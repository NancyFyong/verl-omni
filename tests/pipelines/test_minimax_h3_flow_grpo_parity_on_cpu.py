# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity tests for the MiniMax H3 FlowGRPO rollout loop against its two references.

At ``noise_level=0`` the dual reverse-SDE loop must reproduce vllm-omni's own denoise
loop: the SDE mean collapses to ``x + (s_next - s) * (-v)``, which is algebraically the
vendor's Euler step, so a broken timestep convention, velocity sign, packed layout or
initial noise shows up as a mismatch. At ``noise_level>0`` the training adapter must
recompute the rollout's log probs from the captured trajectory, since the FlowGRPO
importance ratio is 1 at initialisation only if it does.

The stub DiT is a fixed function of (rows, per-row timestep), so this pins the update
rule rather than a checkpoint. Set ``H3_PARITY_DEVICE=cuda`` to run the same checks on
the accelerator.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

denoise_loop_module = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.denoise_loop")
packed_sequence_module = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.packed_sequence")
packed_tokens_module = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.packed_tokens")
rollout_module = pytest.importorskip("verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter")

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import unpack_video_audio_rows  # noqa: E402
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO  # noqa: E402
from verl_omni.pipelines.schedulers import flow_match_shift_sigmas  # noqa: E402

_DEVICE = torch.device(os.environ.get("H3_PARITY_DEVICE", "cpu"))
if _DEVICE.type == "cuda" and not torch.cuda.is_available():
    pytest.skip("H3_PARITY_DEVICE=cuda but no CUDA device is visible", allow_module_level=True)

_SEED = 1234
_NUM_STEPS = 8
_VIDEO_SHIFT, _AUDIO_SHIFT = 12.0, 3.0
_LATENT_T, _LATENT_H, _LATENT_W, _AUDIO_T = 2, 8, 8, 3
_TEXT_LEN, _TEXT_DIM = 16, 5120
_NUM_VIDEO_ROWS = _LATENT_T * (_LATENT_H // 2) * (_LATENT_W // 2)
_NUM_AUDIO_ROWS = _AUDIO_T * 2
_WEIGHT_VIDEO, _WEIGHT_AUDIO = 1.0, 1.0
_NOISE_LEVEL = 0.7


def _video_velocity(rows, timestep):
    return 0.5 * rows * (1.0 + timestep) + 0.25 * torch.sin(torch.tensor(4.0 * timestep, device=rows.device))


def _audio_velocity(rows, timestep):
    return -0.3 * rows * (2.0 - timestep) + 0.1 * torch.cos(torch.tensor(3.0 * timestep, device=rows.device))


class _PackedStubDiT:
    """Stub with the vendor's packed forward interface."""

    def __init__(self, img_pos, audio_pos):
        self.img_pos, self.audio_pos = img_pos, audio_pos
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        row_timesteps = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        video_rows = kwargs["x"][0][self.img_pos]
        audio_rows = kwargs["audio_x"][0][self.audio_pos]
        return (
            _video_velocity(video_rows, float(row_timesteps[self.img_pos[0]])),
            _audio_velocity(audio_rows, float(row_timesteps[self.audio_pos[0]])),
        )


class _TrainingStubDiT:
    """Stub with the training adapter's diffusers-style interface, computing the same function."""

    config = None

    def __call__(self, **kwargs):
        row_timesteps = kwargs["timestep"][kwargs["timestep_indices"]]
        return (
            _video_velocity(kwargs["hidden_states"], float(row_timesteps[kwargs["video_indices"][0]])),
            _audio_velocity(kwargs["audio_hidden_states"], float(row_timesteps[kwargs["audio_indices"][0]])),
        )


def _packed_layout():
    return packed_sequence_module.minimax_h3_packed_sequence(
        text_len=_TEXT_LEN,
        latent_t=_LATENT_T,
        latent_h=_LATENT_H,
        latent_w=_LATENT_W,
        audio_t=_AUDIO_T,
        include_keyframe_cond=False,
    )


def _pipeline(noise_level):
    """A MiniMaxH3PipelineWithLogProb carrying only the attributes ``diffuse`` reads."""
    pipeline = object.__new__(rollout_module.MiniMaxH3PipelineWithLogProb)
    pipeline.device = _DEVICE
    pipeline.default_video_shift, pipeline.default_audio_shift = _VIDEO_SHIFT, _AUDIO_SHIFT
    pipeline._flow_grpo_capture = None
    pipeline._flow_grpo_noise_level = noise_level
    pipeline._flow_grpo_sde_type = "sde"
    pipeline._flow_grpo_video_weight, pipeline._flow_grpo_audio_weight = _WEIGHT_VIDEO, _WEIGHT_AUDIO
    pipeline._flow_grpo_window_size = None
    pipeline._flow_grpo_window_range = None
    pipeline._flow_grpo_window_seed = 42
    packed = _packed_layout()
    pipeline.transformer = _PackedStubDiT(
        packed["img_pos"].view(-1).to(torch.long).to(_DEVICE),
        packed["audio_pos"].view(-1).to(torch.long).to(_DEVICE),
    )
    return pipeline


def _text_inputs():
    generator = torch.Generator().manual_seed(0)
    text_embeddings = torch.randn(_TEXT_LEN, _TEXT_DIM, generator=generator, dtype=torch.float32).to(_DEVICE)
    return text_embeddings, torch.ones(_TEXT_LEN, dtype=torch.long)


def _diffuse(pipeline, text_embeddings, text_tags):
    return pipeline.diffuse(
        task="t2va",
        text_embeddings=text_embeddings,
        text_tags=text_tags,
        seed=_SEED,
        latent_t=_LATENT_T,
        latent_h=_LATENT_H,
        latent_w=_LATENT_W,
        audio_t=_AUDIO_T,
        num_frames=8,
        num_steps=_NUM_STEPS,
        video_shift=_VIDEO_SHIFT,
        audio_shift=_AUDIO_SHIFT,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
    )


def _vendor_reference(pipeline, text_embeddings, text_tags):
    """The vendor denoise loop plus its teardown over the same layout, noise and stub."""
    packed = _packed_layout()
    token_tags = packed["token_tags"].clone()
    token_tags[packed["text_pos"]] = text_tags.cpu()
    branch = denoise_loop_module.MiniMaxH3DenoiseBranch(
        packed=packed, text_embeddings=text_embeddings, token_tags=token_tags, device=_DEVICE
    )
    initial_video, initial_audio = pipeline._initial_noise(
        seed=_SEED, latent_t=_LATENT_T, latent_h=_LATENT_H, latent_w=_LATENT_W, audio_t=_AUDIO_T
    )
    video_rows, audio_rows = denoise_loop_module.minimax_h3_denoise_loop(
        model=pipeline.transformer,
        positive=branch,
        initial_video_rows=initial_video,
        initial_audio_rows=initial_audio,
        keyframe_cond_rows=None,
        audio_ref_rows=None,
        sigmas_video=flow_match_shift_sigmas(num_steps=_NUM_STEPS, shift_scale=_VIDEO_SHIFT),
        sigmas_audio=flow_match_shift_sigmas(num_steps=_NUM_STEPS, shift_scale=_AUDIO_SHIFT),
        device=_DEVICE,
        imgvid_cond_noise_aug_for_inference=denoise_loop_module.MINIMAX_H3_IMGVID_COND_TIMESTEP,
        audio_cond_noise_aug_for_inference=denoise_loop_module.MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    video_latent = packed_tokens_module.minimax_h3_unpatchify_video_tokens(
        video_rows[branch.update_mask_dev],
        latent_shape=(_LATENT_T, _LATENT_H // 2, _LATENT_W // 2, 24),
        patch_size=(1, 2, 2),
    )
    audio_latent = packed_tokens_module.minimax_h3_unpack_audio_tokens(
        audio_rows[branch.audio_update_mask_dev], audio_t=_AUDIO_T * 2, audio_channel=2
    )
    return SimpleNamespace(
        video_rows=video_rows, audio_rows=audio_rows, video_latent=video_latent, audio_latent=audio_latent
    )


@pytest.fixture(scope="module")
def eta0_run():
    text_embeddings, text_tags = _text_inputs()
    pipeline = _pipeline(noise_level=0.0)
    video_latent, audio_latent = _diffuse(pipeline, text_embeddings, text_tags)
    rollout_calls = pipeline.transformer.calls
    reference = _vendor_reference(pipeline, text_embeddings, text_tags)
    return SimpleNamespace(
        capture=pipeline._flow_grpo_capture,
        video_latent=video_latent,
        audio_latent=audio_latent,
        rollout_calls=rollout_calls,
        reference=reference,
    )


@pytest.fixture(scope="module")
def sde_capture():
    text_embeddings, text_tags = _text_inputs()
    pipeline = _pipeline(noise_level=_NOISE_LEVEL)
    _diffuse(pipeline, text_embeddings, text_tags)
    return pipeline._flow_grpo_capture


class TestMiniMaxH3FlowGRPOVendorParity:
    """At eta=0 the capture loop must match vllm-omni's denoise loop."""

    def test_final_rows_match(self, eta0_run):
        final_video, final_audio = unpack_video_audio_rows(
            eta0_run.capture["all_latents"][:, -1].to(_DEVICE), _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS
        )
        torch.testing.assert_close(final_video[0], eta0_run.reference.video_rows, rtol=0, atol=1e-4)
        torch.testing.assert_close(final_audio[0], eta0_run.reference.audio_rows, rtol=0, atol=1e-4)

    def test_decoded_latents_match(self, eta0_run):
        torch.testing.assert_close(eta0_run.video_latent, eta0_run.reference.video_latent, rtol=0, atol=1e-4)
        torch.testing.assert_close(eta0_run.audio_latent, eta0_run.reference.audio_latent, rtol=0, atol=1e-4)

    def test_captures_every_transition_once(self, eta0_run):
        window = _NUM_STEPS - 1
        capture = eta0_run.capture
        assert eta0_run.rollout_calls == window
        assert capture["all_latents"].shape[:2] == (1, window + 1)
        for key in ("all_log_probs", "all_timesteps", "audio_all_timesteps"):
            assert capture[key].shape == (1, window), key


class TestMiniMaxH3FlowGRPOLogProbAgreement:
    """The actor must recompute the rollout's log probs from the captured trajectory."""

    @staticmethod
    def _model_config():
        model_config = MagicMock()
        model_config.pipeline.num_inference_steps = _NUM_STEPS
        model_config.pipeline.video_flow_shift = _VIDEO_SHIFT
        model_config.pipeline.audio_flow_shift = _AUDIO_SHIFT
        model_config.pipeline.av_logprob_video_weight = _WEIGHT_VIDEO
        model_config.pipeline.av_logprob_audio_weight = _WEIGHT_AUDIO
        model_config.algo.noise_level = _NOISE_LEVEL
        model_config.algo.sde_type = "sde"
        return model_config

    def test_rollout_log_probs_are_finite(self, sde_capture):
        assert bool(torch.isfinite(sde_capture["all_log_probs"]).all())

    def test_training_adapter_reproduces_rollout_log_probs(self, sde_capture):
        model_config = self._model_config()
        scheduler = MiniMaxH3FlowGRPO.build_scheduler(model_config)
        MiniMaxH3FlowGRPO.set_timesteps(scheduler, model_config, str(_DEVICE))
        scheduler_inputs = {
            "all_latents": sde_capture["all_latents"].float().to(_DEVICE),
            "all_timesteps": sde_capture["all_timesteps"].float().to(_DEVICE),
            "audio_all_timesteps": sde_capture["audio_all_timesteps"].float().to(_DEVICE),
            "latent_meta": torch.tensor(
                [[_NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS, _LATENT_T, _LATENT_H, _LATENT_W, _AUDIO_T]], dtype=torch.long
            ),
        }
        module = _TrainingStubDiT()
        replayed = []
        for step in range(sde_capture["all_timesteps"].shape[1]):
            model_inputs, _ = MiniMaxH3FlowGRPO.prepare_model_inputs(
                module=module,
                model_config=model_config,
                latents=scheduler_inputs["all_latents"],
                timesteps=scheduler_inputs["all_timesteps"],
                prompt_embeds=torch.zeros(1, _TEXT_LEN, _TEXT_DIM, device=_DEVICE),
                prompt_embeds_mask=torch.ones(1, _TEXT_LEN, dtype=torch.long, device=_DEVICE),
                negative_prompt_embeds=None,
                negative_prompt_embeds_mask=None,
                micro_batch=scheduler_inputs,
                step=step,
            )
            log_prob, _, _, _ = MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
                module=module,
                scheduler=scheduler,
                model_config=model_config,
                model_inputs=model_inputs,
                negative_model_inputs=None,
                scheduler_inputs=scheduler_inputs,
                step=step,
            )
            replayed.append(log_prob)
        torch.testing.assert_close(
            torch.stack(replayed, dim=1), sde_capture["all_log_probs"].float().to(_DEVICE), rtol=0, atol=2e-3
        )

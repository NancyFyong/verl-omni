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

"""Real H3 packing/schedulers with CPU model/encoder doubles, not a GPU kernel test."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as upstream
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY
from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline
from verl_omni.pipelines.minimax_h3_flow_grpo.vllm_omni_rollout_adapter import MiniMaxH3PipelineWithLogProb


class _ToyDiT:
    def __init__(self):
        self.batch_sizes = []

    def modules(self):
        return ()

    def __call__(self, **inputs):
        self.batch_sizes.append(inputs["packed_seq_params"]["num_requests"])
        x, audio = inputs["x"][0], inputs["audio_x"][0]
        feature = x.mean(-1) + audio.mean(-1)
        text_pos = inputs["text_pos_info"]["position_ids"]
        feature = feature.index_add(0, text_pos, inputs["prompt_embeds"].float().mean(-1))
        # Consume the real upstream document boundaries: a broken packing layout
        # would change this request-local reduction and fail serial/batch parity.
        result = torch.zeros_like(feature)
        cu = inputs["packed_seq_params"]["cu_seqlens_q"].tolist()
        for start, end in zip(cu[:-1], cu[1:], strict=True):
            if end > start:
                result[start:end] = feature[start:end].mean()
        result += inputs["unique_timesteps"][inputs["inverse_indices"]]
        video_pos = inputs["img_pos_info"]["position_ids"]
        audio_pos = inputs["audio_pos_info"]["position_ids"]
        return 0.01 * (x[video_pos] + result[video_pos, None]), 0.01 * (audio[audio_pos] + result[audio_pos, None])


def _request(index, *, task="t2va", steps=6, contiguous=True):
    sampling = OmniDiffusionSamplingParams(
        seed=17 + index,
        height=32,
        width=32,
        num_frames=22,
        num_inference_steps=steps,
        num_outputs_per_prompt=1,
        max_sequence_length=8 + index,
        extra_args={
            "task": task,
            "noise_level": 0.6 + index * 0.1,
            "sde_type": "cps",
            "sde_window_size": 2,
            "sde_window_range": [0, steps - 1],
            "sde_contiguous": contiguous,
            "sde_window_seed": 123,
            "global_steps": 2,
            MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True,
        },
    )
    return OmniDiffusionRequest(
        prompt={"prompt_token_ids": list(range(1, index + 3))}, sampling_params=sampling, request_id=f"request-{index}"
    )


@pytest.fixture
def pipeline(monkeypatch):
    pipe = object.__new__(MiniMaxH3PipelineWithLogProb)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.od_config = SimpleNamespace(cache_backend=None)
    pipe.transformer = _ToyDiT()
    pipe._transformer_for_task = lambda task: pipe.transformer
    pipe._packed_batch_supported = lambda transformer: True
    pipe._resident_dit_layers_on_device = lambda enabled: nullcontext()
    pipe.progress_bar = lambda total: nullcontext(SimpleNamespace(update=lambda: None))
    pipe.record_denoise_step = MagicMock()
    pipe.tokenizer = MagicMock()
    monkeypatch.setattr(upstream, "minimax_h3_publish_denoise_progress", lambda *args: None)

    def context(*, sampling, **kwargs):
        ids = pipe._h3_prompt_ids
        assert ids is not None  # Exercise the token-ID-native bridge in prepare_encode.
        task = sampling.extra_args["task"]
        visual = torch.full((4, 96), 0.25) if task != "t2va" else None
        audio = torch.full((4, 32), -0.25) if task == "ref2va" else None
        return dict(
            task=task,
            text_embeddings=ids.float()[:, None].expand(-1, 8).clone(),
            text_tags=torch.zeros(len(ids), dtype=torch.long),
            seed=sampling.seed,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=4,
            num_frames=22,
            num_steps=sampling.num_inference_steps,
            video_shift=12.0,
            audio_shift=3.0,
            visual_condition=visual,
            visual_condition_shape=(1, 4, 4) if visual is not None else None,
            audio_condition=audio,
            ref_audio_t=2 if audio is not None else None,
            visual_condition_shapes=None,
            audio_condition_lengths=None,
            num_outputs=1,
            ref_blocks=[{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 2}]
            if task == "ref2va"
            else None,
            keyframe_frame_indices=[0] if task == "fl2va" else None,
            base_schedule=None,
            height=32,
            width=32,
        )

    pipe._prepare_request_inputs = context
    pipe.decode = lambda video, audio, **kwargs: (
        torch.full((1, 3, 2, 4, 4), video.mean().item()),
        audio.clone(),
    )
    return pipe


@pytest.fixture
def nft_pipeline(pipeline):
    pipeline.__class__ = MiniMaxH3DiffusionNFTPipeline
    pipeline.default_video_shift = 12.0
    pipeline._nft_capture = None
    return pipeline


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_nft_serial_step_and_request_batch_match(nft_pipeline, task):
    pipe = nft_pipeline
    requests = [_request(0, task=task), _request(1, task=task, steps=5)]
    expected = [pipe.forward(req) for req in requests]
    first = _state(pipe, requests[0])
    _advance(pipe, [first])
    states = [first, _state(pipe, requests[1])]
    while active := [state for state in states if not state.denoise_completed]:
        _advance(pipe, list(reversed(active)))
    stepped = [pipe.post_decode(state) for state in states]
    batched = pipe.forward(DiffusionRequestBatch(requests=requests))
    for outputs in (stepped, batched):
        for result, reference in zip(outputs, expected, strict=True):
            assert result.trajectory_latents is None
            assert result.trajectory_log_probs is None
            for group in ("rl", "prompt_embeddings"):
                actual = result.output["metadata"][group]
                wanted = reference.output["metadata"][group]
                assert actual.keys() == wanted.keys()
                for key in wanted:
                    torch.testing.assert_close(actual[key], wanted[key], msg=f"{task}/{group}/{key}")
            assert result.output["metadata"]["rl"]["latents_clean"].dtype == torch.float32
    assert pipe._h3_prompt_ids is None


def test_nft_rejects_mixed_policy_versions(nft_pipeline):
    requests = [_request(0), _request(1)]
    requests[1].sampling_params.extra_args["global_steps"] = 3
    with pytest.raises(ValueError, match="policy versions"):
        nft_pipeline.forward(DiffusionRequestBatch(requests=requests))
    with pytest.raises(ValueError, match="policy versions"):
        _advance(nft_pipeline, [_state(nft_pipeline, req) for req in requests])


def _state(pipe, req):
    return pipe.prepare_encode(
        StepRequestState(request_id=req.request_id, prompt=req.prompt, sampling=req.sampling_params)
    )


def _advance(pipe, states):
    # Exercise upstream InputBatch's row-concatenation and the inherited H3 step
    # forward, not just a manually concatenated list of velocities.
    batch = InputBatch.make_batch(states)
    velocities = pipe.denoise_step(batch, states=states)
    for state, velocity in zip(states, velocities.split([s.latents.shape[0] for s in states]), strict=True):
        pipe.step_scheduler(state, velocity)


def _serial_trajectory(pipe, req):
    pipe._configure_flow_grpo(req)
    pipe._ensure_prompt_text(req)
    try:
        context = pipe._prepare_request_inputs(sampling=req.sampling_params)
        pipe.diffuse(**pipe._denoise_kwargs(context))
    finally:
        pipe._h3_prompt_ids = None
    return pipe._flow_grpo_trajectory


def _assert_trajectory(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6, msg=key)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("sde_type", ["cps", "sde"])
def test_stepwise_matches_serial_and_preserves_conditions(pipeline, task, contiguous, sde_type):
    req = _request(0, task=task, contiguous=contiguous)
    req.sampling_params.extra_args["sde_type"] = sde_type
    expected = _serial_trajectory(pipeline, req)
    state = _state(pipeline, req)
    branch = state.extra[upstream._STEP_BRANCH]
    initial_video = state.latents.clone()
    initial_audio = state.extra[upstream._STEP_AUDIO_ROWS].clone()
    while not state.denoise_completed:
        _advance(pipeline, [state])
        assert state.latents.dtype == torch.float32
        assert state.extra[upstream._STEP_AUDIO_ROWS].dtype == torch.float32
        torch.testing.assert_close(state.latents[~branch.update_mask_dev], initial_video[~branch.update_mask_dev])
        torch.testing.assert_close(
            state.extra[upstream._STEP_AUDIO_ROWS][~branch.audio_update_mask_dev],
            initial_audio[~branch.audio_update_mask_dev],
        )
    _assert_trajectory(pipeline._trajectory(state), expected)
    output = pipeline.post_decode(state)
    assert output.trajectory_log_probs.shape == (1, 2)
    torch.testing.assert_close(output.trajectory_latents, expected["all_latents"])
    assert output.trajectory_latents.device.type == "cpu"
    torch.testing.assert_close(output.output["metadata"]["rl"]["all_next_latents"], expected["all_next_latents"])
    pipeline.tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("packed", [True, False])
def test_interleaved_requests_match_serial_with_distinct_rng_and_timesteps(pipeline, packed):
    requests = [_request(0), _request(1, steps=5, contiguous=False)]
    expected = [_serial_trajectory(pipeline, req) for req in requests]
    pipeline.transformer.batch_sizes.clear()
    pipeline._packed_batch_supported = lambda transformer: packed
    first = _state(pipeline, requests[0])
    _advance(pipeline, [first])
    second = _state(pipeline, requests[1])
    assert first.extra["flow_grpo"].generator is not second.extra["flow_grpo"].generator
    assert first.extra["flow_grpo"].video_scheduler is not second.extra["flow_grpo"].video_scheduler
    while active := [s for s in (second, first) if not s.denoise_completed]:
        _advance(pipeline, active)
    for state, reference in zip((first, second), expected, strict=True):
        _assert_trajectory(pipeline._trajectory(state), reference)
    assert (2 in pipeline.transformer.batch_sizes) is packed


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_request_batch_returns_independent_rl_outputs_and_packs_forwards(pipeline, task):
    requests = [_request(0, task=task), _request(1, task=task, steps=5)]
    expected = [_serial_trajectory(pipeline, req) for req in requests]
    for req, reference in zip(requests, expected, strict=True):
        serial_output = pipeline.forward(req)
        torch.testing.assert_close(serial_output.trajectory_latents, reference["all_latents"])
    pipeline.transformer.batch_sizes.clear()
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))
    assert len(outputs) == 2
    assert pipeline.transformer.batch_sizes == [2, 2, 2, 2, 1]
    for output, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(output.trajectory_latents, reference["all_latents"])
        torch.testing.assert_close(output.trajectory_log_probs, reference["all_log_probs"])
        torch.testing.assert_close(output.output["metadata"]["rl"]["all_next_latents"], reference["all_next_latents"])
        torch.testing.assert_close(
            output.output["metadata"]["prompt_embeddings"]["prompt_embeds"], reference["prompt_embeds"]
        )
    assert pipeline._h3_prompt_ids is None


def test_step_batch_rejects_mixed_policy_versions(pipeline):
    requests = [_request(0), _request(1)]
    requests[1].sampling_params.extra_args["global_steps"] = 3
    states = [_state(pipeline, req) for req in requests]
    with pytest.raises(ValueError, match="policy versions"):
        _advance(pipeline, states)


def test_mixed_dits_use_upstream_per_request_fallback(pipeline):
    reference_dit = _ToyDiT()
    pipeline._transformer_for_task = lambda task: reference_dit if task == "ref2va" else pipeline.transformer
    requests = [_request(0), _request(1, task="ref2va")]
    expected = [_serial_trajectory(pipeline, req) for req in requests]
    pipeline.transformer.batch_sizes.clear()
    reference_dit.batch_sizes.clear()
    outputs = pipeline.forward(DiffusionRequestBatch(requests=requests))
    assert pipeline.transformer.batch_sizes == reference_dit.batch_sizes == [1] * 5
    for output, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(output.trajectory_latents, reference["all_latents"])
        torch.testing.assert_close(output.trajectory_log_probs, reference["all_log_probs"])


def test_request_batch_rejects_mixed_policy_versions(pipeline):
    requests = [_request(0), _request(1)]
    requests[1].sampling_params.extra_args["global_steps"] = 3
    with pytest.raises(ValueError, match="policy versions"):
        pipeline.forward(DiffusionRequestBatch(requests=requests))


@pytest.mark.parametrize("unsupported", ["outputs", "dlo", "cache", "quality"])
def test_step_prepare_rejects_unsupported_shared_state(pipeline, unsupported):
    req = _request(0)
    if unsupported == "outputs":
        req.sampling_params.num_outputs_per_prompt = 2
    elif unsupported == "dlo":
        pipeline._dlo_residency_controller = object()
    elif unsupported == "cache":
        pipeline.od_config.cache_backend = "tea_cache"
    else:
        req.sampling_params.quality = "high"
    with pytest.raises((ValueError, NotImplementedError)):
        _state(pipeline, req)


@pytest.mark.parametrize("step_execution", [False, True])
@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
def test_gpu_smoke_passes_batching_flags(step_execution, algorithm, attention):
    from tests.special_e2e.run_flowgrpo_minimax_h3_tiny import _hydra_overrides

    overrides = _hydra_overrides(
        tiny_model_dir="/unused/model",
        train_parquet="/unused/train.parquet",
        val_parquet="/unused/test.parquet",
        reward_stub_path="/unused/reward.py",
        output_dir="/unused/output",
        task="t2va",
        num_gpus=2,
        rollout_tp=1,
        text_encoder_tp=1,
        total_training_steps=2,
        ray_num_cpus=4,
        height=160,
        width=288,
        num_frames=97,
        num_inference_steps=4,
        step_execution=step_execution,
        max_num_seqs=2,
        algorithm=algorithm,
        attention=attention,
    )
    actor_attn, rollout_attn = (
        ("native", "TORCH_SDPA") if attention == "sdpa" else ("_flash_3_varlen_hub", "FLASH_ATTN")
    )
    assert f"actor_rollout_ref.model.attn_backend={actor_attn}" in overrides
    assert f"actor_rollout_ref.rollout.rollout_attn_backend={rollout_attn}" in overrides
    assert f"actor_rollout_ref.rollout.step_execution={step_execution}" in overrides
    assert "actor_rollout_ref.rollout.max_num_seqs=2" in overrides
    assert f"actor_rollout_ref.rollout.calculate_log_probs={algorithm == 'flow_grpo'}" in overrides
    assert "data.train_batch_size=2" in overrides
    assert "trainer.total_epochs=2" in overrides
    if algorithm == "diffusion_nft":
        assert "actor_rollout_ref.rollout.rollout_adapter=old" in overrides
        assert "actor_rollout_ref.model.policy_state_adapters=['default','old']" in overrides
        assert "algorithm.trainer_type=direct_preference" in overrides


def test_failed_encode_clears_prompt_bridge(pipeline):
    def fail(**kwargs):
        raise RuntimeError("encoder failed")

    pipeline._prepare_request_inputs = fail
    with pytest.raises(RuntimeError, match="encoder failed"):
        _state(pipeline, _request(0))
    assert pipeline._h3_prompt_ids is None

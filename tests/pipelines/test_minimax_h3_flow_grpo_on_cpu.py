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
"""CPU tests for the MiniMax H3 FlowGRPO training adapter."""

from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    VIDEO_ROW_WIDTH,
    build_layout_from_meta,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.minimax_h3_flow_grpo import diffusers_training_adapter as flow_grpo_adapter
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler, flow_match_shift_sigmas
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig

_META = [4, 6, 1, 4, 4, 3]
_NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS = 4, 6
_BATCH = 2
_NUM_STEPS = 5
_TEXT_LEN = 12
_TEXT_DIM = 64
_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0
_WEIGHT_VIDEO = 2.0
_WEIGHT_AUDIO = 0.5


def _module(side_effect):
    """A mock transformer with the given per-call behavior and no real ``config`` (patch falls back)."""
    module = MagicMock(side_effect=side_effect)
    module.config = None
    return module


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def _dual_scheduler(num_steps=_NUM_STEPS):
    scheduler = FlowMatchDualSDEDiscreteScheduler(video_flow_shift=_VIDEO_SHIFT, audio_flow_shift=_AUDIO_SHIFT)
    scheduler.set_timesteps(num_steps, device="cpu")
    return scheduler


def _model_config():
    model_config = MagicMock()
    model_config.pipeline.av_logprob_video_weight = _WEIGHT_VIDEO
    model_config.pipeline.av_logprob_audio_weight = _WEIGHT_AUDIO
    model_config.algo.noise_level = 1.0
    model_config.algo.sde_type = "sde"
    return model_config


def _scheduler_inputs(scheduler, batch=_BATCH):
    window = len(scheduler.video_scheduler.timesteps)
    packed_width = _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH
    all_latents = torch.randn(batch, window + 1, packed_width)
    video_timesteps = scheduler.video_scheduler.timesteps.unsqueeze(0).expand(batch, -1).clone()
    audio_timesteps = scheduler.audio_scheduler.timesteps.unsqueeze(0).expand(batch, -1).clone()
    return TensorDict(
        {
            "all_latents": all_latents,
            "all_timesteps": video_timesteps,
            "audio_all_timesteps": audio_timesteps,
            "latent_meta": torch.tensor([_META] * batch, dtype=torch.long),
        },
        batch_size=batch,
    )


def _model_inputs_for_step(scheduler_inputs, step, batch=_BATCH):
    return MiniMaxH3FlowGRPO.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=scheduler_inputs["all_latents"],
        timesteps=scheduler_inputs["all_timesteps"],
        prompt_embeds=torch.randn(batch, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=torch.ones(batch, _TEXT_LEN, dtype=torch.int32),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=scheduler_inputs,
        step=step,
    )[0]


class TestMiniMaxH3FlowGRPORegistry:
    def test_registered_for_minimax_h3_flow_grpo(self):
        resolved = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "flow_grpo")
        assert resolved is MiniMaxH3FlowGRPO

    def test_importing_the_package_registers_both_h3_algorithms(self):
        import verl_omni.pipelines as pipelines

        for algorithm in ("diffusion_nft", "flow_grpo"):
            assert hasattr(pipelines, f"minimax_h3_{algorithm}")
            assert DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", algorithm) is not None


class TestMiniMaxH3FlowGRPORolloutGuards:
    def test_rejects_a_checkpoint_pinned_distilled_schedule(self):
        from verl_omni.pipelines.minimax_h3_flow_grpo import MiniMaxH3PipelineWithLogProb

        pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
        with pytest.raises(NotImplementedError, match="distilled sigma schedule"):
            pipeline.diffuse(task="t2va", base_schedule=[1.0, 0.5, 0.0])

    def test_rejects_conditional_tasks(self):
        from verl_omni.pipelines.minimax_h3_flow_grpo import MiniMaxH3PipelineWithLogProb

        pipeline = object.__new__(MiniMaxH3PipelineWithLogProb)
        with pytest.raises(NotImplementedError, match="task='t2va' only"):
            pipeline.diffuse(task="fl2va")


class TestMiniMaxH3FlowGRPOBuildScheduler:
    def test_build_scheduler_sets_dual_per_modality_timesteps(self, monkeypatch):
        monkeypatch.setattr(flow_grpo_adapter, "get_device_name", lambda: "cpu")
        cfg = object.__new__(DiffusionModelConfig)
        object.__setattr__(cfg, "architecture", "MiniMaxH3Pipeline")
        object.__setattr__(cfg, "algorithm", "flow_grpo")
        object.__setattr__(
            cfg,
            "pipeline",
            DiffusionPipelineConfig(
                num_inference_steps=_NUM_STEPS, video_flow_shift=_VIDEO_SHIFT, audio_flow_shift=_AUDIO_SHIFT
            ),
        )

        scheduler = MiniMaxH3FlowGRPO.build_scheduler(cfg)
        assert isinstance(scheduler, FlowMatchDualSDEDiscreteScheduler)
        assert len(scheduler.video_scheduler.timesteps) == _NUM_STEPS - 1
        assert len(scheduler.audio_scheduler.timesteps) == _NUM_STEPS - 1

        video_sigmas = [round(float(s), 5) for s in scheduler.video_scheduler.sigmas.tolist()]
        audio_sigmas = [round(float(s), 5) for s in scheduler.audio_scheduler.sigmas.tolist()]
        assert video_sigmas == [round(s, 5) for s in flow_match_shift_sigmas(_NUM_STEPS, _VIDEO_SHIFT)]
        assert audio_sigmas == [round(s, 5) for s in flow_match_shift_sigmas(_NUM_STEPS, _AUDIO_SHIFT)]


class TestMiniMaxH3FlowGRPOPrepareModelInputs:
    def test_unpacks_step_latent_and_builds_per_modality_timesteps(self):
        window = _NUM_STEPS - 1
        packed_width = _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH
        latents = torch.randn(_BATCH, window + 1, packed_width)
        video_timesteps = torch.tensor([[1000.0, 900.0, 800.0, 700.0]] * _BATCH)
        audio_timesteps = torch.tensor([[1000.0, 850.0, 700.0, 550.0]] * _BATCH)
        meta = torch.tensor([_META] * _BATCH, dtype=torch.long)
        micro_batch = TensorDict({"latent_meta": meta, "audio_all_timesteps": audio_timesteps}, batch_size=_BATCH)

        step = 1
        model_inputs, negative_model_inputs = MiniMaxH3FlowGRPO.prepare_model_inputs(
            module=MagicMock(),
            model_config=MagicMock(),
            latents=latents,
            timesteps=video_timesteps,
            prompt_embeds=torch.randn(_BATCH, _TEXT_LEN, _TEXT_DIM),
            prompt_embeds_mask=torch.ones(_BATCH, _TEXT_LEN, dtype=torch.int32),
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=micro_batch,
            step=step,
        )

        assert negative_model_inputs is None
        expected_video, expected_audio = unpack_video_audio_rows(latents[:, step], _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        assert torch.equal(model_inputs["video_rows"], expected_video)
        assert torch.equal(model_inputs["audio_rows"], expected_audio)
        assert model_inputs["latent_meta"] == _META
        torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.1, 0.1]))
        torch.testing.assert_close(model_inputs["audio_timestep"], torch.tensor([0.15, 0.15]))


class TestMiniMaxH3FlowGRPOForward:
    def test_forward_routes_per_modality_timesteps(self):
        model_inputs = {
            "video_rows": torch.randn(1, _NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH),
            "audio_rows": torch.randn(1, _NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH),
            "encoder_hidden_states": torch.randn(1, _TEXT_LEN, _TEXT_DIM),
            "encoder_mask": torch.ones(1, _TEXT_LEN, dtype=torch.int32),
            "timestep": torch.tensor([0.7]),
            "audio_timestep": torch.tensor([0.3]),
            "latent_meta": _META,
        }
        module = _module(_identity)
        MiniMaxH3FlowGRPO.forward(module, MagicMock(), model_inputs)

        kwargs = module.call_args_list[0].kwargs
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.3, 0.7]))
        _, _, video_indices, audio_indices, text_indices, _, _ = build_layout_from_meta(_META, _TEXT_LEN)
        timestep_indices = kwargs["timestep_indices"]
        assert (timestep_indices[audio_indices] == 0).all()
        assert (timestep_indices[video_indices] == 1).all()
        assert (timestep_indices[text_indices] == 1).all()

    def test_equal_timesteps_collapse_to_one_distinct(self):
        model_inputs = {
            "video_rows": torch.randn(1, _NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH),
            "audio_rows": torch.randn(1, _NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH),
            "encoder_hidden_states": torch.randn(1, _TEXT_LEN, _TEXT_DIM),
            "encoder_mask": torch.ones(1, _TEXT_LEN, dtype=torch.int32),
            "timestep": torch.tensor([0.5]),
            "audio_timestep": torch.tensor([0.5]),
            "latent_meta": _META,
        }
        module = _module(_identity)
        MiniMaxH3FlowGRPO.forward(module, MagicMock(), model_inputs)

        kwargs = module.call_args_list[0].kwargs
        seq_len = _TEXT_LEN + _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS
        torch.testing.assert_close(kwargs["timestep"], torch.tensor([0.5]))
        assert torch.equal(kwargs["timestep_indices"], torch.zeros(seq_len, dtype=torch.long))


class TestMiniMaxH3FlowGRPOForwardAndSamplePreviousStep:
    def _run_step(self, scheduler, scheduler_inputs, step):
        model_inputs = _model_inputs_for_step(scheduler_inputs, step)
        module = _module(lambda **kw: (kw["hidden_states"] * 2.0, kw["audio_hidden_states"] * 3.0))
        return MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
            module=module,
            scheduler=scheduler,
            model_config=_model_config(),
            model_inputs=model_inputs,
            negative_model_inputs=None,
            scheduler_inputs=scheduler_inputs,
            step=step,
        )

    def test_combined_log_prob_is_weighted_sum_of_per_modality_log_probs(self):
        scheduler = _dual_scheduler()
        scheduler_inputs = _scheduler_inputs(scheduler)
        step = 1

        log_prob, prev_sample_mean, std_dev_t, sqrt_dt = self._run_step(scheduler, scheduler_inputs, step)

        latents = scheduler_inputs["all_latents"]
        cur_video, cur_audio = unpack_video_audio_rows(latents[:, step].float(), _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        prev_video, prev_audio = unpack_video_audio_rows(latents[:, step + 1].float(), _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        v_video, v_audio = cur_video * -2.0, cur_audio * -3.0
        _, lp_video, _, std_video, sqrt_dt_video = scheduler.video_scheduler.sample_previous_step(
            sample=cur_video,
            model_output=v_video,
            timestep=scheduler_inputs["all_timesteps"][:, step],
            noise_level=1.0,
            prev_sample=prev_video,
            sde_type="sde",
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        _, lp_audio, _, std_audio, sqrt_dt_audio = scheduler.audio_scheduler.sample_previous_step(
            sample=cur_audio,
            model_output=v_audio,
            timestep=scheduler_inputs["audio_all_timesteps"][:, step],
            noise_level=1.0,
            prev_sample=prev_audio,
            sde_type="sde",
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        assert log_prob.shape == (_BATCH,)
        torch.testing.assert_close(log_prob, _WEIGHT_VIDEO * lp_video + _WEIGHT_AUDIO * lp_audio)
        # std/sqrt_dt are the weight-combined dual-stream scales GRPO-Guard / FlowDPPO consume.
        assert prev_sample_mean.shape == latents[:, step].shape
        assert std_dev_t.shape == (_BATCH, 1, 1)
        assert sqrt_dt.shape == (_BATCH,)
        weight_total = _WEIGHT_VIDEO + _WEIGHT_AUDIO
        torch.testing.assert_close(std_dev_t, (_WEIGHT_VIDEO * std_video + _WEIGHT_AUDIO * std_audio) / weight_total)
        torch.testing.assert_close(
            sqrt_dt, (_WEIGHT_VIDEO * sqrt_dt_video + _WEIGHT_AUDIO * sqrt_dt_audio) / weight_total
        )

    def test_every_trajectory_step_is_consumable(self):
        scheduler = _dual_scheduler()
        scheduler_inputs = _scheduler_inputs(scheduler)
        window = len(scheduler.video_scheduler.timesteps)
        assert scheduler_inputs["all_latents"].shape[1] == scheduler_inputs["all_timesteps"].shape[1] + 1

        for step in range(window):
            log_prob, _, _, _ = self._run_step(scheduler, scheduler_inputs, step)
            assert log_prob.shape == (_BATCH,)
            assert torch.isfinite(log_prob).all()

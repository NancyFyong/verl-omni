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
"""CPU contract tests for block-causal Diffusers Wan attention."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from diffusers import WanTransformer3DModel
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.wan21_distillation.causal_attention import (
    WanCausalCache,
    allocate_wan_cache,
    configure_causal_wan,
    wan_causal_forward,
)
from verl_omni.pipelines.wan21_distillation.diffusers_training_adapter import (
    Wan21CausalODE,
    WanConditionProvider,
    WanODEComputer,
    build_wan_causal_timesteps,
)
from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest
from verl_omni.trainer.diffusion.distillation.recipes import build_plan
from verl_omni.utils.dataset.distillation import canonical_manifest_sha256
from verl_omni.workers.config import DiffusionModelConfig


def tiny_wan() -> WanTransformer3DModel:
    """Build a deterministic small Wan transformer without model weights."""
    torch.manual_seed(7)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    ).eval()
    return configure_causal_wan(model)


def forward_wan(model, latents, timesteps, context):
    """Run the minimal Diffusers Wan signature used by the adapter."""
    return model(
        hidden_states=latents,
        timestep=timesteps,
        encoder_hidden_states=context,
        return_dict=False,
    )[0]


class RecordingWanPipeline:
    """Small prompt-encoding stand-in for the local frozen provider."""

    def encode_prompt(self, *, prompt, **kwargs):
        self.prompts = prompt
        return torch.ones(len(prompt), 3, 16), None


class RecordingScheduler:
    """Capture scheduler setup performed by the Wan adapter."""

    def set_timesteps(self, count, *, device, sigmas):
        self.count = count
        self.device = device
        self.sigmas = sigmas


def raise_forward_failure(*args, **kwargs):
    """Fail after earlier transformer blocks have staged cache entries."""
    raise RuntimeError("failure")


class ToyWanRuntime:
    def __init__(self, module):
        self.module = module
        self.engine = SimpleNamespace(get_data_parallel_rank=lambda: 0)

    def engine_for_role(self, role):
        assert role == "student"
        return self.engine

    @contextmanager
    def use_role(self, role, *, grad_enabled=None, input_grad_only=False):
        assert role == "student" and not input_grad_only
        with torch.set_grad_enabled(bool(grad_enabled)):
            yield self.module


class TestWanConditionProvider:
    def test_local_provider_reads_raw_prompt_without_precomputed_embeddings(self):
        provider = WanConditionProvider("/unused", "local_frozen_encoder", 8)
        provider.pipeline = RecordingWanPipeline()
        batch = TensorDict({}, batch_size=[1])
        tu.assign_non_tensor_stack(batch, "raw_prompt", [[{"role": "user", "content": "a swan"}]])
        condition = provider.encode(batch, device=torch.device("cpu"), dtype=torch.float32)
        assert provider.pipeline.prompts == ["a swan"]
        assert condition.tensors["prompt_embeds"].shape == (1, 3, 16)


class TestWanCausalSchedule:
    def test_reference_four_step_shift(self):
        timesteps = build_wan_causal_timesteps([1000, 750, 500, 250], 1000, 8.0)
        torch.testing.assert_close(
            timesteps,
            torch.tensor([1000.0, 960.0, 888.888916015625, 727.272705078125]),
        )

    def test_training_scheduler_defaults_to_wan21_shift(self):
        scheduler = RecordingScheduler()
        model_config = SimpleNamespace(pipeline=OmegaConf.create({"num_inference_steps": 4}))
        Wan21CausalODE.set_timesteps(scheduler, model_config, "cpu")
        assert scheduler.count == 4
        torch.testing.assert_close(torch.as_tensor(scheduler.sigmas), torch.tensor([1.0, 0.9, 0.75, 0.5]))

    @pytest.mark.parametrize("timesteps", [[], [750, 1000], [1001], [True]])
    def test_invalid_schedule_fails_closed(self, timesteps):
        with pytest.raises(ValueError, match="strictly descending"):
            build_wan_causal_timesteps(timesteps, 1000, 8.0)


class TestWanCausalAttention:
    def test_adapter_is_registered_for_wan_ode_regression(self):
        model_config = object.__new__(DiffusionModelConfig)
        object.__setattr__(model_config, "architecture", "WanPipeline")
        object.__setattr__(model_config, "algorithm", "ode_regression")
        assert DiffusionModelBase.get_class(model_config) is Wan21CausalODE

    def test_causal_conversion_preserves_checkpoint_keys(self):
        model = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            rope_max_seq_len=32,
        )
        keys = set(model.state_dict())
        configure_causal_wan(model)
        assert set(model.state_dict()) == keys

    def test_future_blocks_do_not_change_prefix_output(self):
        model = tiny_wan()
        latents = torch.randn(1, 4, 4, 4, 4)
        changed = latents.clone()
        changed[:, :, 2:] += 100
        timesteps = torch.full((1, 16), 500.0)
        condition = torch.randn(1, 3, 16)
        with torch.no_grad(), wan_causal_forward(model, num_frames=4, frames_per_block=2):
            original = forward_wan(model, latents, timesteps, condition)
        with torch.no_grad(), wan_causal_forward(model, num_frames=4, frames_per_block=2):
            perturbed = forward_wan(model, changed, timesteps, condition)
        torch.testing.assert_close(original[:, :, :2], perturbed[:, :, :2])

    def test_full_forward_matches_incremental_cache(self):
        model = tiny_wan()
        latents = torch.randn(1, 4, 4, 4, 4)
        timesteps = torch.tensor([[900.0, 900.0, 400.0, 400.0]]).repeat_interleave(4, dim=1)
        condition = torch.randn(1, 3, 16)
        with torch.no_grad(), wan_causal_forward(model, num_frames=4, frames_per_block=2):
            full = forward_wan(model, latents, timesteps, condition)

        cache = allocate_wan_cache(model, batch_size=1, latent_height=4, latent_width=4, max_frames=4)
        chunks = []
        first_cross_pointers = None
        for start in range(0, 4, 2):
            with (
                torch.no_grad(),
                wan_causal_forward(
                    model,
                    num_frames=2,
                    frames_per_block=2,
                    cache=cache,
                    commit_cache=True,
                ),
            ):
                chunks.append(
                    forward_wan(
                        model,
                        latents[:, :, start : start + 2],
                        timesteps[:, start * 4 : (start + 2) * 4],
                        condition,
                    )
                )
            if first_cross_pointers is None:
                first_cross_pointers = tuple(value[0].data_ptr() for value in cache.cross_key_values if value)
        incremental = torch.cat(chunks, dim=2)
        torch.testing.assert_close(incremental, full, rtol=2e-5, atol=2e-5)
        assert cache.committed_frames == 4
        assert all(value is not None and value[0].shape[1] == condition.shape[1] for value in cache.cross_key_values)
        assert first_cross_pointers == tuple(value[0].data_ptr() for value in cache.cross_key_values if value)

    def test_full_forward_supports_gradient_checkpointing(self):
        model = tiny_wan().train()
        model.enable_gradient_checkpointing()
        latents = torch.randn(1, 4, 4, 4, 4)
        timesteps = torch.full((1, 16), 500.0)
        condition = torch.randn(1, 3, 16)
        with wan_causal_forward(model, num_frames=4, frames_per_block=2):
            output = forward_wan(model, latents, timesteps, condition)
        output.square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())

    def test_read_only_forward_does_not_commit(self):
        model = tiny_wan()
        cache = allocate_wan_cache(model, batch_size=1, latent_height=4, latent_width=4, max_frames=4)
        latents = torch.randn(1, 4, 2, 4, 4)
        timesteps = torch.full((1, 8), 500.0)
        condition = torch.randn(1, 3, 16)
        with (
            torch.no_grad(),
            wan_causal_forward(model, num_frames=2, frames_per_block=2, cache=cache, commit_cache=False),
        ):
            forward_wan(model, latents, timesteps, condition)
        assert cache.committed_frames == 0
        assert cache.key_values == [None, None]
        assert cache.cross_key_values == [None, None]

    def test_failed_forward_never_partially_commits(self, monkeypatch):
        model = tiny_wan()
        cache = allocate_wan_cache(model, batch_size=1, latent_height=4, latent_width=4, max_frames=4)
        latents = torch.randn(1, 4, 2, 4, 4)
        timesteps = torch.full((1, 8), 500.0)
        condition = torch.randn(1, 3, 16)

        monkeypatch.setattr(model.blocks[1], "forward", raise_forward_failure)
        with pytest.raises(RuntimeError, match="failure"):
            with wan_causal_forward(model, num_frames=2, frames_per_block=2, cache=cache, commit_cache=True):
                forward_wan(model, latents, timesteps, condition)
        assert cache.committed_frames == 0
        assert cache.key_values == [None, None]
        assert cache.cross_key_values == [None, None]

    def test_cache_rejects_partial_and_overflowing_commits(self):
        cache = WanCausalCache(layer_count=2, batch_size=1, tokens_per_frame=4, max_frames=2)
        tensor = torch.zeros(1, 8, 2, 8)
        cross = torch.zeros(1, 3, 2, 8)
        cross_pending = [(cross, cross), (cross, cross)]
        with pytest.raises(ValueError, match="Every Wan layer"):
            cache.commit([(tensor, tensor), None], cross_pending, frame_count=2)
        cache.commit([(tensor, tensor), (tensor, tensor)], cross_pending, frame_count=2)
        with pytest.raises(ValueError, match="capacity"):
            cache.commit([(tensor, tensor), (tensor, tensor)], [None, None], frame_count=2)


class TestWanODEComputer:
    @staticmethod
    def build_computer_and_batch():
        trajectory_manifest = {
            "teacher_model": "wan-teacher",
            "teacher_revision": "revision",
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "scheduler_config": {"shift": 8.0},
            "guidance_scale": 6.0,
            "negative_prompt": "",
            "timesteps": [1000.0, 500.0, 0.0],
            "vae": "wan-vae",
            "latent_layout": "SFCHW",
            "dtype": "float32",
            "height": 4,
            "width": 4,
            "num_frames": 6,
            "prompt_tokenizer": "umt5",
            "seed_policy": "fixture",
        }
        digest = canonical_manifest_sha256(trajectory_manifest)
        plan = build_plan(
            "ode_regression",
            {
                "model_path": "/unused",
                "trajectory_manifest_sha256": digest,
                "conditioning_provider": "precomputed",
                "frames_per_block": 2,
                "rng_seed": 3,
            },
            frozenset({"distribution_matching", "autoregressive", "ode_regression"}),
        )
        config = SimpleNamespace(
            path="/unused",
            local_path="/unused",
            pipeline=SimpleNamespace(max_sequence_length=8),
        )
        trajectory = torch.randn(1, 3, 6, 4, 4, 4)
        batch = TensorDict(
            {
                "prompt_embeds": torch.randn(1, 3, 16),
                "ode_latents": trajectory,
                "ode_timesteps": torch.tensor([[1000.0, 500.0, 0.0]]),
                "final_clean_latent": trajectory[:, -1].clone(),
            },
            batch_size=[1],
        )
        tu.assign_non_tensor_stack(batch, "trajectory_manifest", [trajectory_manifest])
        tu.assign_non_tensor_stack(batch, "trajectory_manifest_sha256", [digest])
        return WanODEComputer(config, plan), batch

    def test_blockwise_state_selection_and_ode_backward(self):
        computer, batch = self.build_computer_and_batch()
        model = tiny_wan()
        runtime = ToyWanRuntime(model)
        request = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
        computation = computer.compute_phase(request, batch, runtime)
        computation.losses["student"].backward()
        assert 0 < computation.metrics["ode/active_elements"] <= 384
        assert any(parameter.grad is not None for parameter in model.parameters())

    def test_manifest_mismatch_fails_before_forward(self):
        computer, batch = self.build_computer_and_batch()
        tu.assign_non_tensor_stack(batch, "trajectory_manifest_sha256", ["wrong"])
        with pytest.raises(ValueError, match="manifest"):
            computer.compute_phase(
                PhaseRequest("student", 0, 0, "fresh", ("student",), True), batch, ToyWanRuntime(tiny_wan())
            )

    def test_rng_checkpoint_replays_block_indices(self):
        computer, batch = self.build_computer_and_batch()
        runtime = ToyWanRuntime(tiny_wan())
        trajectory = batch["ode_latents"]
        timesteps = batch["ode_timesteps"]
        computer.select_states(trajectory, timesteps, runtime)
        state = computer.state_dict()
        expected = computer.select_states(trajectory, timesteps, runtime)
        restored, _ = self.build_computer_and_batch()
        restored.load_state_dict(state)
        actual = restored.select_states(trajectory, timesteps, runtime)
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            torch.testing.assert_close(expected_tensor, actual_tensor)

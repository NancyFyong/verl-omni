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
"""Causal inference, decoding and semantic FSDP adapter export contracts."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from diffusers import WanTransformer3DModel
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from examples.distillation_trainer.wan21.generate_ode_data import TrajectoryCollector, ode_timesteps
from verl_omni.pipelines.wan21_distillation.diffusers_training_adapter import WanCausVidComputer
from verl_omni.pipelines.wan21_distillation.inference import (
    decode_wan_latents,
    sample_wan_causal,
    validate_wan_sampling_timesteps,
    wan_reference_sigma,
)
from verl_omni.trainer.diffusion.distillation.contracts import TrainerCounters
from verl_omni.trainer.diffusion.distillation.recipes import build_plan
from verl_omni.utils.dataset.distillation import canonical_manifest_sha256
from verl_omni.utils.fsdp_utils import export_fsdp_lora_adapter
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


def tiny_wan():
    torch.manual_seed(7)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=32,
    ).eval()


class RecordingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(z_dim=3, latents_mean=[0.1, 0.2, 0.3], latents_std=[2.0, 3.0, 4.0])

    def decode(self, latents, return_dict=False):
        self.input = latents
        return (torch.zeros_like(latents),)


class TestWanTeacherTrajectories:
    def test_snapshot_schedule_matches_reference_sigma_min_zero(self):
        times = ode_timesteps()
        torch.testing.assert_close(
            torch.tensor(times), torch.tensor([1000.0, 756.7568, 521.7391, 0.0]), atol=1e-3, rtol=0
        )

    def test_collector_saves_initial_intermediate_and_final_states(self):
        initial = torch.zeros(1, 4, 6, 2, 2)
        collector = TrajectoryCollector(initial, index=0)
        collector(None, 35, torch.tensor(500.0), {"latents": initial + 1})
        collector(None, 43, torch.tensor(200.0), {"latents": initial + 2})
        trajectory = collector.finish(initial + 3)
        assert trajectory.shape == (4, 6, 4, 2, 2)
        for index, state in enumerate(trajectory):
            torch.testing.assert_close(state, torch.full_like(state, index))


class TestWanCausalInference:
    @pytest.mark.parametrize("schedule", [[], [1000, 0], [500, 1000], [True], [999], [1000, float("nan")]])
    def test_bad_resolved_schedule_fails(self, schedule):
        with pytest.raises(ValueError):
            validate_wan_sampling_timesteps(schedule, 1000)

    def test_rng_replay_prefix_isolation_and_cache_reset(self):
        model = tiny_wan()
        noise = torch.randn(1, 4, 4, 4, 4)
        prompt = torch.randn(1, 3, 16)
        kwargs = dict(timesteps=[1000, 600], frames_per_block=2)
        first = sample_wan_causal(model, noise, prompt, generator=torch.Generator().manual_seed(11), **kwargs)
        second = sample_wan_causal(model, noise, prompt, generator=torch.Generator().manual_seed(11), **kwargs)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        perturbed = noise.clone()
        perturbed[:, :, 2:] += 5
        changed = sample_wan_causal(model, perturbed, prompt, generator=torch.Generator().manual_seed(11), **kwargs)
        torch.testing.assert_close(first[:, :, :2], changed[:, :, :2], rtol=0, atol=0)
        assert not torch.equal(first[:, :, 2:], changed[:, :, 2:])
        assert first.dtype == torch.float32 and not first.requires_grad
        assert all(block.attn1.processor.cache is None for block in model.blocks)
        assert all(block.attn2.processor.cache is None for block in model.blocks)

    def test_mixed_precision_preserves_fp32_islands_and_casts_model_inputs(self):
        model = tiny_wan().to(torch.bfloat16)
        for name, parameter in model.named_parameters():
            if any(part in name for part in model._keep_in_fp32_modules):
                parameter.data = parameter.data.float()
        assert next(model.parameters()).dtype == torch.float32
        assert model.dtype == torch.bfloat16
        output = sample_wan_causal(
            model,
            torch.randn(1, 4, 2, 4, 4),
            torch.randn(1, 3, 16),
            timesteps=[1000, 500],
            frames_per_block=2,
            generator=torch.Generator().manual_seed(1),
        )
        assert torch.isfinite(output).all() and output.dtype == torch.float32

    def test_reference_sigma_lookup_includes_nonzero_zero_timestep_boundary(self):
        raw = torch.linspace(1, 0, 1001)[:-1]
        expected = 8 * raw / (1 + 7 * raw)
        times = torch.tensor([1000.0, 757.0, 522.0, 0.0])
        indices = (times[:, None] - expected[None, :] * 1000).abs().argmin(dim=1)
        torch.testing.assert_close(wan_reference_sigma(times), expected[indices], rtol=0, atol=0)
        assert wan_reference_sigma(torch.tensor(0.0)) > 0

    def test_progress_includes_every_denoising_step(self):
        model = tiny_wan()
        events = []
        sample_wan_causal(
            model,
            torch.randn(1, 4, 4, 4, 4),
            torch.randn(1, 3, 16),
            timesteps=[1000, 700, 400],
            frames_per_block=2,
            generator=torch.Generator().manual_seed(1),
            callback=lambda *event: events.append(event),
        )
        assert events == [(block, 2, step, 3) for block in (1, 2) for step in (1, 2, 3)]

    def test_decode_inverts_training_normalization(self):
        vae = RecordingVAE()
        latents = torch.ones(1, 3, 2, 4, 4)
        video = decode_wan_latents(vae, latents)
        expected = torch.tensor([2.1, 3.2, 4.3]).reshape(1, 3, 1, 1, 1).expand_as(latents)
        torch.testing.assert_close(vae.input, expected)
        assert video.shape == (1, 2, 3, 4, 4)
        torch.testing.assert_close(video, torch.full_like(video, 0.5))


class WanRoleEngine:
    def __init__(self):
        self.module = tiny_wan()

    def get_data_parallel_rank(self):
        return 0


class WanRoleRuntime:
    def __init__(self):
        self.engines = {role: WanRoleEngine() for role in ("student", "teacher_score", "fake_score")}
        self.engines["teacher_score"].module.requires_grad_(False)

    def engine_for_role(self, role):
        return self.engines[role]

    @contextmanager
    def use_role(self, role, *, grad_enabled):
        with torch.set_grad_enabled(grad_enabled):
            yield self.engines[role].module


def causvid_computer_and_batch():
    manifest = {"latent_layout": "SFCHW", "height": 4, "width": 4, "num_frames": 4}
    digest = canonical_manifest_sha256(manifest)
    plan = build_plan(
        "causvid",
        {
            "model_path": "/tiny",
            "teacher_cfg_norm": "none",
            "teacher_guidance_scale": 3.5,
            "normalization_epsilon": 0.0,
            "score_timestep_shift": 8.0,
            "frames_per_block": 2,
            "trajectory_manifest_sha256": digest,
            "conditioning_provider": "precomputed",
            "fake_update_ratio": 5,
        },
        frozenset({"distribution_matching", "autoregressive"}),
    )
    config = SimpleNamespace(path="/tiny", local_path="/tiny", pipeline=SimpleNamespace(max_sequence_length=4))
    computer = WanCausVidComputer(config, plan)
    batch = TensorDict(
        {
            "final_clean_latent": torch.randn(1, 4, 4, 4, 4),
            "prompt_embeds": torch.randn(1, 3, 16),
            "negative_prompt_embeds": torch.randn(1, 3, 16),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_stack(batch, "trajectory_manifest", [manifest])
    tu.assign_non_tensor_stack(batch, "trajectory_manifest_sha256", [digest])
    return computer, batch, plan


class TestWanCausVid:
    def test_reference_cycle_reuses_first_critic_batch_and_has_unique_counters(self):
        _, _, plan = causvid_computer_and_batch()
        cycle = plan.update_schedule.next_cycle(TrainerCounters())
        assert [request.kind for request in cycle.requests] == ["student"] + ["fake_score"] * 5
        assert cycle.requests[1].batch_policy == "reuse_student"
        assert all(request.batch_policy == "fresh" for request in cycle.requests[2:])
        assert [request.repeat_index for request in cycle.requests[1:]] == list(range(5))

    def test_student_and_fake_gradient_isolation_and_rng_replay(self):
        computer, batch, plan = causvid_computer_and_batch()
        runtime = WanRoleRuntime()
        cycle = plan.update_schedule.next_cycle(TrainerCounters())
        state = computer.state_dict()
        computation = computer.compute_phase(cycle.requests[0], batch, runtime)
        assert computation.losses["student"].isfinite()
        computation.losses["student"].backward()
        assert any(parameter.grad is not None for parameter in runtime.engine_for_role("student").module.parameters())
        for role in ("fake_score", "teacher_score"):
            assert all(parameter.grad is None for parameter in runtime.engine_for_role(role).module.parameters())
        computer.load_state_dict(state)
        replay = computer.compute_phase(cycle.requests[0], batch, runtime)
        torch.testing.assert_close(replay.losses["student"], computation.losses["student"], rtol=0, atol=0)
        for engine in runtime.engines.values():
            engine.module.zero_grad()
        fake = computer.compute_phase(cycle.requests[1], batch, runtime)
        fake.losses["fake_score"].backward()
        assert any(
            parameter.grad is not None for parameter in runtime.engine_for_role("fake_score").module.parameters()
        )
        assert all(parameter.grad is None for parameter in runtime.engine_for_role("student").module.parameters())

    def test_negative_condition_is_required_only_for_student(self):
        computer, batch, plan = causvid_computer_and_batch()
        del batch["negative_prompt_embeds"]
        runtime = WanRoleRuntime()
        cycle = plan.update_schedule.next_cycle(TrainerCounters())
        with pytest.raises(ValueError, match="negative_prompt_embeds"):
            computer.compute_phase(cycle.requests[0], batch, runtime)
        assert computer.compute_phase(cycle.requests[1], batch, runtime).losses["fake_score"].isfinite()


class TestNamedWanExport:
    def test_shared_checkpoint_exports_only_requested_adapter_and_reloads(self, tmp_path):
        model = tiny_wan()
        config = LoraConfig(r=2, lora_alpha=6, target_modules=["to_q", "to_out.0"])
        for role in ("default", "student", "student_ema"):
            model.add_adapter(config, adapter_name=role)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".student." in name:
                    parameter.fill_(0.5)
                elif ".student_ema." in name:
                    parameter.fill_(0.25)
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        torch.save(model.state_dict(), checkpoint / "model_world_size_1_rank_0.pt")
        (checkpoint / "fsdp_config.json").write_text(json.dumps({"world_size": 1}))
        (checkpoint / "lora_train_meta.json").write_text(json.dumps({"r": 2, "lora_alpha": 6}))
        output = tmp_path / "inference"
        export_fsdp_lora_adapter(checkpoint, output, adapter_name="student")
        exported = {
            key.removeprefix("base_model.model."): value
            for key, value in load_file(output / "adapter_model.safetensors").items()
        }
        expected = get_peft_model_state_dict(model, adapter_name="student")
        assert set(exported) == set(expected)
        assert all(torch.equal(tensor, expected[name]) for name, tensor in exported.items())
        reloaded = tiny_wan()
        reloaded.add_adapter(LoraConfig.from_pretrained(output), adapter_name="inference")
        result = set_peft_model_state_dict(reloaded, exported, adapter_name="inference")
        assert result.unexpected_keys == []
        assert set(get_peft_model_state_dict(reloaded, adapter_name="inference")) == set(expected)
        loader = LoRAAdapterMixin()
        loader.model_config = SimpleNamespace(
            lora_adapter_path=str(output), use_shm=False, policy_state_adapters=("default", "student", "student_ema")
        )
        loaded = loader._build_lora_module(tiny_wan())
        assert loaded.peft_config["default"].lora_alpha == 6
        loaded_weights = get_peft_model_state_dict(loaded, adapter_name="default")
        assert set(loaded_weights) == set(expected)
        for name, tensor in loaded_weights.items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="No lora_ keys"):
            export_fsdp_lora_adapter(checkpoint, tmp_path / "missing", adapter_name="missing")

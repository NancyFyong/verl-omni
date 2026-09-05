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
"""CPU contract tests for the Qwen-Image DMD/DMD2 phase runner."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase, DistributionMatchingModelAdapter
from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import QwenImageDistributionMatching
from verl_omni.pipelines.qwen_image_distillation.phase_runner import (
    QwenImageConditionProvider,
    QwenImageDMDPhaseRunner,
    build_qwen_dmd_sigmas,
)
from verl_omni.pipelines.qwen_image_distillation.vllm_omni_rollout_adapter import QwenImageDMDPipeline
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest
from verl_omni.trainer.diffusion.distillation.equations import ode_euler_step
from verl_omni.trainer.diffusion.distillation.recipes import build_plan


class ToyQwenTransformer(torch.nn.Module):
    def __init__(self, scale: float, *, trainable: bool = True) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale), requires_grad=trainable)
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=False)

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        del kwargs
        condition = encoder_hidden_states.float().mean().to(hidden_states.dtype)
        return (hidden_states * self.scale + condition,)


class ToyEngine:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0]))

    def get_data_parallel_rank(self) -> int:
        return 0


class ToyRuntime:
    def __init__(self) -> None:
        self.modules = {
            "student": ToyQwenTransformer(0.2),
            "fake_score": ToyQwenTransformer(0.4),
            "teacher_score": ToyQwenTransformer(0.7, trainable=False),
        }
        self.engines = {role: ToyEngine(module) for role, module in self.modules.items()}

    def engine_for_role(self, role: str):
        return self.engines[role]

    def scheduler_for_role(self, role: str):
        return self.engines[role].scheduler

    @contextmanager
    def use_role(self, role: str, *, grad_enabled=None):
        module = self.modules[role]
        enabled = bool(grad_enabled and module.scale.requires_grad)
        with torch.set_grad_enabled(enabled):
            yield module


def model_config(algorithm: str = "dmd2"):
    return SimpleNamespace(
        architecture="QwenImagePipeline",
        algorithm=algorithm,
        external_lib=None,
        path="/unused",
        local_path="/unused",
        transformer_config={"in_channels": 4},
        pipeline=SimpleNamespace(
            height=16,
            width=16,
            num_inference_steps=2,
            max_sequence_length=8,
            guidance_scale=None,
        ),
    )


def make_plan(name: str = "dmd2", **overrides):
    config = {
        "model_path": "/unused",
        "conditioning_provider": "precomputed",
        "rng_seed": 17,
        **overrides,
    }
    return build_plan(name, config, frozenset({"distribution_matching"}))


def phase_batch(batch_size: int = 1, *, regression: bool = False) -> TensorDict:
    data = {
        "dummy_tensor": torch.zeros(batch_size, 1),
        "prompt_embeds": torch.ones(batch_size, 2, 3),
        "prompt_embeds_mask": torch.ones(batch_size, 2, dtype=torch.long),
        "negative_prompt_embeds": torch.zeros(batch_size, 2, 3),
        "negative_prompt_embeds_mask": torch.ones(batch_size, 2, dtype=torch.long),
    }
    batch = tu.get_tensordict(data)
    if regression:
        batch["reference_noise"] = torch.full((batch_size, 1, 4), 0.25)
        batch["teacher_target_latents"] = torch.zeros(batch_size, 1, 4)
        tu.assign_non_tensor(batch, teacher_sampling_manifest={"scheduler": "fixture"})
    return batch


def phase_request(kind: str) -> PhaseRequest:
    role = "student" if kind == "student" else "fake_score"
    return PhaseRequest(
        kind=kind,
        global_step=0,
        repeat_index=0,
        batch_policy="fresh",
        trainable_roles=(role,),
        update_ema=kind == "student",
    )


class ToyPromptTokenizer:
    def __init__(self):
        self.rendered = []

    def __call__(self, texts, **kwargs):
        width = max(len(text) for text in texts)
        ids = torch.tensor([[len(text)] * len(text) + [0] * (width - len(text)) for text in texts])
        return SimpleNamespace(input_ids=ids, attention_mask=ids.ne(0).long())


class ToyTextEncoder(torch.nn.Module):
    dtype = torch.float32

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=(input_ids.float().unsqueeze(-1),))


class ToyConditionPipeline:
    prompt_template_encode = "x" * 34 + "{}"
    prompt_template_encode_start_idx = 34
    device = torch.device("cpu")

    def __init__(self):
        from diffusers import QwenImagePipeline

        self.tokenizer = ToyPromptTokenizer()
        self.text_encoder = ToyTextEncoder()
        self._extract_masked_hidden = QwenImagePipeline._extract_masked_hidden.__get__(self)


class ToyVAE(torch.nn.Module):
    config = SimpleNamespace(z_dim=1, latents_mean=[0.0], latents_std=[1.0])

    def decode(self, latent, return_dict=False):
        return (latent.repeat(1, 3, 1, 1, 1),)


class ToyLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).square().flatten(1).mean(1)


def broadcast_selected_exit(value, src):
    assert src == 0
    value.fill_(1)


def capture_qwen_forward(self, req, *args, **kwargs):
    return self.rollout_timestep_shift, kwargs


class TestQwenImageDistillationRegistry:
    @pytest.mark.parametrize("algorithm", ["dmd", "dmd2"])
    def test_registered_for_distribution_matching_algorithms(self, algorithm):
        config = model_config(algorithm)
        adapter = DiffusionModelBase.get_class(config)
        assert adapter is QwenImageDistributionMatching
        assert issubclass(adapter, DistributionMatchingModelAdapter)

    @pytest.mark.parametrize("algorithm", ["dmd", "dmd2"])
    def test_rollout_adapter_registered_for_inference(self, algorithm):
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase

        assert VllmOmniPipelineBase.get_class("QwenImagePipeline", algorithm) is QwenImageDMDPipeline


class TestQwenImageConditionProvider:
    @pytest.mark.parametrize("description", ["cat", "a red apple", ""])
    def test_single_user_chat_uses_the_same_qwen_prefix_as_plain_text(self, description):
        tokenizer = Mock(return_value=SimpleNamespace(input_ids=torch.ones(1, 40), attention_mask=torch.ones(1, 40)))
        pipeline = SimpleNamespace(
            tokenizer=tokenizer,
            prompt_template_encode="fixed-qwen-system:{}:assistant",
            prompt_template_encode_start_idx=34,
        )
        provider = QwenImageConditionProvider("/unused", "local_frozen_encoder", 1024, " ")
        for row in (description, [{"role": "user", "content": description}]):
            provider.tokenize_rows(pipeline, [row], torch.device("cpu"))
            assert tokenizer.call_args.args[0] == [pipeline.prompt_template_encode.format(description)]
        tokenizer.apply_chat_template.assert_not_called()

    @pytest.mark.parametrize(
        "row", [[], [{"role": "system", "content": "custom"}], [{"role": "assistant", "content": "cat"}]]
    )
    def test_unsupported_chat_fails_before_encoding(self, row):
        pipeline = Mock()
        provider = QwenImageConditionProvider("/unused", "local_frozen_encoder", 1024, " ")
        with pytest.raises(ValueError, match="single user message"):
            provider.tokenize_rows(pipeline, [row], torch.device("cpu"))
        pipeline.tokenizer.assert_not_called()

    def test_raw_chat_is_rendered_once_and_negative_prompt_is_encoded(self):
        provider = QwenImageConditionProvider("/unused", "local_frozen_encoder", 8, " ")
        provider.pipeline = ToyConditionPipeline()
        batch = tu.get_tensordict(tensor_dict={"dummy_tensor": torch.zeros(1, 1)})
        tu.assign_non_tensor_stack(
            batch,
            "raw_prompt",
            [[{"role": "user", "content": "prompt"}]],
        )

        positive, negative = provider.encode(
            batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
            require_negative=True,
        )

        assert provider.pipeline.tokenizer.rendered == []
        assert positive.tensors["prompt_embeds"].shape[0] == 1
        assert negative.tensors["prompt_embeds"].shape[0] == 1
        assert positive.tensors["prompt_embeds"].shape[1] == len("prompt")
        assert positive.tensors["prompt_embeds"][0, 0, 0].item() == 34 + len("prompt")
        assert negative.tensors["prompt_embeds"][0, 0, 0].item() == 35


class TestQwenImageDMDPhaseRunner:
    def test_rollout_exit_is_broadcast_across_sharded_and_sequence_parallel_ranks(self, monkeypatch):
        runtime = ToyRuntime()
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan())
        broadcast = Mock(side_effect=broadcast_selected_exit)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
        assert runner.sample_rollout_exit(4, torch.device("cpu"), runtime) == 1
        broadcast.assert_called_once()

    def test_continuous_score_sampling_is_uniform_inside_bounds_without_shift(self, monkeypatch):
        runner = QwenImageDMDPhaseRunner(
            model_config(),
            make_plan(score_discrete_steps=0, score_sigma_min=0.2, score_sigma_max=0.6, score_timestep_shift=8.0),
        )
        monkeypatch.setattr(torch, "rand", Mock(return_value=torch.tensor([0.0, 0.25, 0.5, 0.75])))
        sigma = runner.sample_score_sigma(torch.zeros(4, 1, 4), ToyRuntime())
        torch.testing.assert_close(sigma, torch.tensor([0.2, 0.3, 0.4, 0.5]))

    def test_training_and_inference_shifts_must_match(self):
        config = model_config()
        config.algo = SimpleNamespace(rollout_timestep_shift=4.0)
        with pytest.raises(ValueError, match="must match"):
            QwenImageDMDPhaseRunner(config, make_plan(fake_update_ratio=1))

    def test_student_phase_keeps_only_student_gradient(self, monkeypatch):
        runtime = ToyRuntime()
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan(fake_update_ratio=1))
        monkeypatch.setattr(
            runner,
            "rollout_sigmas",
            lambda scheduler, height, width, device: torch.tensor([1.0, 0.5, 0.0], device=device),
        )

        computation = runner.compute_phase(phase_request("student"), phase_batch(), runtime)
        computation.losses["student"].backward()

        assert computation.losses["student"].requires_grad
        assert runtime.modules["student"].scale.grad is not None
        assert runtime.modules["fake_score"].scale.grad is None
        assert runtime.modules["teacher_score"].scale.grad is None
        assert 0.02 <= computation.metrics["score/sigma"] <= 1.0

    def test_fake_phase_detaches_student_rollout(self, monkeypatch):
        runtime = ToyRuntime()
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan(fake_update_ratio=1))
        monkeypatch.setattr(
            runner,
            "rollout_sigmas",
            lambda scheduler, height, width, device: torch.tensor([1.0, 0.5, 0.0], device=device),
        )

        batch = phase_batch()
        del batch["negative_prompt_embeds"]
        del batch["negative_prompt_embeds_mask"]
        computation = runner.compute_phase(phase_request("fake_score"), batch, runtime)
        computation.losses["fake_score"].backward()

        assert runtime.modules["fake_score"].scale.grad is not None
        assert runtime.modules["student"].scale.grad is None
        assert runtime.modules["teacher_score"].scale.grad is None

    def test_original_dmd_adds_paired_latent_regression(self):
        runtime = ToyRuntime()
        runner = QwenImageDMDPhaseRunner(
            model_config("dmd"),
            make_plan("dmd", regression_type="latent_mse", regression_loss_weight=2.0),
        )

        computation = runner.compute_phase(phase_request("student"), phase_batch(regression=True), runtime)

        assert computation.losses["student"].requires_grad
        assert computation.metrics["regression/loss"] >= 0
        assert computation.metrics["dmd/loss"] >= 0

    def test_four_step_rollout_matches_reference_linear_shift(self):
        config = model_config()
        config.pipeline.num_inference_steps = 4
        runner = QwenImageDMDPhaseRunner(config, make_plan(fake_update_ratio=1))

        sigmas = runner.rollout_sigmas(None, 16, 16, torch.device("cpu"))

        torch.testing.assert_close(sigmas, torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]))
        torch.testing.assert_close(sigmas, build_qwen_dmd_sigmas(4, 3.0))

    def test_vllm_rollout_accepts_request_local_shift_and_restores_state(self, monkeypatch):
        from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

        pipeline = object.__new__(QwenImageDMDPipeline)
        pipeline.rollout_timestep_shift = 3.0
        request = SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={"rollout_timestep_shift": 4.0}),
        )
        monkeypatch.setattr(QwenImagePipelineWithLogProb, "forward", capture_qwen_forward)

        observed_shift, captured = pipeline.forward(request)
        assert observed_shift == 4.0
        assert pipeline.rollout_timestep_shift == 3.0
        assert captured["noise_level"] == 0.0
        assert captured["logprobs"] is False
        assert captured["true_cfg_scale"] == 1.0

    def test_vllm_rollout_rejects_stochastic_sampling(self):
        pipeline = object.__new__(QwenImageDMDPipeline)
        request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={"noise_level": 0.1}))
        with pytest.raises(ValueError, match="noise_level=0"):
            pipeline.forward(request)

    def test_vllm_schedule_uses_identical_training_sigmas(self):
        pipeline = object.__new__(QwenImageDMDPipeline)
        pipeline.rollout_timestep_shift = 3.0
        pipeline._components = {}
        pipeline.device = torch.device("cpu")
        pipeline.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000})

        timesteps, count = pipeline.prepare_timesteps(4, None, image_seq_len=4096)

        assert count == 4
        torch.testing.assert_close(pipeline.scheduler.sigmas, build_qwen_dmd_sigmas(4, 3.0))
        torch.testing.assert_close(timesteps, torch.tensor([1000.0, 900.0, 750.0, 500.0]))

    def test_vllm_deterministic_step_matches_training_euler(self):
        scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        )
        pipeline = object.__new__(QwenImageDMDPipeline)
        pipeline.rollout_timestep_shift = 3.0
        pipeline._components = {}
        pipeline.device = torch.device("cpu")
        pipeline.scheduler = scheduler
        timesteps, _ = pipeline.prepare_timesteps(4, None, image_seq_len=4096)
        scheduler.set_begin_index(0)
        sample = torch.randn(1, 4, 8)
        velocity = torch.randn_like(sample)

        inference = scheduler.step(
            velocity,
            timesteps[0],
            sample,
            noise_level=0.0,
            return_logprobs=False,
        ).prev_sample
        training = ode_euler_step(sample, velocity, scheduler.sigmas[0], scheduler.sigmas[1])

        torch.testing.assert_close(inference, training)

    def test_training_rollout_matches_vllm_deterministic_latents(self, monkeypatch):
        runtime = ToyRuntime()
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan(fake_update_ratio=1))
        monkeypatch.setattr(runner, "sample_rollout_exit", lambda high, device, runtime: high - 1)
        condition, _ = runner.condition_provider.encode(
            phase_batch(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            require_negative=False,
        )
        initial = torch.randn(1, 1, 4)

        training_x0, _, _, _ = runner.rollout(
            runtime,
            condition,
            initial.clone(),
            height=16,
            width=16,
            grad_enabled=False,
        )
        scheduler = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        )
        pipeline = object.__new__(QwenImageDMDPipeline)
        pipeline.rollout_timestep_shift = 3.0
        pipeline._components = {}
        pipeline.device = torch.device("cpu")
        pipeline.scheduler = scheduler
        timesteps, _ = pipeline.prepare_timesteps(2, None, image_seq_len=1)
        scheduler.set_begin_index(0)
        inference = initial.clone()
        for index, timestep in enumerate(timesteps):
            sigma = scheduler.sigmas[index].reshape(1)
            velocity = runner.predict_velocity(
                runtime,
                "student",
                inference,
                sigma,
                condition,
                height=16,
                width=16,
                grad_enabled=False,
            )
            inference = scheduler.step(
                velocity,
                timestep,
                inference,
                noise_level=0.0,
                return_logprobs=False,
            ).prev_sample

        torch.testing.assert_close(training_x0, inference)

    def test_decoded_lpips_regression_retains_student_gradient(self, monkeypatch):
        runner = QwenImageDMDPhaseRunner(
            model_config("dmd"),
            make_plan("dmd", regression_type="decoded_lpips"),
        )

        monkeypatch.setattr(runner, "ensure_vae_and_lpips", Mock(return_value=(ToyVAE(), ToyLPIPS())))
        prediction = torch.randn(1, 1, 4, requires_grad=True)
        target = torch.zeros_like(prediction)

        loss = runner.decoded_lpips_loss(prediction, target, None, height=16, width=16)
        loss.backward()

        assert loss.ndim == 0
        assert prediction.grad is not None
        assert torch.count_nonzero(prediction.grad) > 0

    def test_rng_state_round_trip_replays_next_phase(self, monkeypatch):
        plan = make_plan(fake_update_ratio=1)
        runner = QwenImageDMDPhaseRunner(model_config(), plan)
        restored = QwenImageDMDPhaseRunner(model_config(), plan)
        for instance in (runner, restored):
            monkeypatch.setattr(
                instance,
                "rollout_sigmas",
                lambda scheduler, height, width, device: torch.tensor([1.0, 0.5, 0.0], device=device),
            )
        runtime = ToyRuntime()
        runner.compute_phase(phase_request("fake_score"), phase_batch(), runtime)
        state = runner.state_dict()
        expected = runner.compute_phase(phase_request("fake_score"), phase_batch(), runtime)

        restored.load_state_dict(state)
        actual = restored.compute_phase(phase_request("fake_score"), phase_batch(), runtime)

        torch.testing.assert_close(actual.losses["fake_score"], expected.losses["fake_score"])
        expected_values = {key: value for key, value in expected.metrics.items() if not key.startswith("perf/")}
        actual_values = {key: value for key, value in actual.metrics.items() if not key.startswith("perf/")}
        assert actual_values == expected_values

    def test_rejects_physical_batch_larger_than_one(self):
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan(fake_update_ratio=1))
        with pytest.raises(ValueError, match="physical micro-batch size 1"):
            runner.compute_phase(phase_request("student"), phase_batch(batch_size=2), ToyRuntime())

    def test_precomputed_provider_requires_negative_condition(self):
        batch = phase_batch()
        del batch["negative_prompt_embeds"]
        runner = QwenImageDMDPhaseRunner(model_config(), make_plan(fake_update_ratio=1))
        with pytest.raises(ValueError, match="negative_prompt_embeds"):
            runner.compute_phase(phase_request("student"), batch, ToyRuntime())

    def test_runner_rejects_dmd2_adversarial_profile(self):
        plan = build_plan(
            "dmd2",
            {"model_path": "/unused", "conditioning_provider": "precomputed", "profile": "paper"},
            frozenset({"distribution_matching", "adversarial"}),
        )
        with pytest.raises(NotImplementedError, match="adversarial profile"):
            QwenImageDMDPhaseRunner(model_config(), plan)

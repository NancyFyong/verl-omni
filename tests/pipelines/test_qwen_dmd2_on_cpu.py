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

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_dmd2.diffusers_training_adapter import (
    QwenImageConditionProvider,
    QwenImageDMD2,
    load_qwen_dmd2_adapter,
    qwen_dmd2_base_provenance,
)
from verl_omni.workers.config import DiffusionDMDConfig


class ToyPromptTokenizer:
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


class ToyVelocityPredictor:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def __call__(self, sample, sigma, *, grad_enabled):
        with torch.set_grad_enabled(grad_enabled):
            prediction = sample * self.weight + sigma
        self.calls.append((sample.detach().clone(), sigma, grad_enabled, prediction.requires_grad))
        return prediction


class TestQwenDMDSampling:
    @pytest.mark.parametrize("steps,exit_index", [(1, 0), (4, 0), (4, 1), (4, 2), (4, 3)])
    @pytest.mark.parametrize("grad_enabled", [False, True])
    def test_fixed_exit_matches_pre_refactor_euler_and_gradient(self, steps, exit_index, grad_enabled, monkeypatch):
        monkeypatch.setattr(torch.distributed, "broadcast", Mock(side_effect=AssertionError("engine owns collectives")))
        config = SimpleNamespace(pipeline=SimpleNamespace(num_inference_steps=steps))
        sigmas = QwenImageDMD2.sampling_sigmas(config, DiffusionDMDConfig(), "cpu")
        noise = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
        before = noise.clone()
        predictor = ToyVelocityPredictor()
        global_rng = torch.get_rng_state().clone()

        result = QwenImageDMD2.sample_student(
            noise=noise, sigmas=sigmas, exit_index=exit_index, predict=predictor, grad_enabled=grad_enabled
        )

        sample = noise
        for index in range(exit_index):
            velocity = sample * 0.2 + sigmas[index]
            sample = sample + (sigmas[index + 1] - sigmas[index]) * velocity
        expected = sample - sigmas[exit_index] * (sample * 0.2 + sigmas[exit_index])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(noise, before, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), global_rng)
        assert result.dtype == torch.float32
        assert [call[2] for call in predictor.calls] == [False] * exit_index + [grad_enabled]
        assert [call[3] for call in predictor.calls] == [False] * exit_index + [grad_enabled]
        assert result.requires_grad == grad_enabled
        if grad_enabled:
            result.sum().backward()
            torch.testing.assert_close(predictor.weight.grad, (-sigmas[exit_index] * sample).sum())
        else:
            assert predictor.weight.grad is None

    @pytest.mark.parametrize("discrete_steps", [0, 37, 1000])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_score_corruption_matches_pre_refactor_formula_and_rng(self, discrete_steps, dtype):
        config = DiffusionDMDConfig(score_discrete_steps=discrete_steps)
        total = discrete_steps or 1000
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=total))
        generated = torch.linspace(-1, 1, 48, dtype=dtype).reshape(4, 3, 4).requires_grad_()
        sigma_generator = torch.Generator().manual_seed(10)
        noise_generator = torch.Generator().manual_seed(11)
        saved_sigma, saved_noise = sigma_generator.get_state().clone(), noise_generator.get_state().clone()
        global_rng = torch.get_rng_state().clone()

        result = QwenImageDMD2.prepare_score_inputs(
            generated, scheduler, config, sigma_generator=sigma_generator, noise_generator=noise_generator
        )

        expected_sigma_rng = torch.Generator().set_state(saved_sigma)
        expected_noise_rng = torch.Generator().set_state(saved_noise)
        if discrete_steps:
            timestep = torch.randint(total, (4,), generator=expected_sigma_rng)
            frac = timestep.float() / total
            sigma = config.score_timestep_shift * frac / (1 + (config.score_timestep_shift - 1) * frac) * total
            sigma = (sigma / total).clamp(config.score_sigma_min, config.score_sigma_max)
        else:
            sigma = torch.rand(4, generator=expected_sigma_rng)
            sigma = config.score_sigma_min + (config.score_sigma_max - config.score_sigma_min) * sigma
        noise = torch.randn(generated.shape, dtype=torch.float32, generator=expected_noise_rng)
        expanded = sigma.reshape(-1, 1, 1)
        expected = (1 - expanded) * generated.detach().float() + expanded * noise
        for actual, target in zip(result, (expected, noise, sigma), strict=True):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            assert actual.dtype == torch.float32 and not actual.requires_grad
        assert generated.grad is None
        assert torch.equal(sigma_generator.get_state(), expected_sigma_rng.get_state())
        assert torch.equal(noise_generator.get_state(), expected_noise_rng.get_state())
        assert torch.equal(torch.get_rng_state(), global_rng)
        sigma_generator.set_state(saved_sigma)
        noise_generator.set_state(saved_noise)
        replay = QwenImageDMD2.prepare_score_inputs(
            generated, scheduler, config, sigma_generator=sigma_generator, noise_generator=noise_generator
        )
        for actual, target in zip(replay, result, strict=True):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)

    def test_mismatched_score_grid_fails_before_consuming_noise(self):
        sigma_generator = torch.Generator().manual_seed(10)
        noise_generator = torch.Generator().manual_seed(11)
        sigma_state, noise_state = sigma_generator.get_state().clone(), noise_generator.get_state().clone()
        with pytest.raises(ValueError, match="score_discrete_steps"):
            QwenImageDMD2.prepare_score_inputs(
                torch.ones(2, 3, 4),
                SimpleNamespace(config=SimpleNamespace(num_train_timesteps=37)),
                DiffusionDMDConfig(),
                sigma_generator=sigma_generator,
                noise_generator=noise_generator,
            )
        assert torch.equal(sigma_generator.get_state(), sigma_state)
        assert torch.equal(noise_generator.get_state(), noise_state)


class TestQwenDMD2:
    def test_base_provenance_is_owned_by_the_qwen_integration(self, tmp_path):
        revision = "a" * 40
        root = tmp_path / "snapshots" / revision
        (root / "transformer").mkdir(parents=True)
        config = b'{"in_channels": 64}'
        (root / "transformer/config.json").write_bytes(config)

        value = qwen_dmd2_base_provenance(root)

        assert value == {
            "base_model_revision": revision,
            "base_transformer_config_sha256": hashlib.sha256(config).hexdigest(),
        }

    def test_peft_export_loader_is_owned_by_the_qwen_integration(self, tmp_path):
        from safetensors.torch import save_file

        save_file(
            {"base_model.model.transformer.block.lora_A.weight": torch.ones(2)}, tmp_path / "adapter_model.safetensors"
        )
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2}))
        module = SimpleNamespace(load_lora_adapter=Mock())

        load_qwen_dmd2_adapter(module, tmp_path, "student")

        (weights,) = module.load_lora_adapter.call_args.args
        assert set(weights) == {"block.lora_A.weight"}
        assert module.load_lora_adapter.call_args.kwargs == {
            "adapter_name": "student",
            "prefix": None,
            "metadata": {"r": 2},
        }

    def test_registry_does_not_claim_original_dmd_or_edit_support(self):
        assert DiffusionModelBase.get_class_by_name("QwenImagePipeline", "dmd2") is QwenImageDMD2
        with pytest.raises(NotImplementedError):
            DiffusionModelBase.get_class_by_name("QwenImagePipeline", "dmd")
        with pytest.raises(NotImplementedError):
            DiffusionModelBase.get_class_by_name("QwenImageEditPlusPipeline", "dmd2")

    def test_short_prompts_keep_tokens_after_real_prefix_removal(self):
        provider = QwenImageConditionProvider("unused", 64, " ")
        provider.pipeline = ToyConditionPipeline()
        for size in (1, 3, 2):
            batch = TensorDict({"dummy_tensor": torch.zeros(size, 1)}, batch_size=[size])
            tu.assign_non_tensor_stack(batch, "raw_prompt", [[{"role": "user", "content": "cat"}]] * size)
            positive, negative = provider.encode(
                batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=True
            )
            assert positive["prompt_embeds"].shape == (size, 3, 1)
            assert negative["prompt_embeds"].shape == (size, 1, 1)
            assert positive["prompt_embeds"][0, 0, 0] == 37
            assert not positive["prompt_embeds"].requires_grad

    @pytest.mark.parametrize(
        "row", [[], [{"role": "system", "content": "custom"}], [{"role": "assistant", "content": "cat"}]]
    )
    def test_invalid_chat_is_not_generic_chat_formatted(self, row):
        provider = QwenImageConditionProvider("unused", 64, " ")
        with pytest.raises(ValueError, match="single user"):
            provider.tokenize_rows(Mock(), [row], torch.device("cpu"))

    def test_precomputed_inputs_do_not_load_encoder_for_fake_stage(self):
        provider = QwenImageConditionProvider("unused", 2, " ")
        batch = TensorDict({"prompt_embeds": torch.ones(2, 3, 4, requires_grad=True)}, batch_size=[2])
        positive, negative = provider.encode(
            batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=False
        )
        assert provider.pipeline is None and negative is None
        assert positive["prompt_embeds"].shape == (2, 2, 4)
        assert positive["prompt_embeds_mask"].shape == (2, 2)
        assert not positive["prompt_embeds"].requires_grad
        with pytest.raises(ValueError, match="negative_prompt_embeds"):
            provider.encode(batch, device=torch.device("cpu"), dtype=torch.float32, require_negative=True)

    def test_geometry_uses_vae_config_and_packs_consistently(self, tmp_path):
        (tmp_path / "vae").mkdir()
        (tmp_path / "vae" / "config.json").write_text(json.dumps({"z_dim": 4, "temperal_downsample": [False, True]}))
        model = SimpleNamespace(config=SimpleNamespace(in_channels=16))
        config = SimpleNamespace(local_path=str(tmp_path), pipeline=SimpleNamespace(height=32, width=48))
        shape, geometry = QwenImageDMD2.latent_geometry(model, config, TensorDict({}, batch_size=[2]))
        assert shape == (2, 4, 1, 8, 12)
        assert geometry["vae_scale_factor"] == 4
        latents = QwenImageDMD2.pack_latents(torch.ones(shape))
        assert latents.shape == (2, 24, 16)
        batch = TensorDict({"height": torch.tensor([32, 64])}, batch_size=[2])
        with pytest.raises(ValueError, match="homogeneous"):
            QwenImageDMD2.latent_geometry(model, config, batch)

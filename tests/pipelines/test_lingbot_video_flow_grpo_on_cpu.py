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

"""CPU contract tests for LingBot Dense T2V FlowGRPO helpers and adapter."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.lingbot_video_flow_grpo.common import (
    apply_cfg,
    apply_prompt_template,
    caption_to_json,
    shifted_sigmas,
    validate_t2v_dimensions,
)
from verl_omni.pipelines.lingbot_video_flow_grpo.diffusers_training_adapter import LingBotVideoDenseFlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase


def test_caption_serialization_and_template_are_officially_compact():
    caption = {"subject": "猫", "motion": ["walks", "turns"]}
    encoded = caption_to_json(caption)
    assert encoded == '{"subject":"猫","motion":["walks","turns"]}'
    assert encoded in apply_prompt_template(encoded)
    with pytest.raises(ValueError, match="structured JSON"):
        caption_to_json("a plain text prompt")


@pytest.mark.parametrize("height,width,num_frames", [(480, 832, 121), (16, 16, 1), (512, 512, 81)])
def test_t2v_dimensions_accept_official_shapes(height, width, num_frames):
    validate_t2v_dimensions(height, width, num_frames)


@pytest.mark.parametrize("height,width,num_frames", [(480, 832, 120), (481, 832, 121), (480, 831, 121)])
def test_t2v_dimensions_reject_invalid_shapes(height, width, num_frames):
    with pytest.raises(ValueError):
        validate_t2v_dimensions(height, width, num_frames)


def test_shifted_sigmas_match_flow_matching_shift_contract():
    sigmas = shifted_sigmas(4, 3.0)
    assert sigmas.shape == (4,)
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == pytest.approx(0.5)
    assert all(left > right for left, right in zip(sigmas[:-1], sigmas[1:], strict=True))


def test_cfg_formula_matches_official_lingbot_rule():
    positive = torch.tensor([3.0, -1.0])
    negative = torch.tensor([1.0, 2.0])
    assert torch.equal(apply_cfg(positive, negative, 3.0), torch.tensor([7.0, -7.0]))


def test_dense_adapter_is_registered_without_optional_lingbot_package():
    cfg = SimpleNamespace(architecture="LingBotVideoPipeline", algorithm="flow_grpo", external_lib=None)
    assert DiffusionModelBase.get_class(cfg) is LingBotVideoDenseFlowGRPO


def test_adapter_builds_lingbot_transformer_inputs_and_cfg_pair():
    model_config = SimpleNamespace(pipeline=SimpleNamespace(guidance_scale=3.0, true_cfg_scale=1.0))
    latents = torch.randn(2, 3, 16, 2, 4, 4)
    timesteps = torch.tensor([[1000.0, 500.0, 0.0], [1000.0, 500.0, 0.0]])
    embeds = torch.randn(2, 5, 2560)
    mask = torch.ones(2, 5, dtype=torch.long)
    model_inputs, negative_model_inputs = LingBotVideoDenseFlowGRPO.prepare_model_inputs(
        None,
        model_config,
        latents,
        timesteps,
        embeds,
        mask,
        embeds + 1,
        mask,
        None,
        step=1,
    )
    assert torch.equal(model_inputs["hidden_states"], latents[:, 1])
    assert torch.equal(model_inputs["timestep"], torch.tensor([500.0, 500.0]))
    assert negative_model_inputs is not None
    assert torch.equal(negative_model_inputs["encoder_hidden_states"], embeds + 1)


def test_manual_lora_applies_and_deactivates_standard_linear_layers():
    from verl_omni.pipelines.lingbot_video_flow_grpo.manual_lora import ManualLinearLoRAManager

    transformer = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    with torch.no_grad():
        transformer[0].weight.zero_()
    manager = ManualLinearLoRAManager(transformer)
    request = SimpleNamespace(
        lora_int_id=7,
        peft_config={"lora_alpha": 4},
        lora_tensors={
            "base_model.model.transformer.0.lora_A.default.weight": torch.tensor([[1.0, 2.0, 3.0]]),
            "base_model.model.transformer.0.lora_B.default.weight": torch.tensor([[5.0], [7.0]]),
        },
    )

    manager.add_adapter(request)
    assert manager.list_adapters() == [7]

    x = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.equal(transformer(x), torch.zeros(1, 2))
    manager.set_active_adapter(request, lora_scale=0.5)
    # alpha / rank * external_scale = 4 / 1 * 0.5 = 2; A(x)=6; B(A(x))=[30,42].
    assert torch.equal(transformer(x), torch.tensor([[60.0, 84.0]]))

    manager.set_active_adapter(None)
    assert torch.equal(transformer(x), torch.zeros(1, 2))
    assert manager.remove_adapter(7) is True
    assert manager.list_adapters() == []


def test_manual_lora_rejects_unmatched_rollout_modules():
    from verl_omni.pipelines.lingbot_video_flow_grpo.manual_lora import ManualLinearLoRAManager

    manager = ManualLinearLoRAManager(torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)))
    request = SimpleNamespace(
        lora_int_id=8,
        peft_config={"lora_alpha": 1},
        lora_tensors={
            "transformer.missing.lora_A.default.weight": torch.zeros(1, 3),
            "transformer.missing.lora_B.default.weight": torch.zeros(2, 1),
        },
    )
    with pytest.raises(ValueError, match="No LingBot nn.Linear LoRA tensors matched"):
        manager.add_adapter(request)


def test_custom_pipeline_lora_proxy_routes_worker_lifecycle_calls():
    from verl_omni.workers.rollout.vllm_rollout.utils import (
        _PipelineLoRAProxy,
        _supports_pipeline_lora,
        vLLMOmniColocateWorkerExtension,
    )

    class Pipeline:
        def __init__(self):
            self.calls = []
            self.adapters = []

        def add_lora(self, request):
            self.calls.append(("add", request))
            self.adapters.append(request.lora_int_id)
            return True

        def remove_lora(self, adapter_id):
            self.calls.append(("remove", adapter_id))
            self.adapters.remove(adapter_id)
            return True

        def list_loras(self):
            return self.adapters

        def pin_lora(self, adapter_id):
            self.calls.append(("pin", adapter_id))
            return adapter_id in self.adapters

        def set_active_lora(self, request, scale):
            self.calls.append(("active", request, scale))

    pipeline = Pipeline()
    request = SimpleNamespace(lora_int_id=42)
    assert _supports_pipeline_lora(pipeline)
    proxy = _PipelineLoRAProxy(pipeline)
    assert proxy.add_adapter(request)
    assert proxy.list_adapters() == [42]
    assert proxy.pin_adapter(42)
    proxy.set_active_adapter(request, 0.5)
    assert proxy.remove_adapter(42)
    assert pipeline.calls == [
        ("add", request),
        ("pin", 42),
        ("active", request, 0.5),
        ("remove", 42),
    ]

    class Worker:
        def _get_custom_lora_pipeline(self):
            return pipeline

    worker = Worker()
    vLLMOmniColocateWorkerExtension.init_lora_manager(worker)
    assert isinstance(worker.lora_manager, _PipelineLoRAProxy)


def test_async_server_reads_trajectory_payload_from_multimodal_output():
    """The consumer must read payload["trajectory"] via multimodal_output.

    vLLM-Omni main removed ``DiffusionOutput.custom_output`` (#4922); the engine
    copies ``output["payload"]["trajectory"]`` into
    ``OmniRequestOutput.multimodal_output["trajectory"]`` and the formatter never
    repopulates ``custom_output``. This test drives ``_process_output`` with the
    post-#4922 shape and asserts the training-facing keys survive.
    """

    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

    video = torch.rand(5, 3, 8, 8)
    trajectory = {
        "all_latents": torch.randn(1, 3, 4, 2, 1, 1),
        "all_log_probs": torch.randn(1, 2),
        "all_timesteps": torch.tensor([[750.0, 500.0]]),
        "prompt_embeds": torch.randn(1, 7, 16),
        "prompt_embeds_mask": torch.ones(1, 7, dtype=torch.long),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        # Formatter-facing duplicates that the consumer must drop.
        "latents": torch.randn(1, 3, 4, 2, 1, 1),
        "log_probs": torch.randn(1, 2),
        "timesteps": torch.tensor([[750.0, 500.0]]),
    }
    final_res = SimpleNamespace(
        images=[video],
        custom_output={},
        multimodal_output={"trajectory": trajectory, "metadata": {"trajectory": {"type": "denoising"}}},
        request_output=None,
    )
    server = SimpleNamespace(
        _ar_mode=False,
        global_steps=3,
        _map_stop_reason=lambda self_reason: "stop",
        _to_tensor=None,
    )

    result = vLLMOmniHttpServer._process_output(server, final_res, params=None, sampling_params={"logprobs": True})

    assert torch.equal(result.diffusion_output, video)
    assert torch.equal(result.log_probs, trajectory["all_log_probs"][0])
    extra = result.extra_fields
    assert torch.equal(extra["all_latents"], trajectory["all_latents"][0])
    assert torch.equal(extra["all_timesteps"], trajectory["all_timesteps"][0])
    assert torch.equal(extra["prompt_embeds"], trajectory["prompt_embeds"][0])
    assert extra["negative_prompt_embeds"] is None
    assert extra["global_steps"] == 3
    # The formatter-facing duplicate keys must not leak into training data.
    assert "latents" not in extra and "log_probs" not in extra and "timesteps" not in extra

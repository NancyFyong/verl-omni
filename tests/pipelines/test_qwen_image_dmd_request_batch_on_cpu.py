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
"""Run the real request collation, RNG, Euler loop and output splitting on CPU."""

from types import MethodType

import pytest
import torch
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.qwen_image_distillation.vllm_omni_rollout_adapter import QwenImageDMDPipeline
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


class BatchTransformer(torch.nn.Module):
    in_channels = 4
    guidance_embeds = False

    def __init__(self):
        super().__init__()
        self.img_in = torch.nn.Linear(4, 4)
        self.batch_sizes = []

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, img_shapes, **kwargs):
        self.batch_sizes.append(hidden_states.shape[0])
        assert len(img_shapes) == hidden_states.shape[0]
        mask = encoder_hidden_states_mask.to(encoder_hidden_states.dtype).unsqueeze(-1)
        condition = (encoder_hidden_states * mask).sum(1) / mask.sum(1)
        return (hidden_states * 0.2 + condition.unsqueeze(1),)


def encode_tokens(self, prompt_ids, attention_mask, num_images_per_prompt, **kwargs):
    ids = torch.as_tensor(prompt_ids)
    embeds = ids.unsqueeze(-1).float().expand(-1, -1, 4) / 10
    mask = torch.as_tensor(attention_mask)
    return embeds.repeat_interleave(num_images_per_prompt, 0), mask.repeat_interleave(num_images_per_prompt, 0)


def make_pipeline():
    pipeline = object.__new__(QwenImageDMDPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._components = {}
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 2
    pipeline.transformer = BatchTransformer()
    pipeline.scheduler = FlowMatchSDEDiscreteScheduler()
    pipeline.encode_prompt = MethodType(encode_tokens, pipeline)
    return pipeline


def make_request(index, *, outputs=1, extra_args=None, **overrides):
    tokens = [index + 1] * (index + 1)
    params = dict(
        seed=41 + index,
        height=16,
        width=16,
        num_inference_steps=4,
        output_type="latent",
        true_cfg_scale=1.0,
        num_outputs_per_prompt=outputs,
        extra_args=extra_args or {},
    )
    params.update(overrides)
    return OmniDiffusionRequest(
        request_id=f"request-{index}",
        prompt={"prompt_token_ids": tokens, "prompt_mask": [1] * len(tokens)},
        sampling_params=OmniDiffusionSamplingParams(**params),
    )


class TestQwenDMDRequestBatch:
    @pytest.mark.parametrize("outputs", [1, 2])
    def test_packed_inference_matches_serial_and_preserves_request_order(self, outputs):
        pipeline = make_pipeline()
        with torch.no_grad():
            serial = [pipeline.forward(make_request(i, outputs=outputs)) for i in range(3)]
            pipeline.transformer.batch_sizes.clear()
            order = [2, 0, 1]
            packed = pipeline.forward(DiffusionRequestBatch([make_request(i, outputs=outputs) for i in order]))
        assert pipeline.supports_request_batch
        assert pipeline.transformer.batch_sizes == [3 * outputs] * 4
        assert len(packed) == 3
        for actual, index in zip(packed, order, strict=True):
            torch.testing.assert_close(actual.output["payload"], serial[index].output["payload"])
            actual_condition = actual.output["metadata"]["prompt_embeddings"]
            serial_condition = serial[index].output["metadata"]["prompt_embeddings"]
            torch.testing.assert_close(
                actual_condition["prompt_embeds"][actual_condition["prompt_embeds_mask"].bool()],
                serial_condition["prompt_embeds"][serial_condition["prompt_embeds_mask"].bool()],
            )
            torch.testing.assert_close(actual.trajectory_latents, serial[index].trajectory_latents)
            assert actual.output["payload"]["image"].shape[0] == outputs

    def test_supplied_latents_are_collated_by_the_existing_request_batch(self):
        pipeline = make_pipeline()
        requests = [make_request(i, latents=torch.full((1, 1, 4), float(i))) for i in range(2)]
        with torch.no_grad():
            packed = pipeline.forward(DiffusionRequestBatch(requests))
        for index, result in enumerate(packed):
            torch.testing.assert_close(result.trajectory_latents[:, 0], torch.full((1, 1, 4), float(index)))

    @pytest.mark.parametrize("field,value", [("height", 32), ("num_inference_steps", 2), ("true_cfg_scale", 4.0)])
    def test_scheduler_separates_incompatible_requests_and_direct_calls_fail_closed(self, field, value):
        first, second = make_request(0), make_request(1, **{field: value})
        assert build_request_batch_sampling_params_key(first) != build_request_batch_sampling_params_key(second)
        pipeline = make_pipeline()
        with pytest.raises(ValueError, match="sampling parameters"):
            pipeline.forward(DiffusionRequestBatch([first, second]))
        assert pipeline.transformer.batch_sizes == []

    @pytest.mark.parametrize("shift", [0.0, float("nan"), float("inf")])
    def test_invalid_dmd_shift_fails_before_inference(self, shift):
        with pytest.raises(ValueError, match="finite and at least 1"):
            make_pipeline().forward(make_request(0, extra_args={"rollout_timestep_shift": shift}))

    def test_mixed_dmd_shift_is_rejected_without_mutating_pipeline_state(self):
        pipeline = make_pipeline()
        requests = [make_request(0), make_request(1, extra_args={"rollout_timestep_shift": 4.0})]
        with pytest.raises(ValueError, match="same rollout_timestep_shift"):
            pipeline.forward(DiffusionRequestBatch(requests))
        assert pipeline.rollout_timestep_shift == 3.0

    def test_null_extra_args_keep_defaults_on_the_real_forward_path(self):
        pipeline = make_pipeline()
        request = make_request(0)
        request.sampling_params.extra_args = None
        with torch.no_grad():
            result = pipeline.forward(request)
        assert torch.isfinite(result.output["payload"]["image"]).all()

    def test_unvalidated_step_mode_is_not_silently_enabled(self):
        assert not QwenImageDMDPipeline.supports_step_execution
        with pytest.raises(NotImplementedError, match="step_execution=false"):
            make_pipeline().prepare_encode(None)

    def test_request_shift_is_restored_after_forward_error(self):
        pipeline = make_pipeline()
        request = make_request(0, extra_args={"rollout_timestep_shift": 4.0})
        request.prompt["prompt_token_ids"] = ["invalid"]
        with pytest.raises((TypeError, ValueError)):
            pipeline.forward(request)
        assert pipeline.rollout_timestep_shift == 3.0
        assert pipeline.transformer.batch_sizes == []

    def test_empty_request_batch_fails_before_forward(self):
        with pytest.raises(ValueError, match="empty"):
            make_pipeline().forward(DiffusionRequestBatch([]))

    def test_initial_noise_matches_training_fp32_regardless_of_embedding_dtype(self):
        pipeline = make_pipeline()
        full = pipeline.prepare_latents(1, 1, 16, 16, torch.float32, "cpu", torch.Generator().manual_seed(7))
        low = pipeline.prepare_latents(1, 1, 16, 16, torch.bfloat16, "cpu", torch.Generator().manual_seed(7))
        assert low.dtype == torch.float32
        torch.testing.assert_close(full, low, rtol=0, atol=0)

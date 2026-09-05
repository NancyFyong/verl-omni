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
"""GPU integration test of native vLLM-Omni request scheduling and Qwen DMD inference."""

import asyncio
import os
from unittest.mock import Mock

import pytest
import torch


async def collect_request(engine, request):
    output = None
    async for outputs in engine.step_streaming(request):
        assert len(outputs) == 1
        output = outputs[0]
    assert output is not None and output.finished and output.trajectory_latents is not None
    assert len(output.images) == request.sampling_params.num_outputs_per_prompt
    for key in ("preprocess_time_ms", "diffusion_engine_exec_time_ms", "postprocess_time_ms"):
        assert output.metrics[key] >= 0
    return output


async def compare_request_batches(engine, tokens, num_outputs):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    serial = []
    requests = []
    for index, (ids, mask) in enumerate(tokens):
        for prefix in ("serial", "packed"):
            request = OmniDiffusionRequest(
                prompt={"prompt_token_ids": ids, "prompt_mask": mask},
                sampling_params=OmniDiffusionSamplingParams(
                    seed=41 + index,
                    num_outputs_per_prompt=num_outputs,
                    height=64,
                    width=64,
                    num_inference_steps=4,
                    true_cfg_scale=1.0,
                    output_type="pil",
                    extra_args={"noise_level": 0.0},
                ),
                request_id=f"{prefix}-{index}",
            )
            if prefix == "serial":
                serial.append(await collect_request(engine, request))
            else:
                requests.append(request)
    execute = Mock(wraps=engine.execute_fn)
    engine.execute_fn = execute
    order = [2, 0, 1]
    packed = await asyncio.gather(*(collect_request(engine, requests[index]) for index in order))
    assert any(len(call.args[0].scheduled_request_ids) > 1 for call in execute.call_args_list)
    for output, index in zip(packed, order, strict=True):
        torch.testing.assert_close(output.trajectory_latents, serial[index].trajectory_latents, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_outputs", [1, 2])
def test_native_qwen_dmd_request_batch_matches_serial(num_outputs):
    model_path = os.environ.get("QWEN_IMAGE_MODEL_PATH", os.path.expanduser("~/models/tiny-random/Qwen-Image"))
    if not torch.cuda.is_available() or not os.path.isfile(os.path.join(model_path, "model_index.json")):
        pytest.skip("Requires CUDA and QWEN_IMAGE_MODEL_PATH pointing to a tiny Qwen-Image checkpoint.")
    from diffusers import QwenImagePipeline
    from transformers import AutoTokenizer
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    from verl_omni.pipelines.model_base import VllmOmniPipelineBase
    from verl_omni.pipelines.qwen_image_distillation.phase_runner import QwenImageConditionProvider

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"), local_files_only=True)
    template = QwenImagePipeline(tokenizer=tokenizer, text_encoder=None, vae=None, transformer=None, scheduler=None)
    provider = QwenImageConditionProvider(model_path, "local_frozen_encoder", 64, " ")
    tokens = []
    for text in ("cat", "a red apple", "a house on a hill in the evening"):
        ids, mask = provider.tokenize_rows(template, [text], torch.device("cpu"))
        tokens.append((ids[0].tolist(), mask[0].tolist()))
    config = OmniDiffusionConfig.from_kwargs(
        model=model_path,
        dtype=torch.float32,
        num_gpus=1,
        distributed_executor_backend="uni",
        step_execution=False,
        max_num_seqs=6,
        request_batch_max_wait_ms=50,
        diffusion_attention_backend="TORCH_SDPA",
        custom_pipeline_args={"pipeline_class": VllmOmniPipelineBase.get_pipeline_path("QwenImagePipeline", "dmd2")},
    )
    config.enrich_config()
    engine = None
    try:
        engine = DiffusionEngine(config)
        asyncio.run(compare_request_batches(engine, tokens, num_outputs))
    finally:
        if engine is not None:
            engine.close()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("lengths", [(64, 64), (37, 64)])
def test_flash3_attention_matches_native_forward_and_backward(lengths):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("This FlashAttention-3 validation targets Hopper GPUs.")
    from diffusers.models.attention_dispatch import attention_backend, dispatch_attention_fn

    generator = torch.Generator(device="cuda").manual_seed(17)
    values = [torch.randn(2, 64, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    mask = torch.arange(64, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    mask = mask[:, None, None, :]
    outputs, gradients = [], []
    for backend in ("native", "_flash_3_varlen_hub"):
        inputs = [value.detach().clone().requires_grad_(True) for value in values]
        with attention_backend(backend):
            output = dispatch_attention_fn(*inputs, attn_mask=mask)
            grads = torch.autograd.grad(output.float().square().sum(), inputs)
        outputs.append(output.detach())
        gradients.append(grads)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.003)
    for native, flash in zip(gradients[0], gradients[1], strict=True):
        torch.testing.assert_close(native, flash, rtol=0.03, atol=0.005)

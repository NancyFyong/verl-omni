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

"""Opt-in real-checkpoint GPU smoke test for LingBot Dense T2V FlowGRPO.

Run this only when a local Dense checkpoint is available (the rollout adapter
uses vLLM-Omni's in-tree LingBot transformer, so the pip ``lingbot-video``
package is not required here)::

    LINGBOT_VIDEO_MODEL_PATH=/path/to/lingbot-video-dense-1.3b \\
        pytest -q tests/pipelines/test_lingbot_video_flow_grpo_gpu.py

The deliberately tiny 64x64, five-frame rollout verifies the actual rollout
adapter's transformer forward pass, shifted SDE update/log-prob collection and
Wan VAE decode without requiring an expensive training job.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

MODEL_PATH = os.environ.get("LINGBOT_VIDEO_MODEL_PATH", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not find_spec("vllm_omni"), reason="requires vllm-omni with in-tree LingBot"),
    pytest.mark.skipif(
        not os.path.isdir(MODEL_PATH),
        reason="set LINGBOT_VIDEO_MODEL_PATH to a local Dense checkpoint",
    ),
]


def test_dense_rollout_produces_finite_video_and_flowgrpo_trajectory():
    """Run one real Dense rollout through transformer, SDE scheduler and VAE."""
    from verl_omni.pipelines.lingbot_video_flow_grpo.vllm_omni_rollout_adapter import (
        LingBotVideoPipelineWithLogProb,
    )

    pipeline = LingBotVideoPipelineWithLogProb(od_config=SimpleNamespace(model=MODEL_PATH, dtype=torch.bfloat16))
    assert pipeline.transformer.config.num_experts == 0

    # Supply embeddings directly to keep this smoke focused on FlowGRPO rollout
    # mechanics; the Qwen processor/text-encoder integration is exercised by the
    # normal agent loop and is intentionally not duplicated here.
    prompt_embeds = torch.randn(
        1,
        32,
        pipeline.transformer.config.text_dim,
        device=pipeline.device,
        dtype=torch.bfloat16,
    )
    prompt_embeds_mask = torch.ones(1, 32, device=pipeline.device, dtype=torch.long)
    sampling_params = SimpleNamespace(
        height=64,
        width=64,
        num_inference_steps=2,
        max_sequence_length=32,
        extra_args={
            "num_frames": 5,
            "shift": 3.0,
            "noise_level": 1.0,
            "sde_window_size": 1,
            "sde_window_range": [0, 2],
            "sde_type": "sde",
            "logprobs": True,
            "guidance_scale": 1.0,
        },
        guidance_scale_provided=False,
        guidance_scale=None,
        generator=torch.Generator(device=pipeline.device).manual_seed(123),
        seed=123,
    )
    request = SimpleNamespace(prompts=[{}], sampling_params=sampling_params)
    # The runner always wraps requests in a DiffusionRequestBatch.
    batch = SimpleNamespace(
        num_reqs=1,
        prompts=[{}],
        sampling_params=sampling_params,
        requests=[request],
    )

    output = pipeline(
        batch,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
    )

    payload = output.output["payload"]
    video = payload["video"]
    trajectory = payload["trajectory"]
    assert video.shape == (5, 3, 64, 64)
    assert video.device.type == "cpu"
    assert torch.isfinite(video).all()
    assert 0 <= video.min() <= video.max() <= 1
    assert output.output["metadata"]["trajectory"] == {"type": "denoising"}
    assert trajectory["all_latents"].shape[1] == 2
    assert trajectory["all_log_probs"].shape[1] == 1
    assert trajectory["all_timesteps"].shape == (1, 1)
    assert torch.isfinite(trajectory["all_log_probs"]).all()
    # The formatter lifts these named keys onto OmniRequestOutput.trajectory_*.
    assert torch.equal(trajectory["latents"], trajectory["all_latents"])
    assert torch.equal(output.trajectory_latents, trajectory["all_latents"])
    assert torch.equal(output.trajectory_log_probs, trajectory["all_log_probs"])
    assert torch.equal(output.trajectory_timesteps, trajectory["all_timesteps"])

    # The rollout transformer uses ordinary nn.Linear modules.  Exercise the
    # in-memory LoRA hook on a real GPU-resident transformer layer, including
    # the trainer-style ``base_model.model.transformer`` key prefix.
    module_name, linear = next(
        (name, module) for name, module in pipeline.transformer.named_modules() if isinstance(module, torch.nn.Linear)
    )
    lora_request = SimpleNamespace(
        lora_int_id=123,
        peft_config={"lora_alpha": 1},
        lora_tensors={
            f"base_model.model.transformer.{module_name}.lora_A.default.weight": torch.ones(1, linear.in_features),
            f"base_model.model.transformer.{module_name}.lora_B.default.weight": torch.ones(linear.out_features, 1),
        },
    )
    linear_input = torch.ones(1, linear.in_features, device=pipeline.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        baseline = linear(linear_input)
        assert pipeline.add_lora(lora_request)
        pipeline.set_active_lora(lora_request)
        adapted = linear(linear_input)
        pipeline.set_active_lora(None)
        restored = linear(linear_input)
    assert not torch.equal(adapted, baseline)
    assert torch.equal(restored, baseline)
    assert pipeline.remove_lora(123)

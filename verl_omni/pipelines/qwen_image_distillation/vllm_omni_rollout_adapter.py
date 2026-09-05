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
"""Qwen-Image inference adapter for DMD and DMD2 students."""

from __future__ import annotations

import numpy as np
import torch
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_distillation.phase_runner import build_qwen_dmd_sigmas
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

__all__ = ["QwenImageDMDPipeline"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dmd")
@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dmd2")
class QwenImageDMDPipeline(QwenImagePipelineWithLogProb):
    """Qwen-Image rollout with the same fixed-shift schedule used for DMD training."""

    rollout_timestep_shift = 3.0

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch, *args, **kwargs):
        """Apply one batch-consistent DMD rollout shift before normal Qwen generation."""
        requests = req.requests if isinstance(req, DiffusionRequestBatch) else [req]
        extra_args = [request.sampling_params.extra_args or {} for request in requests]
        shifts = {
            3.0 if values.get("rollout_timestep_shift") is None else float(values["rollout_timestep_shift"])
            for values in extra_args
        }
        if len(shifts) != 1:
            raise ValueError("Packed Qwen DMD requests must use the same rollout_timestep_shift.")
        shift = shifts.pop()
        if shift < 1:
            raise ValueError(f"rollout_timestep_shift must be at least 1, got {shift}.")
        default_noise_level = float(kwargs.get("noise_level", 0.0))
        noise_levels = {
            default_noise_level if values.get("noise_level") is None else float(values["noise_level"])
            for values in extra_args
        }
        if noise_levels != {0.0}:
            raise ValueError("Qwen DMD inference requires noise_level=0 for deterministic Euler sampling.")
        kwargs.setdefault("noise_level", 0.0)
        kwargs.setdefault("logprobs", False)
        kwargs.setdefault("true_cfg_scale", 1.0)
        previous_shift = self.rollout_timestep_shift
        self.rollout_timestep_shift = shift
        try:
            return super().forward(req, *args, **kwargs)
        finally:
            self.rollout_timestep_shift = previous_shift

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        """Build the fixed linear-shift schedule shared with the training phase runner."""
        del image_seq_len
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}.")
        if sigmas is None:
            sigmas = build_qwen_dmd_sigmas(num_inference_steps, self.rollout_timestep_shift)[:-1].numpy()
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)
            if sigmas.ndim != 1 or len(sigmas) != num_inference_steps:
                raise ValueError(f"sigmas must contain exactly {num_inference_steps} values, got {sigmas.shape}.")
            if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0) or np.any(sigmas > 1):
                raise ValueError("Qwen DMD sigmas must be finite and lie in (0, 1].")
            if np.any(sigmas[:-1] < sigmas[1:]):
                raise ValueError("Qwen DMD sigmas must be monotonically non-increasing.")

        device = self.device
        sigma_tensor = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
        self.scheduler.num_inference_steps = num_inference_steps
        self.scheduler.sigmas = torch.cat((sigma_tensor, sigma_tensor.new_zeros(1)))
        self.scheduler.timesteps = sigma_tensor * self.scheduler.config.get("num_train_timesteps", 1000)
        self.scheduler._step_index = None
        self.scheduler._begin_index = None
        return self.scheduler.timesteps, num_inference_steps

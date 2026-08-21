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
"""Dual-stream flow-matching SDE scheduler for audio-video pipelines."""

import numpy as np
import torch

from .flow_match_sde import FlowMatchSDEDiscreteScheduler

__all__ = ["FlowMatchDualSDEDiscreteScheduler", "flow_match_shift_sigmas"]


def flow_match_shift_sigmas(num_steps: int, shift_scale: float) -> list[float]:
    """Build a shift-scaled flow-matching sigma schedule.

    Mirrors vllm-omni's ``minimax_h3_time_shift_sigmas`` so the actor replays the schedule
    the rollout sampled with.
    """
    if shift_scale <= 0:
        raise ValueError(f"shift_scale must be > 0, got {shift_scale}.")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}.")

    base = torch.linspace(1.0, 0.0, int(num_steps), device="cpu", dtype=torch.float32)
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]


class FlowMatchDualSDEDiscreteScheduler:
    """Hold independent video and audio FlowGRPO schedulers with per-stream shifts."""

    def __init__(self, video_flow_shift: float = 12.0, audio_flow_shift: float = 3.0):
        """Create the modality schedulers."""
        self.video_flow_shift = video_flow_shift
        self.audio_flow_shift = audio_flow_shift
        self.video_scheduler = FlowMatchSDEDiscreteScheduler(shift=1.0)
        self.audio_scheduler = FlowMatchSDEDiscreteScheduler(shift=1.0)

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu", **kwargs) -> None:
        """Configure both modality schedules."""
        del kwargs
        self._set_stream(self.video_scheduler, num_inference_steps, self.video_flow_shift, device)
        self._set_stream(self.audio_scheduler, num_inference_steps, self.audio_flow_shift, device)

    @staticmethod
    def _set_stream(scheduler: FlowMatchSDEDiscreteScheduler, num_steps: int, shift_scale: float, device: str) -> None:
        sigmas = flow_match_shift_sigmas(num_steps, shift_scale)
        interior = np.asarray(sigmas[:-1], dtype=np.float32)
        scheduler.set_timesteps(num_inference_steps=len(interior), device=device, sigmas=interior)

    @property
    def timesteps(self) -> torch.Tensor:
        """Return the video schedule used by the shared engine contract."""
        return self.video_scheduler.timesteps

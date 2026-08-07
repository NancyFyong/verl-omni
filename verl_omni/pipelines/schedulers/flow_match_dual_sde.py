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
"""Dual-stream SDE scheduler container for MiniMax H3 policy-gradient rollout.

MiniMax H3 steps video rows (width 96) and audio rows (width 32) on two different
sigma schedules -- video ``flow_shift=12.0``, audio ``audio_flow_shift=3.0`` -- so a
single scalar timestep cannot describe the joint reverse-SDE state and log-probs
must be computed per modality. This module provides two pieces that keep the
training side aligned with the vllm-omni rollout without forking any numerics:

* :func:`minimax_h3_time_shift_sigmas`, a CPU-safe replica of vllm-omni's
  ``_time_shift_sigmas`` so both sides build the same sigma arrays.
* :class:`FlowMatchDualSDEDiscreteScheduler`, a thin container holding one stock
  :class:`~verl_omni.pipelines.schedulers.flow_match_sde.FlowMatchSDEDiscreteScheduler`
  per modality; the adapter samples and scores each stream through the reused,
  already-tested SDE / log-prob math.
"""

import numpy as np
import torch

from .flow_match_sde import FlowMatchSDEDiscreteScheduler

__all__ = ["FlowMatchDualSDEDiscreteScheduler", "minimax_h3_time_shift_sigmas"]


def minimax_h3_time_shift_sigmas(num_steps: int, shift_scale: float) -> list[float]:
    """Replicate vllm-omni's MiniMax H3 ``_time_shift_sigmas`` on CPU.

    Bit-identical to ``_time_shift_sigmas`` in
    ``vllm_omni.diffusion.models.minimax_h3.time_request``: a ``linspace(1, 0,
    num_steps)`` ramp is pushed through the SD3 time-shift ``s·shift / (1 +
    (shift-1)·s)``, consecutive duplicates are collapsed, and a terminal ``0.0``
    is appended when the schedule does not already end at zero. Keeping this
    in-tree lets CPU tests assert train/rollout sigma alignment without importing
    vllm-omni.

    Args:
        num_steps: Number of inference steps ``N``; yields up to ``N`` sigmas.
        shift_scale: Per-stream shift (video ``12.0``, audio ``3.0``).

    Returns:
        list[float]: Sigmas descending from ``1.0`` to ``0.0`` (terminal ``0.0``
            included), one entry per requested step before any dedup.
    """
    if shift_scale <= 0:
        raise ValueError(f"MiniMax H3 shift_scale must be > 0, got {shift_scale}.")
    if num_steps <= 0:
        raise ValueError(f"MiniMax H3 num_steps must be > 0, got {num_steps}.")

    base = torch.linspace(1.0, 0.0, int(num_steps), device="cpu", dtype=torch.float32)
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]


class FlowMatchDualSDEDiscreteScheduler:
    """Pair of per-modality SDE schedulers for MiniMax H3 reverse sampling.

    Holds a ``video_scheduler`` and an ``audio_scheduler``, each a stock
    :class:`FlowMatchSDEDiscreteScheduler` configured with ``shift=1.0`` and fed
    the already-shifted H3 sigmas via ``set_timesteps(sigmas=...)`` -- ``shift=1.0``
    makes the scheduler's built-in shift an identity, so the pre-shifted sigmas
    pass through untouched (no double-shift). The training adapter samples each
    stream's previous step and log-prob separately and combines them; nothing
    here reimplements the SDE. ``set_timesteps`` / ``timesteps`` delegate to the
    video stream so the container drops into the engine's single-scheduler
    contract (the engine takes ``num_timesteps`` from the rollout data, not from
    this object).
    """

    def __init__(self, video_flow_shift: float = 12.0, audio_flow_shift: float = 3.0):
        """Create the two per-modality schedulers (timesteps set later).

        Args:
            video_flow_shift: Video-stream shift (vllm-omni ``flow_shift``).
            audio_flow_shift: Audio-stream shift (vllm-omni ``audio_flow_shift``).
        """
        self.video_flow_shift = video_flow_shift
        self.audio_flow_shift = audio_flow_shift
        self.video_scheduler = FlowMatchSDEDiscreteScheduler(shift=1.0)
        self.audio_scheduler = FlowMatchSDEDiscreteScheduler(shift=1.0)

    def set_timesteps(self, num_inference_steps: int, device: str = "cpu", **kwargs) -> None:
        """Configure both streams with their pre-shifted H3 sigmas.

        Args:
            num_inference_steps: Number of inference steps ``N`` (yields ``N-1``
                reverse-SDE transitions per stream).
            device: Target device for the timestep / sigma buffers.
        """
        del kwargs
        self._set_stream(self.video_scheduler, num_inference_steps, self.video_flow_shift, device)
        self._set_stream(self.audio_scheduler, num_inference_steps, self.audio_flow_shift, device)

    @staticmethod
    def _set_stream(scheduler: FlowMatchSDEDiscreteScheduler, num_steps: int, shift_scale: float, device: str) -> None:
        sigmas = minimax_h3_time_shift_sigmas(num_steps, shift_scale)
        # diffusers re-appends the terminal 0.0, so hand it the interior sigmas;
        # ``scheduler.sigmas`` then round-trips to the full H3 schedule.
        interior = np.asarray(sigmas[:-1], dtype=np.float32)
        scheduler.set_timesteps(num_inference_steps=len(interior), device=device, sigmas=interior)

    @property
    def timesteps(self) -> torch.Tensor:
        """Video-stream timesteps (the engine reads step count from the data)."""
        return self.video_scheduler.timesteps

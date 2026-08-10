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
"""MiniMax H3 rollout adapter for DiffusionNFT (GPU-only).

DiffusionNFT trains from the final clean latent with a forward-process
objective, so rollout does not collect a reverse-SDE trajectory. This adapter
runs the standard MiniMax H3 generation (``t2va`` / ``fl2va``), captures the
clean video+audio latents, patchifies them back into DiT rows, packs both
streams into one flat ``latents_clean`` (see :mod:`.common`), and emits the
per-sample row/shape metadata the training adapter needs to invert the pack.

This module imports vllm_omni at module scope and is therefore GPU-only; the
package ``__init__`` guards the import so the training side stays CPU-importable.

Version note: this targets vllm-omni >= 0.26 (``.github/vllm_omni_pin.txt``), which
dropped ``DiffusionOutput.custom_output``. The adapter still emits ``custom_output`` the
same way as every sibling; :mod:`verl_omni.pipelines._vllm_omni_compat` re-threads that
channel at import so the engine and read-sites keep consuming ``result.custom_output``.
"""

import dataclasses
from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import MiniMaxH3RolloutWeightSyncMixin, pack_video_audio_rows

__all__ = ["MiniMaxH3DiffusionNFTPipeline"]

# Video latent patch size: (temporal, height, width). 24 channels * 1 * 2 * 2 = 96-wide rows.
_VIDEO_PATCH_SIZE = (1, 2, 2)


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFTPipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Rollout pipeline for MiniMax H3 used by DiffusionNFT."""

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        self._nft_capture: dict[str, Any] | None = None

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the parent denoiser and capture the clean latents + shape metadata."""
        video_latent, audio_latent = super().diffuse(**kwargs)
        self._nft_capture = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "text_embeddings": kwargs.get("text_embeddings"),
            "latent_t": int(kwargs.get("latent_t", 0)),
            "latent_h": int(kwargs.get("latent_h", 0)),
            "latent_w": int(kwargs.get("latent_w", 0)),
            "audio_t": int(kwargs.get("audio_t", 0)),
            "num_steps": int(kwargs.get("num_steps", 50)),
            "video_shift": float(kwargs.get("video_shift", self.default_video_shift)),
        }
        return video_latent, audio_latent

    def forward(self, request: Any):
        """Generate video+audio and attach DiffusionNFT training tensors."""
        self._ensure_prompt_text(request)
        output = super().forward(request)
        capture = self._nft_capture
        self._nft_capture = None
        if capture is None:
            return output

        # Patchify the clean latents back into the transformer's row layout so the
        # packed tensor lives in the same space as the model's velocity prediction.
        video_rows = minimax_h3_patchify_video_latent(capture["video_latent"], patch_size=_VIDEO_PATCH_SIZE)
        audio_rows = minimax_h3_pack_audio_latent(capture["audio_latent"])
        latents_clean = pack_video_audio_rows(video_rows, audio_rows).float()
        num_video_rows = int(video_rows.shape[0])
        num_audio_rows = int(audio_rows.shape[0])

        latent_meta = torch.tensor(
            [
                [
                    num_video_rows,
                    num_audio_rows,
                    capture["latent_t"],
                    capture["latent_h"],
                    capture["latent_w"],
                    capture["audio_t"],
                ]
            ],
            dtype=torch.long,
        )

        text_embeddings = capture["text_embeddings"]
        prompt_embeds = text_embeddings.unsqueeze(0)
        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)

        train_timesteps = self._build_train_timesteps(capture).unsqueeze(0)

        return dataclasses.replace(
            output,
            custom_output={
                "latents_clean": latents_clean,
                "train_timesteps": train_timesteps,
                "latent_meta": latent_meta,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            },
            to_cpu=True,
        )

    @staticmethod
    def _build_train_timesteps(capture: dict[str, Any]) -> torch.Tensor:
        """Return the video-stream sigma schedule (x1000) as a candidate timestep pool.

        DiffusionNFT samples one shared timestep per training step from this pool
        (Option C). The audio stream uses a different shift at generation time but
        the packed objective trains at the video timestep. See the RFC.
        """
        sigmas = minimax_h3_time_shift_sigmas(
            num_steps=capture["num_steps"],
            shift_scale=capture["video_shift"],
        )
        return torch.tensor(sigmas[:-1], dtype=torch.float32) * 1000.0

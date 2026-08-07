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
"""MiniMax H3 rollout adapter for FlowGRPO (GPU-only).

FlowGRPO trains from the reverse-SDE trajectory and its per-step
log-probabilities. MiniMax H3's own scheduler is eta=0 deterministic (no noise,
no density), so this adapter re-runs the denoise loop as a stochastic SDE: at
each step it predicts ``(v_video, v_audio)``, steps each modality through its own
:class:`~verl_omni.pipelines.schedulers.FlowMatchDualSDEDiscreteScheduler` leg on
that modality's sigma schedule, captures the per-modality Gaussian log-prob, and
combines them as ``w_video * lp_video + w_audio * lp_audio`` (each stream
mean-reduced inside the scheduler, so the ~269:1 video:audio element ratio does
not drown audio). The reverse trajectory is recorded in packed DiT-row space (see
:mod:`..minimax_h3_diffusion_nft.common`) so it lands in the same layout the
training adapter unpacks; both timestep schedules are threaded through
``custom_output`` (``all_timesteps`` = video, ``audio_all_timesteps`` = audio).

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

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler

from ..minimax_h3_diffusion_nft.common import pack_video_audio_rows

__all__ = ["MiniMaxH3PipelineWithLogProb"]

# Video latent patch size: (temporal, height, width). 24 channels * 1 * 2 * 2 = 96-wide rows.
_VIDEO_PATCH_SIZE = (1, 2, 2)


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3PipelineWithLogProb(MiniMaxH3Pipeline):
    """Rollout pipeline for MiniMax H3 that captures the dual reverse-SDE trajectory."""

    supports_request_batch = False

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        # Two stock SDE legs on the video/audio shifts; step counts are set per request.
        self.scheduler = FlowMatchDualSDEDiscreteScheduler(
            video_flow_shift=getattr(self, "default_video_shift", 12.0),
            audio_flow_shift=getattr(self, "default_audio_shift", 3.0),
        )
        self._flow_grpo_capture: dict[str, Any] | None = None

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a dual reverse-SDE denoise loop and capture the packed trajectory.

        Replaces the parent's deterministic loop: samples each modality through
        its own SDE leg, collects the packed row-space trajectory plus per-step
        combined log-probs and per-modality timesteps, then returns the final
        clean ``(video_latent, audio_latent)`` so the parent's ``forward`` decode
        path is unchanged.

        Args:
            **kwargs: The parent ``diffuse`` inputs (``text_embeddings``,
                ``latent_t`` / ``latent_h`` / ``latent_w`` / ``audio_t``,
                ``num_steps``) plus FlowGRPO knobs (``noise_level``, ``sde_type``,
                ``av_logprob_video_weight``, ``av_logprob_audio_weight``).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The final ``(video_latent,
                audio_latent)`` in latent space for VAE decode.
        """
        num_steps = int(kwargs.get("num_steps", 50))
        noise_level = float(kwargs.get("noise_level", 1.0))
        sde_type = str(kwargs.get("sde_type", "sde"))
        w_video = float(kwargs.get("av_logprob_video_weight", 1.0))
        w_audio = float(kwargs.get("av_logprob_audio_weight", 1.0))

        self.scheduler.set_timesteps(num_steps, device=self.device)
        video_timesteps = self.scheduler.video_scheduler.timesteps
        audio_timesteps = self.scheduler.audio_scheduler.timesteps

        video_rows, audio_rows = self._init_row_noise(kwargs)

        packed_traj = [pack_video_audio_rows(video_rows, audio_rows).detach().float().clone()]
        log_probs: list[torch.Tensor] = []
        for step, (t_video, t_audio) in enumerate(zip(video_timesteps, audio_timesteps, strict=True)):
            v_video, v_audio = self._predict_velocity(video_rows, audio_rows, t_video, t_audio, kwargs)

            video_rows, lp_video, _, _ = self.scheduler.video_scheduler.step(
                v_video.float(),
                t_video,
                video_rows.float(),
                noise_level=noise_level,
                sde_type=sde_type,
                return_logprobs=True,
                return_dict=False,
            )
            audio_rows, lp_audio, _, _ = self.scheduler.audio_scheduler.step(
                v_audio.float(),
                t_audio,
                audio_rows.float(),
                noise_level=noise_level,
                sde_type=sde_type,
                return_logprobs=True,
                return_dict=False,
            )

            packed_traj.append(pack_video_audio_rows(video_rows, audio_rows).detach().float().clone())
            log_probs.append(w_video * lp_video + w_audio * lp_audio)

        self._flow_grpo_capture = {
            "all_latents": torch.stack(packed_traj, dim=1),
            "all_log_probs": torch.stack(log_probs, dim=1),
            "all_timesteps": video_timesteps.unsqueeze(0).expand(video_rows.shape[0], -1).clone(),
            "audio_all_timesteps": audio_timesteps.unsqueeze(0).expand(video_rows.shape[0], -1).clone(),
            "text_embeddings": kwargs.get("text_embeddings"),
            "num_video_rows": int(video_rows.shape[1]),
            "num_audio_rows": int(audio_rows.shape[1]),
            "latent_t": int(kwargs.get("latent_t", 0)),
            "latent_h": int(kwargs.get("latent_h", 0)),
            "latent_w": int(kwargs.get("latent_w", 0)),
            "audio_t": int(kwargs.get("audio_t", 0)),
        }
        return self._rows_to_latents(video_rows, audio_rows, kwargs)

    def forward(self, request: Any):
        """Generate video+audio and attach the FlowGRPO reverse-SDE trajectory."""
        output = super().forward(request)
        capture = self._flow_grpo_capture
        self._flow_grpo_capture = None
        if capture is None:
            return output

        latent_meta = torch.tensor(
            [
                [
                    capture["num_video_rows"],
                    capture["num_audio_rows"],
                    capture["latent_t"],
                    capture["latent_h"],
                    capture["latent_w"],
                    capture["audio_t"],
                ]
            ],
            dtype=torch.long,
        )

        prompt_embeds = capture["text_embeddings"].unsqueeze(0)
        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)
        # CFG-distilled (true_cfg_scale=1.0): the negative branch is inert, but the
        # engine's data contract requires the keys, so emit empty tensors.
        empty = prompt_embeds.new_zeros((prompt_embeds.shape[0], 0, prompt_embeds.shape[2]))
        empty_mask = prompt_embeds_mask.new_zeros((prompt_embeds_mask.shape[0], 0))

        return dataclasses.replace(
            output,
            custom_output={
                "all_latents": capture["all_latents"],
                "all_log_probs": capture["all_log_probs"],
                "all_timesteps": capture["all_timesteps"],
                "audio_all_timesteps": capture["audio_all_timesteps"],
                "latent_meta": latent_meta,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": empty,
                "negative_prompt_embeds_mask": empty_mask,
            },
            to_cpu=True,
        )

    def _init_row_noise(self, kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the initial ``(video_rows, audio_rows)`` Gaussian noise in row space.

        TODO(gpu-bringup): the parent samples initial latents in *latent* space and
        patchifies inside its loop; construct the row-space noise from
        ``latent_t/h/w`` + ``audio_t`` via the same patch layout the transformer
        consumes (``minimax_h3_patchify_video_latent`` / ``minimax_h3_pack_audio_latent``),
        confirmed against the diffusers minimax-h3 branch.
        """
        video_latent, audio_latent = super().prepare_latents(**kwargs)  # type: ignore[misc]
        video_rows = minimax_h3_patchify_video_latent(video_latent, patch_size=_VIDEO_PATCH_SIZE)
        audio_rows = minimax_h3_pack_audio_latent(audio_latent)
        if video_rows.ndim == 2:
            video_rows = video_rows.unsqueeze(0)
        if audio_rows.ndim == 2:
            audio_rows = audio_rows.unsqueeze(0)
        return video_rows, audio_rows

    def _predict_velocity(
        self,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: torch.Tensor,
        t_audio: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(v_video, v_audio)`` for one reverse-SDE step.

        TODO(gpu-bringup): fill in the exact ``MiniMaxH3Transformer3DModel.forward``
        structural kwargs (img_position_ids, unique_timesteps, inverse_indices,
        update_mask, update_audio_mask, token_tags, {img,audio,text}_pos_info,
        packed_seq_params) and per-modality timestep kwarg names -- shared with the
        training adapter's open TODO, so the two forward call sites stay identical.
        """
        result = self.transformer(
            x=video_rows.to(self.transformer.dtype),
            audio_x=audio_rows.to(self.transformer.dtype),
            prompt_embeds=kwargs.get("text_embeddings"),
            timestep=t_video.expand(video_rows.shape[0]) / 1000.0,
            audio_timestep=t_audio.expand(audio_rows.shape[0]) / 1000.0,
            return_dict=False,
        )
        return result[0], result[1]

    def _rows_to_latents(
        self,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpatchify the final rows back to latent space for the parent VAE decode.

        TODO(gpu-bringup): wire the exact inverse of ``_init_row_noise``
        (``minimax_h3_unpatchify_video_latent`` / ``minimax_h3_unpack_audio_latent``
        or the parent's equivalent) using ``latent_t/h/w`` + ``audio_t`` from
        *kwargs*, confirmed against the diffusers minimax-h3 branch.
        """
        raise NotImplementedError(
            "MiniMaxH3PipelineWithLogProb._rows_to_latents is a GPU-bringup stub: "
            "wire the row->latent inverse before the end-to-end rollout smoke test."
        )

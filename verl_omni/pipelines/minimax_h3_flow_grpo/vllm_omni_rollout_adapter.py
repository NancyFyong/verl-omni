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
"""GPU rollout adapter for MiniMax H3 FlowGRPO."""

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler

from ..minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin,
    h3_dit_timestep,
    h3_velocity_to_flow_match,
    pack_video_audio_rows,
)

__all__ = ["MiniMaxH3PipelineWithLogProb"]

_VIDEO_PATCH_SIZE = (1, 2, 2)
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_CHANNELS = 2
_SDE_SEED_OFFSET = 1_000_003


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3PipelineWithLogProb(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Capture MiniMax H3 video/audio reverse-SDE trajectories."""

    supports_request_batch = False

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        self._install_lora_layout()
        self.scheduler = FlowMatchDualSDEDiscreteScheduler(
            video_flow_shift=getattr(self, "default_video_shift", 12.0),
            audio_flow_shift=getattr(self, "default_audio_shift", 3.0),
        )
        self._flow_grpo_capture: dict[str, Any] | None = None
        self._flow_grpo_noise_level = 1.0
        self._flow_grpo_sde_type = "sde"
        self._flow_grpo_video_weight = 1.0
        self._flow_grpo_audio_weight = 1.0
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_window_seed = 42

    def forward(self, request: Any):
        """Generate media and attach the FlowGRPO trajectory."""
        self._configure_flow_grpo(request)
        self._ensure_prompt_text(request)
        try:
            output = super().forward(request)
        finally:
            self._h3_prompt_ids = None
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
        empty = prompt_embeds.new_zeros((prompt_embeds.shape[0], 0, prompt_embeds.shape[2]))
        empty_mask = prompt_embeds_mask.new_zeros((prompt_embeds_mask.shape[0], 0))
        # The audio timestep pool and latent layout have no dedicated slot, so they ride the rl group.
        return with_rollout_data(
            output,
            trajectory_latents=capture["all_latents"],
            trajectory_log_probs=capture["all_log_probs"],
            trajectory_timesteps=capture["all_timesteps"],
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": empty,
                "negative_prompt_embeds_mask": empty_mask,
            },
            rl={
                "audio_all_timesteps": capture["audio_all_timesteps"],
                "latent_meta": latent_meta,
            },
            to_cpu=True,
        )

    def _configure_flow_grpo(self, request: Any) -> None:
        """Read FlowGRPO settings from the request."""
        extra_args = getattr(request.sampling_params, "extra_args", None) or {}
        self._flow_grpo_noise_level = float(extra_args.get("noise_level", 1.0))
        self._flow_grpo_sde_type = str(extra_args.get("sde_type", "sde"))
        self._flow_grpo_video_weight = float(extra_args.get("av_logprob_video_weight", 1.0))
        self._flow_grpo_audio_weight = float(extra_args.get("av_logprob_audio_weight", 1.0))
        window_size = extra_args.get("sde_window_size")
        self._flow_grpo_window_size = None if window_size is None else int(window_size)
        window_range = extra_args.get("sde_window_range")
        self._flow_grpo_window_range = None if window_range is None else [int(bound) for bound in window_range]
        if not bool(extra_args.get("sde_contiguous", True)):
            raise NotImplementedError("MiniMax H3 FlowGRPO supports contiguous SDE windows only.")
        self._flow_grpo_window_seed = int(extra_args.get("sde_window_seed", 42)) + max(
            int(extra_args.get("global_steps", 1)) - 1, 0
        )
        if "video_flow_shift" in extra_args:
            extra_args.setdefault("flow_shift", extra_args["video_flow_shift"])

    def _select_sde_window(self, num_transitions: int) -> tuple[int, int]:
        """Select the contiguous reverse-SDE training window."""
        if self._flow_grpo_window_size is None:
            return 0, num_transitions

        window_size = self._flow_grpo_window_size
        window_range = self._flow_grpo_window_range or [0, num_transitions]
        low = max(int(window_range[0]), 0)
        high = min(int(window_range[1]), num_transitions)
        if window_size <= 0 or high - low < window_size:
            raise ValueError(
                f"Invalid MiniMax H3 SDE window: size={window_size}, range={window_range}, "
                f"transitions={num_transitions}."
            )
        generator = torch.Generator().manual_seed(self._flow_grpo_window_seed)
        start = int(torch.randint(low, high - window_size + 1, (1,), generator=generator).item())
        return start, start + window_size

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the dual reverse-SDE denoiser and capture its trajectory."""
        task = str(kwargs.get("task", "t2va"))
        if task != "t2va":
            raise NotImplementedError(
                f"MiniMax H3 FlowGRPO supports task='t2va' only, got {task!r}. "
                "Conditional rows are not represented by this actor layout."
            )
        if kwargs.get("base_schedule") is not None:
            # This loop rebuilds the schedule from the stream shifts, so a checkpoint-pinned
            # distilled schedule would be replayed by the actor as a schedule it never sampled.
            raise NotImplementedError(
                "MiniMax H3 FlowGRPO does not support checkpoints that pin a distilled sigma schedule."
            )

        text_embeddings = kwargs["text_embeddings"]
        seed = int(kwargs.get("seed", 42))
        latent_t = int(kwargs["latent_t"])
        latent_h = int(kwargs["latent_h"])
        latent_w = int(kwargs["latent_w"])
        audio_t = int(kwargs["audio_t"])
        num_steps = int(kwargs.get("num_steps", 50))
        branch = self._build_branch(
            text_embeddings=text_embeddings,
            text_tags=kwargs["text_tags"],
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )

        self.scheduler = FlowMatchDualSDEDiscreteScheduler(
            video_flow_shift=float(kwargs.get("video_shift", self.default_video_shift)),
            audio_flow_shift=float(kwargs.get("audio_shift", self.default_audio_shift)),
        )
        self.scheduler.set_timesteps(num_steps, device=self.device)
        video_scheduler = self.scheduler.video_scheduler
        audio_scheduler = self.scheduler.audio_scheduler
        video_scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)
        video_timesteps = video_scheduler.timesteps
        audio_timesteps = audio_scheduler.timesteps
        video_dit_timesteps = h3_dit_timestep(video_timesteps.float()).tolist()
        audio_dit_timesteps = h3_dit_timestep(audio_timesteps.float()).tolist()

        video_rows, audio_rows = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        video_rows = video_rows.to(device=self.device, dtype=torch.float32)
        audio_rows = audio_rows.to(device=self.device, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed + _SDE_SEED_OFFSET)

        noise_level = self._flow_grpo_noise_level
        sde_type = self._flow_grpo_sde_type
        return_logprobs = noise_level > 0.0
        window_start, window_end = self._select_sde_window(len(video_timesteps))
        packed_traj: list[torch.Tensor] = []
        window_tail: torch.Tensor | None = None
        log_probs: list[torch.Tensor] = []
        selected_steps: list[int] = []
        for step, (video_timestep, audio_timestep) in enumerate(zip(video_timesteps, audio_timesteps, strict=True)):
            in_window = window_start <= step < window_end
            step_noise_level = noise_level if in_window else 0.0
            if in_window:
                packed_traj.append(pack_video_audio_rows(video_rows, audio_rows))
                selected_steps.append(step)

            video_velocity, audio_velocity = self._predict_velocity(
                branch,
                video_rows=video_rows,
                audio_rows=audio_rows,
                t_video=video_dit_timesteps[step],
                t_audio=audio_dit_timesteps[step],
            )
            next_video, video_log_prob, _, _ = video_scheduler.step(
                h3_velocity_to_flow_match(video_velocity.float()).unsqueeze(0),
                video_timestep,
                video_rows.unsqueeze(0),
                generator=generator,
                noise_level=step_noise_level,
                sde_type=sde_type,
                return_logprobs=return_logprobs and in_window,
                return_dict=False,
            )
            next_audio, audio_log_prob, _, _ = audio_scheduler.step(
                h3_velocity_to_flow_match(audio_velocity.float()).unsqueeze(0),
                audio_timestep,
                audio_rows.unsqueeze(0),
                generator=generator,
                noise_level=step_noise_level,
                sde_type=sde_type,
                return_logprobs=return_logprobs and in_window,
                return_dict=False,
            )
            video_rows = next_video.squeeze(0)
            audio_rows = next_audio.squeeze(0)
            if in_window:
                window_tail = pack_video_audio_rows(video_rows, audio_rows)
                if video_log_prob is None:
                    log_probs.append(video_rows.new_zeros(1))
                else:
                    log_probs.append(
                        self._flow_grpo_video_weight * video_log_prob + self._flow_grpo_audio_weight * audio_log_prob
                    )

        if window_tail is None:
            raise RuntimeError("MiniMax H3 FlowGRPO rollout retained no reverse-SDE transitions.")
        packed_traj.append(window_tail)
        step_index = torch.tensor(selected_steps, dtype=torch.long)
        self._flow_grpo_capture = {
            "all_latents": torch.stack(packed_traj, dim=1),
            "all_log_probs": torch.stack(log_probs, dim=1),
            "all_timesteps": video_timesteps.float().cpu()[step_index].unsqueeze(0),
            "audio_all_timesteps": audio_timesteps.float().cpu()[step_index].unsqueeze(0),
            "text_embeddings": text_embeddings,
            "num_video_rows": int(video_rows.shape[0]),
            "num_audio_rows": int(audio_rows.shape[0]),
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
        }
        return self._rows_to_latents(
            video_rows,
            audio_rows,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )

    def _build_branch(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> MiniMaxH3DenoiseBranch:
        """Build the upstream packed T2VA sequence."""
        packed = minimax_h3_packed_sequence(
            text_len=int(text_embeddings.shape[0]),
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            include_keyframe_cond=False,
        )
        token_tags = packed["token_tags"].clone()
        token_tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=token_tags,
            device=self.device,
        )
        if not bool(branch.update_mask.all()) or not bool(branch.audio_update_mask.all()):
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support condition rows in the packed layout.")
        return branch

    def _predict_velocity(
        self,
        branch: MiniMaxH3DenoiseBranch,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one video/audio velocity pair."""
        forward_kwargs = branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timestep=max(t_video, MINIMAX_H3_IMGVID_COND_TIMESTEP),
            audio_ref_cond_timestep=max(t_audio, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
        )
        with torch.no_grad():
            video_velocity, audio_velocity = self.transformer(**forward_kwargs)
        return video_velocity, audio_velocity

    @staticmethod
    def _rows_to_latents(
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        *,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert final DiT rows to VAE latents."""
        video_latent = minimax_h3_unpatchify_video_tokens(
            video_rows,
            latent_shape=(
                latent_t,
                latent_h // _VIDEO_PATCH_SIZE[1],
                latent_w // _VIDEO_PATCH_SIZE[2],
                _VIDEO_LATENT_CHANNELS,
            ),
            patch_size=_VIDEO_PATCH_SIZE,
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_rows,
            audio_t=audio_t * _AUDIO_CHANNELS,
            audio_channel=_AUDIO_CHANNELS,
        )
        return video_latent, audio_latent

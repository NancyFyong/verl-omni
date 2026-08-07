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
"""MiniMax H3 training adapter for FlowGRPO.

MiniMax H3 is a CFG-distilled joint video+audio rectified-flow transformer whose
video rows (width 96) and audio rows (width 32) step on two different sigma
schedules, so the reverse-SDE log-probability must be computed per modality. The
rollout packs both row streams into the shared ``all_latents`` carrier (see
:mod:`..minimax_h3_diffusion_nft.common`) and threads a second per-modality
``audio_all_timesteps`` schedule through the micro-batch. This adapter unpacks
the packed trajectory, runs the transformer, samples each stream through its own
:class:`~verl_omni.pipelines.schedulers.FlowMatchDualSDEDiscreteScheduler` leg,
and combines the two log-probs with per-modality weights -- each stream is
mean-reduced *before* the weighted sum so the ~269:1 video:audio element ratio
does not drown the audio signal. The shared engine and ``FlowGRPOLoss`` stay
single-tensor and unchanged.

The transformer reads one packed sequence whose row layout is shared across the
batch (its ``token_tags`` / ``position_ids`` / row indices carry no batch dim and
it takes no attention mask), and prompts vary in length, so the forward runs one
micro-batch sample at a time. Unlike DiffusionNFT's shared-timestep Option C,
FlowGRPO's two streams sit at *different* noise levels each step, so every sample
routes its video rows to the video timestep and its audio rows to the audio
timestep via :func:`~..minimax_h3_diffusion_nft.common.build_row_timesteps`.
"""

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    build_layout_from_meta,
    build_row_timesteps,
    pack_video_audio_rows,
    split_dual_velocity,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["MiniMaxH3FlowGRPO"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3FlowGRPO(DiffusionModelBase):
    """Reverse-SDE MiniMax H3 adapter used by FlowGRPO (dual video+audio streams)."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build the dual-stream SDE scheduler (one leg per modality).

        Args:
            model_config: Configuration for the diffusion model.

        Returns:
            FlowMatchDualSDEDiscreteScheduler: Container with per-modality
                ``video_scheduler`` / ``audio_scheduler`` timesteps set.
        """
        pipeline = model_config.pipeline
        scheduler = FlowMatchDualSDEDiscreteScheduler(
            video_flow_shift=pipeline.video_flow_shift,
            audio_flow_shift=pipeline.audio_flow_shift,
        )
        cls.set_timesteps(scheduler, model_config, device="cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Set per-modality timesteps on both scheduler legs.

        Args:
            scheduler: The dual-stream scheduler to configure.
            model_config: Configuration providing the inference-step count.
            device: Target device.
        """
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, None]:
        """Unpack the packed step latent into video + audio rows and build inputs.

        Args:
            module: The MiniMax H3 transformer module.
            model_config: Configuration for the diffusion model.
            latents: Packed reverse-SDE trajectory ``(B, W + 1, Nv * 96 + Na * 32)``.
            timesteps: Video-stream timesteps ``(B, W)`` in ``[0, 1000]``.
            prompt_embeds: Text embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for *prompt_embeds*, shape ``(B, L)``.
            negative_prompt_embeds: Unused; MiniMax H3 is CFG-distilled.
            negative_prompt_embeds_mask: Unused; MiniMax H3 is CFG-distilled.
            micro_batch: Micro-batch carrying ``audio_all_timesteps`` ``(B, W)`` and
                ``latent_meta`` ``(B, 6)`` = ``[Nv, Na, latent_t, latent_h, latent_w, audio_t]``.
            step: Current denoising-step index used to slice the trajectory.

        Returns:
            tuple[dict, None]: ``(model_inputs, None)`` -- no negative branch.
        """
        del negative_prompt_embeds, negative_prompt_embeds_mask
        # All samples in a micro-batch share one resolution/duration, so read row 0.
        meta = micro_batch["latent_meta"][0].tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
        video_rows, audio_rows = unpack_video_audio_rows(latents[:, step], num_video_rows, num_audio_rows)
        audio_timesteps = micro_batch["audio_all_timesteps"][:, step]

        # The static packed-sequence layout depends on each sample's text length, so it is
        # built inside forward; carry the two row streams, text cond, both per-modality
        # timesteps, and the grid dims the layout is derived from.
        model_inputs = {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": timesteps[:, step].float() / 1000.0,
            "audio_timestep": audio_timesteps.float() / 1000.0,
            "latent_meta": meta,
        }
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the transformer per micro-batch sample and re-pack the dual velocity into packed-flat form.

        Each sample is run on its own -- with its true text length and its per-modality timesteps -- because
        the transformer's row layout carries no batch dim and it takes no attention mask. Video rows are
        routed to the video timestep and audio rows to the audio timestep via :func:`build_row_timesteps`.

        Args:
            module: The MiniMax H3 transformer module.
            model_config: Configuration for the diffusion model (unused).
            model_inputs: Inputs from :meth:`prepare_model_inputs`.
            negative_model_inputs: Unused; MiniMax H3 is CFG-distilled.

        Returns:
            torch.Tensor: Packed-flat velocity ``(B, Nv * 96 + Na * 32)``.
        """
        del negative_model_inputs
        video_rows = model_inputs["video_rows"]
        audio_rows = model_inputs["audio_rows"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
        audio_timestep = model_inputs["audio_timestep"]
        meta = model_inputs["latent_meta"]
        device = video_rows.device
        # The video patch is a fixed checkpoint property; fall back for the CPU-test stub module.
        raw_patch = getattr(getattr(module, "config", None), "patch_size", (1, 2, 2))
        patch_size = (int(raw_patch[0]), int(raw_patch[1]), int(raw_patch[2]))

        batch = video_rows.shape[0]
        if encoder_mask is not None:
            text_lengths = encoder_mask.long().sum(dim=1).tolist()  # one host sync per micro-batch
        else:
            text_lengths = [encoder_hidden_states.shape[1]] * batch
        video_ts = timestep.tolist()
        audio_ts = audio_timestep.tolist()

        packed_velocities = []
        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = (
                build_layout_from_meta(meta, num_text_tokens, patch_size)
            )
            # flow_grpo reverses two schedules, so video and audio rows sit at different noise levels;
            # route each modality to its own timestep (text inherits the video timestep). t2va has no
            # conditioning rows, so the condition timesteps are unused.
            unique_timesteps, timestep_indices = build_row_timesteps(
                video_indices,
                audio_indices,
                num_cond_video,
                num_cond_audio,
                num_text_tokens,
                video_timestep=video_ts[index],
                audio_timestep=audio_ts[index],
                condition_video_timestep=video_ts[index],
                condition_audio_timestep=audio_ts[index],
            )
            # TODO(gpu-bringup): fl2va keyframe conditioning tags a keyframe's vision-block rows 0 (video)
            # inside the text stream and passes keyframe_anchors; t2va (this path) tags all text rows 1.
            result = module(
                hidden_states=video_rows[index : index + 1],
                audio_hidden_states=audio_rows[index : index + 1],
                encoder_hidden_states=encoder_hidden_states[index : index + 1, :num_text_tokens],
                timestep=unique_timesteps.to(device),
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags.to(device),
                position_ids=position_ids.to(device),
                video_indices=video_indices.to(device),
                audio_indices=audio_indices.to(device),
                text_indices=text_indices.to(device),
                return_dict=False,
            )
            v_video, v_audio = split_dual_velocity(result)
            packed_velocities.append(pack_video_audio_rows(v_video, v_audio))
        return torch.cat(packed_velocities, dim=0)

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs,
        step: int,
    ):
        """Run the transformer and sample the previous reverse-SDE step per modality.

        Samples the video and audio streams through their own scheduler legs on
        their own timestep schedules, mean-reduces each stream's log-prob (done
        inside the scheduler), and combines them as
        ``w_video * lp_video + w_audio * lp_audio``. Weights come from
        ``model_config.pipeline.av_logprob_{video,audio}_weight``; the rollout
        computes ``all_log_probs`` with the same weights so the training ratio is
        valid.

        Args:
            module: The MiniMax H3 transformer module.
            scheduler: The dual-stream scheduler from :meth:`build_scheduler`.
            model_config: Configuration providing ``algo.noise_level`` /
                ``algo.sde_type`` and the per-modality log-prob weights.
            model_inputs: Positive-prompt inputs from :meth:`prepare_model_inputs`.
            negative_model_inputs: Unused; MiniMax H3 is CFG-distilled.
            scheduler_inputs: Micro-batch with ``all_latents``, ``all_timesteps``,
                ``audio_all_timesteps`` and ``latent_meta``.
            step: Current denoising-step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`` where
                *log_prob* is the combined per-modality log-prob ``(B,)`` and
                *prev_sample_mean* is the packed dual mean; *std_dev_t* / *sqrt_dt*
                are the video stream's (``FlowGRPOLoss`` reads only *log_probs*).
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]
        audio_timesteps = scheduler_inputs["audio_all_timesteps"]
        meta = scheduler_inputs["latent_meta"][0].tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])

        packed_velocity = cls.forward(module, model_config, model_inputs)
        v_video, v_audio = unpack_video_audio_rows(packed_velocity, num_video_rows, num_audio_rows)
        cur_video, cur_audio = unpack_video_audio_rows(latents[:, step].float(), num_video_rows, num_audio_rows)
        prev_video, prev_audio = unpack_video_audio_rows(latents[:, step + 1].float(), num_video_rows, num_audio_rows)

        noise_level = model_config.algo.noise_level
        sde_type = model_config.algo.sde_type

        _, lp_video, mean_video, std_dev_t, sqrt_dt = scheduler.video_scheduler.sample_previous_step(
            sample=cur_video,
            model_output=v_video.float(),
            timestep=timesteps[:, step],
            noise_level=noise_level,
            prev_sample=prev_video,
            sde_type=sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        _, lp_audio, mean_audio, _, _ = scheduler.audio_scheduler.sample_previous_step(
            sample=cur_audio,
            model_output=v_audio.float(),
            timestep=audio_timesteps[:, step],
            noise_level=noise_level,
            prev_sample=prev_audio,
            sde_type=sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        w_video = model_config.pipeline.av_logprob_video_weight
        w_audio = model_config.pipeline.av_logprob_audio_weight
        log_prob = w_video * lp_video + w_audio * lp_audio
        prev_sample_mean = pack_video_audio_rows(mean_video, mean_audio)
        # std_dev_t / sqrt_dt are per-modality; FlowGRPOLoss reads only log_probs so
        # the video stream's are returned. TODO(gpu-bringup): FlowDPPO / GRPOGuard
        # loss modes consume these and would need per-modality handling.
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

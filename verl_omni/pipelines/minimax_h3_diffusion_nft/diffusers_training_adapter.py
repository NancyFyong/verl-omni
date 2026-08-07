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
"""MiniMax H3 training adapter for DiffusionNFT.

MiniMax H3 is a CFG-distilled joint video+audio rectified-flow transformer that
consumes and produces separate video rows (width 96) and audio rows (width 32).
DiffusionNFT's shared engine noises a single ``latents_clean`` tensor at one
timestep per sample and its loss is fully elementwise, so both row streams are
packed into one flat vector on the rollout side (see :mod:`.common`). This
adapter unpacks that vector into the two row streams, runs the transformer, and
re-packs the ``(v_video, v_audio)`` velocity into the same flat layout.

The transformer reads one packed sequence whose timestep plan and row layout are
shared across the batch — its ``timestep``/``token_tags``/``position_ids`` carry
no batch dimension and it takes no attention mask. DiffusionNFT samples a
per-sample timestep and prompts vary in length, so the forward runs one
micro-batch sample at a time, each with its own sampled timestep and true text
length, and stacks the packed results.
"""

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    build_layout_from_meta,
    pack_video_audio_rows,
    split_dual_velocity,
    unpack_video_audio_rows,
)

__all__ = ["MiniMaxH3DiffusionNFT"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFT(DiffusionModelBase):
    """Forward-process MiniMax H3 adapter used by DiffusionNFT."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build a rectified-flow scheduler shifted for the video stream.

        DiffusionNFT noises ``latents_clean`` directly and never samples through
        the scheduler, so it is only used to satisfy the engine contract. Audio
        uses a separate shift at rollout time; the packed training objective
        shares the video timestep (Option C, see the RFC).

        Args:
            model_config: Configuration for the diffusion model.

        Returns:
            FlowMatchEulerDiscreteScheduler: Scheduler with timesteps set.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipeline = model_config.pipeline
        # TODO(gpu-bringup): confirm MiniMax H3 ships a diffusers scheduler config;
        # if so, prefer FlowMatchEulerDiscreteScheduler.from_pretrained(subfolder="scheduler").
        scheduler = FlowMatchEulerDiscreteScheduler(shift=pipeline.get("video_flow_shift", 12.0))
        cls.set_timesteps(scheduler, model_config, device="cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Set the video-stream timesteps on the scheduler.

        Args:
            scheduler: The scheduler whose timesteps will be set.
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
    ) -> tuple[dict, Optional[dict]]:
        """Unpack the packed-flat latent into video + audio rows and gather forward inputs.

        The static packed-sequence layout (``token_tags`` / ``position_ids`` / row indices) depends on
        the per-sample text length, so it is built inside :meth:`forward`; this method only unpacks the
        two row streams and forwards the raw pieces the loop needs.

        Args:
            module: The MiniMax H3 transformer module.
            model_config: Configuration for the diffusion model.
            latents: Packed-flat noised latent ``xt`` of shape ``(B, Nv * 96 + Na * 32)``.
            timesteps: Per-sample timestep of shape ``(B,)`` in ``[0, 1000]``.
            prompt_embeds: Text embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask: Text-length mask, shape ``(B, L)`` (``None`` -> use the full ``L``).
            negative_prompt_embeds: Unused; MiniMax H3 is CFG-distilled.
            negative_prompt_embeds_mask: Unused; MiniMax H3 is CFG-distilled.
            micro_batch: Micro-batch carrying ``latent_meta`` ``(B, 6)`` =
                ``[Nv, Na, latent_t, latent_h, latent_w, audio_t]``.
            step: Current denoising-step index (unused; forward-process objective).

        Returns:
            tuple[dict, None]: ``(model_inputs, None)`` — no negative branch.
        """
        del step, negative_prompt_embeds, negative_prompt_embeds_mask
        # All samples in a micro-batch share one resolution/duration, so read row 0.
        meta = micro_batch["latent_meta"][0].tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
        video_rows, audio_rows = unpack_video_audio_rows(latents, num_video_rows, num_audio_rows)

        model_inputs = {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": timesteps.float() / 1000.0,
            "latent_meta": meta,
        }
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run the transformer per micro-batch sample and re-pack the dual velocity into packed-flat form.

        The transformer reads one packed sequence whose timestep plan and row layout are shared across
        the batch (its ``timestep`` / ``token_tags`` / ``position_ids`` carry no batch dim, and it takes
        no attention mask), so each sample is run on its own — with its sampled ``timestep`` and true
        text length — and the packed velocities are stacked back to ``(B, ...)``.

        Args:
            module: The MiniMax H3 transformer module.
            model_config: Configuration for the diffusion model (unused).
            model_inputs: Inputs from :meth:`prepare_model_inputs`.
            negative_model_inputs: Unused; MiniMax H3 is CFG-distilled.

        Returns:
            torch.Tensor: Packed-flat velocity of shape ``(B, Nv * 96 + Na * 32)``, matching ``xt`` so
                the shared elementwise loss applies directly.
        """
        del negative_model_inputs
        video_rows = model_inputs["video_rows"]
        audio_rows = model_inputs["audio_rows"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
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

        packed_velocities = []
        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = build_layout_from_meta(
                meta, num_text_tokens, patch_size
            )
            # TODO(gpu-bringup): fl2va keyframe conditioning tags a keyframe's vision-block rows 0 (video)
            # inside the text stream and passes keyframe_anchors; t2va (this path) tags all text rows 1.
            # TODO(gpu-bringup): every row shares the sampled timestep (timestep_indices all 0), matching
            # the engine noising the whole packed latent at one level; real H3 may hold text/conditioning
            # rows at a distinct clean level, which would make timestep a 2-vector with text -> index 1.
            result = module(
                hidden_states=video_rows[index : index + 1],
                audio_hidden_states=audio_rows[index : index + 1],
                encoder_hidden_states=encoder_hidden_states[index : index + 1, :num_text_tokens],
                timestep=timestep[index : index + 1],
                timestep_indices=torch.zeros(position_ids.shape[0], dtype=torch.long, device=device),
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
        """Not used by DiffusionNFT (a forward-process objective).

        Reverse-sampling log-probabilities belong to policy-gradient algorithms
        (FlowGRPO/DanceGRPO), which are a separate RFC-0001 milestone.
        """
        raise NotImplementedError(
            "MiniMaxH3DiffusionNFT is a forward-process objective and does not "
            "sample the reverse SDE. Reverse-sampling (flow_grpo) is a separate milestone."
        )

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
"""MiniMax H3 training adapter for FlowGRPO."""

from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    build_layout_from_meta,
    build_row_timesteps,
    h3_dit_timestep,
    h3_velocity_to_flow_match,
    pack_video_audio_rows,
    split_dual_velocity,
    unpack_video_audio_rows,
    validate_lora_target_modules,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["MiniMaxH3FlowGRPO"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3FlowGRPO(DiffusionModelBase):
    """Run dual-stream MiniMax H3 FlowGRPO actor replay."""

    @classmethod
    def validate_lora_config(cls, model_config: DiffusionModelConfig) -> None:
        """Reject LoRA targets the rollout weight sync cannot transport (shares common.py whitelist)."""
        if model_config.lora_rank > 0:
            validate_lora_target_modules(model_config.target_modules)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build the video and audio SDE schedulers."""
        pipeline = model_config.pipeline
        scheduler = FlowMatchDualSDEDiscreteScheduler(
            video_flow_shift=pipeline.video_flow_shift,
            audio_flow_shift=pipeline.audio_flow_shift,
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Set both modality schedules."""
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
        """Unpack one trajectory step and prepare H3 inputs."""
        del negative_prompt_embeds, negative_prompt_embeds_mask
        meta = micro_batch["latent_meta"][0].reshape(-1).tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
        video_rows, audio_rows = unpack_video_audio_rows(latents[:, step], num_video_rows, num_audio_rows)
        audio_timesteps = micro_batch["audio_all_timesteps"][:, step]
        return {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": h3_dit_timestep(timesteps[:, step].float()),
            "audio_timestep": h3_dit_timestep(audio_timesteps.float()),
            "latent_meta": meta,
        }, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run H3 per sample and return packed flow-match velocities."""
        del negative_model_inputs
        video_rows = model_inputs["video_rows"]
        audio_rows = model_inputs["audio_rows"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
        audio_timestep = model_inputs["audio_timestep"]
        meta = model_inputs["latent_meta"]
        device = video_rows.device
        raw_patch = getattr(getattr(module, "config", None), "patch_size", (1, 2, 2))
        patch_size = (int(raw_patch[0]), int(raw_patch[1]), int(raw_patch[2]))

        batch = video_rows.shape[0]
        if encoder_mask is not None:
            text_lengths = encoder_mask.long().sum(dim=1).tolist()
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
            video_velocity, audio_velocity = split_dual_velocity(result)
            packed_velocities.append(
                pack_video_audio_rows(
                    h3_velocity_to_flow_match(video_velocity),
                    h3_velocity_to_flow_match(audio_velocity),
                )
            )
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
        """Recompute and score one video/audio reverse-SDE transition."""
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]
        audio_timesteps = scheduler_inputs["audio_all_timesteps"]
        meta = scheduler_inputs["latent_meta"][0].reshape(-1).tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])

        packed_velocity = cls.forward(module, model_config, model_inputs)
        video_velocity, audio_velocity = unpack_video_audio_rows(packed_velocity, num_video_rows, num_audio_rows)
        current_video, current_audio = unpack_video_audio_rows(latents[:, step].float(), num_video_rows, num_audio_rows)
        previous_video, previous_audio = unpack_video_audio_rows(
            latents[:, step + 1].float(), num_video_rows, num_audio_rows
        )

        noise_level = model_config.algo.noise_level
        sde_type = model_config.algo.sde_type
        _, video_log_prob, video_mean, std_dev_t_video, sqrt_dt_video = scheduler.video_scheduler.sample_previous_step(
            sample=current_video,
            model_output=video_velocity.float(),
            timestep=timesteps[:, step],
            noise_level=noise_level,
            prev_sample=previous_video,
            sde_type=sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        _, audio_log_prob, audio_mean, std_dev_t_audio, sqrt_dt_audio = scheduler.audio_scheduler.sample_previous_step(
            sample=current_audio,
            model_output=audio_velocity.float(),
            timestep=audio_timesteps[:, step],
            noise_level=noise_level,
            prev_sample=previous_audio,
            sde_type=sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        pipeline = model_config.pipeline
        w_video = pipeline.av_logprob_video_weight
        w_audio = pipeline.av_logprob_audio_weight
        log_prob = w_video * video_log_prob + w_audio * audio_log_prob
        # The streams run separate sigma schedules, so weight-average their scales the same way
        # as the log-probs; only GRPO-Guard / FlowDPPO read them, FlowGRPOLoss reads log_probs.
        weight_total = w_video + w_audio
        std_dev_t = (w_video * std_dev_t_video + w_audio * std_dev_t_audio) / weight_total
        sqrt_dt = (w_video * sqrt_dt_video + w_audio * sqrt_dt_audio) / weight_total
        return log_prob, pack_video_audio_rows(video_mean, audio_mean), std_dev_t, sqrt_dt

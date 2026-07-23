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

"""Training-side FlowGRPO adapter for the LingBot-Video Dense T2V model."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_cfg, guidance_scale, shifted_sigmas

__all__ = ["LingBotVideoDenseFlowGRPO"]


def _pipeline_value(pipeline, name: str, default):
    value = getattr(pipeline, name, None)
    if value is None and hasattr(pipeline, "get"):
        value = pipeline.get(name, None)
    return default if value is None else value


@DiffusionModelBase.register("LingBotVideoPipeline", algorithm="flow_grpo")
class LingBotVideoDenseFlowGRPO(DiffusionModelBase):
    """FlowGRPO integration for the non-MoE LingBot Video transformer.

    LingBot's transformer is supplied by the optional, pinned
    ``lingbot-video`` package rather than the installed diffusers package.  It
    is loaded as a bare module so FSDP/PEFT see ``blocks.*`` directly.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> Optional[torch.nn.Module]:
        try:
            from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        except ImportError as exc:
            raise ImportError(
                "LingBot Dense FlowGRPO requires the optional `lingbot-video` dependency. "
                "Install `verl-omni[lingbot-video]` before starting training."
            ) from exc

        module = LingBotVideoTransformer3DModel.from_pretrained(
            model_config.local_path,
            subfolder=model_config.transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        num_experts = int(getattr(module.config, "num_experts", 0))
        if num_experts != 0:
            raise ValueError(
                "LingBotVideoDenseFlowGRPO supports only robbyant/lingbot-video-dense-1.3b "
                f"(transformer.config.num_experts == 0), got {num_experts}."
            )
        return module

    @classmethod
    def configure_train_mode(cls, module: torch.nn.Module) -> None:
        # The upstream implementation explicitly does not implement gradient
        # checkpointing.  Keeping it disabled avoids a misleading config that
        # appears memory-safe but has no effect.
        if getattr(module, "_supports_gradient_checkpointing", False):
            raise RuntimeError("LingBot Dense unexpectedly advertises gradient checkpointing support.")

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        pipeline = model_config.pipeline
        num_steps = int(_pipeline_value(pipeline, "num_inference_steps", 40))
        shift = float(_pipeline_value(pipeline, "shift", 3.0))
        scheduler.set_timesteps(num_steps, device=device, sigmas=shifted_sigmas(num_steps, shift))

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        del module, micro_batch
        selected_latents = latents[:, step]
        # LingBot expects the scheduler's [0, 1000] timestep values, not the
        # normalized sigma.  The official pipeline preserves this scale.
        selected_timesteps = timesteps[:, step].float()
        model_inputs = {
            "hidden_states": selected_latents,
            "timestep": selected_timesteps,
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }

        if guidance_scale(model_config.pipeline) > 1.0:
            if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
                raise ValueError("LingBot CFG requires negative prompt embeddings and their attention mask.")
            negative_model_inputs: Optional[dict] = {
                "hidden_states": selected_latents,
                "timestep": selected_timesteps,
                "encoder_hidden_states": negative_prompt_embeds,
                "encoder_attention_mask": negative_prompt_embeds_mask,
                "return_dict": False,
            }
        else:
            negative_model_inputs = None
        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        if scheduler_inputs is None:
            raise ValueError("LingBot FlowGRPO requires rollout `all_latents` and `all_timesteps`.")
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        scale = guidance_scale(model_config.pipeline)
        if scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LingBot CFG requires negative model inputs.")
            noise_pred = apply_cfg(noise_pred, cls.forward(module, model_config, negative_model_inputs), scale)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

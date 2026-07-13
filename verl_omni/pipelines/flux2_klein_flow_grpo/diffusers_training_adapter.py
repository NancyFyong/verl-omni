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

"""FLUX.2-Klein I2I training adapter for diffusion RL.

Port of the UniRL flux2_klein i2i logic onto the verl-omni
:class:`DiffusionI2IModelBase` contract. The transformer is a
``Flux2Transformer2DModel`` whose forward accepts the official FLUX.2
4-axis RoPE ids ``(T, H, W, L)`` via ``txt_ids`` / ``img_ids``; condition
image tokens are distinguished from noise tokens by a time-axis offset
(``REFERENCE_TIME_SCALE = 10``) on their RoPE ids.
"""

from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["Flux2KleinFlowGRPO"]


# --------------------------------------------------------------------------- #
# FLUX.2-Klein pure helpers (ported from UniRL ``flux2_klein_utils.py`` /
# diffusers ``Flux2KleinPipeline``).                                          #
# --------------------------------------------------------------------------- #

# Time-axis offset for the single reference image: diffusers uses
# ``scale + scale * t``; for one image t=0 → T-coord = scale = 10.
REFERENCE_TIME_SCALE: int = 10


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Mirror ``Flux2KleinPipeline.compute_empirical_mu`` from diffusers."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, H*W, C] -> [B, C, H, W]."""
    batch_size, _, num_channels = tokens.shape
    return tokens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)


def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """FLUX.2 text RoPE ids ``(T=0, H=0, W=0, L=token_idx)``. Shape ``[B, L, 4]``."""
    batch_size, seq_len, _ = prompt_embeds.shape
    t = torch.arange(1, device=prompt_embeds.device)
    h = torch.arange(1, device=prompt_embeds.device)
    w = torch.arange(1, device=prompt_embeds.device)
    s = torch.arange(seq_len, device=prompt_embeds.device)
    coords = torch.cartesian_prod(t, h, w, s)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """FLUX.2 latent RoPE ids for patchified ``[B, C, H, W]``.

    Returns ``[B, H*W, 4]`` with ``(T=0, h_idx, w_idx, L=0)``.
    """
    batch_size, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    s = torch.arange(1, device=latents.device)
    coords = torch.cartesian_prod(t, h, w, s)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def _flux_latent_hw(height: int, width: int, vae_scale_factor: int = 8) -> tuple[int, int]:
    """Patchified latent spatial size for a ``height x width`` image."""
    patch_size = 2
    downsample = vae_scale_factor * patch_size
    if int(height) % downsample != 0 or int(width) % downsample != 0:
        raise ValueError(f"FLUX.2-Klein height ({height}) and width ({width}) must be divisible by {downsample}.")
    return int(height) // downsample, int(width) // downsample


def _flux_image_seq_len(height: int, width: int, vae_scale_factor: int = 8) -> int:
    """Packed token count for a ``height x width`` image."""
    h, w = _flux_latent_hw(height, width, vae_scale_factor)
    return h * w


def _true_cfg_scale(model_config: DiffusionModelConfig) -> float:
    """True CFG scale shared by FLUX.2-Klein rollout and training."""
    value = getattr(model_config.pipeline, "true_cfg_scale", 1.0)
    return 1.0 if value is None else float(value)


def _build_guidance(module, model_config: DiffusionModelConfig, batch_size: int, device, dtype) -> torch.Tensor | None:
    if not bool(getattr(getattr(module, "config", None), "guidance_embeds", False)):
        return None
    guidance_scale = model_config.pipeline.guidance_scale
    if guidance_scale is None:
        raise ValueError("FLUX.2-Klein guidance_embeds models require pipeline.guidance_scale.")
    return torch.full((batch_size,), float(guidance_scale), device=device, dtype=dtype)


def _get_constant_batch_value(micro_batch: TensorDict, key: str):
    value = tu.get(micro_batch, key)
    if not isinstance(value, list | tuple):
        return value
    if not value:
        return None
    first = value[0]
    if any(item != first for item in value[1:]):
        raise ValueError(f"Flux2KleinFlowGRPO: {key} differs across the micro-batch: {value!r}")
    return first


# --------------------------------------------------------------------------- #
# Training adapter                                                             #
# --------------------------------------------------------------------------- #


@DiffusionModelBase.register("Flux2KleinPipeline", algorithm="flow_grpo")
class Flux2KleinFlowGRPO(DiffusionI2IModelBase):
    """Training adapter for FLUX.2-Klein image editing.

    Reuses the standard concat-crop i2i pattern from
    :class:`DiffusionI2IModelBase`, but overrides :meth:`inject_condition` to
    also concatenate the 4-axis ``img_ids`` (FLUX.2 distinguishes condition
    vs noise tokens via the RoPE time-axis offset, not via ``img_shapes``).
    """

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
    ):
        image_seq_len = _flux_image_seq_len(model_config.pipeline.height, model_config.pipeline.width)
        mu = compute_empirical_mu(image_seq_len, model_config.pipeline.num_inference_steps)
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device, mu=mu)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        selected_latents = latents[:, step]  # [B, N, 128] (packed)
        batch_size, num_tokens, _ = selected_latents.shape
        latent_h = _get_constant_batch_value(micro_batch, "latent_h")
        latent_w = _get_constant_batch_value(micro_batch, "latent_w")
        if latent_h is None or latent_w is None:
            latent_h, latent_w = _flux_latent_hw(model_config.pipeline.height, model_config.pipeline.width)
        latent_h, latent_w = int(latent_h), int(latent_w)
        if latent_h * latent_w != num_tokens:
            raise ValueError(
                f"Flux2KleinFlowGRPO.prepare_model_inputs: latent_h*latent_w "
                f"({latent_h}*{latent_w}) does not match packed token count ({num_tokens})."
            )

        packed = selected_latents
        img_ids = prepare_latent_ids(
            torch.empty(batch_size, 1, latent_h, latent_w, device=selected_latents.device)
        )  # [B, N, 4] T=0
        txt_ids = prepare_text_ids(prompt_embeds)  # [B, L, 4]
        guidance = _build_guidance(module, model_config, batch_size, selected_latents.device, packed.dtype)

        model_inputs = {
            "hidden_states": packed,
            "encoder_hidden_states": prompt_embeds,
            "timestep": timesteps[:, step].float() / 1000,
            "guidance": guidance,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            "return_dict": False,
        }

        true_cfg_scale = _true_cfg_scale(model_config)
        if true_cfg_scale > 1.0:
            if negative_prompt_embeds is None:
                raise ValueError("Flux2Klein CFG requires negative prompt embeds when true_cfg_scale > 1.")
            neg_txt_ids = prepare_text_ids(negative_prompt_embeds)
            negative_model_inputs = {
                "hidden_states": packed,
                "encoder_hidden_states": negative_prompt_embeds,
                "timestep": timesteps[:, step].float() / 1000,
                "guidance": guidance,
                "txt_ids": neg_txt_ids,
                "img_ids": img_ids,
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
        assert scheduler_inputs is not None
        all_latents = scheduler_inputs["all_latents"]
        all_timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        true_cfg_scale = _true_cfg_scale(model_config)
        if true_cfg_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("Flux2Klein CFG requires negative model inputs when true_cfg_scale > 1.")
            neg_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=all_latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=all_timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=all_latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt

    # ----- I2I hooks ----- #

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> Optional[dict]:
        del latents, step
        image_latents = micro_batch.get("condition_image_latents", None)
        if image_latents is None:
            # Detect wrong key name: rollout output "image_latents" instead of "condition_image_latents".
            if "image_latents" in micro_batch:
                raise ValueError(
                    "Flux2KleinFlowGRPO.prepare_condition: "
                    "micro_batch has 'image_latents' but not 'condition_image_latents'. "
                    "The rollout adapter likely output the wrong key. "
                    "Use 'condition_image_latents' in custom_output to avoid "
                    "colliding with the MFU FLOPs counter."
                )
            return None
        image_latent_ids = micro_batch.get("condition_image_latent_ids", None)
        if image_latent_ids is None:
            raise ValueError(
                "Flux2KleinFlowGRPO.prepare_condition: `condition_image_latents` "
                "set but `condition_image_latent_ids` is missing. Both are "
                "required for the FLUX.2 4-axis RoPE edit path."
            )
        if image_latent_ids.shape[-2] != image_latents.shape[-2]:
            # One RoPE id per packed condition token — a mismatch means the
            # rollout packed N condition images but emitted ids for M != N.
            raise ValueError(
                "Flux2KleinFlowGRPO.prepare_condition: condition token count mismatch — "
                f"image_latents has {image_latents.shape[-2]} tokens but "
                f"image_latent_ids has {image_latent_ids.shape[-2]}."
            )
        return {
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
            "sp_size": tu.get(micro_batch, "sp_size"),
        }

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        """Concat condition tokens **and** their 4-axis RoPE ids.

        Overrides the base (which only concatenates ``hidden_states``) because
        FLUX.2 locates every token via ``img_ids`` — the condition tokens must
        carry their T-offset ids (``REFERENCE_TIME_SCALE``) so the transformer
        can tell them apart from the noise tokens (T=0).
        """
        if not condition:
            return model_inputs, negative_model_inputs

        image_latents = condition.get("image_latents")
        if image_latents is None:
            return model_inputs, negative_model_inputs
        image_latent_ids = condition.get("image_latent_ids")

        hidden_states = model_inputs["hidden_states"]
        if image_latents.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "inject_condition: condition image_latents batch size "
                f"({image_latents.shape[0]}) does not match hidden_states batch size "
                f"({hidden_states.shape[0]})."
            )

        target_seq_len = hidden_states.shape[1]
        combined_seq_len = target_seq_len + image_latents.shape[1]
        sp_size = condition.get("sp_size")
        if isinstance(sp_size, int) and sp_size > 1 and combined_seq_len % sp_size != 0:
            raise ValueError(
                "inject_condition: combined noise+condition token length "
                f"({combined_seq_len} = {target_seq_len} + {image_latents.shape[1]}) "
                f"is not divisible by sequence-parallel size ({sp_size}). "
                "Choose a condition image resolution whose packed latent length keeps "
                "the combined sequence divisible by sp_size (align at rollout-side VAE "
                "encode); do not zero-pad the condition here."
            )

        for inputs in (model_inputs, negative_model_inputs):
            if inputs is None:
                continue
            inputs["hidden_states"] = torch.cat(
                [
                    inputs["hidden_states"],
                    image_latents.to(
                        device=inputs["hidden_states"].device,
                        dtype=inputs["hidden_states"].dtype,
                    ),
                ],
                dim=1,
            )
            # Concat img_ids for 4-axis RoPE with T-offset for condition tokens.
            if image_latent_ids is not None and "img_ids" in inputs:
                inputs["img_ids"] = torch.cat(
                    [
                        inputs["img_ids"],
                        image_latent_ids.to(
                            device=inputs["img_ids"].device,
                            dtype=inputs["img_ids"].dtype,
                        ),
                    ],
                    dim=1,
                )
            inputs["_target_seq_len"] = target_seq_len

        return model_inputs, negative_model_inputs

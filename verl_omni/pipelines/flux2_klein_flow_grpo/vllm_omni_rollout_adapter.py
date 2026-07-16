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

"""FLUX.2-Klein vLLM-Omni rollout adapter for FlowGRPO with SDE log-prob collection.

Extends the upstream :class:`Flux2KleinPipeline` to:
1. Replace the scheduler with :class:`FlowMatchSDEDiscreteScheduler`.
2. Override :meth:`encode_prompt` to accept pre-tokenized prompt IDs (the
   verl-omni agent loop passes ``prompt_token_ids``, not raw text) and run
   the Qwen3 layer-(9,18,27) concatenation.
3. Collect per-step latents, log-probs, and timesteps during the SDE window.
4. Return prompt embeddings and condition image latents + ids in
   :class:`DiffusionOutput.custom_output` for the trainer.

Condition images are parsed via :class:`ImageGenerationRequest` and encoded
through the upstream :meth:`prepare_image_latents` (VAE encode + patchify +
BN-normalize + pack + 4-axis RoPE ids with T-offset).
"""

from __future__ import annotations

import ast
import os
from typing import Any, Literal

import torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux2_klein import Flux2KleinPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import ImageGenerationRequest

from .diffusers_training_adapter import (
    _flux_latent_hw,
    compute_empirical_mu,
    prepare_latent_ids,
    prepare_text_ids,
    resolve_flux2_klein_sigmas,
    unpack_latents,
)

__all__ = ["Flux2KleinPipelineWithLogProb"]


def _coalesce_not_none(value, default):
    return default if value is None else value


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _uses_guidance_embeds(transformer) -> bool:
    return bool(getattr(getattr(transformer, "config", None), "guidance_embeds", False))


def _build_guidance(transformer, guidance_scale: float | None, batch_size: int, device, dtype) -> torch.Tensor | None:
    if not _uses_guidance_embeds(transformer):
        return None
    if guidance_scale is None:
        raise ValueError("FLUX.2-Klein guidance_embeds models require pipeline.guidance_scale.")
    return torch.full((batch_size,), float(guidance_scale), device=device, dtype=dtype)


def _use_true_cfg(
    true_cfg_scale: float,
    negative_prompt_ids,
    negative_prompt_embeds,
    negative_prompt_embeds_mask,
) -> bool:
    enabled = true_cfg_scale > 1
    has_negative_prompt = negative_prompt_ids is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    if enabled and not has_negative_prompt:
        raise ValueError("FLUX.2-Klein true_cfg_scale > 1 requires negative_prompt_ids or negative prompt embeddings.")
    return enabled


def _preprocess_condition_image(image, image_processor, vae_scale_factor: int) -> torch.Tensor | None:
    """Mirror upstream Flux2KleinPipeline reference-image preprocessing."""
    multiple_of = vae_scale_factor * 2
    if isinstance(image, Image.Image):
        image_processor.check_image_input(image)
        image_width, image_height = image.size
        if image_width * image_height > 1024 * 1024:
            image = image_processor._resize_to_target_area(image, 1024 * 1024)
            image_width, image_height = image.size
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of
        return image_processor.preprocess(image, height=image_height, width=image_width, resize_mode="crop")

    if isinstance(image, torch.Tensor):
        tensor = image if image.ndim == 4 else image.unsqueeze(0)
        height, width = tensor.shape[-2:]
        crop_h = (height // multiple_of) * multiple_of
        crop_w = (width // multiple_of) * multiple_of
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(
                f"FLUX.2-Klein condition tensor spatial size ({height}, {width}) "
                f"must be at least {multiple_of} pixels on each axis."
            )
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        tensor = tensor[..., top : top + crop_h, left : left + crop_w]
        if not torch.is_floating_point(tensor):
            tensor = tensor.float() / 255.0
        tensor_min = tensor.amin()
        tensor_max = tensor.amax()
        if tensor_min >= 0 and tensor_max > 1:
            tensor = tensor / 255.0
            tensor_min = tensor.amin()
            tensor_max = tensor.amax()
        if tensor_min >= 0 and tensor_max <= 1:
            tensor = tensor * 2.0 - 1.0
        return tensor

    return None


def _normalize_sde_window_args(
    sde_window_size: int | str | None,
    sde_window_range: tuple[int, int] | list[int] | str,
) -> tuple[int | None, tuple[int, int]]:
    if sde_window_size is not None:
        sde_window_size = int(sde_window_size)
    if isinstance(sde_window_range, str):
        sde_window_range = ast.literal_eval(sde_window_range)
    if len(sde_window_range) != 2:
        raise ValueError("Flux2Klein rollout sde_window_range must contain exactly two values.")
    return sde_window_size, (int(sde_window_range[0]), int(sde_window_range[1]))


@VllmOmniPipelineBase.register("Flux2KleinPipeline", algorithm="flow_grpo")
class Flux2KleinPipelineWithLogProb(VllmOmniPipelineBase, Flux2KleinPipeline):
    """Rollout pipeline for FLUX.2-Klein that captures per-step log-probabilities.

    Registered under ``"Flux2KleinPipeline"`` for vllm-omni rollout dispatch.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Replace the upstream scheduler with our SDE scheduler.
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    # ----- prompt encoding ----- #

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        condition_images: list | None = None,  # unused (flux text enc is image-agnostic)
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ):
        """Encode pre-tokenized prompt IDs into dense Qwen3 layer-concat embeds.

        Overrides the upstream :meth:`Flux2KleinPipeline.encode_prompt` (which
        accepts raw text strings) to work with pre-tokenized prompt IDs as
        required by the verl-omni rollout loop. FLUX.2-Klein's text encoder is
        a Qwen3 LLM; we take hidden states from layers (9, 18, 27) and
        concatenate them into ``[B, L, 3*hidden]`` (= ``[B, L, 15360]``).
        ``condition_images`` is accepted for interface parity but ignored —
        FLUX.2 conditions on images via token concatenation, not via the text
        encoder vision tower.
        """
        del condition_images  # flux text encoding is image-agnostic

        if prompt_embeds is None:
            if prompt_ids is None:
                return None, None, None
            prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

            prompt_ids = prompt_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)

            output = self.text_encoder(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # Stack selected layers and concat: [B, 3, L, D] -> [B, L, 3*D]
            out = torch.stack([output.hidden_states[k] for k in text_encoder_out_layers], dim=1)
            out = out.to(self.text_encoder.dtype)
            batch_size, num_channels, seq_len, hidden_dim = out.shape
            prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
            prompt_embeds_mask = attention_mask

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if prompt_embeds_mask is not None:
                prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        txt_ids = prepare_text_ids(prompt_embeds).to(self.device)
        return prompt_embeds, prompt_embeds_mask, txt_ids

    # ----- SDE diffusion loop ----- #

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,  # [B, N, 128] packed noise tokens
        txt_ids: torch.Tensor,
        negative_txt_ids: torch.Tensor | None,
        image_latents: torch.Tensor | None,  # [B, Nc, 128] packed condition tokens
        image_latent_ids: torch.Tensor | None,  # [B, Nc, 4] condition RoPE ids (T=10)
        latent_hw: tuple[int, int],
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        guidance_scale: float | None,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator,
        logprobs: bool,
    ):
        """Run the full SDE diffusion loop for image editing with rollout data collection."""
        all_latents = []
        all_log_probs = []
        all_timesteps = []

        batch_size = latents.shape[0]
        noise_seq_len = latents.shape[1]
        latent_h, latent_w = latent_hw
        if latent_h * latent_w != noise_seq_len:
            raise ValueError(
                f"Flux2Klein diffuse: latent_h*latent_w ({latent_h}*{latent_w}) "
                f"does not match packed token count ({noise_seq_len})."
            )

        dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else prompt_embeds.dtype
        guidance = _build_guidance(self.transformer, guidance_scale, batch_size, latents.device, dtype)

        for i, timestep_value in enumerate(timesteps):
            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif sde_window[0] < i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            timestep = timestep_value.expand(batch_size).to(device=latents.device, dtype=torch.float32)
            img_ids = prepare_latent_ids(torch.empty(batch_size, 1, latent_h, latent_w, device=latents.device))

            # Concatenate condition tokens + ids (i2i edit path).
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                full_img_ids = torch.cat([img_ids, image_latent_ids.to(device=latents.device)], dim=1)
            else:
                latent_model_input = latents
                full_img_ids = img_ids

            latent_model_input = latent_model_input.to(dtype=dtype)

            # Positive forward.
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds.to(dtype=dtype),
                timestep=timestep / 1000,
                guidance=guidance,
                txt_ids=txt_ids.to(device=latents.device),
                img_ids=full_img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :noise_seq_len]  # slice back to noise tokens

            # CFG with negative prompt.
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=negative_prompt_embeds.to(dtype=dtype),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    txt_ids=negative_txt_ids.to(device=latents.device),
                    img_ids=full_img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0][:, :noise_seq_len]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            # Scheduler step (operates on packed [B, N, 128] latents).
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if sde_window[0] <= i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(batch_size, -1)
        return latents, all_latents, all_log_probs, all_timesteps

    # ----- end-to-end forward ----- #

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: Literal["latent", "pil", "np"] = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Any | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        noise_level: float = 1.0,
        sde_window_size: int | str | None = None,
        sde_window_range: tuple[int, int] | list[int] | str = (0, 4),
        sde_type: str = "sde",
        true_cfg_scale: float = 4.0,
        logprobs: bool = True,
        max_sequence_length: int = 512,
    ):
        """End-to-end image editing with rollout data collection."""
        custom_prompt = req.prompts[0] if req.prompts else {}

        # Parse condition images via the shared ImageGenerationRequest interface.
        gen_request = ImageGenerationRequest.from_request_payload(custom_prompt) if custom_prompt else None
        condition_images = gen_request.images if gen_request else None

        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_token_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)
            image_latents = custom_prompt.get("image_latents", image_latents)
        assert image_latents is None, "FLUX.2-Klein should not receive pre-encoded image_latents"
        sampling_params = req.sampling_params
        height = sampling_params.height or height or 1024
        width = sampling_params.width or width or 1024
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = _coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        guidance_scale = _coalesce_not_none(sampling_params.guidance_scale, guidance_scale)

        sde_window_size, sde_window_range = _normalize_sde_window_args(sde_window_size, sde_window_range)

        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            return DiffusionOutput(output=None, custom_output={})

        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        do_true_cfg = _use_true_cfg(
            true_cfg_scale,
            negative_prompt_ids,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        )

        # Encode prompts (Qwen3 layer concat). Returns (embeds, mask, txt_ids).
        prompt_embeds, prompt_embeds_mask, txt_ids = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        negative_txt_ids = None
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_txt_ids,
            ) = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        latent_h, latent_w = _flux_latent_hw(height, width)

        # Prepare noise latents: [B, 128, h, w] (patchified) -> packed [B, N, 128].
        num_latents_channels = self.transformer.config.in_channels // 4
        latents, _ = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_latents_channels,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )  # latents: [B, N, 128] packed, latent_ids discarded (recomputed in diffuse)

        # Prepare condition image latents (VAE encode + patchify + BN + pack + ids).
        condition_image_latents = None
        condition_image_latent_ids = None
        if condition_images is not None and len(condition_images) > 0:
            # Encode condition images through the Klein VAE path.
            cond_list = [condition_images] if not isinstance(condition_images, list) else condition_images
            image_tensors = []
            for img in cond_list:
                t = _preprocess_condition_image(img, self.image_processor, self.vae_scale_factor)
                if t is None:
                    # Fail closed: dropping one of N condition images would silently
                    # train on a partial condition with mismatched RoPE T-offsets.
                    raise ValueError(
                        f"Flux2Klein forward: condition image {len(image_tensors)} of "
                        f"{len(cond_list)} has unsupported type {type(img).__name__}; "
                        "expected PIL.Image.Image or torch.Tensor."
                    )
                image_tensors.append(t.to(device=self.device, dtype=self.vae.dtype))
            condition_image_latents, condition_image_latent_ids = self.prepare_image_latents(
                image_tensors,
                batch_size * num_images_per_prompt,
                generator,
                self.device,
                self.vae.dtype,
            )

        # Prepare timesteps with Klein empirical-mu shifting.
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        # Forward the official ``linspace(1, 1/N, N)`` base grid (or a caller/
        # request-pinned schedule) so the dynamic-shift scheduler matches the
        # pretrained FLUX.2-Klein inference trajectory instead of collapsing its
        # small-sigma tail. Training recomputes log-probs against the same grid.
        request_sigmas = sampling_params.sigmas if sampling_params.sigmas is not None else sigmas
        sigmas_schedule = resolve_flux2_klein_sigmas(request_sigmas, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, sigmas=sigmas_schedule, mu=mu)
        timesteps_tensor = self.scheduler.timesteps

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                max(sde_window_range[1] - sde_window_size + 1, sde_window_range[0] + 1),
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(timesteps_tensor) - 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            txt_ids,
            negative_txt_ids,
            condition_image_latents,
            condition_image_latent_ids,
            (latent_h, latent_w),
            timesteps_tensor,
            do_true_cfg,
            true_cfg_scale,
            guidance_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        self._current_timestep = None

        # VAE decode: unpack [B, N, 128] -> BN-denorm (128 ch, patchified)
        # -> unpatchify [B, 32, H, W] -> decode.

        latents_unpacked = unpack_latents(latents, latent_h, latent_w)  # [B, 128, h, w]
        # BN denormalize on patchified latents (128 channels).
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_unpacked.device, latents_unpacked.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1).to(latents_unpacked.device, latents_unpacked.dtype)
            + self.vae.config.batch_norm_eps
        )
        latents_denorm = latents_unpacked * latents_bn_std + latents_bn_mean
        latents_unpatched = self._unpatchify_latents(latents_denorm)  # [B, 32, H, W]
        image = self.vae.decode(latents_unpatched.to(self.vae.dtype), return_dict=False)[0]

        return DiffusionOutput(
            output=_maybe_to_cpu(image),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
                "condition_image_latents": _maybe_to_cpu(condition_image_latents),
                "condition_image_latent_ids": _maybe_to_cpu(condition_image_latent_ids),
                "latent_h": latent_h,
                "latent_w": latent_w,
            },
        )

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
"""Qwen-Image training adapter for DMD and DMD2.

Registers the architecture adapter and owns the differentiable phase computation
following LightX2V's public DMD equations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionAdversarialAdapter, DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    QWEN_IMAGE_VAE_SCALE_FACTOR,
    QwenImageTokenIdPromptMixin,
    build_img_shapes,
)
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
from verl_omni.trainer.diffusion.distillation.contracts import ConditionBundle, DistillationPlan, PhaseRequest
from verl_omni.trainer.diffusion.distillation.utils import (
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    consistency_renoise_step,
    dmd_gradient,
    dmd_surrogate_loss,
    fake_score_loss,
    ode_euler_step,
    standard_cfg,
    timestep_shift,
    velocity_to_x0,
)
from verl_omni.workers.config import DiffusionAdversarialConfig, DiffusionModelConfig

if TYPE_CHECKING:
    from verl_omni.workers.diffusion_distillation_worker import (
        DistillationPhaseComputation,
        DistillationRoleRuntime,
    )

__all__ = [
    "QwenImageDistributionMatching",
    "QwenImageDMDComputer",
    "build_qwen_dmd_sigmas",
    "qwen_vae_config_sha256",
]


def qwen_vae_config_sha256(model_path: str) -> str:
    """Fingerprint the canonical Qwen VAE config used to normalize cached latents."""
    from diffusers import AutoencoderKLQwenImage

    config = AutoencoderKLQwenImage.load_config(model_path, subfolder="vae")
    return hashlib.sha256(json.dumps(dict(config), sort_keys=True).encode()).hexdigest()


def build_qwen_dmd_sigmas(
    num_inference_steps: int,
    shift: float,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the fixed linear-shift schedule used by Qwen-Image DMD training."""
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}.")
    if shift < 1:
        raise ValueError(f"rollout_timestep_shift must be at least 1, got {shift}.")
    raw = torch.linspace(
        1.0,
        1.0 / num_inference_steps,
        num_inference_steps,
        device=device,
        dtype=torch.float32,
    )
    shifted = timestep_shift(raw * 1000.0, 1000, shift) / 1000.0
    return torch.cat((shifted, shifted.new_zeros(1)))


class QwenImageDMDProjection(torch.nn.Linear):
    """Preserve denoiser projection keys and classify pooled final DiT features."""

    def __init__(self, projection: torch.nn.Linear):
        super().__init__(
            projection.in_features,
            projection.out_features,
            bias=projection.bias is not None,
            device=projection.weight.device,
            dtype=projection.weight.dtype,
        )
        self.weight = projection.weight
        self.bias = projection.bias
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(projection.in_features),
            torch.nn.Linear(projection.in_features, 1),
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)
        with torch.no_grad():
            values = torch.arange(
                1,
                projection.in_features + 1,
                device=projection.weight.device,
                dtype=torch.float32,
            ).sin_()
            self.classifier[-1].weight.copy_(values.mul(projection.in_features**-0.5).unsqueeze(0))
            self.classifier[-1].bias.zero_()
        self.classify = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return [B, 1] logits or unchanged [B, tokens, channels] velocity."""
        if self.classify:
            return self.classifier(hidden_states.mean(dim=1))
        return super().forward(hidden_states)


class QwenImageConditionProvider:
    """Encode frozen local or precomputed Qwen prompt conditioning."""

    def __init__(
        self,
        model_path: str,
        provider: str,
        max_sequence_length: int,
        negative_prompt: str,
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self.max_sequence_length = max_sequence_length
        self.negative_prompt = negative_prompt
        self.pipeline = None
        self._negative_condition: Optional[ConditionBundle] = None

    @staticmethod
    def make_condition(prompt_embeds: torch.Tensor, prompt_mask: Optional[torch.Tensor]) -> ConditionBundle:
        """Build a detached [B, L, D] condition with a matching [B, L] mask."""
        if prompt_embeds.ndim != 3:
            raise ValueError(f"Qwen prompt embeddings must have shape [B, L, D], got {tuple(prompt_embeds.shape)}.")
        prompt_embeds = prompt_embeds.detach()
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long)
        elif prompt_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError(
                f"Qwen prompt mask shape {tuple(prompt_mask.shape)} does not match {tuple(prompt_embeds.shape[:2])}."
            )
        return ConditionBundle(
            tensors={"prompt_embeds": prompt_embeds},
            masks={"prompt_embeds": prompt_mask.detach()},
        )

    @staticmethod
    def require_tensor(batch: TensorDict, key: str) -> torch.Tensor:
        """Require a tensor-valued precomputed conditioning field."""
        value = tu.get(batch, key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Precomputed Qwen conditioning requires tensor batch field {key!r}.")
        return value

    def encode_precomputed(
        self,
        batch: TensorDict,
        *,
        require_negative: bool,
    ) -> tuple[ConditionBundle, Optional[ConditionBundle]]:
        """Validate and truncate cached positive and negative conditioning."""
        prompt_embeds = self.require_tensor(batch, "prompt_embeds")[:, : self.max_sequence_length]
        prompt_mask = tu.get(batch, "prompt_embeds_mask")
        if isinstance(prompt_mask, torch.Tensor):
            prompt_mask = prompt_mask[:, : self.max_sequence_length]
        negative_embeds = (
            self.require_tensor(batch, "negative_prompt_embeds")[:, : self.max_sequence_length]
            if require_negative
            else None
        )
        negative_mask = tu.get(batch, "negative_prompt_embeds_mask") if require_negative else None
        if isinstance(negative_mask, torch.Tensor):
            negative_mask = negative_mask[:, : self.max_sequence_length]
        if prompt_mask is not None and not isinstance(prompt_mask, torch.Tensor):
            raise TypeError("prompt_embeds_mask must be a tensor when supplied.")
        if negative_mask is not None and not isinstance(negative_mask, torch.Tensor):
            raise TypeError("negative_prompt_embeds_mask must be a tensor when supplied.")
        positive = self.make_condition(prompt_embeds, prompt_mask)
        negative = self.make_condition(negative_embeds, negative_mask) if negative_embeds is not None else None
        if positive.tensors["prompt_embeds"].shape[0] != batch.batch_size[0]:
            raise ValueError("Precomputed Qwen conditioning batch size does not match the phase batch.")
        if negative is not None and negative.tensors["prompt_embeds"].shape[0] != batch.batch_size[0]:
            raise ValueError("Precomputed Qwen negative conditioning batch size does not match the phase batch.")
        return positive, negative

    def ensure_pipeline(self, device: torch.device, dtype: torch.dtype):
        """Load the frozen checkpoint text encoder once on the execution device."""
        if self.pipeline is not None:
            return self.pipeline

        from diffusers import QwenImagePipeline

        pipeline = QwenImagePipeline.from_pretrained(
            self.model_path,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
            local_files_only=os.path.isdir(self.model_path),
        ).to(device)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder.eval()
        self.pipeline = pipeline
        return pipeline

    @staticmethod
    def prompt_rows(value: Any, batch_size: int, key: str) -> list[Any]:
        """Unwrap one text or chat-message row per phase sample."""
        if hasattr(value, "tolist") and not isinstance(value, torch.Tensor):
            value = value.tolist()
        if batch_size == 1 and (
            isinstance(value, str) or (isinstance(value, list) and value and isinstance(value[0], dict))
        ):
            return [value]
        if not isinstance(value, list) or len(value) != batch_size:
            raise ValueError(f"{key} must contain exactly {batch_size} prompt row(s).")
        return value

    def tokenize_rows(self, pipeline, rows: list[Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the fixed Qwen template before tokenization and prefix removal."""
        rendered = []
        for row in rows:
            if isinstance(row, list):
                if len(row) != 1 or not isinstance(row[0], dict) or row[0].get("role") != "user":
                    raise ValueError(
                        "Qwen DMD raw prompts require a single user message; use precomputed conditioning otherwise."
                    )
                row = row[0].get("content")
            if not isinstance(row, str):
                raise TypeError("Qwen DMD prompts must be strings or single text-only user messages.")
            # The encoder removes this template's fixed prefix, not a generic chat prefix.
            rendered.append(pipeline.prompt_template_encode.format(row))
        tokens = pipeline.tokenizer(
            rendered,
            max_length=self.max_sequence_length + pipeline.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def encode_ids(
        self,
        pipeline,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> ConditionBundle:
        """Reuse Qwen token-ID encoding under no-grad and truncate the result."""
        with torch.no_grad():
            prompt_embeds, prompt_mask = QwenImageTokenIdPromptMixin._get_qwen_prompt_embeds(
                pipeline, prompt_ids, attention_mask=attention_mask
            )
        prompt_embeds = prompt_embeds[:, : self.max_sequence_length]
        if prompt_mask is not None:
            prompt_mask = prompt_mask[:, : self.max_sequence_length]
        if prompt_embeds.shape[1] == 0:
            raise ValueError("Qwen prompt encoding produced no tokens after removing the template prefix.")
        return self.make_condition(prompt_embeds.detach(), prompt_mask.detach() if prompt_mask is not None else None)

    @torch.profiler.record_function("distillation/condition_encode")
    def encode(
        self,
        batch: TensorDict,
        *,
        device: torch.device,
        dtype: torch.dtype,
        require_negative: bool,
    ) -> tuple[ConditionBundle, Optional[ConditionBundle]]:
        """Encode the positive prompt and optional teacher negative condition."""
        if self.provider == "precomputed":
            return self.encode_precomputed(batch, require_negative=require_negative)
        if self.provider != "local_frozen_encoder":
            raise ValueError(f"Unsupported Qwen conditioning provider {self.provider!r}.")

        pipeline = self.ensure_pipeline(device, dtype)
        prompt_ids = tu.get(batch, "prompt_ids", tu.get(batch, "prompts"))
        prompt_mask = tu.get(
            batch,
            "prompt_attention_mask",
            tu.get(batch, "prompt_mask", tu.get(batch, "attention_mask")),
        )
        if prompt_ids is None:
            raw_prompt = tu.get(batch, "raw_prompt")
            if raw_prompt is None:
                raise ValueError("Qwen DMD batches require prompt_ids, prompts, prompt_embeds, or raw_prompt.")
            rows = self.prompt_rows(raw_prompt, batch.batch_size[0], "raw_prompt")
            prompt_ids, prompt_mask = self.tokenize_rows(pipeline, rows, device)
        else:
            if prompt_mask is None:
                raise ValueError("Pre-tokenized Qwen prompts require prompt_attention_mask or attention_mask.")
            if not isinstance(prompt_ids, torch.Tensor):
                prompt_ids = torch.as_tensor(prompt_ids, device=device, dtype=torch.long)
            else:
                prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
        if prompt_mask is not None:
            prompt_mask = torch.as_tensor(prompt_mask, device=device, dtype=torch.long)
        positive = self.encode_ids(pipeline, prompt_ids, prompt_mask)

        if not require_negative:
            return positive, None

        negative_ids = tu.get(batch, "negative_prompt_ids")
        negative_mask = tu.get(batch, "negative_prompt_attention_mask", tu.get(batch, "negative_prompt_mask"))
        if negative_ids is None:
            raw_negative = tu.get(batch, "raw_negative_prompt", tu.get(batch, "negative_prompt"))
            if raw_negative is not None:
                negative_rows = self.prompt_rows(raw_negative, batch.batch_size[0], "negative_prompt")
                negative_ids, negative_mask = self.tokenize_rows(pipeline, negative_rows, device)
                negative = self.encode_ids(pipeline, negative_ids, negative_mask)
            else:
                if self._negative_condition is None:
                    negative_ids, negative_mask = self.tokenize_rows(pipeline, [self.negative_prompt], device)
                    self._negative_condition = self.encode_ids(pipeline, negative_ids, negative_mask)
                negative = self.make_condition(
                    self._negative_condition.tensors["prompt_embeds"].expand(batch.batch_size[0], -1, -1),
                    self._negative_condition.masks["prompt_embeds"].expand(batch.batch_size[0], -1),
                )
        else:
            if negative_mask is None:
                raise ValueError("Pre-tokenized negative Qwen prompts require negative_prompt_attention_mask.")
            negative_ids = torch.as_tensor(negative_ids, device=device, dtype=torch.long)
            negative_mask = torch.as_tensor(negative_mask, device=device, dtype=torch.long)
            negative = self.encode_ids(pipeline, negative_ids, negative_mask)
        return positive, negative


class QwenImageDMDComputer:
    """Differentiable Qwen-Image computation for original DMD and both DMD2 profiles."""

    STREAM_OFFSETS = {
        "initial_noise": 0,
        "rollout_decision": 1,
        "rollout_transition": 2,
        "score_sigma": 3,
        "score_noise": 4,
        "adversarial_sigma": 5,
        "adversarial_noise": 6,
        "adversarial_vae": 7,
    }

    def __init__(self, model_config: DiffusionModelConfig, plan: DistillationPlan) -> None:
        if plan.name not in {"dmd", "dmd2"}:
            raise ValueError(f"QwenImageDMDComputer does not implement recipe {plan.name!r}.")
        self.adversarial = bool(plan.objective.get("adversarial", False))
        self.adversarial_config = DiffusionAdversarialConfig(**dict(plan.objective.get("adversarial_config", {})))
        self.model_config = model_config
        self.plan = plan
        self.height = int(model_config.pipeline.height)
        self.width = int(model_config.pipeline.width)
        self.num_inference_steps = int(model_config.pipeline.num_inference_steps)
        self.strategy = str(plan.rollout["strategy"])
        self.rollout_timestep_shift = float(plan.rollout["rollout_timestep_shift"])
        self.score_sigma_min = float(plan.rollout["score_sigma_min"])
        self.score_sigma_max = float(plan.rollout["score_sigma_max"])
        self.score_timestep_shift = float(plan.rollout["score_timestep_shift"])
        self.score_discrete_steps = int(plan.rollout["score_discrete_steps"])
        self.rng_seed = int(plan.rollout["rng_seed"])
        self.guidance_scale = float(plan.objective["teacher_guidance_scale"])
        self.cfg_norm = str(plan.objective["teacher_cfg_norm"])
        self.normalization_epsilon = float(plan.objective["normalization_epsilon"])
        self.dmd_loss_weight = float(plan.objective["dmd_loss_weight"])
        self.regression_type = str(plan.objective["regression_type"])
        self.regression_loss_weight = float(plan.objective["regression_loss_weight"])
        rollout_config = getattr(model_config, "algo", None)
        if hasattr(rollout_config, "get"):
            inference_shift = rollout_config.get("rollout_timestep_shift")
        else:
            inference_shift = getattr(rollout_config, "rollout_timestep_shift", None)
        self.inference_rollout_timestep_shift = 3.0 if inference_shift is None else float(inference_shift)
        self.condition_provider = QwenImageConditionProvider(
            model_config.local_path or model_config.path,
            str(plan.data_requirements["conditioning_provider"]),
            int(model_config.pipeline.max_sequence_length),
            str(plan.data_requirements["negative_prompt"]),
        )
        self._generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, torch.Tensor] = {}
        self._vae = None
        self._lpips = None
        self.vae_fingerprint = None
        self.validate_config()

    def validate_config(self) -> None:
        """Reject unsupported Qwen sampling, geometry, and objective settings."""
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Qwen DMD height and width must be positive.")
        divisor = QWEN_IMAGE_VAE_SCALE_FACTOR * 2
        if self.height % divisor or self.width % divisor:
            raise ValueError(f"Qwen DMD height and width must be divisible by {divisor}.")
        if self.num_inference_steps <= 0:
            raise ValueError("Qwen DMD num_inference_steps must be positive.")
        if self.rollout_timestep_shift < 1:
            raise ValueError("Qwen DMD rollout_timestep_shift must be at least 1.")
        if self.inference_rollout_timestep_shift != self.rollout_timestep_shift:
            raise ValueError(
                "Qwen DMD training and inference rollout_timestep_shift must match; "
                f"got {self.rollout_timestep_shift} and {self.inference_rollout_timestep_shift}."
            )
        if self.strategy not in {"one_step", "ode_euler", "consistency_renoise", "backward_simulated"}:
            raise ValueError(f"Unsupported Qwen DMD rollout strategy {self.strategy!r}.")
        if not 0 <= self.score_sigma_min < self.score_sigma_max <= 1:
            raise ValueError("Qwen DMD score sigma bounds must satisfy 0 <= min < max <= 1.")
        if self.score_timestep_shift < 1:
            raise ValueError("Qwen DMD score_timestep_shift must be at least 1.")
        if self.score_discrete_steps < 0:
            raise ValueError("Qwen DMD score_discrete_steps must be non-negative.")
        if self.guidance_scale <= 1:
            raise ValueError("Qwen DMD teacher guidance_scale must be greater than 1.")
        if self.cfg_norm not in {"none", "layer_norm", "scalar"}:
            raise ValueError(f"Unsupported Qwen DMD CFG normalization {self.cfg_norm!r}.")
        if self.normalization_epsilon <= 0:
            raise ValueError("Qwen DMD normalization_epsilon must be positive.")
        if self.dmd_loss_weight < 0 or self.regression_loss_weight < 0:
            raise ValueError("Qwen DMD objective weights must be non-negative.")
        if self.plan.name == "dmd2" and self.dmd_loss_weight == 0:
            raise ValueError("Qwen DMD2 requires dmd_loss_weight > 0.")
        if self.plan.name == "dmd" and self.dmd_loss_weight == 0 and self.regression_loss_weight == 0:
            raise ValueError("Qwen DMD requires at least one positive objective weight.")
        if self.regression_type not in {"decoded_lpips", "latent_mse"}:
            raise ValueError(f"Unsupported Qwen DMD regression_type {self.regression_type!r}.")
        if self.condition_provider.provider not in {"local_frozen_encoder", "precomputed"}:
            raise ValueError(f"Unsupported Qwen conditioning provider {self.condition_provider.provider!r}.")
        if self.plan.name == "dmd" and self.strategy != "one_step":
            raise ValueError("The Qwen original-DMD profile requires rollout_strategy='one_step'.")
        targets = getattr(self.model_config, "target_modules", None)
        if self.adversarial and (
            targets == "all-linear"
            or targets is not None
            and not isinstance(targets, str)
            and any("proj_out" in target for target in targets)
        ):
            raise ValueError("Qwen DMD2 adversarial LoRA must exclude proj_out/classifier; select attention targets.")

    @staticmethod
    def module_dtype(module: torch.nn.Module) -> torch.dtype:
        """Read the model parameter dtype for frozen conditioning."""
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.float32

    @staticmethod
    def module_config(module: torch.nn.Module):
        """Read the underlying transformer configuration through a wrapper."""
        return getattr(getattr(module, "module", module), "config", None)

    def generator_for_stream(
        self, name: str, device: torch.device, runtime: DistillationRoleRuntime
    ) -> torch.Generator:
        """Resolve a checkpointable RNG stream, shared by sequence-parallel peers."""
        if name in self._generators:
            return self._generators[name]
        if name not in self.STREAM_OFFSETS:
            raise KeyError(f"Unknown Qwen DMD RNG stream {name!r}.")
        engine = runtime.engine_for_role("student")
        rank = int(engine.get_data_parallel_rank()) if hasattr(engine, "get_data_parallel_rank") else 0
        generator = torch.Generator(device=device)
        offset = self.STREAM_OFFSETS[name]
        # Preserve pre-adversarial checkpoint replay for the five original streams.
        seed_offset = rank * 5 + offset if offset < 5 else 2**32 + rank * 3 + offset - 5
        generator.manual_seed(self.rng_seed + seed_offset)
        pending = self._pending_generator_states.pop(name, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[name] = generator
        return generator

    def sample_noise(self, shape: tuple[int, ...], device: torch.device, runtime: DistillationRoleRuntime, stream: str):
        """Draw fp32 noise from the named independent RNG stream."""
        return torch.randn(
            shape, device=device, dtype=torch.float32, generator=self.generator_for_stream(stream, device, runtime)
        )

    def sample_rollout_exit(self, high: int, device: torch.device, runtime: DistillationRoleRuntime) -> int:
        """Broadcast one exit step so FSDP and sequence-parallel control flow agree."""
        value = torch.randint(
            high,
            (1,),
            device=device,
            generator=self.generator_for_stream("rollout_decision", device, runtime),
        )
        if torch.distributed.is_initialized():
            # FSDP shards and SP peers must enter identical forward/backward collectives.
            torch.distributed.broadcast(value, src=0)
        return int(value.item())

    @staticmethod
    def batch_int(batch: TensorDict, key: str, default: int) -> int:
        """Require homogeneous integer geometry metadata within a micro-batch."""
        value = tu.get(batch, key, default)
        if isinstance(value, torch.Tensor):
            values = value.detach().reshape(-1)
            if values.numel() == 0 or not torch.all(values == values[0]):
                raise ValueError(f"Qwen DMD requires one homogeneous {key} per micro-batch.")
            return int(values[0].item())
        if isinstance(value, list | tuple):
            values = [int(item) for item in value]
            if not values or any(item != values[0] for item in values):
                raise ValueError(f"Qwen DMD requires one homogeneous {key} per micro-batch.")
            return values[0]
        return int(value)

    def latent_geometry(self, batch: TensorDict, module: torch.nn.Module) -> tuple[int, int, int, tuple[int, ...]]:
        """Resolve the declared Qwen packed-token and VAE latent dimensions."""
        if len(batch.batch_size) != 1 or batch.batch_size[0] <= 0:
            raise ValueError("Qwen DMD requires a nonempty leading batch dimension.")
        height = self.batch_int(batch, "height", self.height)
        width = self.batch_int(batch, "width", self.width)
        divisor = QWEN_IMAGE_VAE_SCALE_FACTOR * 2
        if height <= 0 or width <= 0 or height % divisor or width % divisor:
            raise ValueError(
                f"Qwen DMD height and width must be positive multiples of {divisor}, got {height}x{width}."
            )
        module_config = self.module_config(module)
        in_channels = getattr(module_config, "in_channels", None)
        if in_channels is None:
            in_channels = (self.model_config.transformer_config or {}).get("in_channels")
        if not isinstance(in_channels, int) or in_channels <= 0 or in_channels % 4:
            raise ValueError(f"Qwen transformer in_channels must be a positive multiple of four, got {in_channels!r}.")
        latent_channels = in_channels // 4
        latent_height = height // QWEN_IMAGE_VAE_SCALE_FACTOR
        latent_width = width // QWEN_IMAGE_VAE_SCALE_FACTOR
        return height, width, in_channels, (batch.batch_size[0], latent_channels, 1, latent_height, latent_width)

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack normalized [B, C, 1, H, W] latents with the Diffusers Qwen helper."""
        from diffusers import QwenImagePipeline

        batch, channels, _, height, width = latents.shape
        return QwenImagePipeline._pack_latents(latents, batch, channels, height, width)

    @staticmethod
    def expand_sigma(sigma: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar or per-sample sigma over non-batch latent dimensions."""
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if sigma.shape[0] == 1 and tensor.shape[0] != 1:
            sigma = sigma.expand(tensor.shape[0])
        if sigma.shape[0] != tensor.shape[0]:
            raise ValueError(f"Sigma batch {sigma.shape[0]} does not match latent batch {tensor.shape[0]}.")
        return sigma.reshape(sigma.shape[0], *((1,) * (tensor.ndim - 1)))

    def predict_velocity(
        self,
        runtime: DistillationRoleRuntime,
        role: str,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        condition: ConditionBundle,
        *,
        height: int,
        width: int,
        grad_enabled: bool,
        input_grad_only: bool = False,
    ) -> torch.Tensor:
        """Predict Qwen velocity or discriminator logits with explicit gradient ownership."""
        with (
            torch.profiler.record_function(f"distillation/{role}_forward"),
            runtime.use_role(role, grad_enabled=grad_enabled, input_grad_only=input_grad_only) as module,
        ):
            module.eval()
            timestep = sigma.reshape(-1)
            if timestep.shape[0] == 1 and latents.shape[0] != 1:
                timestep = timestep.expand(latents.shape[0])
            module_config = self.module_config(module)
            guidance = None
            if getattr(module_config, "guidance_embeds", False):
                configured = self.model_config.pipeline.guidance_scale
                if configured is None:
                    raise ValueError("Qwen guidance-embedded transformers require model.pipeline.guidance_scale.")
                guidance = torch.full(
                    (latents.shape[0],), float(configured), device=latents.device, dtype=torch.float32
                )
            output = module(
                hidden_states=latents,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states_mask=condition.masks.get("prompt_embeds"),
                encoder_hidden_states=condition.tensors["prompt_embeds"],
                img_shapes=build_img_shapes(height, width, latents.shape[0], QWEN_IMAGE_VAE_SCALE_FACTOR),
                return_dict=False,
            )[0]
        if role == "discriminator":
            if output.shape != (latents.shape[0], 1):
                raise ValueError(f"Qwen discriminator must return [B, 1] logits, got {tuple(output.shape)}.")
            return output.float()
        if output.shape != latents.shape:
            raise ValueError(
                f"Qwen DMD transformer output shape {tuple(output.shape)} does not match latent shape "
                f"{tuple(latents.shape)}."
            )
        return output

    def rollout_sigmas(self, scheduler, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return the inference-matched fixed-shift student sigma schedule."""
        del scheduler, height, width
        steps = 1 if self.strategy == "one_step" else self.num_inference_steps
        return build_qwen_dmd_sigmas(steps, self.rollout_timestep_shift, device=device)

    @torch.profiler.record_function("distillation/student_rollout")
    def rollout(
        self,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        initial_noise: torch.Tensor,
        *,
        height: int,
        width: int,
        grad_enabled: bool,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Run to the shared exit step, retaining only its student graph."""
        scheduler = runtime.scheduler_for_role("student")
        sigmas = self.rollout_sigmas(scheduler, height, width, initial_noise.device)
        exit_index = (
            0
            if self.strategy == "one_step"
            else self.sample_rollout_exit(sigmas.numel() - 1, initial_noise.device, runtime)
        )
        sample = initial_noise
        x0 = None
        for index in range(exit_index + 1):
            sigma = sigmas[index].reshape(1)
            use_grad = grad_enabled and index == exit_index
            velocity = self.predict_velocity(
                runtime,
                "student",
                sample,
                sigma,
                condition,
                height=height,
                width=width,
                grad_enabled=use_grad,
            )
            expanded_sigma = self.expand_sigma(sigma, sample)
            x0 = velocity_to_x0(sample, velocity, expanded_sigma)
            if index == exit_index:
                break
            sigma_next = sigmas[index + 1].reshape(1)
            expanded_next = self.expand_sigma(sigma_next, sample)
            if self.strategy == "consistency_renoise":
                transition_noise = self.sample_noise(tuple(sample.shape), sample.device, runtime, "rollout_transition")
                sample = consistency_renoise_step(x0, transition_noise, expanded_next)
            else:
                sample = ode_euler_step(sample, velocity, expanded_sigma, expanded_next)
        assert x0 is not None
        return x0, exit_index, sigmas[exit_index], sigmas[exit_index + 1]

    def sample_score_sigma(self, generated: torch.Tensor, runtime: DistillationRoleRuntime) -> torch.Tensor:
        """Sample discrete shifted timesteps or continuous unshifted sigma values."""
        generator = self.generator_for_stream("score_sigma", generated.device, runtime)
        if self.score_discrete_steps > 0:
            scheduler = runtime.scheduler_for_role("student")
            scheduler_config = getattr(scheduler, "config", {})
            num_train_timesteps = int(scheduler_config.get("num_train_timesteps", 1000))
            if self.score_discrete_steps != num_train_timesteps:
                raise ValueError(
                    "score_discrete_steps must equal the Qwen scheduler num_train_timesteps; "
                    f"got {self.score_discrete_steps} and {num_train_timesteps}."
                )
            timestep = torch.randint(
                0,
                num_train_timesteps,
                (generated.shape[0],),
                device=generated.device,
                generator=generator,
            ).float()
            sigma = timestep_shift(timestep, num_train_timesteps, self.score_timestep_shift)
            sigma = sigma / num_train_timesteps
        else:
            sigma = torch.rand(
                (generated.shape[0],),
                device=generated.device,
                dtype=torch.float32,
                generator=generator,
            )
            sigma = self.score_sigma_min + (self.score_sigma_max - self.score_sigma_min) * sigma
        return sigma.clamp(self.score_sigma_min, self.score_sigma_max)

    def score_batch(
        self,
        generated: torch.Tensor,
        runtime: DistillationRoleRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-noise detached generated latents using independent score RNG streams."""
        sigma = self.sample_score_sigma(generated, runtime)
        noise = self.sample_noise(tuple(generated.shape), generated.device, runtime, "score_noise")
        expanded = self.expand_sigma(sigma, generated)
        noisy = (1.0 - expanded) * generated.detach().float() + expanded * noise
        return noisy, noise, sigma

    @staticmethod
    def prepare_condition_for_sequence_parallel(
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
    ) -> ConditionBundle:
        """Reuse engine padding for sequence-parallel prompt embeddings."""
        engine = runtime.engine_for_role("student")
        if not getattr(engine, "use_ulysses_sp", False):
            return condition
        embeds, mask = engine._pad_embeds_for_sp(
            condition.tensors["prompt_embeds"],
            condition.masks.get("prompt_embeds"),
            engine.ulysses_sequence_parallel_size,
        )
        return QwenImageConditionProvider.make_condition(embeds, mask)

    def student_loss(
        self,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        negative_condition: ConditionBundle,
        *,
        height: int,
        width: int,
        latent_shape: tuple[int, ...],
    ):
        """Build DMD and optional paired-regression losses for the student only."""
        initial_noise = self.sample_noise(
            latent_shape, condition.tensors["prompt_embeds"].device, runtime, "initial_noise"
        )
        packed_noise = self.pack_latents(initial_noise)
        rollout_start = time.perf_counter()
        generated, exit_index, sigma_from, sigma_to = self.rollout(
            runtime,
            condition,
            packed_noise,
            height=height,
            width=width,
            grad_enabled=True,
        )
        rollout_duration = time.perf_counter() - rollout_start
        noisy, _, score_sigma = self.score_batch(generated, runtime)
        with torch.no_grad():
            fake_start = time.perf_counter()
            fake_velocity = self.predict_velocity(
                runtime,
                "fake_score",
                noisy,
                score_sigma,
                condition,
                height=height,
                width=width,
                grad_enabled=False,
            )
            fake_duration = time.perf_counter() - fake_start
            teacher_start = time.perf_counter()
            teacher_positive = self.predict_velocity(
                runtime,
                "teacher_score",
                noisy,
                score_sigma,
                condition,
                height=height,
                width=width,
                grad_enabled=False,
            )
            teacher_negative = self.predict_velocity(
                runtime,
                "teacher_score",
                noisy,
                score_sigma,
                negative_condition,
                height=height,
                width=width,
                grad_enabled=False,
            )
            teacher_velocity = standard_cfg(
                teacher_positive,
                teacher_negative,
                self.guidance_scale,
                self.cfg_norm,
            )
            expanded_sigma = self.expand_sigma(score_sigma, noisy)
            fake_x0 = velocity_to_x0(noisy, fake_velocity, expanded_sigma)
            teacher_x0 = velocity_to_x0(noisy, teacher_velocity, expanded_sigma)
            gradient, normalizer, nonfinite = dmd_gradient(
                fake_x0,
                teacher_x0,
                generated,
                normalization_epsilon=self.normalization_epsilon,
            )
            teacher_duration = time.perf_counter() - teacher_start
        dmd_loss, active = dmd_surrogate_loss(generated, gradient)
        total = self.dmd_loss_weight * dmd_loss
        metrics = {
            "dmd/loss": float(dmd_loss.detach()),
            "dmd/normalizer": float(normalizer.detach().mean()),
            "dmd/nonfinite": float(nonfinite),
            "dmd/active_elements": float(active),
            "rollout/exit_index": float(exit_index),
            "rollout/sigma_from": float(sigma_from),
            "rollout/sigma_to": float(sigma_to),
            "score/sigma": float(score_sigma.detach().mean()),
            "perf/student_rollout_s": rollout_duration,
            "perf/fake_score_model_s": fake_duration,
            "perf/teacher_score_model_s": teacher_duration,
        }
        if self.plan.name == "dmd":
            regression_start = time.perf_counter()
            regression_loss = self.regression_loss(batch, runtime, condition, height=height, width=width)
            total = total + self.regression_loss_weight * regression_loss
            metrics["regression/loss"] = float(regression_loss.detach())
            metrics["perf/regression_s"] = time.perf_counter() - regression_start
        if self.adversarial:
            adversarial_start = time.perf_counter()
            logits = self.discriminator_logits(
                generated, runtime, condition, height=height, width=width, input_grad_only=True
            )
            generator_loss = adversarial_generator_loss(logits)
            total = total + self.adversarial_config.generator_weight * generator_loss
            metrics["adversarial/generator_loss"] = float(generator_loss.detach())
            metrics["perf/generator_adversarial_s"] = time.perf_counter() - adversarial_start
        return total, metrics

    def fake_loss(
        self,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        *,
        height: int,
        width: int,
        latent_shape: tuple[int, ...],
    ):
        """Train the fake score to denoise detached student samples."""
        initial_noise = self.sample_noise(
            latent_shape, condition.tensors["prompt_embeds"].device, runtime, "initial_noise"
        )
        packed_noise = self.pack_latents(initial_noise)
        rollout_start = time.perf_counter()
        with torch.no_grad():
            generated, exit_index, sigma_from, sigma_to = self.rollout(
                runtime,
                condition,
                packed_noise,
                height=height,
                width=width,
                grad_enabled=False,
            )
            generated = generated.detach()
        rollout_duration = time.perf_counter() - rollout_start
        noisy, score_noise, score_sigma = self.score_batch(generated, runtime)
        fake_start = time.perf_counter()
        fake_velocity = self.predict_velocity(
            runtime,
            "fake_score",
            noisy,
            score_sigma,
            condition,
            height=height,
            width=width,
            grad_enabled=True,
        )
        fake_duration = time.perf_counter() - fake_start
        loss, active = fake_score_loss(fake_velocity, score_noise, generated)
        metrics = {
            "fake_score/denoising_loss": float(loss.detach()),
            "fake_score/active_elements": float(active),
            "rollout/exit_index": float(exit_index),
            "rollout/sigma_from": float(sigma_from),
            "rollout/sigma_to": float(sigma_to),
            "score/sigma": float(score_sigma.detach().mean()),
            "perf/student_rollout_s": rollout_duration,
            "perf/fake_score_model_s": fake_duration,
        }
        return loss, metrics, generated

    def discriminator_loss(
        self,
        batch: TensorDict,
        generated: torch.Tensor,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Classify matched real and detached generated samples for one optimizer."""
        discriminator_start = time.perf_counter()
        real = self.real_latents(batch, generated, runtime, height=height, width=width)
        fake_logits = self.discriminator_logits(generated.detach(), runtime, condition, height=height, width=width)
        real_logits = self.discriminator_logits(real.detach(), runtime, condition, height=height, width=width)
        loss = adversarial_discriminator_loss(fake_logits, real_logits)
        return self.adversarial_config.discriminator_weight * loss, {
            "adversarial/discriminator_loss": float(loss.detach()),
            "adversarial/real_logit": float(real_logits.detach().mean()),
            "adversarial/fake_logit": float(fake_logits.detach().mean()),
            "perf/discriminator_s": time.perf_counter() - discriminator_start,
        }

    def discriminator_logits(
        self,
        latents: torch.Tensor,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        *,
        height: int,
        width: int,
        input_grad_only: bool = False,
    ) -> torch.Tensor:
        """Classify clean or re-noised latents; generator inputs retain their graph."""
        sigma = latents.new_zeros(latents.shape[0], dtype=torch.float32)
        if self.adversarial_config.mode == "diffusion_gan":
            steps = int(runtime.scheduler_for_role("student").config.get("num_train_timesteps", 1000))
            if self.adversarial_config.max_timestep > steps:
                raise ValueError("adversarial.max_timestep exceeds the scheduler training horizon.")
            sigma = (
                torch.randint(
                    self.adversarial_config.max_timestep,
                    (latents.shape[0],),
                    device=latents.device,
                    generator=self.generator_for_stream("adversarial_sigma", latents.device, runtime),
                ).float()
                / steps
            )
            noise = self.sample_noise(tuple(latents.shape), latents.device, runtime, "adversarial_noise")
            expanded = self.expand_sigma(sigma, latents)
            latents = (1 - expanded) * latents.float() + expanded * noise
        return self.predict_velocity(
            runtime,
            "discriminator",
            latents,
            sigma,
            condition,
            height=height,
            width=width,
            grad_enabled=True,
            input_grad_only=input_grad_only,
        )

    def real_latents(
        self,
        batch: TensorDict,
        generated: torch.Tensor,
        runtime: DistillationRoleRuntime,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Load normalized real latents or encode [0, 1] RGB with the frozen Qwen VAE."""
        latents = tu.get(batch, "real_latents")
        pixels = tu.get(batch, "real_pixels")
        if (latents is None) == (pixels is None):
            raise ValueError("DMD2 adversarial batches require exactly one of real_latents or real_pixels.")
        if latents is not None:
            manifests = tu.get(batch, "real_latent_manifest")
            if isinstance(manifests, Mapping) and batch.batch_size[0] == 1:
                manifests = [manifests]
            if self.vae_fingerprint is None:
                self.vae_fingerprint = qwen_vae_config_sha256(self.model_config.local_path or self.model_config.path)
            if (
                not isinstance(manifests, list)
                or len(manifests) != batch.batch_size[0]
                or any(
                    not isinstance(item, Mapping)
                    or item.get("normalization") != "qwen_image"
                    or item.get("vae_config_sha256") != self.vae_fingerprint
                    for item in manifests
                )
            ):
                raise ValueError("real_latent_manifest must match Qwen normalization and the VAE config for every row.")
            return self.coerce_packed(latents, generated.shape, "real_latents", generated.device).detach()
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (generated.shape[0], 3, height, width):
            raise ValueError("real_pixels must have shape [B, 3, height, width].")
        pixels = pixels.detach().to(device=generated.device, dtype=torch.float32)
        if not torch.isfinite(pixels).all() or torch.any((pixels < 0) | (pixels > 1)):
            raise ValueError("real_pixels must contain finite RGB values in [0, 1].")
        vae = self.ensure_vae(generated.device)
        with torch.no_grad():
            encoded = (
                vae.encode(pixels.mul(2).sub(1).unsqueeze(2))
                .latent_dist.sample(generator=self.generator_for_stream("adversarial_vae", generated.device, runtime))
                .float()
            )
            shape = (1, vae.config.z_dim, 1, 1, 1)
            mean = encoded.new_tensor(vae.config.latents_mean).view(shape)
            std = encoded.new_tensor(vae.config.latents_std).view(shape)
            return self.coerce_packed((encoded - mean) / std, generated.shape, "real_latents", generated.device)

    def coerce_packed(self, value: Any, expected_shape: torch.Size, key: str, device: torch.device) -> torch.Tensor:
        """Validate and pack a reference noise or teacher-target tensor."""
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Qwen DMD regression requires tensor batch field {key!r}.")
        value = value.to(device=device, dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"Qwen DMD {key} must contain finite values.")
        if value.ndim == 4:
            value = value.unsqueeze(2)
        if value.ndim == 5:
            value = self.pack_latents(value)
        if value.shape != expected_shape:
            raise ValueError(
                f"Qwen DMD {key} shape {tuple(value.shape)} does not match generated latent shape "
                f"{tuple(expected_shape)}."
            )
        return value

    @torch.profiler.record_function("distillation/regression")
    def regression_loss(
        self,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
        condition: ConditionBundle,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Regress a paired student sample against its teacher target."""
        manifest = tu.get(batch, "teacher_sampling_manifest")
        manifests = [manifest] if isinstance(manifest, Mapping) and batch.batch_size[0] == 1 else manifest
        if (
            not isinstance(manifests, list)
            or len(manifests) != batch.batch_size[0]
            or any(not isinstance(item, Mapping) or not item for item in manifests)
        ):
            raise ValueError("Original DMD regression requires teacher_sampling_manifest provenance for every sample.")
        reference_noise = tu.get(batch, "reference_noise")
        target_latents = tu.get(batch, "teacher_target_latents")
        target_pixels = tu.get(batch, "teacher_target_pixels")
        if target_latents is None and target_pixels is None:
            raise ValueError("Original DMD requires teacher_target_latents or teacher_target_pixels.")
        if target_latents is not None and target_pixels is not None:
            raise ValueError("Provide only one of teacher_target_latents and teacher_target_pixels.")

        with runtime.use_role("student", grad_enabled=True) as module:
            _, _, in_channels, latent_shape = self.latent_geometry(batch, module)
        expected_packed = torch.Size((latent_shape[0], (latent_shape[-2] // 2) * (latent_shape[-1] // 2), in_channels))
        reference_noise = self.coerce_packed(
            reference_noise,
            expected_packed,
            "reference_noise",
            condition.tensors["prompt_embeds"].device,
        )
        prediction, _, _, _ = self.rollout(
            runtime,
            condition,
            reference_noise,
            height=height,
            width=width,
            grad_enabled=True,
        )
        if self.regression_type == "latent_mse":
            if target_pixels is not None:
                raise ValueError("regression_type='latent_mse' requires teacher_target_latents.")
            target = self.coerce_packed(target_latents, prediction.shape, "teacher_target_latents", prediction.device)
            return torch.mean((prediction.float() - target.detach().float()) ** 2)
        return self.decoded_lpips_loss(prediction, target_latents, target_pixels, height=height, width=width)

    def ensure_vae(self, device: torch.device):
        """Load the frozen VAE independently of optional perceptual-loss dependencies."""
        if self._vae is None:
            from diffusers import AutoencoderKLQwenImage

            self._vae = AutoencoderKLQwenImage.from_pretrained(
                self.model_config.local_path or self.model_config.path,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=os.path.isdir(self.model_config.local_path or self.model_config.path),
            ).to(device)
            self._vae.requires_grad_(False)
            self._vae.eval()
        return self._vae

    def ensure_vae_and_lpips(self, device: torch.device):
        """Load frozen VAE and perceptual-loss modules for original DMD."""
        self.ensure_vae(device)
        if self._lpips is None:
            try:
                import piq
            except ImportError as exc:
                raise ImportError(
                    "regression_type='decoded_lpips' requires piq; install verl-omni with the distillation extra."
                ) from exc
            self._lpips = piq.LPIPS(replace_pooling=True, reduction="none").to(device)
            self._lpips.requires_grad_(False)
            self._lpips.eval()
        return self._vae, self._lpips

    def decode_latents(self, packed: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Undo Qwen latent normalization and decode differentiably to RGB."""
        from diffusers import QwenImagePipeline

        vae = self.ensure_vae(packed.device)
        latent = QwenImagePipeline._unpack_latents(
            packed,
            height=height,
            width=width,
            vae_scale_factor=QWEN_IMAGE_VAE_SCALE_FACTOR,
        ).float()
        shape = (1, vae.config.z_dim, 1, 1, 1)
        mean = latent.new_tensor(vae.config.latents_mean).view(shape)
        std = latent.new_tensor(vae.config.latents_std).view(shape)
        decoded = vae.decode(latent * std + mean, return_dict=False)[0][:, :, 0]
        return decoded.float().mul(0.5).add(0.5)

    def decoded_lpips_loss(
        self,
        prediction: torch.Tensor,
        target_latents: Optional[torch.Tensor],
        target_pixels: Optional[torch.Tensor],
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Apply perceptual regression with gradients only through the prediction."""
        vae, lpips = self.ensure_vae_and_lpips(prediction.device)
        self._vae = vae
        prediction_pixels = self.decode_latents(prediction, height=height, width=width)
        if target_pixels is not None:
            target = target_pixels.to(device=prediction.device, dtype=torch.float32)
            if target.ndim == 3:
                target = target.unsqueeze(0)
            if target.shape != prediction_pixels.shape:
                raise ValueError(
                    f"teacher_target_pixels shape {tuple(target.shape)} does not match decoded prediction "
                    f"{tuple(prediction_pixels.shape)}."
                )
            if torch.any((target < 0) | (target > 1)):
                raise ValueError("teacher_target_pixels must be normalized to [0, 1].")
        else:
            target = self.coerce_packed(
                target_latents,
                prediction.shape,
                "teacher_target_latents",
                prediction.device,
            )
            with torch.no_grad():
                target = self.decode_latents(target, height=height, width=width)
        return lpips(prediction_pixels, target.detach()).mean()

    def prepare_phase_inputs(
        self,
        request: PhaseRequest,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
    ) -> tuple[int, int, tuple[int, ...], ConditionBundle, Optional[ConditionBundle], float]:
        """Resolve homogeneous geometry and frozen prompt conditioning once per micro-batch."""
        with runtime.use_role("student", grad_enabled=False) as module:
            height, width, _, latent_shape = self.latent_geometry(batch, module)
            dtype = self.module_dtype(module)
            device = next(module.parameters()).device
        condition_start = time.perf_counter()
        condition, negative_condition = self.condition_provider.encode(
            batch,
            device=device,
            dtype=dtype,
            require_negative=request.kind == "student",
        )
        condition = self.prepare_condition_for_sequence_parallel(runtime, condition)
        if negative_condition is not None:
            negative_condition = self.prepare_condition_for_sequence_parallel(runtime, negative_condition)
        return height, width, latent_shape, condition, negative_condition, time.perf_counter() - condition_start

    def compute_phase(
        self,
        request: PhaseRequest,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
    ) -> DistillationPhaseComputation:
        """Build one role loss; multi-role phases use :meth:`iter_role_computations`."""
        from verl_omni.workers.diffusion_distillation_worker import DistillationPhaseComputation

        if request.kind not in {"student", "fake_score"}:
            raise ValueError(f"Unsupported Qwen DMD phase {request.kind!r}.")
        if len(request.trainable_roles) != 1:
            raise ValueError("Multi-role Qwen DMD2 phases require iter_role_computations().")
        height, width, latent_shape, condition, negative_condition, condition_duration = self.prepare_phase_inputs(
            request, batch, runtime
        )
        if request.kind == "student":
            if negative_condition is None:
                raise ValueError("Qwen DMD student phases require negative teacher conditioning.")
            loss, metrics = self.student_loss(
                batch,
                runtime,
                condition,
                negative_condition,
                height=height,
                width=width,
                latent_shape=latent_shape,
            )
            metrics["perf/condition_encode_s"] = condition_duration
            return DistillationPhaseComputation(losses={"student": loss}, metrics=metrics)
        role = request.trainable_roles[0]
        if role != "fake_score":
            raise ValueError("Discriminator computation must be paired through iter_role_computations().")
        loss, metrics, _ = self.fake_loss(
            runtime,
            condition,
            height=height,
            width=width,
            latent_shape=latent_shape,
        )
        metrics["perf/condition_encode_s"] = condition_duration
        return DistillationPhaseComputation(losses={role: loss}, metrics=metrics)

    def iter_role_computations(
        self,
        request: PhaseRequest,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
    ) -> Iterator[tuple[str, DistillationPhaseComputation]]:
        """Yield fake-score then discriminator graphs for immediate FSDP backward."""
        from verl_omni.workers.diffusion_distillation_worker import DistillationPhaseComputation

        if not self.adversarial or request.kind != "fake_score":
            raise ValueError("Multi-role computation is available only for DMD2 adversarial fake phases.")
        if request.trainable_roles != ("fake_score", "discriminator"):
            raise ValueError("DMD2 adversarial fake phases require fake_score then discriminator roles.")
        height, width, latent_shape, condition, _, condition_duration = self.prepare_phase_inputs(
            request, batch, runtime
        )
        fake_loss, fake_metrics, generated = self.fake_loss(
            runtime,
            condition,
            height=height,
            width=width,
            latent_shape=latent_shape,
        )
        fake_metrics["perf/condition_encode_s"] = condition_duration
        yield "fake_score", DistillationPhaseComputation(losses={"fake_score": fake_loss}, metrics=fake_metrics)
        discriminator_loss, discriminator_metrics = self.discriminator_loss(
            batch, generated, runtime, condition, height=height, width=width
        )
        yield (
            "discriminator",
            DistillationPhaseComputation(losses={"discriminator": discriminator_loss}, metrics=discriminator_metrics),
        )

    def state_dict(self) -> dict:
        """Return independent worker-local RNG stream states."""
        states = {name: generator.get_state().cpu() for name, generator in self._generators.items()}
        states.update(self._pending_generator_states)
        return {"version": 1, "rng_seed": self.rng_seed, "generator_states": states}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore worker-local rollout and score-noise RNG streams."""
        if state.get("version") != 1 or state.get("rng_seed") != self.rng_seed:
            raise ValueError("Qwen DMD computer checkpoint is incompatible with the active RNG configuration.")
        generator_states = state.get("generator_states")
        if not isinstance(generator_states, Mapping) or set(generator_states) - set(self.STREAM_OFFSETS):
            raise ValueError("Qwen DMD computer checkpoint contains invalid RNG streams.")
        self._generators.clear()
        self._pending_generator_states.clear()
        for name, generator_state in generator_states.items():
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError(f"Qwen DMD RNG state for {name!r} must be a tensor.")
            self._pending_generator_states[name] = generator_state


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dmd")
@DiffusionModelBase.register("QwenImagePipeline", algorithm="dmd2")
class QwenImageDistributionMatching(QwenImage, DiffusionAdversarialAdapter):
    """Qwen-Image architecture adapter for DMD and DMD2."""

    discriminator_parameter_prefixes = ("proj_out.classifier.",)

    @classmethod
    def build_discriminator_head(cls, module):
        """Reuse final normalized DiT features without rewriting transformer forward."""
        if isinstance(module.proj_out, QwenImageDMDProjection):
            return
        if not isinstance(module.proj_out, torch.nn.Linear):
            raise TypeError(f"Qwen DMD2 discriminator expects a linear proj_out, got {type(module.proj_out)}.")
        module.proj_out = QwenImageDMDProjection(module.proj_out)

    @classmethod
    def configure_role_parameters(cls, module, role):
        """Activate the discriminator head only for its independently optimized role."""
        super().configure_role_parameters(module, role)
        if isinstance(module.proj_out, QwenImageDMDProjection):
            module.proj_out.classify = role == "discriminator"

    @classmethod
    def build_distribution_matching_computer(
        cls,
        model_config: DiffusionModelConfig,
        plan: DistillationPlan,
    ) -> QwenImageDMDComputer:
        """Build the architecture-owned Qwen DMD phase program."""
        return QwenImageDMDComputer(model_config, plan)

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
"""Wan 2.1 causal training adapter and ODE-regression computation."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import (
    AutoregressiveDistillationAdapter,
    DiffusionModelBase,
    DistributionMatchingModelAdapter,
)
from verl_omni.pipelines.wan22_dance_grpo.diffusers_training_adapter import (
    Wan22DanceGRPO,
    _configure_wan_scheduler,
)
from verl_omni.trainer.diffusion.distillation.contracts import ConditionBundle, DistillationPlan, PhaseRequest
from verl_omni.trainer.diffusion.distillation.utils import (
    consistency_renoise_step,
    dmd_gradient,
    dmd_surrogate_loss,
    fake_score_loss,
    legacy_cfg,
    ode_regression_loss,
    timestep_shift,
    velocity_to_x0,
)
from verl_omni.utils.dataset.distillation import canonical_manifest_sha256
from verl_omni.workers.config import DiffusionModelConfig

from .causal_attention import WanCausalCache, allocate_wan_cache, configure_causal_wan, wan_causal_forward
from .inference import wan_reference_sigma

if TYPE_CHECKING:
    from verl_omni.workers.diffusion_distillation_worker import (
        DistillationPhaseComputation,
        DistillationRoleRuntime,
    )

__all__ = [
    "Wan21CausalODE",
    "Wan21CausVid",
    "WanODEComputer",
    "WanCausVidComputer",
    "WanConditionProvider",
    "build_wan_causal_timesteps",
]


def build_wan_causal_timesteps(
    timesteps: tuple[int, ...] | list[int], num_train_timesteps: int, shift: float
) -> torch.Tensor:
    """Warp an inference timestep fixture exactly once with the Wan rational shift."""
    if (
        not timesteps
        or any(isinstance(value, bool) or not isinstance(value, int) for value in timesteps)
        or any(value <= 0 or value > num_train_timesteps for value in timesteps)
        or any(left <= right for left, right in zip(timesteps, timesteps[1:], strict=False))
    ):
        raise ValueError("Wan causal denoising timesteps must be strictly descending positive integers in range.")
    if num_train_timesteps <= 0 or not torch.isfinite(torch.tensor(shift)) or shift < 1:
        raise ValueError("Wan causal timestep scale must be positive and shift must be finite and at least 1.")
    return timestep_shift(torch.tensor(timesteps, dtype=torch.float32), num_train_timesteps, shift)


class WanConditionProvider:
    """Load frozen Wan text conditioning or consume cached embeddings."""

    def __init__(self, model_path: str, provider: str, max_sequence_length: int) -> None:
        self.model_path = model_path
        self.provider = provider
        self.max_sequence_length = max_sequence_length
        self.pipeline = None

    @staticmethod
    def prompt_rows(value: Any, batch_size: int) -> list[str]:
        """Accept plain prompts or one text-only user message per row."""
        if hasattr(value, "tolist") and not isinstance(value, torch.Tensor):
            value = value.tolist()
        if batch_size == 1 and (
            isinstance(value, str) or isinstance(value, list) and value and isinstance(value[0], dict)
        ):
            value = [value]
        if not isinstance(value, list) or len(value) != batch_size:
            raise ValueError(f"raw_prompt must contain exactly {batch_size} row(s).")
        prompts = []
        for row in value:
            if isinstance(row, list):
                if len(row) != 1 or not isinstance(row[0], dict) or row[0].get("role") != "user":
                    raise ValueError("Wan ODE raw prompts require one text-only user message.")
                row = row[0].get("content")
            if not isinstance(row, str) or not row:
                raise ValueError("Wan ODE prompts must be non-empty strings.")
            prompts.append(row)
        return prompts

    def ensure_pipeline(self, device: torch.device, dtype: torch.dtype):
        """Load only the frozen Wan tokenizer and text encoder."""
        if self.pipeline is None:
            from diffusers import WanPipeline

            self.pipeline = WanPipeline.from_pretrained(
                self.model_path,
                transformer=None,
                vae=None,
                torch_dtype=dtype,
                local_files_only=os.path.isdir(self.model_path),
            ).to(device)
            self.pipeline.text_encoder.requires_grad_(False)
            self.pipeline.text_encoder.eval()
        return self.pipeline

    def encode(self, batch: TensorDict, *, device: torch.device, dtype: torch.dtype) -> ConditionBundle:
        """Return detached `[B, L, D]` Wan prompt embeddings."""
        if self.provider == "precomputed":
            embeds = tu.get(batch, "prompt_embeds")
            if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
                raise ValueError("Precomputed Wan conditioning requires prompt_embeds with shape [B, L, D].")
            embeds = embeds[:, : self.max_sequence_length]
        elif self.provider == "local_frozen_encoder":
            prompts = self.prompt_rows(tu.get(batch, "raw_prompt"), batch.batch_size[0])
            pipeline = self.ensure_pipeline(device, dtype)
            with torch.no_grad():
                embeds, _ = pipeline.encode_prompt(
                    prompt=prompts,
                    do_classifier_free_guidance=False,
                    max_sequence_length=self.max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
        else:
            raise ValueError(f"Unsupported Wan conditioning provider {self.provider!r}.")
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
            raise ValueError("Wan prompt encoder must return embeddings with shape [B, L, D].")
        if embeds.shape[0] != batch.batch_size[0] or embeds.shape[1] == 0:
            raise ValueError("Wan prompt embedding shape does not match the ODE batch.")
        embeds = embeds[:, : self.max_sequence_length].to(device=device, dtype=dtype).detach()
        return ConditionBundle(tensors={"prompt_embeds": embeds})


class WanODEComputer:
    """Train a causal Wan student against provenance-checked ODE trajectories."""

    def __init__(self, model_config: DiffusionModelConfig, plan: DistillationPlan) -> None:
        if plan.name != "ode_regression":
            raise ValueError(f"WanODEComputer requires recipe 'ode_regression', got {plan.name!r}.")
        self.model_config = model_config
        self.plan = plan
        self.frames_per_block = int(plan.rollout["frames_per_block"])
        self.loss_weight = float(plan.objective["loss_weight"])
        self.manifest_sha256 = str(plan.data_requirements["trajectory_manifest_sha256"])
        self.num_train_timesteps = int(plan.objective["num_train_timesteps"])
        self.rng_seed = int(plan.rollout["rng_seed"])
        self.condition_provider = WanConditionProvider(
            model_config.local_path or model_config.path,
            str(plan.data_requirements["conditioning_provider"]),
            int(model_config.pipeline.max_sequence_length),
        )
        self.generator: Optional[torch.Generator] = None
        self.pending_generator_state: Optional[torch.Tensor] = None

    def rng(self, device: torch.device, runtime: DistillationRoleRuntime) -> torch.Generator:
        """Return a checkpointable data-parallel-local ODE index generator."""
        if self.generator is None:
            engine = runtime.engine_for_role("student")
            rank = int(engine.get_data_parallel_rank()) if hasattr(engine, "get_data_parallel_rank") else 0
            self.generator = torch.Generator(device=device).manual_seed(self.rng_seed + rank)
            if self.pending_generator_state is not None:
                self.generator.set_state(self.pending_generator_state)
                self.pending_generator_state = None
        return self.generator

    def validate_manifest(self, batch: TensorDict) -> None:
        """Require every trajectory row to match the configured immutable manifest."""
        manifests = tu.get(batch, "trajectory_manifest")
        hashes = tu.get(batch, "trajectory_manifest_sha256")
        if isinstance(manifests, Mapping) and batch.batch_size[0] == 1:
            manifests = [manifests]
        if isinstance(hashes, str) and batch.batch_size[0] == 1:
            hashes = [hashes]
        if not isinstance(manifests, list) or not isinstance(hashes, list):
            raise ValueError("Wan ODE batches require one trajectory manifest and fingerprint per sample.")
        if len(manifests) != batch.batch_size[0] or len(hashes) != batch.batch_size[0]:
            raise ValueError("Wan ODE manifest batch size does not match the trajectory batch.")
        for manifest, digest in zip(manifests, hashes, strict=True):
            if canonical_manifest_sha256(manifest) != digest or digest != self.manifest_sha256:
                raise ValueError("Wan ODE trajectory manifest does not match the active recipe fingerprint.")

    def select_states(
        self,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        runtime: DistillationRoleRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one trajectory state per temporal block and return its valid-frame mask."""
        if trajectory.ndim != 6:
            raise ValueError("ode_latents must have shape [B, S, F, C, H, W].")
        batch_size, step_count, frames, channels, height, width = trajectory.shape
        if frames % self.frames_per_block:
            raise ValueError("Wan ODE latent frames must be divisible by frames_per_block.")
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(0).expand(batch_size, -1)
        if timesteps.shape != (batch_size, step_count):
            raise ValueError("ode_timesteps must have shape [B, S] matching ode_latents.")
        if not torch.all(timesteps == timesteps[0]):
            raise ValueError("All Wan ODE rows in one physical batch must use the same timestep schedule.")
        schedule = timesteps[0]
        if (
            schedule[-1].item() != 0
            or torch.any(schedule[:-1] <= schedule[1:])
            or torch.any(schedule < 0)
            or torch.any(schedule > self.num_train_timesteps)
        ):
            raise ValueError("Wan ODE timesteps must be strictly descending, in range, and end at zero.")
        block_count = frames // self.frames_per_block
        indices = torch.randint(
            step_count,
            (batch_size, block_count),
            device=trajectory.device,
            generator=self.rng(trajectory.device, runtime),
        )
        frame_indices = indices.repeat_interleave(self.frames_per_block, dim=1)
        frame_timesteps = timesteps.gather(1, frame_indices)
        all_zero = frame_timesteps.eq(0).all(dim=1)
        nonzero_indices = timesteps.ne(0).float().argmax(dim=1)
        indices[:, 0] = torch.where(all_zero, nonzero_indices, indices[:, 0])
        frame_indices = indices.repeat_interleave(self.frames_per_block, dim=1)
        gather_index = frame_indices[:, None, :, None, None, None].expand(
            batch_size, 1, frames, channels, height, width
        )
        noisy = trajectory.gather(1, gather_index).squeeze(1).permute(0, 2, 1, 3, 4).contiguous()
        frame_timesteps = timesteps.gather(1, frame_indices)
        return noisy, frame_timesteps, frame_timesteps.ne(0)

    def compute_phase(
        self,
        request: PhaseRequest,
        batch: TensorDict,
        runtime: DistillationRoleRuntime,
    ) -> DistillationPhaseComputation:
        """Run one full-sequence block-causal ODE-regression update."""
        from verl_omni.workers.diffusion_distillation_worker import DistillationPhaseComputation

        if request.kind != "student" or request.trainable_roles != ("student",):
            raise ValueError("Wan ODE regression accepts only a single student phase.")
        self.validate_manifest(batch)
        trajectory = tu.get(batch, "ode_latents")
        timesteps = tu.get(batch, "ode_timesteps")
        target = tu.get(batch, "final_clean_latent")
        if not all(isinstance(value, torch.Tensor) for value in (trajectory, timesteps, target)):
            raise ValueError("Wan ODE trajectory fields must be tensors.")
        with runtime.use_role("student", grad_enabled=True) as module:
            device = next(module.parameters()).device
            dtype = module.dtype
            trajectory = trajectory.to(device=device, dtype=torch.float32)
            timesteps = timesteps.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            condition = self.condition_provider.encode(batch, device=device, dtype=dtype)
            noisy, frame_timesteps, valid_mask = self.select_states(trajectory, timesteps, runtime)
            if target.shape != trajectory[:, -1].shape or not torch.equal(trajectory[:, -1], target):
                raise ValueError(
                    "final_clean_latent must exactly match the final ODE trajectory state [B, F, C, H, W]."
                )
            target = target.permute(0, 2, 1, 3, 4).contiguous()
            if target.shape != noisy.shape:
                raise ValueError("Converted Wan clean targets must match selected [B, C, F, H, W] latents.")
            patch_t, patch_h, patch_w = module.config.patch_size
            if patch_t != 1 or noisy.shape[-2] % patch_h or noisy.shape[-1] % patch_w:
                raise ValueError(
                    "Wan causal ODE requires temporal patch size 1 and divisible spatial latent dimensions."
                )
            token_timesteps = frame_timesteps.repeat_interleave(
                (noisy.shape[-2] // patch_h) * (noisy.shape[-1] // patch_w), dim=1
            )
            start = time.perf_counter()
            with wan_causal_forward(module, num_frames=noisy.shape[2], frames_per_block=self.frames_per_block):
                velocity = module(
                    hidden_states=noisy.to(dtype=dtype),
                    timestep=token_timesteps,
                    encoder_hidden_states=condition.tensors["prompt_embeds"],
                    return_dict=False,
                )[0]
            duration = time.perf_counter() - start
        sigma = frame_timesteps.div(self.num_train_timesteps).reshape(
            frame_timesteps.shape[0], 1, frame_timesteps.shape[1], 1, 1
        )
        prediction = velocity_to_x0(noisy, velocity, sigma)
        loss, active = ode_regression_loss(prediction, target, valid_mask[:, None, :, None, None])
        weighted_loss = self.loss_weight * loss
        return DistillationPhaseComputation(
            losses={"student": weighted_loss},
            loss_normalizer=active,
            metrics={
                "ode/loss": float(loss.detach()),
                "ode/active_elements": float(active),
                "ode/timestep": float(frame_timesteps.mean()),
                "perf/causal_full_forward_s": duration,
            },
        )

    def state_dict(self) -> dict:
        """Return the ODE trajectory-index RNG state."""
        state = self.generator.get_state().cpu() if self.generator is not None else self.pending_generator_state
        return {"version": 1, "rng_seed": self.rng_seed, "generator_state": state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the ODE trajectory-index RNG state."""
        if state.get("version") != 1 or state.get("rng_seed") != self.rng_seed:
            raise ValueError("Wan ODE computer checkpoint is incompatible with the active RNG configuration.")
        generator_state = state.get("generator_state")
        if generator_state is not None and not isinstance(generator_state, torch.Tensor):
            raise TypeError("Wan ODE generator state must be a tensor.")
        self.generator = None
        self.pending_generator_state = generator_state


@DiffusionModelBase.register("WanPipeline", algorithm="ode_regression")
class Wan21CausalODE(Wan22DanceGRPO, DistributionMatchingModelAdapter, AutoregressiveDistillationAdapter):
    """Wan 2.1 causal architecture adapter for ODE-regression initialization."""

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Configure the inherited training scheduler with Wan 2.1's default shift."""
        shift = model_config.pipeline.get("shift")
        _configure_wan_scheduler(
            scheduler,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            shift=3.0 if shift is None else shift,
            device=device,
        )

    @classmethod
    def distillation_capabilities(cls) -> frozenset[str]:
        """Declare causal and ODE-regression capabilities."""
        return frozenset({"distribution_matching", "autoregressive", "ode_regression"})

    @classmethod
    def configure_distillation_module(cls, module, roles):
        """Install causal attention only on student/EMA physical groups."""
        if set(roles) & {"student", "student_ema"}:
            return configure_causal_wan(module)
        return module

    @classmethod
    def allocate_cache(cls, module, **kwargs) -> WanCausalCache:
        """Allocate architecture-derived per-layer KV cache metadata."""
        return allocate_wan_cache(module, **kwargs)

    @classmethod
    def causal_forward(cls, module, **kwargs):
        """Run one full or incremental causal model call."""
        forward_kwargs = kwargs.pop("forward_kwargs")
        with wan_causal_forward(module, **kwargs):
            return module(**forward_kwargs)

    @classmethod
    def build_distribution_matching_computer(
        cls,
        model_config: DiffusionModelConfig,
        plan: DistillationPlan,
    ) -> WanODEComputer:
        """Build the Wan ODE-regression computation."""
        return WanODEComputer(model_config, plan)


class WanCausVidComputer:
    """Causal student / bidirectional scores with the released real-latent CausVid flow.

    Algorithmic reference: tianweiy/CausVid, causvid/dmd.py (no source copied).
    Uses its conditional-plus-difference CFG and nearest-grid sigma lookup. LoRA,
    fp32 reductions and completed-cycle accounting are explicit runtime choices.
    """

    def __init__(self, model_config: DiffusionModelConfig, plan: DistillationPlan) -> None:
        if plan.name != "causvid" or plan.rollout["strategy"] != "teacher_forced_causal":
            raise ValueError("Wan CausVid requires its real-latent teacher-forced recipe.")
        if plan.objective["teacher_cfg_norm"] != "none":
            raise ValueError("CausVid teacher uses legacy CFG without norm rescaling; set teacher_cfg_norm=none.")
        self.plan = plan
        self.model_config = model_config
        self.frames_per_block = plan.rollout["frames_per_block"]
        self.num_train_timesteps = plan.rollout["num_train_timesteps"]
        if plan.rollout["score_discrete_steps"] != self.num_train_timesteps:
            raise ValueError("CausVid requires discrete score sampling on the full training-timestep range.")
        self.timesteps = tuple(float(value) for value in plan.rollout["denoising_timesteps"])
        if (
            not self.timesteps
            or self.timesteps[0] != self.num_train_timesteps
            or self.timesteps[-1] != 0
            or any(left <= right for left, right in zip(self.timesteps, self.timesteps[1:], strict=False))
        ):
            raise ValueError("CausVid timesteps must descend from the training horizon to zero.")
        self.condition_provider = WanConditionProvider(
            model_config.local_path or model_config.path,
            str(plan.data_requirements["conditioning_provider"]),
            int(model_config.pipeline.max_sequence_length),
        )
        self.generators: dict[str, torch.Generator] = {}
        self.pending_states: dict[str, torch.Tensor] = {}

    def rng(self, stream: str, device: torch.device, runtime) -> torch.Generator:
        """Keep index/noise/score RNG independent and checkpointable on each DP rank."""
        streams = ("student_index", "student_noise", "score_timestep", "score_noise")
        if stream not in self.generators:
            engine = runtime.engine_for_role("student")
            rank = engine.get_data_parallel_rank()
            seed = int(self.plan.rollout["rng_seed"]) + rank * len(streams) + streams.index(stream)
            generator = torch.Generator(device=device).manual_seed(seed)
            if stream in self.pending_states:
                generator.set_state(self.pending_states[stream])
            self.generators[stream] = generator
        return self.generators[stream]

    def conditions(self, batch, runtime, need_negative: bool):
        """Use frozen cached text embeddings, or the existing local Wan encoder."""
        module = runtime.engine_for_role("student").module
        parameter = next(module.parameters())
        dtype = module.dtype
        condition = self.condition_provider.encode(batch, device=parameter.device, dtype=dtype)
        negative = None
        if need_negative:
            cached = tu.get(batch, "negative_prompt_embeds")
            if cached is None:
                if self.condition_provider.provider == "precomputed":
                    raise ValueError("Cached CausVid student phases require negative_prompt_embeds.")
                pipeline = self.condition_provider.ensure_pipeline(parameter.device, dtype)
                with torch.no_grad():
                    cached, _ = pipeline.encode_prompt(
                        prompt=[self.plan.data_requirements["negative_prompt"]] * batch.batch_size[0],
                        do_classifier_free_guidance=False,
                        device=parameter.device,
                        dtype=dtype,
                        max_sequence_length=self.condition_provider.max_sequence_length,
                    )
            if not isinstance(cached, torch.Tensor) or cached.ndim != 3 or cached.shape[0] != batch.batch_size[0]:
                raise ValueError("CausVid negative_prompt_embeds must match [B,L,D].")
            negative = (
                cached[:, : self.condition_provider.max_sequence_length]
                .to(device=parameter.device, dtype=dtype)
                .detach()
            )
        return condition.tensors["prompt_embeds"], negative

    def clean_latents(self, batch: TensorDict, device: torch.device) -> torch.Tensor:
        """Validate every real/teacher-clean FCHW sample against its VAE/trajectory provenance."""
        clean = tu.get(batch, "final_clean_latent")
        manifests = tu.get(batch, "trajectory_manifest")
        hashes = tu.get(batch, "trajectory_manifest_sha256")
        expected = self.plan.data_requirements["trajectory_manifest_sha256"]
        if not expected or not isinstance(manifests, list) or len(manifests) != batch.batch_size[0]:
            raise ValueError("CausVid real latents require one provenance manifest per sample.")
        if not isinstance(hashes, list) or len(hashes) != len(manifests):
            raise ValueError("CausVid real-latent provenance hashes are missing.")
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5 or clean.shape[0] != batch.batch_size[0]:
            raise ValueError("CausVid clean latents must be [B,F,C,H,W].")
        for manifest, digest in zip(manifests, hashes, strict=True):
            if digest != expected or canonical_manifest_sha256(manifest) != digest:
                raise ValueError("CausVid real-latent provenance does not match the configured manifest.")
            if (
                manifest.get("latent_layout") != "SFCHW"
                or manifest.get("num_frames") != clean.shape[1]
                or manifest.get("height") != clean.shape[-2]
                or manifest.get("width") != clean.shape[-1]
            ):
                raise ValueError("CausVid real-latent geometry does not match its provenance.")
        if clean.shape[1] % self.frames_per_block or not torch.isfinite(clean).all():
            raise ValueError("CausVid clean latents must be finite complete temporal blocks.")
        return clean.detach().to(device=device, dtype=torch.float32).permute(0, 2, 1, 3, 4).contiguous()

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Use the same sigma lookup for corruption and velocity-to-x0 conversion."""
        return wan_reference_sigma(timesteps, self.num_train_timesteps, self.plan.rollout["scheduler_shift"])

    def predict(self, runtime, role, latents, timesteps, prompt_embeds, *, grad_enabled: bool):
        """Run causal student or bidirectional score forwards through the shared role runtime."""
        with runtime.use_role(role, grad_enabled=grad_enabled) as module:
            module.eval()
            dtype = module.dtype
            model_timesteps = timesteps
            if role == "student":
                patch_t, patch_h, patch_w = module.config.patch_size
                if patch_t != 1:
                    raise ValueError("Causal Wan requires temporal patch size one.")
                model_timesteps = timesteps.repeat_interleave(
                    (latents.shape[-2] // patch_h) * (latents.shape[-1] // patch_w), dim=1
                )
            inputs = {
                "hidden_states": latents.to(dtype),
                "timestep": model_timesteps,
                "encoder_hidden_states": prompt_embeds.to(dtype),
                "return_dict": False,
            }
            if role == "student":
                with wan_causal_forward(module, num_frames=latents.shape[2], frames_per_block=self.frames_per_block):
                    return module(**inputs)[0].float()
            return module(**inputs)[0].float()

    def student_sample(self, batch, runtime, prompt_embeds, *, grad_enabled: bool):
        """Select independently noised clean states with one shared timestep per motion block."""
        clean = self.clean_latents(batch, prompt_embeds.device)
        batch_size, channels, frames, height, width = clean.shape
        schedule = clean.new_tensor(self.timesteps)
        candidates = []
        for timestep in schedule:
            noise = torch.randn(
                clean.shape, device=clean.device, generator=self.rng("student_noise", clean.device, runtime)
            )
            if timestep.item() == 0:
                candidates.append(clean)
            else:
                candidates.append(consistency_renoise_step(clean, noise, self.sigma(timestep)))
        indices = torch.randint(
            len(schedule),
            (batch_size, frames // self.frames_per_block),
            device=clean.device,
            generator=self.rng("student_index", clean.device, runtime),
        ).repeat_interleave(self.frames_per_block, dim=1)
        gather = indices[:, None, None, :, None, None].expand(batch_size, 1, channels, frames, height, width)
        noisy = torch.stack(candidates, dim=1).gather(1, gather).squeeze(1)
        times = schedule[indices]
        velocity = self.predict(runtime, "student", noisy, times, prompt_embeds, grad_enabled=grad_enabled)
        sigma = self.sigma(times)[:, None, :, None, None]
        return velocity_to_x0(noisy, velocity, sigma)

    def score_input(self, generated, runtime):
        """Sample one integer score timestep per video, shift then clamp, and independently re-noise."""
        times = torch.randint(
            self.num_train_timesteps,
            (generated.shape[0],),
            device=generated.device,
            generator=self.rng("score_timestep", generated.device, runtime),
        )
        times = timestep_shift(times, self.num_train_timesteps, self.plan.rollout["score_timestep_shift"])
        times = times.clamp(
            self.plan.rollout["score_sigma_min"] * self.num_train_timesteps,
            self.plan.rollout["score_sigma_max"] * self.num_train_timesteps,
        )
        sigma = self.sigma(times)[:, None, None, None, None]
        noise = torch.randn(
            generated.shape,
            device=generated.device,
            dtype=torch.float32,
            generator=self.rng("score_noise", generated.device, runtime),
        )
        noisy = consistency_renoise_step(generated.detach(), noise, sigma)
        return noisy, noise, times, sigma

    def compute_phase(self, request: PhaseRequest, batch: TensorDict, runtime) -> DistillationPhaseComputation:
        """Compute one student DMD or fake-score flow-matching loss; no trainer special cases."""
        from verl_omni.workers.diffusion_distillation_worker import DistillationPhaseComputation

        role = request.kind
        if role not in {"student", "fake_score"} or request.trainable_roles != (role,):
            raise ValueError("CausVid requires one student or fake_score optimizer per phase.")
        started = time.perf_counter()
        prompt, negative = self.conditions(batch, runtime, role == "student")
        generated = self.student_sample(batch, runtime, prompt, grad_enabled=role == "student")
        noisy, noise, times, sigma = self.score_input(generated, runtime)
        metrics = {"causvid/score_timestep": float(times.mean())}
        if role == "student":
            with torch.no_grad():
                fake = self.predict(runtime, "fake_score", noisy, times, prompt, grad_enabled=False)
                real_cond = self.predict(runtime, "teacher_score", noisy, times, prompt, grad_enabled=False)
                real_uncond = self.predict(runtime, "teacher_score", noisy, times, negative, grad_enabled=False)
                real = legacy_cfg(real_cond, real_uncond, self.plan.objective["teacher_guidance_scale"])
                gradient, normalizer, nonfinite = dmd_gradient(
                    velocity_to_x0(noisy, fake, sigma),
                    velocity_to_x0(noisy, real, sigma),
                    generated,
                    normalization_epsilon=self.plan.objective["normalization_epsilon"],
                )
            loss, active = dmd_surrogate_loss(generated, gradient)
            loss = self.plan.objective["dmd_loss_weight"] * loss
            metrics.update(
                {
                    "dmd/nonfinite": nonfinite,
                    "dmd/normalizer": float(normalizer.mean()),
                    "dmd/grad_norm": float(gradient.abs().mean()),
                }
            )
        else:
            velocity = self.predict(runtime, "fake_score", noisy, times, prompt, grad_enabled=True)
            loss, active = fake_score_loss(velocity, noise, generated.detach())
        metrics.update({f"{role}/loss": float(loss.detach()), "perf/causvid_compute_s": time.perf_counter() - started})
        return DistillationPhaseComputation(losses={role: loss}, metrics=metrics, loss_normalizer=active)

    def state_dict(self) -> dict:
        """Save every rank-local sampling stream for exact checkpoint replay."""
        states = dict(self.pending_states)
        states.update({name: generator.get_state().cpu() for name, generator in self.generators.items()})
        return {"version": 1, "rng_seed": self.plan.rollout["rng_seed"], "states": states}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore RNG before the next phase without retaining autograd graphs."""
        if state.get("version") != 1 or state.get("rng_seed") != self.plan.rollout["rng_seed"]:
            raise ValueError("Incompatible CausVid RNG checkpoint.")
        self.pending_states = dict(state["states"])
        self.generators = {}


@DiffusionModelBase.register("WanPipeline", algorithm="causvid")
class Wan21CausVid(Wan21CausalODE):
    """Explicit causal-student/bidirectional-score Wan architecture registration."""

    @classmethod
    def build_distribution_matching_computer(cls, model_config, plan) -> WanCausVidComputer:
        """Build the CausVid computer while preserving ODE as its own registry pair."""
        return WanCausVidComputer(model_config, plan)

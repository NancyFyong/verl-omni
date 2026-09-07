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
"""Local causal Wan sampling and video decoding; independent of the training driver."""

from collections.abc import Callable, Sequence

import torch

from verl_omni.trainer.diffusion.distillation.utils import consistency_renoise_step, velocity_to_x0

from .causal_attention import allocate_wan_cache, wan_causal_forward


def validate_wan_sampling_timesteps(timesteps: Sequence[float], num_train_timesteps: int) -> tuple[float, ...]:
    """Validate resolved (already shifted) positive timesteps, excluding terminal zero."""
    values = tuple(float(value) for value in timesteps)
    if (
        num_train_timesteps <= 0
        or not values
        or any(isinstance(value, bool) for value in timesteps)
        or any(not 0 < value <= num_train_timesteps for value in values)
        or any(left <= right for left, right in zip(values, values[1:], strict=False))
    ):
        raise ValueError("Wan sampling timesteps must be finite, positive, strictly descending and in range.")
    if values[0] != num_train_timesteps:
        raise ValueError("Noise-initialized Wan sampling must start at sigma=1.")
    return values


def wan_reference_sigma(timesteps: torch.Tensor, num_train_timesteps: int = 1000, shift: float = 8.0) -> torch.Tensor:
    """Map raw timesteps to the nearest sigma on the released Wan training grid.

    The reference uses a shifted linspace with an extra terminal element removed;
    even raw timestep zero maps to its last nonzero sigma. Keep raw timesteps for
    model conditioning and use this lookup only for corruption/x0 conversion.
    """
    raw = torch.linspace(1, 0, num_train_timesteps + 1, device=timesteps.device, dtype=torch.float32)[:-1]
    grid = shift * raw / (1 + (shift - 1) * raw)
    indices = (timesteps.float().reshape(-1, 1) - grid[None, :] * num_train_timesteps).abs().argmin(dim=1)
    return grid[indices].reshape(timesteps.shape)


@torch.no_grad()
def sample_wan_causal(
    module: torch.nn.Module,
    initial_noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    timesteps: Sequence[float],
    frames_per_block: int,
    generator: torch.Generator,
    num_train_timesteps: int = 1000,
    reference_grid_shift: float | None = None,
    callback: Callable[[int, int, int, int], None] | None = None,
) -> torch.Tensor:
    """Sample `[B,C,F,H,W]` latents with consistency re-noising and clean-block cache commits.

    ``timesteps`` are resolved values, not unshifted schedule indices. Student
    inference is conditional-only: teacher guidance has already shaped its targets.
    Each call owns a fresh cache, including text K/V; no cache crosses prompts.
    """
    schedule = validate_wan_sampling_timesteps(timesteps, num_train_timesteps)
    if initial_noise.ndim != 5 or not initial_noise.is_floating_point():
        raise ValueError("Wan initial noise must be floating point [B,C,F,H,W].")
    batch_size, channels, frames, height, width = initial_noise.shape
    if (
        isinstance(frames_per_block, bool)
        or not isinstance(frames_per_block, int)
        or frames_per_block <= 0
        or frames % frames_per_block
        or min(initial_noise.shape) <= 0
    ):
        raise ValueError("Wan latent frames must be positive and divisible by frames_per_block.")
    if channels != module.config.in_channels:
        raise ValueError("Wan noise channels must match the transformer.")
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != batch_size or prompt_embeds.shape[1] == 0:
        raise ValueError("Wan prompt embeddings must be nonempty [B,L,D] matching noise.")
    parameter = next(module.parameters())
    device, dtype = parameter.device, module.dtype
    cache = allocate_wan_cache(
        module, batch_size=batch_size, latent_height=height, latent_width=width, max_frames=frames
    )
    output = initial_noise.detach().to(device=device, dtype=torch.float32).clone()
    condition = prompt_embeds.detach().to(device=device, dtype=dtype)
    sigma_values = torch.tensor(schedule, device=device, dtype=torch.float32)
    sigmas = (
        sigma_values / num_train_timesteps
        if reference_grid_shift is None
        else wan_reference_sigma(sigma_values, num_train_timesteps, reference_grid_shift)
    )
    was_training = module.training
    module.eval()
    block_count = frames // frames_per_block
    try:
        for block in range(block_count):
            start = block * frames_per_block
            current = output[:, :, start : start + frames_per_block]
            for index, timestep in enumerate(schedule):
                with wan_causal_forward(
                    module, num_frames=frames_per_block, frames_per_block=frames_per_block, cache=cache
                ):
                    velocity = module(
                        hidden_states=current.to(dtype),
                        timestep=torch.full((batch_size,), timestep, device=device, dtype=torch.float32),
                        encoder_hidden_states=condition,
                        return_dict=False,
                    )[0]
                clean = velocity_to_x0(current, velocity, sigmas[index])
                if index + 1 < len(schedule):
                    noise = torch.randn(current.shape, device=device, dtype=torch.float32, generator=generator)
                    current = consistency_renoise_step(clean, noise, sigmas[index + 1])
                if callback is not None:
                    callback(block + 1, block_count, index + 1, len(schedule))
            with wan_causal_forward(
                module, num_frames=frames_per_block, frames_per_block=frames_per_block, cache=cache, commit_cache=True
            ):
                module(
                    hidden_states=clean.to(dtype),
                    timestep=torch.zeros(batch_size, device=device, dtype=torch.float32),
                    encoder_hidden_states=condition,
                    return_dict=False,
                )
            output[:, :, start : start + frames_per_block] = clean
        if not torch.isfinite(output).all():
            raise FloatingPointError("Causal Wan generated non-finite latents.")
        return output
    finally:
        cache.reset()
        module.train(was_training)


@torch.no_grad()
def decode_wan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized `[B,C,F,H,W]` Wan latents to RGB `[B,T,C,H,W]` in `[0,1]`."""
    if latents.ndim != 5 or latents.shape[1] != vae.config.z_dim:
        raise ValueError("Wan VAE input must be [B,z_dim,F,H,W].")
    parameter = next(vae.parameters())
    latents = latents.to(device=parameter.device, dtype=parameter.dtype)
    mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    decoded = vae.decode(latents * std + mean, return_dict=False)[0]
    if not torch.isfinite(decoded).all():
        raise FloatingPointError("Wan VAE returned non-finite video pixels.")
    return decoded.float().add(1).mul(0.5).clamp(0, 1).permute(0, 2, 1, 3, 4).contiguous()

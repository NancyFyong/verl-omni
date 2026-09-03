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
"""Pure DMD-family math used by the distillation trainer.

This module is independent of Ray and model libraries: it implements only the
detached-normalized distribution-matching gradient, the surrogate student loss,
the fake-score target, canonical x0 conversion, and the CFG forms that the
reference implementations use. The equations follow RFC §7 and §8.

Why this is a separate module (and not part of ``recipes.py`` or
``contracts.py``):

- ``contracts.py`` holds *data types* (immutable plan pieces, role layout, the
  execution state machine). It carries no equations.
- ``recipes.py`` holds *declarations* (which objective, which rollout strategy,
  which initialization, how roles map onto groups). It never computes a quantity.
- ``equations.py`` holds the only *executable equations* in the package. Every value
  is a pure function of its tensors; nothing here reads config, weights, or the
  prompt. Keeping these functions together means they can be unit-tested as
  algebraic identities and finite-difference checks without building a plan or an
  executor (see ``test_distillation_dmd_math_on_cpu.py``).

Boundary conditions (all must match the reviewed reference implementations):

- ``normalizer`` is formed over the **entire** ``x_g - x0_real`` tensor across all
  non-batch (block, frame, channel, spatial) dimensions of one sample, ``keepdim``
  per sample. It is **not** restricted by ``gradient_mask``.
- Only the surrogate loss is masked by ``gradient_mask``.
- ``normalization_epsilon`` is applied as ``max(normalizer, normalization_epsilon)``
  before division, and a non-finite ``g / normalizer`` is replaced by
  ``nan_to_num`` and counted.
- All score/objective arithmetic is fp32.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "epsilon_to_x0",
    "velocity_to_x0",
    "dmd_gradient",
    "dmd_surrogate_loss",
    "runaway_gradient",
    "fake_score_target",
    "fake_score_loss",
    "standard_cfg",
    "legacy_cfg",
    "timestep_shift",
]


def epsilon_to_x0(noisy: Tensor, epsilon: Tensor, sigma: Tensor, a_fn, b_fn) -> Tensor:
    """Convert an epsilon prediction to canonical ``x0`` for ``a(sigma)``/``b(sigma)``.

    For rectified flow, ``a(sigma) = 1 - sigma`` and ``b(sigma) = sigma``. For an
    epsilon-prediction model, ``x_sigma = a*x0 + b*epsilon`` so
    ``x0 = (x_sigma - b*epsilon) / a``.
    """
    a = a_fn(sigma)
    b = b_fn(sigma)
    return (noisy - b * epsilon) / a


def velocity_to_x0(noisy: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """Convert a velocity prediction to canonical ``x0`` (rectified flow).

    ``x_sigma = (1 - sigma) * x0 + sigma * epsilon`` and ``v = epsilon - x0``, so
    ``x0 = x_sigma - sigma * v``.
    """
    return noisy - sigma * velocity


def dmd_gradient(
    x0_fake: Tensor,
    x0_real: Tensor,
    x_g: Tensor,
    normalization_epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor, int]:
    """Compute the detached normalized fake-minus-real score gradient.

    Follows Self-Forcing ``_compute_kl_grad`` and LightX2V ``dmd_loss`` exactly:

    ``g = x0_fake - x0_real``
    ``normalizer = mean(abs(x_g - x0_real), non-batch dimensions)``
    ``g_normalized = nan_to_num(g / max(normalizer, normalization_epsilon))``

    Returns ``(g_normalized, normalizer, nonfinite_count)``. ``normalizer`` is
    formed over the entire ``x_g - x0_real`` tensor across all non-batch
    dimensions of one sample, ``keepdim`` per sample, and is **not** restricted by
    ``gradient_mask``.
    """
    x_g = x_g.float()
    x0_fake = x0_fake.float()
    x0_real = x0_real.float()

    g = x0_fake - x0_real
    # All non-batch dims (keep batch dim), per-sample keepdim.
    reduction_dims = tuple(range(1, x_g.dim()))
    normalizer = torch.abs(x_g - x0_real).mean(dim=reduction_dims, keepdim=True)
    normalizer = torch.maximum(normalizer, torch.as_tensor(normalization_epsilon, device=normalizer.device))
    g_normalized = g / normalizer
    nonfinite = (~torch.isfinite(g_normalized)).sum().item()
    g_normalized = torch.nan_to_num(g_normalized)
    return g_normalized, normalizer, nonfinite


def dmd_surrogate_loss(
    x_g: Tensor,
    g_normalized: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Surrogate objective ``L_DMD = 0.5 * mean((x_g - stop_gradient(x_g - g_norm))^2)``.

    Only the surrogate loss is masked by ``gradient_mask``. An all-masked loss is
    an error, not zero. Returns ``(loss, active_elements)``.
    """
    x_g = x_g.float()
    g_normalized = g_normalized.float()
    target = (x_g - g_normalized).detach()
    if gradient_mask is None:
        loss = 0.5 * torch.mean((x_g - target) ** 2)
        active = x_g.numel()
    else:
        mask = gradient_mask.bool()
        active = int(mask.sum().item())
        if active == 0:
            raise ValueError("all-masked DMD loss is an error, not zero.")
        diff = (x_g - target) ** 2
        loss = 0.5 * torch.sum(diff[mask]) / active
    return loss, active


def runaway_gradient(
    x_g: Tensor,
    g_normalized: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """The gradient of the surrogate loss w.r.t. ``x_g`` (used for finite-difference
    checks). Algebraically ``0.5 * mean((x_g - stop_gradient(x_g - g_norm))^2)`` has
    gradient equal to ``g_normalized``.
    """
    x_g = x_g.float()
    g_normalized = g_normalized.float()
    if gradient_mask is None:
        return g_normalized, x_g.numel()
    mask = gradient_mask.bool()
    active = int(mask.sum().item())
    if active == 0:
        raise ValueError("all-masked DMD loss is an error, not zero.")
    grad = torch.zeros_like(x_g)
    grad[mask] = g_normalized[mask]
    return grad, active


def fake_score_target(noise: Tensor, x_g: Tensor) -> Tensor:
    """Fake-score epsilon-prediction target ``v_target = epsilon - x_g``.

    The fake score is trained to denoise the generated clean latent with
    ``x_sigma = (1 - sigma) * x_g + sigma * epsilon``.
    """
    return noise.float() - x_g.float()


def fake_score_loss(
    model_output: Tensor,
    noise: Tensor,
    x_g: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Fake-score denoising MSE against the epsilon target.

    ``L_fake = mean((model_output - (epsilon - x_g))^2)``. With a mask, the loss
    divides by the number of active elements after applying the mask.
    """
    target = fake_score_target(noise, x_g)
    model_output = model_output.float()
    if gradient_mask is None:
        loss = torch.mean((model_output - target) ** 2)
        active = model_output.numel()
    else:
        mask = gradient_mask.bool()
        active = int(mask.sum().item())
        if active == 0:
            raise ValueError("all-masked fake-score loss is an error, not zero.")
        diff = (model_output - target) ** 2
        loss = torch.sum(diff[mask]) / active
    return loss, active


def standard_cfg(cond: Tensor, uncond: Tensor, guidance_scale: float, cfg_norm: str = "none") -> Tensor:
    """Classifier-free guidance in the ``pred_cond + scale*(pred_cond - pred_uncond)`` form.

    ``cfg_norm`` is one of ``none``, ``layer_norm``, or ``scalar``. For
    ``layer_norm`` the guided output is rescaled to the L2 norm of the conditional
    prediction along the last dim; for ``scalar`` it is multiplied by
    ``min(1.0, ||cond|| / ||guided||)``.
    """
    cond = cond.float()
    uncond = uncond.float()
    guided = uncond + guidance_scale * (cond - uncond)
    if cfg_norm == "none":
        return guided
    if cfg_norm == "layer_norm":
        cond_norm = cond.norm(p=2, dim=-1, keepdim=True)
        guided_norm = guided.norm(p=2, dim=-1, keepdim=True)
        scale = torch.where(
            cond_norm > 0,
            cond_norm / torch.clamp(guided_norm, min=1e-8),
            torch.ones_like(cond_norm),
        )
        return guided * scale
    if cfg_norm == "scalar":
        cond_norm = cond.norm(p=2, dim=-1, keepdim=True)
        guided_norm = guided.norm(p=2, dim=-1, keepdim=True)
        scale = torch.clamp(cond_norm / torch.clamp(guided_norm, min=1e-8), max=1.0)
        return guided * scale
    raise ValueError(f"Unknown cfg_norm {cfg_norm!r}; expected one of {{'none', 'layer_norm', 'scalar'}}.")


def legacy_cfg(cond: Tensor, uncond: Tensor, guidance_scale: float) -> Tensor:
    """Self-Forcing CFG form ``cond + legacy_scale*(cond - uncond)``.

    This is *not* numerically identical to :func:`standard_cfg` with the same
    scale. A parity recipe must convert explicitly rather than silently reusing the
    number.
    """
    return cond.float() + guidance_scale * (cond.float() - uncond.float())


def timestep_shift(timestep: Tensor, num_train_timesteps: int, shift: float = 1.0) -> Tensor:
    """Apply the time-shift remapping exactly once.

    ``shifted = shift * (t / T) / (1 + (shift - 1) * (t / T)) * T``. The
    normalization is by the model's ``num_train_timesteps``, so it diverges from a
    hardcoded 1000 for any model whose ``num_train_timesteps`` is not 1000.
    """
    if shift <= 1.0:
        return timestep.float()
    t = timestep.float()
    frac = t / num_train_timesteps
    shifted = shift * frac / (1.0 + (shift - 1.0) * frac) * num_train_timesteps
    return shifted

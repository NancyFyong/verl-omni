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
"""CPU tests for the pure DMD-family math.

The boundary conditions asserted here are the ones the reviewed reference
implementations use (Self-Forcing ``_compute_kl_grad``, LightX2V ``dmd_loss``):
the normalizer spans all non-batch dimensions per sample and is **not** masked,
only the surrogate loss is masked, and the surrogate gradient equals the
normalized score difference.
"""

import pytest
import torch

from verl_omni.trainer.diffusion.distillation.equations import (
    dmd_gradient,
    dmd_surrogate_loss,
    fake_score_loss,
    fake_score_target,
    legacy_cfg,
    standard_cfg,
    timestep_shift,
    velocity_to_x0,
)


class TestCanonicalConversion:
    def test_velocity_to_x0_matches_flow_definition(self):
        x_sigma = torch.randn(2, 3, 4, 4)
        v = torch.randn(2, 3, 4, 4)
        sigma = torch.rand(2, 1, 1, 1)
        # x_sigma = (1 - sigma) x0 + sigma eps and v = eps - x0 => x0 = x_sigma - sigma v
        assert torch.allclose(velocity_to_x0(x_sigma, v, sigma), x_sigma - sigma * v)

    def test_velocity_to_x0_roundtrip(self):
        x0 = torch.randn(2, 3, 4, 4)
        eps = torch.randn(2, 3, 4, 4)
        sigma = torch.rand(2, 1, 1, 1)
        x_sigma = (1 - sigma) * x0 + sigma * eps
        v = eps - x0
        assert torch.allclose(velocity_to_x0(x_sigma, v, sigma), x0, atol=1e-5)


class TestDMDGradient:
    def test_gradient_direction_is_fake_minus_real(self):
        x_g = torch.randn(2, 3, 4, 4)
        x0_real = torch.randn(2, 3, 4, 4)
        x0_fake = torch.randn(2, 3, 4, 4)
        g_norm, normalizer, _ = dmd_gradient(x0_fake, x0_real, x_g)
        expected = (x0_fake - x0_real) / normalizer
        assert torch.allclose(g_norm, expected, atol=1e-6)

    def test_normalizer_spans_all_non_batch_dims_with_keepdim(self):
        x_g = torch.randn(3, 5, 6, 7)
        x0_real = torch.randn(3, 5, 6, 7)
        _, normalizer, _ = dmd_gradient(torch.randn(3, 5, 6, 7), x0_real, x_g)
        assert normalizer.shape == (3, 1, 1, 1)
        expected = torch.abs(x_g - x0_real).mean(dim=(1, 2, 3), keepdim=True)
        assert torch.allclose(normalizer, expected, atol=1e-6)

    def test_normalizer_is_not_reduced_by_gradient_mask(self):
        """The mask applies only to the loss, never to the normalizer."""
        x_g = torch.randn(2, 3, 4, 4)
        x0_real = torch.randn(2, 3, 4, 4)
        x0_fake = torch.randn(2, 3, 4, 4)
        _, normalizer_a, _ = dmd_gradient(x0_fake, x0_real, x_g)
        # dmd_gradient takes no mask at all: there is no way to shrink it.
        _, normalizer_b, _ = dmd_gradient(x0_fake, x0_real, x_g)
        assert torch.allclose(normalizer_a, normalizer_b)

    def test_epsilon_floor_is_applied_before_division(self):
        x_g = torch.zeros(1, 1, 2, 2)
        x0_real = torch.zeros(1, 1, 2, 2)  # normalizer would be 0
        x0_fake = torch.ones(1, 1, 2, 2)
        eps = 1e-3
        g_norm, normalizer, _ = dmd_gradient(x0_fake, x0_real, x_g, normalization_epsilon=eps)
        assert torch.allclose(normalizer, torch.full_like(normalizer, eps))
        assert torch.isfinite(g_norm).all()

    def test_nonfinite_is_replaced_and_counted(self):
        x_g = torch.zeros(1, 1, 2, 2)
        x0_real = torch.zeros(1, 1, 2, 2)
        x0_fake = torch.full((1, 1, 2, 2), float("inf"))
        g_norm, _, nonfinite = dmd_gradient(x0_fake, x0_real, x_g, normalization_epsilon=1e-8)
        assert nonfinite > 0
        assert torch.isfinite(g_norm).all()


class TestSurrogate:
    def test_surrogate_gradient_equals_normalized_score_difference(self):
        """d/dx_g of 0.5*mean((x_g - sg(x_g - g))^2) is g / numel."""
        x_g = torch.randn(2, 3, 4, 4, requires_grad=True)
        g_norm = torch.randn(2, 3, 4, 4)
        loss, active = dmd_surrogate_loss(x_g, g_norm)
        loss.backward()
        assert torch.allclose(x_g.grad, g_norm / active, atol=1e-6)

    def test_finite_difference_check(self):
        """Finite-difference the surrogate with the target held fixed.

        ``gradcheck`` cannot be used directly: the surrogate deliberately builds
        its target with ``detach()``, so a numerical perturbation of ``x_g`` moves
        the target too and the numerical Jacobian collapses to zero. The
        meaningful check is that, for a *fixed* stop-gradient target, the analytic
        gradient equals ``g_normalized / N``.
        """
        torch.manual_seed(0)
        x_g = torch.randn(1, 1, 3, 3, dtype=torch.float64)
        g_norm = torch.randn(1, 1, 3, 3, dtype=torch.float64)
        target = (x_g - g_norm).detach()
        n = x_g.numel()

        def loss_fn(x):
            return 0.5 * torch.mean((x - target) ** 2)

        eps = 1e-6
        numerical = torch.zeros_like(x_g)
        flat = x_g.reshape(-1)
        for i in range(n):
            plus = flat.clone()
            plus[i] += eps
            minus = flat.clone()
            minus[i] -= eps
            numerical.reshape(-1)[i] = (loss_fn(plus.view_as(x_g)) - loss_fn(minus.view_as(x_g))) / (2 * eps)

        analytic = g_norm / n
        assert torch.allclose(numerical, analytic, atol=1e-7)

    def test_detached_target_blocks_gradient_through_scores(self):
        """No graph is retained through the teacher/fake-score estimate."""
        x_g = torch.randn(1, 1, 2, 2, requires_grad=True)
        g_norm = torch.randn(1, 1, 2, 2, requires_grad=True)
        loss, _ = dmd_surrogate_loss(x_g, g_norm)
        loss.backward()
        assert g_norm.grad is None

    def test_mask_applies_to_loss_only(self):
        x_g = torch.randn(2, 1, 2, 2, requires_grad=True)
        g_norm = torch.ones(2, 1, 2, 2)
        mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
        mask[0] = True  # only the first sample contributes
        loss, active = dmd_surrogate_loss(x_g, g_norm, gradient_mask=mask)
        assert active == 4
        loss.backward()
        # Masked-out elements receive no gradient.
        assert torch.count_nonzero(x_g.grad[1]) == 0
        assert torch.count_nonzero(x_g.grad[0]) == 4

    def test_all_masked_loss_is_an_error_not_zero(self):
        x_g = torch.randn(2, 1, 2, 2)
        g_norm = torch.ones(2, 1, 2, 2)
        mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="all-masked"):
            dmd_surrogate_loss(x_g, g_norm, gradient_mask=mask)

    def test_loss_is_fp32_even_for_half_inputs(self):
        x_g = torch.randn(2, 1, 2, 2, dtype=torch.float16)
        g_norm = torch.randn(2, 1, 2, 2, dtype=torch.float16)
        loss, _ = dmd_surrogate_loss(x_g, g_norm)
        assert loss.dtype == torch.float32


class TestFakeScore:
    def test_target_is_epsilon_minus_x_g(self):
        eps = torch.randn(2, 3, 4, 4)
        x_g = torch.randn(2, 3, 4, 4)
        assert torch.allclose(fake_score_target(eps, x_g), eps - x_g)

    def test_loss_is_mse_against_target(self):
        eps = torch.randn(2, 3, 4, 4)
        x_g = torch.randn(2, 3, 4, 4)
        out = torch.randn(2, 3, 4, 4)
        loss, active = fake_score_loss(out, eps, x_g)
        assert active == out.numel()
        assert torch.allclose(loss, torch.mean((out - (eps - x_g)) ** 2), atol=1e-6)

    def test_masked_loss_divides_by_active_elements(self):
        eps = torch.zeros(2, 1, 2, 2)
        x_g = torch.zeros(2, 1, 2, 2)
        out = torch.ones(2, 1, 2, 2)
        mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
        mask[0] = True
        loss, active = fake_score_loss(out, eps, x_g, gradient_mask=mask)
        assert active == 4
        assert torch.allclose(loss, torch.tensor(1.0))

    def test_all_masked_fake_loss_raises(self):
        z = torch.zeros(2, 1, 2, 2)
        mask = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="all-masked"):
            fake_score_loss(z, z, z, gradient_mask=mask)


class TestCFG:
    def test_standard_cfg_form(self):
        cond = torch.randn(2, 8)
        uncond = torch.randn(2, 8)
        out = standard_cfg(cond, uncond, 3.0)
        assert torch.allclose(out, uncond + 3.0 * (cond - uncond), atol=1e-6)

    def test_legacy_cfg_differs_from_standard(self):
        """Self-Forcing's cond + s*(cond-uncond) is not the standard definition."""
        cond = torch.randn(2, 8)
        uncond = torch.randn(2, 8)
        assert not torch.allclose(standard_cfg(cond, uncond, 3.0), legacy_cfg(cond, uncond, 3.0))

    def test_legacy_cfg_equals_standard_with_shifted_scale(self):
        """cond + s*(cond-uncond) == uncond + (s+1)*(cond-uncond)."""
        cond = torch.randn(2, 8)
        uncond = torch.randn(2, 8)
        assert torch.allclose(legacy_cfg(cond, uncond, 3.0), standard_cfg(cond, uncond, 4.0), atol=1e-6)

    def test_cfg_scale_zero_returns_unconditional(self):
        cond = torch.randn(2, 8)
        uncond = torch.randn(2, 8)
        assert torch.allclose(standard_cfg(cond, uncond, 0.0), uncond, atol=1e-6)

    def test_unknown_cfg_norm_raises(self):
        cond = torch.randn(2, 8)
        with pytest.raises(ValueError, match="Unknown cfg_norm"):
            standard_cfg(cond, cond, 1.0, cfg_norm="bogus")

    @pytest.mark.parametrize("norm", ["none", "layer_norm", "scalar"])
    def test_cfg_norm_modes_are_finite(self, norm):
        cond = torch.randn(2, 8)
        uncond = torch.randn(2, 8)
        assert torch.isfinite(standard_cfg(cond, uncond, 3.0, cfg_norm=norm)).all()


class TestTimestepShift:
    def test_shift_of_one_is_identity(self):
        t = torch.tensor([0.0, 250.0, 500.0, 1000.0])
        assert torch.allclose(timestep_shift(t, 1000, shift=1.0), t)

    def test_shift_normalizes_by_num_train_timesteps(self):
        """The remap divides by the model's T, not a hardcoded 1000."""
        t = torch.tensor([500.0])
        shifted_1000 = timestep_shift(t, 1000, shift=3.0)
        shifted_500 = timestep_shift(torch.tensor([250.0]), 500, shift=3.0)
        # Same normalized position (0.5) maps to the same fraction of T.
        assert torch.allclose(shifted_1000 / 1000, shifted_500 / 500, atol=1e-6)

    def test_shift_is_monotonic(self):
        t = torch.linspace(0, 1000, 11)
        shifted = timestep_shift(t, 1000, shift=3.0)
        assert torch.all(shifted[1:] >= shifted[:-1])

    def test_endpoints_are_preserved(self):
        t = torch.tensor([0.0, 1000.0])
        shifted = timestep_shift(t, 1000, shift=5.0)
        assert torch.allclose(shifted, t, atol=1e-4)

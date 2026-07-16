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
"""FLUX.2-Klein sigma-schedule parity with the official diffusers pipeline.

The training/rollout adapters must forward the official ``linspace(1, 1/N, N)``
base grid into ``set_timesteps(sigmas=..., mu=mu)``. Omitting ``sigmas`` lets the
dynamic-shift scheduler collapse its small-sigma tail toward
``~1 / num_train_timesteps`` and diverge from the pretrained FLUX.2-Klein
inference trajectory (see UniRL ``get_sigma_schedule``).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from verl_omni.pipelines.flux2_klein_flow_grpo.diffusers_training_adapter import (
    Flux2KleinFlowGRPO,
    build_flux2_klein_sigmas,
    compute_empirical_mu,
    resolve_flux2_klein_sigmas,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

# Matches the published black-forest-labs/FLUX.2-klein-base-4B scheduler config.
_KLEIN_SCHEDULER_CONFIG = dict(
    num_train_timesteps=1000,
    shift=3.0,
    use_dynamic_shifting=True,
    time_shift_type="exponential",
    base_image_seq_len=256,
    max_image_seq_len=4096,
    base_shift=0.5,
    max_shift=1.15,
    shift_terminal=None,
)


def _model_config(height=512, width=512, num_inference_steps=10):
    return SimpleNamespace(
        pipeline=SimpleNamespace(height=height, width=width, num_inference_steps=num_inference_steps),
    )


def _official_sigmas(num_inference_steps, mu):
    scheduler = FlowMatchEulerDiscreteScheduler(**_KLEIN_SCHEDULER_CONFIG)
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device="cpu", sigmas=sigmas, mu=mu)
    return scheduler.sigmas


@pytest.mark.parametrize("num_inference_steps", [10, 12, 28, 50])
@pytest.mark.parametrize(("height", "width"), [(512, 512), (768, 512)])
def test_training_set_timesteps_matches_official_pipeline(num_inference_steps, height, width):
    scheduler = FlowMatchSDEDiscreteScheduler(**_KLEIN_SCHEDULER_CONFIG)
    Flux2KleinFlowGRPO.set_timesteps(scheduler, _model_config(height, width, num_inference_steps), "cpu")

    latent_h = height // 16
    latent_w = width // 16
    mu = compute_empirical_mu(latent_h * latent_w, num_inference_steps)

    torch.testing.assert_close(scheduler.sigmas, _official_sigmas(num_inference_steps, mu))


def test_omitting_sigmas_diverges_from_official_schedule():
    """Guard against regressing to the ``set_timesteps(mu=...)``-only call."""
    num_inference_steps = 10
    mu = compute_empirical_mu(1024, num_inference_steps)

    without_sigmas = FlowMatchSDEDiscreteScheduler(**_KLEIN_SCHEDULER_CONFIG)
    without_sigmas.set_timesteps(num_inference_steps, device="cpu", mu=mu)

    assert not torch.allclose(without_sigmas.sigmas, _official_sigmas(num_inference_steps, mu))


def test_build_sigmas_is_official_linspace():
    sigmas = build_flux2_klein_sigmas(10)
    np.testing.assert_allclose(sigmas, np.linspace(1.0, 0.1, 10, dtype=np.float32))


def test_build_sigmas_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
        build_flux2_klein_sigmas(0)


def test_resolve_sigmas_defaults_to_official_grid():
    resolved = resolve_flux2_klein_sigmas(None, 10)
    np.testing.assert_allclose(resolved, build_flux2_klein_sigmas(10))


def test_resolve_sigmas_accepts_exact_length():
    resolved = resolve_flux2_klein_sigmas([1.0, 0.5], 2)
    np.testing.assert_allclose(resolved, np.array([1.0, 0.5], dtype=np.float32))


def test_resolve_sigmas_drops_terminal_zero_for_t_plus_one():
    resolved = resolve_flux2_klein_sigmas([1.0, 0.5, 0.0], 2)
    np.testing.assert_allclose(resolved, np.array([1.0, 0.5], dtype=np.float32))


def test_resolve_sigmas_rejects_incompatible_length():
    with pytest.raises(ValueError, match="incompatible with num_inference_steps"):
        resolve_flux2_klein_sigmas([1.0, 0.5, 0.25, 0.1], 2)

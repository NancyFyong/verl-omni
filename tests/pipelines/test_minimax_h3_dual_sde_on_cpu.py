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
"""CPU correctness gate for the MiniMax H3 dual-stream SDE scheduler.

The dual scheduler is a thin container of two stock ``FlowMatchSDEDiscreteScheduler``
instances fed pre-shifted H3 sigmas (video ``flow_shift=12.0``, audio ``3.0``). These
tests pin the two things that can silently go wrong: (1) the sigma schedules must equal
``minimax_h3_time_shift_sigmas`` -- a double-shift would leave the training sigmas
misaligned with the vllm-omni rollout; (2) each stream's reverse-SDE log-prob must be a
correctly-normalized Gaussian transition density, checked both against the closed form
and by Riemann-integrating that density to 1.0.
"""

import math

import pytest
import torch

from verl_omni.pipelines.schedulers import FlowMatchDualSDEDiscreteScheduler, minimax_h3_time_shift_sigmas

_NUM_STEPS = 5
_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0
_BATCH = 2
_ROWS = 4
_COLS = 6


def _dual_scheduler(num_steps=_NUM_STEPS):
    scheduler = FlowMatchDualSDEDiscreteScheduler(video_flow_shift=_VIDEO_SHIFT, audio_flow_shift=_AUDIO_SHIFT)
    scheduler.set_timesteps(num_steps, device="cpu")
    return scheduler


class TestMiniMaxH3TimeShiftSigmas:
    def test_descends_from_one_to_zero(self):
        sigmas = minimax_h3_time_shift_sigmas(_NUM_STEPS, _VIDEO_SHIFT)
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == pytest.approx(0.0)
        assert all(earlier > later for earlier, later in zip(sigmas, sigmas[1:], strict=False))

    def test_larger_shift_pushes_mass_toward_high_noise(self):
        # A bigger shift keeps interior sigmas closer to 1.0 (more high-noise steps).
        video = minimax_h3_time_shift_sigmas(_NUM_STEPS, _VIDEO_SHIFT)
        audio = minimax_h3_time_shift_sigmas(_NUM_STEPS, _AUDIO_SHIFT)
        assert video[1] > audio[1]

    def test_rejects_nonpositive_arguments(self):
        with pytest.raises(ValueError):
            minimax_h3_time_shift_sigmas(_NUM_STEPS, 0.0)
        with pytest.raises(ValueError):
            minimax_h3_time_shift_sigmas(0, _VIDEO_SHIFT)


class TestDualSchedulerSigmaAlignment:
    def test_per_stream_sigmas_match_helper_no_double_shift(self):
        scheduler = _dual_scheduler()
        video = [round(float(s), 5) for s in scheduler.video_scheduler.sigmas.tolist()]
        audio = [round(float(s), 5) for s in scheduler.audio_scheduler.sigmas.tolist()]
        assert video == [round(s, 5) for s in minimax_h3_time_shift_sigmas(_NUM_STEPS, _VIDEO_SHIFT)]
        assert audio == [round(s, 5) for s in minimax_h3_time_shift_sigmas(_NUM_STEPS, _AUDIO_SHIFT)]

    def test_streams_have_distinct_schedules(self):
        scheduler = _dual_scheduler()
        assert not torch.allclose(scheduler.video_scheduler.sigmas, scheduler.audio_scheduler.sigmas)

    def test_timesteps_delegate_to_video_stream(self):
        scheduler = _dual_scheduler()
        assert scheduler.timesteps is scheduler.video_scheduler.timesteps
        assert len(scheduler.timesteps) == _NUM_STEPS - 1


class TestPerModalitySampleStep:
    def _sample(self, stream, timestep):
        sample = torch.randn(_BATCH, _ROWS, _COLS)
        model_output = torch.randn(_BATCH, _ROWS, _COLS)
        return stream.sample_previous_step(
            sample=sample,
            model_output=model_output,
            timestep=timestep.expand(_BATCH),
            noise_level=1.0,
            sde_type="sde",
            generator=torch.Generator().manual_seed(0),
            return_logprobs=True,
            return_sqrt_dt=True,
        )

    def test_returns_batched_fp32_logprob_and_stats(self):
        scheduler = _dual_scheduler()
        step = 1
        for stream in (scheduler.video_scheduler, scheduler.audio_scheduler):
            prev, log_prob, mean, std_dev_t, sqrt_dt = self._sample(stream, stream.timesteps[step])
            assert log_prob.shape == (_BATCH,)
            assert std_dev_t.shape == (_BATCH, 1, 1)
            assert sqrt_dt.shape == (_BATCH,)
            assert prev.dtype == torch.float32
            assert std_dev_t.dtype == torch.float32
            assert torch.isfinite(log_prob).all()

    def test_streams_differ_at_same_step_index(self):
        # Same reverse-step index, different sigma schedules -> different noise std.
        scheduler = _dual_scheduler()
        step = 1
        _, _, _, video_std, _ = self._sample(scheduler.video_scheduler, scheduler.video_scheduler.timesteps[step])
        _, _, _, audio_std, _ = self._sample(scheduler.audio_scheduler, scheduler.audio_scheduler.timesteps[step])
        assert not torch.allclose(video_std, audio_std)


class TestAnalyticGaussianLogProb:
    """The reverse-SDE log-prob must be a correctly-normalized Gaussian density."""

    def _step(self, stream, step):
        sample = torch.randn(_BATCH, _ROWS, _COLS)
        model_output = torch.randn(_BATCH, _ROWS, _COLS)
        return stream.sample_previous_step(
            sample=sample,
            model_output=model_output,
            timestep=stream.timesteps[step].expand(_BATCH),
            noise_level=1.0,
            sde_type="sde",
            generator=torch.Generator().manual_seed(0),
            return_logprobs=True,
            return_sqrt_dt=True,
        )

    def test_logprob_matches_closed_form_gaussian(self):
        scheduler = _dual_scheduler()
        prev, log_prob, mean, std_dev_t, sqrt_dt = self._step(scheduler.video_scheduler, step=1)
        std = std_dev_t * sqrt_dt.view(-1, 1, 1)
        per_element = -((prev - mean) ** 2) / (2 * std**2) - torch.log(std) - math.log(math.sqrt(2 * math.pi))
        torch.testing.assert_close(log_prob, per_element.mean(dim=(1, 2)))

    def test_analytic_density_integrates_to_one(self):
        # Riemann-integrate the per-element Gaussian implied by the scheduler's
        # (mean, std) over +/-8 sigma; a wrong normalizer would not integrate to 1.
        scheduler = _dual_scheduler()
        _, _, mean, std_dev_t, sqrt_dt = self._step(scheduler.audio_scheduler, step=1)
        mu = float(mean[0, 0, 0])
        sigma = float((std_dev_t * sqrt_dt.view(-1, 1, 1))[0, 0, 0])

        grid = torch.linspace(mu - 8 * sigma, mu + 8 * sigma, 20001)
        dx = grid[1] - grid[0]
        log_density = -((grid - mu) ** 2) / (2 * sigma**2) - math.log(sigma) - math.log(math.sqrt(2 * math.pi))
        integral = torch.exp(log_density).sum() * dx
        torch.testing.assert_close(integral, torch.tensor(1.0), atol=1e-3, rtol=1e-3)

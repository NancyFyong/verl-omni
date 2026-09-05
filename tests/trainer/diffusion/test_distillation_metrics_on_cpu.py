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
"""Cycle metrics and existing Tracking/DistProfiler lifecycle regression tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.distillation.contracts import PhaseResult
from verl_omni.trainer.diffusion.distillation.control_plane import FakeBatchProvider, FakePhaseExecutor
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.diffusion.distillation.recipes import build_plan


class MetricExecutor(FakePhaseExecutor):
    def execute_phase(self, request, batch):
        result = super().execute_phase(request, batch)
        return PhaseResult(
            optimizer_steps=result.optimizer_steps,
            metrics={
                f"{request.kind}/loss": float(request.repeat_index + 1),
                "perf/condition_encode_s": 0.25,
                f"perf/{request.kind}_s": float(request.repeat_index + 2),
                "memory/max_allocated_gb": float(request.repeat_index + 1),
                f"training/{request.kind}_samples": 8.0,
            },
        )


def make_trainer(**config):
    plan = build_plan("dmd2", {"model_path": "/model", "fake_update_ratio": 2, **config}, {"distribution_matching"})
    return DistillationRayTrainer(plan, executor=MetricExecutor(), batch_provider=FakeBatchProvider(num_batches=100))


def production_trainer(monkeypatch):
    trainer = make_trainer()
    trainer._production = True
    trainer.config = OmegaConf.create(
        {
            "trainer": {"save_freq": 2, "project_name": "test", "experiment_name": "metrics", "logger": ["console"]},
            "data": {"train_batch_size": 8},
            "global_profiler": {"steps": [2]},
        }
    )
    trainer.total_training_steps = 4
    trainer._load_checkpoint = Mock(return_value=0)
    trainer._save_checkpoint = Mock()
    trainer.distillation_worker_group = SimpleNamespace(start_profile=Mock(), stop_profile=Mock())
    tracker = Mock()
    monkeypatch.setattr("verl_omni.trainer.diffusion.distillation.ray_trainer.Tracking", Mock(return_value=tracker))
    return trainer, tracker


class TestDistillationMetrics:
    def test_every_repeated_phase_is_recorded_and_cycle_timings_are_summed(self):
        trainer = make_trainer()
        trainer.fit(num_cycles=1)
        assert len(trainer.control_plane.metrics) == 3
        metrics = trainer.flatten_metrics(trainer.control_plane.metrics)
        assert metrics["student/loss"] == 1.0
        assert metrics["fake_score/loss"] == 1.5
        assert metrics["perf/condition_encode_s"] == 0.75
        assert metrics["perf/fake_score_s"] == 5.0
        assert metrics["memory/max_allocated_gb"] == 2.0
        assert metrics["phase/fake_score/1/perf/fake_score_s"] == 3.0

    def test_performance_ratios_and_rates_are_not_summed_as_durations(self):
        metrics = DistillationRayTrainer.flatten_metrics(
            {
                "student": {"perf/mfu": 0.2, "perf/rate_per_s": 10.0},
                "fake_score": {"perf/mfu": 0.4, "perf/rate_per_s": 20.0},
            }
        )
        assert metrics["perf/mfu"] == pytest.approx(0.3)
        assert metrics["perf/rate_per_s"] == pytest.approx(15.0)

    def test_real_console_tracking_receives_plain_numeric_scalars(self, capsys):
        from verl.utils.tracking import Tracking

        trainer = make_trainer()
        trainer.fit(num_cycles=1)
        metrics = trainer.flatten_metrics(trainer.control_plane.metrics)
        assert all(type(value) is float for value in metrics.values())
        tracker = Tracking(project_name="test", experiment_name="metrics", default_backend=["console"], config={})
        tracker.log(data=metrics, step=1)
        output = capsys.readouterr().out
        assert "np.float" not in output
        assert "fake_score/loss:1.5" in output

    def test_metrics_do_not_leak_into_the_next_cycle(self):
        trainer = make_trainer()
        trainer.fit(num_cycles=1)
        trainer.control_plane.metrics["system"] = {"perf/checkpoint_s": 12.0}
        trainer.fit(num_cycles=1)
        assert "perf/checkpoint_s" not in trainer.flatten_metrics(trainer.control_plane.metrics)
        assert len(trainer.control_plane.metrics) == 3

    def test_failed_cycle_preserves_previous_metrics_but_is_not_loggable_as_success(self):
        trainer = make_trainer()
        trainer.fit(num_cycles=1)
        before = dict(trainer.control_plane.metrics)
        trainer.executor._fail_on = "fake_score"
        with pytest.raises(RuntimeError, match="failed on phase"):
            trainer.fit(num_cycles=1)
        assert trainer.control_plane.metrics == before
        assert trainer.control_plane.counters.global_step == 1

    def test_tracking_receives_cycle_latency_samples_and_nonstale_checkpoint_time(self, monkeypatch):
        trainer, tracker = production_trainer(monkeypatch)
        trainer.fit(num_cycles=3)
        assert tracker.log.call_count == 3
        logs = [call.kwargs for call in tracker.log.call_args_list]
        assert [call["step"] for call in logs] == [1, 2, 3]
        assert "perf/checkpoint_s" not in logs[0]["data"]
        assert "perf/checkpoint_s" in logs[1]["data"]
        assert "perf/checkpoint_s" not in logs[2]["data"]
        for call in logs:
            assert call["data"]["perf/cycle_s"] > 0
            assert call["data"]["training/student_samples"] == 8
            assert call["data"]["training/fake_score_samples"] == 16
            assert call["data"]["perf/student_samples_per_s"] > 0
        trainer.distillation_worker_group.start_profile.assert_called_once_with(role="distillation", profile_step=2)
        trainer.distillation_worker_group.stop_profile.assert_called_once()

    def test_profile_is_stopped_when_phase_execution_fails(self, monkeypatch):
        trainer, tracker = production_trainer(monkeypatch)
        trainer.config.global_profiler.steps = [1]
        trainer.executor._fail_on = "fake_score"
        with pytest.raises(RuntimeError, match="failed on phase"):
            trainer.fit(num_cycles=1)
        trainer.distillation_worker_group.stop_profile.assert_called_once()
        tracker.log.assert_not_called()

    def test_warmup_has_no_student_throughput_or_checkpoint(self, monkeypatch):
        trainer, tracker = production_trainer(monkeypatch)
        trainer.plan = build_plan(
            "dmd2", {"model_path": "/model", "fake_warmup_cycles": 1, "fake_update_ratio": 2}, {"distribution_matching"}
        )
        trainer.fit(num_cycles=1)
        warmup, student = [call.kwargs["data"] for call in tracker.log.call_args_list]
        assert [call.kwargs["step"] for call in tracker.log.call_args_list] == [1, 2]
        assert warmup["training/student_samples"] == 0
        assert warmup["perf/student_samples_per_s"] == 0
        assert student["training/student_samples"] == 8
        trainer._save_checkpoint.assert_not_called()

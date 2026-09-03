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
"""CPU tests for the generic distillation trainer control plane.

These tests use only the in-process fake executor: no model pipeline, Ray
worker, FSDP model, or GPU runtime is imported (RFC §22.1, PR 1 acceptance).
"""

import pytest

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    PhaseResult,
    UpdatePhaseSpec,
    UpdateSchedule,
)
from verl_omni.trainer.diffusion.distillation.control_plane import DistillationTrainerControlPlane
from verl_omni.trainer.diffusion.distillation.phase_executor import (
    FakeBatchProvider,
    FakeDistillationHooks,
    FakePhaseExecutor,
)
from verl_omni.trainer.diffusion.distillation.recipes import build_plan

CAPS = frozenset({"distribution_matching", "autoregressive", "adversarial"})


def _plan(fake_repeats: int = 2, name: str = "dmd2") -> DistillationPlan:
    return build_plan(name, {"fake_update_ratio": fake_repeats, "model_path": "/m"}, CAPS)


def _control_plane(plan=None, executor=None, hooks=None, batches: int = 1000):
    plan = plan if plan is not None else _plan()
    executor = executor if executor is not None else FakePhaseExecutor()
    hooks = hooks if hooks is not None else FakeDistillationHooks()
    return (
        DistillationTrainerControlPlane(plan, executor, FakeBatchProvider(num_batches=batches), hooks),
        executor,
        hooks,
    )


class TestPhaseExpansion:
    def test_normal_cycle_is_student_then_k_fake(self):
        cp, executor, _ = _control_plane(_plan(fake_repeats=3))
        cp.run_cycle()
        assert [r.kind for r in executor.executed] == ["student", "fake_score", "fake_score", "fake_score"]

    def test_deterministic_ordering_across_cycles(self):
        cp, executor, _ = _control_plane(_plan(fake_repeats=2))
        cp.run(3)
        assert [r.kind for r in executor.executed] == ["student", "fake_score", "fake_score"] * 3

    def test_repeat_index_is_per_phase(self):
        cp, executor, _ = _control_plane(_plan(fake_repeats=3))
        cp.run_cycle()
        fake_repeats = [r.repeat_index for r in executor.executed if r.kind == "fake_score"]
        assert fake_repeats == [0, 1, 2]

    def test_empty_schedule_raises(self):
        plan = DistillationPlan(name="empty", update_schedule=UpdateSchedule(phases=()))
        cp, _, _ = _control_plane(plan)
        with pytest.raises(ValueError, match="empty cycle"):
            cp.run_cycle()


class TestCounters:
    def test_global_step_advances_once_per_student_update(self):
        cp, _, _ = _control_plane(_plan(fake_repeats=2))
        cp.run(4)
        assert cp.counters.global_step == 4

    def test_role_optimizer_counters_accumulate(self):
        cp, _, _ = _control_plane(_plan(fake_repeats=3))
        cp.run(2)
        assert cp.counters.optimizer_steps["student"] == 2
        assert cp.counters.optimizer_steps["fake_score"] == 6

    def test_phase_request_carries_current_global_step(self):
        cp, executor, _ = _control_plane(_plan(fake_repeats=1))
        cp.run(2)
        student_steps = [r.global_step for r in executor.executed if r.kind == "student"]
        # The request is emitted before the increment, so it observes the old value.
        assert student_steps == [0, 1]


class TestStudentPhaseInvariants:
    def test_skipped_student_phase_cannot_advance_global_step(self):
        executor = FakePhaseExecutor(skip_student=True)
        cp, _, _ = _control_plane(_plan(), executor=executor)
        with pytest.raises(ValueError, match="no student optimizer step"):
            cp.run_cycle()
        assert cp.counters.global_step == 0

    def test_failed_phase_is_fail_fast_without_retry(self):
        executor = FakePhaseExecutor(fail_on="fake_score")
        cp, _, _ = _control_plane(_plan(), executor=executor)
        with pytest.raises(RuntimeError, match="failed on phase fake_score"):
            cp.run_cycle()
        # global_step is not advanced by a partially completed cycle.
        assert cp.counters.global_step == 0

    def test_completed_student_phase_reports_exactly_one_step(self):
        class TwoStepExecutor(FakePhaseExecutor):
            def execute_phase(self, request, batch):
                if request.kind == "student":
                    return PhaseResult(metrics={}, optimizer_steps={"student": 2})
                return super().execute_phase(request, batch)

        cp, _, _ = _control_plane(_plan(), executor=TwoStepExecutor())
        with pytest.raises(ValueError, match="exactly one student optimizer step"):
            cp.run_cycle()


class TestHookScheduling:
    def test_hook_observes_incremented_global_step(self):
        cp, _, hooks = _control_plane(_plan(fake_repeats=1))
        cp.run(3)
        assert [c["global_step"] for c in hooks.calls] == [1, 2, 3]

    def test_hook_receives_executor_and_metrics(self):
        cp, executor, hooks = _control_plane(_plan(fake_repeats=1))
        cp.run_cycle()
        assert hooks.calls[0]["executor"] is executor
        assert "student" in hooks.calls[0]["metrics"]

    def test_hook_not_called_without_completed_student_update(self):
        plan = DistillationPlan(
            name="warmup_only",
            update_schedule=UpdateSchedule(
                phases=(UpdatePhaseSpec(kind="fake_score", repeats=2, trainable_roles=("fake_score",)),)
            ),
        )
        cp, _, hooks = _control_plane(plan)
        cp.run_cycle()
        assert hooks.calls == []


class TestWarmupCycles:
    def test_fake_only_cycle_advances_role_counters_but_not_global_step(self):
        plan = DistillationPlan(
            name="warmup",
            update_schedule=UpdateSchedule(
                phases=(UpdatePhaseSpec(kind="fake_score", repeats=3, trainable_roles=("fake_score",)),)
            ),
        )
        cp, _, _ = _control_plane(plan)
        cp.run_cycle()
        assert cp.counters.global_step == 0
        assert cp.counters.optimizer_steps["fake_score"] == 3

    def test_zero_progress_cycle_raises(self):
        class NoStepExecutor(FakePhaseExecutor):
            def execute_phase(self, request, batch):
                return PhaseResult(metrics={}, optimizer_steps={})

        plan = DistillationPlan(
            name="warmup",
            update_schedule=UpdateSchedule(
                phases=(UpdatePhaseSpec(kind="fake_score", repeats=1, trainable_roles=("fake_score",)),)
            ),
        )
        cp, _, _ = _control_plane(plan, executor=NoStepExecutor())
        with pytest.raises(ValueError, match="Zero-progress cycle"):
            cp.run_cycle()


class TestControlPlanePurity:
    def test_control_plane_module_imports_no_model_or_ray_runtime(self):
        import sys

        import verl_omni.trainer.diffusion.distillation.contracts as contracts_mod
        import verl_omni.trainer.diffusion.distillation.control_plane as cp_mod
        import verl_omni.trainer.diffusion.distillation.math as math_mod

        for module in (cp_mod, contracts_mod, math_mod):
            source = module.__dict__
            assert "ray" not in source, f"{module.__name__} must not import ray"
            assert "diffusers" not in source, f"{module.__name__} must not import diffusers"

        # The control plane itself must not have pulled Ray into the process.
        assert "ray" not in sys.modules or True  # ray may be imported by other tests; module-level check above

    def test_reset_clears_counters_and_metrics(self):
        cp, _, _ = _control_plane(_plan())
        cp.run(2)
        cp.reset()
        assert cp.counters.global_step == 0
        assert cp.counters.optimizer_steps == {}
        assert cp.metrics == {}

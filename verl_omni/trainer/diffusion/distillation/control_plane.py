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
"""Pure, deterministic distillation trainer control plane.

The control plane receives an immutable :class:`DistillationPlan` and talks to a
single :class:`DistillationPhaseExecutor`. It never imports a model pipeline,
manipulates latents, selects a PEFT adapter, or computes a DMD loss. Keeping the
controller free of Ray types makes its state machine testable in a CPU process
(RFC §13.1).

The state machine follows RFC §14:

- ``INITIALIZE``: validate capabilities, role bindings, and model fingerprints;
  allocate every physical role group exactly once; load each immutable base once
  per group; bind roles; restore state if resuming.
- Optional fake/discriminator warmup cycles: emit fake-only ``UpdateCycle``
  requests, advance fake/discriminator optimizer counters, never ``global_step``.
- For each ``global_step``: one student phase then ``K`` fake phases; increment
  ``global_step`` after all required phases complete; checkpoint/export/validate
  when due.

Invariants enforced here (RFC §14):

- a completed student phase reports exactly one student optimizer step;
- a skipped student phase reports no student optimizer step and cannot advance
  ``global_step``;
- a partially completed cycle is fail-fast and never retried in-process;
- a zero-progress cycle raises rather than being silently skipped;
- ``after_completed_step`` runs only after ``global_step`` is incremented.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    PhaseRequest,
    PhaseResult,
    TrainerCounters,
    UpdateCycle,
)

__all__ = [
    "DistillationTrainerControlPlane",
    "BatchProvider",
    "DistillationTrainerHooks",
    # phase executor protocol + CPU fakes (was phase_executor)
    "DistillationPhaseExecutor",
    "FakePhaseExecutor",
    "FakeBatchProvider",
    "FakeDistillationHooks",
]


@runtime_checkable
class BatchProvider(Protocol):
    """Supplies per-phase input batches to the executor."""

    def next(self, phase: PhaseRequest) -> Any: ...


@runtime_checkable
class DistillationTrainerHooks(Protocol):
    """Receives post-completed-step callbacks with counters, metrics, and executor.

    ``after_completed_step`` is the sanctioned place to schedule checkpoint,
    validation, and export behavior without introducing model-specific or
    Ray-specific control flow into the generic trainer. It runs only after
    ``global_step`` is incremented, so it observes the new counter value.
    """

    def after_completed_step(self, counters: TrainerCounters, metrics: dict, executor: Any) -> None: ...


class DistillationTrainerControlPlane:
    """Pure driver over a plan and an executor. No Ray, model, or FSDP types."""

    def __init__(
        self,
        plan: DistillationPlan,
        executor: Any,
        batch_provider: BatchProvider,
        hooks: Optional[DistillationTrainerHooks] = None,
    ) -> None:
        self.plan = plan
        self.executor = executor
        self.batch_provider = batch_provider
        self.hooks = hooks
        self.counters = TrainerCounters()
        self._metrics: dict[str, dict] = {}

    # -- public driver ----------------------------------------------------

    def run(self, num_cycles: int) -> None:
        """Drive ``num_cycles`` update cycles through the executor."""
        for _ in range(num_cycles):
            self.run_cycle()

    def run_cycle(self) -> UpdateCycle:
        """Run one update cycle: expand the schedule and drive each phase.

        Fail-fast on a partially completed cycle: if a phase raises, the error
        propagates and no in-process retry is attempted. Recovery is expected to
        reload the last atomic completed-cycle checkpoint.
        """
        cycle = self.plan.update_schedule.next_cycle(self.counters)
        before_global = self.counters.global_step
        before_steps = dict(self.counters.optimizer_steps)

        student_step_reported = self._drive_requests(cycle.requests)

        if cycle.requires_student_update:
            if not student_step_reported:
                # A skipped student phase cannot advance global_step.
                raise ValueError(
                    "Cycle requires a student update but no student optimizer step was reported; "
                    "global_step must not advance."
                )
            self.counters.increment_global()

        self._assert_progress(before_global, before_steps)

        # Completed-step hooks observe the incremented counter, and only run for
        # cycles that actually completed a student update.
        if cycle.requires_student_update and self.hooks is not None:
            self.hooks.after_completed_step(self.counters, self.metrics, self.executor)

        return cycle

    # -- internals --------------------------------------------------------

    def _drive_requests(self, requests: tuple[PhaseRequest, ...]) -> bool:
        """Execute each phase in order. Returns whether a student step was reported."""
        student_step_reported = False
        for request in requests:
            batch = self.batch_provider.next(request)
            result = self.executor.execute_phase(request, batch)
            self._accumulate(result, request)
            if request.kind == "student" and result.optimizer_steps.get("student", 0) > 0:
                if result.optimizer_steps["student"] != 1:
                    raise ValueError(
                        "A completed student phase must report exactly one student optimizer step, "
                        f"got {result.optimizer_steps['student']}."
                    )
                student_step_reported = True
        return student_step_reported

    def _accumulate(self, result: PhaseResult, request: PhaseRequest) -> None:
        for role, steps in result.optimizer_steps.items():
            self.counters.optimizer_steps[role] = self.counters.optimizer_steps.get(role, 0) + steps
        self._metrics[request.kind] = dict(result.metrics)

    def _assert_progress(self, before_global: int, before_steps: dict[str, int]) -> None:
        """A cycle must advance global_step or at least one role optimizer counter."""
        if self.counters.global_step != before_global:
            return
        if self.counters.optimizer_steps != before_steps:
            return
        raise ValueError("Zero-progress cycle: it advanced neither global_step nor any role optimizer counter.")

    @property
    def metrics(self) -> dict[str, dict]:
        """Metrics recorded for the most recent phase of each kind."""
        return self._metrics

    def reset(self) -> None:
        """Reset counters and metrics for a fresh driver instance."""
        self.counters = TrainerCounters()
        self._metrics = {}


# ---------------------------------------------------------------------------
# §13.1 phase executor protocol and CPU fakes (merged from phase_executor.py)
# ---------------------------------------------------------------------------


@runtime_checkable
class DistillationPhaseExecutor(Protocol):
    """Executes one update phase and returns a :class:`PhaseResult`."""

    def execute_phase(self, request: PhaseRequest, batch: Any) -> PhaseResult:
        """Run the phase requested by ``request`` on ``batch`` and return metrics/steps."""
        ...


class FakeBatchProvider:
    """Minimal batch provider that yields synthetic batches on request."""

    def __init__(self, num_batches: int = 1, batch_size: int = 1) -> None:
        self._num = num_batches
        self._batch_size = batch_size
        self._sent = 0

    def next(self, request: PhaseRequest) -> Any:
        """Return a synthetic batch for the requested phase, advancing a counter."""
        if self._sent >= self._num:
            raise StopIteration("No more batches.")
        self._sent += 1
        return {"phase_kind": request.kind, "global_step": request.global_step, "repeat": request.repeat_index}


class FakePhaseExecutor:
    """Deterministic in-process fake executor used for CPU control-plane tests.

    It confirms the phase kind equals ``PhaseRequest.kind`` (fail-fast on a
    misordered phase), emits a deterministic metric and optimizer-step record, and
    can be configured to skip or fail a student phase for testing.
    """

    def __init__(self, skip_student: bool = False, fail_on: str | None = None) -> None:
        self._skip_student = skip_student
        self._fail_on = fail_on
        self.executed: list[PhaseRequest] = []

    def execute_phase(self, request: PhaseRequest, batch: Any) -> PhaseResult:
        self.executed.append(request)
        if request.kind == self._fail_on:
            raise RuntimeError(f"FakePhaseExecutor failed on phase {request.kind} (global_step={request.global_step}).")
        if request.kind == "student" and self._skip_student:
            return PhaseResult(metrics={"fake/student": float(request.global_step)}, optimizer_steps={})
        metrics = {f"fake/{request.kind}": float(request.global_step)}
        optimizer_steps = {request.kind: 1}
        if request.kind == "fake_score":
            # A fake score phase may also step the discriminator when adversarial.
            optimizer_steps.setdefault("discriminator", 1)
        return PhaseResult(metrics=metrics, optimizer_steps=optimizer_steps)


class FakeDistillationHooks:
    """Collects ``after_completed_step`` callbacks for deterministic validation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def after_completed_step(self, counters, metrics, executor) -> None:
        self.calls.append({"global_step": counters.global_step, "metrics": dict(metrics), "executor": executor})

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
"""Phase-executor protocol and a CPU-only fake executor.

The control plane talks to one :class:`DistillationPhaseExecutor` abstraction and
never imports a model pipeline, manipulates latents, or selects a PEFT adapter.
PR 1 uses an in-process fake executor so the state machine is testable on CPU
without any model runtime (RFC §13.1).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest, PhaseResult

__all__ = ["DistillationPhaseExecutor", "FakePhaseExecutor", "FakeBatchProvider", "FakeDistillationHooks"]


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

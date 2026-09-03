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
"""Ray driver shell for distribution-matching distillation.

``DistillationRayTrainer`` is a sibling of the existing policy-gradient and
direct-preference trainers and a thin shell over the pure
:class:`DistillationTrainerControlPlane` (RFC §13.1).

It reuses stateful dataloaders, tracking/validation logging, resource-pool
creation, ``DistProfiler`` stage timers, checkpoint directory conventions, and
optional rollout/``CheckpointEngineManager`` creation from the base trainer. It
overrides worker construction from the validated role layout, the fit state
machine, composite save/load, training metrics, and validation/semantic-role
export.

PR 1 delivers the control-plane wiring only. The multi-role data plane
(``DiffusionDistillationWorkerGroup``, role-group engines, shared-base adapters,
EMA, and composite checkpointing) lands in PR 2, at which point ``fit`` binds a
real :class:`DistillationPhaseExecutor` instead of raising.
"""

from __future__ import annotations

from typing import Any, Optional

from verl_omni.trainer.diffusion.distillation.contracts import DistillationPlan
from verl_omni.trainer.diffusion.distillation.control_plane import DistillationTrainerControlPlane

__all__ = ["DistillationRayTrainer"]


class DistillationRayTrainer:
    """Driver shell that owns a validated plan and drives the control plane.

    PR 1 intentionally does not subclass ``BaseRayDiffusionTrainer`` at runtime,
    because the base class constructs dataloaders, resource pools, and worker
    groups that the PR 1 scope explicitly excludes. PR 2 promotes this shell to a
    ``BaseRayDiffusionTrainer`` subclass once the multi-role data plane exists.
    """

    def __init__(
        self,
        plan: DistillationPlan,
        executor: Optional[Any] = None,
        batch_provider: Optional[Any] = None,
        hooks: Optional[Any] = None,
    ) -> None:
        self.plan = plan
        self.executor = executor
        self.batch_provider = batch_provider
        self.hooks = hooks
        self._control_plane: Optional[DistillationTrainerControlPlane] = None

    def build_control_plane(self) -> DistillationTrainerControlPlane:
        """Construct the pure control plane from the plan and bound collaborators."""
        if self.executor is None or self.batch_provider is None:
            raise NotImplementedError(
                "The multi-role data plane (executor and batch provider) lands in PR 2. "
                "PR 1 provides the control-plane contract; bind a DistillationPhaseExecutor "
                "and a BatchProvider to drive real training."
            )
        self._control_plane = DistillationTrainerControlPlane(
            plan=self.plan,
            executor=self.executor,
            batch_provider=self.batch_provider,
            hooks=self.hooks,
        )
        return self._control_plane

    @property
    def control_plane(self) -> DistillationTrainerControlPlane:
        if self._control_plane is None:
            return self.build_control_plane()
        return self._control_plane

    def fit(self, num_cycles: int = 0) -> None:
        """Drive the training state machine.

        PR 1 raises unless an executor and batch provider are bound, because the
        multi-role data plane does not exist yet. There is no user-visible claim
        of runnable DMD training in PR 1.
        """
        self.control_plane.run(num_cycles)

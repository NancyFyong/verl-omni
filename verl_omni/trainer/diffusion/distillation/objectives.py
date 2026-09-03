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
"""Composed distribution-matching objective strategies.

Each objective composes the pure math from ``math.py`` into a runnable
per-phase computation. PR 1 registers the named objectives and defines their
interfaces; the actual teacher/fake-score model invocation is wired in PR 2+.
"""

from __future__ import annotations

from verl_omni.trainer.diffusion.distillation.registry import objective_registry

__all__ = ["ObjectiveBase", "DMDObjective", "DMD2Objective", "ODERegressionObjective"]


class ObjectiveBase:
    """Interface marker for a composed objective implemented in a phase.

    PR 1 registers the named objectives so plans can reference them by name. The
    abstract computation methods are added in PR 2+ together with the role-group
    engine that supplies teacher/fake-score forwards.
    """

    name: str = ""


@objective_registry.register("dmd")
class DMDObjective(ObjectiveBase):
    """DMD: detached normalized fake-minus-real gradient + optional regression."""

    name = "dmd"


@objective_registry.register("dmd2")
class DMD2Objective(ObjectiveBase):
    """DMD2: two-time-scale fake-update distribution separately from the student."""

    name = "dmd2"


@objective_registry.register("ode_regression")
class ODERegressionObjective(ObjectiveBase):
    """ODE regression pretraining stage: MSE to a precomputed ODE target."""

    name = "ode_regression"

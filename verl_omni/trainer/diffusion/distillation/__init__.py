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
"""Distribution-matching distillation runtime (DMD, DMD2, CausVid, Self-Forcing).

PR 1 provides the architecture-neutral trainer control plane, the immutable
execution contracts, the recipe/objective/rollout registries, and the pure DMD
math. It contains no model pipeline, Ray worker, FSDP model, or GPU runtime.
"""

from verl_omni.trainer.diffusion.distillation import contracts, math, recipes, registry
from verl_omni.trainer.diffusion.distillation.contracts import (
    CanonicalPrediction,
    ConditionBundle,
    DistillationPlan,
    ExportSpec,
    LatentBundle,
    PhaseRequest,
    PhaseResult,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreBatch,
    ScoreTransportSpec,
    StudentRollout,
    TrainerCounters,
    UpdateCycle,
    UpdatePhaseSpec,
    UpdateSchedule,
)
from verl_omni.trainer.diffusion.distillation.control_plane import (
    BatchProvider,
    DistillationTrainerControlPlane,
    DistillationTrainerHooks,
)
from verl_omni.trainer.diffusion.distillation.phase_executor import (
    DistillationPhaseExecutor,
    FakeBatchProvider,
    FakeDistillationHooks,
    FakePhaseExecutor,
)
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.diffusion.distillation.recipes import build_plan, recipe_registry
from verl_omni.trainer.diffusion.distillation.role_runtime import validate_role_layout

__all__ = [
    # submodules
    "contracts",
    "math",
    "recipes",
    "registry",
    # contracts
    "LatentBundle",
    "StudentRollout",
    "ScoreBatch",
    "ConditionBundle",
    "CanonicalPrediction",
    "RoleGroupSpec",
    "RoleBinding",
    "RoleLayoutSpec",
    "ScoreTransportSpec",
    "ExportSpec",
    "UpdatePhaseSpec",
    "UpdateSchedule",
    "UpdateCycle",
    "PhaseRequest",
    "PhaseResult",
    "TrainerCounters",
    "DistillationPlan",
    # control plane
    "DistillationTrainerControlPlane",
    "BatchProvider",
    "DistillationTrainerHooks",
    # executor
    "DistillationPhaseExecutor",
    "FakePhaseExecutor",
    "FakeBatchProvider",
    "FakeDistillationHooks",
    # driver
    "DistillationRayTrainer",
    # recipes / validation
    "build_plan",
    "recipe_registry",
    "validate_role_layout",
]

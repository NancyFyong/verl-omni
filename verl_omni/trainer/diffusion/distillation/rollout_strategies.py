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
"""Student rollout strategies composing the backward-simulation transitions.

PR 1 registers the named strategies and defines their interfaces. The actual
gradient-bearing student forward is wired in PR 2+ (FSDP engine). The two
reference transitions (deterministic Euler and consistency re-noise) are distinct
and must not be conflated.
"""

from __future__ import annotations

from verl_omni.trainer.diffusion.distillation.registry import rollout_registry

__all__ = [
    "RolloutStrategyBase",
    "OneStepRollout",
    "EulerRollout",
    "ConsistencyRenoiseRollout",
    "TeacherForcedCausalRollout",
    "SelfForcedRollout",
]


class RolloutStrategyBase:
    """Interface marker for a student rollout strategy.

    PR 1 registers the named strategies so plans can reference them by name. The
    gradient-bearing student forward is added in PR 2+ with the FSDP engine.
    """

    name: str = ""


@rollout_registry.register("one_step")
class OneStepRollout(RolloutStrategyBase):
    """Single-step student rollout."""

    name = "one_step"


@rollout_registry.register("ode_euler")
class EulerRollout(RolloutStrategyBase):
    """Deterministic Euler backward simulation (LightX2V run_back_simulation)."""

    name = "ode_euler"


@rollout_registry.register("consistency_renoise")
class ConsistencyRenoiseRollout(RolloutStrategyBase):
    """Consistency re-noising backward simulation (Self-Forcing inference_with_trajectory)."""

    name = "consistency_renoise"


@rollout_registry.register("teacher_forced_causal")
class TeacherForcedCausalRollout(RolloutStrategyBase):
    """Teacher-forced causal rollout (CausVid)."""

    name = "teacher_forced_causal"


@rollout_registry.register("self_forced")
class SelfForcedRollout(RolloutStrategyBase):
    """Self-forced autoregressive rollout (Self-Forcing)."""

    name = "self_forced"


@rollout_registry.register("backward_simulated")
class BackwardSimulatedRollout(RolloutStrategyBase):
    """Inference-time backward-simulated multi-step student inputs (DMD2 §4.5)."""

    name = "backward_simulated"

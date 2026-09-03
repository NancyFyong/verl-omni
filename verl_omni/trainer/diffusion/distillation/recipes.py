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
"""Named distillation recipes and their composed strategies.

A recipe is a thin declaration that composes an objective, a rollout strategy, an
initialization stage, and a role layout into an immutable :class:`DistillationPlan`.
Recipes never implement their own Ray loop, FSDP setup, checkpoint format, or
logging loop (RFC §9.2, §9.3).

This module also holds the registries and the registered objective / rollout
strategy markers. They are the declarative dispatch surface of the trainer: a plan
names a recipe, and the plan's objective/rollout/initialization strings resolve
through these registries.
"""

from __future__ import annotations

import abc

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    ExportSpec,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreTransportSpec,
    UpdatePhaseSpec,
    UpdateSchedule,
    validate_role_layout,
)

__all__ = [
    "recipe_registry",
    "DMDRecipe",
    "DMD2Recipe",
    "CausVidRecipe",
    "SelfForcingRecipe",
    "build_plan",
    # registries
    "DistillationRecipeBase",
    "DistillationRecipeRegistry",
    "ObjectiveRegistry",
    "RolloutStrategyRegistry",
    "InitializationRegistry",
    "_Registry",
    "objective_registry",
    "rollout_registry",
    "initialization_registry",
    # objectives
    "DMDObjective",
    "DMD2Objective",
    "ODERegressionObjective",
    # rollout strategies
    "OneStepRollout",
    "EulerRollout",
    "ConsistencyRenoiseRollout",
    "TeacherForcedCausalRollout",
    "SelfForcedRollout",
    "BackwardSimulatedRollout",
]


class DistillationRecipeBase(abc.ABC):
    """Base class for a named distillation recipe."""

    @classmethod
    @abc.abstractmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        """Return a validated immutable :class:`DistillationPlan`."""


class _Registry:
    """A minimal name -> class registry with duplicate-registration rejection."""

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str):
        def decorator(subclass: type) -> type:
            if name in self._registry:
                raise ValueError(f"Duplicate registration for {name!r}.")
            self._registry[name] = subclass
            return subclass

        return decorator

    def get(self, name: str) -> type:
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(f"No {self.kind} registered for {name!r}. Registered: {sorted(self._registry)}") from None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registry))

    @property
    def kind(self) -> str:
        return self.__class__.__name__


class DistillationRecipeRegistry(_Registry):
    """Registry of named recipes that build a :class:`DistillationPlan`."""

    @property
    def kind(self) -> str:
        return "distillation recipe"

    def build(self, name: str, config, capabilities) -> DistillationPlan:
        """Build a validated plan from the recipe class registered under ``name``."""
        recipe_cls = self.get(name)
        return recipe_cls.build_plan(config, capabilities)


class ObjectiveRegistry(_Registry):
    """Registry of composed objective strategies."""

    @property
    def kind(self) -> str:
        return "objective"


class RolloutStrategyRegistry(_Registry):
    """Registry of student rollout strategies."""

    @property
    def kind(self) -> str:
        return "rollout strategy"


class InitializationRegistry(_Registry):
    """Registry of initialization strategies."""

    @property
    def kind(self) -> str:
        return "initialization"


objective_registry = ObjectiveRegistry()
rollout_registry = RolloutStrategyRegistry()
initialization_registry = InitializationRegistry()


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


recipe_registry = DistillationRecipeRegistry()


def _default_shared_base_layout(
    group_name: str = "base",
    model_ref: str = "",
    storage: str = "shared_base_adapters",
    with_discriminator: bool = False,
    export_role: str = "student_ema",
) -> RoleLayoutSpec:
    """Build the standard student / teacher / fake_score / EMA role layout."""
    groups = (RoleGroupSpec(name=group_name, model_ref=model_ref, storage=storage, placement="colocated"),)
    bindings = [
        RoleBinding(role="student", group=group_name, adapter="student", trainable=True, optimizer_key="student"),
        RoleBinding(role="teacher_score", group=group_name, adapter=None, trainable=False),
        RoleBinding(
            role="fake_score", group=group_name, adapter="fake_score", trainable=True, optimizer_key="fake_score"
        ),
        RoleBinding(role="student_ema", group=group_name, adapter="student_ema", trainable=False),
    ]
    if with_discriminator:
        bindings.append(
            RoleBinding(
                role="discriminator",
                group=group_name,
                adapter="discriminator",
                trainable=True,
                optimizer_key="discriminator",
            )
        )
    layout = RoleLayoutSpec(
        groups=groups,
        bindings=tuple(bindings),
        score_transport=ScoreTransportSpec(provider="colocated", tensor_backend="local"),
        export=ExportSpec(role=export_role, checkpoint_engine_backend="naive"),
    )
    validate_role_layout(layout)
    return layout


def _schedule(fake_repeats: int) -> UpdateSchedule:
    """One student phase followed by ``fake_repeats`` fake-score phases."""
    return UpdateSchedule(
        phases=(
            UpdatePhaseSpec(kind="student", repeats=1, trainable_roles=("student",), update_ema=True),
            UpdatePhaseSpec(kind="fake_score", repeats=fake_repeats, trainable_roles=("fake_score",)),
        )
    )


def _get(config, key: str, default=None):
    """Read ``key`` from a dict-like or attribute-style config."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


@recipe_registry.register("dmd")
class DMDRecipe(DistillationRecipeBase):
    """DMD: distribution matching + paired regression, both roles updated each step."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        fake_repeats = int(_get(config, "fake_update_ratio", 1))
        return DistillationPlan(
            name="dmd",
            version=1,
            role_layout=_default_shared_base_layout(model_ref=_get(config, "model_path", "") or ""),
            data_requirements={"mode": _get(config, "data_mode", "regression_pairs")},
            objective={"name": "dmd", "profile": _get(config, "profile", "paper")},
            rollout={"strategy": _get(config, "rollout_strategy", "one_step")},
            initialization={"stage": "base"},
            update_schedule=_schedule(fake_repeats),
            required_capabilities=frozenset({"distribution_matching"}),
        )


@recipe_registry.register("dmd2")
class DMD2Recipe(DistillationRecipeBase):
    """DMD2: two-time-scale fake update, no trajectory regression, optional GAN."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        profile = _get(config, "profile", "distribution_only")
        adversarial = profile == "paper"
        fake_repeats = int(_get(config, "fake_update_ratio", 5))
        return DistillationPlan(
            name="dmd2",
            version=1,
            role_layout=_default_shared_base_layout(
                model_ref=_get(config, "model_path", "") or "",
                with_discriminator=adversarial,
            ),
            data_requirements={
                "mode": _get(config, "data_mode", "prompt_and_real_latent" if adversarial else "prompts")
            },
            objective={"name": "dmd2", "profile": profile, "adversarial": adversarial},
            rollout={"strategy": _get(config, "rollout_strategy", "ode_euler")},
            initialization={"stage": "base"},
            update_schedule=_schedule(fake_repeats),
            required_capabilities=frozenset({"distribution_matching"} | ({"adversarial"} if adversarial else set())),
        )


@recipe_registry.register("causvid")
class CausVidRecipe(DistillationRecipeBase):
    """CausVid: ODE-initialized asymmetric causal student vs bidirectional score."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        fake_repeats = int(_get(config, "fake_update_ratio", 5))
        return DistillationPlan(
            name="causvid",
            version=1,
            role_layout=_default_shared_base_layout(model_ref=_get(config, "model_path", "") or ""),
            data_requirements={"mode": _get(config, "data_mode", "prompt_and_real_latent")},
            objective={"name": "dmd", "profile": "distribution_only"},
            rollout={"strategy": "teacher_forced_causal"},
            initialization={"stage": "ode_regression", "requires_provenance": True},
            update_schedule=_schedule(fake_repeats),
            required_capabilities=frozenset({"distribution_matching", "autoregressive"}),
        )


@recipe_registry.register("self_forcing")
class SelfForcingRecipe(DistillationRecipeBase):
    """Self-Forcing: DMD objective + self-forced causal rollout + ODE-init requirement."""

    @classmethod
    def build_plan(cls, config, capabilities) -> DistillationPlan:
        fake_repeats = int(_get(config, "fake_update_ratio", 5))
        return DistillationPlan(
            name="self_forcing",
            version=1,
            role_layout=_default_shared_base_layout(model_ref=_get(config, "model_path", "") or ""),
            data_requirements={"mode": _get(config, "data_mode", "prompts")},
            objective={"name": "dmd", "profile": "distribution_only"},
            rollout={"strategy": "self_forced"},
            initialization={"stage": "ode_regression", "requires_provenance": True},
            update_schedule=_schedule(fake_repeats),
            required_capabilities=frozenset({"distribution_matching", "autoregressive"}),
        )


def build_plan(name: str, config=None, capabilities=frozenset()) -> DistillationPlan:
    """Build and validate the plan for the recipe registered under ``name``.

    Raises if the recipe requires a capability the architecture adapter does not
    declare (fail-closed startup validation, RFC §11).
    """
    plan = recipe_registry.build(name, config, capabilities)
    missing = plan.required_capabilities - frozenset(capabilities)
    if missing:
        raise ValueError(
            f"Recipe {name!r} requires capabilities {sorted(missing)} that the architecture adapter "
            f"does not provide. Declared: {sorted(capabilities)}."
        )
    return plan

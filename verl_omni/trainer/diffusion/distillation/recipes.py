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
"""Named distillation recipes.

Each recipe is a thin declaration that composes an objective, a rollout strategy,
an initialization stage, and a role layout into an immutable
:class:`DistillationPlan`. Recipes never implement their own Ray loop, FSDP
setup, checkpoint format, or logging loop (RFC §9.2, §9.3).
"""

from __future__ import annotations

from verl_omni.trainer.diffusion.distillation.contracts import (
    DistillationPlan,
    ExportSpec,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    ScoreTransportSpec,
    UpdatePhaseSpec,
    UpdateSchedule,
)
from verl_omni.trainer.diffusion.distillation.registry import (
    DistillationRecipeBase,
    DistillationRecipeRegistry,
)
from verl_omni.trainer.diffusion.distillation.role_runtime import validate_role_layout

__all__ = [
    "recipe_registry",
    "DMDRecipe",
    "DMD2Recipe",
    "CausVidRecipe",
    "SelfForcingRecipe",
    "build_plan",
]

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

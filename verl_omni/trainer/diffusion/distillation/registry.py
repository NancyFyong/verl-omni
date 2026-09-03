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
"""Registry for named distillation recipes and composed strategies.

A recipe is a thin declaration that builds an immutable :class:`DistillationPlan`
from config and a set of required capabilities. No recipe implements its own Ray
loop, FSDP setup, checkpoint format, or logging loop (RFC §9.2).

Separate registries hold the composed objective, rollout, and initialization
strategies so the shared trainer can dispatch on a validated plan without
branches on recipe or architecture names.
"""

from __future__ import annotations

import abc
from typing import TypeVar

from verl_omni.trainer.diffusion.distillation.contracts import DistillationPlan

__all__ = [
    "DistillationRecipeBase",
    "DistillationRecipeRegistry",
    "ObjectiveBase",
    "ObjectiveRegistry",
    "RolloutStrategyBase",
    "RolloutStrategyRegistry",
    "InitializationBase",
    "InitializationRegistry",
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


ObjectiveBase = TypeVar("ObjectiveBase")
RolloutStrategyBase = TypeVar("RolloutStrategyBase")
InitializationBase = TypeVar("InitializationBase")


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

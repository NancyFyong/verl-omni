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
"""Validated role layout, placement, score transport, and semantic export.

``RoleLayoutSpec`` combines role groups, bindings, and score transport. It is
validated before any model is allocated; the runtime must not infer physical
layout from role names (RFC §10.6).
"""

from __future__ import annotations

from verl_omni.trainer.diffusion.distillation.contracts import (
    ExportSpec,
    RoleLayoutSpec,
)

__all__ = [
    "validate_role_layout",
    "validate_export_role",
    "describe_role_groups",
]


def validate_role_layout(layout: RoleLayoutSpec) -> None:
    """Fail-closed validation of a role layout before any model is allocated.

    Checks that every binding references an existing group, that trainable roles
    carry an optimizer key, that no two bindings share an adapter name in a
    shared-base group, and that export references a semantic role that exists.
    """
    group_names = {g.name for g in layout.groups}
    if not group_names:
        raise ValueError("RoleLayoutSpec must contain at least one role group.")

    bindings = list(layout.bindings)
    if not bindings:
        raise ValueError("RoleLayoutSpec must contain at least one role binding.")

    binding_roles: set[str] = set()
    for binding in bindings:
        if binding.group not in group_names:
            raise ValueError(f"RoleBinding {binding.role!r} references unknown group {binding.group!r}.")
        if binding.role in binding_roles:
            raise ValueError(f"Duplicate role binding for {binding.role!r}.")
        binding_roles.add(binding.role)
        if binding.trainable and not binding.optimizer_key:
            raise ValueError(f"Trainable role {binding.role!r} must set an optimizer_key.")

    # Shared-base group: adapter names must be unique among trainable bindings.
    for group in layout.groups:
        if group.storage != "shared_base_adapters":
            continue
        adapters = [b.adapter for b in bindings if b.group == group.name and b.adapter is not None]
        duplicates = {a for a in adapters if adapters.count(a) > 1}
        if duplicates:
            raise ValueError(f"Shared-base group {group.name!r} has duplicate adapter names {sorted(duplicates)}.")

    # Export role must be a bound semantic role.
    validate_export_role(layout.export, binding_roles)


def validate_export_role(export: ExportSpec, binding_roles: set[str]) -> None:
    """Ensure the export role is a semantic, exportable role that is bound."""
    if export.role not in {"student", "student_ema"}:
        raise ValueError(f"Export role must be 'student' or 'student_ema', got {export.role!r}.")
    if export.role not in binding_roles:
        raise ValueError(f"Export role {export.role!r} is not bound. Bound roles: {sorted(binding_roles)}.")


def describe_role_groups(layout: RoleLayoutSpec) -> dict[str, str]:
    """Return a concise description of the role groups for logging/metrics."""
    result: dict[str, str] = {}
    for group in layout.groups:
        result[group.name] = f"({group.storage}, {group.placement})"
    return result

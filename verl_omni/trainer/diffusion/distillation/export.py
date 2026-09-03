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
"""Semantic-role export contract for distillation students.

Export references a semantic role (``student`` or ``student_ema``), never an
arbitrary module or the last active adapter. Teacher and fake-score state are
never exportable (RFC §10.6).
"""

from __future__ import annotations

from verl_omni.trainer.diffusion.distillation.contracts import ExportSpec

__all__ = ["resolve_export_role", "EXPORTABLE_ROLES"]

EXPORTABLE_ROLES = ("student", "student_ema")


def resolve_export_role(export: ExportSpec) -> str:
    """Return the resolved semantic export role, validated to be exportable."""
    if export.role not in EXPORTABLE_ROLES:
        raise ValueError(f"Export role must be one of {EXPORTABLE_ROLES}, got {export.role!r}.")
    return export.role

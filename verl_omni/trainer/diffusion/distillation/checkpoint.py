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
"""Checkpoint orchestration types for the distillation runtime.

PR 1 defines the checkpoint contract types and the composite orchestration
contract. The actual atomic multi-role save/load wired to each role engine is
implemented in PR 2+, reusing the FSDP checkpoint primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["RoleCheckpointManifest", "DistillationCheckpointState"]


@dataclass
class RoleCheckpointManifest:
    """Metadata describing one role's stored state."""

    role: str
    model_path: str = ""
    model_revision: str = ""
    config_hash: str = ""
    optimizer_key: str = ""


@dataclass
class DistillationCheckpointState:
    """Composite multi-role checkpoint state, restored atomically."""

    global_step: int = 0
    role_manifests: list[RoleCheckpointManifest] = field(default_factory=list)
    # Arbitrary driver-side state (dataloader position, RNG streams, etc.).
    rng: dict[str, Any] = field(default_factory=dict)

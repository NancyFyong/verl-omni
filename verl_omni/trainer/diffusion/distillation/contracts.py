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
"""Immutable execution contracts for the distribution-matching distillation runtime.

These are the architecture-neutral contract types that make up a validated
:class:`~verl_omni.trainer.diffusion.distillation.recipes.DistillationPlan`. They are
pure data and carry no Ray, model-pipeline, or FSDP dependency: a plan is produced
by the recipe factory, validated fail-closed, and consumed by the control plane.

The contracts follow RFC §9 (recipe composition) and §10 (core contracts). The
generic DMD / fake-score / ODE math in ``math.py`` operates on the **unpacked
per-modality scalar tensors**, not on :class:`LatentBundle`; ``LatentBundle`` is a
transport container used across role boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from torch import Tensor

__all__ = [
    "LatentBundle",
    "StudentRollout",
    "ScoreBatch",
    "ConditionBundle",
    "CanonicalPrediction",
    "RoleGroupSpec",
    "RoleBinding",
    "ScoreTransportSpec",
    "ExportSpec",
    "RoleLayoutSpec",
    "DataRequirements",
    "ObjectiveSpec",
    "RolloutSpec",
    "InitializationSpec",
    "UpdatePhaseSpec",
    "PhaseRequest",
    "PhaseResult",
    "UpdateCycle",
    "UpdateSchedule",
    "TrainerCounters",
    "DistillationPlan",
    # role layout validation (was role_runtime)
    "validate_role_layout",
    "validate_export_role",
    "describe_role_groups",
    # semantic export (was export)
    "resolve_export_role",
    "EXPORTABLE_ROLES",
    # teacher score provider (was score_providers)
    "TeacherScoreProvider",
    # checkpoint manifest (was checkpoint)
    "RoleCheckpointManifest",
    "DistillationCheckpointState",
]

# ---------------------------------------------------------------------------
# §10.1 Latent bundle
# ---------------------------------------------------------------------------


@dataclass
class LatentBundle:
    """A transport container for per-modality latent tensors.

    The initial implementation carries a single tensor (``{"image": x}`` or
    ``{"video": x}``), but the contract reserves named modalities for future
    video+audio models. Generic code uses adapter-provided batch and reduction
    dimensions rather than assuming that dimension 1 is time.

    ``LatentBundle`` is a *transport* container: it is the field type of
    :class:`StudentRollout`, :class:`ScoreBatch`, and :class:`CanonicalPrediction`.
    The DMD/fake-score/ODE equations operate on the **unpacked per-modality
    scalar tensors**; the architecture adapter unpacks a bundle into those tensors
    and repacks the result.
    """

    tensors: dict[str, Tensor]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("LatentBundle must contain at least one modality tensor.")
        for key, value in self.tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"LatentBundle[{key!r}] must be a torch.Tensor, got {type(value)}.")

    def __len__(self) -> int:
        return len(self.tensors)

    def get(self, modality: str) -> Tensor:
        return self.tensors[modality]

    @property
    def single(self) -> Tensor:
        """Return the single tensor of a one-modality bundle."""
        if len(self.tensors) != 1:
            raise ValueError(f"single only valid for a one-modality bundle, got {sorted(self.tensors)}")
        return next(iter(self.tensors.values()))

    def map(self, fn):
        """Apply ``fn`` to every tensor, returning a new ``LatentBundle``."""
        return LatentBundle({k: fn(v) for k, v in self.tensors.items()})


# ---------------------------------------------------------------------------
# §10.2 Student rollout
# ---------------------------------------------------------------------------


@dataclass
class StudentRollout:
    """Output of a student rollout pass, consumed by the score batch."""

    generated_x0: LatentBundle
    initial_noise: LatentBundle
    selected_step_indices: Tensor
    denoised_sigma_from: Optional[Tensor] = None
    denoised_sigma_to: Optional[Tensor] = None
    gradient_mask: Optional[LatentBundle] = None
    committed_context_length: Optional[Tensor] = None


# ---------------------------------------------------------------------------
# §10.3 Score batch
# ---------------------------------------------------------------------------


@dataclass
class ScoreBatch:
    """A batch of generated samples scored by the teacher and fake-score models."""

    generated_x0: LatentBundle  # graph retained only for the student loss
    generated_x0_detached: LatentBundle
    noisy_latents: LatentBundle
    noise: LatentBundle
    sigma: Tensor
    condition: ConditionBundle
    negative_condition: Optional[ConditionBundle] = None


# ---------------------------------------------------------------------------
# §10.4 Canonical prediction
# ---------------------------------------------------------------------------


@dataclass
class CanonicalPrediction:
    """Teacher / fake-score output converted to canonical fp32 ``x0``."""

    x0: LatentBundle
    raw: Optional[LatentBundle] = None


# ---------------------------------------------------------------------------
# §10.5 Conditioning
# ---------------------------------------------------------------------------


@dataclass
class ConditionBundle:
    """Conditioning tensors, masks, and metadata shared by all roles."""

    tensors: dict[str, Tensor]
    masks: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# §10.6 Role layout, placement, transport, and export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleGroupSpec:
    """A physical model group owning one wrapped model."""

    name: str
    model_ref: str = ""
    storage: Literal["independent_module", "shared_base_adapters"] = "shared_base_adapters"
    placement: Literal["colocated", "standalone"] = "colocated"


@dataclass(frozen=True)
class RoleBinding:
    """A logical algorithm role bound to a group and an optional named adapter."""

    role: str
    group: str
    adapter: Optional[str] = None  # None = frozen base / disabled adapters
    trainable: bool = False
    optimizer_key: Optional[str] = None


@dataclass(frozen=True)
class ScoreTransportSpec:
    """How teacher/fake scores are transported."""

    provider: Literal["colocated", "ray"] = "colocated"
    tensor_backend: Literal["local", "ray_nixl", "mooncake"] = "local"


@dataclass(frozen=True)
class ExportSpec:
    """Which semantic role is exported and through which backend."""

    role: Literal["student", "student_ema"] = "student_ema"
    checkpoint_engine_backend: str = "naive"


@dataclass(frozen=True)
class RoleLayoutSpec:
    """Validated role groups, bindings, and score transport for a plan."""

    groups: tuple[RoleGroupSpec, ...] = ()
    bindings: tuple[RoleBinding, ...] = ()
    score_transport: ScoreTransportSpec = ScoreTransportSpec()
    export: ExportSpec = ExportSpec()


# ---------------------------------------------------------------------------
# §9.1 Immutable plan pieces
# ---------------------------------------------------------------------------

DataRequirements = dict[str, Any]
ObjectiveSpec = dict[str, Any]
RolloutSpec = dict[str, Any]
InitializationSpec = dict[str, Any]


@dataclass(frozen=True)
class UpdatePhaseSpec:
    """A static phase specification that the schedule expands into requests."""

    kind: Literal["student", "fake_score"]
    repeats: int = 1
    batch_policy: Literal["fresh", "reuse_student"] = "fresh"
    trainable_roles: tuple[str, ...] = ()
    update_ema: bool = False


@dataclass(frozen=True)
class PhaseRequest:
    """A concrete phase request emitted by :meth:`UpdateSchedule.next_cycle`."""

    kind: Literal["student", "fake_score"]
    global_step: int
    repeat_index: int
    batch_policy: Literal["fresh", "reuse_student"] = "fresh"


@dataclass
class PhaseResult:
    """The only phase-specific state returned from the executor to the driver."""

    metrics: dict[str, float] = field(default_factory=dict)
    optimizer_steps: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateCycle:
    """A sequence of :class:`PhaseRequest` produced for one cycle."""

    requests: tuple[PhaseRequest, ...] = ()
    requires_student_update: bool = False


@dataclass
class TrainerCounters:
    """Driver-side counters. ``global_step`` is completed student updates only."""

    global_step: int = 0
    optimizer_steps: dict[str, int] = field(default_factory=dict)

    def increment_global(self) -> None:
        self.global_step += 1

    def record_step(self, role: str) -> None:
        self.optimizer_steps[role] = self.optimizer_steps.get(role, 0) + 1


@dataclass(frozen=True)
class UpdateSchedule:
    """Holds the static phase-spec sequence and expands it into cycles."""

    phases: tuple[UpdatePhaseSpec, ...] = ()

    def next_cycle(self, counters: TrainerCounters) -> UpdateCycle:
        """Expand the static phase specs into one concrete :class:`UpdateCycle`.

        A normal cycle emits one student phase followed by ``K`` fake-score
        phases (per the configured ``repeats``). A fake/discriminator-only warmup
        cycle emits fake phases with ``requires_student_update=False`` and never
        advances ``global_step``. A cycle that would advance neither
        ``global_step`` nor any role counter raises rather than being silently
        skipped.
        """
        requests: list[PhaseRequest] = []
        requires_student = False
        for phase in self.phases:
            for repeat_index in range(phase.repeats):
                requests.append(
                    PhaseRequest(
                        kind=phase.kind,
                        global_step=counters.global_step,
                        repeat_index=repeat_index,
                        batch_policy=phase.batch_policy,
                    )
                )
                if phase.kind == "student":
                    requires_student = True

        if not requests:
            raise ValueError("UpdateSchedule produced an empty cycle; zero-progress cycles are not skipped.")

        return UpdateCycle(requests=tuple(requests), requires_student_update=requires_student)


@dataclass(frozen=True)
class DistillationPlan:
    """An immutable, validated plan describing one named distillation recipe."""

    name: str
    version: int = 1
    role_layout: RoleLayoutSpec = RoleLayoutSpec()
    data_requirements: DataRequirements = field(default_factory=dict)
    objective: ObjectiveSpec = field(default_factory=dict)
    rollout: RolloutSpec = field(default_factory=dict)
    initialization: InitializationSpec = field(default_factory=dict)
    update_schedule: UpdateSchedule = field(default_factory=lambda: UpdateSchedule(UpdatePhaseSpec(kind="student")))
    required_capabilities: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# §10.6 role layout validation / semantic export / teacher provider / checkpoint
# ---------------------------------------------------------------------------


# --- merged from role_runtime.py ---
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


# --- merged from export.py ---
EXPORTABLE_ROLES = ("student", "student_ema")


def resolve_export_role(export: ExportSpec) -> str:
    """Return the resolved semantic export role, validated to be exportable."""
    if export.role not in EXPORTABLE_ROLES:
        raise ValueError(f"Export role must be one of {EXPORTABLE_ROLES}, got {export.role!r}.")
    return export.role


# --- merged from score_providers.py ---
@runtime_checkable
class TeacherScoreProvider(Protocol):
    """Provides canonical ``x0`` teacher predictions for a score batch."""

    def predict_x0(self, score_batch: ScoreBatch) -> CanonicalPrediction:
        """Return the teacher's canonical fp32 ``x0`` for ``score_batch``.

        This must not hold a student autograd graph across an unbounded
        synchronous round trip. A standalone provider is responsible for
        versioned weight sync, cancellation, and never materializing tensors on
        the driver via the driver process.
        """
        ...


# --- merged from checkpoint.py ---
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

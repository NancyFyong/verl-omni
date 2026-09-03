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
"""CPU tests for distillation contracts, role layout validation, and recipes."""

import dataclasses

import pytest
import torch

from verl_omni.trainer.diffusion.distillation.contracts import (
    ExportSpec,
    LatentBundle,
    RoleBinding,
    RoleGroupSpec,
    RoleLayoutSpec,
    TrainerCounters,
    UpdatePhaseSpec,
    UpdateSchedule,
)
from verl_omni.trainer.diffusion.distillation.export import resolve_export_role
from verl_omni.trainer.diffusion.distillation.recipes import build_plan, recipe_registry
from verl_omni.trainer.diffusion.distillation.registry import _Registry
from verl_omni.trainer.diffusion.distillation.role_runtime import validate_role_layout

ALL_CAPS = frozenset({"distribution_matching", "autoregressive", "adversarial"})


class TestLatentBundle:
    def test_rejects_empty_bundle(self):
        with pytest.raises(ValueError, match="at least one modality"):
            LatentBundle({})

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            LatentBundle({"image": [1, 2, 3]})

    def test_single_returns_the_only_tensor(self):
        x = torch.randn(2, 3)
        assert torch.equal(LatentBundle({"image": x}).single, x)

    def test_single_rejects_multimodal_bundle(self):
        bundle = LatentBundle({"video": torch.randn(1), "audio": torch.randn(1)})
        with pytest.raises(ValueError, match="one-modality"):
            _ = bundle.single

    def test_map_applies_to_every_modality(self):
        bundle = LatentBundle({"video": torch.ones(2), "audio": torch.ones(3)})
        doubled = bundle.map(lambda t: t * 2)
        assert torch.equal(doubled.get("video"), torch.full((2,), 2.0))
        assert torch.equal(doubled.get("audio"), torch.full((3,), 2.0))


class TestImmutability:
    @pytest.mark.parametrize(
        "instance,field_name",
        [
            (RoleGroupSpec(name="base"), "name"),
            (RoleBinding(role="student", group="base"), "role"),
            (ExportSpec(), "role"),
            (UpdatePhaseSpec(kind="student"), "kind"),
            (UpdateSchedule(), "phases"),
        ],
    )
    def test_plan_pieces_are_frozen(self, instance, field_name):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)

    def test_plan_is_frozen(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, ALL_CAPS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.name = "other"


class TestRoleLayoutValidation:
    def _layout(self, **kwargs):
        defaults = dict(
            groups=(RoleGroupSpec(name="base"),),
            bindings=(
                RoleBinding(role="student", group="base", adapter="student", trainable=True, optimizer_key="student"),
                RoleBinding(role="student_ema", group="base", adapter="student_ema"),
            ),
            export=ExportSpec(role="student_ema"),
        )
        defaults.update(kwargs)
        return RoleLayoutSpec(**defaults)

    def test_valid_layout_passes(self):
        validate_role_layout(self._layout())

    def test_binding_to_unknown_group_raises(self):
        layout = self._layout(
            bindings=(RoleBinding(role="student", group="missing", trainable=True, optimizer_key="student"),),
            export=ExportSpec(role="student"),
        )
        with pytest.raises(ValueError, match="unknown group"):
            validate_role_layout(layout)

    def test_trainable_role_without_optimizer_key_raises(self):
        layout = self._layout(
            bindings=(RoleBinding(role="student", group="base", adapter="student", trainable=True),),
            export=ExportSpec(role="student"),
        )
        with pytest.raises(ValueError, match="must set an optimizer_key"):
            validate_role_layout(layout)

    def test_duplicate_adapter_in_shared_base_raises(self):
        layout = self._layout(
            bindings=(
                RoleBinding(role="student", group="base", adapter="dup", trainable=True, optimizer_key="student"),
                RoleBinding(role="fake_score", group="base", adapter="dup", trainable=True, optimizer_key="fake"),
                RoleBinding(role="student_ema", group="base", adapter="student_ema"),
            ),
        )
        with pytest.raises(ValueError, match="duplicate adapter names"):
            validate_role_layout(layout)

    def test_duplicate_role_binding_raises(self):
        layout = self._layout(
            bindings=(
                RoleBinding(role="student", group="base", adapter="a", trainable=True, optimizer_key="s"),
                RoleBinding(role="student", group="base", adapter="b", trainable=True, optimizer_key="s2"),
            ),
            export=ExportSpec(role="student"),
        )
        with pytest.raises(ValueError, match="Duplicate role binding"):
            validate_role_layout(layout)

    def test_unbound_export_role_raises(self):
        layout = self._layout(
            bindings=(RoleBinding(role="student", group="base", adapter="s", trainable=True, optimizer_key="s"),),
            export=ExportSpec(role="student_ema"),
        )
        with pytest.raises(ValueError, match="is not bound"):
            validate_role_layout(layout)

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="at least one role group"):
            validate_role_layout(RoleLayoutSpec(groups=(), bindings=()))


class TestExportRole:
    @pytest.mark.parametrize("role", ["student", "student_ema"])
    def test_exportable_roles(self, role):
        assert resolve_export_role(ExportSpec(role=role)) == role

    @pytest.mark.parametrize("role", ["teacher_score", "fake_score", "discriminator"])
    def test_non_exportable_roles_raise(self, role):
        with pytest.raises(ValueError, match="Export role must be"):
            resolve_export_role(ExportSpec(role=role))


class TestRegistry:
    def test_all_four_recipes_registered(self):
        assert set(recipe_registry.names) == {"dmd", "dmd2", "causvid", "self_forcing"}

    def test_duplicate_registration_raises(self):
        registry = _Registry()

        @registry.register("thing")
        class _A:
            pass

        with pytest.raises(ValueError, match="Duplicate registration"):

            @registry.register("thing")
            class _B:
                pass

    def test_unknown_name_raises_with_registered_list(self):
        registry = _Registry()
        with pytest.raises(KeyError, match="Registered"):
            registry.get("nope")


class TestRecipePlans:
    @pytest.mark.parametrize("name", ["dmd", "dmd2", "causvid", "self_forcing"])
    def test_every_recipe_builds_a_validated_plan(self, name):
        plan = build_plan(name, {"model_path": "/m"}, ALL_CAPS)
        assert plan.name == name
        assert plan.role_layout.bindings
        validate_role_layout(plan.role_layout)

    def test_missing_capability_is_fail_closed(self):
        with pytest.raises(ValueError, match="requires capabilities"):
            build_plan("self_forcing", {"model_path": "/m"}, frozenset({"distribution_matching"}))

    def test_dmd2_paper_profile_adds_discriminator(self):
        plan = build_plan("dmd2", {"profile": "paper", "model_path": "/m"}, ALL_CAPS)
        roles = {b.role for b in plan.role_layout.bindings}
        assert "discriminator" in roles
        assert plan.objective["adversarial"] is True

    def test_dmd2_distribution_only_has_no_discriminator(self):
        plan = build_plan("dmd2", {"profile": "distribution_only", "model_path": "/m"}, ALL_CAPS)
        roles = {b.role for b in plan.role_layout.bindings}
        assert "discriminator" not in roles
        assert plan.objective["adversarial"] is False

    def test_causvid_and_self_forcing_require_ode_provenance(self):
        for name in ("causvid", "self_forcing"):
            plan = build_plan(name, {"model_path": "/m"}, ALL_CAPS)
            assert plan.initialization["stage"] == "ode_regression"
            assert plan.initialization["requires_provenance"] is True

    def test_fake_update_ratio_controls_phase_repeats(self):
        plan = build_plan("dmd2", {"fake_update_ratio": 7, "model_path": "/m"}, ALL_CAPS)
        fake_phase = [p for p in plan.update_schedule.phases if p.kind == "fake_score"][0]
        assert fake_phase.repeats == 7

    def test_teacher_role_is_frozen_base_without_adapter(self):
        plan = build_plan("dmd2", {"model_path": "/m"}, ALL_CAPS)
        teacher = [b for b in plan.role_layout.bindings if b.role == "teacher_score"][0]
        assert teacher.adapter is None
        assert teacher.trainable is False


class TestUpdateSchedule:
    def test_next_cycle_flags_student_requirement(self):
        schedule = UpdateSchedule(
            phases=(UpdatePhaseSpec(kind="student"), UpdatePhaseSpec(kind="fake_score", repeats=2))
        )
        cycle = schedule.next_cycle(TrainerCounters())
        assert cycle.requires_student_update is True
        assert len(cycle.requests) == 3

    def test_fake_only_schedule_does_not_require_student(self):
        schedule = UpdateSchedule(phases=(UpdatePhaseSpec(kind="fake_score", repeats=2),))
        cycle = schedule.next_cycle(TrainerCounters())
        assert cycle.requires_student_update is False

    def test_empty_schedule_raises(self):
        with pytest.raises(ValueError, match="empty cycle"):
            UpdateSchedule(phases=()).next_cycle(TrainerCounters())


class TestTrainerCounters:
    def test_counters_start_at_zero(self):
        counters = TrainerCounters()
        assert counters.global_step == 0
        assert counters.optimizer_steps == {}

    def test_record_step_accumulates_per_role(self):
        counters = TrainerCounters()
        counters.record_step("student")
        counters.record_step("student")
        counters.record_step("fake_score")
        assert counters.optimizer_steps == {"student": 2, "fake_score": 1}

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
"""CPU tests for ``algorithm.trainer_type=distillation`` routing.

The DMD-family trainer is routed by ``algorithm.trainer_type``, which is
orthogonal to the existing on-policy distillation (OPD) path. OPD keeps using
``distillation.enabled=true`` together with ``trainer_type=policy_gradient``.
"""

import pytest

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls


class _Algo:
    def __init__(self, trainer_type):
        self.trainer_type = trainer_type


class _Config:
    def __init__(self, trainer_type):
        self.algorithm = _Algo(trainer_type)


class TestTrainerRouting:
    def test_distillation_routes_to_distillation_trainer(self):
        assert _get_trainer_cls(_Config("distillation")) is DistillationRayTrainer

    def test_policy_gradient_is_unchanged(self):
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

        assert _get_trainer_cls(_Config("policy_gradient")) is PolicyGradientRayTrainer

    def test_direct_preference_is_unchanged(self):
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer

        assert _get_trainer_cls(_Config("direct_preference")) is DirectPreferenceRayTrainer

    def test_unknown_trainer_type_lists_distillation(self):
        with pytest.raises(ValueError, match="distillation"):
            _get_trainer_cls(_Config("bogus"))


class TestAlgorithmConfig:
    def test_distillation_is_a_valid_trainer_type(self):
        config = DiffusionAlgoConfig(trainer_type="distillation")
        assert config.trainer_type == "distillation"

    def test_existing_trainer_types_still_valid(self):
        assert DiffusionAlgoConfig(trainer_type="policy_gradient").trainer_type == "policy_gradient"
        assert DiffusionAlgoConfig(trainer_type="direct_preference").trainer_type == "direct_preference"

    def test_default_is_policy_gradient(self):
        assert DiffusionAlgoConfig().trainer_type == "policy_gradient"

    def test_invalid_trainer_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trainer_type"):
            DiffusionAlgoConfig(trainer_type="bogus")


class TestPR1DataPlaneBoundary:
    def test_fit_without_executor_reports_pr2_boundary(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        plan = build_plan("dmd2", {"model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer = DistillationRayTrainer(plan=plan)
        with pytest.raises(NotImplementedError, match="PR 2"):
            trainer.fit(num_cycles=1)

    def test_control_plane_binds_when_collaborators_are_supplied(self):
        from verl_omni.trainer.diffusion.distillation.phase_executor import (
            FakeBatchProvider,
            FakePhaseExecutor,
        )
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        plan = build_plan("dmd2", {"fake_update_ratio": 1, "model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer = DistillationRayTrainer(
            plan=plan,
            executor=FakePhaseExecutor(),
            batch_provider=FakeBatchProvider(num_batches=100),
        )
        trainer.fit(num_cycles=2)
        assert trainer.control_plane.counters.global_step == 2

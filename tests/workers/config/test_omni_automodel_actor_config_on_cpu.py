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
"""CPU tests for OmniAutomodelActorConfig.

Guarded by ``pytest.importorskip("verl")`` because the config subclasses verl's
``ActorConfig`` and composes ``AutomodelEngineConfig`` / ``AutomodelOptimizerConfig``.
"""

import pytest

pytest.importorskip("verl")

from verl.workers.config import AutomodelEngineConfig, AutomodelOptimizerConfig  # noqa: E402

from verl_omni.workers.config.omni import OmniAutomodelActorConfig, OmniLossConfig  # noqa: E402


class TestOmniAutomodelActorConfig:
    def test_defaults_and_engine_wiring(self):
        cfg = OmniAutomodelActorConfig(
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
        )
        assert cfg.strategy == "automodel"
        assert isinstance(cfg.automodel, AutomodelEngineConfig)
        assert isinstance(cfg.optim, AutomodelOptimizerConfig)
        # __post_init__ points engine at the automodel block and stamps its strategy.
        assert cfg.engine is cfg.automodel
        assert cfg.engine.strategy == "automodel"

    def test_direct_preference_includes_omni_loss(self):
        cfg = OmniAutomodelActorConfig(
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            trainer_type="direct_preference",
        )
        assert cfg.trainer_type == "direct_preference"
        assert isinstance(cfg.omni_loss, OmniLossConfig)

    def test_accepts_policy_gradient_trainer_type(self):
        cfg = OmniAutomodelActorConfig(
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
            trainer_type="policy_gradient",
        )
        assert cfg.trainer_type == "policy_gradient"

    def test_invalid_trainer_type_raises(self):
        with pytest.raises(ValueError):
            OmniAutomodelActorConfig(
                rollout_n=1,
                ppo_micro_batch_size_per_gpu=1,
                trainer_type="invalid",
            )

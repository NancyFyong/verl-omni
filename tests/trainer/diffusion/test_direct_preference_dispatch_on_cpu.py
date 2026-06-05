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
"""CPU tests for ``DirectPreferenceRayTrainer.fit`` dispatch.

Necessity: ``DirectPreferenceRayTrainer.fit`` is a thin dispatcher that routes
to ``_fit_dpo_offline`` or ``_fit_nft_online`` based on
``actor.diffusion_loss.loss_mode``. This contract is what keeps the upstream
DPO offline path unmodified while letting NFT live as a sibling. These tests
pin the dispatch table without standing up a Ray cluster.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer


def _make_trainer_with_loss_mode(loss_mode: str) -> DirectPreferenceRayTrainer:
    """Construct a DirectPreferenceRayTrainer instance bypassing __init__.

    We only need ``self.config.actor_rollout_ref.actor.diffusion_loss.loss_mode``
    to drive the dispatcher. Bypassing ``__init__`` avoids needing a tokenizer,
    role-worker mapping, dataset, etc. This is the same pattern used by
    ``test_qwen_image_edit_adapter_on_cpu.py`` / ``test_sd3_dpo_adapter_on_cpu.py``
    when constructing config objects.
    """
    trainer = object.__new__(DirectPreferenceRayTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                diffusion_loss=SimpleNamespace(loss_mode=loss_mode),
            ),
        ),
    )
    return trainer


# ===========================================================================
# 1. Dispatch table
# ===========================================================================


class TestDirectPreferenceFitDispatch:
    def test_dpo_loss_mode_routes_to_fit_dpo_offline(self):
        trainer = _make_trainer_with_loss_mode("dpo")
        # Patch the two private fit helpers to spy.
        trainer._fit_dpo_offline = MagicMock(return_value="dpo-result")
        trainer._fit_nft_online = MagicMock(return_value="nft-result")

        result = trainer.fit()

        assert result == "dpo-result"
        trainer._fit_dpo_offline.assert_called_once_with()
        trainer._fit_nft_online.assert_not_called()

    def test_nft_loss_mode_routes_to_fit_nft_online(self):
        trainer = _make_trainer_with_loss_mode("nft")
        trainer._fit_dpo_offline = MagicMock(return_value="dpo-result")
        trainer._fit_nft_online = MagicMock(return_value="nft-result")

        result = trainer.fit()

        assert result == "nft-result"
        trainer._fit_nft_online.assert_called_once_with()
        trainer._fit_dpo_offline.assert_not_called()

    def test_unknown_loss_mode_raises_with_actionable_message(self):
        trainer = _make_trainer_with_loss_mode("nonexistent_mode")
        trainer._fit_dpo_offline = MagicMock()
        trainer._fit_nft_online = MagicMock()

        with pytest.raises(NotImplementedError) as excinfo:
            trainer.fit()

        msg = str(excinfo.value)
        # Error must echo the offending mode and list supported modes so the
        # user knows what to set in their config.
        assert "nonexistent_mode" in msg
        assert "dpo" in msg
        assert "nft" in msg
        # No private fit ran.
        trainer._fit_dpo_offline.assert_not_called()
        trainer._fit_nft_online.assert_not_called()

    def test_dispatcher_does_not_consume_self_state(self):
        """Calling ``fit()`` must not set or mutate any other attribute on the
        trainer — that would smuggle state across the DPO/NFT boundary.
        """
        trainer = _make_trainer_with_loss_mode("dpo")
        trainer._fit_dpo_offline = MagicMock(return_value=None)
        trainer._fit_nft_online = MagicMock(return_value=None)

        attrs_before = set(vars(trainer).keys())
        trainer.fit()
        attrs_after = set(vars(trainer).keys())

        assert attrs_before == attrs_after, f"fit() dispatcher leaked attributes: added={attrs_after - attrs_before}"


# ===========================================================================
# 2. Helper methods exist (regression for the merge that broke them)
# ===========================================================================


class TestDirectPreferenceFitHelpersExist:
    """During the merge that landed NFT, the upstream offline DPO ``fit()`` was
    silently overwritten. These tests freeze the recovery so a future refactor
    that drops ``_fit_dpo_offline`` produces a clear test failure rather than
    a runtime regression on the SD3 offline DPO example.
    """

    def test_fit_dpo_offline_method_exists(self):
        assert hasattr(DirectPreferenceRayTrainer, "_fit_dpo_offline")
        assert callable(DirectPreferenceRayTrainer._fit_dpo_offline)

    def test_fit_nft_online_method_exists(self):
        assert hasattr(DirectPreferenceRayTrainer, "_fit_nft_online")
        assert callable(DirectPreferenceRayTrainer._fit_nft_online)

    def test_upstream_offline_dpo_helpers_preserved(self):
        # These are the upstream methods that the offline DPO fit calls. They
        # were imported verbatim during the merge. Removing any of them is a
        # breaking change against upstream PR #95 (offline DPO).
        for method in ("init_workers", "_validate", "_update_actor", "_compute_ref_noise_pred"):
            assert hasattr(DirectPreferenceRayTrainer, method), (
                f"Upstream DPO contract method {method!r} is missing from DirectPreferenceRayTrainer"
            )


# ===========================================================================
# 3. NFT enhancement metadata threading (off_policy / kl_beta)
# ===========================================================================


class TestNFTEnhancementMetadata:
    """The trainer reads ``algorithm.nft_off_policy`` and
    ``algorithm.nft_kl_beta`` and threads them into the actor update batch
    via ``tu.assign_non_tensor`` — these are the only two switches the
    engine path consumes for the off-policy and KL-anchor enhancements.
    Pin the field/types so downstream serialization (TensorDict NonTensor
    stash) cannot silently drop them.
    """

    def test_diffusion_algo_config_has_nft_enhancement_fields(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig

        cfg = DiffusionAlgoConfig()
        # nft_off_policy default OFF — strict on-policy.
        assert cfg.nft_off_policy is False
        # nft_kl_beta default 0 — no KL anchor.
        assert cfg.nft_kl_beta == 0.0
        # Types are concrete (not Optional) so OmegaConf merge cannot
        # accidentally substitute None.
        assert isinstance(cfg.nft_off_policy, bool)
        assert isinstance(cfg.nft_kl_beta, float)

    def test_overriding_nft_enhancement_fields_through_algo_config(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig

        cfg = DiffusionAlgoConfig(nft_off_policy=True, nft_kl_beta=0.123)
        assert cfg.nft_off_policy is True
        assert cfg.nft_kl_beta == pytest.approx(0.123)

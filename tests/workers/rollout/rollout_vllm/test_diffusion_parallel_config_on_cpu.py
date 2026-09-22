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

"""Rollout topology and VAE knobs must survive the actual engine boundary."""

from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as strategy_module


def _prepare(monkeypatch, config, *, algorithm="flow_grpo", nested=None, **kwargs):
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    monkeypatch.setattr(strategy_module.VllmOmniPipelineBase, "get_pipeline_path", lambda **_: None)
    server = SimpleNamespace(
        config=config, model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm)
    )
    args = Namespace(tensor_parallel_size=config.tensor_model_parallel_size, **kwargs)
    engine_args = asdict(OmniEngineArgs.from_cli_args(args))
    if nested is not None:
        engine_args["parallel_config"] = nested
    strategy_module.DiffusionStrategy(server).prepare_engine_args(engine_args, args)
    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)[0]
    return OmniDiffusionConfig(
        parallel_config=stage["engine_args"]["parallel_config"],
        vae_use_tiling=stage["engine_args"]["vae_use_tiling"],
    )


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("nested", [None, {"tensor_parallel_size": 2}])
def test_vae_parallel_reaches_od_config(monkeypatch, algorithm, nested):
    config = DiffusionRolloutConfig(
        tensor_model_parallel_size=2, vae_patch_parallel_size=2, vae_parallel_mode="tile", vae_use_tiling=True
    )
    od = _prepare(monkeypatch, config, algorithm=algorithm, nested=nested)
    assert od.parallel_config.tensor_parallel_size == 2
    assert od.parallel_config.vae_patch_parallel_size == 2
    assert od.parallel_config.vae_parallel_mode == "tile"
    assert od.vae_use_tiling is True
    if nested is not None:
        assert nested == {"tensor_parallel_size": 2}


@pytest.mark.parametrize("field", ["ulysses_degree", "ring_degree", "vae_patch_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_parallel_degrees_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        DiffusionRolloutConfig(**{field: value})


def test_vae_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="vae_parallel_mode"):
        DiffusionRolloutConfig(vae_parallel_mode="unknown")


def test_default_parallelism_is_unchanged(monkeypatch):
    od = _prepare(monkeypatch, DiffusionRolloutConfig(tensor_model_parallel_size=2))
    assert od.parallel_config.world_size == 2
    assert od.parallel_config.ulysses_degree == 1
    assert od.parallel_config.ring_degree == 1
    assert od.parallel_config.vae_patch_parallel_size == 1
    assert od.vae_use_tiling is False


def test_nested_vae_conflict_is_not_silently_ignored(monkeypatch):
    config = DiffusionRolloutConfig(tensor_model_parallel_size=2, vae_patch_parallel_size=2)
    with pytest.raises(ValueError, match="Conflicting.*vae_patch_parallel_size"):
        _prepare(monkeypatch, config, nested={"vae_patch_parallel_size": 1})


def test_legacy_sp_cannot_bypass_resource_allocation(monkeypatch):
    with pytest.raises(ValueError, match="ulysses_degree"):
        _prepare(monkeypatch, DiffusionRolloutConfig(), ulysses_degree=2)


def test_nested_sp_cannot_bypass_resource_allocation(monkeypatch):
    with pytest.raises(ValueError, match="ring_degree"):
        _prepare(monkeypatch, DiffusionRolloutConfig(), nested={"ring_degree": 2})


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("nested", [False, True])
def test_pure_ulysses_and_shared_encoder_vae_group(monkeypatch, algorithm, nested):
    config = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        ulysses_degree=4,
        text_encoder_tp_size=4,
        vae_patch_parallel_size=4,
        vae_use_tiling=True,
    )
    od = _prepare(monkeypatch, config, algorithm=algorithm, nested={"tensor_parallel_size": 1} if nested else None)
    assert od.parallel_config.tensor_parallel_size == 1
    assert od.parallel_config.sequence_parallel_size == 4
    assert od.parallel_config.world_size == 4
    assert od.parallel_config.text_encoder_tp_size == 4
    assert od.parallel_config.vae_patch_parallel_size == 4


@pytest.mark.parametrize("key,field", [("usp", "ulysses_degree"), ("ring-degree", "ring_degree")])
def test_explicit_one_cannot_override_allocated_sp(monkeypatch, key, field):
    config = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        engine_kwargs={"vllm_omni": {key: 1}},
        **{field: 4},
    )
    with pytest.raises(ValueError, match=field):
        _prepare(monkeypatch, config, **{field: 1})


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"ulysses_degree": 2, "ring_degree": 2}, "hybrid"),
        ({"vae_patch_parallel_size": 2}, "full DiT group"),
        ({"vae_parallel_mode": "spatial_shard_height"}, "tile only"),
        ({"cfg_parallel_size": 2}, "cfg_parallel_size"),
        ({"text_encoder_tp_size": 2}, "text_encoder_tp_size"),
    ],
)
def test_h3_rejects_unsupported_groups(overrides, error):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import validate_h3_parallel_config

    with pytest.raises(ValueError, match=error):
        validate_h3_parallel_config(DiffusionParallelConfig(tensor_parallel_size=4, **overrides))


@pytest.mark.parametrize("tp,usp,vae,etp", [(4, 1, 4, 4), (1, 4, 4, 4), (2, 2, 1, 1)])
def test_h3_accepts_group_reuse(tp, usp, vae, etp):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import validate_h3_parallel_config

    validate_h3_parallel_config(
        DiffusionParallelConfig(
            tensor_parallel_size=tp,
            ulysses_degree=usp,
            vae_patch_parallel_size=vae,
            text_encoder_tp_size=etp,
        )
    )


@pytest.mark.parametrize("key", ["vae_use_tiling", "vae-use-tiling"])
def test_explicit_legacy_tiling_conflict(monkeypatch, key):
    config = DiffusionRolloutConfig(vae_use_tiling=True, engine_kwargs={"vllm_omni": {key: False}})
    with pytest.raises(ValueError, match="Conflicting vae_use_tiling"):
        _prepare(monkeypatch, config, vae_use_tiling=False)


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_h3_parallel_validation_precedes_weight_loading(monkeypatch, algorithm):
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    import verl_omni.pipelines  # noqa: F401

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid H3 topology reached model loading")

    monkeypatch.setattr(MiniMaxH3Pipeline, "__init__", unexpected_load)
    pipeline_cls = strategy_module.VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", algorithm)
    parallel = DiffusionParallelConfig(tensor_parallel_size=4, vae_patch_parallel_size=2)
    with pytest.raises(ValueError, match="vae_patch_parallel_size"):
        pipeline_cls(od_config=SimpleNamespace(parallel_config=parallel))


def test_hydra_exposes_parallel_fields_without_plus():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    config_dir = Path(__file__).resolve().parents[4] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                "actor_rollout_ref.rollout.name=vllm_omni",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
                "actor_rollout_ref.rollout.vae_patch_parallel_size=4",
                "actor_rollout_ref.rollout.vae_use_tiling=true",
            ],
        )
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    assert rollout.vae_patch_parallel_size == 4
    assert rollout.vae_use_tiling is True
    assert rollout.ulysses_degree == rollout.ring_degree == 1

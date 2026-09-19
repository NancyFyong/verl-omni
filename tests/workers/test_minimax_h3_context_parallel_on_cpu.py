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
"""CPU contract tests for MiniMax H3 Diffusers context parallelism."""

import pytest

from tests.special_e2e.run_flowgrpo_minimax_h3_tiny import _hydra_overrides, _validate_actor_sp
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import validate_standard_ulysses_sequence_length


def test_minimax_h3_uses_diffusers_standard_ulysses_configuration() -> None:
    from diffusers import ContextParallelConfig

    config = ContextParallelConfig(ulysses_degree=2)

    assert config.ulysses_anything is False
    assert config.ring_anything is False


@pytest.mark.parametrize(("sequence_length", "sp_size"), [(7, 1), (8, 2), (8, 4), (12, 4)])
def test_minimax_h3_accepts_standard_ulysses_compatible_layout(sequence_length, sp_size) -> None:
    validate_standard_ulysses_sequence_length(sequence_length, sp_size)


@pytest.mark.parametrize(("sequence_length", "sp_size"), [(7, 2), (9, 4), (10, 4)])
def test_minimax_h3_rejects_uneven_standard_ulysses_layout(sequence_length, sp_size) -> None:
    with pytest.raises(ValueError, match=rf"sequence_length={sequence_length} and sp_size={sp_size}"):
        validate_standard_ulysses_sequence_length(sequence_length, sp_size)


def test_minimax_h3_rejects_nonpositive_standard_ulysses_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        validate_standard_ulysses_sequence_length(sequence_length=8, sp_size=0)


@pytest.mark.parametrize(("task", "train_batch_size"), [("t2va", 8), ("fl2va", 8), ("ref2va", 4)])
def test_minimax_h3_smoke_config_enables_sp_for_each_task(task, train_batch_size) -> None:
    overrides = _hydra_overrides(
        tiny_model_dir="/tmp/model",
        train_parquet="/tmp/train.parquet",
        val_parquet="/tmp/val.parquet",
        reward_stub_path="/tmp/reward.py",
        output_dir="/tmp/output",
        task=task,
        num_gpus=4,
        actor_sp=2,
        rollout_tp=2,
        text_encoder_tp=1,
        total_training_steps=1,
        ray_num_cpus=4,
        height=160,
        width=288,
        num_frames=97,
        num_inference_steps=4,
    )

    assert "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2" in overrides
    assert f"data.train_batch_size={train_batch_size}" in overrides
    assert f"actor_rollout_ref.rollout.pipeline.task={task}" in overrides


def test_minimax_h3_smoke_config_rejects_invalid_sp_partition() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_actor_sp(num_gpus=4, actor_sp=3)

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

"""CPU checks for LingBot Dense T2V example script launchers."""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "examples" / "flowgrpo_trainer" / "lingbot_video"


def _read_script(script: str) -> str:
    return (_SCRIPT_DIR / script).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script, experiment_name",
    [
        ("run_lingbot_dense_t2v_lora.sh", "lingbot_dense_t2v_lora"),
        ("run_lingbot_dense_t2v_lora_fsdp2.sh", "lingbot_dense_t2v_lora_fsdp2"),
    ],
)
def test_lingbot_training_scripts_use_simple_example_launcher_shape(script, experiment_name):
    text = _read_script(script)

    assert "set -x" in text
    assert "CHECK_CONFIG_ONLY" not in text
    assert "must_divide" not in text
    assert "python3 -m verl_omni.trainer.main_diffusion" in text
    assert f"EXPERIMENT_NAME=${{EXPERIMENT_NAME:-{experiment_name}}}" in text
    assert "output_dir=${OUTPUT_DIR:-$WORKSPACE/outputs/$EXPERIMENT_NAME}" in text
    assert "checkpoint_dir=${CHECKPOINT_DIR:-$output_dir/checkpoints}" in text
    assert 'exec > >(tee -a "$log_file") 2>&1' in text
    assert 'echo "Logging to $log_file"' in text


@pytest.mark.parametrize(
    "script",
    ["run_lingbot_dense_t2v_lora.sh", "run_lingbot_dense_t2v_lora_fsdp2.sh"],
)
def test_lingbot_training_scripts_keep_validated_defaults(script):
    text = _read_script(script)

    expected_snippets = [
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}",
        "VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}",
        "ROLLOUT_GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-8}",
        "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}",
        "PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}",
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}",
        "ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}",
        "NUM_FRAMES=${NUM_FRAMES:-81}",
        "ROLLOUT_NOISE_LEVEL=${ROLLOUT_NOISE_LEVEL:-0.7}",
        "ROLLOUT_SDE_TYPE=${ROLLOUT_SDE_TYPE:-dance_sde}",
        "FLOW_SHIFT=${FLOW_SHIFT:-3.0}",
        "GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.0}",
        "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}",
        "VAL_NUM_INFERENCE_STEPS=${VAL_NUM_INFERENCE_STEPS:-40}",
        "LR=${LR:-1e-5}",
        "LORA_RANK=${LORA_RANK:-64}",
        "LORA_ALPHA=${LORA_ALPHA:-128}",
        "TRAINER_LOGGER=${TRAINER_LOGGER:-",
        "actor_rollout_ref.rollout.pipeline.shift=$FLOW_SHIFT",
        "actor_rollout_ref.rollout.algo.sde_type=$ROLLOUT_SDE_TYPE",
        'actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE"',
        "reward.custom_reward_function.name=$REWARD_FUNCTION_NAME",
        "trainer.rollout_data_dir=$rollout_data_dir",
        "trainer.validation_data_dir=$val_data_dir",
        "trainer.resume_mode=$RESUME_MODE",
    ]
    for snippet in expected_snippets:
        assert snippet in text


def test_lingbot_fsdp2_script_sets_fsdp2_specific_knobs():
    text = _read_script("run_lingbot_dense_t2v_lora_fsdp2.sh")

    assert "ROLLOUT_TP=${ROLLOUT_TP:-2}" in text
    assert "ACTOR_SP=${ACTOR_SP:-1}" in text
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}" in text
    assert "ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}" in text
    assert "actor_rollout_ref.actor.strategy=$ACTOR_STRATEGY" in text
    assert "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP" in text


def test_lingbot_non_fsdp2_script_keeps_fp32_actor_init():
    text = _read_script("run_lingbot_dense_t2v_lora.sh")

    assert "ROLLOUT_TP=${ROLLOUT_TP:-1}" in text
    assert "actor_rollout_ref.actor.strategy=fsdp2" not in text
    assert "MODEL_DTYPE=${MODEL_DTYPE:-fp32}" in text
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE" in text
    assert "LORA_DTYPE=${LORA_DTYPE:-bf16}" in text
    assert "actor_rollout_ref.model.lora_dtype=$LORA_DTYPE" in text

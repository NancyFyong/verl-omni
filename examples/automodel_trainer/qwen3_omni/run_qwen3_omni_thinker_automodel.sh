#!/usr/bin/env bash
# Qwen3-Omni Thinker offline DPO training on the Automodel (nemo_automodel) engine.
#
# This demonstrates the automodel backend for omni models (image+text). It is an
# opt-in example: it selects OmniAutomodelActorConfig (actor.strategy=automodel),
# which routes to OmniAutomodelEngine, instead of the default FSDP engine.
#
# Offline DPO is used deliberately -- it exercises model build, forward, loss and
# the optimizer step without requiring rollout weight sync
# (engine.get_per_tensor_param), which is validated separately for online RL.
# No reward model is needed: preference pairs come from the dataset.
#
# Requirements (all Ray worker nodes):
#   - nemo_automodel installed (optional dependency; the engine is only registered
#     when it is importable).
#   - GPU. This example has not been run on CPU; the automodel build ->
#     configure_model ordering is validated on GPU.
#   - pip install qwen-vl-utils  # required for multimodal data processing
#
# Data preparation (run once) -- Omni-Preference, image split only:
#   export DATASET_ROOT=$HOME/Omni-Preference
#   python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
#       --dataset_root "$DATASET_ROOT" \
#       --output_dir "$DATASET_ROOT/parquet_dpo" \
#       --modalities image
#   See examples/dpo_trainer/data_process/omni_preference_dpo_dataset.md for the
#   download steps and the full parquet schema.

set -x

# Make verl_omni available to Ray workers.
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
DATA_DIR=${DATA_DIR:-"$HOME/Omni-Preference/parquet_dpo/image"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/test.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m verl_omni.trainer.main_omni \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_omni_thinker_automodel \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"

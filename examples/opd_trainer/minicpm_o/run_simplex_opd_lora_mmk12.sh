#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker OPD on MMK12 (image + text -> text): the AVQA OPD wrapper
# with the MMK12 data and rule-based reward used by the Qwen3-Omni OPD recipe.
#
# Data preparation (run once):
#   pip install math-verify    # required by mmk12_reward.py
#   python examples/gspo_trainer/data_process/mmk12.py \
#       --local_dataset_path <path_to_raw_mmk12> \
#       --local_save_dir ~/data/mmk12
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"${REPO_ROOT}/verl_omni/utils/reward_score/mmk12_reward.py"}

TRAIN_FILE="${TRAIN_FILE}" VAL_FILE="${VAL_FILE}" exec bash "${SCRIPT_DIR}/run_simplex_opd_lora.sh" \
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.experiment_name=minicpm_o_4_5_thinker_lora_mmk12_opd \
    "$@"

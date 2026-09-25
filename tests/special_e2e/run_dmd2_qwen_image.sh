#!/usr/bin/env bash
# Multi-GPU DMD2 production smoke with complete resumable FSDP checkpoints.
set -euo pipefail

export NUM_GPUS=${NUM_GPUS:-2}
export TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}
export MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/dmd2_smoke}
DATA_DIR=${DATA_DIR:-${OUTPUT_DIR}/data}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "$((NUM_GPUS * TOTAL_TRAIN_STEPS * 3))" \
    --val_size "${NUM_GPUS}" \
    --user_prompt_only

export TRAIN_FILES=${DATA_DIR}/train.parquet
export VAL_FILES=${DATA_DIR}/test.parquet
export SAVE_FREQ=1
export RESUME_MODE=${RESUME_MODE:-disable}

bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
    actor_rollout_ref.model.lora_rank=2 \
    actor_rollout_ref.model.lora_alpha=2 \
    actor_rollout_ref.model.pipeline.height=64 \
    actor_rollout_ref.model.pipeline.width=64 \
    actor_rollout_ref.model.pipeline.max_sequence_length=64 \
    data.max_prompt_length=64 \
    trainer.logger=console \
    "$@"

CHECKPOINT="${OUTPUT_DIR}/global_step_${TOTAL_TRAIN_STEPS}"
test -f "${CHECKPOINT}/trainer.pt"
test -f "${CHECKPOINT}/data.pt"
for ((rank = 0; rank < NUM_GPUS; rank++)); do
    for kind in model optim extra_state; do
        test -f "${CHECKPOINT}/actor/${kind}_world_size_${NUM_GPUS}_rank_${rank}.pt"
    done
    test -f "${CHECKPOINT}/actor/dmd_state_rank_${rank}.pt"
done
test "$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt")" = "${TOTAL_TRAIN_STEPS}"
echo 'DMD2 training and resumable FSDP checkpoint completed.'

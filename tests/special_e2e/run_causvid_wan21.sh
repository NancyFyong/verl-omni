#!/usr/bin/env bash
# Use a prior ODE adapter plus synthetic clean latents to test production orchestration.
set -euo pipefail

export NUM_GPUS=${NUM_GPUS:-8}
export MODEL_PATH=${MODEL_PATH:?set a local Wan Diffusers checkpoint}
export STUDENT_ADAPTER_PATH=${STUDENT_ADAPTER_PATH:?export an ODE student adapter first}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_wan_causvid}
export TRAIN_FILES=${DATA_DIR}/train.parquet
export VAL_FILES=${DATA_DIR}/test.parquet

python3 tests/special_e2e/create_dummy_wan_ode_data.py \
    --output_dir "${DATA_DIR}" --model_path "${MODEL_PATH}" \
    --train_size "$((NUM_GPUS * 6))" --val_size "${NUM_GPUS}" \
    --latent_frames 6 --latent_height 8 --latent_width 8
export TRAJECTORY_MANIFEST_SHA256
TRAJECTORY_MANIFEST_SHA256=$(cat "${DATA_DIR}/manifest.sha256")

bash examples/distillation_trainer/wan21/run_wan21_causvid_lora.sh \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.pipeline.height=64 \
    actor_rollout_ref.model.pipeline.width=64 \
    actor_rollout_ref.model.pipeline.num_frames=21 \
    actor_rollout_ref.model.pipeline.max_sequence_length=32 \
    data.max_prompt_length=32 \
    distillation.distribution_matching.fake_update_ratio=2 \
    trainer.logger=console \
    trainer.save_freq=1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=2 \
    "$@"

echo "Wan CausVid production smoke completed. Synthetic-data execution is not quality validation."

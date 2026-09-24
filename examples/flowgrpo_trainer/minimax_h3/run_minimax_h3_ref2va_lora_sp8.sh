#!/usr/bin/env bash
# Eight-GPU MiniMax H3 Ref2VA LoRA FlowGRPO preset with Actor SP=8 (Ulysses + padding).
#
# Thin variant of run_minimax_h3_ref2va_lora.sh: reuses the base entrypoint and
# only selects overrides. Compared with the base recipe it applies the MiniMax H3
# FL2VA V1 hyperparameters (rollout TP=2, 256x384 train / 512x768 val, 121 frames,
# SDE window [0,8]) and a smaller reference image (short edge 512). Prompt-length
# caps keep the ref2va defaults (4096 / 12288): ref2va encodes up to ~6.5k text
# tokens, so the V1 prompt cap of 1024 is too small for this task.
#
# Required environment (same contract as the base recipe):
#   DATA_DIR   - parquet directory produced by prepare_ref2va_data.py
#   MODEL_PATH - MiniMax-H3 root containing Ref2VA/ and transformer_ref/
#   IMAGEBIND_MODEL_PATH / CLAP_MODEL_PATH - reward scorer weights
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(readlink -f "$script_dir/../../..")

export N_GPUS=8
export ACTOR_SP=8
export ROLLOUT_TP=2
export TEXT_ENCODER_TP=2
export TOTAL_TRAINING_STEPS=100
# V1 hyperparameters
export HEIGHT=256
export WIDTH=384
export VAL_HEIGHT=512
export VAL_WIDTH=768
export NUM_FRAMES=121
# Smaller reference images shorten the packed condition rows; val follows train.
export REF_IMAGE_SHORT_EDGE=${REF_IMAGE_SHORT_EDGE:-512}
export OUTPUT_DIR=${OUTPUT_DIR:-$repo_root/outputs/h3-ref2va-sp8-pad-ref512-v1}

exec "$script_dir/run_minimax_h3_ref2va_lora.sh" \
    '++actor_rollout_ref.rollout.pipeline.output_type=np' \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,8]' \
    '++reward.reward_functions.clap.required=true' \
    '++reward.reward_functions.imagebind.required=true' \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.experiment_name=h3-ref2va-sp8-pad-ref512-v1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}" \
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=${VLLM_INIT_TIMEOUT:-3600} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=${VLLM_STAGE_INIT_TIMEOUT:-1800} \
    "$@"

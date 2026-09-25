#!/usr/bin/env bash
# MiniMax H3 reference-conditioned FlowGRPO with the VeOmni actor.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 repository root containing Ref2VA/transformer/}"
: "${DATA_DIR:?Set DATA_DIR to the parquet directory produced by prepare_ref2va_data.py}"
script_dir=$(dirname "$(readlink -f "$0")")
export MODEL_PATH="$MODEL_PATH/Ref2VA"
export DATA_DIR
export OUTPUT_DIR=${OUTPUT_DIR:-$script_dir/../../../outputs/run_minimax_h3_ref2va_lora_veomni}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export HEIGHT=${HEIGHT:-288} WIDTH=${WIDTH:-448}
export VAL_HEIGHT=${VAL_HEIGHT:-576} VAL_WIDTH=${VAL_WIDTH:-928}
export NUM_FRAMES=${NUM_FRAMES:-96}
export REF_IMAGE_SHORT_EDGE=${REF_IMAGE_SHORT_EDGE:-2048}
VAL_REF_IMAGE_SHORT_EDGE=${VAL_REF_IMAGE_SHORT_EDGE:-$REF_IMAGE_SHORT_EDGE}
MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-12288}

exec bash "$script_dir/run_minimax_h3_t2va_lora_veomni.sh" \
    data.max_prompt_length=4096 \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    "actor_rollout_ref.model.tokenizer_path=$MODEL_PATH/tokenizer" \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload=True \
    "actor_rollout_ref.rollout.max_prompt_embed_length=$MAX_PROMPT_EMBEDS" \
    actor_rollout_ref.rollout.pipeline.task=ref2va \
    "actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_EMBEDS" \
    "actor_rollout_ref.rollout.pipeline.reference_image_short_edge=$REF_IMAGE_SHORT_EDGE" \
    actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
    actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline.task=ref2va \
    "actor_rollout_ref.rollout.val_kwargs.pipeline.max_sequence_length=$MAX_PROMPT_EMBEDS" \
    "actor_rollout_ref.rollout.val_kwargs.pipeline.reference_image_short_edge=$VAL_REF_IMAGE_SHORT_EDGE" \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    trainer.experiment_name=minimax_h3_ref2va_lora_veomni \
    "$@"

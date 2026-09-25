#!/usr/bin/env bash
# MiniMax H3 first-frame conditioned FlowGRPO with the VeOmni actor.
set -euo pipefail

script_dir=$(dirname "$(readlink -f "$0")")
export DATA_DIR=${DATA_DIR:-${WORKSPACE:-$HOME}/data/fl2va/verl_omni}
export OUTPUT_DIR=${OUTPUT_DIR:-$script_dir/../../../outputs/run_minimax_h3_fl2va_lora_veomni}

exec bash "$script_dir/run_minimax_h3_t2va_lora_veomni.sh" \
    actor_rollout_ref.rollout.pipeline.task=fl2va \
    actor_rollout_ref.rollout.pipeline.frame_indices='[0]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.task=fl2va \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_indices='[0]' \
    trainer.experiment_name=minimax_h3_fl2va_lora_veomni \
    "$@"

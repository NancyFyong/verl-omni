#!/usr/bin/env bash
# Shared batching profile for the six H3 example entrypoints. Reuse the existing
# task/algorithm recipes, with common hyperparameters from FL2VA FlowGRPO V1;
# keep main_diffusion rather than switching trainer implementations.
set -euo pipefail

algorithm=$1
task=$2
shift 2
case "$algorithm" in
    flow_grpo) family=flowgrpo_trainer ;;
    diffusion_nft) family=diffusionnft_trainer ;;
    *) echo "Unsupported algorithm: $algorithm" >&2; exit 1 ;;
esac
case "$task" in
    t2va|fl2va) partition=FL2VA; actor_subdir=transformer ;;
    ref2va) partition=Ref2VA; actor_subdir=transformer_ref ;;
    *) echo "Unsupported task: $task" >&2; exit 1 ;;
esac

export WORKSPACE=${WORKSPACE:-$HOME}
model_root=${MODEL_PATH:-$WORKSPACE/models/MiniMax-H3}
export DATA_DIR=${DATA_DIR:-$WORKSPACE/data/$task/verl_omni}
export NUM_GPUS=${NUM_GPUS:-8}
export N_GPUS=$NUM_GPUS
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
if (( NUM_GPUS <= 0 || ROLLOUT_TP <= 0 || NUM_GPUS % ROLLOUT_TP != 0 )); then
    echo "NUM_GPUS must be a positive multiple of ROLLOUT_TP." >&2
    exit 1
fi
if (( TEXT_ENCODER_TP != 1 && TEXT_ENCODER_TP != ROLLOUT_TP )); then
    echo "TEXT_ENCODER_TP must be 1 or ROLLOUT_TP." >&2
    exit 1
fi

ROLLOUT_MODE=${ROLLOUT_MODE:-request}
case "$ROLLOUT_MODE" in
    request) step_execution=False ;;
    stepwise) step_execution=True ;;
    *) echo "ROLLOUT_MODE must be request or stepwise." >&2; exit 1 ;;
esac
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-50}
if (( MAX_NUM_SEQS < 1 || REQUEST_BATCH_MAX_WAIT_MS < 0 )); then
    echo "MAX_NUM_SEQS must be positive and REQUEST_BATCH_MAX_WAIT_MS nonnegative." >&2
    exit 1
fi

export HEIGHT=${HEIGHT:-256} WIDTH=${WIDTH:-384}
export VAL_HEIGHT=${VAL_HEIGHT:-512} VAL_WIDTH=${VAL_WIDTH:-768}
export NUM_FRAMES=${NUM_FRAMES:-121} INFER_STEPS=${INFER_STEPS:-10}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export ROLLOUT_N=${ROLLOUT_N:-8}
export ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-_flash_3_varlen_hub}
# FLASH_ATTN enables H3 multi-document packing; FLASH_ATTN_3_HUB currently does not.
export ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-FLASH_ATTN}
export MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-1024}
if [[ "$task" == ref2va ]]; then
    # Preserve capacity for multimodal references, unlike the FL2VA text limit.
    export MAX_PROMPT_EMBEDS=${REF_MAX_PROMPT_EMBEDS:-12288}
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-$repo_root/outputs/minimax_h3_${task}_${algorithm}_batch_${ROLLOUT_MODE}}
export WANDB_RUN_ID=${WANDB_RUN_ID:-minimax_h3_${task}_${algorithm}_batch_${ROLLOUT_MODE}}
export ACTOR_CONFIG_PATH="$model_root/$actor_subdir"
if [[ "$algorithm" == flow_grpo && "$task" != ref2va ]]; then
    export MODEL_PATH="$model_root/$partition"
else
    export MODEL_PATH="$model_root"
fi

args=(
    "data.train_batch_size=32"
    "data.val_max_samples=128"
    "data.max_prompt_length=1024"
    "actor_rollout_ref.model.attn_backend=$ACTOR_ATTN_BACKEND"
    "actor_rollout_ref.model.lora_rank=64"
    "actor_rollout_ref.model.lora_alpha=128"
    "actor_rollout_ref.actor.optim.lr=3e-4"
    "actor_rollout_ref.actor.optim.weight_decay=0.0001"
    "actor_rollout_ref.actor.ppo_mini_batch_size=16"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.step_execution=$step_execution"
    "actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS"
    "actor_rollout_ref.rollout.enforce_eager=True"
    "actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=$REQUEST_BATCH_MAX_WAIT_MS"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload=False"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_cpu_offload=False"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.max_prompt_embed_length=$MAX_PROMPT_EMBEDS"
    "actor_rollout_ref.rollout.pipeline.task=$task"
    "actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_EMBEDS"
    "actor_rollout_ref.rollout.val_kwargs.pipeline.task=$task"
    "actor_rollout_ref.rollout.val_kwargs.pipeline.max_sequence_length=$MAX_PROMPT_EMBEDS"
    "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40"
    "actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0"
    "reward.num_workers=${REWARD_NUM_WORKERS:-1}"
    "++reward.reward_functions.clap.model_name_or_path=${CLAP_MODEL_PATH:-laion/larger_clap_general}"
    "++reward.reward_functions.imagebind.model_name_or_path=${IMAGEBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}"
    "++reward.reward_functions.clap.device=${REWARD_DEVICE:-cuda}:0"
    "++reward.reward_functions.imagebind.device=${REWARD_DEVICE:-cuda}:1"
    "++reward.reward_functions.clap.required=True"
    "++reward.reward_functions.imagebind.required=True"
    "trainer.project_name=$algorithm"
    "trainer.experiment_name=minimax_h3_${task}_lora_batch_${ROLLOUT_MODE}"
    "trainer.save_freq=10"
    "trainer.max_actor_ckpt_to_keep=1"
    "trainer.test_freq=10"
    "trainer.total_epochs=15"
)
if [[ "$task" == fl2va ]]; then
    args+=(
        "actor_rollout_ref.rollout.pipeline.frame_indices=${FRAME_INDICES:-[0]}"
        "actor_rollout_ref.rollout.val_kwargs.pipeline.frame_indices=${FRAME_INDICES:-[0]}"
    )
elif [[ "$task" == ref2va ]]; then
    args+=("data.max_prompt_length=4096")
fi
if [[ "$algorithm" == flow_grpo ]]; then
    args+=(
        "actor_rollout_ref.rollout.algo.noise_level=0.8"
        "actor_rollout_ref.rollout.algo.sde_type=cps"
        "actor_rollout_ref.rollout.algo.sde_window_range=[0,${SDE_WINDOW_END:-8}]"
        "actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-3}"
        "actor_rollout_ref.rollout.algo.sde_contiguous=True"
        "actor_rollout_ref.rollout.algo.sde_window_seed=42"
    )
fi

echo "H3 $task / $algorithm: mode=$ROLLOUT_MODE, max_num_seqs=$MAX_NUM_SEQS, attention=$ROLLOUT_ATTN_BACKEND"
exec bash "$repo_root/examples/$family/minimax_h3/run_minimax_h3_${task}_lora.sh" "${args[@]}" "$@"

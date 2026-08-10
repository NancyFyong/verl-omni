#!/usr/bin/env bash
# Dense LingBot T2V FlowGRPO LoRA FSDP2 recipe.
# Requires structured-caption parquet from prepare_structured_captions.py.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-$WORKSPACE/models/lingbot-video-dense-1.3b}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH/processor}
TRAIN_PATH=${TRAIN_PATH:-$WORKSPACE/data/lingbot_video/train.parquet}
VAL_PATH=${VAL_PATH:-$WORKSPACE/data/lingbot_video/val.parquet}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:?Set an existing video reward function path.}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:?Set its callable name.}

HPSV3_MODEL_PATH=${HPSV3_MODEL_PATH:-$WORKSPACE/models/HPSv3/HPSv3.safetensors}
HPSV3_REWARD_DEVICE=${HPSV3_REWARD_DEVICE:-cuda}
export custom_reward_model_path=${custom_reward_model_path:-$HPSV3_MODEL_PATH}
export custom_reward_device=${custom_reward_device:-$HPSV3_REWARD_DEVICE}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ACTOR_SP=${ACTOR_SP:-1}
NUM_CPUS=${NUM_CPUS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}

PROJECT_NAME=${PROJECT_NAME:-flow_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-lingbot_dense_t2v_lora_fsdp2}
CKPT_DIR=${CKPT_DIR:-checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}
RUN_DIR=${RUN_DIR:-$CKPT_DIR}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$RUN_DIR/rollout}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$RUN_DIR/val}
LOG_FILE=${LOG_FILE:-$RUN_DIR/run.log}
WANDB_DIR=${WANDB_DIR:-$RUN_DIR/wandb}
export WANDB_DIR
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ROLLOUT_DATA_SAVE_FREQ=${ROLLOUT_DATA_SAVE_FREQ:-5}
VALIDATION_DATA_MAX_SAMPLES=${VALIDATION_DATA_MAX_SAMPLES:-8}

# Defaults target 16 prompts x 8 rollouts on 8 GPUs for 81-frame videos.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
ROLLOUT_GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
NUM_FRAMES=${NUM_FRAMES:-81}
ROLLOUT_NOISE_LEVEL=${ROLLOUT_NOISE_LEVEL:-0.7}
ROLLOUT_SDE_TYPE=${ROLLOUT_SDE_TYPE:-dance_sde}
CHECK_CONFIG_ONLY=${CHECK_CONFIG_ONLY:-False}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
PYTHON_BIN=${PYTHON_BIN:-python3}

is_truthy() {
    [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "True" ]]
}

require_positive_int() {
    local name=$1
    local value=$2

    if ! [[ "$value" =~ ^[0-9]+$ ]] || ((value <= 0)); then
        echo "[config error] $name must be a positive integer, got: $value" >&2
        exit 2
    fi
}

require_divisible() {
    local dividend_name=$1
    local dividend=$2
    local divisor_name=$3
    local divisor=$4
    local reason=$5

    if ((divisor <= 0)); then
        echo "[config error] $divisor_name must be > 0 for $reason, got: $divisor" >&2
        exit 2
    fi
    if ((dividend % divisor != 0)); then
        echo "[config error] $dividend_name=$dividend must be divisible by $divisor_name=$divisor ($reason)" >&2
        exit 2
    fi
}

validate_batch_config() {
    require_positive_int NUM_GPUS "$NUM_GPUS"
    require_positive_int ROLLOUT_TP "$ROLLOUT_TP"
    require_positive_int TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE"
    require_positive_int VAL_BATCH_SIZE "$VAL_BATCH_SIZE"
    require_positive_int ROLLOUT_GROUP_SIZE "$ROLLOUT_GROUP_SIZE"
    require_positive_int PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE"
    require_positive_int PPO_MICRO_BATCH_SIZE_PER_GPU "$PPO_MICRO_BATCH_SIZE_PER_GPU"
    require_positive_int LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    require_positive_int REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"

    require_divisible NUM_GPUS "$NUM_GPUS" ROLLOUT_TP "$ROLLOUT_TP" "rollout worker sharding"

    GLOBAL_ROLLOUT_BATCH=$((TRAIN_BATCH_SIZE * ROLLOUT_GROUP_SIZE))
    ACTOR_UPDATE_GLOBAL_MINI_BATCH=$((PPO_MINI_BATCH_SIZE * ROLLOUT_GROUP_SIZE))

    require_divisible GLOBAL_ROLLOUT_BATCH "$GLOBAL_ROLLOUT_BATCH" NUM_GPUS "$NUM_GPUS" "rollout DP sharding"
    require_divisible \
        ACTOR_UPDATE_GLOBAL_MINI_BATCH "$ACTOR_UPDATE_GLOBAL_MINI_BATCH" \
        NUM_GPUS "$NUM_GPUS" \
        "actor mini-batch sharding"

    ROLLOUT_BATCH_PER_GPU=$((GLOBAL_ROLLOUT_BATCH / NUM_GPUS))
    ACTOR_UPDATE_MINI_BATCH_PER_GPU=$((ACTOR_UPDATE_GLOBAL_MINI_BATCH / NUM_GPUS))

    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        ACTOR_UPDATE_MINI_BATCH_PER_GPU "$ACTOR_UPDATE_MINI_BATCH_PER_GPU" \
        "local rollout/train mini-batches"
    require_divisible \
        ACTOR_UPDATE_MINI_BATCH_PER_GPU "$ACTOR_UPDATE_MINI_BATCH_PER_GPU" \
        PPO_MICRO_BATCH_SIZE_PER_GPU "$PPO_MICRO_BATCH_SIZE_PER_GPU" \
        "actor micro-batches"
    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
        "old-logprob micro-batches"
    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
        "ref-logprob micro-batches"
}

print_batch_config() {
    echo "[config check] batch settings are self-consistent:" >&2
    echo "  global_rollout_batch=$GLOBAL_ROLLOUT_BATCH, rollout_batch_per_gpu=$ROLLOUT_BATCH_PER_GPU" >&2
    printf '  actor_update_global_mini_batch=%s, actor_update_mini_batch_per_gpu=%s\n' \
        "$ACTOR_UPDATE_GLOBAL_MINI_BATCH" "$ACTOR_UPDATE_MINI_BATCH_PER_GPU" >&2
    echo "  ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU" >&2
    echo "  log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" >&2
    echo "  ref_log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" >&2
    echo "  enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING" >&2
    echo "  rollout_noise_level=$ROLLOUT_NOISE_LEVEL, rollout_sde_type=$ROLLOUT_SDE_TYPE" >&2
}

validate_batch_config
print_batch_config
if is_truthy "$CHECK_CONFIG_ONLY"; then
    exit 0
fi

mkdir -p "$RUN_DIR" "$WANDB_DIR"

"$PYTHON_BIN" -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=37698 \
    data.seed=42 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    'actor_rollout_ref.model.fsdp_layer_prefixes=["blocks."]' \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_GROUP_SIZE \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=lingbot_dense_t2v_agent \
    actor_rollout_ref.rollout.prompt_length=37698 \
    actor_rollout_ref.rollout.pipeline.height=480 \
    actor_rollout_ref.rollout.pipeline.width=832 \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.0 \
    actor_rollout_ref.rollout.pipeline.shift=3.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=37698 \
    actor_rollout_ref.rollout.algo.noise_level=$ROLLOUT_NOISE_LEVEL \
    actor_rollout_ref.rollout.algo.sde_type=$ROLLOUT_SDE_TYPE \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,5]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$REWARD_FUNCTION_PATH \
    reward.custom_reward_function.name=$REWARD_FUNCTION_NAME \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$NUM_CPUS \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_model_path=$custom_reward_model_path \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_device=$custom_reward_device \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    trainer.rollout_data_save_freq=$ROLLOUT_DATA_SAVE_FREQ \
    trainer.validation_data_max_samples=$VALIDATION_DATA_MAX_SAMPLES \
    trainer.save_freq=30 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 "$@" \
    2>&1 | tee "$LOG_FILE"

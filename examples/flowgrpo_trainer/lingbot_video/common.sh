#!/usr/bin/env bash
# Shared helpers for LingBot-Video example shell scripts.

is_truthy() {
    local value=${1:-}
    [[ "$value" == "1" || "$value" == "true" || "$value" == "True" ]]
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

validate_lingbot_batch_config() {
    require_positive_int NUM_GPUS "$NUM_GPUS"
    require_positive_int ROLLOUT_TP "$ROLLOUT_TP"
    require_positive_int TRAIN_BATCH_SIZE "$TRAIN_BATCH_SIZE"
    require_positive_int VAL_BATCH_SIZE "$VAL_BATCH_SIZE"
    require_positive_int ROLLOUT_GROUP_SIZE "$ROLLOUT_GROUP_SIZE"
    require_positive_int PPO_MINI_BATCH_SIZE "$PPO_MINI_BATCH_SIZE"
    require_positive_int PPO_MICRO_BATCH_SIZE_PER_GPU "$PPO_MICRO_BATCH_SIZE_PER_GPU"
    require_positive_int LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    require_positive_int REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"

    require_divisible \
        NUM_GPUS "$NUM_GPUS" \
        ROLLOUT_TP "$ROLLOUT_TP" \
        "rollout agent workers = NUM_GPUS / ROLLOUT_TP"

    GLOBAL_ROLLOUT_BATCH=$((TRAIN_BATCH_SIZE * ROLLOUT_GROUP_SIZE))
    ACTOR_UPDATE_GLOBAL_MINI_BATCH=$((PPO_MINI_BATCH_SIZE * ROLLOUT_GROUP_SIZE))

    require_divisible \
        GLOBAL_ROLLOUT_BATCH "$GLOBAL_ROLLOUT_BATCH" \
        NUM_GPUS "$NUM_GPUS" \
        "rollout samples must shard evenly across actor/data-parallel ranks"
    require_divisible \
        ACTOR_UPDATE_GLOBAL_MINI_BATCH "$ACTOR_UPDATE_GLOBAL_MINI_BATCH" \
        NUM_GPUS "$NUM_GPUS" \
        "trainer passes actor mini_batch_size=PPO_MINI_BATCH_SIZE*ROLLOUT_GROUP_SIZE"

    ROLLOUT_BATCH_PER_GPU=$((GLOBAL_ROLLOUT_BATCH / NUM_GPUS))
    ACTOR_UPDATE_MINI_BATCH_PER_GPU=$((ACTOR_UPDATE_GLOBAL_MINI_BATCH / NUM_GPUS))

    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        ACTOR_UPDATE_MINI_BATCH_PER_GPU "$ACTOR_UPDATE_MINI_BATCH_PER_GPU" \
        "engine_workers.train_mini_batch local tensordict batch must divide local mini-batch"
    require_divisible \
        ACTOR_UPDATE_MINI_BATCH_PER_GPU "$ACTOR_UPDATE_MINI_BATCH_PER_GPU" \
        PPO_MICRO_BATCH_SIZE_PER_GPU "$PPO_MICRO_BATCH_SIZE_PER_GPU" \
        "actor forward/backward micro-batches must divide each local actor mini-batch"
    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
        "old-logprob infer micro-batches must divide local rollout samples"
    require_divisible \
        ROLLOUT_BATCH_PER_GPU "$ROLLOUT_BATCH_PER_GPU" \
        REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU "$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
        "ref logprob infer micro-batches must divide local rollout samples"
}

print_lingbot_batch_config() {
    echo "[config check] batch settings are self-consistent:" >&2
    echo "  global_rollout_batch=$GLOBAL_ROLLOUT_BATCH, rollout_batch_per_gpu=$ROLLOUT_BATCH_PER_GPU" >&2
    echo "  actor_update_global_mini_batch=$ACTOR_UPDATE_GLOBAL_MINI_BATCH, actor_update_mini_batch_per_gpu=$ACTOR_UPDATE_MINI_BATCH_PER_GPU" >&2
    echo "  ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU, log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU, ref_log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" >&2
    echo "  enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING" >&2
    echo "  rollout_noise_level=$ROLLOUT_NOISE_LEVEL, rollout_sde_type=$ROLLOUT_SDE_TYPE" >&2
}

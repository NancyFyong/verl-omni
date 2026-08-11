#!/usr/bin/env bash
# LingBot-Video MoE (30B-A3B) T2V FlowGRPO — VeOmni engine + vllm_omni rollout.
#
# Training side: VeOmni FSDP2 with expert parallelism, full fine-tune (the
# VeOmni backend rejects LoRA).  The transformer is verl-omni's VeOmni-native
# LingBot registration (verl_omni/models/veomni/lingbot_video); grouped
# experts are Shard(0)-sliced across the EP mesh and computed through the
# fused Triton grouped-GEMM MoE kernel (moe_implementation=fused_triton is
# REQUIRED whenever EXPERT_PARALLEL_SIZE > 1 — the eager path cannot dispatch
# across EP ranks and the engine config rejects that combination).
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-robbyant/lingbot-video-moe-30b-a3b}
TRAIN_PATH=${TRAIN_PATH:-$WORKSPACE/data/lingbot_video/train.parquet}
VAL_PATH=${VAL_PATH:-$WORKSPACE/data/lingbot_video/val.parquet}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:?Set an existing video reward function path.}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:?Set its callable name.}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}
# 128 experts / EP=8 -> 16 experts per rank.  Must divide both num_experts
# (asserted by ParallelPlan.apply) and the FSDP world size.
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-8}
TRAINER_BACKEND=veomni
EXPERIMENT_NAME=lingbot_moe_t2v_${TRAINER_BACKEND}_ep${EXPERT_PARALLEL_SIZE}

run_timestamp=$(date +"%Y%m%d_%H%M")
run_dir=${RUN_DIR:-$WORKSPACE/logs/$EXPERIMENT_NAME/$run_timestamp}
checkpoint_dir=${CHECKPOINT_DIR:-$run_dir/checkpoints}
rollout_data_dir=${ROLLOUT_DATA_DIR:-$run_dir/rollout_videos}
validation_data_dir=${VALIDATION_DATA_DIR:-$run_dir/val_videos}
log_file=${LOG_FILE:-$run_dir/${NODE_RANK:-0}.log}
mkdir -p "$checkpoint_dir" "$rollout_data_dir" "$validation_data_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    diffusion/model_engine=veomni_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=37698 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni_config.expert_parallel_size=$EXPERT_PARALLEL_SIZE \
    actor_rollout_ref.actor.veomni_config.moe_implementation=fused_triton \
    actor_rollout_ref.actor.veomni_config.param_offload=True \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=True \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.ref.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.ref.veomni_config.expert_parallel_size=$EXPERT_PARALLEL_SIZE \
    actor_rollout_ref.ref.veomni_config.moe_implementation=fused_triton \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=lingbot_dense_t2v_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=37698 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.0 \
    actor_rollout_ref.rollout.pipeline.shift=3.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=37698 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,5]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=480 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=832 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.custom_reward_function.path=$REWARD_FUNCTION_PATH \
    reward.custom_reward_function.name=$REWARD_FUNCTION_NAME \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.val_before_train=True \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_training_steps=10000 \
    trainer.total_epochs=100 "$@"

#!/bin/bash
# Wan2.2 LoRA RL with Flash-GRPO (temporal gradient rectification)
#
# Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (text+image-to-video, used in T2V mode)
# Algorithm: Flash-GRPO — FlowGRPO clipped PPO with per-timestep coe weighting
#            that balances gradient magnitude across denoising steps.
# Reward: HPSv3 (Human Preference Score v3) - custom reward model
#
# Key differences from the DanceGRPO recipe (examples/dancegrpo_trainer/wan22/):
#   - algorithm/adv_estimator/loss_mode = flash_grpo (instead of dance_grpo)
#   - sde_type = sde (NOT dance_sde — coe requires the FlowGRPO SDE density form)
#   - The loss automatically applies temporal gradient rectification via the coe
#     weight produced by the scheduler's sample_previous_step (return_coe=True).
#
# Note: This recipe implements the temporal-gradient-rectification component of
# Flash-GRPO (https://github.com/Shredded-Pork/Flash-GRPO). Single-step training
# and iso-temporal grouping (the paper's other two innovations) are planned for a
# future phase and are not yet active here — the training still iterates over the
# full sde_window, but with balanced per-timestep gradients.
#
# Reference: https://github.com/Shredded-Pork/Flash-GRPO
#            https://arxiv.org/abs/2605.15980
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

hpsv3_train_path=$WORKSPACE/data/hpsv3/train.parquet
hpsv3_test_path=$WORKSPACE/data/hpsv3/test.parquet

model_name=Wan-AI/Wan2.2-TI2V-5B-Diffusers
export custom_reward_model_path=$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors
custom_reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

# 8-GPU single-node configuration
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}

ENGINE=vllm_omni

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flash_grpo \
    actor_rollout_ref.model.algorithm=flash_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flash_grpo \
    data.train_files=$hpsv3_train_path \
    data.val_files=$hpsv3_test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=480 \
    actor_rollout_ref.rollout.pipeline.width=832 \
    actor_rollout_ref.rollout.pipeline.num_frames=81 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=20 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,10]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$custom_reward_function_path \
    reward.custom_reward_function.name=compute_score_hpsv3 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flash_grpo \
    trainer.experiment_name=wan22_5b_t2v_hpsv3_flash \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@" \
    2>&1 | tee run_wan22_5b_t2v_hpsv3_flash.log

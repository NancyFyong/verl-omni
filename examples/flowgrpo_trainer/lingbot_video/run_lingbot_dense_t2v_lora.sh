#!/usr/bin/env bash
# Dense LingBot T2V FlowGRPO starter recipe.  Preprocess structured captions
# first; the rollout agent rejects plain-text prompts by design.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-robbyant/lingbot-video-dense-1.3b}
TRAIN_PATH=${TRAIN_PATH:-$WORKSPACE/data/lingbot_video/train.parquet}
VAL_PATH=${VAL_PATH:-$WORKSPACE/data/lingbot_video/val.parquet}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:?Set an existing video reward function path.}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:?Set its callable name.}

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=8 \
    data.max_prompt_length=37698 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    'actor_rollout_ref.model.fsdp_layer_prefixes=["blocks."]' \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=lingbot_dense_t2v_agent \
    actor_rollout_ref.rollout.prompt_length=37698 \
    actor_rollout_ref.rollout.pipeline.height=480 \
    actor_rollout_ref.rollout.pipeline.width=832 \
    actor_rollout_ref.rollout.pipeline.num_frames=121 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.0 \
    actor_rollout_ref.rollout.pipeline.shift=3.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=37698 \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,5]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.custom_reward_function.path=$REWARD_FUNCTION_PATH \
    reward.custom_reward_function.name=$REWARD_FUNCTION_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=lingbot_dense_t2v_lora "$@"

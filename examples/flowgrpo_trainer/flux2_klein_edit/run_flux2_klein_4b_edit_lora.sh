#!/usr/bin/env bash
# FLUX.2-Klein-4B I2I LoRA RL with PickScore and DanceSDE.
set -xeuo pipefail

export RAY_DEDUP_LOGS=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

MODEL_PATH=${MODEL_PATH:-black-forest-labs/FLUX.2-klein-base-4B}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.pickscore_reward}
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8}
ACTOR_SP=${ACTOR_SP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
SDE_TYPE=${SDE_TYPE:-dance_sde}
PYTHON_BIN=${PYTHON_BIN:-python3}

WORKSPACE=${WORKSPACE:-$PWD}
TRAIN_FILES=${TRAIN_FILES:-$WORKSPACE/data/flux2_klein_edit/train.parquet}
VAL_FILES=${VAL_FILES:-$WORKSPACE/data/flux2_klein_edit/test.parquet}

"$PYTHON_BIN" -m verl_omni.trainer.main_diffusion \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=64 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.filter_overlong_prompts=False \
    data.image_key=__none__ \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','ff.linear_in','ff.linear_out','ff_context.linear_in','ff_context.linear_out','to_qkv_mlp_proj']" \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=5e-3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type=$SDE_TYPE \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$REWARD_FUNCTION_PATH \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=flux2_klein_4b_edit_pickscore_dance_sde_lora \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=checkpoints/flow_grpo/flux2_klein_4b_edit_pickscore_dance_sde_lora/rollouts \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=1000 \
    trainer.total_training_steps=1000 \
    trainer.resume_mode=disable "$@"

#!/usr/bin/env bash
# Qwen-Image four-step DMD2 LoRA training with colocated student, teacher, fake-score, and EMA roles.
set -xeuo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen-Image}
TRAIN_FILES=${TRAIN_FILES:-${WORKSPACE}/data/ocr/qwen_image/train.parquet}
VAL_FILES=${VAL_FILES:-${WORKSPACE}/data/ocr/qwen_image/test.parquet}
NUM_GPUS=${NUM_GPUS:-8}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${NUM_GPUS} \
    data.max_prompt_length=1024 \
    algorithm.trainer_type=distillation \
    algorithm.sample_source=offline \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.algorithm=dmd2 \
    actor_rollout_ref.model.model_type=diffusion_distillation_model \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0']" \
    actor_rollout_ref.model.pipeline.height=1024 \
    actor_rollout_ref.model.pipeline.width=1024 \
    actor_rollout_ref.model.pipeline.num_inference_steps=4 \
    actor_rollout_ref.model.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.rollout_timestep_shift=3.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.001 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    distillation.enabled=false \
    distillation.distribution_matching.recipe=dmd2 \
    distillation.distribution_matching.profile=distribution_only \
    distillation.distribution_matching.role_storage=shared_base_adapters \
    distillation.distribution_matching.conditioning_provider=local_frozen_encoder \
    distillation.distribution_matching.fake_update_ratio=2 \
    distillation.distribution_matching.student_micro_batch_size_per_gpu=1 \
    distillation.distribution_matching.fake_score_micro_batch_size_per_gpu=1 \
    distillation.distribution_matching.fake_score_optim.lr=2e-5 \
    distillation.distribution_matching.fake_score_optim.weight_decay=0.001 \
    distillation.distribution_matching.teacher_guidance_scale=4.0 \
    distillation.distribution_matching.teacher_cfg_norm=layer_norm \
    'distillation.distribution_matching.negative_prompt=" "' \
    distillation.distribution_matching.rollout_timestep_shift=3.0 \
    distillation.distribution_matching.score_timestep_shift=3.0 \
    distillation.distribution_matching.score_sigma_min=0.02 \
    distillation.distribution_matching.score_sigma_max=0.98 \
    distillation.distribution_matching.score_discrete_steps=1000 \
    distillation.distribution_matching.ema_decay=0.999 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=qwen-image-distillation \
    trainer.experiment_name=qwen-image-dmd2-lora \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=100 \
    trainer.total_training_steps=1000 \
    "$@"

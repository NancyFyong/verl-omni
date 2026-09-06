#!/usr/bin/env bash
# Wan 2.1 T2V causal-student ODE-regression initialization.
set -xeuo pipefail

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}
TRAIN_FILES=${TRAIN_FILES:-${WORKSPACE}/data/wan21_ode/train.parquet}
VAL_FILES=${VAL_FILES:-${WORKSPACE}/data/wan21_ode/test.parquet}
TRAJECTORY_MANIFEST_SHA256=${TRAJECTORY_MANIFEST_SHA256:?set the canonical trajectory manifest SHA-256}
NUM_GPUS=${NUM_GPUS:-8}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${NUM_GPUS} \
    data.max_prompt_length=512 \
    data.dataloader_num_workers=0 \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.wan_ode_dataset \
    data.custom_cls.name=WanODETrajectoryDataset \
    algorithm.trainer_type=distillation \
    algorithm.sample_source=offline \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.algorithm=ode_regression \
    actor_rollout_ref.model.model_type=diffusion_distillation_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0']" \
    actor_rollout_ref.model.pipeline.height=480 \
    actor_rollout_ref.model.pipeline.width=832 \
    actor_rollout_ref.model.pipeline.num_frames=81 \
    actor_rollout_ref.model.pipeline.max_sequence_length=512 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.001 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    distillation.enabled=false \
    distillation.distribution_matching.recipe=ode_regression \
    distillation.distribution_matching.causal_model_path=${MODEL_PATH} \
    distillation.distribution_matching.role_storage=shared_base_adapters \
    distillation.distribution_matching.conditioning_provider=precomputed \
    distillation.distribution_matching.student_micro_batch_size_per_gpu=1 \
    distillation.distribution_matching.frames_per_block=3 \
    distillation.distribution_matching.trajectory_manifest_sha256=${TRAJECTORY_MANIFEST_SHA256} \
    distillation.distribution_matching.ode_num_train_timesteps=1000 \
    distillation.distribution_matching.ode_loss_weight=1.0 \
    'distillation.distribution_matching.causal_denoising_timesteps=[1000,750,500,250]' \
    distillation.distribution_matching.causal_timestep_shift=8.0 \
    distillation.distribution_matching.ema_decay=0.999 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=wan21-distillation \
    trainer.experiment_name=wan21-ode-lora \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=100 \
    trainer.total_training_steps=10000 \
    "$@"

#!/usr/bin/env bash
# Qwen-Image DMD2 multi-role training smoke test.
# Requires a tiny checkpoint at ~/models/tiny-random/Qwen-Image.
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "$((NUM_GPUS * TOTAL_TRAIN_STEPS))" \
    --val_size "${NUM_GPUS}" \
    --user_prompt_only

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${NUM_GPUS} \
    data.max_prompt_length=64 \
    data.dataloader_num_workers=0 \
    algorithm.trainer_type=distillation \
    algorithm.sample_source=offline \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.algorithm=dmd2 \
    actor_rollout_ref.model.model_type=diffusion_distillation_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.rollout_timestep_shift=3.0 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.pipeline.height=64 \
    actor_rollout_ref.model.pipeline.width=64 \
    actor_rollout_ref.model.pipeline.num_inference_steps=2 \
    actor_rollout_ref.model.pipeline.max_sequence_length=64 \
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
    distillation.distribution_matching.fake_update_ratio=1 \
    distillation.distribution_matching.student_micro_batch_size_per_gpu=1 \
    distillation.distribution_matching.fake_score_micro_batch_size_per_gpu=1 \
    distillation.distribution_matching.teacher_guidance_scale=4.0 \
    distillation.distribution_matching.teacher_cfg_norm=layer_norm \
    distillation.distribution_matching.rollout_timestep_shift=3.0 \
    distillation.distribution_matching.score_timestep_shift=3.0 \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen-image-dmd2-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "Qwen-Image DMD2 e2e test passed."

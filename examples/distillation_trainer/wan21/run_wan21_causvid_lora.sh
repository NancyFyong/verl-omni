#!/usr/bin/env bash
# CausVid real-latent DMD after causal ODE initialization (not Self-Forcing).
set -euo pipefail

STUDENT_ADAPTER_PATH=${STUDENT_ADAPTER_PATH:?set the exported ODE student adapter path}
MODEL_PATH=${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}
export MODEL_PATH

bash examples/distillation_trainer/wan21/run_wan21_ode_lora.sh \
    actor_rollout_ref.model.algorithm=causvid \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    distillation.distribution_matching.recipe=causvid \
    distillation.distribution_matching.bidirectional_model_path="${MODEL_PATH}" \
    distillation.distribution_matching.student_adapter_path="${STUDENT_ADAPTER_PATH}" \
    distillation.distribution_matching.fake_update_ratio=5 \
    distillation.distribution_matching.fake_score_optim.lr=2e-6 \
    distillation.distribution_matching.fake_score_optim.weight_decay=0.01 \
    distillation.distribution_matching.teacher_guidance_scale=3.5 \
    distillation.distribution_matching.teacher_cfg_norm=none \
    distillation.distribution_matching.normalization_epsilon=0.0 \
    distillation.distribution_matching.score_timestep_shift=8.0 \
    distillation.distribution_matching.score_sigma_min=0.02 \
    distillation.distribution_matching.score_sigma_max=0.98 \
    'distillation.distribution_matching.causvid_timesteps=[1000,757,522,0]' \
    trainer.experiment_name=wan21-causvid-lora \
    "$@"

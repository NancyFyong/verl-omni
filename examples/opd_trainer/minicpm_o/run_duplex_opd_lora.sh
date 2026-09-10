#!/usr/bin/env bash
# Experimental bounded native-duplex Thinker OPD; no transcript reconstruction.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_OMNI_MINICPM_DUPLEX_OPD=1
export VERL_OMNI_DUPLEX_ARTIFACT_DIR=${VERL_OMNI_DUPLEX_ARTIFACT_DIR:-"$PWD/duplex_artifacts"}
export PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
export RESPONSE_LENGTH=1

bash "${SCRIPT_DIR}/run_simplex_opd_lora.sh" \
    data.train_batch_size=2 \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=minicpm_duplex_agent \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.pipelines.minicpm.duplex_agent_loop.MiniCPMDuplexAgentLoopManager \
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5_duplex \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=full \
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.async_chunk=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.session_mode=duplex \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_1_gpu_memory_utilization=0.10 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_2_gpu_memory_utilization=0.10 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=1 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=true \
    distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5_duplex \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_mode=thinker_only \
    distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.async_chunk=false \
    trainer.experiment_name=minicpm-o45-duplex \
    trainer.total_training_steps=3 \
    trainer.save_freq=1 \
    "$@"

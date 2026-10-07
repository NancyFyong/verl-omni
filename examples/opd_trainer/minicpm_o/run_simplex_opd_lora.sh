#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker on-policy distillation (OPD) on AVQA: the GSPO AVQA recipe
# plus a frozen vllm_omni teacher pool that scores every student rollout.
#
# Student data, actor, and rollout settings come from
# examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh
# (prepare its AVQA data first); this wrapper selects the student checkpoint and
# adds the distillation group, as the Qwen3-Omni OPD recipe does for its GSPO base.
# The teacher must share the student's tokenizer and multimodal preprocessing.
#
# Defaults: four student GPUs (from the base recipe) plus TEACHER_GPUS teacher GPUs
# on the same node. Teacher prompt/response lengths must match the student's.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a MiniCPM-o 4.5 checkpoint}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a frozen checkpoint sharing the student tokenizer}"

TEACHER_GPUS=${TEACHER_GPUS:-4}
TEACHER_TP=${TEACHER_TP:-2}
if (( TEACHER_TP <= 0 || TEACHER_GPUS % TEACHER_TP != 0 )); then
    echo "TEACHER_GPUS (${TEACHER_GPUS}) must be a positive multiple of TEACHER_TP (${TEACHER_TP})." >&2
    exit 1
fi

MODEL_PATH="${STUDENT_MODEL}" exec bash "${REPO_ROOT}/examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh" \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm_omni \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
    distillation.teacher_models.teacher_model.inference.max_model_len=16640 \
    distillation.teacher_models.teacher_model.inference.prompt_length=4160 \
    distillation.teacher_models.teacher_model.inference.response_length=12288 \
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=false \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.output_mode="ar" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.trust_remote_code=true \
    distillation.distillation_loss.loss_mode=kl \
    distillation.distillation_loss.use_policy_gradient=true \
    trainer.project_name=opd \
    trainer.experiment_name=minicpm_o_4_5_thinker_lora_avqa_opd \
    "$@"

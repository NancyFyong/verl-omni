#!/usr/bin/env bash
# MiniMax H3 T2VA FlowGRPO LoRA training with vLLM-Omni rollout.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the official MiniMax-H3 FL2VA rollout checkpoint}"
: "${ACTOR_TRANSFORMER_PATH:?Set ACTOR_TRANSFORMER_PATH to the converted Diffusers transformer}"
: "${DATA_DIR:?Set DATA_DIR to a directory containing train.parquet and test.parquet}"

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-448}
# Required by H3 t2va; one of 21:9, 16:9, 4:3, 1:1, 3:4, 9:16. HEIGHT/WIDTH still set the pixel canvas.
ASPECT_RATIO=${ASPECT_RATIO:-16:9}
NUM_FRAMES=${NUM_FRAMES:-124}
INFER_STEPS=${INFER_STEPS:-50}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-512}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
NOISE_LEVEL=${NOISE_LEVEL:-0.7}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-"[0,$((INFER_STEPS - 1))]"}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}
ENABLE_LAYERWISE_OFFLOAD=${ENABLE_LAYERWISE_OFFLOAD:-True}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/minimax_h3_t2va_flow_grpo}
# ImageBind ships no HF repo; see the README for the download step.
IMAGEBIND_CKPT=${IMAGEBIND_CKPT:-.checkpoints/imagebind_huge.pth}
CLAP_MODEL=${CLAP_MODEL:-laion/larger_clap_general}
CLAP_DEVICE=${CLAP_DEVICE:-cuda:0}
IMAGEBIND_DEVICE=${IMAGEBIND_DEVICE:-cuda:1}

repo_root=$(dirname "$(readlink -f "$0")")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
  repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
  echo "Unable to locate repo root from $0: no LICENSE found" >&2
  exit 1
fi

if (( N_GPUS % ROLLOUT_TP != 0 )); then
  echo "N_GPUS ($N_GPUS) must be divisible by ROLLOUT_TP ($ROLLOUT_TP)." >&2
  exit 1
fi
if (( TEXT_ENCODER_TP > ROLLOUT_TP )); then
  echo "TEXT_ENCODER_TP ($TEXT_ENCODER_TP) must not exceed ROLLOUT_TP ($ROLLOUT_TP)." >&2
  exit 1
fi
if (( ROLLOUT_N < 2 )); then
  echo "ROLLOUT_N must be at least 2 for group-relative advantages." >&2
  exit 1
fi
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by N_GPUS for FSDP actor dispatch." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 -m verl_omni.trainer.main_diffusion \
  algorithm.trainer_type=policy_gradient \
  algorithm.sample_source=online \
  algorithm.adv_estimator=flow_grpo \
  algorithm.adv_mode=continuous \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_SEQUENCE_LENGTH" \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/tokenizer" \
  actor_rollout_ref.model.config_path="$ACTOR_TRANSFORMER_PATH" \
  +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
  actor_rollout_ref.model.external_lib=verl_omni.pipelines.minimax_h3_flow_grpo \
  actor_rollout_ref.model.algorithm=flow_grpo \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]' \
  actor_rollout_ref.model.attn_backend="$ACTOR_ATTN_BACKEND" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=3e-4 \
  actor_rollout_ref.actor.optim.weight_decay=0.0001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
  actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.rollout_attn_backend="$ROLLOUT_ATTN_BACKEND" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.agent.num_workers="$((N_GPUS / ROLLOUT_TP))" \
  actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
  actor_rollout_ref.rollout.max_prompt_embed_length="$MAX_SEQUENCE_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.text_encoder_tp_size="$TEXT_ENCODER_TP" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_layerwise_offload="$ENABLE_LAYERWISE_OFFLOAD" \
  actor_rollout_ref.rollout.pipeline.aspect_ratio="$ASPECT_RATIO" \
  actor_rollout_ref.rollout.pipeline.height="$HEIGHT" \
  actor_rollout_ref.rollout.pipeline.width="$WIDTH" \
  actor_rollout_ref.rollout.pipeline.num_frames="$NUM_FRAMES" \
  actor_rollout_ref.rollout.pipeline.num_inference_steps="$INFER_STEPS" \
  actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
  actor_rollout_ref.rollout.pipeline.max_sequence_length="$MAX_SEQUENCE_LENGTH" \
  actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
  actor_rollout_ref.rollout.pipeline.audio_flow_shift=3.0 \
  actor_rollout_ref.rollout.pipeline.av_logprob_video_weight=1.0 \
  actor_rollout_ref.rollout.pipeline.av_logprob_audio_weight=1.0 \
  actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
  actor_rollout_ref.rollout.algo.noise_level="$NOISE_LEVEL" \
  actor_rollout_ref.rollout.algo.sde_type=sde \
  actor_rollout_ref.rollout.algo.sde_window_size="$SDE_WINDOW_SIZE" \
  actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE" \
  actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
  reward.num_workers=1 \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
  reward.custom_reward_function.name=_multi_reward_placeholder \
  reward.reward_manager.name=MultiVisualRewardManager \
  reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
  "+reward.reward_functions.clap.path=$repo_root/verl_omni/utils/reward_score/clap.py" \
  '+reward.reward_functions.clap.name=compute_score' \
  '+reward.reward_functions.clap.weight=1.0' \
  "+reward.reward_functions.clap.device=$CLAP_DEVICE" \
  "+reward.reward_functions.clap.model_name_or_path=$CLAP_MODEL" \
  "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
  '+reward.reward_functions.imagebind.name=compute_score' \
  '+reward.reward_functions.imagebind.weight=1.0' \
  "+reward.reward_functions.imagebind.device=$IMAGEBIND_DEVICE" \
  "+reward.reward_functions.imagebind.model_name_or_path=$IMAGEBIND_CKPT" \
  '+reward.reward_functions.imagebind.mode=audio_video' \
  reward.aggregation=weighted_sum \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=flow_grpo \
  trainer.experiment_name=minimax_h3_t2va_lora \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.log_val_generations=4 \
  trainer.video_fps=24 \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  "$@"

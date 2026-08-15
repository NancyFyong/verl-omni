#!/usr/bin/env bash
# MiniMax H3 T2VA DiffusionNFT LoRA training with vLLM-Omni rollout.
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
NUM_FRAMES=${NUM_FRAMES:-124}
INFER_STEPS=${INFER_STEPS:-50}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-512}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}
ENABLE_LAYERWISE_OFFLOAD=${ENABLE_LAYERWISE_OFFLOAD:-True}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/minimax_h3_t2va_diffusion_nft}

if (( N_GPUS % ROLLOUT_TP != 0 )); then
  echo "N_GPUS ($N_GPUS) must be divisible by ROLLOUT_TP ($ROLLOUT_TP)." >&2
  exit 1
fi
if (( TEXT_ENCODER_TP > ROLLOUT_TP )); then
  echo "TEXT_ENCODER_TP ($TEXT_ENCODER_TP) must not exceed ROLLOUT_TP ($ROLLOUT_TP)." >&2
  exit 1
fi
if (( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS != 0 )); then
  echo "TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by N_GPUS for FSDP actor dispatch." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python3 -m verl_omni.trainer.main_diffusion \
  algorithm.trainer_type=direct_preference \
  algorithm.sample_source=online \
  algorithm.adv_mode=continuous \
  algorithm.timestep_fraction=1.0 \
  algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
  algorithm.old_policy_update_interval=2 \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_SEQUENCE_LENGTH" \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/tokenizer" \
  actor_rollout_ref.model.config_path="$ACTOR_TRANSFORMER_PATH" \
  +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
  actor_rollout_ref.model.external_lib=verl_omni.pipelines.minimax_h3_diffusion_nft \
  actor_rollout_ref.model.algorithm=diffusion_nft \
  actor_rollout_ref.model.model_type=diffusion_nft_model \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
  actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]' \
  actor_rollout_ref.model.attn_backend="$ACTOR_ATTN_BACKEND" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=3e-4 \
  actor_rollout_ref.actor.optim.weight_decay=0.0001 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
  actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
  actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
  actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
  actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.rollout_attn_backend="$ROLLOUT_ATTN_BACKEND" \
  actor_rollout_ref.rollout.rollout_adapter=old \
  actor_rollout_ref.rollout.calculate_log_probs=False \
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
  actor_rollout_ref.rollout.pipeline.height="$HEIGHT" \
  actor_rollout_ref.rollout.pipeline.width="$WIDTH" \
  actor_rollout_ref.rollout.pipeline.num_frames="$NUM_FRAMES" \
  actor_rollout_ref.rollout.pipeline.num_inference_steps="$INFER_STEPS" \
  actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
  actor_rollout_ref.rollout.pipeline.max_sequence_length="$MAX_SEQUENCE_LENGTH" \
  actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
  actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
  reward.num_workers=1 \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path=verl_omni/utils/reward_score/pickscore_reward.py \
  reward.custom_reward_function.name=compute_score_pickscore \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=diffusion_nft \
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

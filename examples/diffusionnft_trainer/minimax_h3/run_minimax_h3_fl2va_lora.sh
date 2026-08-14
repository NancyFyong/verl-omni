#!/usr/bin/env bash
# MiniMax H3 FL2VA DiffusionNFT. MODEL_PATH is the official rollout checkpoint;
# ACTOR_TRANSFORMER_PATH is the converted diffusers transformer for FSDP.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the official MiniMax-H3 FL2VA rollout checkpoint}"
: "${ACTOR_TRANSFORMER_PATH:?Set ACTOR_TRANSFORMER_PATH to the converted diffusers transformer}"
: "${DATA_DIR:?Set DATA_DIR to the parquet directory produced by prepare_data.py}"

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-448}
NUM_FRAMES=${NUM_FRAMES:-96}
INFER_STEPS=${INFER_STEPS:-50}
MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-1024}
FRAME_INDICES=${FRAME_INDICES:-'[0,-1]'}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}

python3 -m verl_omni.trainer.main_diffusion \
  algorithm.trainer_type=direct_preference \
  algorithm.sample_source=online \
  algorithm.adv_mode=continuous \
  algorithm.timestep_fraction=1.0 \
  algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
  algorithm.old_policy_update_interval=2 \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=2 \
  data.max_prompt_length=512 \
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
  actor_rollout_ref.actor.optim.lr=3e-4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.rollout.name=vllm_omni \
  actor_rollout_ref.rollout.rollout_attn_backend="$ROLLOUT_ATTN_BACKEND" \
  actor_rollout_ref.rollout.rollout_adapter=old \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.enable_cpu_offload=True \
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_N" \
  actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
  actor_rollout_ref.rollout.max_prompt_embed_length="$MAX_PROMPT_EMBEDS" \
  actor_rollout_ref.rollout.pipeline.task=fl2va \
  actor_rollout_ref.rollout.pipeline.frame_indices="$FRAME_INDICES" \
  actor_rollout_ref.rollout.pipeline.height="$HEIGHT" \
  actor_rollout_ref.rollout.pipeline.width="$WIDTH" \
  actor_rollout_ref.rollout.pipeline.num_frames="$NUM_FRAMES" \
  actor_rollout_ref.rollout.pipeline.num_inference_steps="$INFER_STEPS" \
  actor_rollout_ref.rollout.pipeline.max_sequence_length="$MAX_PROMPT_EMBEDS" \
  actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
  actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
  reward.reward_model.enable=False \
  reward.custom_reward_function.path=verl_omni/utils/reward_score/pickscore_reward.py \
  reward.custom_reward_function.name=compute_score_pickscore \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.video_fps=24 \
  trainer.val_before_train=False \
  "$@"

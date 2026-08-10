# LingBot Dense T2V LoRA FSDP2 RL with HPSv3 reward.
set -x

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}
model_name=${MODEL_PATH:-$WORKSPACE/models/lingbot-video-dense-1.3b}
tokenizer_path=${TOKENIZER_PATH:-$model_name/processor}
reward_function_path=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.hpsv3_reward}

export custom_reward_model_path=${custom_reward_model_path:-$WORKSPACE/models/HPSv3/HPSv3.safetensors}
export custom_reward_device=${custom_reward_device:-cuda}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8}
ACTOR_SP=${ACTOR_SP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
REWARD_WORKERS=${REWARD_WORKERS:-1}
NUM_CPUS=${NUM_CPUS:-64}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-37698}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-480}
IMAGE_WIDTH=${IMAGE_WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
ROLLOUT_GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
ROLLOUT_NOISE_LEVEL=${ROLLOUT_NOISE_LEVEL:-0.7}
ROLLOUT_SDE_TYPE=${ROLLOUT_SDE_TYPE:-dance_sde}

ENGINE=vllm_omni

train_path=${TRAIN_FILES:-$WORKSPACE/data/lingbot_video/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/lingbot_video/val.parquet}

output_dir=$WORKSPACE/outputs/lingbot_dense_t2v_lora_fsdp2
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_videos
val_data_dir=$output_dir/logs/$run_timestamp/val_videos
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_path \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    'actor_rollout_ref.model.fsdp_layer_prefixes=["blocks."]' \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_GROUP_SIZE \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=lingbot_dense_t2v_agent \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_WIDTH \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.0 \
    actor_rollout_ref.rollout.pipeline.shift=3.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=$ROLLOUT_NOISE_LEVEL \
    actor_rollout_ref.rollout.algo.sde_type=$ROLLOUT_SDE_TYPE \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    reward.num_workers=$REWARD_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_hpsv3 \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=lingbot_dense_t2v_lora_fsdp2 \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$val_data_dir \
    trainer.rollout_data_save_freq=5 \
    trainer.validation_data_max_samples=8 \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=$NUM_CPUS \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_model_path=$custom_reward_model_path \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_device=$custom_reward_device \
    trainer.save_freq=30 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto "$@"

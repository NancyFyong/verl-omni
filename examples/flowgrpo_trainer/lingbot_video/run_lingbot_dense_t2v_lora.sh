# LingBot Dense T2V LoRA RL with HPSv3 reward.
set -x

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}
model_name=${MODEL_PATH:-$WORKSPACE/models/lingbot-video-dense-1.3b}
tokenizer_path=${TOKENIZER_PATH:-$model_name/processor}
reward_function_path=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.hpsv3_reward}

export custom_reward_model_path=${custom_reward_model_path:-$WORKSPACE/models/HPSv3/HPSv3.safetensors}
export custom_reward_device=${custom_reward_device:-cuda}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_WORKERS=${REWARD_WORKERS:-1}
NUM_CPUS=${NUM_CPUS:-64}
ENGINE=${ENGINE:-vllm_omni}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-flow_grpo}
MODEL_ALGORITHM=${MODEL_ALGORITHM:-flow_grpo}
LOSS_MODE=${LOSS_MODE:-flow_grpo}
DATA_SEED=${DATA_SEED:-42}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-37698}

IMAGE_HEIGHT=${IMAGE_HEIGHT:-480}
IMAGE_WIDTH=${IMAGE_WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-81}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
VAL_NUM_INFERENCE_STEPS=${VAL_NUM_INFERENCE_STEPS:-40}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.0}
FLOW_SHIFT=${FLOW_SHIFT:-3.0}
ROLLOUT_NOISE_LEVEL=${ROLLOUT_NOISE_LEVEL:-0.7}
VAL_NOISE_LEVEL=${VAL_NOISE_LEVEL:-0.0}
ROLLOUT_SDE_TYPE=${ROLLOUT_SDE_TYPE:-dance_sde}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-"[0,5]"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
ROLLOUT_GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}

LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DTYPE=${LORA_DTYPE:-bf16}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
FSDP_LAYER_PREFIXES=${FSDP_LAYER_PREFIXES:-'["blocks."]'}
TARGET_MODULES=${TARGET_MODULES:-"['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']"}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-True}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}

LOAD_FORMAT=${LOAD_FORMAT:-safetensors}
LAYERED_SUMMON=${LAYERED_SUMMON:-True}
AGENT_LOOP=${AGENT_LOOP:-lingbot_dense_t2v_agent}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score_hpsv3}
REWARD_MODEL_ENABLE=${REWARD_MODEL_ENABLE:-False}

PROJECT_NAME=${PROJECT_NAME:-flow_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-lingbot_dense_t2v_lora}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console", "tensorboard", "wandb"]'}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
NNODES=${NNODES:-1}
SAVE_FREQ=${SAVE_FREQ:-30}
TEST_FREQ=${TEST_FREQ:-30}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
RESUME_MODE=${RESUME_MODE:-auto}
ROLLOUT_DATA_SAVE_FREQ=${ROLLOUT_DATA_SAVE_FREQ:-5}
VALIDATION_DATA_MAX_SAMPLES=${VALIDATION_DATA_MAX_SAMPLES:-32}

train_path=${TRAIN_FILES:-$WORKSPACE/data/lingbot_video/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/lingbot_video/val.parquet}

output_dir=${OUTPUT_DIR:-$WORKSPACE/outputs/$EXPERIMENT_NAME}
checkpoint_dir=${CHECKPOINT_DIR:-$output_dir/checkpoints}
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=${LOG_FILE:-$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log}
rollout_data_dir=${ROLLOUT_DATA_DIR:-$output_dir/logs/$run_timestamp/rollout_videos}
val_data_dir=${VALIDATION_DATA_DIR:-$output_dir/logs/$run_timestamp/val_videos}
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=$DATA_SEED \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_path \
    actor_rollout_ref.model.algorithm=$MODEL_ALGORITHM \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.model.fsdp_layer_prefixes="$FSDP_LAYER_PREFIXES" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=$LORA_DTYPE \
    actor_rollout_ref.model.target_modules="$TARGET_MODULES" \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=$LOSS_MODE \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.load_format=$LOAD_FORMAT \
    actor_rollout_ref.rollout.layered_summon=$LAYERED_SUMMON \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=$ROLLOUT_GROUP_SIZE \
    actor_rollout_ref.rollout.seed=$ROLLOUT_SEED \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=$AGENT_LOOP \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_WIDTH \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.guidance_scale=$GUIDANCE_SCALE \
    actor_rollout_ref.rollout.pipeline.shift=$FLOW_SHIFT \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=$ROLLOUT_NOISE_LEVEL \
    actor_rollout_ref.rollout.algo.sde_type=$ROLLOUT_SDE_TYPE \
    actor_rollout_ref.rollout.algo.sde_window_size=$SDE_WINDOW_SIZE \
    actor_rollout_ref.rollout.algo.sde_window_range="$SDE_WINDOW_RANGE" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=$VAL_NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=$VAL_NOISE_LEVEL \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    reward.num_workers=$REWARD_WORKERS \
    reward.reward_model.enable=$REWARD_MODEL_ENABLE \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=$REWARD_FUNCTION_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$val_data_dir \
    trainer.rollout_data_save_freq=$ROLLOUT_DATA_SAVE_FREQ \
    trainer.validation_data_max_samples=$VALIDATION_DATA_MAX_SAMPLES \
    trainer.log_val_generations=$LOG_VAL_GENERATIONS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=$NNODES \
    ray_kwargs.ray_init.num_cpus=$NUM_CPUS \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_model_path=$custom_reward_model_path \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_device=$custom_reward_device \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=$RESUME_MODE "$@"

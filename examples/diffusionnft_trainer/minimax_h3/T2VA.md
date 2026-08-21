# MiniMax H3 T2VA DiffusionNFT

Text-to-audio-video recipe: trains a rank-64 MiniMax H3 LoRA with online
DiffusionNFT. A Diffusers transformer is trained with FSDP2 while vLLM-Omni
generates joint video and audio rollouts. CLAP and ImageBind provide the
default multi-reward (audio-video alignment).

For the image-conditioned FL2VA variant, see [README.md](README.md) for the
data pipeline and the FL2VA PR for the training recipe.

## Install

Follow the project [installation guide](../../../docs/start/install.md), then
install the repository-pinned vLLM-Omni revision:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@245d78fb48f1c87dfb560a94be6e191c9f9f1c0"
```

The explicit Diffusers revision is the tested API target that provides
`MiniMaxH3Transformer3DModel`.

## Checkpoint

`MODEL_PATH` points at the official MiniMax-H3 `FL2VA` rollout directory
(tokenizer, text encoder, VAE, fused QKV and GEGLU transformer used by
vLLM-Omni), while `ACTOR_TRANSFORMER_PATH` points at the converted Diffusers
`MiniMaxH3Transformer3DModel` for FSDP training. Do not replace the official
rollout transformer with a symlink to the Diffusers conversion.

## Prepare data

Convert prompt lists to text-only parquet:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

Input is `train.txt`/`test.txt` (one prompt per line) or
`train.jsonl`/`test.jsonl` (`prompt`/`text`/`caption` fields).

## Launch

```bash
export MODEL_PATH=/path/to/MiniMax-H3/FL2VA
export ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer
export DATA_DIR=/path/to/h3_t2va_data

bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

MiniMax H3 t2va requires an explicit named `aspect_ratio` (one of
`21:9/16:9/4:3/1:1/3:4/9:16`); the launch script sets `16:9` and explicit
`height`/`width` control the actual canvas (must be multiples of 32).

The H3-specific agent loop (`minimax_h3_diffusion_single_turn_agent`) is
required: it tokenizes raw text once and sends those token IDs directly to the
H3 text encoder.

The checkpoint is not a low-step distilled model; `INFER_STEPS=50` is the
default. Common overrides:

```bash
N_GPUS=8 ROLLOUT_TP=4 ROLLOUT_N=4 INFER_STEPS=50 \
TOTAL_TRAINING_STEPS=100 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

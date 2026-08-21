# MiniMax H3 DiffusionNFT Recipes

Two recipes share this directory: **T2VA** (text-to-audio-video) and **FL2VA**
(first/last-frame-conditioned text-to-audio-video). Both train rank-64 MiniMax H3
LoRA with online DiffusionNFT: a Diffusers transformer is trained with FSDP2
while vLLM-Omni generates joint video and audio rollouts. CLAP and ImageBind
provide the default multi-reward (audio-video alignment).

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

Both recipes consume the same two checkpoint layouts. `MODEL_PATH` points at
the official MiniMax-H3 `FL2VA` rollout directory (tokenizer, text encoder,
VAE, fused QKV and GEGLU transformer used by vLLM-Omni), while
`ACTOR_TRANSFORMER_PATH` points at the converted Diffusers
`MiniMaxH3Transformer3DModel` for FSDP training. Do not replace the official
rollout transformer with a symlink to the Diffusers conversion — that silently
breaks the rollout weight loader.

## T2VA (text-to-audio-video)

### Prepare data

The recipe accepts RLHFDataset-compatible text-only parquet files. Convert
prompt lists with:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

Input is `train.txt`/`test.txt` (one prompt per line) or
`train.jsonl`/`test.jsonl` (`prompt`/`text`/`caption` fields).

### Launch

```bash
export MODEL_PATH=/path/to/MiniMax-H3/FL2VA
export ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer
export DATA_DIR=/path/to/h3_t2va_data

bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

MiniMax H3 t2va requires an explicit named `aspect_ratio` (one of
`21:9/16:9/4:3/1:1/3:4/9:16`); the launch script sets `16:9` and explicit
`height`/`width` control the actual canvas (must be multiples of 32).

## FL2VA (image-conditioned)

> The FL2VA scripts (`prepare_data.py`, `run_minimax_h3_fl2va_lora.sh`, and
> the checkpoint parity test) ship in the companion FL2VA PR.

### Prepare data

Prepare `train.jsonl` and `test.jsonl`. Each row has a prompt and either an
`images` list or explicit first/last names:

```json
{"prompt":"A sunrise becomes a starry night.","images":["images/first.png","images/last.png"]}
```

Convert with:

```bash
python examples/diffusionnft_trainer/minimax_h3/prepare_data.py \
  --input_dir /path/to/raw \
  --output_dir /path/to/parquet \
  --frame_mode first_last
```

`frame_mode` can be `first`, `last`, or `first_last`. Set the matching launcher
value: `export FRAME_INDICES='[0,-1]'` (or `'[0]'` / `'[-1]'`).

### Launch

```bash
MODEL_PATH=/path/to/MiniMax-H3/FL2VA \
ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer \
DATA_DIR=/path/to/parquet \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

The vLLM-Omni contract requires 4–15 seconds at 24 FPS. `NUM_FRAMES=96` is
aligned by vLLM-Omni to the next valid `17n+5` boundary. The checkpoint is not
a low-step distilled model, so `INFER_STEPS=50` is the default; 2–4 steps are
suitable only for contract smoke tests.

To verify that the two checkpoint directories represent the same base policy
after fused-QKV and GEGLU conversion:

```bash
python tests/special_e2e/minimax_h3_checkpoint_parity.py \
  --vllm-transformer "$MODEL_PATH/transformer" \
  --diffusers-transformer "$ACTOR_TRANSFORMER_PATH"
```

## Shared notes

- The H3-specific agent loop (`minimax_h3_diffusion_single_turn_agent`) is
  required for both recipes: it tokenizes raw text once and sends those token
  IDs directly to the H3 text encoder. Replacing it with the generic diffusion
  agent loop changes the prompt contract.
- The actor uses `native` attention and rollout uses `TORCH_SDPA` by default.
  Override `ACTOR_ATTN_BACKEND` and `ROLLOUT_ATTN_BACKEND` together only after
  validating the replacement backends.
- Common environment overrides:

```bash
N_GPUS=8 ROLLOUT_TP=4 ROLLOUT_N=4 INFER_STEPS=50 \
TOTAL_TRAINING_STEPS=100 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

## Offline first-frame data generation

An offline data pipeline turns a prompt list into FLUX reference images and
train/test JSONL pairs:

- `gen_flux_images.py` — multi-GPU FLUX batch image generator (prompt file ->
  one JPEG per prompt, deterministic per-index seeds, resume by skipping
  existing files). Defaults mirror DanceGRPO's online reference pipeline
  (400x640, 30 steps, guidance 3.5, max_sequence_length 512).
- `build_fl2va_jsonl.py` — pairs each prompt with its same-index image,
  shuffles with a fixed seed, and writes `train.jsonl` / `test.jsonl` with
  relative paths for `prepare_data.py`.

A ready-made dataset built with this pipeline (27,815 prompt/image pairs from
the DanceGRPO ConsisID prompt list) is published at
[zyfenghit/dancegrpo-t2av](https://huggingface.co/datasets/zyfenghit/dancegrpo-t2av).

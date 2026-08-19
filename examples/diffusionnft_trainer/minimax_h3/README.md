# MiniMax H3 T2VA DiffusionNFT

This recipe trains a rank-64 MiniMax H3 LoRA with online DiffusionNFT. A
Diffusers transformer is trained with FSDP2, while vLLM-Omni generates joint
video and audio rollouts from text prompts. PickScore provides the default
video reward.

## Install

Follow the project [installation guide](../../../docs/start/install.md), then
install the repository-pinned vLLM-Omni revision:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@245d78fb48f1c87dfb560a94bea6e191c9f9f1c0"
```

The explicit Diffusers revision is the tested API target that provides
`MiniMaxH3Transformer3DModel`.

## Prepare the model

The actor and rollout consume different transformer checkpoint layouts:

- `MODEL_PATH` must point to the official MiniMax-H3 `FL2VA` rollout directory,
  including the tokenizer, text encoder, VAE, and official fused transformer.
- `ACTOR_TRANSFORMER_PATH` must point to the converted Diffusers transformer
  directory containing its `config.json` and safetensor shards.

Do not replace the official rollout transformer with a symlink to the
Diffusers transformer. vLLM-Omni expects fused QKV and GEGLU weights, while the
actor expects the Diffusers split layout.

## Prepare text prompts

The recipe accepts RLHFDataset-compatible text-only parquet files. The LTX-2
prompt converter can be reused for `train.txt`/`test.txt` or
`train.jsonl`/`test.jsonl` inputs:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

Each JSONL row may use a `prompt`, `text`, or `caption` field.

## Launch

```bash
export MODEL_PATH=/path/to/MiniMax-H3/FL2VA
export ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer
export DATA_DIR=/path/to/h3_t2va_data

bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

The launch script explicitly selects:

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent
```

This AgentLoop extracts the plain text from each dataset message, tokenizes it
with `add_special_tokens=False`, and sends those token IDs directly to the H3
text encoder. The H3 rollout rejects generic chat-template token IDs instead
of silently decoding and re-tokenizing them.

The default topology uses eight GPUs, rollout TP=4, text-encoder TP=2, four
rollouts per prompt, FSDP2 actor training, and rank-64/alpha-128 LoRA. The actor
uses native attention and rollout uses `TORCH_SDPA`; override both backends
together if using a matched FA3 installation.

The checkpoint is not a 2–4-step distilled model. The recipe therefore defaults
to 50 inference steps. Lower values are useful only for contract smoke tests
and generally produce noisy video and audio.

Common environment overrides include:

```bash
N_GPUS=8 \
ROLLOUT_TP=4 \
TEXT_ENCODER_TP=2 \
ROLLOUT_N=4 \
INFER_STEPS=50 \
TOTAL_TRAINING_STEPS=100 \
OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

Additional Hydra overrides may be appended to the command.

## FLUX first-frame data (FL2VA)

For the image-conditioned FL2VA variant, an offline data pipeline turns a
prompt list into FLUX reference images and train/test JSONL pairs:

- `gen_flux_images.py` — multi-GPU FLUX batch image generator (prompt file ->
  one JPEG per prompt, deterministic per-index seeds, resume by skipping
  existing files). Defaults mirror DanceGRPO's online reference pipeline
  (400x640, 30 steps, guidance 3.5, max_sequence_length 512).
- `build_fl2va_jsonl.py` — pairs each prompt with its same-index image,
  shuffles with a fixed seed, and writes `train.jsonl` / `test.jsonl` with
  relative paths for `prepare_data.py`.

A ready-made dataset built with this pipeline (27,815 prompt/image pairs from
the DanceGRPO ConsisID prompt list) is published at
https://huggingface.co/datasets/zyfenghit/dancegrpo-t2av

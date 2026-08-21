# MiniMax H3 FL2VA First-Frame Data

Last updated: 08/21/2026

Offline data pipeline for MiniMax H3 FL2VA (text+image to audio-video) RL
training: turn a prompt list into FLUX reference images, pair them into
train/test JSONL, and feed the FL2VA `prepare_data.py` converter.

A ready-made dataset built with this pipeline is published at
https://huggingface.co/datasets/zyfenghit/dancegrpo-t2av

## Pipeline

```
prompts.txt ──► gen_flux_images.py ──► images/{index:06d}.jpg
                     │                        │
                     │                        ▼
                     │              build_fl2va_jsonl.py
                     │                        │
                     ▼                        ▼
              (same index)          train.jsonl / test.jsonl
                                             │
                                             ▼
                              prepare_data.py --frame_mode first
                                             │
                                             ▼
                                train.parquet / test.parquet
```

## Getting the prompt file

`dancegrpo_consist-id.txt` is the filtered ConsisID prompt list released by
DanceGRPO (27,815 prompts, one per line). Download it directly from the
DanceGRPO repository:

```bash
curl -L -o dancegrpo_consist-id.txt \
  https://raw.githubusercontent.com/XueZeyue/DanceGRPO/main/assets/consist-id.txt
```

The prompts originate from
[ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
captions; DanceGRPO ships the filtered result as-is and does not document the
exact filtering criteria, so downloading the released file is the reproducible
way to get the identical prompt set (verified line-for-line against the copy
used for the published dataset).

## Scripts

### `gen_flux_images.py`

Multi-GPU FLUX batch image generator. Reads one prompt per line, shards the
prompts across ranks (`torchrun`), and writes one JPEG per prompt plus a
per-rank `metadata_rankN.jsonl` (image <-> prompt <-> index mapping).
Deterministic per-prompt seed (`seed + index`), and re-running skips images
that already exist, so interrupted jobs resume safely. Defaults mirror
DanceGRPO's online reference pipeline (400x640, 30 steps, guidance 3.5,
max_sequence_length 512), so prompt and condition image are semantically
aligned by construction.

```bash
torchrun --nproc_per_node=8 examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --model_path /path/to/FLUX.1-dev \
    --output_dir data/flux_images \
    --height 400 --width 640
```

### `build_fl2va_jsonl.py`

Pairs each prompt with its same-index image, verifies all images exist,
shuffles with a fixed seed, and writes `train.jsonl` / `test.jsonl` with
relative image paths — the input format of `prepare_data.py`:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/build_fl2va_jsonl.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --image_dir data/flux_images/images \
    --output_dir data/flux_images \
    --test_size 128 --seed 42
```

## Reference dataset recipe

- Prompts: 27,815 English video captions from
  [ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data),
  as filtered by [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
  (`assets/consist-id.txt`)
- Images: FLUX.1-dev, 400x640, 30 steps, guidance 3.5, per-index seeds
- Split: seed-42 shuffle -> 27,687 train / 128 test
- Convert: `prepare_data.py --frame_mode first`, then train with
  `rollout.pipeline.task=fl2va` and `frame_indices='[0]'`. The rollout
  pipeline LANCZOS-resizes condition images to the sampling resolution, so
  training at e.g. 288x464 (same ~1:1.61 aspect) works directly.

## FL2VA (image-conditioned) training

This recipe trains the FL2VA checkpoint with online DiffusionNFT. The rollout
uses vLLM-Omni's official first/last-frame contract and the Actor applies the
NFT forward-process objective only to generated video/audio rows.

### Data

Prepare `train.jsonl` and `test.jsonl`. Each row has a prompt and either an
`images` list or explicit first/last names:

```json
{"prompt":"A sunrise becomes a starry night.","images":["images/first.png","images/last.png"]}
```

Convert it with:

```bash
python examples/diffusionnft_trainer/minimax_h3/prepare_data.py \
  --input_dir /path/to/raw \
  --output_dir /path/to/parquet \
  --frame_mode first_last
```

`frame_mode` can be `first`, `last`, or `first_last`. Set the matching launcher
value:

```bash
export FRAME_INDICES='[0,-1]'  # '[0]' or '[-1]' for one-image datasets
```

### Checkpoint

The two runtimes intentionally use different transformer layouts. `MODEL_PATH`
points at the official FL2VA checkpoint used by vLLM-Omni (fused QKV and GEGLU
weights), while `ACTOR_TRANSFORMER_PATH` points at the converted diffusers
`MiniMaxH3Transformer3DModel`. Do not replace the official checkpoint's
`transformer/` directory with the diffusers conversion: that silently breaks the
rollout weight loader.

### Run

```bash
MODEL_PATH=/path/to/MiniMax-H3/FL2VA \
ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer \
DATA_DIR=/path/to/parquet \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

The latest vLLM-Omni contract requires 4–15 seconds at 24 FPS. The launcher's
`NUM_FRAMES=96` is aligned by vLLM-Omni to the next valid `17n+5` boundary.
This checkpoint is not a low-step distilled model, so the launcher defaults to
`INFER_STEPS=50`; 2–4 steps are suitable only for contract smoke tests and
produce noisy video and audio. Do not use the old short 22/29-frame settings.

To verify that the two checkpoint directories represent exactly the same base
policy after fused-QKV and GEGLU conversion, run:

```bash
python tests/special_e2e/minimax_h3_checkpoint_parity.py \
  --vllm-transformer "$MODEL_PATH/transformer" \
  --diffusers-transformer "$ACTOR_TRANSFORMER_PATH"
```

The H3-specific agent loop is required: it tokenizes raw text once and lets
vLLM-Omni prepend the `<Picture N>` vision presentation. Replacing it with the
generic diffusion agent loop changes the prompt contract.

CPU offload is enabled and the default rollout TP is 4: with TP=2, the
colocated dummy-load phase places a full text encoder beside one DiT shard
before offload activates and leaves no room for Actor-to-rollout weight
synchronization on a 96 GB GPU. With the 8-GPU recipe, `ROLLOUT_N=4` also
makes the two-prompt Actor batch divisible by the FSDP data-parallel world
size.

The launcher uses the diffusers `native` Actor attention and vLLM-Omni
`TORCH_SDPA` rollout attention by default. This validated combination does not
need to download the optional `kernels-community/flash-attn3` Hub kernel at
startup. Override `ACTOR_ATTN_BACKEND` and `ROLLOUT_ATTN_BACKEND` together only
after validating the replacement backends.

Rollout quantization is intentionally not enabled. On the pinned vLLM-Omni
commit, a native custom pipeline combined with online FP8 can hit a meta-tensor
placement failure during custom-pipeline initialization; BF16 TP=4 is the
validated path.

## License

- Prompts: CC-BY-4.0 (ConsisID-preview-Data)
- Images: generated with FLUX.1-dev (non-commercial license); datasets built
  with this pipeline inherit the non-commercial restriction

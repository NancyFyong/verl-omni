# MiniMax H3 T2VA FlowGRPO

Last updated: 08/22/2026

This recipe trains a rank-64 MiniMax H3 LoRA with FlowGRPO. The rollout uses
separate video and audio reverse-SDE schedules, records a contiguous training
window, and recomputes the same weighted transition log probabilities on the
Diffusers actor.

## Install

Follow the project [installation guide](../../../docs/start/install.md), then
install the pinned vLLM-Omni and tested Diffusers revisions:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
uv pip install "diffusers @ git+https://github.com/huggingface/diffusers.git@245d78fb48f1c87dfb560a94bea6e191c9f9f1c0"
```

## Prepare the model

Use separate transformer layouts for actor and rollout:

- `MODEL_PATH` points to the official MiniMax-H3 `FL2VA` directory, including
  tokenizer, text encoder, VAE, and the official fused transformer.
- `ACTOR_TRANSFORMER_PATH` points to the converted Diffusers transformer.

Do not replace the official rollout transformer with the Diffusers conversion.
vLLM-Omni expects fused QKV and GEGLU weights; the actor expects the Diffusers
split layout.

## Prepare text prompts

Reuse the DiffusionNFT t2va converter, which reads `train.txt`/`test.txt` or
`train.jsonl`/`test.jsonl` and writes the same prompt-only schema:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2av_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

Do not use the LTX-2 converter here: it emits a `negative_prompt` column, and
`RLHFDataset` keys off that column's presence to tokenize a negative branch. H3
runs without classifier-free guidance (`true_cfg_scale=1.0`), so that branch is
dead weight the pipeline never consumes.

## Prepare the rewards

The recipe scores both generated streams, matching the DiffusionNFT t2va recipe:
CLAP for text-audio alignment and ImageBind for audio-video alignment, combined
as a weighted sum. An image-only scorer such as PickScore cannot consume the
`[1, 3, T, H, W]` video the rollout returns.

CLAP resolves from the Hub. ImageBind ships no Hub repo and is licensed
CC-BY-NC-SA 4.0 (NonCommercial):

```bash
uv pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
uv pip install --no-deps pytorchvideo fvcore iopath portalocker ftfy timm
```

`--no-deps` is required: ImageBind pins `torch==2.0.1`. `pytorchvideo` still
imports the `torchvision.transforms.functional_tensor` module removed in
torchvision 0.17, so add a shim next to the installed package:

```bash
python3 -c 'import pathlib, torchvision; (pathlib.Path(torchvision.__file__).parent / "transforms/functional_tensor.py").write_text("from torchvision.transforms._functional_tensor import *\n")'
```

The reward downloads `imagebind_huge.pth` (4.5 GB) on first use, or point
`IMAGEBIND_CKPT` at an existing copy.

## Launch

```bash
export MODEL_PATH=/path/to/MiniMax-H3/FL2VA
export ACTOR_TRANSFORMER_PATH=/path/to/MiniMax-H3-diffusers/transformer
export DATA_DIR=/path/to/h3_t2va_data
export IMAGEBIND_CKPT=/path/to/imagebind_huge.pth

bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

The recipe explicitly selects:

```bash
actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent
```

The dedicated AgentLoop tokenizes the verbatim prompt with
`add_special_tokens=False`. Its marker lets the rollout reject generic
chat-template IDs rather than decoding and re-tokenizing them.

The default topology uses eight GPUs, rollout TP=4, text-encoder TP=2, four
samples per prompt, and FSDP2 actor training. The actor uses native attention
and rollout uses `TORCH_SDPA`; override both backends together when using a
matched FA3 installation.

MiniMax H3 uses separate video and audio sigma shifts (`12.0` and `3.0`). The
rollout captures two contiguous stochastic transitions by default while still
running the full schedule. The checkpoint is not a 2–4-step distilled model, so
quality runs default to 50 inference steps. Lower values are suitable only for
contract smoke tests. Both streams' per-step log probs are combined with
`av_logprob_video_weight` / `av_logprob_audio_weight`, using the same weights in
the rollout and the actor. Only `diffusion_loss.loss_mode=flow_grpo` is
supported; the other reverse-SDE losses assume a single sigma schedule.

Common overrides include:

```bash
NOISE_LEVEL=0.7 \
SDE_WINDOW_SIZE=2 \
SDE_WINDOW_RANGE='[0,49]' \
INFER_STEPS=50 \
TOTAL_TRAINING_STEPS=100 \
OUTPUT_DIR=/path/to/output \
bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

Additional Hydra overrides may be appended to the command.

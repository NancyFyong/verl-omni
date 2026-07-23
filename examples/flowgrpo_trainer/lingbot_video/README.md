# LingBot Dense T2V FlowGRPO

This example supports only `robbyant/lingbot-video-dense-1.3b` text-to-video
LoRA training.  It intentionally excludes the MoE checkpoint, image input,
the refiner, and SGLang Diffusion.

Install the optional model package in every actor and rollout environment:

```bash
uv pip install -e '.[vllm-omni,lingbot-video,train,dev]'
```

Run the prompt rewriter offline if desired, then write one JSONL object per
caption with at least `{"caption": {...}}`. Convert it to the dataset schema:

```bash
python examples/flowgrpo_trainer/lingbot_video/prepare_structured_captions.py \
  --train-jsonl captions/train.jsonl --val-jsonl captions/val.jsonl \
  --output-dir ~/data/lingbot_video
```

Set a video reward function already available in this repository/environment,
then launch `run_lingbot_dense_t2v_lora.sh`. The recipe uses the official
480×832, 121-frame, 40-step, guidance-3, shift-3 baseline. For inexpensive
training experiments lower `num_inference_steps` explicitly, while retaining
the 40-step validation setting until quality is measured.

## GPU rollout smoke test

The opt-in GPU test loads a real Dense checkpoint and runs a 64×64, five-frame,
two-step rollout through the LingBot transformer, FlowGRPO SDE trajectory and
Wan VAE. It also verifies rollout-side in-memory LoRA activation on an actual
LingBot `nn.Linear` layer. Point it at an existing local checkpoint:

```bash
LINGBOT_VIDEO_MODEL_PATH=/path/to/lingbot-video-dense-1.3b \
  pytest -q -s tests/pipelines/test_lingbot_video_flow_grpo_gpu.py
```

The test skips automatically when CUDA, `lingbot-video`, or the checkpoint is
unavailable; it does not download model weights during normal test runs.

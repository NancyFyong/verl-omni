# Wan 2.1 causal ODE initialization and CausVid

Last updated: 09/07/2026.

This recipe initializes a causal Wan 2.1 T2V student from deterministic teacher
ODE trajectories. It is the initialization stage used before CausVid-style DMD;
it does not run teacher/fake-score distribution matching itself.

## Scope

- Target: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (the dimensions are read from the
  checkpoint; they are not hardcoded into the adapter).
- Training: full-sequence block-causal attention with one timestep per latent
  motion block.
- Inference contract: incremental self-attention K/V caching with atomic block
  commits and temporal RoPE offsets. vLLM-Omni causal serving is intentionally
  deferred.
- Parallelism: FSDP1 or FSDP2, with Ulysses sequence parallelism disabled for
  this initial implementation.

## ODE data

Use `WanODETrajectoryDataset`. Each row contains:

- precomputed `prompt_embeds` for the default launcher, or `prompt` when
  `conditioning_provider=local_frozen_encoder`;
- `ode_latents` with the reference layout `[steps, latent_frames, channels,
  latent_height, latent_width]`;
- `ode_timesteps` with one raw Wan timestep per trajectory state;
- `final_clean_latent`, exactly equal to the final trajectory state;
- `trajectory_manifest`, including teacher/revision, full scheduler config,
  guidance and negative prompt, timestep list, VAE identity, `SFCHW` layout,
  dtype and shape, tokenizer identity, and seed policy.

The dataset computes a canonical SHA-256 over the manifest. Pass that digest as
`TRAJECTORY_MANIFEST_SHA256`; training fails before forward if any row differs.
Generate trajectories with a frozen deterministic teacher and retain the initial
noise state, selected intermediate states, and final clean latent. Synthetic
trajectories from `tests/special_e2e/create_dummy_wan_ode_data.py` are only for
execution tests and are not valid training data.

## Run

```bash
MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
TRAJECTORY_MANIFEST_SHA256=<canonical-sha256> \
NUM_GPUS=8 \
bash examples/distillation_trainer/wan21/run_wan21_ode_lora.sh
```

The default trajectory layout has 21 latent frames grouped into seven
three-frame motion blocks. A sampled trajectory index is shared by all frames in
a block. Positions whose selected timestep is zero are excluded from the ODE MSE.
The causal attention path allows all tokens in the current block to attend to one
another and to every committed prior block, but never to future blocks.

The ODE config retains a raw `[1000,750,500,250]` schedule fixture with shift 8.0;
the user-facing exporter below instead records the actual supplied trajectory
timesteps to avoid silently using that fixture for incompatible training data.
Only `student` or `student_ema` LoRA is exported; the causal attention processor
adds no checkpoint parameters, so standard Diffusers Wan transformer keys load
without conversion. A downstream causal inference runtime must install the same
mask, temporal-offset RoPE, and cache semantics before using the export.

## User-facing video tools

Run from the repository root. `wan21_video.py` implements local causal generation;
it does not route requests through ordinary bidirectional `WanPipeline.__call__`
or claim vLLM-Omni serving support.

```bash
# Trusted FSDP checkpoints only. Never deserialize an untrusted pickle checkpoint.
python examples/distillation_trainer/wan21/wan21_video.py export \
  --checkpoint /path/to/global_step_1000 --role student \
  --manifest /path/to/trajectory_manifest.json \
  --output /path/to/inference/student --trust-checkpoint

python examples/distillation_trainer/wan21/wan21_video.py generate \
  --base-model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --adapter /path/to/inference/student \
  --prompt "A river flowing through a forest" --seed 42 \
  --output /path/to/videos/student.mp4

# Decode a teacher target; this does not show student quality.
python examples/distillation_trainer/wan21/wan21_video.py decode \
  --base-model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --latent /path/to/000000.clean.pt --prompt "A river flowing through a forest" \
  --output /path/to/videos/teacher.mp4

python examples/distillation_trainer/wan21/wan21_video.py gallery \
  --directory /path/to/videos
```

The inference artifact contains one adapter (`student` or `student_ema`), its PEFT
config, and `inference_manifest.json` with a weights hash, base revision, geometry,
block size, **resolved** denoising timesteps and transition semantics. Retain the
full training checkpoint separately for resume. The exporter never overwrites an
existing artifact and selects one role rather than mixing all saved adapters.

For an untrained causal baseline use `generate --manifest <trajectory manifest>`
instead of `--adapter`. `--trajectory <SFCHW file>` reuses its initial noise.
Use the same prompt, initial noise, transition-noise seed, geometry and resolved
schedule for a weight-only before/after comparison. Without training, switching
a bidirectional Wan to causal attention and three-step sampling can produce noise.

Causal sampling predicts x0, adds **fresh noise** at the next sigma, and commits
completed clean blocks at t=0. This is the reference consistency sampler, not Euler
integration. Each video owns fresh self/cross-attention caches. A schedule ending
in zero contains one fewer prediction than stored states; `[1000,757,522,0]` is
three denoising predictions, not four. A final zero-time cache refresh is not a
fourth denoising step.

## Real teacher trajectories

`generate_ode_data.py` runs independent frozen Wan teachers on each rank and saves
resumable, prompt-indexed trajectories plus cached positive/negative embeddings.
It records actual Euler snapshot timesteps; do not label them with approximate
values from another scheduler. The default grid has sigma_min=0, 50 steps and
shift 8; its snapshots are approximately `[1000,756.757,521.739,0]`. These are
the resolved values underlying the released rounded `[1000,757,522,0]` fixture.
Generation uses standard Diffusers CFG (6.0), while CausVid DMD scoring below uses
the reference's legacy CFG. Keep the actual timestep metadata even when using a
different explicit teacher grid; do not relabel existing trajectories. Checkpoint
conversion, precision and scheduler implementation can still differ numerically
from the reference and must not be advertised as bit-identical reproduction.

```bash
torchrun --standalone --nproc_per_node=8 \
  examples/distillation_trainer/wan21/generate_ode_data.py \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --prompts /path/to/filtered_prompts.txt --output-dir /path/to/ode_data \
  --teacher-dtype bfloat16 --validation-size 16
# Publish parquet only after every requested trajectory has completed.
python examples/distillation_trainer/wan21/generate_ode_data.py \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --prompts /path/to/filtered_prompts.txt --output-dir /path/to/ode_data \
  --teacher-dtype bfloat16 --validation-size 16 --finalize
```

The supplied local checkpoint must include Hugging Face revision metadata.
FP32 teacher compute is the default; BF16 is a documented compute/storage tradeoff,
not FP32 parity. Use unique prompts and keep evaluation prompts outside training.
Each rank logs every five denoising steps. Do not use synthetic smoke trajectories
to claim quality or convergence.

## CausVid DMD after ODE initialization

```bash
MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
STUDENT_ADAPTER_PATH=/path/to/inference/ode_student \
TRAIN_FILES=/path/to/ode_data/train.parquet \
VAL_FILES=/path/to/ode_data/test.parquet \
TRAJECTORY_MANIFEST_SHA256=<canonical-sha256> NUM_GPUS=8 \
bash examples/distillation_trainer/wan21/run_wan21_causvid_lora.sh \
  trainer.total_training_steps=1000
```

The cached-data path needs `negative_prompt_embeds` as well as positive embeddings
and final clean latents. The existing `WanODETrajectoryDataset` validates their
trajectory provenance; this implementation consumes the final clean state for
CausVid input construction, not the stored intermediate ODE states. Raw-prompt
runs can use `conditioning_provider=local_frozen_encoder` and an explicit negative
prompt instead.

Two physical groups reuse the existing distillation worker, optimizer, EMA,
accumulation and checkpoint implementation:

- `causal_base`: ODE-initialized student and its EMA.
- `bidirectional_base`: independently optimized fake-score adapter and an
  adapter-disabled frozen teacher. The ODE student adapter is **never** loaded
  into this group.

The released reference path is `backward_simulation=false`: independently re-noise
clean latents at each configured student timestep, select one state per motion
block and predict x0 with a full block-causal forward. Score timesteps are shared
across all frames of each sample; integer sampling precedes shift and [2%,98%]
clamping. Corruption and x0 conversion use the same nearest point on Wan's shifted
1000-step sigma grid, including its nonzero sigma at raw t=0. The teacher uses
`cond + 3.5*(cond-uncond)`, with no norm rescaling; fake scoring is conditional-only.
The normalizer covers every non-batch element, is unmasked, and uses epsilon=0 in
the reference profile. The fake-score target is `noise - generated_x0.detach()`.

One completed cycle is one student update followed by K fake updates (default 5).
The first fake update reuses the student batch; the remaining K-1 consume fresh
batches, matching the reference generator-every-K-critic-iterations ordering.
Role optimizer counters remain independent, and global_step counts completed
student cycles rather than critic iterations. EMA, LoRA, fp32 surrogate reductions
and atomic completed-cycle checkpoint/resume are framework choices, not a claim
of exact reproduction of the full-weight reference experiment.

Reference: [CausVid](https://github.com/tianweiy/CausVid), especially
`causvid/dmd.py`, `causvid/train_distillation.py`, and
`causvid/models/wan/causal_inference.py`. Its CC BY-NC-SA source is used only to
verify behavior; this integration does not import or copy that source.

Execution tests and finite losses do **not** demonstrate a useful model. Evaluate
held-out prompts with matched seeds, report failed or degraded outputs, and retain
teacher, untrained causal, ODE student and CausVid student videos together.


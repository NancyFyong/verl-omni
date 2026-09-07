# Wan 2.1 causal ODE initialization

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

The configured export schedule starts from raw timesteps
`[1000, 750, 500, 250]` and applies the Wan rational shift `8.0` exactly once.
Only `student` or `student_ema` LoRA is exported; the causal attention processor
adds no checkpoint parameters, so standard Diffusers Wan transformer keys load
without conversion. A downstream causal inference runtime must install the same
mask, temporal-offset RoPE, and cache semantics before using the export.

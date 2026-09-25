# Qwen-Image DMD2 distribution-only

Last updated: 09/25/2026.

See the [algorithm and runtime contract](../../../docs/algo/diffusion_distillation.md)
for the objectives, role ownership, configuration and checkpoint semantics.

This MVP trains a conditional-only few-step **Qwen-Image T2I student** from prompts.
It is **DMD2 distribution-only**, not original DMD: there are no paired teacher
trajectories, LPIPS regression, discriminator, GAN, rewards or PPO advantages.
Qwen-Image Edit, causal video and Self-Forcing are not included.

## Train

Use the repository's GPU + training installation and a base `Qwen/Qwen-Image`
checkpoint. Prompt parquet follows the existing `RLHFDataset` schema:

```python
{"prompt": [{"role": "user", "content": "A red apple on a wooden table"}]}
```

The frozen encoder applies the checkpoint's Qwen system template. Do not add a
custom system message: raw inputs accept a string or one text-only user message.
A custom dataset may instead return `prompt_embeds` `[B,L,D]`, optional
`prompt_embeds_mask` `[B,L]`, and matching negative embeddings for teacher scoring.
Pre-tokenized inputs must include attention masks and the Qwen template prefix.

```bash
MODEL_PATH=/path/to/Qwen-Image \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/test.parquet \
OUTPUT_DIR=outputs/qwen_dmd2 \
NUM_GPUS=8 TOTAL_TRAIN_STEPS=1000 \
bash examples/dmd2_trainer/qwen_image/run_qwen_image_dmd2_lora.sh
```

The script selects the new route:

```yaml
algorithm:
  trainer_type: distribution_matching
  sample_source: offline
actor_rollout_ref:
  model:
    algorithm: dmd2
    model_type: diffusion_dmd_model
```

Here `sample_source: offline` means **engine-local sampling**, not offline RL or
training on pre-generated images. The current student generates fresh samples
from prompts and noise during training inside FSDP, retaining the graph needed
for its objective. No independent vLLM rollout server or reward workers are
started. Keep this configuration value `offline`; it does not make the student
samples precomputed.

`dmd` is a separate top-level configuration group. Do **not** enable the existing
OPD `distillation.enabled` or actor `use_distill_loss` flags. Existing
policy-gradient, direct-preference and OPD routing is unchanged.

Defaults: 1024×1024, four student steps, LoRA rank/alpha 32, student LR `1e-4`,
fake-score LR `2e-5`, fake update ratio 2, teacher CFG 4 with `layer_norm`, negative
prompt `" "`, and EMA decay 0.999. Student and fake-score are conditional-only.
Teacher CFG is computed in **packed velocity space**, before conversion to x0.
Sampling uses fixed sigmas `[1, .9, .75, .5, 0]`, not Qwen's native
resolution-dependent shift. The text encoder is frozen; no training VAE is loaded.

## Execution and accounting

`DistributionMatchingRayTrainer` reuses `BaseRayDiffusionTrainer`'s offline
initialization, dataloaders and profiling. `DMDTrainingWorker` subclasses the
existing `TrainingWorker`; `DMDDiffusersFSDPEngine` reuses its model loading, FSDP,
LoRA and checkpoint services. The Qwen adapter remains a stateless registry class.

Each **cycle** fetches one fresh student batch, then K fresh fake-score batches.
Each role call attempts at most one optimizer update. `training/global_step`
counts completed cycles; per-role optimizer counters count successful updates.
All ranks agree on nonfinite skips. A skipped role advances neither its scheduler
nor EMA. Even an all-skipped cycle consumes the finite budget; a run with no
successful updates for either role ends with an error rather than a success claim.
Unexpected failures stop the run: partially applied GPU updates are not rolled
back, and recovery requires the last complete checkpoint in a new trainer.

One frozen base holds independently optimized `default` (student) and
`fake_score` adapters, a non-optimized `student_ema`, and adapter-disabled teacher
scoring. This MVP requires LoRA; independent full modules and external teachers
are deferred, not mathematical requirements of DMD2. FSDP1 requires
`use_orig_params=true`; FSDP2 is the default.

## Batching and diagnostics

`GLOBAL_BATCH_SIZE` defaults to `NUM_GPUS`. Each role independently uses
`STUDENT_MICRO_BATCH_SIZE` / `FAKE_MICRO_BATCH_SIZE` (default 1). Accumulation uses
the existing worker and microbatch splitter. For a non-divisible tail, it uses
native TensorDict splitting, with sample-weighted losses and identical chunk
counts across DP ranks; it does not pad in extra training samples.
Physical batches must have homogeneous image geometry. For example, eight DP
ranks and physical batch 2 use global batch 16. The baseline validation uses SP=1.

The rollout exit is synchronized across FSDP ranks. Physical batching and
accumulation can sample different rollout depths; scalar CFG additionally uses a
batch-wide norm. Neither is a controlled performance comparison without fixing
those inputs.

Metrics are phase-qualified (`student/0/...`, `fake_score/0/...`,
`fake_score/1/...`). Within a role attempt, losses are sample-weighted and host
component durations/counts are summed over microbatches; DP aggregation reports
rank means, not slowest-rank wall time. Cycle/checkpoint timings are fresh each
cycle. Component timings overlap and are not isolated CUDA kernel costs. The
inherited MFU fallback may report zero without diffusion FLOPs metadata. Peak
allocated/reserved memory reports the maximum across ranks, cumulative since
worker initialization (not a separately reset peak for each phase).

Use the existing `global_profiler.steps` and
`actor_rollout_ref.actor.profiler` settings for Torch traces. No new profiler or
inference server is required for training.

## Checkpoint and resume

The only saved training artifact is the complete FSDP checkpoint:

```text
OUTPUT_DIR/
  global_step_N/
    actor/          # shared model, student optimizer, fake optimizer/scheduler,
                    # EMA, per-rank RNG and successful/skipped counters
    data.pt
    trainer.pt      # completed cycle, driver RNG and configuration fingerprint
  latest_checkpointed_iteration.txt
```

The launcher saves every `SAVE_FREQ` cycles (default 100) and after the final
completed cycle. `SAVE_FREQ=-1` disables saving; no separate artifact is then
written at the end. Saving is synchronous and atomic on a shared local filesystem.
All ranks and role clocks are checked before publication and restore. `max_actor_ckpt_to_keep`
controls retention of this run's complete checkpoints. Restore requires model,
optimizer and extra state. Legacy generic-distillation checkpoints are rejected;
there is no implicit format/counter migration. Mathematical/data/optimizer
configuration drift (including a changed configured training horizon) is rejected.
Existing experimental checkpoints are not converted or modified.

The script defaults to `RESUME_MODE=auto`. To replay from a chosen checkpoint,
keep the same data/model/training configuration and use a fresh output directory:

```bash
# Append to the training command above:
# trainer.resume_mode=resume_path \
# trainer.resume_from_path=/path/to/global_step_N
```

Student, fake-score and EMA adapters remain in the resumable model state. No
separate student/EMA LoRA, `inference/` directory or generation tool is produced.
`dmd.export_role` is deprecated and ignored; retain its original value when
resuming an existing checkpoint because it remains part of the configuration
fingerprint. Existing checkpoints and inference files are left untouched.

FSDP checkpoints are not directly loadable with `QwenImagePipeline.from_pretrained`.
Inference conversion, CheckpointEngine validation replicas, vLLM request batching
and new transports are outside this training integration. A completed training
smoke does not establish generation-quality improvement.

## Validation commands

```bash
# CPU suite (includes existing OPD regressions)
python -m pytest -o 'python_files=*_on_cpu.py' --asyncio-mode=auto tests/

# Real two-rank FSDP1/FSDP2 tiny-model updates, EMA and checkpoint resume
QWEN_IMAGE_MODEL_PATH=/path/to/tiny-random/Qwen-Image \
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/workers/test_dmd_fsdp.py

# Full Ray production entrypoint: three student / six fake-score updates
MODEL_PATH=/path/to/tiny-random/Qwen-Image NUM_GPUS=2 \
bash tests/special_e2e/run_dmd2_qwen_image.sh
```

The tiny model is an execution fixture, not a useful generator. Real-model smoke,
controlled prototype comparison and resume results belong in the PR's validation
evidence, with their actual scope and hardware, not inferred from CPU tests.

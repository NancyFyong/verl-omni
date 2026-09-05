# Qwen-Image DMD/DMD2

This example trains a few-step Qwen-Image generator with the distribution-matching runtime introduced by RFC #519.
Sampling is **offline and differentiable inside the FSDP actor**; it does not use vLLM-Omni for the training rollout.

## Install

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,distillation]"
```

The `distillation` extra installs `piq`, which is required only by the paper-oriented original-DMD `decoded_lpips` regression profile. Distribution-only DMD2 does not import it.

## Data

The DMD2 distribution-only recipe accepts the normal prompt parquet contract. Each row needs `prompt=[{"role": "user", "content": "..."}]`. The standard `RLHFDataset` passes this as `raw_prompt`; the Qwen adapter applies the checkpoint pipeline's fixed template before encoding and removing its 34-token prefix. Plain strings are also supported; custom system messages and multi-turn chats require precomputed conditioning.

For PickScore / Pick-a-Pic prompts, use the existing SFW converter:

```bash
python examples/flowgrpo_trainer/data_process/sd3_pickscore_sfw.py \
    --dataset CarperAI/pickapic_v1_no_images_training_sfw \
    --output-dir ~/data/pickscore_sfw/qwen_image
```

To preserve the upstream Flow-GRPO split, download its `dataset/pickscore_sfw/train.txt` and `test.txt`, then pass their directory with `--input-dir`. The converter records source and split metadata. Set `TRAIN_FILES` and `VAL_FILES` to the resulting parquet files. DMD2 uses only these prompts, **not a PickScore reward model**; prompt-only data does not satisfy original DMD's paired-regression requirements.

Precomputed conditioning is also supported. Set:

```bash
distillation.distribution_matching.conditioning_provider=precomputed
```

and provide `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, and `negative_prompt_embeds_mask` tensors.

Original DMD additionally requires paired `reference_noise`, either `teacher_target_latents` or normalized `[0, 1]` `teacher_target_pixels`, and a non-empty `teacher_sampling_manifest`. Use `data.custom_cls.path=pkg://verl_omni.utils.dataset.qwen_image_distillation_dataset` and `data.custom_cls.name=QwenImageDMDPairDataset` to convert inline arrays, serialized tensor bytes, or absolute `.pt` paths into fp32 tensors. Its default `decoded_lpips` regression is paper-oriented; `latent_mse` is available only as a non-paper diagnostic.

## Run DMD2

```bash
MODEL_PATH=/path/to/Qwen-Image \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
NUM_GPUS=8 \
bash examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh
```

The reference-aligned defaults are four student denoising steps, rollout and score-noise time shifts of `3.0`, score sigma range `[0.02, 0.98]`, teacher CFG `4.0` with per-token norm preservation, student LR `1e-4`, fake-score LR `2e-5`, and two fake-score updates after each student update.

Physical micro-batches support multiple samples at the same resolution. The existing worker splits each rank-local batch, sample-weights gradients (including a smaller final micro-batch), and steps each role optimizer once. Student and fake-score micro-batch sizes are independent. Original DMD additionally requires one provenance manifest per sample. Mixed-resolution micro-batches fail closed.

For eight data-parallel ranks, compare these configurations at **the same effective batch of 16**:

```bash
# Accumulation: two physical micro-batches of one per rank.
bash examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
  data.train_batch_size=16 \
  distillation.distribution_matching.student_micro_batch_size_per_gpu=1 \
  distillation.distribution_matching.fake_score_micro_batch_size_per_gpu=1

# Physical batching: one micro-batch of two per rank.
bash examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
  data.train_batch_size=16 \
  distillation.distribution_matching.student_micro_batch_size_per_gpu=2 \
  distillation.distribution_matching.fake_score_micro_batch_size_per_gpu=2
```

Pass one set of overrides to the launch script above. Rollout exit decisions are broadcast across the training group: FSDP shards must execute identical forward counts and gradient exits, even though their prompts and sample noise differ. A physical batch shares one exit; accumulation samples an exit for each micro-batch. Consequently, compare forward counts as well as wall time rather than attributing random exit-depth differences to batching. The default `layer_norm` CFG is sample-separable; optional `scalar` CFG uses a batch-wide norm, so changing physical batch size also changes that normalization. Shared-base FSDP1 additionally requires `use_orig_params=true`; the script uses FSDP2.

Only `student` or `student_ema` is exportable. Teacher and fake-score parameters remain training-only state. The registered vLLM-Omni `dmd`/`dmd2` rollout adapter uses the same fixed-shift sigma schedule, fp32 initial noise, deterministic sampling (`noise_level=0`) and no inference CFG by default. It accepts `rollout_timestep_shift` through request `extra_args` when a non-default training shift is used.

## Request batching

The adapter explicitly advertises `supports_request_batch=True` and uses the existing vLLM-Omni request scheduler, request-local generators, prompt collation, denoising loop, and output splitting. Requests must share compatible sampling parameters and the DMD shift. Seeds and prompt lengths may differ; multiple images per request are supported. Use `step_execution=false`: stepwise continuous batching has not been validated for this DMD adapter and is rejected rather than silently using FlowGRPO defaults.

Start inference with a small `actor_rollout_ref.rollout.max_num_seqs` (for example, `2`) and measure memory before increasing it; multiple images per request further increase the effective tensor batch.

Request batching concerns non-autograd inference, **not the offline FSDP training rollout**. Export APIs do not by themselves wire validation replicas into the distillation trainer. The tiny-GPU test covers native request scheduling, variable prompt lengths, seeds, multiple images, and serial-versus-packed trajectories. It does not establish real-model inference throughput or numerical parity between the training and inference backends.

## Profiling and metric semantics

Reuse the existing `DistProfiler` and `Tracking` backends. Run a short job in a separate output directory after other training finishes; keep the same checkpoint, effective batch, schedule, seeds, and hardware for each comparison. Example profiling overrides to the launch script:

```bash
bash examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
  trainer.total_training_steps=6 \
  trainer.resume_mode=disable trainer.save_freq=-1 \
  trainer.logger='[console,tensorboard]' \
  global_profiler.tool=torch 'global_profiler.steps=[3,4]' \
  global_profiler.save_path=/path/to/profile/traces \
  actor_rollout_ref.actor.profiler.enable=true \
  'actor_rollout_ref.actor.profiler.ranks=[0,1]' \
  'actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda]'
```

Both the global step selection and worker profiler enable/rank selection are required. The Torch trace includes `distillation/condition_encode`, `student_rollout`, role forwards, `backward`, role optimizers, `ema`, and original-DMD `regression` ranges. Profiled runs measure attribution, not clean throughput: compare unprofiled warm runs for speed, and report other GPU workloads.

- `perf/cycle_s`: driver wall time for the entire completed cycle, including data fetch, all phase RPCs and any checkpoint; excludes trace export and logging.
- `perf/*_s`: component **host** durations summed across micro-batches and repeated phases, averaged across DP ranks. These ranges overlap; do not sum them to derive cycle time or GPU kernel time.
- `perf_max_rank/*_s`: sums of the corresponding per-phase slowest-rank durations, not an independently measured cycle critical path.
- `phase/<kind>[/<repeat>]/...`: individual phase metrics, so no fake-score update is overwritten. Unprefixed losses are means across phase updates; element/nonfinite counts are summed across micro-batches and phases, then DP-averaged.
- `memory/max_*`: maximum across data-parallel replicas and phases, rather than average memory usage.
- `training/<role>_samples`: actual global samples processed this cycle (reused samples count as processing again). `perf/<role>_samples_per_s` divides that count by cycle time. Fake updates do not inflate student throughput.
- `batch/<role>_micro_batches`: total micro-batches per rank this cycle; `training/<role>_optimizer_steps`: cumulative successful updates.
- Checkpoint timing is emitted only on checkpoint cycles. Tracking uses `completed_cycles` as its monotonic logging step, including warmup; `training/global_step` remains the completed student-update counter.

Choose tuning targets from the trace. Compare batching at fixed effective batch, then independently test the existing `actor_rollout_ref.model.enable_gradient_checkpointing` and `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` settings, monitoring memory and repeating correctness tests.

On supported Hopper GPUs, the existing Diffusers/Kernels FA3 Hub backend is another option; it does not require implementing a new attention kernel in this repository. Configure both sides to satisfy the existing attention-consistency validator, even when the current run only samples offline:

```bash
bash examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh \
  actor_rollout_ref.model.attn_backend=_flash_3_varlen_hub \
  actor_rollout_ref.rollout.rollout_attn_backend=FLASH_ATTN_3_HUB
```

The Hub kernel must be available locally or downloadable. `tests/pipelines/test_qwen_image_dmd_request_batch.py` checks masked/unmasked FA3 forward/backward against native attention as well as the real inference request path. `tests/workers/test_distillation_fsdp_roles.py` covers multi-rank phase execution for DMD/DMD2 at physical batch sizes one and two. Keep backend/precision tolerance checks separate from throughput benchmarks.

Do not change the fake-update ratio or denoising schedule and label it an implementation speedup. These optimizations are opt-in; the example does not automatically disable checkpointing or resharding.

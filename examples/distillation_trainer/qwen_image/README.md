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

The DMD2 distribution-only recipe accepts the normal prompt parquet contract. Each row needs a `prompt` chat-message list. The standard `RLHFDataset` passes the messages as `raw_prompt`; the Qwen adapter renders and encodes them once per micro-batch with the frozen checkpoint text encoder.

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

The initial implementation requires physical micro-batch size one per data-parallel rank. Gradient accumulation still provides a larger global batch. Shared-base FSDP1 additionally requires `use_orig_params=true`; the script uses FSDP2.

Only `student` or `student_ema` is exportable. Teacher and fake-score parameters remain training-only state. The registered vLLM-Omni `dmd`/`dmd2` rollout adapter uses the same fixed-shift sigma schedule, defaults to deterministic sampling (`noise_level=0`) and no inference CFG, and accepts `rollout_timestep_shift` through request `extra_args` when a non-default training shift is used.

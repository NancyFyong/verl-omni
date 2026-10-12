# MiniCPM-o 4.5 simplex thinker OPD

Last updated: 10/07/2026.

This recipe implements the first, thinker-stage part of [RFC #565](https://github.com/verl-project/verl-omni/issues/565), under [#345](https://github.com/verl-project/verl-omni/issues/345). The student generates a complete text response; a frozen teacher scores the same tokens and verl's on-policy distillation loss updates the student.

It is the [MiniCPM-o GSPO recipe](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/README.md#training-with-minicpm-o-45-avqa) plus a `distillation.*` group, the same composition as the [Qwen3-Omni OPD recipe](../../../docs/algo/omni_opd.md). Data loading, prompt parity, rollout, and the actor path are unchanged from GSPO; only the teacher is added.

## Supported scope

- MiniCPM-o **4.5**, text output from the thinker stage.
- Text, images, and at most one audio clip per prompt.
- A separate frozen `vllm_omni` teacher pool from the same model family; teacher and student must share the tokenizer and special-token IDs, which is checked before worker allocation.

Video, multiple audio clips per prompt, talker/codec-policy training, and streaming duplex are not supported. Speech-policy OPD is a separate RFC milestone; streaming duplex is tracked by [#566](https://github.com/verl-project/verl-omni/issues/566).

## Weights and dependencies

Use the repository's [installation instructions](../../../docs/start/install.md) and pinned vLLM-Omni revision, and review the checkpoint's remote code before enabling `trust_remote_code`. Use Python 3.12: FlashInfer's communication module fails during worker initialization on Python 3.11.

```bash
hf download openbmb/MiniCPM-o-4_5 \
  --revision 503e754207c94da6bb26850b4469f367c9ea3582 \
  --local-dir "$HOME/models/MiniCPM-o-4_5"
```

Use the original checkpoint as the frozen teacher. To reproduce the noise-perturbed-student setup from [Qwen3-Omni OPD PR #375](https://github.com/verl-project/verl-omni/pull/375) with 20% noise, create a separate student checkpoint:

```bash
python examples/opd_trainer/minicpm_o/prepare_noised_student.py \
  --source-model "$HOME/models/MiniCPM-o-4_5" \
  --output-model "$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
  --noise-ratio 0.20 \
  --seed 42
```

The script copies the checkpoint and perturbs only nonzero floating-point `llm.*` tensors, each by a Gaussian direction rescaled to `||ΔW||₂ / ||W||₂ = 0.20` within the checkpoint dtype's rounding. Encoders, processor, and tokenizer stay identical to the teacher, and `noise_manifest.json` records the scope, ratio, and seed. It is diagnostic tooling, not a validated recovery recipe: it perturbs embeddings, MLPs, and norms, while the recipe trains attention LoRA only.

## Data

AVQA (image + audio → text) uses the GSPO recipe's data:

```bash
python examples/gspo_trainer/data_process/avqa.py \
  --input_dir <path_to_raw_AVQA_R1> \
  --output_dir ~/data/avqa_r1_6k
```

MMK12 (image + text → text) uses the canonical converter shared with Qwen3-Omni; the scorer needs `math-verify`:

```bash
pip install math-verify
python examples/gspo_trainer/data_process/mmk12.py \
  --local_dataset_path <path_to_raw_mmk12> \
  --local_save_dir ~/data/mmk12
```

## Training

Run from the repository root; the base GSPO recipe resolves its reward path relative to it.

```bash
STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
TEACHER_MODEL="$HOME/models/MiniCPM-o-4_5" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora.sh

STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5-Thinker-Noise20" \
TEACHER_MODEL="$HOME/models/MiniCPM-o-4_5" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora_mmk12.sh
```

The student uses the GSPO recipe's four GPUs and settings; the teacher adds `TEACHER_GPUS` (default 4) GPUs with `TEACHER_TP` (default 2) per replica on the same node. `TRAIN_FILE` / `VAL_FILE` select other data, and extra CLI arguments override the defaults.

The default objective is task reward plus OPD, with `loss_mode=kl` and `use_policy_gradient=true`. For teacher-only OPD, add `distillation.distillation_loss.use_task_rewards=false`. Rewards decode through the GSPO recipe's `minicpm_naive` manager, which keeps MiniCPM's `<answer>` tags.

| Setting | Value |
| --- | --- |
| `distillation.teacher_models.teacher_model.inference.name` | `vllm_omni` |
| teacher `engine_kwargs.vllm_omni` | `output_mode=ar`, `pipeline_name=minicpmo_4_5`, `trust_remote_code=true` |
| teacher `prompt_length` / `response_length` | the student's, 4160 / 12288; `max_model_len` 16640 |
| `distillation.distillation_loss` | `loss_mode=kl`, `use_policy_gradient=true` |

Using an identical unperturbed checkpoint on both sides is a consistency smoke test, **not evidence of useful distillation**.

## Teacher scoring and verification

The teacher scores the student's prompt and response in one forward pass. The rollout adapter collapses only as many expanded media spans as the request carries media, so the response passes through verbatim even when the student samples a media marker. The AR strategy rejects a teacher result whose scored ids differ from the request or whose log probabilities are not finite.

The thinker adapter rebuilds the resampler's non-persistent positional cache after loading and keeps Qwen3 RoPE frequencies in FP32 through whole-model casts. Floating weight synchronization follows `actor.fsdp_config.mixed_precision.param_dtype` (BF16 by default). None of this establishes BF16 actor/rollout parity; validate the chosen precision with an identical-student/teacher control before interpreting a learning curve.

CPU contracts:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q --asyncio-mode=auto \
  tests/pipelines/test_minicpm_prompt_parity_on_cpu.py \
  tests/pipelines/test_minicpm_adapter_on_cpu.py \
  tests/trainer/omni/test_omni_distillation_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_omni_teacher_on_cpu.py \
  tests/utils/test_minicpm_rl_dataset_on_cpu.py
```

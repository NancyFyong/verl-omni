---
name: review-train-rollout-consistency
description: "Review checks for a verl-omni PR that can change how the actor replays a rollout step: schedules and step counts, sampling parameters, conditioning inputs and masks, CFG and optimizer numerics, and the ratio and log-prob evidence. Use from pr-review or self-review."
---

# Review: train/rollout consistency

The actor recomputes the log-prob of every stored transition through the training
adapter's `forward_and_sample_previous_step` (see `verl_omni/pipelines/model_base.py`),
which returns `(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)`. Before the first
update, actor and rollout must agree: after any difference in schedule,
conditioning, CFG, masks or precision, the loss trains a policy other than the one
that sampled. Nothing raises. Precision root causes live in the
[common pitfalls](../../../docs/contributing/common_pitfalls.md); dump collection
lives in [train-infer-consistency](../train-infer-consistency/SKILL.md).

## When it applies

```bash
git diff <base>...<head> | grep -nE 'sigmas|timesteps|linspace|time_shift|num_inference_steps|noise_level|sde_|guidance|true_cfg|negative_prompt|prompt_embeds|attention_mask|max_sequence_length|chat_template|torch_dtype|param_dtype'
```

Any new training or rollout adapter, and any change to a scheduler, text encoder,
processor or sampling default on either side.

## Step math

- **Build the schedule the same way on both sides.** FLUX (#532) builds the base
  schedule with NumPy `linspace` on both rollout and actor, then applies the same
  float32 shift, so exact timestep lookup holds at non-dyadic step counts such as
  20. A `torch` versus NumPy schedule differs by an ULP and misses the lookup. Ask
  for a CPU test that asserts exact equality at a non-power-of-two step count.
- **Step-count mapping follows the pinned engine.** BAGEL mapped
  `num_inference_steps` n → n−1 to match vllm-omni 0.22; after the engine switched
  to the official semantics, the engine sampled a 14-point schedule while both
  scheduler tables held 15 points (#279). Recheck every mapping on a pin bump
  ([review-dependencies](../review-dependencies/SKILL.md)).
- **SDE window and noise seeding.** The window is chosen per GPU, not per request
  ([pitfall](../../../docs/contributing/common_pitfalls.md#sde-window-per-request-vs-per-gpu-seeding)).
- **Stored latents and stepwise scheduler math stay float32** (the two float32
  precision-loss pitfalls in the same guide).

## Conditioning inputs

- **Every sampling parameter reaches the rollout server.** #25 found that
  `guidance_scale` was never passed from config, so rollout always ran with
  `None`. Trace each new config field to the engine request.
- **A deliberate actor/rollout difference is stated and tested.** FLUX runs rollout
  guidance 3.5 and actor guidance 1.0 by design (#532), with a CPU test for the
  actor default.
- **Broadcast order at `num_images_per_prompt > 1`.** #25 fixed a
  `prompt_embeds_mask` expanded as ABAB instead of AABB; batch 1 hid it.
- **No truncation on one side.** SD3.5 actor replay cropped a 333-token embedding
  to 256 (#357). Compare the maximum sequence length and chat template on both
  sides; the SD3.5 v1 scripts lacked the template (#384).
- **Masks match the pinned engine.** #585 review found a replay path that dropped
  the text attention masks; the reviewer asked to keep them or cite the engine's
  behavior with parity evidence. RoPE lengths have their own
  [pitfall](../../../docs/contributing/common_pitfalls.md#rope-sequence-length-mismatch).

## Numerics

- **CFG matches the reference implementation.** A guard epsilon that the official
  pipeline lacks changes the output; #260 review removed one by citing the
  diffusers source.
- **Updates must be representable.** #428 trained bf16 parameters with plain AdamW
  at lr 2e-7 and no fp32 master copy; the step is below half a bf16 ULP, so over
  99% of parameters rounded back unchanged. Check parameter dtype against the
  recipe's learning rate.
- **Identical objects do not prove identical log-probs.** Registering the Qwen3-Omni
  processor through the registry hook dropped the rollout/actor log-prob
  correlation from about 0.99 to 0.13 with byte-identical processor objects
  (#113). Judge by the metric.

## Evidence

- Parity on the submitted recipe with real weights, read from the right metric. By
  default (`rollout_correction.bypass_mode=false`) the actor recomputes
  `old_log_probs`, so step-1 `actor/ratio_mean` ≈ 1.0 only shows the actor agrees
  with itself (the precision pitfalls). Actor/rollout parity is then
  `rollout_corr/*` (for example `rollout_corr/logprob_abs_diff_mean`), computed
  only with `calculate_log_probs=true`. With `bypass_mode=True`, `old_log_probs`
  are the rollout's, and step-1 `ratio_mean` ≈ 1.0 and `ppo_kl` ≈ 0 are the parity
  check. AR paths report upstream verl's `training/rollout_actor_probs_pearson_corr`.
- A CPU replay test with a mask-sensitive mock and exact `ratio_mean` 1.0 is a
  boundary test, not real-weight parity; #585 said so in its body.
- A mismatch goes to [train-infer-consistency](../train-infer-consistency/SKILL.md)
  for paired dumps, not to a tolerance bump.
- Parity does not show learning; convergence is a separate rung in
  [review-tests-ci](../review-tests-ci/SKILL.md).

<!--
MAINTAINER GUIDE — The return tuple and metric names are read from model_base.py,
diffusion_algos.py and rollout_correction.py; recheck them when those change. Put
new precision root causes in common_pitfalls.md and link them here.
-->

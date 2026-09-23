---
name: review-weight-sync
description: "Review checks for a verl-omni PR that changes actor-to-rollout weight sync: full-weight export, name mapping, fused-layer loading, LoRA collection and engine binding, and the evidence that the rollout actually runs the trained weights. Use from pr-review or self-review."
---

# Review: actor → rollout weight sync

Weight sync failures are almost never loud. A tensor that loads under the wrong name,
a LoRA that binds to zero modules or a buffer that is not exported leaves the
rollout running base (or subtly wrong) weights while training looks healthy. The
export side lives in `verl_omni/workers/engine/`, LoRA collection in
`verl_omni/utils/fsdp_utils.py`, engine-side adapter loading in
`verl_omni/utils/vllm_omni/`.

## When it applies

```bash
git diff <base>...<head> | grep -nE 'load_weights|get_per_tensor_param|state_dict\(|collect_lora_params|layer_prefixes|map_lora_update_to_engine|target_modules|merged_lora'
```

A missing mapper adds no diff line, so this grep cannot find it. Also apply the skill
to every new pipeline package whose recipe or smoke enables LoRA.

## Full-weight export

- **Non-persistent buffers are not exported.** The engine sends `state_dict()`,
  which omits buffers registered with `persistent=False`. MiniMax H3's
  `rope.inv_freq` stayed uninitialized on the rollout side and produced patch-grid
  videos with healthy statistics. The rollout module must compute such buffers
  itself; check the buffer list of any new model.
- **fp32 islands are cast on the wire.** Training keeps `_keep_in_fp32_modules` in
  fp32 ([common pitfalls](../../../docs/contributing/common_pitfalls.md#float32-islands-flattened-by-a-blanket-bf16-cast)),
  but the export casts sharded tensors to bf16 (see the TODO in the export path).
  A model whose rollout depends on fp32 gates or norms needs an explicit exception.
- **Names get a `transformer.` prefix.** Compare the exported key set with the
  rollout module's `load_weights` mapping; renamed modules, fused QKV/GEGLU weights
  and per-head interleaved layouts need an explicit loader. #477 fixed an H3 loader
  that reported source names instead of the canonical fused names.
- **Merged LoRA export is a generator over live storage.** Tensors must be
  materialized inside `merged_lora_context`; after it exits, the base weights are
  sent without error.
- A full-weight path added to a pipeline worker must forward to the engine's real
  `load_weights`, not a local no-op.

## LoRA collection and binding

- **Collection walks only `<prefix><i>` blocks.** LoRA on top-level modules such as
  `proj_out` or `norm_out` is not collected, and `collect_lora_params` raises only
  when the total is zero. A wrong `fsdp_layer_prefixes` that still matches one
  block family passes silently.
- **Engine names differ from diffusers names.** Without a correct
  `map_lora_update_to_engine` on the rollout pipeline, the adapter binds to zero
  modules and rollout stays base-identical. #332 shipped Boogu and Qwen-Image
  LoRA whose `to_out.0` and Boogu `.processor.` targets never bound (fixed in
  #661); BAGEL LoRA silently trained nothing after a vllm-omni bump (#553). Take
  each recipe `target_modules` pattern, apply the mapper, and find the matching
  engine module; a module the engine renames in `load_weights` (Boogu drops
  `.processor.`, `to_out.0` becomes `to_out`) binds only if the mapper renames it too.
- **A mapper on a shared pipeline affects every model and engine using it.**
  #624 review found a mapper rewriting all Qwen-Image FlowGRPO LoRA, including the
  FSDP path; #661 then limited the change to renaming. Check each subclass.
- **A validator must run.** A bind-count or name validator that no runtime path
  calls is dead code (#624). Trace its caller.
- A compatibility loader for a pinned vllm-omni bug (#368) is a temporary patch
  under [review-dependencies](../review-dependencies/SKILL.md#patching-upstream).

## Evidence

- A CPU test that pushes the adapter through the real engine-side LoRA manager on a
  tiny native model and asserts the bound module count and exact per-module A/B
  values, as #661 did (the partial mapper bound 36/44; the full one 44/44).
- Adapter IDs, "loaded" log lines or tensor counts do not show that values bound.
- Step-1 parity cannot show LoRA binding: with the default `gaussian`
  `lora_init_weights`, LoRA B starts at zero, so a rollout on base weights matches
  the actor until the first update. The bind-count test is the evidence;
  later-step `rollout_corr/*` drift is the symptom.
- For full weights: after one sync, the same prompt and seed on the actor and the
  rollout give matching outputs or log-probs
  ([review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md)),
  and a generated sample is actually viewed.

<!--
MAINTAINER GUIDE — The bf16 cast and the block-prefix walk are current engine
behavior with TODOs; update the first two bullets of each list when they change.
-->

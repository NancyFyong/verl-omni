---
name: review-new-architecture
description: "Review checks for a verl-omni PR that adds a model architecture (new pipeline package or registry pair): registration, silent rollout-registry misses, integration-checklist gaps, shared-path leakage and past misses. Use from pr-review or self-review."
---

# Review: new architecture

`docs/contributing/` owns the integration steps and [add-pipeline](../add-pipeline/SKILL.md)
picks the guide. Work that guide's final checklist against
`git diff --stat <base>...<head>` first; the checks below cover what the checklists
miss and what past architecture PRs got wrong. A new package almost always also
needs [review-weight-sync](../review-weight-sync/SKILL.md),
[review-rollout-contract](../review-rollout-contract/SKILL.md),
[review-media-contract](../review-media-contract/SKILL.md) and
[review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md).

## Both registries must resolve

Registration is an import side effect ([pipelines rule](../../rules/pipelines.md)).
The two sides fail differently: a training-registry miss raises, but a rollout miss
returns `None` and the rollout silently runs the stock vllm-omni pipeline with no
log-probs adapter, so the failure surfaces later and far from its cause. Both keys
are `(architecture, algorithm)` since #67. Check with the checkpoint's own key:

```bash
jq -r ._class_name <ckpt>/model_index.json
python - <<'EOF'
import verl_omni.pipelines  # noqa: F401 -- registers every adapter
from verl_omni.pipelines.model_base import DiffusionModelBase, VllmOmniPipelineBase
arch, algo = "<ClassName>", "<algorithm>"
print(DiffusionModelBase.get_class_by_name(arch, algo))
assert VllmOmniPipelineBase.get_class(arch, algo) is not None
EOF
```

- Request batching is opt-in. Without `supports_request_batch = True`, the strategy
  clamps `max_num_seqs` to 1 — correct but slow (the Boogu recipe notes about 4x).
  Ask whether that is deliberate; enabling it needs the batch > 1 checks in
  review-rollout-contract.
- A custom agent loop registers the same way, via its `@register(...)` on import.

## Gaps in the integration checklists

None of the guide checklists mention these:

- A row in `_PRIMARY_MODALITY` (search `tests/pipelines/` for it). The io-spec test
  covers only listed adapters, and its `importorskip` turns an import error into a
  skip.
- `fsdp_layer_prefixes` in the recipe when the blocks are not `transformer_blocks.`;
  it also decides which LoRA tensors sync (review-weight-sync).
- LoRA support implies `map_lora_update_to_engine` plus a bind-count test
  (review-weight-sync).
- The doc surfaces: an entry in `docs/start/models.md`, a row in the README model
  table, and an example page in the `docs/index.md` toctree, each doc with its
  `Last updated:` line.
- A non-standard checkpoint layout (review-config-recipe) and GPU smoke selection
  ([review-tests-ci](../review-tests-ci/SKILL.md)).

## Keep shared code generic

Maintainers push back on per-model copies in shared layers:

- Reuse an agent loop across a model's algorithms and put shared prompt helpers
  in `verl_omni/agent_loop/utils.py` (#368 extracted `messages_to_text`; #383
  reused the H3 raw-text loop). Check RFC #463 before adding another raw-text path.
- Keep new base-class hooks minimal; a maintainer asked #428 to cut its hooks
  down.
- A rollout adapter must not decode token IDs the agent loop already encoded
  back to text (#178, #383).
- A local non-diffusers model copy is a maintenance cost, accepted only with a
  TODO to drop it once a first-class engine supports the model (#66).
- Subclass or import from the closest sibling before copying
  ([code-style](../../rules/code-style.md#reuse-over-duplication)).

## Existing pipelines must not regress

New architectures often edit a shared helper, agent loop or strategy. List every
other adapter that calls the changed function and ask for evidence each still
works:

```bash
git diff --stat <base>...<head> -- verl_omni | grep -v "pipelines/<new_package>/"
```

#585 added TI2VA on the path shared with T2AV and had to add regression tests for
T2AV condition replay and attention masks; #527 had to show that new
`OmniRolloutPipelineBase` functions leave existing rollout intact.

## What past architecture PRs missed

| Original | Fixed in | Missed |
| --- | --- | --- |
| #332 Boogu, Qwen-Image LoRA | #661 | LoRA `to_out.0` and Boogu `.processor.` targets never bound on rollout |
| #260 Qwen-Image-Edit | #583, #488, #584 | image tokens vs visual features; `img_shapes` lost in the V1 TransferQueue path; no e2e CI |
| #178 SD3.5 | #357, #384 | 333-token embedding cropped to 256; chat template missing from v1 scripts |
| BAGEL | #279, #553 | stale step-count mapping after a vllm-omni bump; LoRA silently trained nothing on vllm-omni ≥ 0.24 |
| #368 MiniMax H3 | #477 | fused QKV/GEGLU loader reported source names, so the strict post-load check rejected them |
| #485 H3 Ref2VA | #534 | nested condition tensors crashed before the first training step |

Nearly all are silent train/rollout divergences that mock-only unit tests passed.
The evidence a new model must show is rung 4 of the
[evidence ladder](../review-tests-ci/SKILL.md#evidence-ladder).

<!--
MAINTAINER GUIDE — Recheck the registry behavior against model_base.py and the
diffusion rollout strategy when dispatch changes. Move a gap out of this list once
the integration guide's checklist covers it.
-->

---
name: review-rollout-contract
description: "Review checks for a verl-omni PR that changes what crosses the rollout boundary: request fields, multi_modal_data, extra_fields, rollout_output keys, log-prob gating and batch > 1 collation. Use from pr-review or self-review."
---

# Review: rollout request and output contract

The agent loop, the vLLM-Omni strategy and the rollout adapter exchange plain dicts.
A field that is misspelled, dropped or shaped differently on one side is not a type
error; it is a silent change in what the actor replays. The request contract lives
in `verl_omni/pipelines/rollout_request.py`, the output envelope in
`verl_omni/pipelines/diffusion_rollout_output.py`.

## When it applies

```bash
git diff --name-only <base>...<head> | grep -E 'agent_loop/|workers/rollout/|rollout_(request|output)|vllm_omni_rollout_adapter'
git diff <base>...<head> | grep -nE 'extra_fields|multi_modal_data|mm_processor_kwargs|rollout_output\(|with_rollout_data\('
```

## Request side

- The diffusion strategy sends `prompt_token_ids`, not text. An adapter that needs
  text must decode it itself (BAGEL does) or its agent loop must pass the raw prompt
  (see [review-new-architecture](../review-new-architecture/SKILL.md) on raw-text
  loops).
- `multi_modal_data` is written twice, top-level and under
  `extra_args.multi_modal_data`, and `mm_processor_kwargs` carries reference fps and
  sample rate. A change must keep both copies, and the payload for an already-valid
  request must stay byte-identical (#480 made that a stated invariant).
- Conflicting inputs fail closed, but only real aliases conflict. Condition images
  resolve through `condition_images_from_payload`; vllm-omni also stores a derived,
  resized copy, and treating it as an alias broke Qwen-Image-Edit (#583).
- A duplicate modality in one request raises; do not replace that with last-wins.

## Output side

- Only the `trajectory_*` arguments and the `prompt_embeddings` and `rl` groups of
  `rollout_output()` reach training. `metadata=` stays in the envelope and never
  becomes an `extra_fields` key; a replay input put there is lost.
- `custom_output` was removed with the vllm-omni 0.27 pin (#363);
  `git grep -n custom_output` on the head must be empty.
- A key that appears in two groups raises `Duplicate rollout metadata field`.
  Rename it; do not merge the groups.
- Rollout log-probs are returned only when the request sets `logprobs`. Check that
  the agent loop still requests them for every algorithm that reads
  `rollout_log_probs`.
- Every new replay field needs a matching consumer in the training adapter and, for
  the V1 TransferQueue trainer, in the written field set
  ([review-async-state](../review-async-state/SKILL.md)); #488 lost `img_shapes`
  there.

## Batch > 1 collation

The agent loop concatenates each tensor `extra_fields` entry on dim 0, taking the
key list from the first sample. Only prompt embeddings, their masks, token tags and
`*_rows` reference inputs are padded first. So:

- A variable-length tensor field works at batch 1 and crashes, or silently
  misaligns, at batch > 1. Ask for padding plus a mask, or a `*_rows` field.
- A key missing from the first sample is dropped for the whole batch; a key
  missing from a later sample raises. Mixed presence needs an explicit check.
- Nested or jagged tensors do not survive dim-0 slicing in actor replay (#534).
- Opting into `supports_request_batch` without a batch ≥ 2 test ships this bug.

## Evidence

A CPU test that builds a batch of at least two samples with different prompt or
condition lengths, runs the real collation and asserts the replayed fields. A test
at batch 1, or one that mocks the agent loop's concatenation, does not cover this
boundary. For request changes, a before/after comparison of the engine payload:
feed the same inputs to the strategy's `preprocess_input` at base and head, as
`test_vllm_omni_strategy_on_cpu.py` does, and compare the prompts it returns. For
a derived field, build the input from what the pinned engine's pre-process writes.

<!--
MAINTAINER GUIDE — Recheck the padded-key list in diffusion_agent_loop.py and the
group whitelist in the diffusion strategy when either changes; both are the facts
this skill rests on.
-->

---
name: review-omni-ar
description: "Review checks for a verl-omni PR that adds or changes an omni or autoregressive model: OmniModelBase and OmniRolloutPipelineBase adapters, multi-stage pipeline_mode topology, processor loading on Ray workers, shared stage hooks and omni CI. Use from pr-review or self-review."
---

# Review: omni and autoregressive models

The [omni integration guide](../../../docs/contributing/integrating_an_omni_model.md)
owns the adapter hooks, their contracts and registration, but has no final
checklist. Work its [common pitfalls](../../../docs/contributing/integrating_an_omni_model.md#6-common-pitfalls)
against the diff, check each overridden rollout hook against its contract in
[§3](../../../docs/contributing/integrating_an_omni_model.md#3-create-the-rollout-adapter), then work the checks below.

## When it applies

```bash
git grep -l -E 'OmniModelBase|OmniRolloutPipelineBase' <head> -- verl_omni/pipelines  # omni packages
git diff --name-only <base>...<head> | grep -E 'trainer/(config/)?omni/|vllm_omni_ar_strategy|vllm_omni_async_server|engine/fsdp/omni_impl'
git diff <base>...<head> | grep -nE 'OmniModelBase|OmniRolloutPipelineBase|pipeline_mode|combine_engine_outputs|prepare_engine_prompt|policy_stage_id|weight_sync_stage_ids|postprocess_agent_loop_output'
```

## One server, two rollout configs

- **Shared code reads only what both configs have.** `vLLMOmniHttpServer` serves
  diffusion and AR rollouts. It read the diffusion-only `rollout_attn_backend`
  unconditionally, so every AR rollout crashed at server launch (#211). For a field
  added to one rollout config, find each reader and confirm the other path never
  reaches it; launch both kinds of rollout.
- **Deviations from the verl parent are tracked.** `vllm_omni_async_server.py` is
  being realigned with upstream verl under RFC #377. #527's `collective_rpc`
  override returned the result where the parent discards it; the review asked for
  a row in #377. Any new override or return-contract change there gets one.

## Processor and tokenizer on every process

- **The processor must load on Ray workers, not only in the driver.** #211: a patch
  of `verl.utils.tokenizer.hf_processor` missed the snapshot re-exported through
  `from verl.utils import hf_processor`. #224: on `AgentLoopWorker`, `model.py`
  bound the unpatched function before the patch ran, so the processor was `None`,
  `SingleTurnAgentLoop` fell back to a bare tokenizer and failed with "chat_template
  is not set". Patch placement is covered in
  [review-dependencies](../review-dependencies/SKILL.md#patching-upstream); here,
  ask for a run that reaches `AgentLoopWorker.generate_sequences`, since a
  driver-side unit test cannot see a worker's import order. A processor that loads
  but shifts log-probs is a [consistency](../review-train-rollout-consistency/SKILL.md)
  finding (#113).

## Stage topology

- **Every existing `pipeline_mode` still deploys.** #527 first made Qwen3-Omni
  `pipeline_mode=full` hard-fail on "multiple final pipeline outputs": that
  topology has two `final_output=True` stages and the adapter had no combiner. The
  review offered three ways out: state that the mode was already unusable, mark the
  PR `[BREAKING]`, or add a combiner. Retaining several outputs became opt-in through
  `combine_engine_outputs`. For each adapter touched, list the modes
  `build_stage_configs` accepts and check each one.
- **The policy stage is the trained and synced stage.** `policy_stage_id` selects
  the sampling parameters and log-probs; `weight_sync_stage_ids` selects which
  stages receive actor weights (default: all). Check both against the stage the
  training adapter registers. Qwen3-TTS syncs only its trainable Talker stage
  (#428).

## Hooks on the shared base classes

- **Generalize, interface first.** #428 review asked that `talker_forward.py` be
  made general because the Qwen3-Omni Talker and later models need it too, with the
  interface landing first. It became the shared Talker hooks (#504) and the `[1/N]`
  interface PR #527 with RFC #538. Stage logic another omni model would need goes
  into an `OmniModelBase` or `OmniRolloutPipelineBase` hook, not a model package.
- **Defaults keep existing adapters unchanged, with a test.** #527 review: "We might
  need some test to cover newly function added to `OmniRolloutPipelineBase` make
  sure it doesnt break the existing rollout that already adapt". The answer was
  `test_optional_rollout_hooks_preserve_existing_ar_defaults`, run on the base class
  and on `Qwen3OmniRolloutAdapter`. Each new hook's default returns the pre-PR
  behavior, and the test gains a line for it. Any other behavior a hook adds is
  named in the body and tested; #527's per-stage capacity handling was neither.
- **Validate only real mistakes.** #527 dropped about 40 lines checking the return
  types of in-repo adapter classmethods and kept the unknown-stage check
  ([code-style](../../rules/code-style.md#runtime-boundaries-and-state)).
- **Keep per-request work cheap and owned.** #527 had stashed fields on engine
  output objects (replaced by a dict keyed by request ID) and deep-copied the
  default sampling parameters on every request (now copied once, with only the
  policy stage copied per request).
- **Talker outputs stay aligned.** `postprocess_agent_loop_output` aligns
  `response_ids`, `response_mask` and `response_logprobs` one-to-one and keeps
  model data under one namespaced `extra_fields` key; `prepare_model_inputs` raises
  when that key or a field in it is missing, as the Qwen3-TTS Talker adapter does
  (replay rules: [review-rollout-contract](../review-rollout-contract/SKILL.md)).

## Dependencies and CI

- **A model's upstream package is pinned and installed in CPU CI.** #428: the
  released `qwen-tts` used the Transformers 4 form of `@check_model_inputs()`, so
  the PR pinned a source revision (`.github/qwen_tts_pin.txt`) and made CPU CI
  install it, which also stopped its contract tests from skipping. A local
  auto-class registration carries a TODO on the upstream PR that removes it (#428
  tracks huggingface/transformers#44517; rules in [review-dependencies](../review-dependencies/SKILL.md)).
- **Request the omni group.** Of the source paths, only the Qwen3-TTS package,
  `trainer/omni/` and the omni config select `ci-e2e-omni`; a change under
  `pipelines/qwen3_omni/` or the shared AR strategy does not
  ([selection](../review-tests-ci/SKILL.md#the-test-sits-where-ci-runs-it)). Ask for
  the `ci-e2e-omni` [label](../../../docs/contributing/gpu_smoke_tests.md#pr-labels-for-gpu-smoke).

## Evidence

- CPU tests that call each new hook through every existing omni adapter and assert
  the default behavior, plus the hook's edge cases (unknown stage, abort output).
- `ci-e2e-omni` green on the head SHA, and the config generator run with verl
  installed, which otherwise skips the omni YAMLs ([config rule](../../rules/config.md#generated-yaml)).
- A real run that reaches `AgentLoopWorker.generate_sequences` and trains, with
  step-1 parity from review-train-rollout-consistency; a new model also owes rung 4
  of the [evidence ladder](../review-tests-ci/SKILL.md#evidence-ladder).

<!--
MAINTAINER GUIDE — Hook names and defaults are read from model_base.py and
vllm_omni_ar_strategy.py; group paths from select_gpu_smoke_groups.py. When the
omni guide gains a final checklist, work it first and drop what it covers here.
-->

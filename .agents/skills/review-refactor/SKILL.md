---
name: review-refactor
description: "Review checks for a verl-omni refactor or stacked [N/N] slice: behavior preservation, deleted tests, fail-fast semantics, every consumer of a shared helper, leftover references, breaking changes and slice scope. Use from pr-review or self-review."
---

# Review: refactor

A refactor's claim is "nothing observable changed except what the body lists". The
review job is to falsify that claim. Past refactors broke in the places below, and
most were caught only because a reviewer searched outside the diff.

## Behavior preservation

- The first slice of a stack should be behavior-preserving extraction; #444 (first
  slice of RFC #403) states "public RPC, outputs and lifecycle unchanged".
- Every intentional behavior change must be listed. #478 removed a MiniMax-only
  32 kHz default from the shared strategy, which changed other audio models to
  their declared rates; that belongs in the body, not only the diff.
- Fail-fast must stay fail-fast. Flag a synchronous `raise` that became a
  background-thread warning or a per-sample fail-open: #481 moved the uint8 check
  into the export thread, so a dtype bug silently cost every dump. "Silent
  fallback → silent bug" (#96; tracked in RFC #388).
- Compare wire payloads for valid requests byte for byte; a request-contract
  refactor must not change what the engine receives
  ([review-rollout-contract](../review-rollout-contract/SKILL.md)).

## Tests that disappeared

Deleted or rewritten tests are the most common refactor blocker. List them and find
where each assertion went:

```bash
git diff --diff-filter=D --name-only <base>...<head> -- tests
git diff <base>...<head> -- tests | grep -E '^-\s*(def test_|class Test)'
```

For each removed name, `grep -rn` the head tree. #480's first head deleted
`TestProcessorPreparationHook` with an alias-test file, dropping the only coverage
of the `external_lib` override and processor loading; #481 deleted the pre-submit
dtype test. Review caught both and they were restored before merge.

## Every consumer of a changed helper

A shared parser, strategy or base-class method has callers the diff does not show:

```bash
git grep -n "<changed_function>" <head> -- verl_omni tests examples
```

Each consumer needs a test on the head, or an explicit risk note. #480 unified
condition-image aliases; Qwen-Image-Edit had no e2e test, and #583 fixed it the next
day because the raw and derived (resized) images tripped the new conflict check.
A green smoke group does not show the consumer ran
([review-tests-ci](../review-tests-ci/SKILL.md#the-test-sits-where-ci-runs-it)).

## Leftover references

Grep the whole tree, not only code, for every removed or renamed symbol, file,
config key and CLI flag:

```bash
git grep -n "<old_name>" <head> -- docs .agents examples tests verl_omni
```

#480's first head left the I2I integration guide calling a deleted class; #373 had to update
`docs/contributing` and the reward rule; `docs/contributing/gpu_smoke_tests.md`
still routes `qwen3_omni_thinker.py`, which #522 removed.

## Breaking changes and in-flight work

- A renamed or removed public API, config key, import path or CLI argument needs
  `[BREAKING]` and the deprecate-then-remove sequence in the
  [config rule](../../rules/config.md#backward-compatibility) (#522 removed a path
  deprecated in #312 and #359). Do not rename files without a reason (#96).
- Registry key or adapter signature changes break open PRs built on the old
  shape: #67 re-keyed both registries by `(architecture, algorithm)` while
  MixGRPO (#58) was in review. Search open PRs for the old symbol.
- Concurrent merges invalidate evidence: #593 merged the latest `main` and reran
  the suite before CI. Ask for results on a head that contains current `main`.
- A change that depends on an upstream verl or vllm-omni PR waits for it and for
  the install docs (#56). Regenerated `_generated_*.yaml` is a golden file:
  review its diff line by line (#350 fixed the omni YAML).

## Slice scope

- An `[N/N]` PR states its stack parent and review range in the body (#557:
  "PR6 review range `6eb8db3..05998d0`"). Review only that range; lower-slice
  findings go to the lower PR.
- Revert what the slice does not need: an unrelated smoke script change (#559),
  a doc rename (#96), extra test plumbing (#8).
- Code copied from vllm-omni stays side-by-side comparable with its source; do not
  restyle it (#8).
- New persisted state or a changed checkpoint layout goes to
  [review-async-state](../review-async-state/SKILL.md).

## Evidence

A refactor needs the full CPU suite on the head plus the smoke group of every
touched consumer ([review-tests-ci](../review-tests-ci/SKILL.md)). "Tests pass" is
not enough when tests were deleted in the same PR; state where each moved.

<!--
MAINTAINER GUIDE — Each check cites the refactor that broke it. Add a check only
with a PR where it was missed; drop one when a CI gate enforces it.
-->

---
name: review-tests-ci
description: "Review checks, loaded for every verl-omni PR, on whether the tests exercise the change, sit at a layer CI actually runs, and whether the claimed evidence (CPU, smoke, real-weight run, curve, profile) supports the claim. Use from pr-review or self-review."
---

# Review: tests, CI and evidence

The [testing guide](../../../docs/contributing/testing_guide.md) owns the layers and
naming, the [testing rule](../../rules/testing.md) the placement gate,
[GPU smoke tests](../../../docs/contributing/gpu_smoke_tests.md) the groups and
labels, and [run-cpu-tests](../run-cpu-tests/SKILL.md) the commands. This skill asks
whether the tests and results in a PR show what the PR claims.

## When it applies

Every PR. Start from what the body claims, then find the test or run behind each claim.

```bash
git diff --name-only <base>...<head> -- tests/ examples/
git diff --name-only --diff-filter=D <base>...<head> -- tests/
```

## A test has to test the change

- **The assertion can fail.** #428 review called its GPU smoke a vacuous gate: the
  success check `grep -q "training/global_step.*2"` also matched the step-1 log
  line. Revert the fix locally, or break the input, and confirm the test fails.
- **It calls the code it protects.** In the same review, a claimed regression test
  never called the config writer it was meant to guard, and most new tests asserted
  their own mocks. A mock-only test cannot see a train/rollout divergence
  ([review-new-architecture](../review-new-architecture/SKILL.md#what-past-architecture-prs-missed)).
- **It runs at all.** The `_on_cpu.py` suffix selects CPU tests but is not gated, so
  a misnamed file commits and never runs ([testing rule](../../rules/testing.md)).
  `pytest.importorskip` turns a missing dependency into a skip; read the `-rs` skip
  summary, not only the pass count.
- **Deleted or moved tests** are traced one by one
  ([review-refactor](../review-refactor/SKILL.md)).
- A bug fix carries a test that fails at the base and passes at the head.

## The test sits where CI runs it

- **Lowest layer that can catch it, in an environment that has what it needs.**
  #559 put a GPU FA3 test in nightly; review moved it out of nightly because the
  L2 environment cannot reliably download FA3 kernels, and the main guard became a
  CPU test.
- **Smoke tests stay self-contained.** Build a tiny-random checkpoint in the
  script (see the `build_*_tiny_random.py` helpers under `tests/special_e2e/`);
  #428 review flagged a smoke that downloaded the real 0.6B checkpoint and ran
  `uv pip install` inside the shared CI job.
- **Each changed consumer is reached by a registered run.** An e2e script needs a
  `run_test` line in its group script, or no CI run ever executes it. The selector
  below lists groups, not tests, so a green group can skip the consumer: at #480,
  Qwen-Image-Edit and Boogu had e2e scripts that no group ran. Check with
  `git grep -n 'run_test' <head> -- tests/gpu_smoke`.
- **`ready-for-ci` selects the group the change needs.** The selector maps paths to
  groups; every package under `verl_omni/pipelines/` except `qwen3_tts` selects
  only `ci-e2e-diffusion`, so an omni or AR adapter change does not run the omni
  e2e group unless its paths are added. Check the plan:

  ```bash
  python tests/gpu_smoke/select_gpu_smoke_groups.py $(git diff --name-only <base>...<head>)
  ```

- **Device counts agree.** #233 review found the omni group forcing `NUM_GPUS=2`
  while exporting the whole `CUDA_DEVICE_LIST`, which breaks once more devices are
  exposed.

## CI results

- **On the head that will merge.** #481 review: "the Sept 7 GPU smoke predates the
  rebase. Pls rerun the MiniMax H3 V1 GPU smoke on the rebased head before merge."
  Check the SHA of each run (the pr-review hub's first step).
- **A failure blamed on `main` is reproduced on `main`.** #593 got that answer with
  evidence: the same LoRA sync failure, `collect_lora_params` returning none,
  reproduced on `main`. Then judge whether the PR worsens it.
- **A timeout is not a pass.** #589 counted a CPU suite that timed out after 600
  seconds as not passing.
- `ci` labels are removed on every push, so CI may not have run since the last push
  ([label auto-removal](../../../docs/contributing/gpu_smoke_tests.md#label-auto-removal)).

## Evidence ladder

Each rung supports only the claims at its level:

1. **CPU contract tests** — config, registry, shapes, mappings, error paths.
2. **GPU smoke** — tiny-random, one or two steps: the loop runs and the flags wire up.
3. **Real-weight end-to-end** — actor/rollout log-prob parity on the right metric
   ([review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md)),
   and generated samples actually viewed.
4. **Convergence curve** — the committed recipe, pretrained weights, about 100 steps,
   train and validation reward rising. #332 review: "do you have a val result curve,
   it is necessary for the pr." #568 review asked for "a clear reward_curve.png";
   its 100-step tiny-random run stayed flat by construction, which shows the loop
   runs, not that the model learns.
5. **Performance** — the same workload before and after ([profile](../profile/SKILL.md)).

What each kind of PR owes:

| PR | Minimum rung |
| --- | --- |
| new model or algorithm | 4, from the committed recipe ([review-config-recipe](../review-config-recipe/SKILL.md)) |
| refactor | 1 and 2 for every consumer it touches |
| bug fix | 1 (or 2) failing before the fix |
| speed or memory claim | 5 |

"Engine initialized", healthy pixel statistics and finite losses are not rungs.
The body lists the rungs that ran, with head SHA and device, and names the ones
that did not.

<!--
MAINTAINER GUIDE — Group mappings are read from select_gpu_smoke_groups.py; update
the selection bullet when its patterns change. Keep layer definitions in
testing_guide.md and label mechanics in gpu_smoke_tests.md.
-->

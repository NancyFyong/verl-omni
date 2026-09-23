---
name: review-dependencies
description: "Review checks for a verl-omni PR that bumps a pin or version bound, patches or hijacks upstream code, adds an import or version guard, or relies on an unmerged upstream change. Use from pr-review or self-review."
---

# Review: upstream pins and patches

verl-omni runs on verl, vllm-omni (with vLLM or vLLM-Ascend), diffusers,
transformers and peft. Exact revisions live in `.github/*_pin.txt` and
`pyproject.toml`; CI installs verl and vllm-omni from the pin files (see
`.github/workflows/cpu_unit_tests.yml`). RFC #445 is the audited ledger of
temporary patches, and RFC #388 is the fail-fast policy.

## When it applies

```bash
git diff --name-only <base>...<head> | grep -E 'pyproject\.toml|\.github/.*_pin\.txt|docker/|docs/start/install'
git diff <base>...<head> | grep -nE 'Hijack|_patches|setattr\(|__version__|version\.parse|ImportError|importlib|hasattr\(|getattr\([^)]*None\)'
```

## Reproduce at the pins

- **Match the pins before attributing a failure.** Install the way CI does. The
  `vllm-omni` extra in `pyproject.toml` names a PyPI release, not the git pin
  (search `vllm-omni==`); a failure on another revision is not the PR's. A #428
  review checked the installed versions against each pin before judging.
- **One revision, every reader.** The verl revision sits in both
  `.github/verl_pin.txt` and the `verl @ git+…` line in `pyproject.toml`;
  Dockerfiles, workflows and install docs read the pin files. After a bump,
  `git grep -n <old-sha-prefix> <head>` returns nothing.
- **An upstream API must exist at the pin.** #113 review found `register_rollout_adapter`
  called from verl at a pin that did not have it. For each new upstream symbol,
  check it at the pinned revision, not upstream `main`.

## A pin bump is a behavior change

Upstream moves break adapters without raising. On a bump, recheck:

- **Removed or renamed contracts.** #363 moved to vllm-omni 0.27, which removed
  `custom_output` and `LTX23Pipeline` (the port moved to `LTX2Pipeline`). Wan2.2
  then broke on `DiffusionRequestBatch`, request fields, sampling parameters and
  the NumPy output contract (#435). Grep the new pin for every upstream symbol the
  adapters import, subclass or override.
- **New upstream calls into code an adapter bypassed.** The verl re-pin in #408
  invoked a continuous-token entry point that MiniMax H3's bypassed base init
  lacked, so rollout failed before generation (#454). An adapter that skips
  `super().__init__` is retested on every bump.
- **Guards that go no-op.** RFC #445 P26: `install_h3_lora_layout` installs its
  mapping only when the transformer has no `stacked_params_mapping`; the new pin
  added a QKV-only native one, so the guard skipped and `fc1` LoRA never bound. Look
  for `if not getattr(<upstream object>, …)` guards.
- **Semantics under an unchanged name.** Step-count mapping
  ([review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md), #279)
  and LoRA binding ([review-weight-sync](../review-weight-sync/SKILL.md), #553).
- **The Python floor.** A pinned vllm-omni used `enum.StrEnum`, so the 3.10 CPU job
  failed before starting (#341); #497 raised `requires-python` to 3.11. A bump says
  what floor it needs.
- **Bump first, integrate second.** #332: Boogu needed vllm-omni#4995, so the
  proposal was a standalone bump PR with the integration rebased on top. A model
  PR that also bumps a pin mixes two reviews.
- The body names the #445 items the bump lands or reopens (the #450 and #497
  re-pins landed P1–P8).

## Patching upstream

- **Prefer a public hook, added upstream if needed.** #113 review: "sorry, we
  cannot accept a PR with huge patches". The author listed every registry hook and
  every monkey-patch with its reason, then replaced the `ray.init` patch with a
  verl plugin entry point. Ask for that inventory whenever a PR adds patches.
- **A remaining patch is temporary and tracked:** an upstream issue or PR, a
  `# TODO` naming it and its removal condition
  ([comment rules](../../rules/code-style.md#comments)), and an RFC #445 entry with
  its fate (drop after bump, drop now, blocked on upstream). Code that exists
  because upstream does not cover diffusion or omni training is an intended
  adaptation, not debt.
- **Wrap, do not replace.** RFC #445 P10: `VLLMOmniHijack` replaces `_load_adapter`
  wholesale and so disables vllm-omni#6476's loader hook for path-based LoRA. Call
  the original, and subclass the base hijack (#113 review asked `VLLMOmniHijack` to
  subclass `VLLMHijack`).
- **Patches reach every process.** Ray workers import on their own; a patch applied
  only in the driver is absent in workers.

## No compatibility fallback

- **Unsupported versions get no code.** #428 review rejected a Transformers 4.x
  compatibility module because the project does not support 4.x. Floors and caps
  live in `pyproject.toml`; a version branch or shim below them is a finding.
- **An import guard reports the real error.** RFC #388 C8: the Wan2.2 package stub
  turned any `ImportError`, `RuntimeError` or `AttributeError`, version drift
  included, into "requires GPU". Catch the narrow error and chain the cause; the
  general rule is in [code-style](../../rules/code-style.md#runtime-boundaries-and-state).
- **Cross-repo order is explicit.** #56 dropped reward-loop patches for upstream
  verl's `assemble_rm_scores`; the reviewer held it until the verl PR merged and the
  install doc was updated. The body names the upstream PR and the pin that contains it.

## Evidence

- `pip list` (or `uv pip list`) for verl, vllm-omni, vllm, transformers and
  diffusers, next to the pin files.
- For a bump: the CPU suite, the GPU smoke groups for every model family whose
  adapters touch a changed upstream symbol, and each #445 item checked
  ([review-tests-ci](../review-tests-ci/SKILL.md)).
- For a patch: a test that fails without it at the current pin, so it is deleted
  when upstream fixes the cause.

<!--
MAINTAINER GUIDE — Pin locations and version floors are read from .github/*_pin.txt,
pyproject.toml and cpu_unit_tests.yml. Patch fates belong in RFC #445, not here;
cite a #445 item only as an example of a failure class.
-->

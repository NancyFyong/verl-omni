# `.agents/` — Agent rules & skills for verl-omni

Repo-local guidance for AI-assisted contributions, complementing the mandatory
contribution policy in [`AGENTS.md`](../AGENTS.md) / [`CLAUDE.md`](../CLAUDE.md).

Check enforcement claims against the current hook scripts, config and call sites.
Distinguish automated gates from review conventions; a passing hook covers only
its configured scope. External templates are references, not additional policies.

## Division of labour

- **[`docs/contributing/`](../docs/contributing/) is authoritative for procedures.**
  Integration guides, testing procedures and the pitfalls reference live there.
  Link to them instead of maintaining another copy here.
  When you add a new guide there, also list it in
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- **`skills/`** are **routers and deltas**: they classify the task, name the guide
  that owns it, and add only what the guide does not cover. Invoked as `/<skill>` or
  auto-loaded by description.
- **`rules/`** carry **contracts and invariants**: the signature a manager calls,
  what a pre-commit hook actually greps for, which filename suffix CI selects. They
  load automatically for files matching their `paths:` frontmatter.

A fact belongs in exactly one place. When a guide and a file here would say the same
thing, the guide wins and the file links to it.

## `rules/`

| Rule                              | Applies to                                                    | Key point |
| --------------------------------- | ------------------------------------------------------------- | --------- |
| [code-style](rules/code-style.md) | everywhere                                                    | automated scope versus conventions; runtime boundaries, shell recipes, contextual reuse and performance |
| [pipelines](rules/pipelines.md)   | `verl_omni/pipelines/**`                                      | dispatch is by registry key and registration is an import side effect; training adapters are never instantiated |
| [reward](rules/reward.md)         | `verl_omni/utils/reward_score/**`, `verl_omni/reward_loop/**` | scorers are selected by config, not by the `data_source` dispatcher; managers call by keyword |
| [config](rules/config.md)         | `verl_omni/trainer/config/**`, `verl_omni/workers/config/**`  | inherit verl's `BaseConfig`, declare `_mutable_fields`, regenerate the YAMLs |
| [testing](rules/testing.md)       | `tests/**`, `test_*.py`                                       | placement is the only commit gate — the `_on_cpu` suffix that decides what CI runs is not enforced |

## `skills/`

| Skill                                                | Use for                                                        |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| [commit-and-pr](skills/commit-and-pr/SKILL.md)       | commit trailers, `[{modules}] {type}:` titles, duplicate-work checks, AI-assistance disclosure |
| [add-pipeline](skills/add-pipeline/SKILL.md)         | routing a model / algorithm integration to the right guide under `docs/contributing/` |
| [add-reward-score](skills/add-reward-score/SKILL.md) | a new reward scorer plus the config overrides that select it    |
| [run-cpu-tests](skills/run-cpu-tests/SKILL.md)       | what the CPU job does that `testing_guide.md`'s local commands don't |
| [self-review](skills/code-review/SKILL.md)           | report-only review: purpose, code quality, goal completeness, and validation/accountability; severity separate from category |
| [pr-review](skills/pr-review/SKILL.md)               | review someone's PR, or address review on your own: exact head state, all feedback threads, routing to the area skills below |
| [profile](skills/profile/SKILL.md)                  | select a profiler and capture the relevant processes/workload |
| [train-infer-consistency](skills/train-infer-consistency/SKILL.md) | rollout / actor consistency collection and analysis using MindStudio skills |

`commit-and-pr` holds the authoritative module list; other files link to it rather
than duplicating it.

### Review area skills

Loaded from the `pr-review` routing table or from `self-review`. Each covers one
system boundary with the past PRs that broke it and the evidence that settles it.

| Skill | Boundary |
| --- | --- |
| [review-new-architecture](skills/review-new-architecture/SKILL.md) | both registries, checklist gaps, shared-path leakage, past misses |
| [review-refactor](skills/review-refactor/SKILL.md) | behavior preservation, deleted tests, every consumer, leftovers, slicing |
| [review-rollout-contract](skills/review-rollout-contract/SKILL.md) | request payload and `rollout_output()` fields reaching training |
| [review-media-contract](skills/review-media-contract/SKILL.md) | declared media kind, layouts, sample rates, reward inputs |
| [review-weight-sync](skills/review-weight-sync/SKILL.md) | full-weight export, name mapping, LoRA binding |
| [review-train-rollout-consistency](skills/review-train-rollout-consistency/SKILL.md) | step math, conditioning and numerics that must match on both sides |
| [review-distributed-memory](skills/review-distributed-memory/SKILL.md) | parallel sizes, FSDP paths, gradient sync, colocated memory, NPU |
| [review-async-state](skills/review-async-state/SKILL.md) | TransferQueue, background threads, sleep/wake, checkpoint resume |
| [review-config-recipe](skills/review-config-recipe/SKILL.md) | config surfaces, checkpoint layout, the recipe behind the evidence |
| [review-dependencies](skills/review-dependencies/SKILL.md) | pins, pin bumps, upstream patches, compatibility code |
| [review-omni-ar](skills/review-omni-ar/SKILL.md) | omni and AR adapters, stage topology, processors on workers |
| [review-tests-ci](skills/review-tests-ci/SKILL.md) | every PR: tests that can fail, CI selection, the evidence ladder |

## Other agent tools

The files live here. `.claude/` and `.codex/` are symlinks to the `skills/` and
`rules/` directories in this one, so each tool finds the same content at the path it
looks for:

```
.claude/skills -> ../.agents/skills      .codex/skills -> ../.agents/skills
.claude/rules  -> ../.agents/rules       .codex/rules  -> ../.agents/rules
```

Whole directories, not per-file links — a skill added under `.agents/` shows up in
both without anyone remembering to link it. Add content only here; `CLAUDE.md` →
`AGENTS.md` is the same arrangement one level up.

## Maintaining these files

- Verify before you edit. Read the enforcement script, count the occurrences, read
  the source — a convention with zero occurrences in the tree is not a convention.
- Prefer a command over a snapshotted number, so the file cannot silently go stale.
- Each skill ends with a `MAINTAINER GUIDE` comment naming what invalidates it.
- Editing agent instructions is itself governed by
  [`docs/contributing/editing-agent-instructions.md`](../docs/contributing/editing-agent-instructions.md).

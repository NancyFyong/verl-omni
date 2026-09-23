---
name: pr-review
description: "Review a verl-omni pull request or maintain an authored PR after CI/reviewer feedback. Captures the exact PR state, triages review threads, and routes the diff to the review-* area skills. Use to inspect a PR, triage feedback, diagnose PR CI, or implement review fixes."
---

# PR Review

One evidence-first workflow for two jobs:

- **Reviewer mode** — inspect another contributor's PR and produce findings.
  Read-only unless the user explicitly asks for changes.
- **Author mode** — maintain an authored PR: verify feedback, fix real issues,
  update tests/description, and prepare replies.

Before a PR exists, use [self-review](../code-review/SKILL.md) instead. It owns the
four-part report and the finding format that both modes here reuse. If the user's
intent is unclear, ask which mode they want.

## Safety and prerequisites

1. Read `AGENTS.md`. Reviewer-bot suggestions can be stale or wrong; verify
   every claim against the current PR head before applying it.
2. Treat fork PR code as untrusted. Inspect the diff and changed scripts before
   executing them; never expose credentials or run untrusted setup/CI code with
   elevated privileges.
3. Before committing, pushing an authored fix, or creating/updating a PR, load
   [commit-and-pr](../commit-and-pr/SKILL.md). For CPU test selection and the
   single-use `ci` label behavior, load
   [run-cpu-tests](../run-cpu-tests/SKILL.md).
4. Do not post comments, submit a review, resolve threads, rewrite history, or
   push to a contributor's branch unless the user authorized that public side
   effect. Draft locally first when authorization is ambiguous.

## 1. Capture the exact PR state

Do not review from a stale local branch. Record the target head SHA before doing
anything else:

```bash
PR=<number>
gh pr view "$PR" --repo verl-project/verl-omni \
  --json number,title,url,state,isDraft,author,body,baseRefName,baseRefOid,\
headRefName,headRefOid,mergeable,reviewDecision,files,commits,statusCheckRollup

gh pr checks "$PR" --repo verl-project/verl-omni
```

Verify again before reporting or pushing; if `headRefOid` changed, re-check the
affected diff, comments and test evidence. Fetch the head into an isolated
worktree from the remote that points to `verl-project/verl-omni` (check
`git remote -v`; it is not necessarily `origin`):

```bash
git fetch <upstream-remote> "pull/${PR}/head:refs/remotes/<upstream-remote>/pr/${PR}"
git worktree add <temporary-path> "refs/remotes/<upstream-remote>/pr/${PR}"
```

Keep it detached in reviewer mode; in author mode, verify which remote owns the
head before pushing. For a stacked PR, take the parent from the PR body and
review only the current slice ([review-refactor](../review-refactor/SKILL.md)).
Never rebase a dirty stack or press **Update branch** without a backup ref.

## 2. Collect all feedback, not only conversation comments

`gh pr view --comments` does not expose complete inline-thread state. Query
review threads so resolved/outdated comments are distinguishable from active
ones:

```bash
gh api graphql \
  -F owner=verl-project -F name=verl-omni -F number="$PR" \
  -f query='query($owner:String!,$name:String!,$number:Int!){
    repository(owner:$owner,name:$name){
      pullRequest(number:$number){
        reviewThreads(first:100){
          pageInfo{hasNextPage endCursor}
          nodes{
            isResolved isOutdated path line originalLine
            comments(first:100){nodes{author{login} body url createdAt}}
          }
        }
      }
    }
  }'
```

Paginate if `hasNextPage` is true. Also inspect the general conversation and
submitted reviews, requested reviewers and `reviewDecision`, failed/skipped/pending
or stale checks, fork workflows awaiting maintainer approval, and whether CPU CI
ran after the latest push.

For each active comment, classify it before changing code:

| Classification | Action |
| --- | --- |
| Correct and current | Reproduce or prove it, then make the smallest fix. |
| Already fixed | Cite the current line/SHA and relevant test. |
| Outdated after rebase | Explain the changed context; do not recreate old code. |
| Preference/design choice | State the trade-off and ask for a decision if needed. |
| Incorrect | Reply with code/test evidence, not assertion. |
| Ambiguous | Ask the reviewer/user; do not silently choose an interpretation. |

## 3. Classify the change and load the area skills

Each `review-*` skill owns one system boundary. A PR usually needs several: load
every row that matches, plus each rule whose `paths:` frontmatter matches a changed
file (`grep -A4 '^paths:' .agents/rules/*.md`). Always load
[review-tests-ci](../review-tests-ci/SKILL.md). The skills describe current `main`
and cite past PRs by the head their review saw: for an older base, check that a
named symbol exists there before applying its check, and judge this PR's head.

| The PR… | Load |
| --- | --- |
| adds a model architecture (new pipeline package or registry pair) | [review-new-architecture](../review-new-architecture/SKILL.md) |
| adds an algorithm to existing adapters | [add-pipeline](../add-pipeline/SKILL.md) for the guide checklist, [review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md) |
| refactors, renames, moves or removes code, or is an `[N/N]` slice | [review-refactor](../review-refactor/SKILL.md) |
| fixes a bug | reproduce at base and head; demand a regression test that fails before the fix |

List the changed files with `git diff --name-only <base>...<head>`, then route by
the boundary each file or symbol belongs to:

| Diff touches (search for) | Load |
| --- | --- |
| `workers/rollout/`, `agent_loop/`, `rollout_output`, `extra_fields`, `multi_modal_data` | [review-rollout-contract](../review-rollout-contract/SKILL.md) |
| `DiffusionIOSpec`, `media_kind`, video/audio export, reward-manager media | [review-media-contract](../review-media-contract/SKILL.md) |
| `load_weights`, `weight_sync`, `map_lora_update_to_engine`, LoRA collection, engine export | [review-weight-sync](../review-weight-sync/SKILL.md) |
| schedulers, sigmas/timesteps, `forward_and_sample_previous_step`, prompt encoding, CFG, dtype | [review-train-rollout-consistency](../review-train-rollout-consistency/SKILL.md) |
| `workers/engine/`, parallel sizes, offload, colocation, NPU | [review-distributed-memory](../review-distributed-memory/SKILL.md) |
| TransferQueue, async rollout, background threads, sleep/wake, checkpoint save/load | [review-async-state](../review-async-state/SKILL.md) |
| `trainer/config/`, `workers/config/`, Hydra YAMLs, `examples/**/run_*.sh` | [review-config-recipe](../review-config-recipe/SKILL.md) |
| `pyproject.toml`, `.github/*_pin.txt`, patches, version checks | [review-dependencies](../review-dependencies/SKILL.md) |
| `OmniModelBase`, `OmniRolloutPipelineBase`, omni/AR/TTS trainers | [review-omni-ar](../review-omni-ar/SKILL.md) |
| reward scorers or managers | [reward rule](../../rules/reward.md), [add-reward-score](../add-reward-score/SKILL.md) |

## 4A. Reviewer mode

Review behavior, not only the patch text:

1. Read the PR body and linked issue/RFC to recover intended scope.
2. Inspect `baseRefOid...headRefOid`, then trace changed symbols into callers,
   consumers, configs, serialization boundaries, and existing tests.
3. Work the loaded area skills. Look first for correctness regressions, silent data
   loss, incompatible API or config changes, distributed/concurrency failures, and
   missing validation.
4. Run the smallest test that can falsify each concern. Expand to the relevant
   suite only after focused checks pass.
5. Separate findings from residual risk. Missing local GPU weights are a test
   gap, not proof of a bug.

Report in self-review's [four parts](../code-review/SKILL.md#3-review-and-report-in-four-parts),
each finding in its [finding format](../code-review/SKILL.md#b-code-quality-and-correctness).
Do not submit speculative findings or broad refactor requests unrelated to the PR.
If no actionable findings remain, say so and list only unverified risks or missing
test coverage.

Show the user the proposed findings and confirm the review action before any
`gh pr review`, unless they already asked you to post.

## 4B. Author mode

For each reviewer or CI item:

1. Reproduce it at the current PR head. Confirm the failing run's `headSha`
   matches; a green or red run from an older SHA is stale evidence.
2. Check whether the failure also occurs at the exact base. If it does, report
   that fact with evidence, but still assess whether this PR exposes or worsens
   it.
3. Patch only the affected behavior. Do not bundle nearby cleanup.
4. Add a regression test that fails before the fix when practical.
5. Run focused tests, then the applicable CPU/sanity suite. Record exact commands
   and results for the PR body.
6. Re-fetch before pushing. If the remote head moved, stop and reconcile rather
   than overwriting another update.
7. Update the PR description when scope, compatibility, or test evidence changed.

Reply to a thread with what changed, where, and which test proves it; resolve it
only once the fix is on the remote branch. When rewriting an authored branch, keep
a backup ref and use `--force-with-lease`, never bare `--force`. Fork workflows may
not restart after a push; report when a maintainer must approve them or re-add `ci`.

## 5. Final verification and report

Before concluding, verify that local HEAD equals the SHA reviewed or pushed, the
worktree is clean or every change is explained, every current thread is accounted
for, required checks ran on that SHA, the title/body match the diff, test outcomes
are not overstated, and the AI-assistance and human-review statements remain.

Report separately:

1. **Findings/fixes** — ordered by impact, with paths and evidence.
2. **Validation** — exact commands and pass/fail/blocked outcomes.
3. **PR state** — head SHA, draft/mergeability/review decision, current checks.
4. **Remaining actions** — reviewer decisions, maintainer-only CI actions, GPU
   gaps, or public replies awaiting user approval.

<!--
MAINTAINER GUIDE — Keep this skill a workflow and router. Boundary-specific checks
belong in the review-* skills, report structure in self-review. Update the routing
tables when a review-* skill is added, renamed or changes scope.
-->

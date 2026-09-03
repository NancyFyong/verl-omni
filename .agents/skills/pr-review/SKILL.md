---
name: pr-review
description: "Review a verl-omni pull request or maintain an authored PR after CI/reviewer feedback. Use when asked to inspect a PR, triage review threads, diagnose PR CI, implement review fixes, draft replies, or decide whether a reviewer suggestion is still valid. Supports read-only reviewer mode and author-maintenance mode."
---

# PR Review

Use one evidence-first workflow for two different jobs:

- **Reviewer mode** — inspect another contributor's PR and produce findings.
- **Author mode** — maintain an authored PR: verify feedback, fix real issues,
  update tests/description, and prepare replies.

If the user's intent is unclear, ask which mode they want. Reviewer mode is
read-only unless the user explicitly asks for changes.

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

Verify again before reporting or pushing. If `headRefOid` changed, re-check the
affected diff, comments, and test evidence.

Fetch the PR head into an isolated worktree or temporary branch. First identify
the git remote that points to `verl-project/verl-omni`; do not assume it is named
`origin` or `upstream`.

```bash
git remote -v
git fetch <upstream-remote> \
  "pull/${PR}/head:refs/remotes/<upstream-remote>/pr/${PR}"
git worktree add <temporary-path> \
  "refs/remotes/<upstream-remote>/pr/${PR}"
```

In reviewer mode, keep the worktree detached/read-only. In author mode, verify
which remote owns the PR head before checking out or pushing its branch.

For a stacked PR, identify the intended parent from the PR body and commit graph.
Do not review lower-stack commits as if they belong to the current slice. Do not
blindly rebase a dirty stack or use GitHub's **Update branch** button: preserve
work in a stash/backup ref and use the actual old/new stack parents.

## 2. Collect all feedback, not only conversation comments

`gh pr view --comments` does not expose complete inline-thread state. Query
review threads so resolved/outdated comments are distinguishable from active
ones:

```bash
NWO=$(gh repo view verl-project/verl-omni \
  --json nameWithOwner --jq .nameWithOwner)
OWNER=${NWO%/*}
NAME=${NWO#*/}

gh api graphql \
  -F owner="$OWNER" -F name="$NAME" -F number="$PR" \
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

Paginate if `hasNextPage` is true. Also inspect:

- general conversation and submitted reviews;
- requested reviewers and `reviewDecision`;
- failed, skipped, pending, or stale checks;
- whether fork workflows are awaiting maintainer approval;
- whether CPU CI actually ran after the latest push (`ci` is single-use).

For each active comment, classify it before changing code:

| Classification | Action |
| --- | --- |
| Correct and current | Reproduce or prove it, then make the smallest fix. |
| Already fixed | Cite the current line/SHA and relevant test. |
| Outdated after rebase | Explain the changed context; do not recreate old code. |
| Preference/design choice | State the trade-off and ask for a decision if needed. |
| Incorrect | Reply with code/test evidence, not assertion. |
| Ambiguous | Ask the reviewer/user; do not silently choose an interpretation. |

## 3A. Reviewer mode

Review behavior, not only the patch text:

1. Read the PR body and linked issue/RFC to recover intended scope.
2. Inspect `baseRefOid...headRefOid`, then trace changed symbols into callers,
   consumers, configs, serialization boundaries, and existing tests.
3. Look first for correctness regressions, silent data loss, incompatible API or
   config changes, distributed/concurrency failures, and missing validation.
4. Run the smallest test that can falsify each concern. Expand to the relevant
   suite only after focused checks pass.
5. Separate findings from residual risk. Missing local GPU weights are a test
   gap, not proof of a bug.

A useful finding must contain:

- exact file and current line/range;
- a concrete input or execution path that triggers it;
- observable impact;
- evidence from code, test, or logs;
- the minimum viable correction (when clear).

Do not submit speculative findings or broad refactor requests unrelated to the
PR. If no actionable findings remain, say so and list only unverified risks or
missing test coverage.

Before a public review, show the user the proposed findings unless they already
asked you to post. Use `gh pr review --comment`, `--approve`, or
`--request-changes` only after confirming the desired review action.

## 3B. Author mode

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

Reply to a review thread with a short evidence-based note: what changed, where,
and which test proves it. Do not claim a GPU path passed when it only reached
engine initialization. Do not resolve a thread until the fix is visible on the
remote branch and the reply accurately describes the current head.

When rewriting an authored branch, create a backup ref and use
`--force-with-lease`, never bare `--force`. Fork PR workflows may not restart
after a push; report when a maintainer must approve workflows or re-add `ci`.

## 4. Final verification and report

Before concluding, verify:

- local HEAD equals the SHA reviewed or pushed;
- the worktree is clean (or every remaining change is explained);
- unresolved current review threads are accounted for;
- required checks actually ran on the current SHA;
- PR title/body still match the real diff;
- test commands and outcomes are recorded without overstating coverage;
- AI-assistance and human-review requirements remain present.

Report separately:

1. **Findings/fixes** — ordered by impact, with paths and evidence.
2. **Validation** — exact commands and pass/fail/blocked outcomes.
3. **PR state** — head SHA, draft/mergeability/review decision, current checks.
4. **Remaining actions** — reviewer decisions, maintainer-only CI actions, GPU
   gaps, or public replies awaiting user approval.

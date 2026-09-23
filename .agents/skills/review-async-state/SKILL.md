---
name: review-async-state
description: "Review checks for a verl-omni PR that touches V1 TransferQueue fields, async rollout, background threads, rollout server sleep/wake, or checkpoint save and resume. Use from pr-review or self-review."
---

# Review: TransferQueue, async lifecycle and persisted state

State that crosses a queue, a thread, a sleep/wake cycle or a restart is where a
bug waits the longest before it shows. The V1 trainer lives in
`verl_omni/trainer/diffusion/v1/` (field lists in `tq_utils.py`), the TransferQueue
agent loop in `verl_omni/agent_loop/diffusion_agent_loop_tq.py`, and the server
lifecycle in `verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py`.

## When it applies

```bash
git diff --name-only <base>...<head> | grep -E 'trainer/diffusion/v1/|_tq\.py|async_server|checkpoint'
git diff <base>...<head> | grep -nE 'kv_batch_put|select_fields|fields=|_persisted_tq_fields|_metric_tq_fields|Thread|executor|sleep\(|wake_up|resume_|_save_checkpoint|_load_checkpoint'
```

## TransferQueue field contract

- **Only listed fields cross.** The TransferQueue agent loop forwards tensors and
  an allowlist of non-tensor metadata; #488 found `img_shapes` dropped there, so
  Qwen-Image-Edit replay lost its RoPE shapes. A new non-tensor replay field needs
  an allowlist entry and a V1 test
  ([review-rollout-contract](../review-rollout-contract/SKILL.md)).
- **A projected read asks only for fields some writer persists.** #593 review found
  metric projections requesting fields no active trainer path wrote, so metric
  reads failed. Check every name against `diffusion_persisted_tq_fields` and each
  writer, and keep one list rather than a second copy in a script.
- **Partial writes keep the row.** A tags-only `kv_batch_put` without `fields` may
  replace the whole row; #470 review asked for proof it does not wipe restored
  prompt tensors. Check the TransferQueue version's semantics for each write.
- V0 and V1 trainers both still exist (#650 made V1 the default and deprecated V0).
  A change to one path names what happens on the other.

## Threads and async work

- **Background workers get independent copies.** V1 copies visual, audio and
  sample-rate tensors to separate CPU storage before its background exporter runs
  (#481). A tensor handed to a thread while the batch buffer is reused or freed is
  a race.
- **Correctness checks stay on the producing thread.** Moving a contract check into
  a background exporter turns an error into late warnings
  ([review-media-contract](../review-media-contract/SKILL.md)).
- **Server lifecycle ordering.** #497 replaced a workaround stack with the engine's
  real sequence: abort then pause, ACK-validated sleep and wake, a multimodal cache
  clear after sleep and after an abort pause, and an admission resume after every
  wake. Without the resume, only the `omni_sync` bridge could generate after a
  sleep/wake. A lifecycle change needs a CPU test of that order, including timeouts and
  a partial abort.

## Checkpoint and resume

- **Everything the loop needs is saved.** Diffusion V1 used to restore the actor and
  dataloader but not TransferQueue, so `separate_async` lost queued prompt groups
  and trajectories on restart (#470). New queue, buffer or counter state needs
  save, load and a round-trip test.
- **In-flight work is re-issued, finished work is kept, warmup is not overfilled.**
  #470 re-issues pending and running groups, clears their partial trajectories and
  tops warmup up from the restored state; its review caught a rule that skipped all
  warmup whenever any row existed.
- **An old checkpoint is handled out loud.** A missing `transfer_queue/` snapshot or
  a TransferQueue without checkpoint APIs must warn like the dataloader path does;
  #470 review flagged a silent skip that looks like a successful recovery.
- A changed layout needs a load test from a checkpoint written by the base branch.

## Evidence

- A CPU test that writes through the real writer and reads through the real reader
  for each new field, on the V1 path.
- For resume: save with work in flight, restart with `trainer.resume_mode=auto`,
  and show the restored queue entries and continued training (#470's GPU smoke
  saved at step 1, restored the queue and saved step 2).
- For lifecycle: a sleep, wake and generate cycle in each trainer mode that uses
  the server.

<!--
MAINTAINER GUIDE — Field names and the allowlist are read from tq_utils.py and
diffusion_agent_loop_tq.py; the lifecycle order from vllm_omni_async_server.py.
Update the bullets when TransferQueue's write semantics or the V0 trainer change.
-->

---
name: review-distributed-memory
description: "Review checks for a verl-omni PR that touches parallelism, sharding, weight-sync modes, colocation memory or device-specific paths: TP/ETP propagation, per-rank routing, engine-mode support, gradient sync, peak memory and NPU layouts. Use from pr-review or self-review."
---

# Review: distributed execution and memory

CPU tests run in one process. Parallel degrees, per-rank routing, sharding hooks,
colocated sleep/wake and device-specific layouts only exist on several ranks, so a
green CPU suite says little here. Engine code lives in `verl_omni/workers/engine/`,
rollout parallel config in `verl_omni/workers/config/diffusion/rollout.py`, and the
sync modes in the `update_weights` docstring of `verl_omni/workers/engine_workers.py`.

## When it applies

```bash
git diff <base>...<head> | grep -nE 'tensor_model_parallel|text_encoder_tp|ulysses|fsdp|FSDP|no_sync|gradient_sync|summon|offload|sleep|wake_up|colocat|checkpoint_engine|zmq|rank|world_size|is_npu_available|FusedMoE'
```

## Parallel degrees reach the engine

- **A config value can be dropped at a conversion boundary.** The pinned vllm-omni
  CLI accepted `text_encoder_tp_size`, but `OmniEngineArgs.from_cli_args` filtered
  it out, so a requested ETP=4 ran as ETP=1 with no error (#589). Test through the
  real CLI or engine-arg conversion, not the config object.
- **One validator, and docs that match it.** `DiffusionRolloutConfig` accepts ETP of
  1 or the full rollout TP only; #589 review blocked recipe docs that still listed
  1/2/4/8. Degree and divisibility checks in launchers follow the
  [shell recipe rules](../../rules/code-style.md#example-shell-recipes).
- **Every rank routes its own traffic.** `update_weights` runs on all ranks, but
  only rollout rank 0 forwards the RPC; #341 had to stop forwarding one concrete
  socket address and let each receiver derive its own. Look for a rank-0 value
  broadcast to every rank.
- A changed recipe default for a degree is a behavior change the body lists, with
  its opt-out (#589 moved the NFT launchers to ETP = TP).

## Engine and sync-mode matrix

- **A feature valid on one engine rejects the others at config time.** Regional
  compile rejects FSDP1 and Ulysses SP (#559); deferred gradient sync raises on
  VeOmni and Megatron (#614). Validation belongs in `__post_init__`
  ([config rule](../../rules/config.md#validation-lives-in-__post_init__)).
- **Order against FSDP wrapping.** #559 compiles repeated blocks after structural
  and trainability changes and before FSDP2 registers its hooks, so activation
  checkpoint recomputation runs the compiled blocks too. Ask where a new wrap,
  compile or hook sits in that order.
- **Gradient sync count and memory move together.** #614 reduce-scatters once per
  mini-batch instead of once per denoise timestep and stays opt-in so the default
  peak memory is unchanged. Its test plan names the evidence to ask for:
  actor-update time and `max_memory_allocated` with the flag on and off.
- **Colocated (`naive`) and disaggregated checkpoint-engine sync are separate
  paths.** #624 review found a LoRA change where the colocated path still exported
  adapters but the other path sent base weights on step 1 and raised on step 2. A
  sync change needs both ([review-weight-sync](../review-weight-sync/SKILL.md)).

## Memory

- **Peak memory during reload.** #514 review found that materializing
  `list(model.named_parameters())` kept every MoE parameter alive while each
  transposed copy was allocated, duplicating the expert weights and defeating the
  bucketed transfer. Look for whole-model lists, clones and dict copies on weight
  paths.
- **Meta-tensor init must survive config quirks.** `tie_word_embeddings=True`
  disabled meta-tensor init and ran out of memory on Qwen3-Omni 30B-A3B (#113; the
  workaround is tracked in RFC #445).
- **Colocated modules must be offloadable.** Rollout sleeps while the actor trains;
  a pipeline built outside the engine's loader has to live where `sleep()` can
  offload it. A scattered `empty_cache()` is not a fix
  ([code-style](../../rules/code-style.md#runtime-boundaries-and-state)).
- Treat a variant the body calls CPU-only (the colocated scorer in #480 and #481)
  as uncovered on GPU.

## Device-specific paths

- **NPU weight layouts differ after load.** vLLM-Ascend transposes FusedMoE
  weights for `npu_grouped_matmul`, so a reload must restore the checkpoint layout
  first (#514), and only for unquantized models, which the review required.
- **Mixins can collide with engine workers.** An NPU mixin applied unconditionally
  to the worker extension redefined an attribute the vLLM worker already had and
  failed on GPU (#113). Gate device mixins and test the other device's import.
- Device checks use `verl.utils.device` ([code-style](../../rules/code-style.md#device-handling)).

## Evidence

- A run at the degrees the recipe uses, for each affected algorithm; #589 ran
  TP=2/ETP=2 for both FlowGRPO and DiffusionNFT, plus stubbed argv checks of every
  launcher.
- Memory or speed claims: the same workload before and after, with
  `max_memory_allocated` and timings ([profile](../profile/SKILL.md)).
- A CUDA run does not cover NPU, and the reverse; the body must say which devices
  ran (#589 listed NPU as not revalidated).

<!--
MAINTAINER GUIDE — The ETP rule, the rejected engine combinations and the two sync
modes are read from rollout.py, the engine configs and engine_workers.py; recheck
them when those validators change.
-->

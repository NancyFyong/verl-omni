# RFC-0001: MiniMax-H3 FL2VA — Text/First-frame → Video+Audio RL

- **Status:** Draft
- **Scope (this RFC):** the `FL2VA` checkpoint partition — tasks `t2va` (text → video+audio) and `fl2va` (first-frame → video+audio)
- **Companion:** [RFC-0002](rfc-0002-minimax-h3-ref2va.md) covers the `Ref2VA` partition (`ref2va`). Together the two RFCs cover all three H3 tasks.
- **Prior art / template:** PR #341 (LTX2.3 `t2av` FlowGRPO + CLAP/ImageBind rewards) is the closest existing joint audio-video integration and the structural template for this work. This RFC is **not** a duplicate of it — H3 is a different architecture with a different (non-diffusers, CFG-distilled, dual-schedule) runtime.
- **Last updated:** 2026-08-05

---

## 0. TL;DR

MiniMax-H3 is a 33B CFG-distilled joint video+audio diffusion transformer served today only by `vllm-omni`. We want to fine-tune it with verl-omni's diffusion RL stack. This RFC scopes the **first** phase: the `FL2VA` partition, which serves text-to-video+audio (`t2va`) and first-frame-to-video+audio (`fl2va`).

The integration is not a "drop in another diffusers T2I model" job. Six properties of H3 each break an assumption baked into the current verl-omni diffusion path, and each needs an explicit design decision here:

1. **diffusers has no classic `DiffusionPipeline` for H3 — only a Modular pipeline** (`MiniMaxH3ModularPipeline`), staged on the diffusers `minimax-h3` branch; the merge-to-`main` PR #14355 is still **open** as of 2026-08-05 (nothing minimax on `main` yet), while #14371 merged review fixes *into that branch*. But the DiT is now a standard `ModelMixin` — **`MiniMaxH3Transformer3DModel`**, publicly exported — so the training module can load via the normal diffusers `AutoModel` path once we pin a diffusers build that includes it; `NonDiffusersModelBase` (BAGEL precedent) is the fallback only for older diffusers. Rollout uses vllm-omni's pipeline either way.
2. **H3's sampler is deterministic (eta=0)** despite the "euler_ancestral" filename. FlowGRPO needs a genuine reverse-time SDE with a per-step Gaussian transition. → we supply a **dual-schedule SDE scheduler**.
3. **Two coupled sigma schedules** (video `flow_shift=12.0`, audio `audio_flow_shift=3.0`). A single scalar `t` does not describe the state. → log-probs must be **per-modality**.
4. **The rollout trajectory is a joint A+V object**, but the engine derives the number of RL timesteps from a single `all_latents` tensor and slices `[:, step]`. → we add a **parallel `audio_latents` stream** rather than concatenating (a 269:1 element-count ratio would otherwise drown audio in the mean-reduced log-prob).
5. **verl-omni has zero audio support** — no audio reward scorer, no audio dump, no A/V muxing, no audio config fields. → all of that is built here.
6. **Rollout weight sync is a blind `f"transformer.{name}"` prefix** with no `map_weights` hook, and H3's rollout weights are structurally fused/reordered (fused-then-reordered qkv, chunked `fc1`) relative to the training-side layout. → we add a per-model weight-mapping seam.

Plus two hard constraints from vllm-omni: **`cfg_parallel_size` must stay 1** (CFG-distilled, no negative branch), and **one generation per diffusion request** (group rollouts must issue `n` separate requests).

We land this in milestones, cheapest-risk first: `diffusion_nft` / `dpo` (no reverse-SDE log-prob needed) → `flow_grpo`.

---

## 1. Background

### 1.1 What H3 is

MiniMax-H3 is a joint video+audio rectified-flow diffusion transformer:

- **DiT**: 50 layers, hidden size 5376, 56 heads × 128 dim, ffn 14336, patch `(1,2,2)`, video latent dim 24, audio latent dim 32, text conditioning dim 5120 (`vllm-omni: minimax_h3_transformer.py:47-78`).
- **Text encoder**: Qwen3-VL truncated at layer 50, emitting a `[L, 5120]` hidden state + a per-token `tags` vector. ~51.5 GB BF16 — the memory hotspot; sharded by `--text-encoder-tp-size` (`encoder.py:898-1211`).
- **Two VAEs**: a video VAE (fp32, spatial compression 16, 24 latent channels) and an audio VAE (32 kHz stereo, 32 latent channels) (`vae.py:97-357`).
- **Packed sequence**: one flat token sequence carries text / conditioning / audio / video rows. Video rows are 96-wide (24 latent ch × 1·2·2 patch), audio rows 32-wide, aligned to 64 (`packed_sequence.py`, `packed_tokens.py`).

### 1.2 The three tasks and the two partitions

| Task | Input | Partition dir | Served by |
|------|-------|---------------|-----------|
| `t2va` | text | `FL2VA/` | this RFC |
| `fl2va` | text + first frame (image) | `FL2VA/` | this RFC |
| `ref2va` | text + reference image+audio, or 1+ reference videos | `Ref2VA/` | [RFC-0002](rfc-0002-minimax-h3-ref2va.md) |

One server process loads exactly one partition. Both partitions declare `_class_name: "MiniMaxH3Pipeline"` in their `model_index.json`, so **one registered adapter pair `("MiniMaxH3Pipeline", <algorithm>)` serves both RFCs**; the task is selected at request time via `extra_args["task"]`, constrained by which partition is loaded — never by a registry key (this is why the split is by config, not by two registry entries).

### 1.3 Checkpoint layout (HF `MiniMaxAI/MiniMax-H3`, `gated: false`, 280 files)

```
FL2VA/       (81 files: trust_remote_code .py + model.safetensors)   <- vllm-omni rollout format
Ref2VA/      (81 files)                                              <- vllm-omni rollout format
transformer/     (config.json + diffusion_pytorch_model-*.safetensors, 14 shards)   <- diffusers-converted
transformer_ref/ (same shape, for ref2va)                                            <- diffusers-converted
vae/  audio_vae/  text_encoder/  processor/  tokenizer/  scheduler/  audio_scheduler/
model_index.json  modular_model_index.json
```

Rollout reads `$MODEL_ROOT/FL2VA`; training reads the shared `transformer/` (diffusers-converted) plus the shared `text_encoder/`, `vae/`, `audio_vae/`, `processor/`, `tokenizer/`. **The two sides speak different weight-name dialects for the same DiT** — see §5.6.

### 1.4 Where FL2VA sits in the H3 request API

Task and shape are per-request `extra_args`:

```python
OmniDiffusionSamplingParams(
    height=256, width=448, num_inference_steps=4, seed=1234,
    extra_args={"task": "t2va", "duration": 1.6, "flow_shift": 12.0, "audio_flow_shift": 3.0},
)
```

`fl2va` additionally passes an ordered image list in `multi_modal_data["image"]`; it is wired **first-frame only** today (`pipeline_minimax_h3.py:909-911` hardcodes `keyframe_frame_indices=[0]`; `_load_image` rejects >1 image; `model_metadata.py:41` caps `max_multimodal_image_inputs=1`) even though `packed_sequence.py:21-25` structurally accepts `(0,)`, `(-1,)`, `(0,-1)`. True first+last is a **non-goal** here (§3).

---

## 2. Motivation

RL post-training (FlowGRPO/DPO/NFT) has measurably improved the aesthetic/prompt-alignment quality of the image diffusion models already integrated in verl-omni, and (via the in-flight PR #341) is being extended to audio-video. H3 is the strongest open joint-A/V model available and is already production-served by vllm-omni; wiring it into verl-omni lets us apply the same reward-driven fine-tuning to synchronized video+audio generation. Doing `FL2VA` first gives us the whole plumbing (audio latents, dual-schedule SDE, A/V reward, weight-sync map) against the simplest conditioning (text, optionally a single first frame) before we take on reference conditioning.

---

## 3. Goals / Non-goals

**Goals (this RFC)**

- G1. `t2va` and `fl2va` rollout through a `MiniMaxH3PipelineWithLogProb` subclass in vllm-omni, exposing per-step joint (video, audio) latents + log-probs on `DiffusionOutput.trajectory_*`.
- G2. A training-side module load path that does not depend on diffusers pipeline support for H3.
- G3. A dual-schedule reverse-SDE scheduler producing per-modality Gaussian log-probs consistent with `FlowMatchSDEDiscreteScheduler`'s contract.
- G4. Joint A+V trajectory transport through the FSDP engines and losses without silently ignoring the audio modality.
- G5. Weight sync (actor → rollout) across the fused/reordered name mismatch.
- G6. Audio-aware reward + rollout/validation dump (mp4 with muxed stereo audio).
- G7. Land `diffusion_nft` and/or `dpo` first, then `flow_grpo`, with CPU tests and one GPU smoke test per milestone.

**Non-goals (deferred or out of scope)**

- N1. `ref2va` — [RFC-0002](rfc-0002-minimax-h3-ref2va.md).
- N2. **True first+last-frame `fl2va`.** Needs an upstream vllm-omni change to `keyframe_frame_indices` and the `max_multimodal_image_inputs=1` cap. We ship first-frame-only and note the upstream ask.
- N3. **Request-level batching** (`supports_request_batch`). vllm-omni currently executes one generation per diffusion batch; group rollouts issue `n` separate requests. Enabling batching is upstream work.
- N4. **FP8 training/rollout.** FP8 is inference-only in vllm-omni and incompatible with layerwise offload; a weight push would have to re-run online quantization. bf16 only.
- N5. **Step-execution / continuous batching** for H3. H3 does not implement `SupportsStepExecution`; the stepwise engine driver is unavailable. Full-forward rollout only.
- N6. `sleep(level=2)` for colocated memory reclaim — `wake_up()` after level-2 raises `NotImplementedError` in vllm-omni. Only `sleep(level=1)` round-trips.

---

## 4. High-level architecture

```
                        model.path = $MODEL_ROOT/FL2VA        (rollout, vllm-omni format)
                        transformer_subfolder = transformer   (training, diffusers-converted)
                                       │
   ┌───────────────────────────────────┴───────────────────────────────────┐
   │ TRAINING (FSDP)                          │ ROLLOUT (vllm-omni, colocated) │
   │                                          │                                │
   │ verl_omni/pipelines/minimax_h3_<algo>/   │ MiniMaxH3PipelineWithLogProb   │
   │   diffusers_training_adapter.py          │  (subclass of MiniMaxH3Pipeline)│
   │     : diffusers-native load (default)    │  - drop @torch.no_grad /        │
   │       DiT from transformer/              │    inference_mode               │
   │       (Wan/Qwen-Image path)              │  - dual-schedule SDE step       │
   │   minimax_h3_model.py                     │  - capture via on_step          │
   │     : NonDiffusersModelBase (fallback)   │  - populate trajectory_*        │
   │   vllm_omni_rollout_adapter.py ──────────┼─▶ injected via                  │
   │                                          │   custom_pipeline_args          │
   │   schedulers/flow_match_dual_sde.py      │                                │
   └──────────────────────────────────────────┴────────────────────────────────┘
                 registry keys: ("MiniMaxH3Pipeline", "diffusion_nft" | "dpo" | "flow_grpo")
                 registered by star-import in verl_omni/pipelines/__init__.py
```

Files added (mirrors the LTX2 package shape from PR #341):

```
verl_omni/pipelines/minimax_h3_<algo>/
    __init__.py                       # imports adapters => triggers @register
    common.py                         # shared packing/schedule helpers (grows across milestones)
    minimax_h3_model.py               # NonDiffusersModelBase fallback loader (only if diffusers predates the H3 merge)
    diffusers_training_adapter.py     # DiffusionModelBase subclass (<Model><Algo>)
    vllm_omni_rollout_adapter.py      # VllmOmniPipelineBase + MiniMaxH3Pipeline subclass
verl_omni/pipelines/schedulers/flow_match_dual_sde.py
verl_omni/utils/reward_score/<av_reward>.py    # audio-aware scorer (CLAP/ImageBind-style)
tests/pipelines/test_minimax_h3_<algo>_on_cpu.py
tests/pipelines/test_minimax_h3_dual_sde_on_cpu.py
tests/special_e2e/run_<algo>_minimax_h3.sh      # wired into tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh
examples/<algo>_trainer/minimax_h3/{README.md,prepare_data.py,run_*.sh}
```

Config edited (both must stay in sync — §5.7):

```
verl_omni/workers/config/diffusion/rollout.py     # new audio + H3 fields on DiffusionPipelineConfig
verl_omni/trainer/config/diffusion/rollout/diffusion_rollout.yaml     # top-level + val_kwargs
verl_omni/trainer/config/diffusion/model/diffusion_model.yaml         # ${oc.select:...}
verl_omni/trainer/config/_generated_diffusion_trainer.yaml            # regenerated, never hand-edited
verl_omni/trainer/config/_generated_diffusion_veomni_trainer.yaml     # regenerated
verl_omni/pipelines/__init__.py                                       # star-import the new package
docs/index.md                                                          # rfcs toctree
```

---

## 5. Detailed design

### 5.1 Registry keys and dispatch

`architecture` is auto-detected from `model_index.json::_class_name` (`DiffusionModelConfig` auto-detect at `model.py:133-136`); `algorithm` comes from `DiffusionModelConfig.algorithm`. **Caveat on auto-detection:** the partition dirs (`FL2VA/`, `Ref2VA/`) declare `_class_name: "MiniMaxH3Pipeline"`, but the repo **root** `model_index.json` declares `"MiniMaxH3ModularPipeline"` (the new diffusers Modular class). Rollout points at the partition dir while training points at the root, so the two would auto-detect *different* architecture strings for one run. We therefore **pin `model.architecture=MiniMaxH3Pipeline` explicitly** (the BAGEL precedent, `+actor_rollout_ref.model.architecture=...`) and register exactly one adapter pair under that single key:

```python
@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")   # milestone 1
@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")       # milestone 3
@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
```

Registration is an import side effect; the only importer is `verl_omni/pipelines/__init__.py`, so the new package **must** be star-imported there or it silently does not exist (per `.agents/rules/pipelines.md`). Task differentiation (`t2va` vs `fl2va`) is by config/request, never by registry key.

### 5.2 Training-side module load — diffusers-native, `NonDiffusersModelBase` fallback

As of 2026-08-05 diffusers defines `MiniMaxH3Transformer3DModel` (a standard `ModelMixin`, `models/transformers/transformer_minimax_h3.py`) and exports it from `diffusers/__init__.py`, alongside `AutoencoderKLMiniMaxH3` / `AutoencoderKLMiniMaxH3Audio`, `MiniMaxH3Scheduler`, and the Modular pipeline (`MiniMaxH3ModularPipeline` / `MiniMaxH3Blocks`). This code lives on the `minimax-h3` integration branch; the →`main` PR #14355 is still open. What matters for **training** is that the **DiT is now a standard diffusers model class** — the pipeline being Modular-only is irrelevant to the training module (rollout uses vllm-omni's pipeline regardless).

So the training-side module load is a **dependency-pin decision**, not a hard non-diffusers requirement:

- **Preferred — diffusers-native (pin to PR #14355).** The upstream H3 doc (`docs/source/en/api/pipelines/minimax_h3.md`) gives the exact install: `pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head`. Mirror it in `pyproject.toml` — replace the `diffusers>=0.37.1` line (`:26`) with a `git+https` pin (the `verl` pin at `:64` is the precedent). For a **reproducible** build, freeze the moving `refs/pull/14355/head` to a specific commit SHA (branch head `99ced1b` as of 2026-08-05) rather than the PR ref — same as how `verl` is pinned to a fixed SHA, not a branch. Then the training adapter uses the **standard `DiffusionModelBase` diffusers path** — like Wan/Qwen-Image — loading the DiT from the diffusers-converted `transformer/` (or `transformer_ref/`) with `transformer_subfolder=transformer`. This pulls in only the `MiniMaxH3Transformer3DModel` `ModelMixin`; the doc's `ModularPipeline.from_pretrained(..., workflow=...)` example is the **inference** path and is *not* used here (rollout is vllm-omni). No vendored loader. `DiffusionModelConfig` auto-detects `transformer_config` from `<local>/transformer/config.json` (`model.py:138-151`). Switch to a released `diffusers>=<version>` once #14355 lands on `main`.
- **Fallback — `NonDiffusersModelBase`** (BAGEL precedent, `integrating_a_non_diffusers_model.md`) only if we must run on a diffusers predating the merge: a `minimax_h3_model.py` subclass implementing `from_pretrained` (load the 14 `transformer/` shards into a vendored `nn.Module` matching `transformer/config.json`), `forward`, and `_no_split_modules`, with the training adapter overriding `build_module()` to return it.

Either way:

- The training adapter (`DiffusionModelBase` subclass) stays **stateless** — all `@classmethod`, no `__init__` (`.agents/rules/pipelines.md`). Diffusers-native uses the default `build_module()`; the fallback overrides it.
- **Path divergence to resolve at implementation:** rollout needs the `FL2VA/` partition dir (vllm-omni trust_remote_code format) while training needs the root `transformer/` (diffusers format). `model.path` is a single value, so this needs either a rollout-side path override or a staging layout (symlink `transformer/` under a path vllm-omni ignores). Flag as an implementation detail; it does not change the design.
- `fsdp_layer_prefixes` must match the H3 DiT block prefix from the diffusers `transformer/config.json` (default `["transformer_blocks."]` may be wrong — vllm-omni's rollout module names its stack `blocks` at `minimax_h3_transformer.py:595`; confirm the diffusers-side name at implementation). Getting this wrong makes `collect_lora_params` raise on 0 collected params (`fsdp_utils.py:256-261`).
- One checkpoint-manager caveat survives regardless of path: verl-omni's FSDP save needs a non-frozen `config.save_pretrained` for the DiT. A diffusers `ModelMixin` config supports `save_pretrained`, so the diffusers-native path is *safer* here than a custom module — the `FrozenDict`/`can_generate` patch (`diffusers_impl.py:293-299`) is only needed on the custom-loader path (`fsdp_checkpoint_manager.py:341`).

### 5.2.1 Transformer forward interface — unified packed sequence, per-sample loop (verified 2026-08-05)

`MiniMaxH3Transformer3DModel.forward` (diffusers PR #14355) reads **one unified packed sequence**, not a batched pair of media tensors: `hidden_states` `(B, Nv, 96)` video rows, `audio_hidden_states` `(B, Na, 32)` audio rows, `encoder_hidden_states` `(B, Ntext, D)` text, plus the static layout — `token_tags` `(seq,)`, `position_ids` `(seq, 3)`, and `video_indices` / `audio_indices` / `text_indices` — and the timestep plan `timestep` `(num_distinct,)` + `timestep_indices` `(seq,)`; it returns `(v_video, v_audio)`. **The layout and timestep-plan tensors carry no batch dim, and the forward takes no attention mask** — one forward serves exactly one row layout at one timestep plan.

Two facts force a **per-sample loop** in the training adapters (`forward`): each sample has its own unpadded text length — which shifts the entire media rotary clock (`common.py::build_packed_sequence`, §5.5) — and, for M3, its own per-modality sampled timesteps. So the adapter iterates the micro-batch one sample at a time: it builds that sample's layout with `build_layout_from_meta` (→ the verbatim-ported `build_packed_sequence`, deriving the args from the `latent_meta` `[Nv, Na, latent_t, latent_h, latent_w, audio_t]` row), calls the transformer on the 1-sample slice sliced to the sample's true text length, splits `(v_video, v_audio)`, and `torch.cat`s the re-packed rows back to `(B, Nv·96 + Na·32)`. The loop lives entirely under `forward`, so the engine's `(B, packed)` contract is untouched — **zero engine edits**, consistent with the §5.5c packed-transport decision.

The timestep plan is the one thing that differs by algorithm:

- **M1 `diffusion_nft` (Option C).** The engine noised the whole packed latent at one level, so every row shares it: `timestep_indices` is all-zero and there is a single distinct timestep.
- **M3 `flow_grpo` (per-modality).** Video and audio rows sit at *different* noise levels each step (dual schedules, §5.4), so the adapter routes per modality via the verbatim-ported `common.py::build_row_timesteps` — video rows → video timestep, audio rows → audio timestep, text inherits the video timestep — which reduces (`torch.unique(sorted=True, return_inverse=True)`) to the `(distinct timesteps, per-row index)` pair the forward expects (two distinct when video ≠ audio, collapsing to one when equal). t2va has no conditioning rows, so the condition-timestep arguments are inert here.

Both ports are checked against diffusers: `build_row_timesteps` is **bit-identical** to `MiniMaxH3SetTimestepsStep.build_row_timesteps`, and both adapters' per-sample loops were run against a tiny real `MiniMaxH3Transformer3DModel` in an isolated diffusers-PR-#14355 venv (distinct video/audio timesteps produce two routed distinct timesteps and measurably change the velocity). This supersedes the earlier working assumption of a flat, scalar-timestep media interface.

### 5.3 Rollout adapter — `MiniMaxH3PipelineWithLogProb`

The rollout adapter subclasses both `VllmOmniPipelineBase` and the upstream `MiniMaxH3Pipeline`, and is injected into the vllm-omni engine via `custom_pipeline_args={"pipeline_class": "<qualname>"}` (the sanctioned path; blueprint is `vllm-omni: tests/e2e/features/helpers/custom_pipeline.py`, the `QwenImagePipelineWithLogProbForTest` shape). It must:

1. **Remove `@torch.no_grad()` (`pipeline_minimax_h3.py:1028`) and the `torch.inference_mode()` in `denoise_loop.py:208-209`** in the subclass path — inference-mode tensors cannot re-enter autograd, and even for RL rollout we must clone captured latents out of inference mode. (Rollout itself is no-grad, but the captured trajectory tensors are consumed later by the training engine; they must be ordinary tensors.)
2. **Swap the deterministic step for the dual-schedule SDE step** (§5.4), gated on `sampling_params.extra_args["logprobs"]` so the production deterministic path is untouched when the flag is absent.
3. **Capture per-step latents/log-probs.** The natural hook already exists: `minimax_h3_denoise_loop(..., on_step=Callable[[int, video_rows, audio_rows], None])` (`denoise_loop.py:129-144`, invoked `:236-237`). We pass a capturing callback rather than the progress-bar lambda the pipeline uses today (`pipeline_minimax_h3.py:986`). Note the loop already `.clone()`s at `:219,:232`, so captured tensors are not aliased. We also need `v_video`/`v_audio` (the velocity, `:209-211`) for the SDE mean — these are **not** exposed by `on_step` today, so either (a) extend `on_step`'s signature in our subclass by reimplementing the loop body, or (b) recompute the mean from consecutive latents. **Decision: reimplement the loop in the subclass** (cleaner, and we already override `diffuse`).
4. **Override `encode_prompt`** to accept pre-tokenized `prompt_ids` + mask from the dataset (the dataset hands token ids, not raw text, on the RL path) and return padded `(B, L, D)` + `(B, L)`. H3's encoder emits `(hidden[L,5120], tags[L])`; the `tags` vector must be threaded through as an extra output (it is a required transformer kwarg, `token_tags` in `_FORWARD_SUPPORTED_KWARGS`).
5. **Override `forward(req, ...)`** to populate `DiffusionOutput.trajectory_{latents,log_probs,timesteps}` (transport already exists, `vllm-omni: data.py:1290-1352`) **and** put `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, `negative_prompt_embeds_mask` in `custom_output` under exactly those names (the integration guide says "do not rename them"). H3 is CFG-distilled and has no negative branch — we still must provide the keys; `negative_*` are zero-length / empty tensors with matching dtype, and the loss must tolerate that (documented, §5.8).

Naming per convention: rollout adapter for a policy-gradient algo is `XxxWithLogProb` → `MiniMaxH3PipelineWithLogProb`; for DPO/NFT it is `<Model><Algo>Pipeline` → `MiniMaxH3DiffusionNFTPipeline`.

### 5.4 Dual-schedule reverse-SDE scheduler

H3's `scheduling_minimax_h3_euler_ancestral.py` is **eta=0 deterministic** — `minimax_h3_euler_eta0_step` is a pure lerp `out = (σ_next/σ_cur)·state + (1 − σ_next/σ_cur)·denoised` (`:98-99`); there is no `generator`, no noise, no `torch.randn` in the file. The "ancestral" name is a misnomer. So there is **no per-step transition distribution today**, hence no log-prob, hence no policy gradient.

verl-omni's `FlowMatchSDEDiscreteScheduler` (`schedulers/flow_match_sde.py`) already turns a deterministic flow-match scheduler into a stochastic one with a Gaussian per-step transition and the matching log-prob (`sample_previous_step`, `:148-321`). We create `flow_match_dual_sde.py` next to it that:

- Runs **two coupled sigma schedules** — video (`flow_shift=12.0`) and audio (`audio_flow_shift=3.0`) — because a single `t` does not describe the joint state. Concretely, two instances of the SDE machinery (or one class holding two sigma arrays) stepping the video rows and audio rows with their own `sigma`, `sigma_next`, `std_dev_t`.
- Produces a **per-modality** `(log_prob, prev_sample_mean, std_dev_t)`. The two log-probs are combined **inside the training/rollout adapter** (§5.5), not silently summed and not in the loss/engine.
- Keeps the hard fp32 asserts (`flow_match_sde.py:194-197`) and the `sde`/`cps`/`dance_sde` branch structure. H3's DiT and video VAE run fp32-critical params (asserted in `post_load_weights`, `minimax_h3_transformer.py:930-936`); storing the trajectory in fp32 is mandatory (`common_pitfalls.md` "Float32 Precision Loss in Stored Rollout Latents" and "…in Stepwise Scheduler").
- Matches the ancestral formulation so `std_dev_t * sqrt(-dt)` equals the intended `sigma_up`, keeping `FlowDPPOLoss`/`GRPOGuardLoss`/`KLLoss` correct if they are later selected as loss modes.

The rollout adapter replaces the upstream H3 scheduler with this one (the guide requires the rollout adapter to swap in the SDE scheduler; here it is the dual-schedule variant).

### 5.5 Joint A+V trajectory representation — **packed transport (chosen)**

The engine derives the number of RL timesteps from a single timesteps key and slices `[:, step]` (`diffusers_impl.py:784`, `:895-921`); the rollout stacks a single `all_latents` into `(B, T, ...)` (`qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:620`). H3's trajectory is a joint (video, audio) pair per step, so we must choose how to carry both:

- **(a) Naive concatenate** — one latent tensor, log-prob reduced with the engine's **mean over all non-batch dims** (`flow_match_sde.py:313`). **Rejected**: at full quality video has ≈62,496×96 ≈ 6.0M elements vs audio ≈696×32 ≈ 22k (~269:1), so audio contributes ~0.37% of the mean-reduced signal and is effectively ignored.
- **(b) Parallel `audio_latents` stream** alongside `latents`, with the engine taught a second per-step slice (`audio_latents[:, step]`). **Original choice; superseded** — it requires editing `PPODiffusersFSDPEngine` (`postprocess_batch_func` `:525-558`, `_run_forward_backward_batch` `:776-813`, the slice sites `:895-921`), i.e. forking the shared engine for one model.
- **(c) Packed transport — CHOSEN and implemented (2026-08-05).** Keep (a)'s single `all_latents` carrier and its zero engine changes, but defeat the 269:1 drowning the way (b) intended: pack video and audio rows flat into `all_latents` (`minimax_h3_diffusion_nft/common.py::pack_video_audio_rows`, with a `latent_meta` `(B, 6)` = `[Nv, Na, latent_t, latent_h, latent_w, audio_t]` custom_output key driving the unpack), then **unpack inside the adapter and mean-reduce each modality's log-prob separately before the weighted combine** `w_v·lp_v + w_a·lp_a`. The mean-over-all-dims that drowned audio in (a) never runs on the concatenated tensor — each stream is reduced on its own element count first (inside its `FlowMatchSDEDiscreteScheduler` leg, `flow_match_sde.py:313`), so the two log-probs are commensurate regardless of the element ratio. The combine and the dual-SDE both live entirely in the training adapter's `forward_and_sample_previous_step` and the rollout adapter; the dual-schedule scheduler (§5.4) is a thin container of two stock `FlowMatchSDEDiscreteScheduler` legs.

Why (c) needs no engine edit: `PPODiffusersFSDPEngine.forward_step` passes the **whole micro-batch** as `scheduler_inputs` (`diffusers_impl.py:887`), so the adapter reads `all_latents`, the video `all_timesteps`, the extra per-modality `audio_all_timesteps`, and `latent_meta` directly — the engine never needs to know the tensor is dual-modality. **Result: ZERO changes to `PPODiffusersFSDPEngine` and `FlowGRPOLoss`** (no engine subclass, no `all_audio_latents` key, no second slice site), honoring "don't fork the shared engine" (`.agents/rules/code-style.md`). `w_v`/`w_a` default `1.0/1.0` and are exposed as `av_logprob_{video,audio}_weight` (§5.7). M1 (`diffusion_nft`) already carries both modalities through `common.py`; M3 adds only the per-modality `audio_all_timesteps` key and the dual-SDE + weighted log-prob combine.

### 5.6 Weight sync (actor → rollout) across the name mismatch

**This is a hard blocker with no existing seam.** Rollout weight sync uses one blind prefix — `get_per_tensor_param` yields `f"transformer.{name}"` (`diffusers_impl.py:771-772`), and `DiffusionModelBase` has **no `map_weights` hook** (`model_base.py:126-137` exposes only `build_module`/`configure_trainable_params`). But the two sides are structurally different for the same DiT:

- Rollout (`FL2VA/`, trust_remote_code): `*.attn.qkv_proj.weight` is fused **and reordered** by `_reorder_grouped_qkv_to_qkv` (`minimax_h3_transformer.py:139-168`, applied at load `:952-961`); `mlp.fc1` is one tensor chunked into gate/up (`:962-970`).
- Training (`transformer/`, diffusers-converted): separate projections / diffusers names.

So a blind-prefix push will silently miss or reject keys inside vllm-omni's loader. We add a **per-model weight-mapping seam**:

- Add an optional `map_weights(cls, named_params_iter) -> Iterable[tuple[str, Tensor]]` classmethod to `DiffusionModelBase` (default identity), called at `diffusers_impl.py:771-772` in place of the bare prefixing. This is a small, backward-compatible base-class addition (every existing adapter inherits the identity default). **This is an API addition and must be flagged in the PR title area — but it is additive, not breaking.**
- H3's `map_weights` performs the inverse of the rollout-side transforms: fuse+reorder q/k/v into `qkv_proj`, concatenate gate/up into `fc1`, apply the diffusers→vllm-omni name remap. The forward transforms are the authoritative reference (`_reorder_grouped_qkv_to_qkv`, the fc1 split, and the encoder `_map_weight_name` at `encoder.py:955-993`), so `map_weights` is their inverse.
- **LoRA path is worse and is a non-goal for milestone 1.** A fused rollout projection has no 1:1 LoRA counterpart for separate q/k/v adapters (`vllm_rollout/utils.py:112-128` hands raw PEFT-named A/B tensors to `LoRAModel.from_lora_tensors`). Full-parameter training first; LoRA support is a follow-up that needs a LoRA-aware `map_weights` (fuse A/B per-projection). Documented as a known limitation.

Because H3's rollout weights include fused/reordered layouts, the mapping is non-trivial and must be **unit-tested on CPU** against a tiny random H3 config, round-tripping training-names → rollout-names → loaded (§7).

### 5.7 Config surface

`DiffusionPipelineConfig` (`rollout.py:58-70`) today has **no audio knobs** and `num_frames=1`. New fields (mirrored in BOTH `diffusion_rollout.yaml` top-level *and* `val_kwargs.pipeline:`, and referenced from `diffusion_model.yaml` via `${oc.select:actor_rollout_ref.rollout.pipeline.<field>,<default>}` — the §5.1 config-hygiene rule of the integration guide):

| Field | Default | Meaning |
|-------|---------|---------|
| `task` | `"t2va"` | H3 task; validated against loaded partition |
| `duration` | `2.0` | seconds; → frame count via H3's planner |
| `fps` | `24` | fixed by H3 (planner raises otherwise) |
| `flow_shift` | `12.0` | video sigma shift |
| `audio_flow_shift` | `3.0` | audio sigma shift |
| `audio_sample_rate` | `32000` | for dump/muxing/reward |
| `av_logprob_video_weight` | `1.0` | `w_v` in §5.5 |
| `av_logprob_audio_weight` | `1.0` | `w_a` in §5.5 |

`num_frames` already exists (`:70`); we reuse it (derived from `duration×fps` if unset). All new fields are additive with defaults, so no `[BREAKING]`. The generated yamls are regenerated via `scripts/generate_trainer_config.sh` — never hand-edited (pre-commit `autogen-trainer-cfg` gate).

### 5.8 Losses / algorithm classification

Per `integrating_a_new_direct_preference_algorithm_for_diffusion_model.md` "Classify the Algorithm First":

| Milestone | `algorithm` | `trainer_type` | engine | loss | needs reverse-SDE log-prob? |
|-----------|-------------|----------------|--------|------|------------------------------|
| M1 | `diffusion_nft` | `direct_preference` | `NFTDiffusersFSDPEngine` | `DiffusionNFTLoss` | **No** |
| M2 (optional) | `dpo` | `direct_preference` | `DPODiffusersFSDPEngine` | `DPOLoss` | **No** |
| M3 | `flow_grpo` | `policy_gradient` | `PPODiffusersFSDPEngine` | `FlowGRPOLoss` | **Yes** |

M1 is the cheapest correct entry point: `DiffusionNFTLoss` consumes `forward_prediction`/`old_prediction`/`ref_forward_prediction`/`x0`/`xt`/`t_expanded` + `reward_prob` (`diffusion_algos.py:781`), none of which requires a correct reverse-SDE transition — so we can validate the whole A/V plumbing (module load, joint trajectory, weight sync, reward, dump) **before** the dual-schedule SDE has to be numerically exact. M3 turns on `FlowGRPOLoss` (`old_log_probs`/`advantages`, `:268`) and depends on §5.4 being right.

For M3, the joint log-prob fed to `FlowGRPOLoss` as `log_probs` is the §5.5 weighted combination. `FlowGRPOLoss` reduces per-sample; we supply one scalar log-prob per sample already combined across modalities.

The CFG-distilled `negative_*` empty-tensor convention (§5.3) is inert for these losses (none consult a negative branch), but the rollout contract still requires the keys present.

### 5.9 Trajectory-length invariant

`num_inference_steps=N` → N sigmas → **N−1** transformer forwards (`time_request.py:47-61`, confirmed by the H3 contract test). So `len(all_latents) == len(all_timesteps) + 1` and there are `N−1` RL steps. Both the video and audio schedules step `N−1` times (same step count, different sigmas), so `num_timesteps` derived from the video timesteps key is correct for the audio stream too. This off-by-one must be asserted in the CPU test to avoid a silent shape mismatch in the engine's `[:, step]` loop.

### 5.10 Audio reward and rollout/validation dump

verl-omni has **zero** audio today (no scorer, no `wandb.Audio`, no `soundfile`/`torchaudio.save`, no A/V mux). We build:

- **Reward scorer** under `verl_omni/utils/reward_score/` — **out of M1 scope and not yet built**. Its intended template, PR #341's `clap.py` (audio-text) and `imagebind.py` (joint A/V binding), is **open/unmerged as of 2026-08-05**, so neither file exists in-tree today; the H3 audio-aware scorer is a from-scratch follow-up. When built, rewards consume **in-memory tensors, never file paths** (the `VisualRewardManager` contract, `visual.py:70-88`), so the scorer takes the decoded video array + audio waveform + sample rate that H3's post-process already returns (`np` output: `{"video":[ndarray(T,H,W,3)], "audio":..., "audio_sample_rate":32000, "fps":24}`). `assemble_rm_scores` returns `(bsz, 1)` (`visual.py:40-43`).
- **Dump**: extend `ray_diffusion_trainer._dump_generations` (`:280-337`, currently `export_to_video` at `:300`) to mux stereo audio into the mp4 (H3's server output is H.264 + synchronized stereo). `ffmpeg`/`ffprobe` on PATH is required (H3 recipe). Guard against the known channels-last bug being fixed by upstream PR #340; the mp4 RGB/BGR contract is asserted by `test_dump_generations_video_on_cpu.py:80`.
- New trainer config already exists for video (`video_fps`, `rollout_data_save_freq`, etc. in `diffusion_trainer.yaml`); we add audio-sample-rate plumbing there.

### 5.11 Memory, parallelism, cost envelope

- **bf16 only** (N4). Peak hotspot is the Qwen3-VL text encoder (~51.5 GB BF16); shard with `--text-encoder-tp-size N` (N divides 64 heads / 8 KV heads).
- **`cfg_parallel_size` must stay 1** — vllm-omni raises otherwise (`pipeline_minimax_h3.py:280-282`); H3 is CFG-distilled with no negative branch. This is asserted by the H3 contract test and we must not attempt CFG-parallel rollout.
- USP (Ulysses+ring), HSDP, VAE patch-parallel (tile mode, size 1 or full group) are available and orthogonal.
- **Cost**: full-quality FL2VA is ~87 s/generation (209 frames, 1248×768, 4×B300). RL is infeasible at that size. RFC target rollout config: reduced resolution/duration/steps — e.g. 512×896, ~39 frames, `num_inference_steps≈10`. Latent budget at that size ≈ 2.06 MB/step video → ~22.7 MB/sample (T=10) → ~182 MB per `n=8` group (vs ~2.1 GB per group at full quality). Exact numbers to be pinned in the example script; the point is the rollout config must be small.
- One generation per request (N3): a group of `n=8` issues 8 separate `AsyncOmni.generate` calls. Per-request vs per-GPU SDE seeding is a known pitfall (`common_pitfalls.md` "SDE Window Per-Request vs Per-GPU Seeding") — the dual-SDE scheduler must seed per-request.

---

## 6. Milestones

| # | Deliverable | Gate |
|---|-------------|------|
| M0 | Package skeleton + registry + star-import + config fields + CPU import test | `test_minimax_h3_*_on_cpu.py` imports adapters on CPU (lazy diffusers/vllm_omni imports) |
| M1 | `diffusion_nft` end-to-end: module load, joint A+V trajectory, weight-sync `map_weights`, A/V reward, dump | GPU smoke `run_diffusionnft_minimax_h3.sh`; reward increases on a tiny run |
| M2 | (optional) `dpo` offline path | reuses M1 plumbing; `DirectPreferenceRayTrainer` |
| M3 | `flow_grpo`: dual-schedule SDE numerically validated, `FlowGRPOLoss` | log-prob matches a finite-difference reference within tol (CPU test); GPU smoke |
| M4 | Docs: `docs/algo` note only if we introduce algorithm-level concepts; example READMEs; this RFC marked Accepted | — |

Rationale for order: M1 exercises every new piece of plumbing **except** the reverse-SDE math, so a bug in §5.4 cannot masquerade as a plumbing bug. M3 isolates the one genuinely novel numerical component.

**Status (2026-08-05):** M0 and the CPU-provable portions of M1 (`diffusion_nft`) and M3 (`flow_grpo`) have landed on the `minimax-h3-support` branch — package skeletons, registry keys, the packed-transport training adapters (§5.5c) calling the real unified packed-sequence forward via a per-sample loop (§5.2.1), the verbatim-ported layout/timestep helpers in `common.py` (`build_packed_sequence`, `build_layout_from_meta`, `build_row_timesteps`), the dual-schedule SDE scheduler (`flow_match_dual_sde.py`, §5.4), the `av_logprob_{video,audio}_weight` config fields (§5.7), and passing CPU tests (`test_minimax_h3_diffusion_nft_on_cpu.py`, `test_minimax_h3_flow_grpo_on_cpu.py`, `test_minimax_h3_dual_sde_on_cpu.py`). The `common.py` ports and both adapters' per-sample forward loops are additionally validated against a tiny real `MiniMaxH3Transformer3DModel` in an isolated diffusers-PR-#14355 venv (§5.2.1). The rollout adapters are written but GPU-only (import-guarded). Everything requiring the 33B checkpoint or live diffusers/vllm-omni weights — the diffusers pin (§5.2), weight-sync `map_weights` (§5.6), the A/V reward and mp4 dump (§5.10), the GPU smoke, and the vllm-omni pin bump (§8) — is deferred to GPU bring-up. No PR is opened in this pass (CLAUDE.md §1).

---

## 7. Test plan

**CPU tests** (`test_*_on_cpu.py`; CI selects by suffix; `TORCH_COMPILE_DISABLE=1`):

- `test_minimax_h3_<algo>_on_cpu.py` — adapters import on CPU; transformer `MagicMock`ed; `DiffusionModelConfig` built via `object.__new__`/`object.__setattr__`; `prepare_model_inputs` slices `[:, step]` correctly for both video and audio streams; trajectory-length invariant `len(latents)==len(timesteps)+1` (§5.9).
- `test_minimax_h3_dual_sde_on_cpu.py` — dual-schedule SDE: fp32 asserts hold; per-modality log-prob shape; and a **finite-difference check** that the analytic Gaussian log-prob matches a numerical estimate of the transition density for both schedules within tolerance. This is the correctness gate for M3.
- `test_minimax_h3_weight_map_on_cpu.py` — `map_weights` round-trip on a tiny random H3 config: training-names → `map_weights` → rollout-side loader accepts every key, no missing/unexpected params; qkv fuse+reorder and fc1 concat verified element-wise.
- Reuse `test_diffusion_model_base_on_cpu.py:95`'s contract (NFT rollout must not override the SDE trajectory loop) and `test_dump_generations_video_on_cpu.py:80` (no RGB/BGR inversion).

**GPU smoke** (`tests/special_e2e/run_<algo>_minimax_h3.sh`, wired into `tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh`; trigger `ci-e2e-diffusion` fires on `verl_omni/pipelines/**` etc., `select_gpu_smoke_groups.py:73-83`): tiny random H3, reduced resolution/steps, one training step, assert reward computed and a muxed mp4 written.

**PR requirements** (CLAUDE.md §1): PR body states why this is not a duplicate (H3 ≠ LTX2, different architecture/runtime), the exact test commands run and their output, and that AI assistance was used. Title `[pipelines, cfg, docs] feat: MiniMax-H3 FL2VA (t2va+fl2va) flow_grpo/nft integration`. The `map_weights` base-class addition is additive; if any reviewer deems it API-breaking, prefix `[BREAKING]`.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| diffusers H3 not yet on `main` (PR #14355 open) and the `minimax-h3` branch is live WIP (head `99ced1b` still has failing upstream tests today) | Med | Pin to PR #14355 as the upstream doc instructs (`git+https@refs/pull/14355/head`, frozen to a commit SHA for reproducibility), re-freeze when the branch stabilizes / merges; `NonDiffusersModelBase` fallback needs no diffusers PR at all |
| Dual-schedule SDE log-prob subtly wrong → M3 silently trains on a bad gradient | Med | Finite-difference CPU gate before any GPU run; M1/M2 validate plumbing without it |
| Weight-sync `map_weights` inverse drifts from upstream forward transforms | Med | Round-trip CPU test; reference the exact upstream functions; pin vllm-omni commit |
| Cost: even reduced rollout too slow for practical RL | Med | Example uses smallest viable config; N3 (no batching) is the ceiling — flag to users |
| Upstream vllm-omni changes the H3 pipeline internals (on_step signature, loop) | Low-Med | We reimplement `diffuse`/loop in the subclass and pin the vllm-omni commit in `.github/vllm_omni_pin.txt` |
| **Enabling H3 rollout requires bumping the vllm-omni pin, and that is a repo-wide migration.** The current pin (`fe478a95`) predates `minimax_h3` (added ~236 commits later, `24741961`); every commit that has `minimax_h3` post-dates PR #4922, which **removed `DiffusionOutput.custom_output`** in favor of an `output["payload"]` envelope (`1a4a042a`). No single vllm-omni commit has **both** `minimax_h3` and `custom_output`. | High (certain for GPU bring-up) | Treat the pin bump as its own PR that migrates the engine (`pipelines/request_batch.py`) **and every** diffusion rollout adapter (qwen NFT/DPO, sd3, wan22, bagel, this one) from `custom_output` → the payload envelope **together**. Keep the H3 rollout adapter on the current `custom_output` + `dataclasses.replace(..., to_cpu=True)` contract so it migrates uniformly with its siblings rather than being special-cased early. **Hard prerequisite for the M1 GPU smoke.** |
| verl spelling / license / docstring / naming gates | Low | pre-commit; new files carry the 2026 Apache header; docs carry "Last updated" |

---

## 9. Open questions

- **Q1. [resolved 2026-08-05]** diffusers now defines and exports `MiniMaxH3Transformer3DModel` (+ `AutoencoderKLMiniMaxH3`/`…Audio`, `MiniMaxH3Scheduler`, `MiniMaxH3Blocks`, `MiniMaxH3ModularPipeline`) on the `minimax-h3` branch; the →`main` PR #14355 is open. **Plan (per the upstream doc):** pin diffusers to PR #14355 — `git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head`, frozen to a commit SHA in `pyproject.toml` for reproducibility — and load the DiT through the standard diffusers path (§5.2). Switch to a released `diffusers>=<version>` when #14355 merges to `main`. `NonDiffusersModelBase` only if we must support pre-merge diffusers. *(Does not block the RFC; the branch is still WIP so the pinned SHA will need one re-freeze near merge.)*
- **Q2.** `w_v`/`w_a` defaults: is `1.0/1.0` (equal per-dim-mean weighting) the right prior, or should audio be up-weighted given its 269:1 element disadvantage is already removed by using per-modality means? Start at `1.0/1.0`, expose the knob, tune empirically.
- **Q3.** Should `map_weights` live on `DiffusionModelBase` (my proposal, benefits any future fused-weight model) or be an H3-local override at the sync call site? Base-class is cleaner and additive; confirm with a maintainer since it touches shared code.
- **Q4.** For M3, do we also want `flow_dppo`/`grpo_guard` as loss modes over the `flow_grpo` key (they are loss modes, not registry entries)? Cheap to add once the SDE is correct; defer to a follow-up.

---

## 10. Appendix — key upstream references

vllm-omni (`/group/40173/zionyfeng/vllm-omni`):
- Pipeline `MiniMaxH3Pipeline` `pipeline_minimax_h3.py:247`; `forward` `@torch.no_grad` `:1028`; single-request assert `:1030-1031`; CFG-parallel reject `:280-282`; `_resolve_task` `:384-402`; `_resolve_shape` `:403-444`; task cross-validate `:1053-1063`.
- Deterministic scheduler `scheduling_minimax_h3_euler_ancestral.py:49,72`.
- Denoise loop + `on_step` `denoise_loop.py:129-144,236-237`; DiT under `inference_mode` `:208-209`.
- DiT arch `minimax_h3_transformer.py:47-78`; qkv reorder `:139-168`; fc1 split `:962-970`; `load_weights` `:938-974`; block stack `blocks` `:595`; `_repeated_blocks`/`_layerwise_offload_blocks_attrs` `:781-782`.
- Encoder `encoder.py:898-1211`; `_map_weight_name` `:955-993`; rank gating `:921-922`.
- VAEs `vae.py:97-357` (no `load_weights`; `from_pretrained` trust-remote-code).
- Trajectory transport `DiffusionOutput` `data.py:1290-1352`; sampling params trajectory flags `inputs/data.py:301-307`.
- Custom-pipeline injection blueprint `tests/e2e/features/helpers/custom_pipeline.py`; collective_rpc weight-sync seam `diffusion_worker.py:1375-1400`, `CustomPipelineWorkerExtension.re_init_pipeline` `:763-787`.
- Recipe/usage `recipes/MiniMaxAI/MiniMax-H3.md`.

diffusers (`huggingface/diffusers`, `minimax-h3` branch as of 2026-08-05; →`main` PR #14355 open, #14371 merged into the branch):
- DiT `src/diffusers/models/transformers/transformer_minimax_h3.py` → `MiniMaxH3Transformer3DModel` (standard `ModelMixin`, `AutoModel`-loadable).
- VAEs `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py` → `AutoencoderKLMiniMaxH3`; `…_audio.py` → `AutoencoderKLMiniMaxH3Audio`.
- Modular pipeline `src/diffusers/modular_pipelines/minimax_h3/` → `MiniMaxH3ModularPipeline`, `MiniMaxH3Blocks` (**Modular only — no classic `DiffusionPipeline`**); `packing.py`, `references.py`, `packing_ref2va.py`, `denoise.py`, `encoders.py`, `decoders.py`.
- Public exports in `src/diffusers/__init__.py`: `MiniMaxH3Transformer3DModel`, `AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio`, `MiniMaxH3Scheduler`, `MiniMaxH3Blocks`, `MiniMaxH3ModularPipeline`.
- Doc/usage `docs/source/en/api/pipelines/minimax_h3.md` — install `pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head`; inference load `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="t2va"|"fl2va"|"ref2va").load_components(dtype=bf16)` (partition selected by `workflow=`; this is the diffusers *inference* path, **not** verl-omni's training load, which uses only the `transformer/` `ModelMixin`). Reference blocks `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` / `MiniMaxH3AudioReference`. Branch head SHA `99ced1b` (2026-08-05, live WIP).
- Independent third-party impl (not diffusers): DiffSynth-Studio `diffsynth/pipelines/minimax_h3_audio_video.py` (`BasePipeline`, no diffusers imports) — corroborating, not on verl-omni's path.

verl-omni (this repo):
- `DiffusionModelBase`/`DiffusionI2IModelBase` `verl_omni/pipelines/model_base.py:126-406`.
- FlowGRPO rollout reference `verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:346-640`.
- Wan video precedent `verl_omni/pipelines/wan22_dance_grpo/`.
- SDE scheduler `verl_omni/pipelines/schedulers/flow_match_sde.py:148-321` (reduction `:313`).
- Engines `verl_omni/workers/engine/fsdp/diffusers_impl.py`: PPO `:816-935`, DPO `:938-1121`, NFT `:1124-1248`; batch postprocess `:525-558`; fwd/bwd `:776-813`; per-step slices `:895-921`; **weight-sync blind prefix `:771-772`**.
- Losses `verl_omni/trainer/diffusion/diffusion_algos.py`: FlowGRPO `:268`, DPO `:590`, NFT `:781`.
- Config dataclasses `verl_omni/workers/config/diffusion/{model.py,rollout.py,actor.py}`; trainer types `main_diffusion.py:102-111`.
- Reward `verl_omni/reward_loop/reward_manager/visual.py:40-88`.
- Non-diffusers guide `docs/contributing/integrating_a_non_diffusers_model.md`; integration guide `docs/contributing/integrating_a_diffusion_model.md`; pitfalls `docs/contributing/common_pitfalls.md`.
- Joint-A/V template PR #341 (LTX2.3), **open/unmerged as of 2026-08-05** — the paths it would add (`verl_omni/pipelines/ltx2_flow_grpo/`, `verl_omni/utils/reward_score/{clap.py,imagebind.py}`) **do not exist in-tree yet**; treat it as a structural reference, not an importable dependency.

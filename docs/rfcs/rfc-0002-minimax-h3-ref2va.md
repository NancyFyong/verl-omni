# RFC-0002: MiniMax-H3 Ref2VA — Reference-conditioned Video+Audio RL

- **Status:** Draft (depends on [RFC-0001](rfc-0001-minimax-h3-fl2va.md))
- **Scope (this RFC):** the `Ref2VA` checkpoint partition — task `ref2va`, in both sub-modes:
  - **image+audio**: exactly one reference image + one reference audio → video+audio
  - **video**: one or more reference videos (and no separate audio) → video+audio
- **Companion:** [RFC-0001](rfc-0001-minimax-h3-fl2va.md) covers the `FL2VA` partition (`t2va`, `fl2va`) and establishes the shared machinery this RFC builds on. Together they cover all three H3 tasks.
- **Last updated:** 2026-08-05

---

## 0. TL;DR

Ref2VA is the second phase. It reuses **everything** RFC-0001 builds — the RFC-0001 training-side module load (diffusers-native `MiniMaxH3Transformer3DModel` preferred, `NonDiffusersModelBase` fallback), the dual-schedule reverse-SDE scheduler, the separate `audio_latents` trajectory stream, the `map_weights` weight-sync seam, the audio reward/dump, and the `("MiniMaxH3Pipeline", <algorithm>)` registry pair (both partitions share `_class_name`). The **only** genuinely new surface is **reference conditioning**:

1. A different served checkpoint dir (`$MODEL_ROOT/Ref2VA`) and a different training transformer subfolder (`transformer_ref/`).
2. Reference **inputs** the dataset must carry and the rollout adapter must feed: one image + one audio, or ≥1 videos. Reference **videos are file-path-only** in vllm-omni — a trainer holding decoded frames must write temp files.
3. A **reference-fidelity reward** dimension (does the output honor the reference identity/subject/audio?), on top of RFC-0001's A/V quality reward.
4. A materially **larger cost envelope** — two-video Ref2VA is ~784 s/generation at full quality, ~9× FL2VA.

Because the runtime is one adapter pair for both partitions, no new registry entry is added; `ref2va` is selected by loading the `Ref2VA` partition and setting `extra_args["task"]="ref2va"`. This RFC is deliberately thin: it is RFC-0001 plus a conditioning front-end and a reference reward.

---

## 1. Why a separate RFC

The user asked for the two phases split. They map exactly onto H3's two checkpoint partitions, which is architecturally the right seam:

- `FL2VA/` serves `t2va` + `fl2va`; `Ref2VA/` serves `ref2va`. One server process loads exactly one partition (`recipes/MiniMaxAI/MiniMax-H3.md`, "Known limitations").
- The two partitions declare the **same** `_class_name: "MiniMaxH3Pipeline"`, so the verl-omni registry key is identical — the adapters from RFC-0001 are reused, not duplicated. Splitting by RFC (not by registry key) keeps the shared code shared and isolates the new conditioning work.
- Sequencing FL2VA first means the dual-SDE math, joint A+V plumbing, and weight-sync map are already proven before we add reference conditioning, which is pure front-end.

If RFC-0001 is not yet landed, this RFC is blocked on it — every §5 mechanism below assumes RFC-0001's plumbing exists.

---

## 2. Background specific to Ref2VA

### 2.1 The two sub-modes

`ref2va` in vllm-omni is two distinct conditioning shapes behind one task string:

| Sub-mode | Reference input | Separate audio? | Default frames |
|----------|-----------------|-----------------|----------------|
| image+audio | exactly 1 image + exactly 1 audio | yes (the reference audio) | 124 |
| video | 1 or more reference videos | **no** (audio comes from/with the videos) | 124 |

Hard constraints (`recipes/MiniMaxAI/MiniMax-H3.md`, "Known limitations"): image+audio Ref2VA is **exactly one image + one audio**; video Ref2VA is **1+ videos and no separate audio**. The task default frame count is 124 (vs 209 for t2va/fl2va) — the branch is at `pipeline_minimax_h3.py:419`.

### 2.2 How vllm-omni ingests references

- **Images**: `_load_image` (`pipeline_minimax_h3.py:131-152`) — path / PIL / tensor / 1-element list. Reference-image geometry `_reference_image_shape` `:213-228` (short edge 2048, multiple of 32, aspect clamped to [1:4, 4:1]).
- **Audio**: `_load_audio` `:155-172` — path, `(waveform, sample_rate)` tuple, or dict.
- **Reference videos — file paths only**: `reference_video.py:142-179` `prepare_reference_videos` → `load_video_frames` `:182` opens paths from disk. The HTTP layer already writes temp files for this (`serving_video.py:49` `ReferenceVideo(data, cleanup_paths)`). **A trainer holding decoded frames in memory must write temp files and clean them up.**
- Reference encoders: `_encode_visual_condition` `:706`, `_encode_audio_condition` `:718`, `_encode_video_conditions` `:742`, `_encode_video_audio_conditions` `:789`.
- Task/shape read from `extra_args` and `multi_modal_data` at `pipeline_minimax_h3.py:1044-1063`.

The same reference-conditioning shapes are mirrored in the diffusers Modular pipeline on the `minimax-h3` branch (`modular_pipelines/minimax_h3/references.py`, `packing_ref2va.py`), exposed as `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` / `MiniMaxH3AudioReference` blocks and selected by `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="ref2va")` (upstream doc `docs/source/en/api/pipelines/minimax_h3.md`). This corroborates vllm-omni's contract — the image+audio vs video sub-modes and the exactly-1-image+1-audio cardinality match — but the rollout path stays on vllm-omni (the diffusers side is Modular-only, no classic `DiffusionPipeline`). See [RFC-0001 §10](rfc-0001-minimax-h3-fl2va.md) for the full diffusers reference group.

### 2.3 Checkpoint

Rollout: `model.path = $MODEL_ROOT/Ref2VA`. Training: shared `transformer_ref/` (diffusers-converted, same shape as `transformer/`) plus the shared `text_encoder/`, `vae/`, `audio_vae/`, `processor/`, `tokenizer/`. Set `transformer_subfolder=transformer_ref` on `DiffusionModelConfig`. Everything else about module load is identical to RFC-0001 §5.2.

---

## 3. Goals / Non-goals

**Goals**

- G1. `ref2va` rollout in both sub-modes through the RFC-0001 adapter pair (loading the `Ref2VA` partition), with reference inputs fed correctly.
- G2. Dataset support for reference columns (image, audio, video paths/bytes) and their decode → temp-file plumbing on the RL path.
- G3. A reference-fidelity reward dimension composable with RFC-0001's A/V quality reward.
- G4. Same milestone order as RFC-0001: `diffusion_nft`/`dpo` before `flow_grpo`, reusing RFC-0001's engine/loss/scheduler work unchanged.
- G5. A documented cost envelope and a smallest-viable rollout config given the ~9× cost of two-video Ref2VA.

**Non-goals**

- N1. Anything in RFC-0001's non-goals (batching, FP8, step-execution, level-2 sleep, true first+last fl2va) — inherited.
- N2. Relaxing vllm-omni's ref2va cardinality limits (1 image + 1 audio; the video mode having no separate audio). These are upstream constraints; we respect them.
- N3. New registry keys or a new adapter class. Ref2VA reuses RFC-0001's adapters; if a sub-mode needs branch logic, it is a config/request branch inside the existing adapter, not a new `(architecture, algorithm)` entry (collapsing/duplicating across a registry boundary is disallowed by `.agents/rules/code-style.md`).

---

## 4. High-level architecture (delta over RFC-0001)

```
   model.path = $MODEL_ROOT/Ref2VA            transformer_subfolder = transformer_ref
                     │
   ┌──────────────── same adapters as RFC-0001 ────────────────┐
   │  MiniMaxH3PipelineWithLogProb / MiniMaxH3DiffusionNFTPipeline │
   │     + reference front-end:                                   │
   │        - accept reference image / audio / video(s) from      │
   │          the dataset (in-memory tensors + paths)             │
   │        - write temp files for reference videos               │
   │        - call _encode_{visual,audio,video}_condition          │
   │        - pass task="ref2va" via extra_args                    │
   └──────────────────────────────────────────────────────────────┘
                     │
   Dataset: reference columns (image bytes / audio / video paths)
   Reward:  RFC-0001 A/V quality reward  +  reference-fidelity reward
```

Files touched (small delta):

```
verl_omni/pipelines/minimax_h3_<algo>/vllm_omni_rollout_adapter.py   # + reference-conditioning path
verl_omni/pipelines/minimax_h3_<algo>/common.py                      # + temp-file / ref-encode helpers
verl_omni/utils/dataset/rl_dataset.py (or a Ref2VA dataset)          # reference columns
verl_omni/utils/reward_score/<ref_reward>.py                         # reference-fidelity scorer
verl_omni/workers/config/diffusion/rollout.py                        # + ref2va fields (default frames 124, sub-mode)
verl_omni/trainer/config/diffusion/{rollout,model}/*.yaml            # mirror the new fields (both files)
examples/<algo>_trainer/minimax_h3_ref2va/{README.md,prepare_data.py,run_*.sh}
tests/pipelines/test_minimax_h3_ref2va_on_cpu.py
tests/special_e2e/run_<algo>_minimax_h3_ref2va.sh
```

No engine, scheduler, or loss changes beyond RFC-0001.

---

## 5. Detailed design

### 5.1 Reference inputs on the RL path

The rollout adapter (already overriding `encode_prompt`/`forward` from RFC-0001) gains a reference front-end, gated on `task == "ref2va"`:

- **image+audio sub-mode**: pull one reference image and one reference audio from the sample; hand the image to `_load_image` and the audio to `_load_audio`; both are in-memory-friendly (tensor/PIL image, `(waveform, sr)` tuple). Enforce the exact-1+exact-1 cardinality with a clear error mirroring the upstream limit.
- **video sub-mode**: reference videos must be **file paths on disk**. The dataset stores paths (preferred) or bytes; if bytes, the adapter writes temp files via a `common.py` helper (`with_temp_reference_videos(...)` → list of paths + cleanup) modeled on `serving_video.py:49`'s `ReferenceVideo(data, cleanup_paths)`, and deletes them after the request. No separate audio is passed in this sub-mode (upstream constraint).
- Task is set via `extra_args={"task":"ref2va", ...}`; shape defaults to 124 frames unless overridden. Reference-image geometry follows `_reference_image_shape` (short edge 2048, multiple of 32, aspect [1:4,4:1]).

Because rollout is "one generation per request" (inherited N1), a group of `n` reference-conditioned rollouts issues `n` separate requests, each re-supplying the same reference inputs. Temp-file writes are therefore per-request; the helper must be cheap and clean up deterministically.

### 5.2 Dataset

Reference conditioning needs columns the FL2VA path does not:

- image+audio: a reference-image column (`[{"bytes": png_bytes}]`, decoded exactly like the existing `images` path — `verl/utils/dataset/rl_dataset.py:299-384`, with a matching `<image>` placeholder and equal placeholder count in prompt and negative prompt) and a reference-audio column (waveform bytes or path).
- video: a reference-video column carrying **paths** (a decoded-frames-to-temp-file fallback lives in the adapter, §5.1, not the dataset).

Two viable wirings:
- **(a)** extend the existing `RLDataset` override (`verl_omni/utils/dataset/rl_dataset.py:54-62`) to surface the extra columns in `raw_prompt`/non-tensor batch (they survive collate as object arrays, `collate_fn` at `verl/utils/dataset/rl_dataset.py:41-70`);
- **(b)** a dedicated `Ref2VADataset` selected by `data.custom_cls` (the offline-DPO precedent, `OfflineDPODataset` wired via `data.custom_cls.path/name/collate_fn`).

**Decision: (a) for image+audio** (reuses the proven image-decode path), **paths-in-column for video** (no decode needed). A dedicated dataset only if (a) proves too tangled — deferred, not chosen up front (don't invent a shared abstraction for a single caller, `.agents/rules/code-style.md`).

Note the `images`/`video`/`audio` keys are **popped** upstream (`__getitem__` :391-393) and re-extracted in the agent loop via `process_vision_info(raw_prompt)` (`single_turn_agent_loop.py:48`) — reference columns must ride the same `raw_prompt` channel to reach the rollout adapter.

### 5.3 Reference-fidelity reward

RFC-0001's A/V quality reward answers "is this a good video+audio for the prompt?". Ref2VA additionally needs "does the output honor the reference?":

- image+audio: subject/identity consistency between the reference image and the generated video (a CLIP/DINO-style image-video similarity), and audio consistency between the reference audio and the generated audio (a CLAP-style audio-audio similarity). PR #341's `clap.py` is the intended template, but that PR is **open/unmerged as of 2026-08-05** — the file does not exist in-tree yet, so this scorer is from-scratch.
- video: consistency between the reference video(s) and the generated video (ImageBind-style joint embedding). Likewise PR #341's `imagebind.py` is unmerged and not yet in-tree.

Both consume **in-memory tensors** (the reference inputs are already in memory on the adapter side; the outputs come from H3's `np` post-process) — never file paths, per the `VisualRewardManager` contract (`visual.py:70-88`). The reference reward is a **second scorer** composed with the quality reward via the existing multi-reward manager (`MultiVisualRewardManager`, `reward_manager/multi.py:49`); weighting between quality and fidelity is a reward-config knob. `assemble_rm_scores` still returns `(bsz, 1)`.

### 5.4 Config surface (delta)

Add to `DiffusionPipelineConfig` (mirrored in both yamls, §5.7 of RFC-0001):

| Field | Default | Meaning |
|-------|---------|---------|
| `ref2va_submode` | `"image_audio"` | `"image_audio"` or `"video"` |
| `num_frames` (reuse) | 124 for ref2va | task default is 124, not 209 |

The `task`/`duration`/`fps`/`flow_shift`/`audio_flow_shift`/audio-reward fields from RFC-0001 are reused unchanged. All additive with defaults → no `[BREAKING]`. Regenerate the `_generated_*.yaml` via `scripts/generate_trainer_config.sh`.

### 5.5 Cost envelope (the reason Ref2VA is phase 2)

- Two-video Ref2VA at full quality: **~784 s/generation** (362 frames, 4×B300) — ~9× FL2VA's ~87 s. RL is infeasible at that size; the example must use the smallest viable reference config (fewest reference frames, reduced output resolution/duration/steps).
- image+audio Ref2VA is cheaper than two-video but still above FL2VA (default 124 frames + reference encode).
- The rollout config for RL should target the low end: reduced output resolution/duration, `num_inference_steps≈10`, single short reference. Exact numbers pinned in the example script.

### 5.6 What is explicitly reused from RFC-0001 (no change)

- RFC-0001 training-side module load (diffusers-native `MiniMaxH3Transformer3DModel` preferred, `NonDiffusersModelBase` fallback), only `transformer_subfolder=transformer_ref`.
- Dual-schedule reverse-SDE scheduler (`flow_match_dual_sde.py`).
- Separate `audio_latents` trajectory stream + engine slice sites.
- `map_weights` weight-sync seam (the Ref2VA DiT has the same fused/reordered layout as FL2VA — the same inverse transforms apply; verify on the `Ref2VA` weights in the round-trip test).
- Audio dump + A/V muxing in `ray_diffusion_trainer._dump_generations`.
- Losses, trainer types, milestone order.

---

## 6. Milestones

| # | Deliverable | Gate |
|---|-------------|------|
| R0 | RFC-0001 landed and green | prerequisite |
| R1 | `ref2va` image+audio rollout + reference-fidelity reward, `diffusion_nft` | GPU smoke `run_diffusionnft_minimax_h3_ref2va.sh`; reward computed on a tiny run |
| R2 | `ref2va` video sub-mode (temp-file plumbing, 1+ videos) | CPU test for temp-file lifecycle; GPU smoke |
| R3 | `flow_grpo` for `ref2va` (reuses RFC-0001's dual-SDE unchanged) | GPU smoke; reward increases |
| R4 | Docs + example READMEs; this RFC marked Accepted | — |

Image+audio before video because video adds temp-file lifecycle risk and the highest cost.

---

## 7. Test plan (delta over RFC-0001)

**CPU**

- `test_minimax_h3_ref2va_on_cpu.py` — adapter accepts reference image+audio and reference-video paths (mocked encoders); cardinality errors raised (2 images, or audio in video sub-mode) with clear messages; `task="ref2va"` and 124-frame default resolved correctly; the temp-file helper writes and **cleans up** files (assert no leak) on both success and exception paths.
- Reference-reward scorer CPU test on tiny tensors (shape + range).

**GPU smoke** — `tests/special_e2e/run_<algo>_minimax_h3_ref2va.sh` (both sub-modes), wired into `run_gpu_smoke_diffusion_e2e.sh`; tiny random H3 `Ref2VA`, smallest reference config, one training step, assert reward + muxed mp4.

**PR requirements** (CLAUDE.md §1) — same as RFC-0001: not-a-duplicate rationale (Ref2VA reference conditioning, distinct from FL2VA and from LTX2), test commands + output, AI-assistance statement. Title `[pipelines, cfg, docs] feat: MiniMax-H3 Ref2VA (ref2va) reference-conditioned integration`.

---

## 8. Risks (delta)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Cost makes video-mode RL impractical | High | image+audio first; document the envelope; smallest viable config; N1 (no batching) is the ceiling |
| Reference temp-file leaks under exception/abort | Med | context-manager helper with guaranteed cleanup; CPU test on the exception path |
| Reference-fidelity reward hard to define well (identity/subject) | Med | start with CLIP/DINO + CLAP similarities (PR #341 scorers, **unmerged as of 2026-08-05** — build from scratch until it lands); treat weighting as a tunable |
| Dataset reference columns don't survive collate | Low | ride the proven `raw_prompt` object-array channel; assert in the CPU test |
| Ref2VA weights differ from FL2VA in a way that breaks `map_weights` | Low | round-trip test against the actual `Ref2VA`/`transformer_ref` weights |

---

## 9. Open questions

- **Q1.** Dataset wiring (a) vs (b) — confirmed (a) for image+audio, paths for video; revisit only if (a) is unwieldy.
- **Q2.** Reference-fidelity reward composition: separate scorer + `MultiVisualRewardManager` (proposed) vs a single joint scorer emitting a combined score. Separate is more modular and would reuse PR #341's scorers **once it merges** (unmerged today); confirm.
- **Q3.** For the video sub-mode, is one reference video the practical RL target (cost), with multi-video left as an inference-time capability only? Likely yes — flag in the example.
- **Q4.** Inherited from RFC-0001 Q1/Q3 (diffusers load path, `map_weights` home) — resolved there.

---

## 10. Appendix — Ref2VA-specific upstream references

vllm-omni (`/group/40173/zionyfeng/vllm-omni`):
- Reference-video ingest (paths only) `reference_video.py:142-179`, `load_video_frames` `:182`.
- Reference encoders `pipeline_minimax_h3.py:706,718,742,789`; task-branch points incl. ref2va frame default `:419`; `_reference_image_shape` `:213-228`; image/audio loaders `:131-172`.
- HTTP temp-file precedent `serving_video.py:49` (`ReferenceVideo`), `:42` (`ReferenceImage`), `:57` (`ReferenceAudio`).
- `Ref2VA/model_index.json` — `_class_name: "MiniMaxH3Pipeline"`, `partition:"ref2va"`, `tasks:["ref2va"]`.
- Recipe `recipes/MiniMaxAI/MiniMax-H3.md` (cardinality limits; ~784 s two-video latency).

diffusers (`huggingface/diffusers`, `minimax-h3` branch; →`main` PR #14355 open as of 2026-08-05):
- ref2va conditioning/packing `src/diffusers/modular_pipelines/minimax_h3/references.py`, `packing_ref2va.py`; reference blocks `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` / `MiniMaxH3AudioReference`; `workflow="ref2va"` selector and install command in doc `docs/source/en/api/pipelines/minimax_h3.md` (Modular pipeline only — corroborates vllm-omni's shapes; not the rollout path). Full diffusers reference group in [RFC-0001 §10](rfc-0001-minimax-h3-fl2va.md).

verl-omni (this repo):
- Reused machinery — see [RFC-0001 §10](rfc-0001-minimax-h3-fl2va.md).
- Dataset image-decode path `verl/utils/dataset/rl_dataset.py:299-384`; verl-omni override `verl_omni/utils/dataset/rl_dataset.py:54-62`; collate `verl/utils/dataset/rl_dataset.py:41-70`; agent-loop re-extract `single_turn_agent_loop.py:48`; offline-DPO custom-dataset precedent `verl_omni/utils/dataset/offline_dpo_dataset.py:170`.
- Multi-reward manager `verl_omni/reward_loop/reward_manager/multi.py:49`; reward contract `visual.py:40-88`.
- Reference scorers to extend: PR #341 `verl_omni/utils/reward_score/{clap.py,imagebind.py}` — **open/unmerged as of 2026-08-05; these paths do not exist in-tree yet**.

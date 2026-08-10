# [RFC] Unified multi-input / multi-output interface for diffusion rollout

> Structured against `.github/ISSUE_TEMPLATE/feature-request.yml` — the repo has no dedicated
> RFC template — following the two `nemo_automodel` design drafts that set the precedent
> (`.agents/RFC_ISSUE.md`, `.agents/RFC_ISSUE_DIFFUSION.md`; they live on their own branch).
> §0 TL;DR is the template's *Feature request* field, §2 is *Motivation*, and a *Your
> contribution* answer would be scoped from §6 Milestones. §1 and §3-§10 are the technical
> body.

- **Status:** Draft
- **Scope:** the request and output objects crossing the agent-loop → rollout-server →
  rollout-adapter → reward → training-adapter path for **diffusion** pipelines. Multimodal
  *conditioning inputs* (text + image + video + audio) and multimodal *generated outputs*
  (image, video, audio) are both in scope.
- **Not in scope:** the autoregressive omni track, the `TokenOutput` path, and any change to
  the pinned vllm-omni protocol (§3).
- **Applies to:** the whole diffusion tree, not one model integration — 10 pipeline packages, 10
  registered `(architecture, algorithm)` keys, 9 rollout adapters, all re-deciding the same
  questions independently today.
- **Last updated:** 2026-08-10
- **Verified against:** verl-omni `8c0f5fa`, vllm-omni `0.24.1.dev26+gfe478a95a`. Every
  `file:line` below was read at those revisions.

---

## 0. TL;DR

**text + image conditioning already works** (§1.4). What is missing is not the capability but
the contract, in three places:

1. **The generated modality is recorded nowhere in this repo.** Seven sites sniff tensor rank —
   five to decide *which modality*, two to decide *is this batched* — and because the two
   questions look identical in the code, they collide: `ndim == 4` means *video* in `utils/tracking.py:156` and *a
   batch of images* in `http_scorer_client.py:40`, which keeps frame 0 and silently discards
   the rest (§1.3).
2. **The layer this repo owns is the only one on the path with no request type** — 12 flat
   kwargs, nine describing the input, re-listed verbatim by two intermediaries (§1.1-1.2).
3. **Everything else rides in a `dict[str, Any]`** merged with upstream's by `setdefault`, so an
   adapter key silently wins any collision — `audio_sample_rate` already has two producers and no
   declared owner — and one past collision plus one name reserved by the MFU FLOPs
   counter are policed by a hand-written runtime guard instead of by a type
   (`pipelines/model_base.py:355-371`) (§1.3).

Proposal: one additive, CPU-importable module `verl_omni/pipelines/io.py` holding `MediaRef` +
`PromptBundle` + `MediaRequest` on the way in and `MediaOut` + `MediaOutput` on the way out.
Modality becomes **declared** data, conditioning inputs become a **list** instead of N optional
kwargs, and the RL trajectory gets a slot separate from the pipeline-private escape hatch.
`MediaRequest` is the existing `ImageGenerationRequest` (`pipelines/utils.py:45`) generalised
past images — the follow-up its own call site already asked for (§1.4).

Five milestones, cheapest-risk first. **M1+M2 carry most of the value and are revertible.**

---

## 1. Background

### 1.1 The request crosses three representations

| # | Layer | Type | Shape of the abstraction |
| --- | --- | --- | --- |
| 1 | vllm-omni HTTP protocol | `ImageGenerationRequest` (`protocol/images.py:33`), `VideoGenerationRequest` (`videos.py:97`), `OpenAICreateAudioGenerateRequest` (`audio.py:321`) | three per-modality Pydantic models, **no shared base** |
| 2 | vllm-omni engine | `OmniDiffusionRequest` (`diffusion/request.py:14`) | modality-agnostic, 4 fields, unified **by union** |
| 3 | **verl-omni rollout** | `vLLMOmniAsyncServer.generate` (`vllm_omni_async_server.py:330`) | **no request object at all** — 12 flat kwargs |

Layers 1 and 2 are each internally coherent. **Layer 3 — the one this repo owns — is the one
with no type.**

Layer 1's split exists to satisfy the OpenAI Images and Videos API shapes at an HTTP boundary,
which verl-omni does not have (§3 N2). The one idea worth taking from it is the **tagged-union
reference** (`ImageReference = UrlImageReference | FileImageReference`, `videos.py:63-94`): each
conditioning input carries its own kind, so a consumer never guesses from the payload's Python
type. That is what §5.2's `MediaRef` is.

Layer 2 is the load-bearing precedent — **vllm-omni keeps the per-modality split at the HTTP
edge and exactly one modality-agnostic request inside**:

```python
@dataclass
class OmniDiffusionRequest:            # vllm_omni/diffusion/request.py:14
    prompt: OmniPromptType
    sampling_params: OmniDiffusionSamplingParams   # 80 fields, every modality's knobs at once
    request_id: str
    kv_sender_info: dict | None = None
```

It unifies by **union**, not by structure: `OmniDiffusionSamplingParams` (`inputs/data.py:178`,
80 fields counted with `dataclasses.fields`) carries every modality's knobs simultaneously,
reconciled by defaulting. verl-omni sits entirely inside that edge, so it should have the inside
shape. Today it has neither.

One nearby type is deliberately **not** counted as a fourth representation:
`DiffusionPipelineConfig` (`config/diffusion/rollout.py:58-74`) is a Hydra-side declaration of
nine sampling knobs (`height`, `width`, `num_inference_steps`, `output_type`, `true_cfg_scale`,
`max_sequence_length`, `guidance_scale`, `num_frames`, `frame_rate`) that feed
`sampling_params` — it never carries a prompt or a conditioning input, so it is a static source
*for* the request, not a copy *of* it. Unifying it is out of scope (§3 N6). It needs one note
though: `output_type` looks like a modality declaration and is not one. It defaults to `"image"`
for every pipeline including video ones and is a diffusers-style *format* selector (`"pil"` /
`"pt"` / `"np"` / `"latent"`), which is why LTX-2 rewrites it
(`ltx2_flow_grpo/common.py:51-53`).

### 1.2 Input side: flat kwargs and a relay chain

`generate` (`vllm_omni_async_server.py:330-344`) takes twelve keyword arguments, nine describing
the input: `prompt_ids`, `negative_prompt_ids`, `prompt_mask`, `extra_prompt_ids`,
`negative_extra_prompt_ids`, `image_data`, `video_data`, `audio_data`, `mm_processor_kwargs`.
Consequences present in the tree:

- **Every intermediary re-lists the arguments.** `DiffusionRetryLLMServer.generate`
  (`diffusion_llm_server.py:55-80`) names six verbatim — none of which it uses, it only forwards
  them — purely to wrap the call in a retry loop; `DiffusionSingleTurnAgentLoop.run`
  (`single_turn_agent_loop.py:104-113`) restates seven. Adding a modality means editing two of
  those signatures: the retry server also takes `**kwargs` and forwards it, so a new kwarg
  *reaches* `generate` through it — but the six it does name are pure transcription that has to
  be kept in sync by hand.
- **The only unification already present** is `_build_multi_modal_data` (`:367-380`), which
  folds three optional lists into `{"image":…, "video":…, "audio":…}` — i.e. it converts
  positional kwargs back into the tagged list they should have been.
- **Modality is hardcoded on the way in too:** `_preprocess_input` sets
  `custom_prompt["modalities"] = ["image"]` (`:495-497`) whenever the pipeline has more than one
  stage (`len(default_params_list) > 1`). It is a stage-routing tag for the orchestrator rather
  than a claim about the output, and multi-stage video pipelines get `["image"]` too — a second
  place where "which modality" is answered by a literal instead of by the pipeline.
- **Unknown sampling keys are silently accepted.** `_preprocess_input:510-517` partitions
  `sampling_params` by `hasattr(OmniDiffusionSamplingParams, k)` and files anything unrecognised
  under `extra_args`, so a typo in a knob name is a no-op that surfaces as a quality regression.

### 1.3 Output side: the modality is inferred from tensor rank

```python
class DiffusionOutput(BaseModel):      # verl_omni/workers/rollout/replica.py:20-32
    diffusion_output: Any
    """generated image tensor (CHW format) / video tensor (TCHW format)"""
    log_probs: Optional[Any] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
    extra_fields: dict[str, Any] = {}
```

No modality field, so **seven** sites sniff tensor rank instead. Five of them are asking *"which
modality is this?"*; the last two are asking *"is this batched?"* — a legitimate use of rank that
is indistinguishable from the first at the call site, which is precisely how the two get confused.
The last column is the M2 edit (§6), listed here so the audit and the fix read as one table:

| Site | Today's test | Meaning assigned | Becomes |
| --- | --- | --- | --- |
| `trainer/diffusion/ray_diffusion_trainer.py:309` | `outputs.ndim == 5` | video → mp4, else jpg | `out.modality == "video"`; audio track from `MediaOutput.get("audio")` instead of the asymmetric `batch.batch.get("audio", …)` (`:391-393`) |
| `utils/tracking.py:156` | `out.ndim == 4` | **video** → `wandb.Video` | same predicate; `fps` / `sample_rate` from the `MediaOut` instead of positional tuple slots (`:154-155`) |
| `utils/reward_score/http_scorer_client.py:40` | `image.ndim == 4` | **a batch** → keep `[0]` | batching handled by the caller; the scorer takes one `MediaOut` and refuses a non-image modality explicitly |
| `utils/reward_score/hpsv3_reward.py:388-410` | 3/4/5-way + `shape[-1] in (1,3)` | image / video / batched video | modality from the field; the channels-last guess stays (§9 q3) |
| `utils/reward_score/genrm_ocr.py:141-162` | same | same | same |
| `utils/reward_score/unified_reward.py:70-76` | 3/4 only, `raise` otherwise | image / video | same; its `raise` becomes reachable only for a genuinely unsupported modality |
| `pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py:381` | `video.ndim == 5` | leading batch axis to squeeze | **unchanged** — this one is about batching, not modality |

`ndim == 4` therefore carries opposite meanings in two files. Nothing prevents a video pipeline
from being pointed at the HTTP scorer, and if that happens the reward is computed on frame 0 of
each clip **without any error**. Both files are locally correct; the bug is that "is this a
video?" has no single answer to consult — and the rank convention is not even stable along the
path, since the dump site sees batched 5-D where the logging site sees per-sample 4-D. That the
last row is *correctly* using rank is only discoverable by auditing all seven together, itself
an argument for doing M2 in one pass.

Upstream admits the union honestly — `DiffusionOutput.output` is typed `torch.Tensor |
tuple[Any, ...] | dict[str, Any] | None` (`diffusion/data.py:1202`) and LTX-2 uses the tuple
arm, `output.output = (video[0], audio)` (`ltx2_flow_grpo/vllm_omni_rollout_adapter.py:383`).
verl-omni flattens it back to `Any`.

Everything that is *not* the primary tensor rides in `extra_fields`, assembled by
`_process_output` (`vllm_omni_async_server.py:610-635`) from each adapter's `custom_output` unioned with upstream's
`multimodal_output`. Nine rollout adapters exist and eight emit `custom_output`
(`qwen_image_mix_grpo` emits none); their key sets overlap without
agreeing, and the disagreement does not follow the algorithm boundary: `qwen_image_flow_grpo` and
`wan22_dance_grpo` emit `all_latents` / `all_log_probs` / `all_timesteps` plus four prompt-embed
keys, `sd3_flow_grpo` adds the two pooled variants, but `bagel_flow_grpo` — same algorithm —
emits the three `all_*` keys and **no** prompt-embed keys at all. NFT and DPO substitute
`latents_clean` (+ `train_timesteps` for NFT), and `qwen_image_edit` and `ltx2` each add their own
(`img_shapes`, `condition_image_latents`; `audio_prompt_embeds`, `audio_sample_rate`). Upstream's
`_build_multimodal_output` (`diffusion/output_formatter.py:158-171`) merges in `audio`,
`audio_sample_rate`, `fps` and `actions` on top. Four consequences, all present in the tree:

1. **A hand-written guard exists to catch a key collision.**
   `DiffusionI2IModelBase.inject_condition` raises if the `model_inputs` it is about to populate
   already contains `image_latents` — a name **reserved by the MFU FLOPs counter** for the
   denoised latent — with the message *"the rollout adapter likely output 'image_latents' instead
   of 'condition_image_latents'"* (`pipelines/model_base.py:355-371`). Two separate undeclared
   contracts collide in one dict: the rollout→training key set, and the counter's reserved name.
   The guard is the shape of the problem — both are policed at runtime because neither is
   declared. Note it inspects the **training-side** `model_inputs`, so §5.3's output types do not
   remove it (§6 M4).
2. **Batch slicing is decided by shape coincidence.** `_slice_batch_value`
   (`request_batch.py:197-207`) slices a tensor iff `value.shape[0] == req.num_reqs *
   num_outputs_per_prompt`. Whether a key is per-sample or shared is inferred from a number, not
   known. Unbatching is likewise `isinstance` plus `[0]` (`_maybe_unbatch`, `vllm_omni_async_server.py:619-628`).
3. **The container a key lands in is not statically knowable.** Tensor-valued extra fields go to
   the `TensorDict` and everything else to `non_tensor_batch` by `isinstance`
   (`diffusion_agent_loop.py:348-352`, `:383-388`), so the trainer must look in both — and does
   so in *inconsistent order* for two related keys, reading `batch.batch` first for `audio` and
   `non_tensor_batch` first for `audio_sample_rate` (`ray_diffusion_trainer.py:391-393`).
4. **Merge order silently picks the winner.** The union is
   `extra_fields.setdefault(key, ...)` (`vllm_omni_async_server.py:630-634`), which folds upstream's `multimodal_output` in
   *under* the adapter's `custom_output` — so on any collision the adapter wins and upstream's
   value is discarded without a warning. That collision is not hypothetical: `audio_sample_rate`
   is emitted both by `ltx2_flow_grpo/vllm_omni_rollout_adapter.py:397` (the vocoder's rate) and
   by upstream's `_build_multimodal_output`, and nothing in the tree declares which is
   authoritative. The other two keys the trainer reads there, `audio` and `fps`, have **no**
   in-repo rollout producer at all, so a pin bump renaming either fails at **training** time with
   a `KeyError`, or worse yields `None` and silently drops the audio track from every dumped mp4.

### 1.4 What is already partially unified — and where it stopped

**text + image conditioning already works end to end.** This RFC is not introducing multimodal
input; it is generalising a mechanism that exists, is exercised by an in-repo example
(`examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py:63-71` carries text and image in
separate dataset columns), and whose author explicitly deferred the generalisation:

```python
@dataclass
class ImageGenerationRequest:           # verl_omni/pipelines/utils.py:45
    prompt: Any
    images: list[Any] = field(default_factory=list)
    """Condition images: empty for t2i, single-element for image editing, multi-element
    for multi-image conditioning."""
    negative_prompt: Any | None = None
    metadata: Mapping[str, Any] | None = None
```

This is the right idea, and it is why this RFC proposes a structure rather than a per-modality
split. How far it got:

1. **The generalisation was deferred in the source.** Its one call site carries a `NOTE`: *"only
   this image-edit pipeline consumes `ImageGenerationRequest` for now; migrating the existing
   T2I pipelines onto it is left to a follow-up PR"*
   (`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`). **This RFC is that
   follow-up, widened from images to all four modalities.** It is image-only today — no `videos`
   / `audios` field, no modality tag, no `role`, so it cannot express "video conditioned on a
   keyframe plus a reference clip" (M-3).
2. **It searches five candidate locations for one value** — `images`, `image`,
   `multi_modal_data.image`, `extra_args.multi_modal_data.image`,
   `additional_information.condition_images` (`utils.py:79-83`). **Three of the five have no
   producer** anywhere on the diffusion rollout path: nothing writes `images` or `image` into a
   `custom_prompt`, and `additional_information` has consumers (`request_batch.py:75`) and CPU-test
   fixtures (`tests/pipelines/test_image_edit_interface_on_cpu.py:55`) but no writer. Of the two
   that are live, one exists only because `_preprocess_input:506-508` writes the same dict into
   two places. This is not defensive coding; it is the cost of having no declared contract.
3. **One of the ten diffusion pipelines bypasses it and hand-rolls the same lookup** —
   `wan22_dance_grpo:373-381` reads `multi_modal_data["image"]` directly. `bagel_flow_grpo` is a
   second shape of the same problem rather than a third spelling: it does not hand-roll the
   lookup, it *copies* `extra_args.multi_modal_data` up into `custom_prompt` first (`:285-288`) —
   a workaround that exists only because `_preprocess_input:506-508` wrote the dict into two
   places and the two consumers disagree on which one to read. (`ImageGenerationRequest` also
   collides by name with upstream's unrelated HTTP model, `protocol/images.py:33`; both are
   importable in the same process.)

Video and audio are plumbed to different depths, which matters because "the parameter exists" is
not "the modality works":

| Modality in | Reaches the engine prompt? | Any consumer? | Status |
| --- | --- | --- | --- |
| text | yes | all 10 pipelines | **live** |
| image | yes, written twice (`:506-508`) | `qwen_image_edit_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo` | **live**, two lookup styles + one copy-up workaround |
| video | yes — `multi_modal_data["video"]` (`:377`) | **none** — `:377` is the only occurrence of the key under `verl_omni/` | **write-only** |
| audio | only if a caller passes `audio_data` | **none** — no caller on this branch supplies it | **unreachable** |

A contributor reading `generate`'s signature sees four conditioning modalities and gets one
working, one silently discarded, and one unreachable. Nothing in the type system says which is
which.

### 1.5 What upstream already tags and this repo discards

- `OmniRequestOutput.final_output_type: str` (`outputs.py:87`). The docstring at `:74` lists
  `"text" | "image" | "audio" | "latents"`, but that is stale — upstream also emits `"video"` and
  `"videos"` (`stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29`) and
  `"actions"` (`models/gr00t/pipeline.py:22`), and verl-omni's own AR track sets `"codec"`
  (`qwen3_omni/omni_rollout_adapter.py:91`). **Nothing in verl-omni reads the field off an
  output** — the only hits are stage-config literals (`qwen3_omni/omni_rollout_adapter.py:90-95`
  plus three shipped configs). §5.4 explains why it stays that way.
- `OmniRequestOutput.images: list[Image.Image]` (`:91`) is declared as PIL images and in practice
  carries video tensors and `(video, audio)` tuples. verl-omni inherits the ambiguity by reading
  `final_res.images[0]` into an `Any`.
- `multimodal_output` carries `actions` — a fifth modality already present in the pinned engine.
  Today it would land in `extra_fields` and be dropped without comment.

---

## 2. Motivation

The tree holds **10 diffusion pipeline packages** (11 counting the AR-track `qwen3_omni`, out of
scope by N4), **10 registered `(architecture, algorithm)` keys** and **9 rollout adapters**. Each
one independently re-decides the same three questions — how do I pass conditioning, how do I say
what I emitted, where do I put the per-step trajectory — because no type answers them. The result
is **3 representations of one request** — and the one this repo owns is the untyped one (§1.1-1.2)
— **7 sites sniffing tensor rank, 5 of them to answer "which modality"** (§1.3) and **5 candidate
locations for one conditioning image, 3 of them with no producer** (§1.4). Nothing here
is a bug in any single file; every entry below is what that missing type costs, and the cost is
paid once per pipeline, forever.

Five failure modes follow directly:

- **M-1: wrong rewards, no error.** The `ndim == 4` collision (§1.3). A video pipeline
  configured with `http_scorer` scores frame 0 and reports a plausible number.
- **M-2: pin bumps break at training time, not request time.** The contract is string keys spread
  over eight adapters plus upstream's `multimodal_output`, unioned by `setdefault` so the adapter
  silently wins any collision (§1.3). Already live: `audio_sample_rate` has two producers —
  `ltx2_flow_grpo` and upstream's formatter — and merge order alone decides which one the trainer
  sees. `audio` and `fps` have no in-repo rollout producer at all, so an upstream rename surfaces
  as a `KeyError` minutes into a run, or as a silently muted mp4.
- **M-3: adding a modality costs three signature edits.** A second conditioning image with a
  different role (keyframe vs identity reference) has nowhere to go but a new `*_data` kwarg
  threaded through three signatures (§1.2), or an untyped `extra_args` key — the route
  `bagel_flow_grpo:278-282` already takes to smuggle a `negative_prompt` through
  `sampling_params`, where a conditioning input does not belong.
- **M-4: a model emitting two modalities has no representation.** LTX-2 already does, and had to
  smuggle audio through the tuple arm of upstream's union plus an `audio_sample_rate` key
  (§1.3). Every future joint A/V pipeline repeats that choice independently.
- **M-5: one conditioning image is looked for in five places** (§1.4), while
  `multi_modal_data["video"]` is write-only and `audio_data` is unreachable. Each is
  individually harmless; collectively the input contract is whatever the last adapter happened
  to check.

None of these is specific to a model or an algorithm — they are properties of the path every
pipeline shares, so they recur on the next integration whoever writes it.

---

## 3. Goals / Non-goals

**Goals**

- **G1.** One request type covering text + image + video + audio conditioning, such that adding
  a conditioning modality or a second input with a different role requires **no signature
  change** on the relay path — by generalising the existing image-only
  `ImageGenerationRequest`, not adding a second partial answer beside it.
- **G1b.** One place to read a conditioning input, replacing the five candidates, the one
  hand-rolled lookup and the one copy-up workaround (§1.4); and resolve the two dead paths (write-only video, unreachable audio) in
  one direction or the other.
- **G2.** One output type in which the generated modality is **declared data**, and one request
  may declare more than one output medium.
- **G3.** **No site infers the generated modality where a declaration exists.** Not "zero rank
  checks": §1.3's last column keeps rank at the two sites where rank is genuinely the question —
  `ltx2_flow_grpo:381` (squeeze a leading batch axis) and `http_scorer_client.py:40` (is this a
  batch?) — and keeps `tracking.py:156`'s predicate for the `fps` shape. The five sites that today
  answer *"which modality is this?"* by rank read `MediaOut.modality` instead, and the two that
  answer *"is this batched?"* stop being confusable with them.
- **G4.** A first-class slot for the RL trajectory, so a rollout↔training key mismatch is a
  construction-time error rather than a training-time `KeyError`.
- **G5.** Fully additive: every milestone landable and revertible alone, and **no milestone may
  require editing all 10 diffusion pipeline packages at once.**
- **G6.** CPU-testable — importable without diffusers, vllm-omni, or a GPU.

**Non-goals**

- **N1.** Changing the vllm-omni protocol or engine types. Pinned upstream; this RFC adapts at
  the boundary and never forks them.
- **N2.** Mirroring the `ImageGenerationRequest` / `VideoGenerationRequest` HTTP split — it serves
  an API boundary this repo does not have, and cannot express "video conditioned on an image and
  an audio clip" without growing the union anyway (§1.1).
- **N3.** Typing `sampling_params`. A parallel `MediaSamplingParams` would be a fifth
  representation silently tracking an 80-field upstream dataclass across pin bumps — exactly
  M-2. Type only what this repo owns. (The narrower win, independent of this RFC: make the
  `hasattr` partition *warn* on unknown keys.)
- **N4.** Touching the autoregressive omni track or `TokenOutput`. `generate` serves both via
  `self._ar_mode`; this RFC changes only the diffusion arm.
- **N5.** Redesigning `compute_score_*` signatures. Making the scorers read a declared modality
  is in scope (G3); their signatures are not.
- **N6.** Folding `DiffusionPipelineConfig`'s nine sampling knobs into the request type (§1.1).
  Config lives on a different lifecycle (Hydra + `_generated_*.yaml`).
- **N7.** Renaming the two existing `DiffusionOutput` classes (§5.5).

---

## 4. High-level architecture

Today — the same hops, no type at either end:

```
dataset row → agent loop (7 kwargs) → retry server (6 + **kwargs) → generate (12 kwargs)
  ▼  _preprocess_input → OmniCustomPrompt + OmniDiffusionSamplingParams(80)
vllm-omni engine → rollout adapter → DiffusionOutput(output=Tensor|tuple|dict)
  ▼  images[0] → Any;  custom_output ∪ multimodal_output → extra_fields
DiffusionOutput(diffusion_output: Any, extra_fields: dict[str, Any])
  ├──► reward scorers / dump / logging  ── ndim 3/4/5 guess ×6 sites (7th is ltx2's own squeeze)
  └──► training adapter                 ── string keys, policed by a runtime guard
```

Proposed:

```
dataset row
  ▼
DiffusionSingleTurnAgentLoop.run
  │  MediaRequest(prompt=PromptBundle, conditions=[MediaRef, …], sampling_params={…})
  ▼
DiffusionRetryLLMServer.generate(request)      ← relay becomes a pass-through
  ▼
vLLMOmniAsyncServer.generate(request)
  │  request.to_engine_prompt()          ← the single adaptation point to pinned upstream
  ▼
vllm-omni engine → rollout adapter
  │  MediaOutput.from_diffusion_output(…)  ← the single adaptation point back
  ▼
MediaOutput(media=[MediaOut(video), MediaOut(audio)], trajectory={…}, extra={…})
  ├──► reward scorers / dump / logging  ── read MediaOut.modality
  └──► training adapter                 ── read MediaOutput.trajectory
```

Two adaptation points, both in `workers/rollout/`, both against the pinned engine. Everything
upstream and downstream of them speaks the repo's own types — the 10 diffusion pipeline packages,
the scorers, the trainer and the tracking layer never import a vllm-omni type to answer a question
about a modality.

---

## 5. Detailed design

### 5.1 Module placement

One new module, `verl_omni/pipelines/io.py`, with `torch` as its only third-party import so it
stays CPU-importable (G6). It sits **below** the pipeline registry and imports nothing from it,
so it cannot collapse behaviour across an `(architecture, algorithm)` boundary — which
`.agents/rules/pipelines.md` forbids.

`verl_omni/pipelines/` rather than `workers/rollout/`, because both rollout and training
adapters are consumers and `request_batch.py` already establishes that shared plumbing lives
here. A new module rather than extending `pipelines/utils.py` (where `ImageGenerationRequest`
lives): that module imports `diffusers`, `tensordict` and `verl.utils.device` at module scope
(`:21-29`), so anything added there fails G6. `utils.py` will import from `io.py`, not the
reverse.

### 5.2 The input types

```python
Modality = Literal["text", "image", "video", "audio"]   # this repo's own vocabulary — deliberately
                                                        # NOT upstream's, which also carries
                                                        # "video"/"videos", "actions" and "codec"
                                                        # and is never ingested here (§5.4)


@dataclass
class MediaRef:
    """One conditioning input, carrying its own modality tag."""

    modality: Modality
    data: Any                    # decoded PIL / ndarray / tensor, as ImageGenerationRequest.images
    role: str = "condition"      # condition | reference | keyframe | identity — the M-3 case
    meta: dict[str, Any] = field(default_factory=dict)   # fps, frame_index: per-input scalars
                                                         # today lost or promoted to sampling knobs


@dataclass
class PromptBundle:
    """Text side of one request; collapses four of the twelve kwargs."""

    ids: list[int]
    text: str | None = None      # required, not cosmetic: four in-tree paths need the string —
                                 # bagel decodes ids back to text (:273-276) and smuggles a text
                                 # negative_prompt through extra_args (:280-282), qwen_image_edit
                                 # synthesizes prompt="" (:359), wan22 keys warmup off
                                 # prompt == "dummy run" (:384). ids alone cannot express any.
    mask: torch.BoolTensor | None = None
    extra_ids: dict[str, list[int]] = field(default_factory=dict)   # per-text-encoder, as
                                                                    # _tokenize_per_encoder emits


@dataclass
class MediaRequest:
    request_id: str
    prompt: PromptBundle
    negative_prompt: PromptBundle | None = None
    conditions: list[MediaRef] = field(default_factory=list)
    sampling_params: dict[str, Any] = field(default_factory=dict)    # left untyped by N3
    mm_processor_kwargs: dict[str, Any] | None = None
    priority: int = 0

    def multi_modal_data(self) -> dict[str, list[Any]]:
        """Group ``conditions`` by modality, byte-identical to ``_build_multi_modal_data``."""

    def to_generate_kwargs(self) -> dict[str, Any]:
        """Lower to the twelve-kwarg form, so M1 needs no server change."""
```

Twelve kwargs become seven fields, and the nine input-describing ones become three. **The
property that matters: `conditions` is a list.** A fifth modality, a second image with a
different role, or a reference video alongside a keyframe all extend the list; none touches a
signature. `Modality` is a `Literal` rather than an `Enum` because these values round-trip
through `sampling_params` dicts, `extra_args` and `non_tensor_batch` object arrays — all
plain-data channels.

`MediaRequest` subsumes `ImageGenerationRequest` field for field: `images` becomes the `"image"`
slice of `conditions`, `metadata` becomes `sampling_params` plus per-`MediaRef` `meta`. A
classmethod projecting a `MediaRequest` down to the old shape keeps `qwen_image_edit_flow_grpo`
working unchanged through M1–M3. Note the direction of the fix: the five candidates are a
*decoding* concern that disappears once the encoder stops writing the same dict twice
(`:506-508`).

### 5.3 The output types

```python
@dataclass
class MediaOut:
    """One generated medium."""

    modality: Modality           # declared, never inferred — replaces every ndim test in §1.3
    data: torch.Tensor           # image [C,H,W]; video [T,C,H,W]; audio [S] or [C,S]
    fps: float | None = None     # attached to the medium it describes, not to a flat namespace
    sample_rate: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaOutput:
    media: list[MediaOut]        # >1 for joint A/V: LTX-2's (video, audio) tuple
    log_probs: torch.Tensor | None = None
    stop_reason: str | None = None
    num_preempted: int | None = None
    trajectory: dict[str, Any] = field(default_factory=dict)   # per-step RL data (G4)
    extra: dict[str, Any] = field(default_factory=dict)        # pipeline-private escape hatch

    @property
    def primary(self) -> MediaOut:
        """The medium the reward and the dump path act on."""

    def get(self, modality: Modality) -> MediaOut | None:
        """The first medium of ``modality``, or None."""
```

`trajectory` takes the per-step keys (`all_latents`, `all_timesteps`, `all_log_probs`,
`latents_clean`, `train_timesteps`) and `extra` keeps the genuinely private ones (`img_shapes`,
`condition_image_latents`). It stays a `dict[str, Any]` rather than becoming a dataclass because
its key set genuinely differs by algorithm — FlowGRPO wants `all_*`, NFT wants `latents_clean` +
`train_timesteps`, DPO wants neither — and freezing that would force a per-algorithm union or
`Optional` on everything. What M4 buys is that the *slot* is declared and its keys are documented
per algorithm, so a mismatch is localised.

### 5.4 The two adaptation points

- **In:** `MediaRequest.to_engine_prompt()` — in M1, `to_generate_kwargs()` — producing the
  `OmniCustomPrompt` + `OmniDiffusionSamplingParams` pair `_preprocess_input` builds today
  (`:429-523`). This is where the 80-field upstream dataclass is populated, and the only place a
  pin bump can require an edit.
- **Out:** `MediaOutput.from_diffusion_output(final_res, …)`, absorbing `_process_output`
  (`vllm_omni_async_server.py:546-654`) including the `images[0]` read, `_maybe_unbatch`, and the
  `custom_output` / `multimodal_output` merge.

**The modality must be declared by this repo, not read from upstream's `final_output_type`.**
That field *cannot express video on the path verl-omni uses.* The single-stage diffusion
constructor sets `final_output_type = "audio" if supports_audio_output(model_class_name) else
"image"` (`vllm_omni/engine/async_omni_engine.py:1000`) — there is no branch that can produce
`"video"`. Upstream does use `"video"`, but only in multi-stage stage configs
(`model_executor/stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29`),
which this path never builds. So for `wan22_dance_grpo` the field is populated with a confidently
wrong `"image"` — and it is never empty, so a prefer-declared-else-infer rule would never reach
its fallback. Trusting it would make M2 **cause** M-1 rather than fix it: `ray_diffusion_trainer.py:309`
would read `"image"` for a wan22 rollout and dump `0.jpg` from a `[T,C,H,W]` tensor,
`tracking.py:156` would stop emitting `wandb.Video`, and a video pipeline pointed at
`http_scorer_client` would be *declared* an image and silently scored on frame 0.

Modality is therefore declared by the **rollout adapter**, the only component that knows what it
produced: a class attribute on the adapter, or `MediaOut(modality=…)` at construction, consistent
with `DiffusionPipelineConfig.num_frames` (`config/diffusion/rollout.py:71` — `> 1` implies video).
`final_output_type` is read only as a cross-check that logs a warning on disagreement, never as the
source. This removes the RFC's dependency on an upstream field that cannot answer the question, and
it is why R4 is a *design constraint* rather than an accepted risk.

### 5.5 Compatibility seam, and naming

M1 adds two derived properties to the existing `DiffusionOutput` (`replica.py:20-32`), computed
from today's conventions — `modality` (adapter-declared where available, else `ndim`; **not**
`final_output_type`, §5.4) and `media` (the primary tensor plus audio/fps out of `extra_fields`). Every consumer can migrate to the declared
field before anything about the wire format changes, and `DiffusionOutput` keeps its name,
fields and Pydantic base. This is what makes M1 revertible: deleting the two properties and the
new module restores the tree exactly.

An earlier draft called these `OmniGenRequest` / `OmniGenOutput`. Both prefixes are wrong here:
**`Omni*`** denotes the autoregressive track (`OmniModelBase`, `OmniAlgoConfig`, …), which N4
puts out of scope; **`Diffusion*Output`** already names two different classes coexisting in the
same modules (`replica.py:20` and `vllm_omni/diffusion/data.py:1196`); and
**`*GenerationRequest`** already collides too (`pipelines/utils.py:45` vs
`protocol/images.py:33` — two unrelated types, one name, both importable in one process), so
`MediaGenerationRequest` would add a third near-homonym. `MediaRequest` / `MediaOutput` /
`MediaRef` / `MediaOut` / `PromptBundle` / `Modality` are all unused in the tree (verified: no
`class Media*`, no bare `Modality`).

---

## 6. Milestones

Ordered by risk. Each is a separate PR with its own tests, revertible without touching the next.

**M1 — `io.py` plus derived accessors. No behaviour change.**
The types in §5.2-5.3, the two `DiffusionOutput` properties (§5.5), and `to_generate_kwargs()`.
Nothing calls them yet. Entirely CPU-testable.
*Title:* `[rollout, pipelines] feat: add unified media I/O types`.

**M2 — read the declared modality at the five modality sites.**
The last column of §1.3's table, all seven rows: five stop inferring modality, and the two that
legitimately test for batching are left alone but no longer confusable with them. Fixes M-1, and is where the real review effort belongs because
it touches the reward path. Each site keeps its current behaviour for the modality it currently
handles; only how the modality is determined changes.
*Title:* `[trainer, reward] refactor: read declared modality instead of tensor rank`.

**M3 — accept the request object.**
`generate(request: MediaRequest)` as an overload. Deletes the six-kwarg relay
(`diffusion_llm_server.py:55-80`) and the seven-kwarg relay
(`single_turn_agent_loop.py:104-113`), and stops `:506-508` writing `multi_modal_data` twice.
Fixes M-3.
*Title:* `[rollout] refactor: accept MediaRequest in the diffusion generate path`.

**M4 — promote the known keys.**
`audio`, `audio_sample_rate`, `fps` into `MediaOut`; per-algorithm trajectory keys into
`MediaOutput.trajectory`; `extra` stays the escape hatch. Fixes M-2 and M-4. It does **not**
retire the `image_latents` guard (`model_base.py:355-371`): that guard defends a name the MFU
FLOPs counter reserves inside the **training-side** `model_inputs` dict, which no field of
`MediaOut`/`MediaOutput` governs. What M4 does remove is the guard's *trigger* — an adapter
emitting `image_latents` where `condition_image_latents` was meant — since the condition latent
becomes a named slot instead of a string key. Making the reservation itself a declaration is
follow-up work on the training side, out of scope here.
*Title:* `[rollout, trainer] refactor: promote trajectory and media keys out of extra_fields`.

**M5 — retire the five-candidate lookup.**
Collapse `utils.py:79-83` to one place and move `wan22_dance_grpo:373-381` onto it.
`bagel_flow_grpo:285-288` is **deleted rather than migrated** — it only copies
`extra_args.multi_modal_data` up into `custom_prompt`, and M3 stops `:506-508` writing the dict
twice, so there is nothing left for it to reconcile. Also where the dead paths get resolved: either give
`multi_modal_data["video"]` a consumer or stop writing it, and either wire `audio_data` to the
agent loop or delete the parameter. Fixes M-5; deletes more than it adds.
*Title:* `[pipelines, rollout] refactor: single condition-input lookup via MediaRef`.

M3 and M4 are worth doing only if M1's types survive contact with a second pipeline; if they do
not, M1+M2 stand alone and M-1 is still fixed. M5 depends on M3.

---

## 7. Test plan

Placement under `tests/<module>/` is the commit gate and the `_on_cpu.py` suffix is what CI
selects on, so both are load-bearing (`.agents/rules/testing.md`).

**M1 — `tests/pipelines/test_unified_io_on_cpu.py`.** `io.py` imports with `diffusers` and
`vllm_omni` absent from `sys.modules` (G6, the test that keeps the module clean);
`multi_modal_data()` byte-identical to `_build_multi_modal_data` for all eight
image/video/audio present-absent combinations including the empty dict; `to_generate_kwargs()`
round-trips from the twelve kwargs with `None`s included; `DiffusionOutput.modality` is
`"image"` for `[C,H,W]` and `"video"` for `[T,C,H,W]`, and a `[T,C,H,W]` output whose
`final_output_type` says `"image"` — the wan22 case, which is what the engine actually produces
(§5.4) — still resolves to `"video"`; `get("audio")` returns `None` rather than raising, and
`primary` errors clearly on empty `media`.

**M2 — regression, not new coverage.** The three rank-sniffing scorers keep their existing CPU
tests green **unchanged** — any diff there means M2 changed behaviour, which it must not. One new
case pins the §1.3 collision: an image `MediaOut` and a video `MediaOut` with the *same*
`data.ndim` route differently. This is the test that would have caught M-1, and it is only
expressible once modality is a field. `tests/trainer/` gains a case for mp4-vs-jpg selection
from the declared modality.

**M3-M5 — contract tests.** `generate(request)` and `generate(**kwargs)` produce identical
`OmniCustomPrompt` + `OmniDiffusionSamplingParams` pairs, field by field. Per-algorithm
trajectory-key tables (FlowGRPO / NFT / DPO) asserted against all eight adapters, so a missing
key fails in CI rather than mid-run. `tests/pipelines/test_image_edit_interface_on_cpu.py`
already pins the five-candidate precedence (`:47-63`, `:84-85`): it is M3's compatibility test
(must stay green unchanged) and is deliberately **rewritten** in M5 — as an explicit hunk in
that PR, never a silent deletion. M5 also asserts a `MediaRef(modality="video")` reaches the
adapter, or, if M5 deletes `video_data` instead, that the parameter is gone — one of the two
must be true and CI records which.

**GPU.** One image pipeline (`qwen_image_flow_grpo`) and one video+audio pipeline
(`ltx2_flow_grpo`) re-run unchanged after M2 and M4 — the only checks covering the
`multimodal_output` merge against a real engine. The image-conditioned path needs its own: one
`qwen_image_edit_flow_grpo` run after M3 and M5, since it is the only live multi-input path.

**Every milestone.** `pre-commit run --files <changed>`. Note `autogen-trainer-cfg` fails in the
current venv with a pre-existing `ModuleNotFoundError: omegaconf` (`scripts/print_cfg.py:16`),
unrelated to these files; all other hooks must pass.

---

## 8. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | **M2 changes a reward silently** — the reward path is the one place a wrong refactor yields a plausible number instead of an error. | M2 keeps each scorer's behaviour byte-identical **for the modality that scorer handles today**, and changes only how the modality is chosen; the existing scorer tests must pass *unchanged*. The one intended behaviour change is the M-1 fix itself: `http_scorer_client` starts **raising** on a declared non-image modality where it used to score frame 0. That is a new error on a path that was silently wrong, it needs its own test, and it is the only diff in the reward path that may land — any other blocks the PR. |
| R2 | **A fifth representation** — adding types without deleting the old ones. | M3 and M4 are the deletions and they are in the plan. If M3 never lands, `to_generate_kwargs()` is one function and M2 replaced inference with field reads — no new representation, only accessors. |
| R3 | **`trajectory` becomes the new `extra_fields`** under a nicer name. | The per-algorithm key tables in §7 are the constraint; §5.3 states why it stays a dict and §9 q1 tracks promoting it. |
| R4 | **`final_output_type` cannot express video** on this path — the single-stage constructor only ever picks `"audio"` or `"image"` (`async_omni_engine.py:1000`), so it is confidently wrong for `wan22_dance_grpo` and never empty. | This is why §5.4 makes the **adapter** the source of truth and demotes `final_output_type` to a warn-on-disagreement cross-check. §7's M1 case pins the wan22 shape explicitly. Had this RFC kept prefer-declared-else-infer, M2 would have caused M-1. |
| R5 | **Ten packages, one shared type** — a type fitting eight image pipelines may not fit a joint A/V one. | M1 lands with two consumers of *different* shape (one image pipeline and `ltx2_flow_grpo`), not eight of the same. G5 makes each milestone revertible. |
| R6 | **Merge conflicts with in-flight work** — anything touching `_process_output` or a rollout adapter. | M1 adds a file and two properties; M2 edits one line per site. M3/M4 touch the adapters and should be sequenced after whatever pipeline work is in flight, rather than racing it. |

---

## 9. Open questions

1. **`num_outputs_per_prompt` placement.** It lives in `sampling_params` and is re-derived as an
   explicit argument to `split_diffusion_output_by_request` (`request_batch.py:210`), where it
   drives the shape-coincidence slicing of §1.3. Should `MediaRequest` own it — making the group
   size declared and the slicing exact — or keep deferring? Turns on whether request batching
   stays a rollout-adapter concern.
2. **May `MediaRef.data` be a `str`?** Accepting a URL or `file_id` matches upstream's reference
   types, but every dataset here hands over decoded objects. Allowing both without a resolver is
   how upstream's `image_reference` ended up needing one.
3. **Should tensor layout be declared too?** `shape[-1] in (1, 3)` channels-last guessing
   survives in two scorers (§1.3). A `layout: Literal["chw", "hwc"]` on `MediaOut` would close
   it, at the cost of auditing every producer. Not in M1–M4.
4. **`"actions"` as a fifth modality.** Upstream already emits it (§1.5). Add it pre-emptively,
   or wait for a consumer? Adding it costs nothing; using it untested costs a false claim of
   support.
5. **Does `verl` want this, or only verl-omni?** The rank-sniffing pattern is local to the
   diffusion path, which is verl-omni's. Confirm before proposing anything upstream.

---

## 10. Appendix — key upstream references

Verified at vllm-omni `0.24.1.dev26+gfe478a95a`. verl-omni references are cited inline above and
are greppable in-tree; only the pinned upstream ones are collected here.

| What | Where |
| --- | --- |
| `ImageGenerationRequest` (HTTP), size validator | `entrypoints/openai/protocol/images.py:33`, `:66-79` |
| `VideoGenerationRequest`; `SizeStr`; `VideoParams` | `entrypoints/openai/protocol/videos.py:97`, `:30`, `:47-60` |
| reference tagged unions (the idea `MediaRef` takes) | `entrypoints/openai/protocol/videos.py:63-94` |
| audio request | `entrypoints/openai/protocol/audio.py:321` |
| `OmniDiffusionRequest` | `diffusion/request.py:14` |
| `OmniDiffusionSamplingParams` (80 fields) | `inputs/data.py:178` |
| `OmniCustomPrompt`, `OmniPromptType` | `inputs/data.py:110-129`, `:137` |
| `OmniRequestOutput`, `final_output_type`, `images` | `outputs.py:63`, `:87`, `:91` |
| `from_diffusion` default `"image"`; `is_image_output` reads the field (R4) | `outputs.py:165-205`, default at `:178`, `:318` |
| **the single-stage `final_output_type` decision — `"audio"` or `"image"`, never `"video"` (R4)** | `engine/async_omni_engine.py:1000` |
| upstream's only `"video"` producers — multi-stage stage configs this path never builds | `model_executor/stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29` |
| `DiffusionOutput.output` union | `diffusion/data.py:1196-1202` |
| `_build_multimodal_output` keys; `final_output_type="audio"` | `diffusion/output_formatter.py:158-171`, `:225` |

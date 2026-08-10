# RFC-0003: Unified multi-input / multi-output interface for diffusion rollout

- **Status:** Draft
- **Scope (this RFC):** the request and output objects that cross the
  agent-loop → rollout-server → rollout-adapter → reward → training-adapter path for
  **diffusion** pipelines. Multimodal *conditioning inputs* (text + image + video +
  audio) and multimodal *generated outputs* (image, video, audio) are both in scope.
- **Not in scope:** the autoregressive omni track (`OmniModelBase`,
  `OmniRolloutPipelineBase`), the `TokenOutput` path, and any change to the pinned
  vllm-omni protocol or engine types. See §3.
- **Companion:** RFC-0001 / RFC-0002 (MiniMax-H3) are the first consumers that need
  two output modalities from one request; this RFC generalises the ad-hoc seams those
  integrations had to open. Neither depends on this RFC landing.
- **Prior art / template:** two things in-tree already point this way and are the
  reason this RFC proposes a *structure* rather than a per-modality split.
  `verl_omni/pipelines/utils.py:45` already defines an `ImageGenerationRequest`, whose
  only call site carries a `NOTE` deferring "migrating the existing T2I pipelines onto
  it" to a follow-up PR
  (`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`) — **this RFC is
  that follow-up, widened from images to all four modalities** (§1.9). And the LTX-2.3
  `t2av` pipeline (`verl_omni/pipelines/ltx2_flow_grpo/`) is the only merged pipeline
  that already emits video **and** audio from one request; every workaround it needed
  is cited below as evidence rather than as a pattern to copy.
- **Last updated:** 2026-08-10
- **Verified against:** verl-omni `8c0f5fa`, vllm-omni `0.24.1.dev26+gfe478a95a`.
  Every `file:line` below was read at those revisions.

---

## 0. TL;DR

**text + image conditioning already works** (§1.9) — a dataset row carries text and
image in separate columns, and three pipelines consume the image. What is missing is
not the capability but the contract: one generation request is re-encoded **four**
times between a dataset row and the DiT and no two encodings agree on shape; the one
conditioning image is written to two places and looked for in five; and on the output
side **the generated modality is not recorded anywhere in this repo — it is re-derived
from tensor rank at seven independent call sites**, with everything that is not the
primary tensor travelling as untyped string keys in a `dict[str, Any]`.

Six facts, each verified:

1. `vLLMOmniAsyncServer.generate` takes **twelve flat keyword arguments**, nine of
   which describe the input (`vllm_omni_async_server.py:330-344`). Two intermediaries
   re-list six and seven of them verbatim just to pass them through
   (`diffusion_llm_server.py:55-79`, `single_turn_agent_loop.py:104-113`).
2. `DiffusionOutput.diffusion_output` is typed `Any` and documented as
   "image tensor (CHW format) / video tensor (TCHW format)"
   (`replica.py:20-32`). Consumers recover the modality with `ndim` tests.
3. **Those tests disagree with each other.** `ndim == 4` means *video* in
   `utils/tracking.py:156` and *a batch of images* in
   `utils/reward_score/http_scorer_client.py:40`, where it silently keeps frame 0 and
   discards the rest.
4. The rollout↔training contract is a string-keyed dict. It is already policed by a
   hand-written runtime guard whose whole job is to catch one key being confused for
   another (`model_base.py:363-372`), and one whose keys are batch-sliced by a
   **shape-coincidence heuristic** (`request_batch.py:197-207`).
5. **vllm-omni already tags the output modality** — `OmniRequestOutput.final_output_type`
   (`outputs.py:87`, one of `"text" | "image" | "audio" | "latents"`) — and verl-omni
   never reads it (grep: zero hits outside a `qwen3_omni` stage-config literal).
6. **On the input side, one conditioning image is written to two places and looked for
   in five**, `multi_modal_data["video"]` is written and never read, and `audio_data`
   has no caller at all (§1.9). Three pipelines consume the image, each with its own
   lookup.

Proposal: one small, additive module, `verl_omni/pipelines/io.py`, holding
`MediaRef` + `PromptBundle` + `MediaRequest` on the way in and `MediaOut` +
`MediaOutput` on the way out. Modality becomes **declared** data, conditioning inputs
become a **list** instead of N optional kwargs, and the RL trajectory gets a
first-class slot separate from the pipeline-private escape hatch. `MediaRequest` is
the existing `ImageGenerationRequest` (`pipelines/utils.py:45`) generalised past
images, which its own call-site `NOTE` already asked for.

We land it in four milestones, cheapest-risk first: types + derived accessors (no
behaviour change) → replace the seven inference sites → accept the request object →
promote the known dict keys. **M1+M2 carry most of the value and are reversible.**

Explicitly **not** proposed: mirroring vllm-omni's per-modality
`ImageGenerationRequest` / `VideoGenerationRequest` HTTP split, and typing
`sampling_params`. §5.11 gives the reasons.

---

## 1. Background

### 1.1 The request crosses four representations

| # | Layer | Type | Shape of the abstraction |
| --- | --- | --- | --- |
| 1 | vllm-omni HTTP protocol | `ImageGenerationRequest` (`entrypoints/openai/protocol/images.py:33`), `VideoGenerationRequest` (`videos.py:97`), `OpenAICreateAudioGenerateRequest` (`audio.py:321`) | three per-modality Pydantic models, **no shared base** |
| 2 | vllm-omni engine | `OmniDiffusionRequest` (`diffusion/request.py:15`) | modality-agnostic, 4 fields, unified **by union** |
| 3 | **verl-omni rollout** | `vLLMOmniAsyncServer.generate` (`vllm_omni_async_server.py:330`) | **no request object at all** — 12 flat kwargs |
| 4 | verl-omni config | `DiffusionPipelineConfig` (`workers/config/diffusion/rollout.py:60-74`) | the same 10 knobs, declared statically a fourth time |

Layers 1, 2 and 4 are each internally coherent. Layer 3 — the one this repo owns — is
the one with no type.

### 1.2 Layer 1: per-modality by design, and that design is not ours

`ImageGenerationRequest` and `VideoGenerationRequest` share roughly 70% of their
fields with no common base class:

| | `ImageGenerationRequest` | `VideoGenerationRequest` |
| --- | --- | --- |
| shared | `prompt`, `negative_prompt`, `model`, `size`, `user`, `num_inference_steps`, `guidance_scale`, `true_cfg_scale`, `flow_shift` | same |
| `size` type | free-form `str` + a hand-written validator that only checks for an `"x"` (`images.py:66-79`) | `SizeStr = ^\d+x\d+$` (`videos.py:30`) |
| modality-only | `n`, `layers`, `response_format`, `bot_task`, `system_prompt`, `use_system_prompt` | `seconds`, `fps`, `num_frames`, `guidance_scale_2`, `boundary_ratio`, `generate_sound`, `sound_duration` |
| conditioning | none | `image_reference`, `video_reference`, `audio_reference` |

Two details matter, because one is the part to copy and one is the part not to:

- **Copy the tagged-union reference.** `ImageReference = UrlImageReference |
  FileImageReference` (`videos.py:63-73`, and the same shape for video and audio at
  `:76-94`) makes each conditioning input carry its own kind, so a consumer never has
  to guess from the payload's Python type. This is exactly the property §5.3 wants.
- **Do not copy the duplicated block.** `width` / `height` / `fps` / `num_frames`
  exist both top-level (`videos.py:135-138`) **and** inside the optional `VideoParams`
  block (`videos.py:47-60`). Two spellings of one value with no documented precedence
  is a live ambiguity, not a pattern.

The split exists to satisfy the OpenAI Images and Videos API shapes at an HTTP
boundary. **verl-omni has no such boundary** — the rollout is an in-process RL loop.
§5.11 records this as an explicit non-goal.

### 1.3 Layer 2: modality-agnostic, unified by union

Both entrypoints collapse into one object:

```python
@dataclass
class OmniDiffusionRequest:            # vllm_omni/diffusion/request.py:15
    prompt: OmniPromptType
    sampling_params: OmniDiffusionSamplingParams
    request_id: str
    kv_sender_info: dict | None = None
```

It unifies by **union**, not by structure. `OmniDiffusionSamplingParams`
(`inputs/data.py:178`) is an **80-field** dataclass — counted with
`dataclasses.fields` at the pinned revision, not estimated — carrying every modality's
knobs simultaneously (`num_frames`, `fps`, `frame_rate`, `audio_latents`,
`boundary_ratio`, `layers`, and a `kv_metadata` family), reconciled by defaulting
rather than by type. `OmniPromptType` (`inputs/data.py:137`) is a five-way alias whose
`OmniCustomPrompt` member (`inputs/data.py:110-129`) — `prompt_ids`,
`negative_prompt_ids`, `prompt_mask`, `negative_prompt_mask`, `extra_args` — is the
escape hatch verl-omni actually uses.

**This is the important precedent: vllm-omni keeps the per-modality split at the HTTP
edge and exactly one modality-agnostic request inside.** verl-omni sits entirely
inside that edge, so it should have the inside shape. Today it has neither.

### 1.4 Layer 3, input side: flat kwargs and a relay chain

```python
async def generate(                    # vllm_omni_async_server.py:330-344
    self,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    request_id: str,
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    audio_data: Optional[list[Any]] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    negative_prompt_ids: Optional[list[int]] = None,
    prompt_mask: torch.BoolTensor | None = None,
    extra_prompt_ids: Optional[dict[str, list[int]]] = None,
    negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
    priority: int = 0,
) -> DiffusionOutput | TokenOutput:
```

Consequences visible in the tree today:

- **Every intermediary re-lists the arguments.**
  `DiffusionRetryLLMServer.generate` (`diffusion_llm_server.py:55-79`) restates six
  parameters verbatim and forwards the rest through `**kwargs`, purely to wrap the
  call in a retry loop. `DiffusionSingleTurnAgentLoop.run`
  (`single_turn_agent_loop.py:104-113`) restates seven. Adding a modality means
  editing all three signatures.
- **The only real unification already present** is
  `_build_multi_modal_data` (`vllm_omni_async_server.py:366-380`), which folds three
  optional lists into `{"image": ..., "video": ..., "audio": ...}` — i.e. it converts
  positional kwargs back into the tagged list they should have been.
- **Modality is hardcoded on the way in, too.** For multi-stage pipelines
  `_preprocess_input` sets `custom_prompt["modalities"] = ["image"]`
  (`vllm_omni_async_server.py:497`) unconditionally.
- **Unknown sampling keys are silently accepted.** `_preprocess_input:510-517`
  partitions `sampling_params` by `hasattr(OmniDiffusionSamplingParams, k)`; anything
  unrecognised becomes an `extra_args` entry. A typo in a knob name is not an error —
  it is a no-op that surfaces as a quality regression.

### 1.5 Layer 3, output side: the modality is inferred from tensor rank

```python
class DiffusionOutput(BaseModel):      # verl_omni/workers/rollout/replica.py:20-32
    diffusion_output: Any
    """generated image tensor (CHW format) / video tensor (TCHW format)"""
    log_probs: Optional[Any] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
    extra_fields: dict[str, Any] = {}
```

There is no modality field, so **seven** independent sites re-derive it:

| Site | Test | Meaning assigned |
| --- | --- | --- |
| `trainer/diffusion/ray_diffusion_trainer.py:309` | `outputs.ndim == 5` | video (`[N,T,C,H,W]`) → mp4, else jpg |
| `utils/tracking.py:156` | `out.ndim == 4` | **video** (`[T,C,H,W]`) → `wandb.Video`, else `wandb.Image` |
| `utils/reward_score/http_scorer_client.py:40` | `image.ndim == 4` | **a batch** → keep `[0]`, drop the rest |
| `utils/reward_score/hpsv3_reward.py:388-410` | 3 / 4 / 5-way, plus `shape[-1] in (1, 3)` to guess channels-last | image / video / batched video |
| `utils/reward_score/genrm_ocr.py:141-162` | same 3 / 4 / 5-way + channels-last guess | image / video / batched video |
| `utils/reward_score/unified_reward.py:70-76` | 3 / 4 only, `raise` otherwise | image / video |
| `pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py:381` | `video.ndim == 5` | leading batch axis to squeeze |

Two of these are worth stating plainly:

- **`ndim == 4` carries opposite meanings in two files.** `tracking.py:156` reads it
  as a video; `http_scorer_client.py:40` reads it as a batch of images and keeps only
  the first element. Nothing in the config system prevents a video pipeline from being
  pointed at the HTTP scorer, and if that happens the reward is computed on frame 0 of
  each clip **without any error** — the RL signal is simply wrong. The two files are
  each locally correct; the bug is that the question "is this a video?" has no single
  answer to consult.
- **The rank convention itself is not stable across the path.** The dump path sees a
  batched 5-D tensor (`ray_diffusion_trainer.py:309`) while the validation-logging
  path sees per-sample 4-D (`tracking.py:156`). Same question, two different magic
  numbers, in the same repo, for the same data.

Upstream, meanwhile, admits the union honestly: `DiffusionOutput.output` is typed
`torch.Tensor | tuple[Any, ...] | dict[str, Any] | None` (`diffusion/data.py:1202`),
and LTX-2 uses the tuple arm — `output.output = (video[0], audio)`
(`ltx2_flow_grpo/vllm_omni_rollout_adapter.py:383`). verl-omni's `DiffusionOutput`
then flattens that back to `Any`.

### 1.6 The rollout↔training contract is an untyped dict

Everything that is not the primary tensor rides in `extra_fields`, assembled in
`_process_output`:

```python
extra_fields = {k: _maybe_unbatch(v) for k, v in custom_output.items() if k != "all_log_probs"}
multimodal_output = final_res.multimodal_output or {}
if isinstance(multimodal_output, dict):
    for key, value in multimodal_output.items():
        extra_fields.setdefault(key, _maybe_unbatch(value))
extra_fields["global_steps"] = self.global_steps
#                                       vllm_omni_async_server.py:619-635
```

The union of keys the eight diffusion rollout adapters actually emit:

| Producer | `custom_output` keys |
| --- | --- |
| `qwen_image_flow_grpo` | `all_latents`, `all_log_probs`, `all_timesteps`, `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, `negative_prompt_embeds_mask` |
| `sd3_flow_grpo` | the above + `pooled_prompt_embeds`, `negative_pooled_prompt_embeds`, and conditionally `latents_clean` |
| `qwen_image_edit_flow_grpo` | the above + `img_shapes`, `condition_image_latents` |
| `wan22_dance_grpo` | as `qwen_image_flow_grpo` |
| `bagel_flow_grpo` | `all_latents`, `all_log_probs`, `all_timesteps` |
| `qwen_image_diffusion_nft` | `latents_clean`, `train_timesteps`, `prompt_embeds`, `prompt_embeds_mask`, `negative_prompt_embeds`, `negative_prompt_embeds_mask` |
| `qwen_image_dpo` | `latents_clean` + the four prompt-embed keys |
| `ltx2_flow_grpo` | the trajectory dict + `audio_prompt_embeds`, `negative_audio_prompt_embeds`, `audio_sample_rate` |

Plus whatever upstream merges in from `multimodal_output`, which
`_build_multimodal_output` (`diffusion/output_formatter.py:158-171`) populates with
`audio`, `audio_sample_rate`, `fps`, and `actions`.

Four consequences, all present in the tree:

1. **A hand-written guard exists solely to catch a key collision.**
   `DiffusionI2IModelBase.inject_condition` raises if `model_inputs` already contains
   `image_latents`, with the message *"the rollout adapter likely output
   'image_latents' instead of 'condition_image_latents'. Check the rollout adapter's
   custom_output keys."* (`pipelines/model_base.py:363-372`). That guard is the shape
   of the problem: the contract has to be policed at runtime because it is not
   declared anywhere.
2. **Batch slicing is decided by shape coincidence.**
   `_slice_batch_value` (`request_batch.py:197-207`) slices a tensor iff
   `value.shape[0] == req.num_reqs * num_outputs_per_prompt`, and leaves it alone
   otherwise. Whether a key is per-sample or shared is therefore inferred from a
   number, not known. A schedule tensor whose length happens to equal the batch size
   is sliced; a per-sample tensor that lost its leading axis is not.
3. **Unbatching is decided by `isinstance` plus `[0]`.** `_maybe_unbatch`
   (`vllm_omni_async_server.py:619-628`) takes element 0 of any tensor, array, list or
   tuple with a leading axis.
4. **The container a key lands in is not statically knowable.** In the agent loop,
   tensor-valued extra fields are promoted into the `TensorDict` and everything else
   into `non_tensor_batch` by `isinstance` (`diffusion_agent_loop.py:348-352` and
   `:383-388`). The trainer then has to look in both, in inconsistent order for two
   related keys:

   ```python
   audios=batch.batch.get("audio", batch.non_tensor_batch.get("audio")),
   audio_sample_rates=batch.non_tensor_batch.get(
       "audio_sample_rate", batch.batch.get("audio_sample_rate")
   ),
   #                              trainer/diffusion/ray_diffusion_trainer.py:391-393
   ```

   Neither `"audio"` nor `"audio_sample_rate"` is produced anywhere under
   `verl_omni/pipelines/` — both originate upstream. A pin bump that renames either
   one fails at **training** time with a `KeyError`, or worse, silently yields `None`
   and drops the audio track from every dumped mp4.

### 1.7 Layer 4: a fourth copy in config

`DiffusionPipelineConfig` (`workers/config/diffusion/rollout.py:60-74`) statically
declares `height`, `width`, `num_inference_steps`, `output_type`, `true_cfg_scale`,
`max_sequence_length`, `guidance_scale`, `num_frames`, `frame_rate` — the same knobs
again, later funnelled into the `sampling_params` dict.

`output_type` looks like a modality declaration and is not one: it defaults to
`"image"` for every pipeline including video ones, and LTX-2 has to rewrite it —
`return "pt" if output_type == "image" else output_type`
(`ltx2_flow_grpo/common.py:51-53`). It is a diffusers-style *format* selector
(`"pil"` / `"pt"` / `"np"` / `"latent"`), not a statement about what the model emits.

### 1.8 What upstream already tags and this repo discards

Three modality signals already exist on the objects verl-omni receives:

- `OmniRequestOutput.final_output_type: str` (`outputs.py:87`), one of
  `"text" | "image" | "audio" | "latents"`, set by the formatter — e.g.
  `final_output_type="audio"` at `diffusion/output_formatter.py:225`. **verl-omni
  never reads it.**
- `OmniRequestOutput.images: list[Image.Image]` (`outputs.py:91`) is declared as PIL
  images and in practice carries video tensors and `(video, audio)` tuples. The
  declared type is already wrong upstream; verl-omni inherits the ambiguity by
  reading `final_res.images[0]` into an `Any`.
- `multimodal_output` carries `fps` and `actions` alongside audio
  (`output_formatter.py:158-171`). `actions` is a fifth modality already present in
  the pinned engine; on today's path it would land in `extra_fields` and be dropped
  without comment.

### 1.9 What is already partially unified — and where it stopped

**text + image conditioning already works end to end today.** This RFC is not
introducing multimodal input; it is generalising a mechanism that exists, is
exercised by an in-repo example, and whose author explicitly deferred the
generalisation.

The live path, for image-conditioned generation:

| Step | Where |
| --- | --- |
| dataset row carries text and image in **separate** columns — `prompt` chat messages with an `<image>` placeholder, plus an `images: [{"bytes": …}]` column | `examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py:63-71` |
| images extracted from the rendered messages | `single_turn_agent_loop.py:81-83`, via verl's `AgentLoopBase.process_vision_info` (`verl/experimental/agent_loop/agent_loop.py:239`) |
| passed as a separate kwarg | `single_turn_agent_loop.py:108-109` (`image_data=`, `video_data=`) |
| folded into one dict | `_build_multi_modal_data` (`vllm_omni_async_server.py:366-380`) |
| written to the engine prompt **twice** | `_preprocess_input:506-508` |
| parsed back out | `ImageGenerationRequest.from_request_payload` (`pipelines/utils.py:64-105`) |

And there is already a request dataclass for it:

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

This is the right idea, and it is why this RFC proposes a structure rather than a
per-modality split. Five facts about how far it got:

1. **It is image-only.** There is no `videos` or `audios` field, no modality tag, and
   no `role`, so it cannot express "video conditioned on a keyframe plus a reference
   clip" — the case §2 M-3 describes.
2. **The generalisation was explicitly deferred, in the source.** The one call site
   carries a `NOTE`: *"only this image-edit pipeline consumes
   `ImageGenerationRequest` for now; migrating the existing T2I pipelines onto it is
   left to a follow-up PR to keep this change focused"*
   (`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`). **This RFC is
   that follow-up, widened from images to all four modalities.**
3. **It has to search five candidate locations for one value** —
   `images`, `image`, `multi_modal_data.image`,
   `extra_args.multi_modal_data.image`, `additional_information.condition_images`
   (`pipelines/utils.py:79-85`). That is not defensive coding; it is the cost of
   having no declared contract. Two of the five have no producer on this branch at
   all, and one exists only because `_preprocess_input:506-508` writes the same dict
   into two places:

   ```python
   if multi_modal_data:
       custom_prompt["multi_modal_data"] = multi_modal_data
       custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}
   ```

4. **Two of eleven pipelines bypass it and hand-roll the same lookup.**
   `bagel_flow_grpo:285-288` copies `extra_args["multi_modal_data"]` up to the
   top level; `wan22_dance_grpo:373-381` checks both containers itself and then
   reads `.get("image")`. Three consumers, three spellings of one question.
5. **It collides by name with upstream.** `ImageGenerationRequest` is also the name
   of vllm-omni's HTTP protocol model (`entrypoints/openai/protocol/images.py:33`),
   which is a different thing entirely. Both are importable in the same process.

Video and audio conditioning are plumbed to different depths, which is worth stating
precisely because "the parameter exists" is not the same as "the modality works":

| Modality in | Reaches the engine prompt? | Any consumer? | Status |
| --- | --- | --- | --- |
| text | yes | all 11 pipelines | **live** |
| image | yes — `multi_modal_data["image"]`, written twice (`:506-508`) | `qwen_image_edit_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo` | **live**, three lookup styles |
| video | yes — `multi_modal_data["video"]` (`:377`), populated from `process_vision_info` (`single_turn_agent_loop.py:83`) | **none** — `:377` is the only occurrence of the `"video"` key under `verl_omni/` | **write-only** |
| audio | only if a caller passes `audio_data`; `diffusion_llm_server.py:78` relays it | **none** — no caller on this branch supplies it | **unreachable** |

So a contributor reading `generate`'s signature sees four conditioning modalities and
gets one working, one silently discarded, and one that cannot be reached. Nothing in
the type system says which is which.

---

## 2. Motivation

Five concrete failure modes, each traceable to a missing type rather than to a bug in
any individual file:

- **M-1: wrong rewards, no error.** The `ndim == 4` collision in §1.5. A video
  pipeline configured with `http_scorer` scores frame 0 and reports a plausible
  number. Nothing raises.
- **M-2: pin bumps break at training time, not request time.** The rollout↔training
  contract is a set of string keys spread over eight adapters plus upstream's
  `multimodal_output`. This is not hypothetical: the MiniMax-H3 bring-up (RFC-0001)
  hit exactly this when a `custom_output` key moved between vllm-omni revisions, and
  the failure surfaced as a `KeyError` several minutes into a multi-GPU run.
- **M-3: adding a modality costs three signature edits and a guard.** A second
  conditioning image with a different role (keyframe vs identity reference) has
  nowhere to go today except a new `*_data` kwarg threaded through
  `single_turn_agent_loop.py` → `diffusion_llm_server.py` →
  `vllm_omni_async_server.py`, or an untyped `extra_args` key. Both were used during
  the H3 and LTX-2 bring-ups.
- **M-4: a model that emits two modalities has no representation.** LTX-2 already
  does, and had to smuggle audio through the tuple arm of upstream's `output` union
  plus an `audio_sample_rate` key in `custom_output`
  (`ltx2_flow_grpo/vllm_omni_rollout_adapter.py:383, 389-397`). Every future joint A/V
  pipeline repeats that choice independently.
- **M-5: one conditioning image is looked for in five places.** §1.9. The producer
  writes it twice (`_preprocess_input:506-508`), one consumer searches five candidate
  keys (`pipelines/utils.py:79-85`), and two more hand-roll their own two-place
  lookup. Meanwhile `multi_modal_data["video"]` is written and never read, and
  `audio_data` cannot be reached at all. Every one of these is individually harmless
  and collectively means the input contract is whatever the last adapter happened to
  check.

The cost of *not* fixing this scales with pipeline count. There are currently 11
pipeline packages and 10 registered `(architecture, algorithm)` keys; each new one
re-decides the same three questions (how do I pass conditioning, how do I say what I
emitted, where do I put the trajectory) with no type to answer them.

---

## 3. Goals / Non-goals

**Goals**

- **G1.** One request type covering text + image + video + audio conditioning, such
  that adding a conditioning modality or a second input with a different role
  requires **no signature change** on the relay path. Concretely: generalise the
  existing image-only `ImageGenerationRequest` (`pipelines/utils.py:45`, §1.9) rather
  than add a second partial answer beside it.
- **G1b.** One place to read a conditioning input, replacing the five candidate keys
  (`pipelines/utils.py:79-85`) and the two hand-rolled lookups
  (`bagel_flow_grpo:285-288`, `wan22_dance_grpo:373-381`); and resolve the two dead
  paths — `multi_modal_data["video"]` written but never read, `audio_data` with no
  caller — in one direction or the other.
- **G2.** One output type in which the generated modality is **declared data**, and a
  single request may declare more than one output medium (video + audio).
- **G3.** Reduce the seven modality-inference sites (§1.5) to **zero**, by having each
  read a declared field.
- **G4.** Give the RL trajectory a first-class slot, distinct from the
  pipeline-private escape hatch, so a rollout↔training key mismatch becomes a
  construction-time error rather than a training-time `KeyError`.
- **G5.** Fully additive. Every milestone must be landable and revertible on its own,
  and no milestone may require editing all 11 pipeline packages at once.
- **G6.** CPU-testable. The new module must import without diffusers, vllm-omni, or a
  GPU, per `.agents/rules/pipelines.md`.

**Non-goals**

- **N1.** Changing the vllm-omni protocol, `OmniDiffusionRequest`, or
  `OmniDiffusionSamplingParams`. Those are pinned upstream; this RFC adapts at the
  boundary and never forks them.
- **N2.** Mirroring the `ImageGenerationRequest` / `VideoGenerationRequest` split
  (§5.11).
- **N3.** Typing `sampling_params` (§5.11).
- **N4.** Touching the autoregressive omni track or `TokenOutput`. `generate` serves
  both via `self._ar_mode`; this RFC changes only the diffusion arm.
- **N5.** Unifying the reward-scorer *signatures*. Making the scorers read a declared
  modality instead of guessing rank is in scope (G3); redesigning
  `compute_score_*` is not.
- **N6.** Removing the fourth config copy (§1.7). Config lives on a different
  lifecycle (Hydra + `_generated_*.yaml`) and is a separate change.
- **N7.** Renaming the two existing `DiffusionOutput` classes. See §5.10.

---

## 4. High-level architecture

Today, with the number of shape-changing hops marked:

```
dataset row
  │
  ▼  raw_prompt (chat messages)
DiffusionSingleTurnAgentLoop.run                        ──┐
  │  7 flat kwargs                                        │ each hop re-lists
  ▼                                                       │ the argument set
DiffusionRetryLLMServer.generate                        ──┤
  │  6 flat kwargs + **kwargs                             │
  ▼                                                       │
vLLMOmniAsyncServer.generate  (12 kwargs)               ──┘
  │  _build_multi_modal_data  →  {"image":…, "video":…, "audio":…}
  ▼  _preprocess_input        →  OmniCustomPrompt + OmniDiffusionSamplingParams(80)
vllm-omni engine  →  rollout adapter  →  DiffusionOutput(output=Tensor|tuple|dict)
  │  _process_output: images[0] → Any;  custom_output ∪ multimodal_output → extra_fields
  ▼
DiffusionOutput(diffusion_output: Any, extra_fields: dict[str, Any])
  │
  ├──► reward scorers        ──  ndim 3/4/5 guess  ×4 sites
  ├──► rollout dump          ──  ndim == 5 guess
  ├──► wandb val logging     ──  ndim == 4 guess
  └──► training adapter      ──  string keys, policed by a runtime guard
```

Proposed, with the same hops but one type at each end:

```
dataset row
  │
  ▼
DiffusionSingleTurnAgentLoop.run
  │  MediaRequest(prompt=PromptBundle, conditions=[MediaRef, …], sampling_params={…})
  ▼
DiffusionRetryLLMServer.generate(request)      ← relay becomes a pass-through
  ▼
vLLMOmniAsyncServer.generate(request)
  │  request.to_engine_prompt()   ← the single adaptation point to pinned upstream
  ▼
vllm-omni engine  →  rollout adapter
  │  MediaOutput.from_diffusion_output(...)    ← the single adaptation point back
  ▼
MediaOutput(media=[MediaOut(video), MediaOut(audio)], trajectory={…}, extra={…})
  │
  ├──► reward scorers        ──  read MediaOut.modality
  ├──► rollout dump          ──  read MediaOut.modality
  ├──► wandb val logging     ──  read MediaOut.modality
  └──► training adapter      ──  read MediaOutput.trajectory
```

Two adaptation points, both in `workers/rollout/`, both against the pinned engine.
Everything upstream and downstream of them speaks the repo's own types.

---

## 5. Detailed design

### 5.1 Module placement

One new module, `verl_omni/pipelines/io.py`.

It sits **below** the pipeline registry and imports nothing from it, so it does not
interact with `(architecture, algorithm)` dispatch — collapsing behaviour across a
registry boundary is forbidden by `.agents/rules/pipelines.md`, and these types
deliberately carry no behaviour that could. Third-party imports are limited to
`torch`, keeping the module importable on CPU (G6) and usable from `tests/pipelines/`.

`verl_omni/pipelines/` is the right home rather than `workers/rollout/`: both the
rollout adapters and the training adapters are consumers, and `request_batch.py`
already establishes that shared rollout/training plumbing lives here.

A new module rather than extending `verl_omni/pipelines/utils.py`, where
`ImageGenerationRequest` lives today: that module imports `diffusers`, `tensordict`
and `verl.utils.device` at module scope (`utils.py:21-29`), so anything added there
inherits a heavy import and cannot satisfy G6. `utils.py` will import from `io.py`,
not the other way round.

### 5.2 `Modality`

```python
Modality = Literal["text", "image", "video", "audio"]
```

A `Literal` rather than an `Enum`: these values must round-trip through
`sampling_params` dicts, `extra_args`, and `non_tensor_batch` object arrays, all of
which are plain-data channels. `"text"` is included because conditioning is text-first
and `MediaRef` is also how a future text-side reference would arrive; no output path
uses it in M1–M4.

Values match upstream's `final_output_type` vocabulary (`outputs.py:87`) minus
`"latents"`, which is a format not a modality. §9 tracks `"actions"`.

### 5.3 `MediaRef` — one conditioning input

```python
@dataclass
class MediaRef:
    """One conditioning input, carrying its own modality tag."""

    modality: Modality
    data: Any                                    # decoded PIL / ndarray / tensor
    role: str = "condition"                      # condition | reference | keyframe | identity
    meta: dict[str, Any] = field(default_factory=dict)   # fps, frame_index, ...
```

This is the tagged-union idea from `videos.py:63-94` reduced to the one thing the
in-process path needs: the kind travels with the value. `role` distinguishes
same-modality inputs that condition differently — the case that today has to become a
new kwarg or an `extra_args` key (M-3). `meta` holds per-input scalars (`fps` for a
video reference, `frame_index` for a keyframe) that are currently either lost or
promoted to top-level sampling knobs.

`data` is a decoded object, not a URL or `file_id`. Accepting a `str` would require a
resolver, which is precisely how upstream's `image_reference` ended up needing one;
§9 keeps this open rather than deciding it now. This matches what the existing
`ImageGenerationRequest.images` already receives — decoded PIL images out of the
parquet `images` column, never a reference.

Relation to `ImageGenerationRequest.images: list[Any]` (`pipelines/utils.py:47`):
that field is `[MediaRef, …]` with `modality` fixed to `"image"`, `role` fixed to
`"condition"`, and no `meta`. Its docstring already anticipates the list being
heterogeneous in count ("empty for t2i, single-element for image editing,
multi-element for multi-image conditioning"); this makes it heterogeneous in *kind*
as well, which is the part §1.9 shows is missing.

### 5.4 `PromptBundle` — the text side

```python
@dataclass
class PromptBundle:
    """Text side of one request, keyed per encoder for multi-encoder models."""

    ids: list[int]
    mask: torch.BoolTensor | None = None
    extra_ids: dict[str, list[int]] = field(default_factory=dict)
```

This exists to collapse four of the twelve `generate` kwargs (`prompt_ids`,
`prompt_mask`, `extra_prompt_ids`, and — via a second instance — `negative_prompt_ids`
with `negative_extra_prompt_ids`) into two symmetric objects. `extra_ids` keeps the
per-text-encoder tokenisation that `_tokenize_per_encoder`
(`single_turn_agent_loop.py:37-63`) already produces for multi-encoder models such as
SD3.5.

### 5.5 `MediaRequest`

```python
@dataclass
class MediaRequest:
    request_id: str
    prompt: PromptBundle
    negative_prompt: PromptBundle | None = None
    conditions: list[MediaRef] = field(default_factory=list)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    mm_processor_kwargs: dict[str, Any] | None = None
    priority: int = 0

    def multi_modal_data(self) -> dict[str, list[Any]]:
        """Group ``conditions`` by modality, matching ``_build_multi_modal_data``."""

    def to_generate_kwargs(self) -> dict[str, Any]:
        """Lower to the twelve-kwarg form, so M1 needs no server change."""
```

Twelve kwargs become seven fields, and the nine input-describing kwargs become three.
`multi_modal_data()` reproduces `_build_multi_modal_data`'s output exactly
(`vllm_omni_async_server.py:366-380`), which is what makes M1 a no-op at runtime.

The property that matters: **`conditions` is a list.** A fifth modality, a second
image with a different role, or a reference video alongside a keyframe all extend the
list. None of them touches a signature.

`MediaRequest` also subsumes `ImageGenerationRequest`
(`pipelines/utils.py:45`) field for field: `prompt` / `negative_prompt` become
`PromptBundle`s, `images` becomes the `"image"` slice of `conditions`, and `metadata`
becomes `sampling_params` plus per-`MediaRef` `meta`. A classmethod on
`ImageGenerationRequest` that projects a `MediaRequest` down to the old shape keeps
`qwen_image_edit_flow_grpo` working unchanged through M1–M3; M4 retires the
five-candidate search (`utils.py:79-85`) because there is one declared place to look.
Note the direction: the five candidates are a *decoding* concern that disappears when
the encoder stops writing the same dict twice (`_preprocess_input:506-508`).

### 5.6 `MediaOut` and `MediaOutput`

```python
@dataclass
class MediaOut:
    """One generated medium."""

    modality: Modality
    data: torch.Tensor          # image [C,H,W]; video [T,C,H,W]; audio [S] or [C,S]
    fps: float | None = None
    sample_rate: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaOutput:
    media: list[MediaOut]
    log_probs: torch.Tensor | None = None
    stop_reason: str | None = None
    num_preempted: int | None = None
    trajectory: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> MediaOut:
        """The medium the reward and the dump path act on."""
        return self.media[0]

    def get(self, modality: Modality) -> MediaOut | None:
        """The first medium of ``modality``, or None."""
```

Three properties make this worth doing:

1. **Modality is declared, never inferred** (G3). `MediaOut.modality` replaces every
   `ndim` test in §1.5. The `ndim == 4` collision cannot recur, because there is
   nothing left to disagree about.
2. **A request may emit several media** (G2). LTX-2's `(video, audio)` tuple and H3's
   joint video+audio become two `MediaOut` entries instead of a tuple plus a
   convention key. `fps` and `sample_rate` attach to the medium they describe, not to
   a flat namespace shared with prompt embeddings.
3. **`trajectory` is separated from `extra`** (G4). RL per-step data (`all_latents`,
   `all_timesteps`, `all_log_probs`, `latents_clean`, `train_timesteps`) is a named
   slot; genuinely pipeline-private keys (`img_shapes`, `condition_image_latents`)
   stay in `extra`. The rollout↔training contract stops being one flat namespace
   shared with upstream's `multimodal_output`.

`trajectory` stays a `dict[str, Any]` in M1–M3 rather than becoming a dataclass:
its key set genuinely differs by algorithm (FlowGRPO wants `all_*`, NFT wants
`latents_clean` + `train_timesteps`, DPO wants neither), and freezing that into one
type would either grow a per-algorithm union or force `Optional` on everything. What
M4 buys is that the *slot* is declared and the keys inside it are documented per
algorithm, so a mismatch is localised.

### 5.7 The two adaptation points

Only two functions know about both worlds:

- **In:** `MediaRequest.to_engine_prompt()` — or, in M1, `to_generate_kwargs()` —
  producing the `OmniCustomPrompt` + `OmniDiffusionSamplingParams` pair that
  `_preprocess_input` builds today (`vllm_omni_async_server.py:429-523`). This is
  where the 80-field upstream dataclass is populated, and the only place a pin bump
  can require an edit.
- **Out:** `MediaOutput.from_diffusion_output(final_res, ...)`, absorbing
  `_process_output`'s logic (`vllm_omni_async_server.py:546-654`), including the
  `images[0]` read, `_maybe_unbatch`, and the `custom_output` /
  `multimodal_output` merge. This is where upstream's `final_output_type`
  (`outputs.py:87`) finally gets read, as the primary source for
  `MediaOut.modality`, with the current rank heuristic kept as a documented fallback
  for pipelines that do not set it.

Confining both to `workers/rollout/` means the 11 pipeline packages, the reward
scorers, the trainer, and the tracking layer never import a vllm-omni type to answer
a question about a modality.

### 5.8 Replacing the seven inference sites

For each site in §1.5, the M2 edit:

| Site | Becomes |
| --- | --- |
| `ray_diffusion_trainer.py:309` | `out.modality == "video"` on the `MediaOut`, and the audio track read from `MediaOutput.get("audio")` instead of `batch.batch.get("audio", …)` (`:391-393`) |
| `utils/tracking.py:156` | same predicate; `fps` / `sample_rate` read from the `MediaOut` instead of from positional tuple slots — `audio = sample[3]`, `audio_sample_rate = sample[4]` (`:154-155`) |
| `http_scorer_client.py:40` | the batch dimension is handled by the caller; the scorer receives one `MediaOut` and refuses a non-image modality explicitly |
| `hpsv3_reward.py:388-410` | modality from the field; the channels-last guess (`shape[-1] in (1, 3)`) stays, since layout is genuinely not declared today (§9) |
| `genrm_ocr.py:141-162` | same |
| `unified_reward.py:70-76` | same, and its `raise` becomes reachable only for a genuinely unsupported modality |
| `ltx2 adapter:381` | unchanged in M2 — it is squeezing a batch axis, not identifying a modality; it is listed here only because it reads as one |

Note that the last row is a case where the current `ndim == 5` test is *correct* and
about batching, not modality. Auditing all seven together is the only way to tell
those apart, which is itself an argument for doing it in one pass.

### 5.9 Compatibility seam on `DiffusionOutput`

M1 adds two derived properties to the existing `DiffusionOutput`
(`replica.py:20-32`), computed from today's conventions:

```python
@property
def modality(self) -> Modality: ...     # from final_output_type, else ndim
@property
def media(self) -> list[MediaOut]: ...  # primary tensor + audio/fps from extra_fields
```

Every consumer can migrate to the declared field before anything about the wire format
changes, and `DiffusionOutput` keeps its name, its fields, and its Pydantic base. This
is what makes M1 revertible: deleting the two properties and the new module restores
the tree exactly.

### 5.10 Naming

An earlier draft of this design called these `OmniGenRequest` / `OmniGenOutput`. This
RFC moves away from both prefixes deliberately:

- **`Omni*` is taken.** In this repo it denotes the autoregressive omni-model track —
  `OmniModelBase`, `OmniRolloutPipelineBase`, `OmniAlgoConfig`, `OmniDPOLoss`. Naming
  a diffusion-track type `OmniGenRequest` would suggest membership in a track it is
  explicitly out of scope for (N4).
- **`Diffusion*Output` collides twice.** `DiffusionOutput` already names both
  `verl_omni/workers/rollout/replica.py:20` and
  `vllm_omni/diffusion/data.py:1196`, and the two coexist in the same modules today.
  Adding a third spelling of the same word is how the current confusion started.
- **`*GenerationRequest` collides too, and would collide worse.**
  `ImageGenerationRequest` already names both `verl_omni/pipelines/utils.py:45` and
  vllm-omni's HTTP model at `entrypoints/openai/protocol/images.py:33` — two unrelated
  types, one name, both importable in the same process. Calling the new type
  `MediaGenerationRequest` would put a third near-homonym next to them; `MediaRequest`
  does not read as a member of that family.

`MediaRequest` / `MediaOutput` / `MediaRef` / `MediaOut` / `PromptBundle` /
`Modality` are all unused in the tree (verified: no `class Media*`, no bare
`Modality`; the nearest neighbours are `ModalityGroupedBatchSampler` in
`utils/dataset/offline_mllm_dpo_dataset.py:767` and imagebind's imported
`ModalityType`). The names also read correctly for a repo where the payload may be an
image, a video, an audio clip, or two of them at once.

### 5.11 What this deliberately does not unify

**Do not mirror vllm-omni's per-modality HTTP split** — `ImageGenerationRequest`
(`images.py:33`) vs `VideoGenerationRequest` (`videos.py:97`). That split serves the
OpenAI Images and Videos API shapes at an HTTP boundary (§1.2). This repo has no such
boundary. Importing the split would mean maintaining two ~40-field models with 70%
overlap plus a converter between them, and buying compatibility with nothing — while
giving up the one property G1 is about, since a per-modality model cannot express
"video conditioned on an image and an audio clip" without growing the union anyway.
vllm-omni's own internal answer is a single modality-agnostic request (§1.3); that is
the layer verl-omni corresponds to. Note this is the same conclusion the in-tree
`ImageGenerationRequest` is currently stuck on from the other side: it is
modality-*specific* and therefore had to be worked around twice (§1.9 point 4) rather
than extended.

**Do not type `sampling_params`.** It is forwarded to `OmniDiffusionSamplingParams`,
an 80-field dataclass owned by the pinned vllm-omni. A parallel `MediaSamplingParams`
would become a fifth representation (§1.1) that has to track an upstream dataclass
across pin bumps, and the tracking would be silent when it drifted — exactly the
failure mode M-2 describes. Keep it a `dict`; type only what this repo owns. The
narrower win available here is to make `_preprocess_input`'s silent
`hasattr`-partition (§1.4) *warn* on unknown keys, which is a two-line change
independent of this RFC.

**Do not remove the config copy** (N6) and **do not rename the existing
`DiffusionOutput`** (N7). Both are breaking changes on different lifecycles.

### 5.12 Backward compatibility

- M1 and M2 add code and change call sites; the wire format, the `TensorDict` keys,
  and every config field are untouched. No checkpoint or config migration.
- M3 adds a `generate(request: MediaRequest)` overload **alongside** the kwarg form.
  The kwarg form is marked deprecated in its docstring and kept, so no out-of-tree
  adapter (registered via `DiffusionModelConfig.external_lib`,
  `.agents/rules/pipelines.md`) breaks on this repo's schedule.
- M4 is the only milestone that moves keys between namespaces. It ships with the
  promotion table in §5.6 and reads the old key as a fallback for one release, logging
  once per key.
- No PR in this series carries `[BREAKING]`. If one turns out to need it, that is the
  signal to stop and re-scope.

---

## 6. Milestones

Ordered by risk. Each is a separate PR with its own tests, and each is revertible
without touching the next.

**M1 — `io.py` plus derived accessors. No behaviour change.**
Add `verl_omni/pipelines/io.py` with the six types in §5.2-5.6. Add the two derived
properties to `DiffusionOutput` (§5.9) and `MediaRequest.to_generate_kwargs()`
(§5.5). Nothing calls them yet. Entirely CPU-testable.
*Title:* `[rollout, pipelines] feat: add unified media I/O types`.

**M2 — read the declared modality at all seven sites.**
Replace the `ndim` tests in §5.8. This is the milestone that fixes M-1 and is where
the real review effort belongs, because it touches the reward path. Each site keeps
its current behaviour for the modality it currently handles; the change is only in how
the modality is determined.
*Title:* `[trainer, reward] refactor: read declared modality instead of tensor rank`.

**M3 — accept the request object.**
`generate(request: MediaRequest)` as an overload. Deletes the six-kwarg relay in
`diffusion_llm_server.py:55-79` and the seven-kwarg relay in
`single_turn_agent_loop.py:104-113`. Stops `_preprocess_input:506-508` writing
`multi_modal_data` into two places. Fixes M-3.
*Title:* `[rollout] refactor: accept MediaRequest in the diffusion generate path`.

**M4 — promote the known keys.**
Move `audio`, `audio_sample_rate`, `fps` into `MediaOut`; move the per-algorithm
trajectory keys into `MediaOutput.trajectory`; leave `extra` as the escape hatch.
Retire the `image_latents` guard (`model_base.py:363-372`) in favour of a typed slot.
Fixes M-2 and M-4.
*Title:* `[rollout, trainer] refactor: promote trajectory and media keys out of extra_fields`.

**M5 — retire the five-candidate image lookup.**
With one declared place to read a condition input, collapse
`ImageGenerationRequest.from_request_payload`'s five candidates
(`pipelines/utils.py:79-85`) to one, and move `bagel_flow_grpo:285-288` and
`wan22_dance_grpo:373-381` onto it — the migration the in-tree `NOTE`
(`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`) already asked for.
This is also where the dead paths get resolved: either give
`multi_modal_data["video"]` a consumer or stop writing it, and either wire
`audio_data` to the agent loop or delete the parameter. Fixes M-5.
*Title:* `[pipelines, rollout] refactor: single condition-input lookup via MediaRef`.

M3 and M4 are worth doing only if M1's types survive contact with a second pipeline;
if they do not, M1+M2 still stand on their own and M-1 is still fixed. M5 depends on
M3 and is the smallest of the five — it deletes more than it adds.

---

## 7. Test plan

Per `docs/contributing/testing_guide.md` and `.agents/rules/testing.md`: placement
under `tests/<module>/` is the commit gate, and the `_on_cpu.py` suffix is what CI
selects on, so it is load-bearing rather than cosmetic.

**M1 — `tests/pipelines/test_unified_io_on_cpu.py`**

1. `io.py` imports with `diffusers` and `vllm_omni` absent from `sys.modules`
   (G6). This is the test that keeps the module CPU-clean.
2. `MediaRequest.multi_modal_data()` is byte-identical to
   `_build_multi_modal_data` for all eight combinations of image / video / audio
   present-or-absent, including the empty dict.
3. `MediaRequest.to_generate_kwargs()` round-trips: building a request from the
   twelve kwargs and lowering it back yields the same dict, `None`s included.
4. `DiffusionOutput.modality` on a `[C,H,W]` tensor is `"image"`, on `[T,C,H,W]` is
   `"video"`, and prefers upstream's `final_output_type` when present — including the
   case where the two disagree, which must resolve to the declared value.
5. `MediaOutput.get("audio")` returns `None` rather than raising for an image-only
   output, and `primary` raises a clear error on an empty `media` list.

**M2 — regression, not new coverage**

6. The three reward scorers with a rank-sniffing branch
   (`hpsv3_reward.py`, `genrm_ocr.py`, `unified_reward.py`) keep their existing CPU
   tests green unchanged. Any diff in those tests means M2 changed behaviour, which it
   must not.
7. A new case pinning the §1.5 collision: an image `MediaOut` and a video `MediaOut`
   with the *same* `data.ndim` are routed differently. This is the test that would
   have caught M-1, and it is only expressible once modality is a field.
8. `tests/trainer/` gains a CPU case for `_dump_samples` selecting mp4 vs jpg from the
   declared modality, with a fake `_export_video`.

**M3 / M4 / M5 — contract tests**

9. `generate(request)` and `generate(**kwargs)` produce identical
   `OmniCustomPrompt` + `OmniDiffusionSamplingParams` pairs, asserted field by field.
10. Per-algorithm trajectory-key tables (FlowGRPO / NFT / DPO) are asserted against
    what each of the eight adapters emits, so a missing key fails in CI rather than
    mid-run.
11. `tests/pipelines/test_image_edit_interface_on_cpu.py` already pins the
    five-candidate precedence order (`:47-63` asserts top-level `images` wins over
    four lower-priority keys; `:84-85` asserts the
    `additional_information.condition_images` fallback). It becomes the compatibility
    test for M3 — the projection `MediaRequest → ImageGenerationRequest` must keep it
    green unchanged — and the test that is deliberately *rewritten* in M5, since
    collapsing five candidates to one is exactly what M5 does. Rewriting it must be an
    explicit hunk in the M5 PR, never a silent deletion.
12. M5 adds a case asserting that a `MediaRef(modality="video")` in `conditions`
    reaches the adapter, closing the §1.9 write-only path. If M5 instead deletes
    `video_data`, the same case asserts the parameter is gone — the point is that one
    of the two must be true and CI records which.

**GPU**

13. One existing GPU smoke test per output modality — one image pipeline
    (`qwen_image_flow_grpo`) and one video+audio pipeline (`ltx2_flow_grpo`) — re-run
    unchanged after M2 and after M4, per `docs/contributing/gpu_smoke_tests.md`. These
    are the only checks that cover the `multimodal_output` merge with a real engine.
14. The image-conditioned path needs its own GPU check, since it is the only live
    multi-input path: one `qwen_image_edit_flow_grpo` run from
    `examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh`, re-run
    after M3 and after M5.

**Every milestone**

15. `pre-commit run --files <changed>`. Note: in the current venv the
    `autogen-trainer-cfg` hook fails with a pre-existing
    `ModuleNotFoundError: No module named 'omegaconf'` from `scripts/print_cfg.py:16`,
    unrelated to these files; all other hooks must pass.

---

## 8. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | **M2 changes a reward silently.** The reward path is the one place a wrong refactor produces a plausible number instead of an error. | M2 keeps each scorer's per-modality behaviour byte-identical and changes only how the modality is chosen; test 6 requires the existing scorer tests to pass *unchanged*, and any diff there blocks the PR. |
| R2 | **A fifth representation.** Adding types without deleting the old ones leaves five encodings instead of four. | M3 and M4 are the deletions, and they are in the plan. If M3 does not land, M1's `to_generate_kwargs()` is one function and M2 replaced inference with field reads — no new *representation*, only accessors. |
| R3 | **`trajectory` becomes the new `extra_fields`.** A `dict[str, Any]` under a nicer name. | The per-algorithm key tables in test 10 are the constraint. §5.6 states the reason it stays a dict and §9 tracks promoting it. |
| R4 | **Upstream's `final_output_type` is unreliable** — it defaults to `"text"` on the dataclass (`outputs.py:87`) but to `"image"` in the `from_diffusion` constructor (`:165-205`, default at `:178`), so a video pipeline that never sets it is mislabelled `"image"`. | §5.7 keeps the rank heuristic as an explicit documented fallback, and test 4 pins the disagreement case. Prefer-declared-else-infer is strictly better than today's infer-only. |
| R5 | **Eleven pipeline packages, one shared type.** A type that fits eight image pipelines may not fit a joint A/V one. | M1 lands with two consumers of different shape — one image pipeline and `ltx2_flow_grpo` — rather than eight of the same shape. G5 makes each milestone revertible if it does not fit. |
| R6 | **Merge conflicts with in-flight pipeline work** (RFC-0001/0002 H3, and anything touching `_process_output`). | M1 adds a file and two properties; M2 edits one line per site. Both are small and rebase cleanly. M3/M4 should follow the H3 bring-up rather than race it. |
| R7 | **The channels-last guess survives.** `shape[-1] in (1, 3)` (`hpsv3_reward.py:389`) is a second inferred property this RFC does not fix. | Acknowledged in §5.8 and §9; declaring layout is a strictly larger change and is out of scope here. |

---

## 9. Open questions

1. **`n` / `num_outputs_per_prompt` placement.** Today it lives in `sampling_params`
   and is re-derived as an explicit argument to
   `split_diffusion_output_by_request(..., num_outputs_per_prompt=)`
   (`request_batch.py:210`), where it drives the shape-coincidence slicing of §1.6.
   Should `MediaRequest` own it — making the group size declared and the slicing
   exact — or keep deferring to `sampling_params`? This turns on whether request
   batching stays a rollout-adapter concern.
2. **May `MediaRef.data` be a `str`?** Accepting a URL or `file_id` matches upstream's
   reference types (`videos.py:63-94`), but every dataset in this repo hands over
   decoded objects. Allowing both without a resolver is how upstream's
   `image_reference` ended up needing one.
3. **Should tensor layout be declared too?** `shape[-1] in (1, 3)` channels-last
   guessing appears in two scorers (§5.8). A `layout: Literal["chw", "hwc"]` field on
   `MediaOut` would close it, at the cost of every producer having to be audited. Not
   in M1–M4.
4. **`"actions"` as a fifth modality.** Upstream already emits it
   (`output_formatter.py:158-171`). Add it to `Modality` pre-emptively, or wait for a
   consumer? Adding it costs nothing; using it untested costs a false claim of
   support.
5. **Does `verl` want this at all, or only verl-omni?** The rank-sniffing pattern is
   local to the diffusion path, which is verl-omni's. Confirm before proposing
   anything upstream of this repo.
6. **RFC numbering.** RFC-0001 and RFC-0002 (MiniMax-H3) exist on unmerged branches.
   If they land in a different order, this file is renumbered; nothing references it
   by number yet.

---

## 10. Appendix — key references

All line numbers verified at verl-omni `8c0f5fa` and vllm-omni
`0.24.1.dev26+gfe478a95a`.

**verl-omni — the request path**

| What | Where |
| --- | --- |
| `generate`, 12 kwargs | `verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py:330-360` |
| `_build_multi_modal_data` | `…/vllm_omni_async_server.py:366-380` |
| `_preprocess_input` | `…/vllm_omni_async_server.py:429-523` |
| hardcoded `modalities = ["image"]` | `…/vllm_omni_async_server.py:497` |
| `hasattr`-partition of `sampling_params` | `…/vllm_omni_async_server.py:510-517` |
| `_process_output`, `_maybe_unbatch` | `…/vllm_omni_async_server.py:546-654`, `:619-628` |
| six-kwarg retry relay | `verl_omni/workers/rollout/diffusion_llm_server.py:55-79` |
| seven-kwarg agent-loop relay | `verl_omni/agent_loop/single_turn_agent_loop.py:104-113` |
| per-encoder tokenisation | `verl_omni/agent_loop/single_turn_agent_loop.py:37-63` |
| `multi_modal_data` written twice | `…/vllm_omni_async_server.py:506-508` |
| in-tree `ImageGenerationRequest` | `verl_omni/pipelines/utils.py:45`; five-candidate search `:79-85`; heavy module-scope imports `:21-29` |
| the `NOTE` deferring the generalisation | `verl_omni/pipelines/qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`; consumed at `:361-363` |
| hand-rolled condition-image lookups | `verl_omni/pipelines/bagel_flow_grpo/vllm_omni_rollout_adapter.py:285-288`, `verl_omni/pipelines/wan22_dance_grpo/vllm_omni_rollout_adapter.py:373-381` |
| live text+image dataset shape | `examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py:63-71` |
| existing five-candidate precedence test | `tests/pipelines/test_image_edit_interface_on_cpu.py:47-63`, `:84-85` |
| `additional_information` alias lookup | `verl_omni/pipelines/request_batch.py:69-80` |

**verl-omni — the output path**

| What | Where |
| --- | --- |
| `DiffusionOutput` | `verl_omni/workers/rollout/replica.py:20-32` |
| `DiffusionAgentLoopOutput` / internal | `verl_omni/agent_loop/diffusion_agent_loop.py:69-86`, `:88-102` |
| tensor-vs-object promotion | `verl_omni/agent_loop/diffusion_agent_loop.py:348-352`, `:383-388` |
| `is_video = outputs.ndim == 5` | `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:309` |
| asymmetric `audio` lookup | `verl_omni/trainer/diffusion/ray_diffusion_trainer.py:391-393` |
| `out.ndim == 4` → video | `verl_omni/utils/tracking.py:156` |
| `image.ndim == 4` → batch | `verl_omni/utils/reward_score/http_scorer_client.py:38-44` |
| 3/4/5-way + channels-last | `verl_omni/utils/reward_score/hpsv3_reward.py:388-410`, `genrm_ocr.py:141-162`, `unified_reward.py:68-77` |
| `image_latents` collision guard | `verl_omni/pipelines/model_base.py:363-372` |
| shape-coincidence slicing | `verl_omni/pipelines/request_batch.py:197-207`, `:210-254` |
| `(video, audio)` tuple output | `verl_omni/pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py:381-397` |
| `output_type` is a format, not a modality | `verl_omni/pipelines/ltx2_flow_grpo/common.py:51-53` |
| fourth config copy | `verl_omni/workers/config/diffusion/rollout.py:60-74` |

**vllm-omni (pinned)**

| What | Where |
| --- | --- |
| `ImageGenerationRequest` | `entrypoints/openai/protocol/images.py:33`, size validator `:66-79` |
| `VideoGenerationRequest` | `entrypoints/openai/protocol/videos.py:97`; `SizeStr` `:30`; `VideoParams` `:47-60`; duplicated top-level block `:135-138` |
| reference tagged unions | `entrypoints/openai/protocol/videos.py:63-94` |
| audio request | `entrypoints/openai/protocol/audio.py:321` |
| `OmniDiffusionRequest` | `diffusion/request.py:15` |
| `OmniDiffusionSamplingParams` (80 fields) | `inputs/data.py:178` |
| `OmniCustomPrompt`, `OmniPromptType` | `inputs/data.py:110-129`, `:137` |
| `OmniRequestOutput`, `final_output_type`, `images` | `outputs.py:63`, `:87`, `:91` |
| `DiffusionOutput.output` union | `diffusion/data.py:1196-1202` |
| `_build_multimodal_output` keys | `diffusion/output_formatter.py:158-171`; `final_output_type="audio"` `:225` |

**Repo rules this design is constrained by**

- `.agents/rules/pipelines.md` — registry dispatch by `(architecture, algorithm)`;
  registration is an import side effect; lazy heavy imports so adapters stay
  CPU-importable.
- `.agents/rules/code-style.md` — reuse over duplication; do not merge across a
  registry boundary; ~4% comment density; Apache 2026 header on new `.py`.
- `.agents/rules/testing.md` — `tests/<module>/` placement is the only commit gate;
  `_on_cpu.py` is what CI selects on.
- `AGENTS.md` §1 — duplicate-work checks, no pure code-agent PRs, AI-assistance
  disclosure in every PR description.

# [RFC] Upgrade `ImageGenerationRequest` to a full-modality omni interface for the diffusion rollout→train path

- **Status:** Draft
- **Scope:** the request and output objects crossing agent-loop → rollout-server → rollout-adapter
  → reward → training-adapter, for **diffusion** pipelines. **All four modalities — text + image +
  video + audio — in one shape**, in both directions.
- **Not in scope:** the AR omni track / `TokenOutput`, and any change to the pinned vllm-omni
  protocol (that ground is #391's; §2.1).
- **Applies to:** the whole diffusion tree — all 10 pipeline packages re-decide the same questions
  independently today.
- **Last updated:** 2026-08-16
- **Verified against:** verl-omni `4c77b29`, vllm-omni `0.24.1.dev26+gfe478a95a`.

---

## 0. TL;DR

`ImageGenerationRequest` (`verl_omni/pipelines/utils.py:45`) already lives at the verl-omni-side
rollout→adapter boundary, and its own call site carries the note *"only this image-edit pipeline
consumes `ImageGenerationRequest` for now; migrating the existing T2I pipelines onto it is left to
a follow-up PR"* (`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`). **This RFC is
that follow-up, widened past images to a full-modality omni interface: text + image + video +
audio, in one shape, in both directions.**

Four things this fixes, all shipping today:

1. **Generated modality is nowhere in the repo.** Seven sites sniff tensor rank — five to decide
   *which modality*, two *is this batched* — and they collide: `ndim == 4` means *video* in
   `utils/tracking.py:156` and *a batch of images* in `http_scorer_client.py:40`, which keeps
   frame 0 and silently discards the rest (§1.1).
2. **The verl-omni-side input contract is `dict[str, Any]` with five aliases.**
   `ImageGenerationRequest` today scans **five** candidate keys for one condition image
   (`utils.py:79-83`) — three of the five have no producer; one exists only because
   `_preprocess_input:506-508` writes the dict twice. Video is write-only. Audio is unreachable
   (§1.2).
3. **The input *protocol* is undeclared too — and that costs a whole class.** Nothing says how a
   pipeline wants text rendered, so LTX-2 subclasses the agent loop for two overrides unrelated to
   media (§1.3); nothing says an input may need more than one representation, so Qwen-Image-Edit's
   VAE reads two fields (`vae_images`, `vae_image_sizes`) that **no code in the repo can write**.
4. **The pattern is live.** PR #373 pays it again — one uint8 transport decision spread across ten
   modules (§2.2).

**Layer:** above vllm-omni's engine, below training-side DataProto. Ground verl-omni owns.

**Composition with #391 (Stable vLLM-Omni Rollout Interface):** orthogonal, complementary. #391
refactors engine internals (scheduler injection, `TrajectoryCollector`, `output_type` enum +
`custom_output` re-add, `OmniCustomPrompt.extra_prompt_ids` + `multi_modal_data`). This RFC types
the verl-omni-side boundary that consumes those seams. Naming aligns with upstream where they touch
(§2.1). Author of #391 accepted the alignment in principle: *"Good idea!"*
([#391 comment 2026-08-16](https://github.com/verl-project/verl-omni/issues/391)).

**Proposal:** one additive, CPU-importable module `verl_omni/pipelines/io.py`:

```
OmniMediaRequest    ← input:  prompt (PromptBundle) + conditions[MediaRef] + sampling
OmniMediaOutput     ← output: media[OmniMediaOut] + trajectory + extra
```

Modality becomes **declared** data, conditioning inputs become a **typed list** instead of five
candidate keys, the RL trajectory gets a first-class slot separate from the pipeline-private escape
hatch, and the input protocol each pipeline needs becomes a declaration next to its consumer rather
than a subclass of the shared loop.

Six milestones, cheapest-risk first. **M1+M2 carry most of the value and are revertible.**

---

## 1. Evidence

### 1.1 Output side: modality inferred from tensor rank

```python
class DiffusionOutput(BaseModel):      # verl_omni/workers/rollout/replica.py:20-32
    diffusion_output: Any              # "image tensor (CHW) / video tensor (TCHW)"
    log_probs: Optional[Any] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
    extra_fields: dict[str, Any] = {}
```

No modality field, so **seven** sites sniff tensor rank: five ask *"which modality is this?"*, two
ask *"is this batched?"* — a legitimate use of rank, indistinguishable from the first at the call
site. Last column is the M2 edit (§5):

| Site | Test | Meaning | Becomes |
| --- | --- | --- | --- |
| `trainer/diffusion/ray_diffusion_trainer.py:309` | `ndim == 5` | video→mp4 else jpg | `out.modality == "video"`; audio track from `OmniMediaOutput.get("audio")` instead of the asymmetric `batch.batch.get("audio", …)` (`:391-393`) |
| `utils/tracking.py:156` | `ndim == 4` | **video** → `wandb.Video` | same predicate; `fps` / `sample_rate` from `OmniMediaOut` instead of positional tuple slots (`:154-155`) |
| `utils/reward_score/http_scorer_client.py:40` | `ndim == 4` | **a batch** → keep `[0]` | caller handles batching; the scorer takes one `OmniMediaOut` and refuses a non-image modality explicitly |
| `hpsv3_reward.py:388-410` | 3/4/5 + `shape[-1] in (1,3)` | image / video / batched video | modality from field; channels-last guess stays (§8 q5) |
| `genrm_ocr.py:141-162` | same | same | same |
| `unified_reward.py:70-76` | 3/4 only, `raise` otherwise | image / video | same; its `raise` becomes reachable only for a genuinely unsupported modality |
| `ltx2_flow_grpo/vllm_omni_rollout_adapter.py:381` | `ndim == 5` | leading batch axis to squeeze | **unchanged** — batching, not modality |

`ndim == 4` therefore carries opposite meanings in two files: point a video pipeline at the HTTP
scorer and the reward is computed on frame 0 of each clip **without any error**.

Upstream admits the union honestly — `DiffusionOutput.output` is typed `torch.Tensor |
tuple[Any, ...] | dict[str, Any] | None` (`vllm_omni/diffusion/data.py:1202`), and LTX-2 uses the
tuple arm: `output.output = (video[0], audio)` (`ltx2_flow_grpo:383`). verl-omni flattens it to
`Any`.

Everything that is *not* the primary tensor rides in `extra_fields`, assembled by `_process_output`
(`vllm_omni_async_server.py:610-635`) from each adapter's `custom_output` unioned with upstream's
`multimodal_output`. Eight of nine rollout adapters emit `custom_output`; the key sets overlap
without agreeing, and disagreement does not follow the algorithm boundary. Four consequences, all
present in the tree:

1. **A runtime guard exists to catch a key collision.** `DiffusionI2IModelBase.inject_condition`
   raises if `model_inputs` holds `image_latents` — a name **reserved by the MFU FLOPs counter** —
   saying *"the rollout adapter likely output 'image_latents' instead of
   'condition_image_latents'"* (`pipelines/model_base.py:355-371`). Two undeclared contracts collide
   in one dict.
2. **Batch slicing decides by shape coincidence.** `_slice_batch_value` slices iff
   `value.shape[0] == req.num_reqs * num_outputs_per_prompt` (`request_batch.py:197-207`). Per-sample
   vs shared is inferred from a number.
3. **The container is not statically knowable.** Tensor-valued extras go to `TensorDict`, everything
   else to `non_tensor_batch`, by `isinstance` (`diffusion_agent_loop.py:348-352`, `:383-388`). The
   trainer reads `batch.batch` first for `audio` but `non_tensor_batch` first for
   `audio_sample_rate` (`ray_diffusion_trainer.py:391-393`).
4. **Merge order silently picks the winner.** `extra_fields.setdefault(key, ...)` folds
   `multimodal_output` in **under** `custom_output` — on a collision the adapter wins and upstream's
   value is dropped without a warning. `audio_sample_rate` has two producers already
   (`ltx2_flow_grpo:397` + upstream's `_build_multimodal_output`); `audio` and `fps` have no in-repo
   producer, so an upstream rename fails at **training** time with a `KeyError` — or yields `None`
   and silently drops the audio track from every dumped mp4.

### 1.2 Input side: `ImageGenerationRequest` scans a dict with five aliases

This is the type the RFC generalises. It exists today, and its shape tells the story:

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

**One modality** (image). **One call site**, carrying the *"migrating the existing T2I pipelines
onto it is left to a follow-up PR"* deferral note. **Image-only even for that one call site**:
`from_request_payload` (`utils.py:63-90`) scans **five** candidate keys for one condition image —
`images`, `image`, `multi_modal_data.image`, `extra_args.multi_modal_data.image`,
`additional_information.condition_images` — first-wins.

Why five? Because the input contract into `custom_prompt` is a `dict[str, Any]`. Of the five:

- **Three have no producer** anywhere on the diffusion rollout path.
- One (`extra_args.multi_modal_data.image`) exists only because `_preprocess_input:506-508` writes
  the same dict into two places.
- Only `multi_modal_data.image` is genuinely live.

And even that one isn't uniform: `wan22_dance_grpo:373-381` reads `multi_modal_data["image"]`
directly (hand-rolled lookup), and `bagel_flow_grpo:285-288` *copies* `extra_args.multi_modal_data`
up into `custom_prompt` — a workaround existing only because `:506-508` wrote the dict into two
places.

**`additional_information` is unwritable by construction.** `_preprocess_input` writes a **closed
set of seven keys** into `custom_prompt` (`vllm_omni_async_server.py:494-508`), reached through a
`generate()` call passing **seven fixed keyword arguments** (`single_turn_agent_loop.py:104-113`);
`additional_information` is on neither list. Yet `qwen_image_edit_flow_grpo` reads `vae_images` /
`vae_image_sizes` out of it (`:370-372`) and raises `ValueError("Qwen-Image-Edit requires non-empty
additional_information['vae_image_sizes']")` when they are absent (`:73-74`) — **live consumers, no
possible producer** (§1.3 shows what those fields were reaching for).

The other three modalities have never had their `ImageGenerationRequest` equivalent:

| Modality in | Reaches the engine prompt? | Consumer? | Status |
| --- | --- | --- | --- |
| text | yes | all 10 pipelines | **live** |
| image | yes, written twice (`:506-508`) | 3 pipelines, 2 lookup styles + 1 copy-up workaround | **live** |
| video | yes — `multi_modal_data["video"]` (`:377`) | **none** — `:377` is the only occurrence in the tree | **write-only** |
| audio | only if a caller passes `audio_data` | **none** | **unreachable** |

A contributor reading `generate`'s signature sees four conditioning modalities; one is silently
discarded, one is unreachable, and nothing in the type system says which.

### 1.3 The input protocol needs a declaration too

§1.1-§1.2 are about *what* travels; this is about **how the shared producer should build it for a
given pipeline** being undeclared. Already paid for with whole classes.

**Working precedent.** SD3.5 needs two text encoders with two tokenizers and **no agent loop of its
own** — the requirement is declared as data on the CLI:

```bash
# examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh:56
actor_rollout_ref.model.extra_tokenizers='{clip: ..., t5: ...}'
```

`DiffusionModelConfig.__post_init__` resolves that into `extra_tokenizer_map`
(`workers/config/diffusion/model.py:69,74,173-186`), the base loop reads it
(`diffusion_agent_loop.py:242`) and `_tokenize_per_encoder` (`single_turn_agent_loop.py:37-63`)
emits one `extra_prompt_ids[key]` per declared tokenizer without knowing anything about SD3.5.
**This is the shape the rest of this section argues for.** Its one weakness: key names are
hardcoded (`SD3_CLIP_TOKENS_KEY = "clip"` at `sd3_flow_grpo/common.py:23-29`, indexed at `:421`) —
the contract is a shell string matching a Python constant, checked at neither end.

**How the text is *rendered* is not part of the declaration at all, and that costs a class.**
LTX-2 needs raw text, and has no way to say so, so it subclasses the shared loop:

```python
@register("ltx2_diffusion_single_turn_agent")   # pipelines/ltx2_flow_grpo/agent_loop.py:49
class LTX2DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    def __init__(self, ...): ...                # :53-80  works around AgentLoopBase's chat-template probe
    def apply_chat_template(self, messages, ...): ...  # :82-106  raw text
```

Exactly **two** overrides, neither about media. Costs: a registry key (`:49`), an export
(`ltx2_flow_grpo/__init__.py:18,39`), ten transcribed `__init__` lines, and a `default_agent_loop`
line in each of two run scripts. The next raw-text pipeline pays it again.

**And nothing says an input may need more than one representation.** Qwen-Image-Edit feeds **one
source image to two consumers at two different geometries**:

| Consumer | Wants | Why | Site |
| --- | --- | --- | --- |
| Qwen2.5-VL text encoder | processor's patch grid + `image_grid_thw` | grid length must match the `<\|image_pad\|>` span | `_get_qwen_prompt_embeds:135-137` |
| VAE | image at **source** resolution + original size | condition latents concat with noise latents: `torch.cat([latents, condition_image_latents], dim=1)` (`:255`) | `prepare_latents(vae_images, …)` `~:447-448` |

The shared loop decodes once, keeps the resized view, drops the source on the next line
(`single_turn_agent_loop.py:77-82`). The resize is not configurable away: upstream's
`_process_multi_modal_info` has a single branch, comment calling the work *"synchronous PNG decode
+ smart_resize (CPU-heavy)"* (`verl/utils/dataset/rl_dataset.py:445-446, :479-500`). **The two
consumers do not want one shared correct size; they want different sizes.**

The dataset ships `raw_prompt` un-decoded (`rl_dataset.py:389`) — the source is available where the
loop runs. `vae_images` / `vae_image_sizes` were reaching for exactly this slot, and a key cannot
supply it: §1.2's closed seven-key set has no room, and even with room, a dict entry does not say
what geometry the value is in.

---

## 2. Composition with in-flight upstream work

### 2.1 #391 — Stable vLLM-Omni Rollout Interface (engine-side, orthogonal)

**#391 owns:** the vllm-omni-side seams. Four of them:

1. Scheduler injection (`build_pipeline_scheduler`, replacing 5 hardcoded `from_pretrained` sites).
2. `TrajectoryCollector` protocol on the scheduler (`configure()` / `reset()` / `get_trajectory()`).
3. `output_type` enum + re-add `custom_output` to `DiffusionOutput`, plumbed through
   `format_diffusion_outputs → from_diffusion` + IPC.
4. `OmniCustomPrompt.extra_prompt_ids: dict[str, list[int]]` + masks + `multi_modal_data`; rename
   `prompt_token_ids` → `prompt_ids`.

Goal: verl-omni rollout adapters stop rewriting `forward` — ~900-line adapters shrink to ~100 lines.

**This RFC owns:** the verl-omni-side types that consume those seams (`OmniMediaRequest` on the way
in, `OmniMediaOutput` on the way out), plus the declared input protocol (§5.3 G7) that lives
entirely on the adapter and doesn't touch upstream.

**Three places where the two compose:**

- **Naming aligned.** `PromptBundle`'s field names match `OmniCustomPrompt`: `prompt_ids`,
  `extra_prompt_ids`, `prompt_mask`, `negative_prompt_ids`, `negative_prompt_mask`. When seam 4
  lands, `PromptBundle` becomes a straight passthrough to upstream — same names, no translation.
- **Trajectory reads upstream.** M4 ("promote known keys", §6) reads `DiffusionOutput.trajectory_*`
  directly after seam 2/3 land — no `custom_output` shim on the verl-omni side.
- **`conditions` list ↔ `multi_modal_data` dict.** After seam 4,
  `OmniMediaRequest.multi_modal_data()` projects `conditions` (list) to upstream's `multi_modal_data`
  (dict-of-lists by modality). What upstream doesn't need to type — `MediaRef.role` / `view` /
  `source` — stays on our side as a companion field (§8 q6: whether to push those upstream is open).

**Three places this RFC still has ground even with #391 fully landed:**

- **The 5 downstream sites (trainer, tracker, three scorers) never read
  `OmniRequestOutput.final_output_type`**, and can't: `async_omni_engine.py:1000` only picks
  `"audio"` or `"image"`, never `"video"`. So even with #391's engine-side `output_type` enum
  landed, `ndim == 4` still means *video* in `tracking.py:156` vs *a batch of images* in
  `http_scorer_client.py:40`. `OmniMediaOut.modality` at the rollout-output boundary is what
  actually removes the rank-sniff.
- **`OmniCustomPrompt.multi_modal_data` is a dict-of-lists by modality**, so it cannot express "two
  views of the same source" — Qwen-Image-Edit's VL-grid + VAE-native case (§1.3), what `MediaRef.role`
  + `view` handles.
- **The declared input protocol (`prompt_render` / `input_views`)** retires LTX-2's
  `apply_chat_template` override + registry key + export + two run-script lines, and lets
  `vae_images` / `vae_image_sizes` be deleted rather than raised-on. Entirely verl-omni side.

**Acknowledgement.** #391 author replied *"Good idea!"* to the proposal to generalise
`ImageGenerationRequest` this way
([#391 comment 2026-08-16](https://github.com/verl-project/verl-omni/issues/391)). No design
commitment; direction is aligned.

### 2.2 #373 — uint8 response transport (subsumed by `OmniMediaOut.dtype`)

PR #373 adds `actor_rollout_ref.rollout.response_transport_dtype` ∈ {`float32`, `uint8`} to shrink
the rollout→train payload (`dispatch_lazy_compute_data_proto`: 5.1 s → 1.188 s once fields
`update_actor` never reads are dropped). One dtype decision with no declared metadata costs §1.1's
fan-out again:

- `visual_tensor_to_uint8()` lands in `utils/reward_score/reward_utils.py` and is imported by
  **10 modules** — four scorers, `jpeg_compressibility.py`, `http_scorer_client.py`,
  `utils/tracking.py`, both trainer entry points, and `workers/rollout/vllm_rollout/
  vllm_omni_async_server.py`: the **rollout server imports a reward util** for want of a shared
  media-IO module.
- Four scorers gain an `if image.dtype != np.uint8:` branch *inside functions that already sniff
  rank*, and `torch.empty(0, dtype=torch.uint8 if … else torch.float32)` re-derives the decision on
  the empty path.
- `_diffusion_output_type(sampling_params)` in the async server is a **second implementation** of
  `sd3_flow_grpo/vllm_omni_rollout_adapter.py:137-142`.
- `http_scorer_client.py:40`'s frame-0 discard (M-1) survives the PR untouched.

`dtype` / `value_range` on `OmniMediaOut` (§5.2) make quantisation one producer-side declaration
every consumer reads. The deeper fix — per-worker `DataProto` field projection — is q4.

---

## 3. Motivation

**10 diffusion pipeline packages, 10 registered `(architecture, algorithm)` keys, 9 rollout
adapters.** Each re-decides the same four questions — how to pass conditioning, how to say what was
emitted, where the per-step trajectory goes, how to get the shared loop to produce text and images
the way its encoders need — because no type answers them. Result: **3 representations of one
request** (the untyped one being ours), **7 sites sniffing rank, 5 to answer "which modality"**,
**5 candidate locations for one conditioning image, 3 with no producer**, and **2 fields with live
consumers no code can write**.

Seven failure modes:

- **M-1: wrong rewards, no error.** `ndim == 4` collision (§1.1). Video pipeline + `http_scorer`
  scores frame 0 and reports a plausible number.
- **M-2: pin bumps break at training time, not request time.** String keys unioned by `setdefault`;
  adapter silently wins a collision (§1.1). `audio_sample_rate` has two producers; `audio` and `fps`
  have no in-repo producer.
- **M-3: adding a modality costs three signature edits.** A second conditioning image with a
  different role has nowhere to go but a new `*_data` kwarg through three signatures, or an untyped
  `extra_args` key — the route `bagel_flow_grpo:278-282` already takes for `negative_prompt`.
- **M-4: a model emitting two modalities has no representation.** LTX-2 already does, via the tuple
  arm of upstream's union plus an `audio_sample_rate` key.
- **M-5: one conditioning image is looked for in five places** (§1.2), while video is write-only and
  audio is unreachable.
- **M-6: one boolean about text rendering costs a whole agent-loop subclass** (§1.3).
- **M-7: an input has one representation, and two consumers may need two.** `vae_images` /
  `vae_image_sizes` are live consumers with no possible producer (§1.3) — a `ValueError` on the only
  path that reads it.

None of these is model- or algorithm-specific; they are properties of the shared path.

---

## 4. Goals / Non-goals

**Goals**

- **G1.** One request type covering text + image + video + audio conditioning; adding a modality or
  a second input with a different role requires **no signature change** — by generalising the
  existing `ImageGenerationRequest`, not adding a second partial answer beside it.
- **G1b.** One place to read a conditioning input, replacing the five candidates, the hand-rolled
  lookup, and the copy-up workaround. Resolve the two dead paths (write-only video, unreachable
  audio) one way or the other.
- **G2.** One output type; generated modality is **declared data**; a request may declare more than
  one output medium.
- **G3.** **No site infers modality where a declaration exists.** Not "zero rank checks": rank stays
  at the two sites that legitimately test batching. The five that answer *"which modality"* read
  `OmniMediaOut.modality`.
- **G4.** A first-class slot for the RL trajectory, so a rollout↔training key mismatch is
  construction-time, not a `KeyError` mid-training.
- **G5.** Fully additive: every milestone landable and revertible alone, and **no milestone may
  require editing all 10 diffusion pipeline packages at once.**
- **G6.** CPU-testable — importable without diffusers, vllm-omni, or a GPU.
- **G7.** The **input protocol becomes declared data on the consumer**, read by the shared loop:
  how a pipeline wants text rendered (chat template vs raw text) and which views of a conditioning
  input it needs. Generalises `extra_tokenizer_map`. It is what makes M-7 fixable at all:
  `OmniMediaRequest` supplies the *slot*, G7 says the second view is wanted. Bounded by **N8**.
- **G8.** **Naming aligns with #391 where the two touch upstream** (`PromptBundle` field names match
  `OmniCustomPrompt`; `to_omni_custom_prompt()` projects on down). Where they don't touch (typed
  outputs, declared input protocol) the verl-omni-side naming is chosen for clarity, not for
  upstream mirroring.

**Non-goals**

- **N1.** Changing the vllm-omni protocol or engine types. This RFC adapts at the boundary and
  never forks them.
- **N2.** Mirroring upstream's HTTP `ImageGenerationRequest` / `VideoGenerationRequest` split — an
  API boundary this repo does not have.
- **N3.** Typing `sampling_params`. A parallel `OmniMediaSamplingParams` would be a fifth
  representation silently tracking upstream's 80-field dataclass — exactly M-2. Type only what this
  repo owns.
- **N4.** Touching the AR omni track or `TokenOutput`. This is why the type family is named
  `OmniMedia*`, not bare `Omni*` — the `Omni*` prefix in this tree is the AR track's
  (`OmniModelBase`, `OmniAlgoConfig`).
- **N5.** Redesigning `compute_score_*` signatures. Making the scorers read a declared modality is
  in scope (G3); their signatures are not.
- **N6.** Folding `DiffusionPipelineConfig`'s sampling knobs into the request type.
- **N7.** Renaming the two existing `DiffusionOutput` classes.
- **N8.** Deleting `LTX2DiffusionSingleTurnAgentLoop`, or forking upstream verl to make that
  possible. G7 retires its `apply_chat_template` override and its registry key — the reasons a
  *reader* has to care. Its `__init__` override survives: what it works around is
  `AgentLoopBase.__init__`'s chat-template probe (`verl/experimental/agent_loop/agent_loop.py:239-270`),
  and N1's rule — never fork the pin — applies to `verl` too. Upstream ask filed as q3.
- **N9.** Duplicating #391's scope. This RFC does not add a scheduler registry, does not modify
  `custom_output` plumbing on the engine side, does not touch `OmniCustomPrompt`'s schema. Those
  four seams (§2.1) are #391's. This RFC types the boundary above them.

---

## 5. Design

Two adaptation points against pinned upstream, both in `workers/rollout/`. Everything either side
speaks the repo's own types — the 10 pipeline packages, the scorers, the trainer and the tracking
layer never import a vllm-omni type to answer a modality question.

### 5.1 Input: `OmniMediaRequest` (upgrading `ImageGenerationRequest`)

```python
Modality = Literal["text", "image", "video", "audio"]


@dataclass
class MediaRef:
    """One conditioning input, carrying its own modality tag.

    Generalises ``ImageGenerationRequest.images`` (image-only, list of Any)
    to all four modalities plus explicit role and view.
    """

    modality: Modality
    data: Any                    # decoded, as ImageGenerationRequest.images today
    role: str = "condition"      # condition | reference | keyframe | identity — the M-3 case
    view: str = "native"         # geometry of `data`; two refs may share source+role and differ
                                 # only here: "vl_grid" for the VL text encoder, "native" for the
                                 # VAE — the M-7 case (§1.3)
    source: Any | None = None    # un-decoded origin from raw_prompt, so a consumer can derive a
                                 # view the loop did not materialise (q2)
    meta: dict[str, Any] = field(default_factory=dict)   # fps, frame_index, size — replaces
                                                         # vae_image_sizes


@dataclass
class PromptBundle:
    """Text side of one request. Field names match ``OmniCustomPrompt``
    (`vllm_omni/inputs/data.py:110-129`) so that once #391 seam 4 lands
    (`extra_prompt_ids` + `prompt_ids` rename), ``PromptBundle`` is a
    straight passthrough — no translation layer.
    """

    prompt_ids: list[int]                                            # matches OmniCustomPrompt
    prompt: str | None = None                                        # required by four in-tree
                                                                     # paths — bagel decodes ids
                                                                     # back to text (:273-276),
                                                                     # bagel smuggles a text
                                                                     # negative_prompt through
                                                                     # extra_args (:280-282),
                                                                     # qwen_image_edit synthesizes
                                                                     # prompt="" (:359), wan22 keys
                                                                     # warmup off prompt == "dummy
                                                                     # run" (:384)
    prompt_mask: torch.BoolTensor | None = None                      # matches OmniCustomPrompt
    negative_prompt_ids: list[int] | None = None                     # matches OmniCustomPrompt
    negative_prompt_mask: torch.BoolTensor | None = None             # matches OmniCustomPrompt
    extra_prompt_ids: dict[str, list[int]] = field(default_factory=dict)   # per-encoder;
                                                                           # matches #391 seam 4
    render: Literal["chat_template", "raw_text"] = "chat_template"   # verl-omni side only (§5.3)


@dataclass
class OmniMediaRequest:
    """Full-modality omni request; generalises ImageGenerationRequest past images.

    Sits at the verl-omni-side rollout→adapter boundary. Lowers to today's
    twelve-kwarg ``generate()`` form via ``to_generate_kwargs()``, and to
    upstream's ``OmniCustomPrompt`` via ``to_omni_custom_prompt()`` once
    #391 seam 4 lands.
    """

    request_id: str
    prompt: PromptBundle
    conditions: list[MediaRef] = field(default_factory=list)
    sampling_params: dict[str, Any] = field(default_factory=dict)    # left untyped by N3
    mm_processor_kwargs: dict[str, Any] | None = None
    priority: int = 0

    @classmethod
    def from_image_generation_request(cls, req: ImageGenerationRequest) -> "OmniMediaRequest":
        """Adopt the existing image-only type into the omni shape; used until
        every T2I/I2I call site migrates. Deleted at end of M5."""

    def multi_modal_data(self) -> dict[str, list[Any]]:
        """Group ``conditions`` by modality — byte-identical to today's
        ``_build_multi_modal_data`` for M1 revert-parity."""

    def to_generate_kwargs(self) -> dict[str, Any]:
        """Lower to today's twelve-kwarg form so M1 needs no server change."""

    def to_omni_custom_prompt(self) -> "OmniCustomPrompt":
        """Lower to upstream's ``OmniCustomPrompt`` once #391 seam 4 lands.
        Fields align by name; ``conditions.group_by_modality()`` projects to
        ``multi_modal_data``."""
```

**Property that matters: `conditions` is a list.** A fifth modality, a second image with a
different role, a reference video alongside a keyframe — each extends the list; none touches a
signature. M-7 is expressible: one source at two geometries is two `MediaRef` entries sharing
`source` and `role`, differing in `view`; `vae_image_sizes` becomes `meta["size"]` on the `native`
entry, present rather than raised-on.

**Subsumes `ImageGenerationRequest` field for field.** `images` becomes the `"image"` slice of
`conditions`; `metadata` becomes `sampling_params` plus per-`MediaRef` `meta`. The
`from_image_generation_request` classmethod keeps `qwen_image_edit_flow_grpo` working unchanged
through M1–M4; deleted at end of M5 when the last caller has migrated.

**Naming.** `Omni*` in this tree is the AR track prefix (`OmniModelBase`, `OmniAlgoConfig`), so
bare `Omni*` on a diffusion type would be misleading. `OmniMedia*` explicitly scopes the type to
multi-modal media I/O (this RFC's whole subject), avoids the AR-track collision, and reads
symmetrically to upstream's `OmniDiffusion*` engine types without claiming to be them. `MediaRef` /
`MediaOut` / `PromptBundle` are field-level and stay unprefixed. `OmniMediaRequest` /
`OmniMediaOutput` / `OmniMediaOut` / `MediaRef` / `PromptBundle` / `Modality` are all unused in the
tree today (verified with `git grep`).

### 5.2 Output: `OmniMediaOutput`

```python
@dataclass
class OmniMediaOut:
    """One generated modality, carrying its own tag."""

    modality: Modality           # declared, never inferred — replaces every ndim test in §1.1
    data: torch.Tensor           # image [C,H,W]; video [T,C,H,W]; audio [S] or [C,S]
    dtype: str = "float32"       # transport dtype, declared rather than sniffed (§2.2)
    value_range: tuple[float, float] = (0.0, 1.0)   # the pair a consumer needs with `dtype`
    fps: float | None = None     # attached to the medium it describes


@dataclass
class OmniMediaOutput:
    """Full-modality omni output; generalises DiffusionOutput past a single Any tensor.

    ``media`` — one or more OmniMediaOut, so LTX-2's (video, audio) fits
                without the tuple arm of upstream's union.
    ``trajectory`` — per-algorithm RL keys in a first-class slot. Once #391
                     seam 2/3 land, this slot consumes ``DiffusionOutput.trajectory_*``
                     directly rather than shimming ``custom_output``.
    ``extra`` — pipeline-private escape hatch; no cross-pipeline consumer allowed.
    """

    media: list[OmniMediaOut]                                # >1 modality (LTX-2 today)
    trajectory: dict[str, torch.Tensor]                      # per-algorithm keys, one owner
    extra: dict[str, Any] = field(default_factory=dict)      # escape hatch
```

`dtype` and `value_range` on `OmniMediaOut` exist because #373 (§2.2) must spread one transport
decision across ten modules for want of them: a consumer reads the declared pair instead of testing
`image.dtype != np.uint8`.

M1 adds two derived properties to the existing `DiffusionOutput` (`replica.py:20-32`), computed
from today's conventions — `modality` (adapter-declared where available, else `ndim`) and `media`
(the primary tensor plus audio/fps out of `extra_fields`). Every consumer migrates to the declared
field before the wire format changes. This is what makes M1 revertible: deleting the two properties
and the new module restores the tree exactly.

### 5.3 The declared input protocol (G7)

`OmniMediaRequest` gives the second view a slot; this fills it. The declaration lives on the
**rollout adapter** as class attributes — same choice as the modality declaration, same reason: the
component that knows what its encoders need is the component a reader goes to when the answer looks
wrong.

```python
class QwenImageEditPlusFlowGRPO(QwenImage):
    prompt_render = "chat_template"
    input_views = {"image": ("vl_grid", "native")}   # VL text encoder + VAE (§1.3)


class LTX23PipelineWithLogProb(...):
    prompt_render = "raw_text"                        # replaces the apply_chat_template override
    input_views = {}                                  # text-only
```

`DiffusionSingleTurnAgentLoop` reads both: renders the prompt per `prompt_render`, emits one
`MediaRef` per (source, requested view), tagged with the geometry it was produced at. Three
consequences:

- **No dataset change.** Both views derive on the producing side from `raw_prompt`, still in scope
  at `single_turn_agent_loop.py:77` — one line before today's code discards it (`:82`). `"vl_grid"`
  is exactly what `process_vision_info` returns now, so the existing default becomes one of the two
  views. N1/N8 hold.
- **`vae_images` / `vae_image_sizes` are deleted, not given a producer.** The VAE reads the
  `native` entry's `meta["size"]`. `_validate_condition_image_sizes:65-92` validates a field the
  type guarantees is there, instead of raising because nothing could have written it.
- **A wrong declaration fails at startup.** An `input_views` entry naming a view the loop cannot
  produce, or a `prompt_render` outside the `Literal`, is a construction-time error — where §1.3's
  shell-string-to-Python-constant contract fails as a mid-rollout `KeyError`.

LTX-2 keeps its `__init__` override (N8). M6's measurable win is the `apply_chat_template` override,
the registry key, the export and the two run-script lines gone — what a contributor must read and
copy for the next raw-text encoder.

---

## 6. Milestones

Ordered by risk. Each is a separate PR with its own tests, revertible without touching the next.
`#391 seam N` dependencies are called out where they apply.

**M1 — `io.py` plus derived accessors. No behaviour change.** The types in §5.1-5.2, two
`DiffusionOutput` derived properties (`modality`, `media`), `to_generate_kwargs()`, and
`from_image_generation_request()`. Nothing calls them yet. Entirely CPU-testable.
*Depends on #391:* nothing. *Title:* `[rollout, pipelines] feat: add unified media I/O types`.

**M2 — read the declared modality at the five modality sites.** The last column of §1.1's table,
all seven rows. Fixes M-1. Each site keeps its behaviour for the modality it handles today; only
how the modality is determined changes.
*Depends on #391:* nothing. *Title:* `[trainer, reward] refactor: read declared modality instead of tensor rank`.

**M3 — accept the request object.** `generate(request: OmniMediaRequest)` as an overload. Deletes
the six-kwarg relay (`diffusion_llm_server.py:55-80`) and the seven-kwarg relay
(`single_turn_agent_loop.py:104-113`), and stops `:506-508` writing `multi_modal_data` twice. Fixes
M-3.
*Depends on #391:* nothing (lowering uses today's `custom_prompt`). *Consumes seam 4 when it lands*
(swap `to_generate_kwargs()` for `to_omni_custom_prompt()`).
*Title:* `[rollout] refactor: accept OmniMediaRequest in the diffusion generate path`.

**M4 — promote the known keys.** `audio`, `audio_sample_rate`, `fps` into `OmniMediaOut`;
per-algorithm trajectory keys into `OmniMediaOutput.trajectory`; `extra` stays the escape hatch.
Fixes M-2 and M-4. Does **not** retire the `image_latents` guard (`model_base.py:355-371`) — that
name is reserved in the training-side `model_inputs` dict, out of scope — but removes its *trigger*.
*Depends on #391:* **seam 2/3 land ⇒ `trajectory` reads `DiffusionOutput.trajectory_*` directly**;
without them, M4 keeps a small `custom_output` shim (deleted on the seam-2/3 pin bump).
*Title:* `[rollout, trainer] refactor: promote trajectory and media keys out of extra_fields`.

**M5 — retire the five-candidate lookup and delete `ImageGenerationRequest`.** Collapse
`utils.py:79-83` to one place, move `wan22_dance_grpo:373-381` onto it. `bagel_flow_grpo:285-288`
is **deleted, not migrated** — M3 stops `:506-508` writing the dict twice, so nothing left to
reconcile. Also where the dead paths get resolved: give `multi_modal_data["video"]` a consumer or
stop writing it, and wire `audio_data` to the agent loop or delete the parameter. **End of M5:
delete `ImageGenerationRequest` and `from_image_generation_request()`.** Fixes M-5.
*Depends on #391:* nothing (independent). *Title:* `[pipelines, rollout] refactor: single condition-input lookup via MediaRef`.

M3 and M4 are worth doing only if M1's types survive contact with a second pipeline; if they do
not, M1+M2 stand alone and M-1 is still fixed. M5 depends on M3.

**M6 — declare the input protocol (G7).** `prompt_render` and `input_views` as adapter class
attributes read by `DiffusionSingleTurnAgentLoop` (§5.3). Fixes M-6 and M-7. Depends on M3: a
second view of one input needs `conditions` to be a list first. **Should land as two PRs**: the
text half retires `LTX2DiffusionSingleTurnAgentLoop`'s `apply_chat_template` override, its registry
key and export, and the `default_agent_loop` line in two run scripts (its `__init__` override
stays, per N8); the image half emits the `native` view for `qwen_image_edit_flow_grpo` and
**deletes** `vae_images` / `vae_image_sizes` with the `ValueError` guarding their absence. Bounded
by N8.
*Depends on #391:* nothing — entirely verl-omni side.
*Title:* `[rollout, pipelines] feat: declare per-pipeline prompt rendering and input views`.

---

## 7. Test plan

Follow `.agents/rules/testing.md`. Each milestone:

- **M1:** CPU-only unit tests for `io.py` — type round-trips, `to_generate_kwargs()` byte-identical
  to today's twelve-kwarg form on every existing pipeline fixture
  (`tests/pipelines/test_image_edit_interface_on_cpu.py:47-63`, `:84-85`);
  `from_image_generation_request()` round-trips every `ImageGenerationRequest` fixture; the two
  derived `DiffusionOutput` properties reproduce today's `ndim`/`extra_fields` reads.
- **M2:** existing scorer tests pass **unchanged**; one new test that `http_scorer_client`
  **raises** on a declared non-image modality (M-1 fix, the only reward-path diff allowed).
- **M3-M5:** per-milestone CPU pipeline tests re-verified; adapter key sets pinned by name so a
  rename becomes a failing test, not a `KeyError` mid-training. M5 asserts
  `ImageGenerationRequest` and `from_image_generation_request` are gone from the tree.
- **M6:** adapter class-attribute declarations validate at import; the text half asserts LTX-2's
  `apply_chat_template` override, registry key, export, and the two `default_agent_loop` lines are
  gone; the image half asserts `qwen_image_edit_flow_grpo` runs without `vae_images` /
  `vae_image_sizes` and the `ValueError` guard is removed.
- **GPU:** one existing e2e run per milestone (Qwen-Image + one video pipeline for M4; LTX-2 and
  Qwen-Image-Edit for M6).
- **Every milestone:** `pre-commit run --all-files`. `autogen-trainer-cfg` fails in some venvs on
  `omegaconf` (`scripts/print_cfg.py:16`), pre-existing.

---

## 8. Risks / open questions

**Risks**

- **R1: M2 changes a reward silently.** M2 keeps each scorer byte-identical *for the modality it
  handles today* and changes only modality selection; existing scorer tests must pass unchanged. The
  one intended change is the M-1 fix.
- **R2: A fifth representation.** M3-M5 are deletions (`ImageGenerationRequest`, the 5-candidate
  lookup, the bagel copy-up) and they are in the plan. If they never land,
  `to_generate_kwargs()`/`from_image_generation_request()` are one function each; no new
  representation.
- **R3: `trajectory` becomes the new `extra_fields`.** The per-algorithm key tables in §6 M3-M5 are
  the constraint; q1 tracks promoting it.
- **R4: `final_output_type` cannot express video** on the single-stage path
  (`async_omni_engine.py:1000` only picks `"audio"` or `"image"`). §5.2 makes the adapter the source
  of truth; `final_output_type` becomes a warn-on-disagreement cross-check.
- **R5: Ten packages, one shared type.** M1 lands with two consumers of *different* shape, not
  eight of the same. G5 makes each milestone revertible.
- **R6: Two views double the conditioning payload.** `input_views` is opt-in and defaults to
  today's single view; where the second view is large, `MediaRef.source` is cheaper (q2). M6 lands
  the explicit-views form first because its cost is the visible one.
- **R7: G7 stops halfway.** M6's success is written as the artefacts that disappear (the
  `apply_chat_template` override, the registry key, `vae_images` / `vae_image_sizes` and their
  `ValueError`), so a half-landed M6 is visibly half-landed.
- **R8: #391 lands differently from its RFC.** Naming aligned in §5.1 could drift (e.g. `prompt_ids`
  reverts). Mitigation: `PromptBundle` is a plain dataclass; a rename PR is mechanical. `M3` /
  `M4` are structured so the `#391`-consuming code paths are the last to land in each milestone,
  behind a one-line lowering swap.

**Open questions**

1. **`num_outputs_per_prompt` placement.** Lives in `sampling_params`, re-derived by
   `split_diffusion_output_by_request` (`request_batch.py:210`) to drive shape-coincidence slicing.
   Should `OmniMediaRequest` own it?
2. **Second materialised view, or a source reference?** §5.3 emits both views eagerly;
   `MediaRef.source` would ship the origin and let the consumer decode. Eager keeps decode where it
   already happens; lazy halves the payload but needs the rollout worker to reach the origin.
3. **Upstream ask: `AgentLoopBase.__init__` chat-template probe as opt-out?** Only reason
   `LTX2DiffusionSingleTurnAgentLoop.__init__` survives M6
   (`verl/experimental/agent_loop/agent_loop.py:239-270`). Out of scope by N1/N8.
4. **Does `DataProto` field projection follow?** #373 defers its real fix to per-worker field
   declarations plus a projecting dispatcher (§2.2). Out of scope here (N3), but `OmniMediaOutput`'s
   separate `media` / `trajectory` / `extra` slots are its precondition.
5. **Tensor layout declaration.** `shape[-1] in (1, 3)` channels-last guessing survives in two
   scorers (§1.1). A `layout: Literal["chw", "hwc"]` on `OmniMediaOut` would close it; not in M1-M4.
6. **Push `role` / `view` / `source` upstream?** #391 seam 4 gives `OmniCustomPrompt.multi_modal_data`
   a dict-of-lists; upstream's HTTP protocol already has a tagged-union `ImageReference`
   (`entrypoints/openai/protocol/videos.py:63-94`). Extending that to
   `dict[str, list[MediaReference]]` upstream would let M-7 be fixed at the engine boundary; the
   cost is one more upstream PR after #391. Deferred pending a real second consumer with the same
   shape.

---

## 9. Upstream references

Verified at vllm-omni `0.24.1.dev26+gfe478a95a`; verl-omni citations are inline above.

| What | Where |
| --- | --- |
| `ImageGenerationRequest` (HTTP), size validator | `entrypoints/openai/protocol/images.py:33`, `:66-79` |
| `VideoGenerationRequest`, `SizeStr`, reference tagged unions | `entrypoints/openai/protocol/videos.py:97`, `:30`, `:63-94` |
| `OmniDiffusionRequest` (engine payload — distinct from this RFC's `OmniMediaRequest`) | `diffusion/request.py:14` |
| `OmniCustomPrompt` (what #391 seam 4 extends; `PromptBundle` field names track it) | `inputs/data.py:110-129` |
| `OmniDiffusionSamplingParams` (80 fields — N3, N9) | `inputs/data.py:178` |
| `OmniRequestOutput`, `final_output_type`, `images` | `outputs.py:63`, `:87`, `:91` |
| single-stage `final_output_type` — only `"audio"` or `"image"`, never `"video"` (R4) | `engine/async_omni_engine.py:1000` |
| upstream `"video"` producers — multi-stage stage configs this path never builds | `model_executor/stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29` |
| `DiffusionOutput.output` union | `diffusion/data.py:1196-1202` |
| `_build_multimodal_output` keys | `diffusion/output_formatter.py:158-171` |

§1.3 and N8 also lean on upstream `verl`: `_build_messages` leaves image elements un-decoded
(`utils/dataset/rl_dataset.py:299-311`, `:389`); `_process_multi_modal_info` is a single branch
(`:479-500`), comment calling the work *"synchronous PNG decode + smart_resize (CPU-heavy)"*
(`:445-446`); `AgentLoopBase.__init__` chat-template probe
(`experimental/agent_loop/agent_loop.py:239-270`).

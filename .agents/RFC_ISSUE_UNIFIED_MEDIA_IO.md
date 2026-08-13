# [RFC] Unified multi-input / multi-output interface for the diffusion rollout→train path

- **Status:** Draft
- **Scope:** the request and output objects crossing agent-loop → rollout-server → rollout-adapter
  → reward → training-adapter, for **diffusion** pipelines. Multimodal *conditioning inputs*
  (text + image + video + audio) and multimodal *generated outputs* (image, video, audio) are both
  in scope.
- **Not in scope:** the AR omni track, `TokenOutput`, and any change to the pinned vllm-omni
  protocol.
- **Applies to:** the whole diffusion tree — all 10 pipeline packages re-decide the same
  questions independently today.
- **Last updated:** 2026-08-13
- **Verified against:** verl-omni `8c0f5fa`, vllm-omni `0.24.1.dev26+gfe478a95a`.

---

## 0. TL;DR

**text + image conditioning already works** (§1.2). What is missing is not the capability but the
contract, in four places:

1. **The generated modality is recorded nowhere in this repo.** Seven sites sniff tensor rank —
   five to decide *which modality*, two *is this batched* — and they collide: `ndim == 4` means
   *video* in `utils/tracking.py:156` and *a batch of images* in `http_scorer_client.py:40`, which
   keeps frame 0 and silently discards the rest (§1.1).
2. **The layer this repo owns is the only one on the path with no request type** — 12 flat kwargs;
   the contract for everything else is `dict[str, Any]` merged by `setdefault`, so an adapter key
   silently wins any collision, and one past collision plus a name reserved by the MFU FLOPs counter
   are policed by a hand-written runtime guard instead of a type (§1.1).
3. **The *input protocol* is undeclared too, and that costs a whole class.** Nothing says how a
   pipeline wants text rendered, so LTX-2 subclasses the agent loop for two overrides unrelated to
   media (§1.3); and nothing says an input may need more than one representation, so
   Qwen-Image-Edit's VAE reads two fields (`vae_images`, `vae_image_sizes`) that **no code in the
   repo can write** (§1.3).
4. **The pattern is live.** PR #373 pays it again — one uint8 transport decision spread across ten
   modules, plus a second copy of an existing helper (§1.4).

**Proposal:** one additive, CPU-importable module `verl_omni/pipelines/io.py` holding `MediaRef` +
`PromptBundle` + `MediaRequest` (input) and `MediaOut` + `MediaOutput` (output). Modality becomes
**declared** data, conditioning inputs become a **list** instead of N optional kwargs, the RL
trajectory gets a slot separate from the pipeline-private escape hatch, and the input protocol each
pipeline needs becomes a declaration next to its consumer rather than a subclass of the shared loop.
`MediaRequest` generalises the existing `ImageGenerationRequest` (`pipelines/utils.py:45`) past
images — the follow-up its own call site already asked for.

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
site. Last column is the M2 edit (§4):

| Site | Test | Meaning | Becomes |
| --- | --- | --- | --- |
| `trainer/diffusion/ray_diffusion_trainer.py:309` | `ndim == 5` | video→mp4 else jpg | `out.modality == "video"`; audio track from `MediaOutput.get("audio")` instead of the asymmetric `batch.batch.get("audio", …)` (`:391-393`) |
| `utils/tracking.py:156` | `ndim == 4` | **video** → `wandb.Video` | same predicate; `fps` / `sample_rate` from `MediaOut` instead of positional tuple slots (`:154-155`) |
| `utils/reward_score/http_scorer_client.py:40` | `ndim == 4` | **a batch** → keep `[0]` | caller handles batching; the scorer takes one `MediaOut` and refuses a non-image modality explicitly |
| `hpsv3_reward.py:388-410` | 3/4/5 + `shape[-1] in (1,3)` | image / video / batched video | modality from field; the channels-last guess stays (§7 q5) |
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

### 1.2 Input side: one partial answer, deferred

**text + image conditioning already works end to end** — this RFC generalises a mechanism that
exists, is exercised by `examples/flowgrpo_trainer/qwen_image_edit/prepare_data.py:63-71`, and whose
author explicitly deferred the generalisation:

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

Its one call site carries the note *"only this image-edit pipeline consumes `ImageGenerationRequest`
for now; migrating the existing T2I pipelines onto it is left to a follow-up PR"*
(`qwen_image_edit_flow_grpo/vllm_omni_rollout_adapter.py:348-351`). **This RFC is that follow-up,
widened from images to all four modalities.** It is image-only today — no `videos` / `audios` field,
no modality tag, no `role`, so it cannot express "video conditioned on a keyframe plus a reference
clip".

The lookup itself searches five candidate locations for one value — `images`, `image`,
`multi_modal_data.image`, `extra_args.multi_modal_data.image`,
`additional_information.condition_images` (`utils.py:79-83`). **Three of the five have no producer**
anywhere on the diffusion rollout path. Of the two that are live, one exists only because
`_preprocess_input:506-508` writes the same dict into two places.

`additional_information` is not merely unwritten — it is **unwritable by construction**.
`_preprocess_input` writes a **closed set of seven keys** into `custom_prompt`
(`vllm_omni_async_server.py:494-508`), reached through a `generate()` call passing **seven fixed
keyword arguments** (`single_turn_agent_loop.py:104-113`); `additional_information` is on neither
list. Yet `qwen_image_edit_flow_grpo` reads `vae_images` / `vae_image_sizes` out of it (`:370-372`)
and raises `ValueError("Qwen-Image-Edit requires non-empty
additional_information['vae_image_sizes']")` when they are absent (`:73-74`) — **live consumers, no
possible producer** (§1.3 shows what those two fields were reaching for).

One pipeline hand-rolls the lookup — `wan22_dance_grpo:373-381` reads `multi_modal_data["image"]`
directly — and `bagel_flow_grpo:285-288` *copies* `extra_args.multi_modal_data` up into
`custom_prompt`, a workaround existing only because `:506-508` wrote the dict into two places.

Video and audio are plumbed to different depths:

| Modality in | Reaches the engine prompt? | Consumer? | Status |
| --- | --- | --- | --- |
| text | yes | all 10 pipelines | **live** |
| image | yes, written twice (`:506-508`) | `qwen_image_edit_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo` | **live**, 2 lookup styles + 1 copy-up workaround |
| video | yes — `multi_modal_data["video"]` (`:377`) | **none** — `:377` is the only occurrence in the tree | **write-only** |
| audio | only if a caller passes `audio_data` | **none** | **unreachable** |

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
**This is the shape the rest of the RFC argues for.** Its one weakness: the key names are
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

### 1.4 Same pattern, live in review: PR #373

**PR #373** (*add configurable uint8 response transport*, open) adds
`actor_rollout_ref.rollout.response_transport_dtype` ∈ {`float32`, `uint8`} to shrink the
rollout→train payload — `dispatch_lazy_compute_data_proto` costs **5.1 s**, **1.188 s** once fields
`update_actor` never reads are dropped. The goal is sound; expressing *one* dtype decision with no
declared output metadata costs §1.1's fan-out again:

- `visual_tensor_to_uint8()` lands in `utils/reward_score/reward_utils.py` and is imported by
  **10 modules** — four scorers, `jpeg_compressibility.py`, `http_scorer_client.py`,
  `utils/tracking.py`, both trainer entry points, and
  `workers/rollout/vllm_rollout/vllm_omni_async_server.py`: the **rollout server imports a reward
  util**, for want of a shared media-IO module.
- Four scorers gain an `if image.dtype != np.uint8:` branch *inside functions that already sniff
  rank*, and `torch.empty(0, dtype=torch.uint8 if … else torch.float32)` re-derives the decision on
  the empty path.
- `_diffusion_output_type(sampling_params)` in the async server is a **second implementation** of
  `sd3_flow_grpo/vllm_omni_rollout_adapter.py:137-142`.
- `http_scorer_client.py:40`'s frame-0 discard (M-1) survives the PR untouched.

`dtype` / `value_range` on `MediaOut` (§4.1) make quantisation one producer-side declaration every
consumer reads; the deeper fix #373 itself defers — per-worker field declarations — is q4.

---

## 2. Motivation

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

## 3. Goals / Non-goals

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
  `MediaOut.modality`.
- **G4.** A first-class slot for the RL trajectory, so a rollout↔training key mismatch is
  construction-time, not a `KeyError` mid-training.
- **G5.** Fully additive: every milestone landable and revertible alone, and **no milestone may
  require editing all 10 diffusion pipeline packages at once.**
- **G6.** CPU-testable — importable without diffusers, vllm-omni, or a GPU.
- **G7.** The **input protocol becomes declared data on the consumer**, read by the shared loop:
  how a pipeline wants text rendered (chat template vs raw text) and which views of a conditioning
  input it needs. Generalises `extra_tokenizer_map`. It is what makes M-7 fixable at all:
  `MediaRequest` supplies the *slot*, G7 says the second view is wanted. Bounded by **N8**.

**Non-goals**

- **N1.** Changing the vllm-omni protocol or engine types. This RFC adapts at the boundary and
  never forks them.
- **N2.** Mirroring upstream's HTTP `ImageGenerationRequest` / `VideoGenerationRequest` split — an
  API boundary this repo does not have.
- **N3.** Typing `sampling_params`. A parallel `MediaSamplingParams` would be a fifth representation
  silently tracking upstream's 80-field dataclass — exactly M-2. Type only what this repo owns.
- **N4.** Touching the AR omni track or `TokenOutput`.
- **N5.** Redesigning `compute_score_*` signatures. Making the scorers read a declared modality is
  in scope (G3); their signatures are not.
- **N6.** Folding `DiffusionPipelineConfig`'s sampling knobs into the request type.
- **N7.** Renaming the two existing `DiffusionOutput` classes.
- **N8.** Deleting `LTX2DiffusionSingleTurnAgentLoop`, or forking upstream verl to make that
  possible. G7 retires its `apply_chat_template` override and its registry key — the reasons a
  *reader* has to care. Its `__init__` override survives: what it works around is
  `AgentLoopBase.__init__`'s chat-template probe (`verl/experimental/agent_loop/agent_loop.py:239-270`),
  and N1's rule — never fork the pin — applies to `verl` too. Upstream ask filed as q3.

---

## 4. Design

Two adaptation points against pinned upstream, both in `workers/rollout/`. Everything either side
speaks the repo's own types — the 10 pipeline packages, the scorers, the trainer and the tracking
layer never import a vllm-omni type to answer a modality question.

### 4.1 The types

```python
Modality = Literal["text", "image", "video", "audio"]   # this repo's own vocabulary


@dataclass
class MediaRef:
    """One conditioning input, carrying its own modality tag."""

    modality: Modality
    data: Any                    # decoded, as ImageGenerationRequest.images
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
    """Text side of one request; collapses four of the twelve kwargs."""

    ids: list[int]
    text: str | None = None      # required: four in-tree paths need the string — bagel decodes ids
                                 # back to text (:273-276) and smuggles a text negative_prompt through
                                 # extra_args (:280-282), qwen_image_edit synthesizes prompt="" (:359),
                                 # wan22 keys warmup off prompt == "dummy run" (:384)
    mask: torch.BoolTensor | None = None
    extra_ids: dict[str, list[int]] = field(default_factory=dict)   # per-text-encoder, as
                                                                    # _tokenize_per_encoder emits
    render: Literal["chat_template", "raw_text"] = "chat_template"  # how ids were produced (§4.2)


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
        """Lower to the twelve-kwarg form so M1 needs no server change."""


@dataclass
class MediaOut:
    modality: Modality           # declared, never inferred — replaces every ndim test in §1.1
    data: torch.Tensor           # image [C,H,W]; video [T,C,H,W]; audio [S] or [C,S]
    dtype: str = "float32"       # transport dtype, declared rather than sniffed (§1.4)
    value_range: tuple[float, float] = (0.0, 1.0)   # the pair a consumer needs with dtype (§1.4)
    fps: float | None = None     # attached to the medium it describes


@dataclass
class MediaOutput:
    media: list[MediaOut]                                    # can hold >1 modality (LTX-2 today)
    trajectory: dict[str, torch.Tensor]                      # per-algorithm keys, one owner
    extra: dict[str, Any] = field(default_factory=dict)      # escape hatch, no consumer allowed
```

**Property that matters: `conditions` is a list.** A fifth modality, a second image with a
different role, a reference video alongside a keyframe — each extends the list; none touches a
signature. M-7 is expressible: one source at two geometries is two `MediaRef` entries sharing
`source` and `role`, differing in `view`; `vae_image_sizes` becomes `meta["size"]` on the `native`
entry, present rather than raised-on.

`MediaRequest` subsumes `ImageGenerationRequest` field for field. A classmethod projecting a
`MediaRequest` down to the old shape keeps `qwen_image_edit_flow_grpo` working unchanged through
M1–M3.

`dtype` and `value_range` on `MediaOut` exist because #373 (§1.4) must spread one transport decision
across ten modules for want of them: a consumer reads the declared pair instead of testing
`image.dtype != np.uint8`.

**Naming.** `Omni*` is the AR track (N4); `Diffusion*Output` already names two classes coexisting
in the same modules (`replica.py:20` and `vllm_omni/diffusion/data.py:1196`); `*GenerationRequest`
already collides (`pipelines/utils.py:45` vs `protocol/images.py:33`). `MediaRequest` / `MediaOutput`
/ `MediaRef` / `MediaOut` / `PromptBundle` / `Modality` are all unused in the tree (verified).

### 4.2 The input protocol as declared data (G7)

`MediaRequest` gives the second view a slot; this fills it. The declaration lives on the **rollout
adapter** as class attributes — same choice as the modality declaration, same reason: the component
that knows what its encoders need is the component a reader goes to when the answer looks wrong.

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
- **`vae_images` / `vae_image_sizes` are deleted, not given a producer.** The VAE reads the `native`
  entry's `meta["size"]`. `_validate_condition_image_sizes:65-92` validates a field the type
  guarantees is there, instead of raising because nothing in the repo could have written it.
- **A wrong declaration fails at startup.** An `input_views` entry naming a view the loop cannot
  produce, or a `prompt_render` outside the `Literal`, is a construction-time error — where §1.3's
  shell-string-to-Python-constant contract fails as a mid-rollout `KeyError`.

LTX-2 keeps its `__init__` override (N8). M6's measurable win is the `apply_chat_template` override,
the registry key, the export and the two run-script lines gone — what a contributor must read and
copy for the next raw-text encoder.

---

## 5. Milestones

Ordered by risk. Each is a separate PR with its own tests, revertible without touching the next.

**M1 — `io.py` plus derived accessors. No behaviour change.** The types in §4.1, two
`DiffusionOutput` derived properties (M1 adds `modality` computed from the adapter declaration where
available, else `ndim`; and `media` reading the primary tensor plus audio/fps out of
`extra_fields`), and `to_generate_kwargs()`. Nothing calls them yet. Entirely CPU-testable.
*Title:* `[rollout, pipelines] feat: add unified media I/O types`.

**M2 — read the declared modality at the five modality sites.** The last column of §1.1's table,
all seven rows. Fixes M-1, and is where review effort belongs because it touches the reward path.
Each site keeps its behaviour for the modality it handles today; only how the modality is
determined changes.
*Title:* `[trainer, reward] refactor: read declared modality instead of tensor rank`.

**M3 — accept the request object.** `generate(request: MediaRequest)` as an overload. Deletes the
six-kwarg relay (`diffusion_llm_server.py:55-80`) and the seven-kwarg relay
(`single_turn_agent_loop.py:104-113`), and stops `:506-508` writing `multi_modal_data` twice.
Fixes M-3.
*Title:* `[rollout] refactor: accept MediaRequest in the diffusion generate path`.

**M4 — promote the known keys.** `audio`, `audio_sample_rate`, `fps` into `MediaOut`; per-algorithm
trajectory keys into `MediaOutput.trajectory`; `extra` stays the escape hatch. Fixes M-2 and M-4.
Does **not** retire the `image_latents` guard (`model_base.py:355-371`) — that name is reserved in
the training-side `model_inputs` dict, out of scope — but removes its *trigger*, since the condition
latent becomes a named slot.
*Title:* `[rollout, trainer] refactor: promote trajectory and media keys out of extra_fields`.

**M5 — retire the five-candidate lookup.** Collapse `utils.py:79-83` to one place, move
`wan22_dance_grpo:373-381` onto it. `bagel_flow_grpo:285-288` is **deleted, not migrated** — M3
stops `:506-508` writing the dict twice, so nothing left to reconcile. Also where the dead paths
get resolved: give `multi_modal_data["video"]` a consumer or stop writing it, and wire `audio_data`
to the agent loop or delete the parameter. Fixes M-5.
*Title:* `[pipelines, rollout] refactor: single condition-input lookup via MediaRef`.

M3 and M4 are worth doing only if M1's types survive contact with a second pipeline; if they do
not, M1+M2 stand alone and M-1 is still fixed. M5 depends on M3.

**M6 — declare the input protocol (G7).** `prompt_render` and `input_views` as adapter class
attributes read by `DiffusionSingleTurnAgentLoop` (§4.2). Fixes M-6 and M-7. Depends on M3: a second
view of one input needs `conditions` to be a list first. **Should land as two PRs**: the text half
retires `LTX2DiffusionSingleTurnAgentLoop`'s `apply_chat_template` override, its registry key and
export, and the `default_agent_loop` line in two run scripts (its `__init__` override stays, per
N8); the image half emits the `native` view for `qwen_image_edit_flow_grpo` and **deletes**
`vae_images` / `vae_image_sizes` with the `ValueError` guarding their absence. Bounded by N8.
*Title:* `[rollout, pipelines] feat: declare per-pipeline prompt rendering and input views`.

---

## 6. Test plan

Follow `.agents/rules/testing.md`. Each milestone:

- **M1:** CPU-only unit tests for `io.py` — type round-trips, `to_generate_kwargs()` byte-identical
  to today's twelve-kwarg form on every existing pipeline fixture
  (`tests/pipelines/test_image_edit_interface_on_cpu.py:47-63`, `:84-85`), the two derived
  `DiffusionOutput` properties reproduce today's ndim/`extra_fields` reads.
- **M2:** existing scorer tests pass **unchanged**; one new test that `http_scorer_client` **raises**
  on a declared non-image modality (M-1 fix, the only reward-path diff allowed).
- **M3-M5:** per-milestone CPU pipeline tests re-verified; adapter key sets pinned by name so a
  rename becomes a failing test, not a `KeyError` mid-training.
- **M6:** adapter class-attribute declarations validate at import; the text half asserts LTX-2's
  `apply_chat_template` override, registry key, export, and the two `default_agent_loop` lines are
  gone; the image half asserts `qwen_image_edit_flow_grpo` runs without `vae_images` /
  `vae_image_sizes` and the `ValueError` guard is removed.
- **GPU:** one existing e2e run per milestone (Qwen-Image + one video pipeline for M4; LTX-2 and
  Qwen-Image-Edit for M6).
- **Every milestone:** `pre-commit run --all-files`. `autogen-trainer-cfg` fails in some venvs on
  `omegaconf` (`scripts/print_cfg.py:16`), pre-existing.

---

## 7. Risks / open questions

**Risks**

- **R1: M2 changes a reward silently.** M2 keeps each scorer byte-identical *for the modality it
  handles today* and changes only modality selection; existing scorer tests must pass unchanged. The
  one intended change is the M-1 fix.
- **R2: A fifth representation.** M3-M4 are deletions and they are in the plan. If they never land,
  `to_generate_kwargs()` is one function; no new representation.
- **R3: `trajectory` becomes the new `extra_fields`.** The per-algorithm key tables in §6 M3-M5 are
  the constraint; q1 tracks promoting it.
- **R4: `final_output_type` cannot express video** on the single-stage path
  (`async_omni_engine.py:1000` only picks `"audio"` or `"image"`). §4.1 makes the adapter the source
  of truth; `final_output_type` becomes a warn-on-disagreement cross-check.
- **R5: Ten packages, one shared type.** M1 lands with two consumers of *different* shape, not eight
  of the same. G5 makes each milestone revertible.
- **R6: Two views double the conditioning payload.** `input_views` is opt-in and defaults to today's
  single view; where the second view is large, `MediaRef.source` is cheaper (q2). M6 lands the
  explicit-views form first because its cost is the visible one.
- **R7: G7 stops halfway.** M6's success is written as the artefacts that disappear (the
  `apply_chat_template` override, the registry key, `vae_images` / `vae_image_sizes` and their
  `ValueError`), so a half-landed M6 is visibly half-landed.

**Open questions**

1. **`num_outputs_per_prompt` placement.** Lives in `sampling_params`, re-derived by
   `split_diffusion_output_by_request` (`request_batch.py:210`) to drive shape-coincidence slicing.
   Should `MediaRequest` own it?
2. **Second materialised view, or a source reference?** §4.2 emits both views eagerly;
   `MediaRef.source` would ship the origin and let the consumer decode. Eager keeps decode where it
   already happens; lazy halves the payload but needs the rollout worker to reach the origin.
3. **Upstream ask: `AgentLoopBase.__init__` chat-template probe as opt-out?** Only reason
   `LTX2DiffusionSingleTurnAgentLoop.__init__` survives M6
   (`verl/experimental/agent_loop/agent_loop.py:239-270`). Out of scope by N1/N8.
4. **Does `DataProto` field projection follow?** #373 defers its real fix to per-worker field
   declarations plus a projecting dispatcher (§1.4). Out of scope here (N3), but `MediaOutput`'s
   separate `media` / `trajectory` / `extra` slots are its precondition.
5. **Tensor layout declaration.** `shape[-1] in (1, 3)` channels-last guessing survives in two
   scorers (§1.1). A `layout: Literal["chw", "hwc"]` on `MediaOut` would close it; not in M1-M4.

---

## 8. Upstream references

Verified at vllm-omni `0.24.1.dev26+gfe478a95a`; verl-omni citations are inline above.

| What | Where |
| --- | --- |
| `ImageGenerationRequest` (HTTP), size validator | `entrypoints/openai/protocol/images.py:33`, `:66-79` |
| `VideoGenerationRequest`, `SizeStr`, reference tagged unions | `entrypoints/openai/protocol/videos.py:97`, `:30`, `:63-94` |
| `OmniDiffusionRequest` | `diffusion/request.py:14` |
| `OmniDiffusionSamplingParams` (80 fields) | `inputs/data.py:178` |
| `OmniCustomPrompt`, `OmniPromptType` | `inputs/data.py:110-129`, `:137` |
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

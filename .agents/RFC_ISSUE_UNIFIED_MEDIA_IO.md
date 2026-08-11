# [RFC] Unified multi-input / multi-output interface for the diffusion rollout→train path

> Structured against `.github/ISSUE_TEMPLATE/feature-request.yml` — the repo has no dedicated RFC
> template — following the two `nemo_automodel` design drafts that set the precedent
> (`.agents/RFC_ISSUE.md`, `.agents/RFC_ISSUE_DIFFUSION.md`, on their own branch). §0 is the
> template's *Feature request* field, §2 *Motivation*, and a *Your contribution* answer would be
> scoped from §6; §1 and §3-§10 are the technical body.

- **Status:** Draft
- **Scope:** the request and output objects crossing the agent-loop → rollout-server →
  rollout-adapter → reward → training-adapter path for **diffusion** pipelines. Multimodal
  *conditioning inputs* (text + image + video + audio) and multimodal *generated outputs*
  (image, video, audio) are both in scope.
- **Not in scope:** the autoregressive omni track, the `TokenOutput` path, and any change to
  the pinned vllm-omni protocol (§3).
- **Applies to:** the whole diffusion tree, not one model integration — all 10 pipeline packages
  re-decide the same questions independently today (§2).
- **Last updated:** 2026-08-11
- **Verified against:** verl-omni `8c0f5fa`, vllm-omni `0.24.1.dev26+gfe478a95a`. Every
  `file:line` below was read at those revisions.

---

## 0. TL;DR

**text + image conditioning already works** (§1.4). What is missing is not the capability but
the contract, in four places:

1. **The generated modality is recorded nowhere in this repo.** Seven sites sniff tensor rank —
   five to decide *which modality*, two *is this batched* — and because the two questions look
   identical in code they collide: `ndim == 4` means *video* in `utils/tracking.py:156` and *a batch
   of images* in `http_scorer_client.py:40`, which keeps frame 0 and silently discards the rest
   (§1.3).
2. **The layer this repo owns is the only one on the path with no request type** — 12 flat
   kwargs, nine describing the input, re-listed verbatim by two intermediaries (§1.1-1.2).
3. **Everything else rides in a `dict[str, Any]`** merged with upstream's by `setdefault`, so an
   adapter key silently wins a collision — `audio_sample_rate` already has two producers and no
   declared owner — and one past collision plus a name reserved by the MFU FLOPs counter are
   policed by a hand-written runtime guard instead of a type
   (`pipelines/model_base.py:355-371`) (§1.3).
4. **The *input protocol* is undeclared too, and that one costs a whole class.** Nothing says how a
   pipeline wants text rendered, so LTX-2 subclasses the agent loop for two overrides unrelated to
   media (§1.6); and nothing says an input may need more than one representation, so
   Qwen-Image-Edit's VAE — which needs the condition image at its own resolution while the VL text
   encoder needs the processor's patch grid — reads two fields (`vae_images`, `vae_image_sizes`)
   that **no code in the repo can write** (§1.7).

The pattern is not historical: PR #373, open now, pays it a fourth time — one uint8 transport
decision spread across ten modules, plus a second copy of an existing helper (§1.8).

Proposal: one additive, CPU-importable module `verl_omni/pipelines/io.py` holding `MediaRef` +
`PromptBundle` + `MediaRequest` on the way in and `MediaOut` + `MediaOutput` on the way out.
Modality becomes **declared** data, conditioning inputs become a **list** instead of N optional
kwargs, the RL trajectory gets a slot separate from the pipeline-private escape hatch, and the
input protocol each pipeline needs becomes a declaration next to its consumer rather than a
subclass of the shared loop. `MediaRequest` is the existing `ImageGenerationRequest`
(`pipelines/utils.py:45`) generalised past images — the follow-up its own call site already asked
for (§1.4).

Six milestones, cheapest-risk first. **M1+M2 carry most of the value and are revertible.**

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

Layer 1's split exists to satisfy the OpenAI Images and Videos API shapes at an HTTP boundary
verl-omni does not have (§3 N2). The one idea worth taking from it is the **tagged-union
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
nine sampling knobs feeding `sampling_params` — it never carries a prompt or a conditioning input,
so it is a static source *for* the request, not a copy *of* it (§3 N6). One note though:
`output_type` looks like a modality declaration and is not one. It defaults to `"image"` for every
pipeline including video ones and is a diffusers-style *format* selector (`"pil"` / `"pt"` / `"np"` /
`"latent"`), which is why LTX-2 rewrites it (`ltx2_flow_grpo/common.py:51-53`).

### 1.2 Input side: flat kwargs and a relay chain

`generate` (`vllm_omni_async_server.py:330-344`) takes twelve keyword arguments, nine describing
the input: `prompt_ids`, `negative_prompt_ids`, `prompt_mask`, `extra_prompt_ids`,
`negative_extra_prompt_ids`, `image_data`, `video_data`, `audio_data`, `mm_processor_kwargs`.
Consequences present in the tree:

- **Every intermediary re-lists the arguments.** `DiffusionRetryLLMServer.generate`
  (`diffusion_llm_server.py:55-80`) names six verbatim — none of which it uses, it only forwards
  them — purely to wrap the call in a retry loop; `DiffusionSingleTurnAgentLoop.run`
  (`single_turn_agent_loop.py:104-113`) restates seven. Adding a modality means editing two of
  those signatures: the retry server also takes `**kwargs`, so a new kwarg *reaches* `generate`
  through it — but the six it does name are pure transcription kept in sync by hand.
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

No modality field, so **seven** sites sniff tensor rank: five ask *"which modality is this?"*, two
ask *"is this batched?"* — a legitimate use of rank, indistinguishable from the first at the call
site, which is how the two get confused. The last column is the M2 edit (§6), so audit and fix
read as one table:

| Site | Today's test | Meaning assigned | Becomes |
| --- | --- | --- | --- |
| `trainer/diffusion/ray_diffusion_trainer.py:309` | `outputs.ndim == 5` | video → mp4, else jpg | `out.modality == "video"`; audio track from `MediaOutput.get("audio")` instead of the asymmetric `batch.batch.get("audio", …)` (`:391-393`) |
| `utils/tracking.py:156` | `out.ndim == 4` | **video** → `wandb.Video` | same predicate; `fps` / `sample_rate` from the `MediaOut` instead of positional tuple slots (`:154-155`) |
| `utils/reward_score/http_scorer_client.py:40` | `image.ndim == 4` | **a batch** → keep `[0]` | batching handled by the caller; the scorer takes one `MediaOut` and refuses a non-image modality explicitly |
| `utils/reward_score/hpsv3_reward.py:388-410` | 3/4/5-way + `shape[-1] in (1,3)` | image / video / batched video | modality from the field; the channels-last guess stays (§9 q3) |
| `utils/reward_score/genrm_ocr.py:141-162` | same | same | same |
| `utils/reward_score/unified_reward.py:70-76` | 3/4 only, `raise` otherwise | image / video | same; its `raise` becomes reachable only for a genuinely unsupported modality |
| `pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py:381` | `video.ndim == 5` | leading batch axis to squeeze | **unchanged** — this one is about batching, not modality |

`ndim == 4` therefore carries opposite meanings in two files: point a video pipeline at the HTTP
scorer and the reward is computed on frame 0 of each clip **without any error**. Both are locally
correct; the bug is that "is this a video?" has no single answer to consult — and the convention is
not stable along the path, the dump site seeing batched 5-D where the logging site sees
per-sample 4-D.

Upstream admits the union honestly — `DiffusionOutput.output` is typed `torch.Tensor |
tuple[Any, ...] | dict[str, Any] | None` (`diffusion/data.py:1202`), and LTX-2 uses the tuple
arm: `output.output = (video[0], audio)` (`ltx2_flow_grpo/vllm_omni_rollout_adapter.py:383`).
verl-omni flattens it to `Any`.

Everything that is *not* the primary tensor rides in `extra_fields`, assembled by
`_process_output` (`vllm_omni_async_server.py:610-635`) from each adapter's `custom_output` unioned with upstream's
`multimodal_output`. Eight of the nine rollout adapters emit `custom_output`
(`qwen_image_mix_grpo` none); their key sets overlap without
agreeing, and the disagreement does not follow the algorithm boundary: `qwen_image_flow_grpo` and
`wan22_dance_grpo` emit `all_latents` / `all_log_probs` / `all_timesteps` plus four prompt-embed
keys, `sd3_flow_grpo` adds the two pooled variants, but `bagel_flow_grpo` — same algorithm —
emits the three `all_*` keys and **no** prompt-embed keys at all. NFT and DPO substitute
`latents_clean` (+ `train_timesteps` for NFT); `qwen_image_edit` and `ltx2` each add their own
(`img_shapes`, `condition_image_latents`; `audio_prompt_embeds`, `audio_sample_rate`). Upstream's
`_build_multimodal_output` (`diffusion/output_formatter.py:158-171`) merges in `audio`,
`audio_sample_rate`, `fps` and `actions` on top. Four consequences, all present in the tree:

1. **A hand-written guard exists to catch a key collision.**
   `DiffusionI2IModelBase.inject_condition` raises if the `model_inputs` it is about to populate
   already holds `image_latents` — a name **reserved by the MFU FLOPs counter** for the denoised
   latent — saying *"the rollout adapter likely output 'image_latents' instead of
   'condition_image_latents'"* (`pipelines/model_base.py:355-371`). Two undeclared contracts
   collide in one dict, and both are policed at runtime because neither is declared. It inspects
   the **training-side** `model_inputs`, so §5.3's output types do not remove it (§6 M4).
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
   `extra_fields.setdefault(key, ...)` (`vllm_omni_async_server.py:630-634`), folding upstream's
   `multimodal_output` in *under* the adapter's `custom_output` — on a collision the adapter wins
   and upstream's value is dropped without a warning. Not hypothetical: `audio_sample_rate` is
   emitted both by `ltx2_flow_grpo/vllm_omni_rollout_adapter.py:397` (the vocoder's rate) and by
   `_build_multimodal_output`, with nothing declaring which is authoritative. The trainer's other
   two keys there, `audio` and `fps`, have **no** in-repo rollout producer, so a pin bump renaming
   either fails at **training** time with a `KeyError` — or yields `None` and silently drops the
   audio track from every dumped mp4.

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

This is the right idea, and why this RFC proposes a structure rather than a per-modality
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
   two places. This is not defensive coding but the cost of having no declared contract.

   `additional_information` is not merely unwritten — it is **unwritable by construction**:
   `_preprocess_input` writes a **closed set of seven keys** into `custom_prompt`
   (`vllm_omni_async_server.py:494-508`), reached through a `generate()` call passing **seven fixed
   keyword arguments** (`single_turn_agent_loop.py:104-113`), and it is on neither list. Two live
   consumers depend on it anyway: `qwen_image_edit_flow_grpo` reads `vae_images` / `vae_image_sizes`
   out of it (`:370-372`) and raises
   `ValueError("Qwen-Image-Edit requires non-empty additional_information['vae_image_sizes']")` when
   they are absent (`:73-74`). §1.7 shows what those two fields were reaching for, and why an eighth
   key cannot supply it.
3. **One of the ten pipelines bypasses it and hand-rolls the same lookup** —
   `wan22_dance_grpo:373-381` reads `multi_modal_data["image"]` directly. `bagel_flow_grpo` is a
   second shape of the problem: rather than hand-rolling the lookup it *copies*
   `extra_args.multi_modal_data` up into `custom_prompt` (`:285-288`) — a workaround existing only
   because `_preprocess_input:506-508` wrote the dict into two places and the consumers disagree on
   which copy to read.

Video and audio are plumbed to different depths, which matters because "the parameter exists" is
not "the modality works":

| Modality in | Reaches the engine prompt? | Any consumer? | Status |
| --- | --- | --- | --- |
| text | yes | all 10 pipelines | **live** |
| image | yes, written twice (`:506-508`) | `qwen_image_edit_flow_grpo`, `bagel_flow_grpo`, `wan22_dance_grpo` | **live**, two lookup styles + one copy-up workaround |
| video | yes — `multi_modal_data["video"]` (`:377`) | **none** — `:377` is the only occurrence of the key under `verl_omni/` | **write-only** |
| audio | only if a caller passes `audio_data` | **none** — no caller on this branch supplies it | **unreachable** |

A contributor reading `generate`'s signature sees four conditioning modalities; one is silently
discarded and one unreachable, with nothing in the type system saying which is which.

### 1.5 What upstream already tags and this repo discards

- `OmniRequestOutput.final_output_type: str` (`outputs.py:87`). Its docstring (`:74`) lists
  `"text" | "image" | "audio" | "latents"` and is stale — upstream also emits `"video"` / `"videos"`
  (`stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29`) and `"actions"`
  (`models/gr00t/pipeline.py:22`), and verl-omni's AR track sets `"codec"`
  (`qwen3_omni/omni_rollout_adapter.py:91`). **Nothing in verl-omni reads the field off an
  output** — the only hits are stage-config literals (`:90-95`); §5.4 says why that stays.
- `OmniRequestOutput.images: list[Image.Image]` (`:91`) is declared as PIL images and in practice
  carries video tensors and `(video, audio)` tuples; verl-omni inherits the ambiguity by reading
  `final_res.images[0]` into an `Any`. `multimodal_output` also carries `actions` — a fifth modality
  already in the pinned engine, which today lands in `extra_fields` and is dropped without comment.

### 1.6 The text-encoding protocol has no owner

§1.1-§1.5 are about *what* travels on the request; this one and the next are a second gap —
**how the shared producer should build it for a given pipeline** is undeclared too, and that is
already being paid for with whole classes.

There is a working precedent. SD3.5 needs two text encoders with two tokenizers and **no agent loop
of its own** — the requirement is declared as data on the CLI and a generic producer honours it.

```bash
# examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh:56
actor_rollout_ref.model.extra_tokenizers='{clip: ..., t5: ...}'
```

`DiffusionModelConfig.__post_init__` resolves that into `extra_tokenizer_map`
(`workers/config/diffusion/model.py:69,74,173-186`), the base loop reads it
(`diffusion_agent_loop.py:242`) and `_tokenize_per_encoder` (`single_turn_agent_loop.py:37-63`)
emits one `extra_prompt_ids[key]` per declared tokenizer without knowing anything about SD3.5.
**This is the shape the rest of this section argues for**: the consumer declares, the shared base
produces per declaration. Two things it does not cover:

1. **The key names are not part of the declaration.** The consumer hardcodes them —
   `SD3_CLIP_TOKENS_KEY = "clip"` / `SD3_T5_TOKENS_KEY = "t5"` (`sd3_flow_grpo/common.py:23-29`) —
   and indexes the dict with them at `sd3_flow_grpo/vllm_omni_rollout_adapter.py:421`. The contract
   is *a shell string matching a Python constant*, checked at neither end; a typo in the
   run script surfaces as a `KeyError` mid-rollout.
2. **How the text is *rendered* is not part of it at all** — and that is what costs a class.
   `extra_tokenizer_map` says *which tokenizers to run*, never *chat template or raw text*. A
   pipeline whose text encoder wants raw text has no way to say so, so it subclasses the shared loop:

```python
@register("ltx2_diffusion_single_turn_agent")          # pipelines/ltx2_flow_grpo/agent_loop.py:49
class LTX2DiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    def __init__(self, ...):                           # :53-80  ten transcribed attribute assignments
        # "AgentLoopBase.__init__ would probe its optional chat template with two consecutive
        #  user messages, which strict templates reject before the LTX-specific raw-text path
        #  gets a chance to run."
    def apply_chat_template(self, messages, ...):      # :82-106  raw text, add_special_tokens=True
        ...                                            # right-truncated to rollout_config.prompt_length
```

Exactly **two** overrides, and **neither is about media** — both are text rendering. The
class exists so LTX-2 can say "this model takes raw text", and saying it costs a registry key
(`:49`), an export (`ltx2_flow_grpo/__init__.py:18,39`), ten transcribed `__init__` lines, and a
`default_agent_loop` line in each of two run scripts
(`examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh:78`, `…_npu.sh:87`). The next pipeline with
a raw-text encoder pays it again, and a reader cannot tell from the registry key that the difference
is one boolean.

### 1.7 One source image, two encoders, one geometry

The second half of the same gap, and what turns `additional_information` from a loose end into a
diagnosis: Qwen-Image-Edit's rollout adapter feeds **one source image to two consumers that
want it at two different geometries**:

| Consumer | Wants | Why | Site |
| --- | --- | --- | --- |
| Qwen2.5-VL text encoder | the processor's patch grid + `image_grid_thw` | the grid length must match the `<\|image_pad\|>` span it replaces in the prompt | `_get_qwen_prompt_embeds:135-137` |
| VAE | the image at its **source** resolution, plus its original size | condition latents are concatenated to the noise latents, so they must match the generation geometry | `prepare_latents(vae_images, …)` `~:447-448`, `torch.cat([latents, condition_image_latents], dim=1)` `:255` |

The request the loop hands over can carry only one of the two: the loop decodes the image once,
keeps the resized view, and drops the source on the next line:

```python
raw_prompt = kwargs["raw_prompt"]                              # :77  un-decoded source, still in scope
multi_modal_data = await self.process_vision_info(raw_prompt)   # :81  decode + smart_resize
images = multi_modal_data.get("images")                         # :82  only the resized view survives
```

(`verl_omni/agent_loop/single_turn_agent_loop.py`.) The resize is not configurable away:
upstream's `_process_multi_modal_info` has a **single branch** — `if has_visual: process_vision_info(...)`
else `None, None` (`verl/utils/dataset/rl_dataset.py:479-500`) — its own comment describes the work
as *"synchronous PNG decode + smart_resize (CPU-heavy)"* (`:445-446`), and the patch size it grids
to is read off the processor (`:113`), not off our config. No setting makes it a no-op, and making
it one would be wrong anyway: the VL tower needs its grid. **The two consumers do not want one
shared correct size; they want different sizes.**

Two things follow, both scoping the fix:

- **The dataset is not the problem.** `RLHFDataset` ships the messages through untouched —
  `row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)` (`:389`), where
  an image element is still `{"type": "image", "image": image}` (`:299-311`). The original source is
  available where the loop runs; nothing needs to change upstream or in the dataset (N1, N8).
- **`vae_images` / `vae_image_sizes` were reaching for exactly this slot**, and a key cannot supply
  it: §1.4's closed seven-key set has no room, and even with room, a `dict` entry does not say what
  geometry the value is in. The consumer is not wrong to want a second view; the request type cannot
  express one, so the adapter invented a field name and left the producer side blank. **That
  is the whole failure, in one file, today.**

### 1.8 The same pattern, live in review: PR #373

#373 (*add configurable uint8 response transport*, open) adds
`actor_rollout_ref.rollout.response_transport_dtype` ∈ {`float32`, `uint8`} to shrink the
rollout→train payload: `dispatch_lazy_compute_data_proto` costs **5.1 s**, **1.188 s** once fields
`update_actor` never reads are dropped. The goal is sound; expressing *one* dtype decision with no
declared output metadata costs §1.3's fan-out again:

- `visual_tensor_to_uint8()` lands in `utils/reward_score/reward_utils.py` and is imported by
  **10** modules — four scorers, `jpeg_compressibility.py`, `http_scorer_client.py`,
  `utils/tracking.py`, both trainer entry points, and
  `workers/rollout/vllm_rollout/vllm_omni_async_server.py`: the **rollout server imports a reward
  util**, for want of a shared media-IO module (§5.1).
- Four scorers gain an `if image.dtype != np.uint8:` branch *inside functions that already sniff
  rank*, and `torch.empty(0, dtype=torch.uint8 if … else torch.float32)` re-derives the decision on
  the empty path — a second implicit tensor property, tested wherever the first one is.
- `_diffusion_output_type(sampling_params)` in the async server is a **second implementation** of
  `sd3_flow_grpo/vllm_omni_rollout_adapter.py:137-142`, reading a dict where that one reads an
  object and hardcoding `"image"` where that one takes a caller default.
- `http_scorer_client.py:40`'s frame-0 discard (M-1) survives the PR untouched.

`dtype` / `value_range` on `MediaOut` (§5.3) make quantisation one producer-side declaration every
consumer reads; the deeper fix the PR itself defers — per-worker field declarations — is q8.

---

## 2. Motivation

The tree holds **10 diffusion pipeline packages** (11 counting the AR-track `qwen3_omni`, out of
scope by N4), **10 registered `(architecture, algorithm)` keys** and **9 rollout adapters**. Each
re-decides the same four questions — how do I pass conditioning, how do I say what I emitted, where
does the per-step trajectory go, how do I get the shared loop to produce text and images the way my
encoders need them — because no type answers them. Result: **3 representations of one request**, the
one this repo owns being untyped (§1.1-1.2), **7 sites sniffing tensor rank, 5 to answer "which
modality"** (§1.3), **5 candidate locations for one conditioning image, 3 with no producer**
(§1.4), and **2 fields with live consumers no code in the repo can write** (§1.7). Nothing here is a
bug in any single file; each entry below is what the missing type costs, once per pipeline, forever.

Seven failure modes follow directly:

- **M-1: wrong rewards, no error.** The `ndim == 4` collision (§1.3). A video pipeline
  configured with `http_scorer` scores frame 0 and reports a plausible number.
- **M-2: pin bumps break at training time, not request time.** The contract is string keys spread
  over eight adapters plus upstream's `multimodal_output`, unioned by `setdefault` so the adapter
  silently wins a collision (§1.3). Already live: `audio_sample_rate` has two producers and merge
  order decides which the trainer sees; `audio` and `fps` have no in-repo producer, so an upstream
  rename surfaces as a `KeyError` minutes into a run, or as a muted mp4.
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
  individually harmless; collectively the input contract is whatever the last adapter checked.
- **M-6: one boolean about text rendering costs a whole agent-loop subclass.** `extra_tokenizer_map`
  lets SD3.5 declare *which* tokenizers to run and needs no subclass; nothing lets LTX-2 declare
  *raw text instead of a chat template*, so it ships `LTX2DiffusionSingleTurnAgentLoop` — two
  overrides, neither about media — plus a registry key, an export and a run-script line in two
  scripts (§1.6). The next raw-text encoder pays it again.
- **M-7: an input has exactly one representation, and two consumers may need two.** Qwen-Image-Edit's
  VL text encoder wants the processor's patch grid while its VAE wants the source resolution; the
  loop materialises one view and drops the source one line later (`single_turn_agent_loop.py:81-82`).
  The adapter's answer was `vae_images` / `vae_image_sizes`, which the seven-key `custom_prompt` set
  cannot carry — **live consumers, no possible producer** (§1.7), and a `ValueError` on the only
  path that reads it.

None of these is model- or algorithm-specific: they are properties of the shared path, so they
recur on the next integration.

---

## 3. Goals / Non-goals

**Goals**

- **G1.** One request type covering text + image + video + audio conditioning, such that adding
  a conditioning modality or a second input with a different role requires **no signature
  change** on the relay path — by generalising the existing image-only
  `ImageGenerationRequest`, not adding a second partial answer beside it.
- **G1b.** One place to read a conditioning input, replacing the five candidates, the hand-rolled
  lookup and the copy-up workaround (§1.4), and resolving the two dead paths (write-only video,
  unreachable audio) one way or the other.
- **G2.** One output type in which the generated modality is **declared data**, and one request
  may declare more than one output medium.
- **G3.** **No site infers the generated modality where a declaration exists.** Not "zero rank
  checks": §1.3's last column keeps rank at the two sites where rank is genuinely the question —
  `ltx2_flow_grpo:381` (squeeze a leading batch axis) and `http_scorer_client.py:40` (is this a
  batch?) — and keeps `tracking.py:156`'s predicate for the `fps` shape. The five sites that answer
  *"which modality is this?"* by rank read `MediaOut.modality` instead; the two that answer *"is
  this batched?"* stop being confusable with them.
- **G4.** A first-class slot for the RL trajectory, so a rollout↔training key mismatch is a
  construction-time error rather than a training-time `KeyError`.
- **G5.** Fully additive: every milestone landable and revertible alone, and **no milestone may
  require editing all 10 diffusion pipeline packages at once.**
- **G6.** CPU-testable — importable without diffusers, vllm-omni, or a GPU.
- **G7.** The **input protocol becomes declared data on the consumer**, read by the shared agent
  loop: how a pipeline wants its text rendered (chat template vs raw text, §1.6) and which views of
  a conditioning input it needs (VL patch grid, source resolution, or both, §1.7). This generalises
  `extra_tokenizer_map`, moving the declaration from a shell string to a class attribute beside the
  code that consumes it. It is what makes M-7 fixable at all: `MediaRequest` supplies the *slot* for
  a second view, and G7 is what says the second view is wanted. Bounded by **N8**.

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
- **N8.** Deleting `LTX2DiffusionSingleTurnAgentLoop`, or changing upstream verl to make that
  possible. G7 retires that class's `apply_chat_template` override and its registry key — the
  reasons a *reader* has to care. Its `__init__` override survives: what it works around is
  upstream's own chat-template probe in `AgentLoopBase.__init__`
  (`verl/experimental/agent_loop/agent_loop.py:239-270`), and N1's rule — adapt at the boundary,
  never fork the pin — applies to `verl` too. Same boundary for §1.7:
  `RLHFDataset._process_multi_modal_info` stays untouched; G7 works from `raw_prompt`, which the
  dataset already ships intact (`rl_dataset.py:389`). An upstream ask to make the probe opt-out is
  filed as q7, not assumed.

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
  │  reads adapter.prompt_render / adapter.input_views   ← the declaration (§5.6), as it already
  │                                                       reads extra_tokenizer_map today
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
either side of them speaks the repo's own types — the 10 pipeline packages, the scorers, the
trainer and the tracking layer never import a vllm-omni type to answer a modality question. The
declaration line at the top removes the *third* kind of per-pipeline code: neither a request field
nor an output field, but a subclass of the shared producer (§1.6) or a field nothing can write
(§1.7).

---

## 5. Detailed design

### 5.1 Module placement

One new module, `verl_omni/pipelines/io.py`, with `torch` as its only third-party import so it
stays CPU-importable (G6). It sits **below** the pipeline registry and imports nothing from it,
so it cannot collapse behaviour across an `(architecture, algorithm)` boundary — which
`.agents/rules/pipelines.md` forbids.

`verl_omni/pipelines/` rather than `workers/rollout/`: both rollout and training adapters are
consumers, and `request_batch.py` already puts shared plumbing here. A new module rather than
extending `pipelines/utils.py` (where `ImageGenerationRequest` lives), because that module imports
`diffusers`, `tensordict` and `verl.utils.device` at module scope (`:21-29`), so anything added
there fails G6. `utils.py` imports from `io.py`, not the reverse.

### 5.2 The input types

```python
Modality = Literal["text", "image", "video", "audio"]   # this repo's own vocabulary, deliberately
                                                        # NOT upstream's ("videos", "actions",
                                                        # "codec"), never ingested here (§5.4)


@dataclass
class MediaRef:
    """One conditioning input, carrying its own modality tag."""

    modality: Modality
    data: Any                    # decoded PIL / ndarray / tensor, as ImageGenerationRequest.images
    role: str = "condition"      # condition | reference | keyframe | identity — the M-3 case
    view: str = "native"         # geometry of `data`. Two refs may share source+role and differ
                                 # only here: "vl_grid" = the processor's patch grid the VL text
                                 # encoder needs, "native" = the source resolution the VAE needs
                                 # — the M-7 case (§1.7)
    source: Any | None = None    # the un-decoded origin from raw_prompt (path / dict / bytes), so a
                                 # consumer can derive a view the loop did not materialise (q6)
    meta: dict[str, Any] = field(default_factory=dict)   # fps, frame_index: per-input scalars
                                                         # today lost or promoted to sampling knobs
                                                         # size lands here, replacing vae_image_sizes


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
    render: Literal["chat_template", "raw_text"] = "chat_template"  # how `ids` were produced;
                                                                    # declared per pipeline (§5.6)


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

Twelve kwargs become seven fields, the nine input-describing ones become three. **The
property that matters: `conditions` is a list.** A fifth modality, a second image with a
different role, a reference video alongside a keyframe — each extends the list; none touches a
signature. That is what makes M-7 expressible: one source image needed at two geometries is two
`MediaRef` entries sharing `source` and `role` and differing in `view`, rather than a second field
name, and `vae_image_sizes` becomes `meta["size"]` on the `native` entry — a field that is *present*
rather than one whose absence must be raised on. **What the request type does not do is decide that
the second view is wanted**; that is §5.6's declaration, and the two are only useful together.
`Modality` is a `Literal` rather than an `Enum` because these values round-trip through
`sampling_params`, `extra_args` and `non_tensor_batch` object arrays — all plain-data channels.

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
    dtype: str = "float32"       # transport dtype, declared rather than sniffed off the tensor
    value_range: tuple[float, float] = (0.0, 1.0)   # the pair a consumer needs with `dtype` (§1.8)
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

`dtype` and `value_range` exist because #373 (§1.8) must spread one transport decision across ten
modules for want of them: a consumer reads the declared pair instead of testing
`image.dtype != np.uint8`, and the empty-output path carries the declaration instead of re-deriving
it.

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
"image"` (`vllm_omni/engine/async_omni_engine.py:1000`) — no branch can produce `"video"`. Upstream
does use `"video"`, but only in multi-stage stage configs this path never builds
(`model_executor/stage_configs/wan2_2_ti2v_dit_fp8.yaml:32`, `hunyuan_video_15_dit_fp8.yaml:29`). So
for `wan22_dance_grpo` the field is a confidently wrong `"image"` — and never empty, so a
prefer-declared-else-infer rule would never reach its fallback. Trusting it would make M2 **cause**
M-1 rather than fix it: `ray_diffusion_trainer.py:309` would dump `0.jpg` from a `[T,C,H,W]` tensor
and `tracking.py:156` would stop emitting `wandb.Video`.

Modality is therefore declared by the **rollout adapter**, the only component that knows what it
produced: a class attribute, or `MediaOut(modality=…)` at construction, consistent with
`DiffusionPipelineConfig.num_frames` (`config/diffusion/rollout.py:71` — `> 1` implies video).
`final_output_type` is read only as a cross-check logging a warning on disagreement, never as the
source — which is why R4 is a *design constraint* rather than an accepted risk.

### 5.5 Compatibility seam, and naming

M1 adds two derived properties to the existing `DiffusionOutput` (`replica.py:20-32`), computed
from today's conventions — `modality` (adapter-declared where available, else `ndim`; **not**
`final_output_type`, §5.4) and `media` (the primary tensor plus audio/fps out of `extra_fields`).
Every consumer can migrate to the declared field before the wire format changes, and
`DiffusionOutput` keeps its name, fields and Pydantic base. This is what makes M1 revertible:
deleting the two properties and the new module restores the tree exactly.

An earlier draft called these `OmniGenRequest` / `OmniGenOutput`. Both prefixes are wrong here:
**`Omni*`** denotes the autoregressive track (`OmniModelBase`, `OmniAlgoConfig`, …), out of scope
by N4; **`Diffusion*Output`** already names two different classes coexisting in the
same modules (`replica.py:20` and `vllm_omni/diffusion/data.py:1196`); and
**`*GenerationRequest`** already collides too (`pipelines/utils.py:45` vs
`protocol/images.py:33` — two unrelated types, one name, both importable in one process), so
`MediaGenerationRequest` would add a third near-homonym. `MediaRequest` / `MediaOutput` /
`MediaRef` / `MediaOut` / `PromptBundle` / `Modality` are all unused in the tree (verified: no
`class Media*`, no bare `Modality`).

### 5.6 The input protocol as declared data (G7)

`MediaRequest` gives the second view a slot; this is what fills it. The declaration lives on the
**rollout adapter** as class attributes — the same choice as §5.4's modality declaration, and for
the same reason: it knows what its encoders need, and it is where a reader goes when the answer
looks wrong.

```python
class QwenImageEditPlusFlowGRPO(QwenImage):
    prompt_render = "chat_template"
    input_views = {"image": ("vl_grid", "native")}   # VL text encoder + VAE (§1.7)


class LTX23PipelineWithLogProb(...):
    prompt_render = "raw_text"                        # replaces the apply_chat_template override
    input_views = {}                                  # text-only; nothing to materialise
```

`DiffusionSingleTurnAgentLoop` reads both: it renders the prompt per `prompt_render` and emits one
`MediaRef` per (source, requested view), tagged with the geometry it was produced at. Three
consequences:

- **No new data source, and no dataset change.** Both views are derived on the producing side from
  `raw_prompt`, still in scope at `single_turn_agent_loop.py:77` — one line before today's code
  discards it (`:82`). `"vl_grid"` is exactly what `process_vision_info` returns now, so the existing
  default becomes one of the two views rather than something replaced. N1/N8 hold: `RLHFDataset` and
  `AgentLoopBase` are untouched.
- **`vae_images` / `vae_image_sizes` are deleted, not given a producer.** The VAE reads the `native`
  entry of `conditions` and its `meta["size"]`. `_validate_condition_image_sizes:65-92` then
  validates a field the type guarantees is there, instead of raising because nothing in the repo
  could have written it.
- **A wrong declaration fails at startup.** An `input_views` entry naming a view the loop cannot
  produce, or a `prompt_render` outside the `Literal`, is a construction-time error — where §1.6's
  shell-string-to-Python-constant contract fails as a mid-rollout `KeyError`.

What it does not do: LTX-2 keeps its `__init__` override, which works around upstream's
chat-template probe (N8). M6's measurable win is the `apply_chat_template` override, the registry
key, the export and the two run-script lines — what a contributor must read and copy for the next
raw-text encoder.

---

## 6. Milestones

Ordered by risk. Each is a separate PR with its own tests, revertible without touching the next.

**M1 — `io.py` plus derived accessors. No behaviour change.**
The types in §5.2-5.3, the two `DiffusionOutput` properties (§5.5), and `to_generate_kwargs()`.
Nothing calls them yet. Entirely CPU-testable.
*Title:* `[rollout, pipelines] feat: add unified media I/O types`.

**M2 — read the declared modality at the five modality sites.**
The last column of §1.3's table, all seven rows: five stop inferring modality, and the two that
legitimately test for batching stay but are no longer confusable with them. Fixes M-1, and is where
review effort belongs because it touches the reward path. Each site keeps its behaviour for the
modality it handles today; only how the modality is determined changes.
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
retire the `image_latents` guard (`model_base.py:355-371`) — that name is reserved in the
**training-side** `model_inputs` dict, which no `MediaOut` / `MediaOutput` field governs — but it
does remove the guard's *trigger*, since the condition latent becomes a named slot. Declaring the
reservation is a training-side follow-up.
*Title:* `[rollout, trainer] refactor: promote trajectory and media keys out of extra_fields`.

**M5 — retire the five-candidate lookup.**
Collapse `utils.py:79-83` to one place and move `wan22_dance_grpo:373-381` onto it.
`bagel_flow_grpo:285-288` is **deleted rather than migrated** — it only copies
`extra_args.multi_modal_data` up into `custom_prompt`, and M3 stops `:506-508` writing the dict
twice, so there is nothing left for it to reconcile. Also where the dead paths get resolved: give
`multi_modal_data["video"]` a consumer or stop writing it, and wire `audio_data` to the
agent loop or delete the parameter. Fixes M-5; deletes more than it adds.
*Title:* `[pipelines, rollout] refactor: single condition-input lookup via MediaRef`.

M3 and M4 are worth doing only if M1's types survive contact with a second pipeline; if they do
not, M1+M2 stand alone and M-1 is still fixed. M5 depends on M3.

**M6 — declare the input protocol (G7).**
`prompt_render` and `input_views` as adapter class attributes read by
`DiffusionSingleTurnAgentLoop` (§5.6). Fixes M-6 and M-7. Depends on M3: a second view of one
input needs `conditions` to be a list first. The two halves are independent and **should land as two
PRs**: the text half retires `LTX2DiffusionSingleTurnAgentLoop`'s `apply_chat_template` override, its
registry key and export, and the `default_agent_loop` line in two run scripts (its `__init__`
override stays, per N8); the image half emits the `native` view for `qwen_image_edit_flow_grpo` and
**deletes** `vae_images` / `vae_image_sizes` with the `ValueError` guarding their absence. Bounded by
N8 — no upstream verl change, no dataset change.
*Title:* `[rollout, pipelines] feat: declare per-pipeline prompt rendering and input views`.

---

## 7. Test plan

Placement under `tests/<module>/` is the commit gate and the `_on_cpu.py` suffix is what CI
selects on, so both are load-bearing (`.agents/rules/testing.md`).

**M1 — `tests/pipelines/test_unified_io_on_cpu.py`.** `io.py` imports with `diffusers` and
`vllm_omni` absent from `sys.modules` (G6); `multi_modal_data()` byte-identical to
`_build_multi_modal_data` for all eight image/video/audio present-absent combinations; a
twelve-kwarg round-trip through `to_generate_kwargs()` with `None`s included;
`DiffusionOutput.modality` is `"image"` for `[C,H,W]`, `"video"` for `[T,C,H,W]`, and still
`"video"` when `final_output_type` says `"image"` — the wan22 case the engine produces (§5.4);
`get("audio")` returns `None` rather than raising.

**M2 — regression, not new coverage.** The three rank-sniffing scorers keep their existing CPU
tests green **unchanged**; any diff means M2 changed behaviour, which it must not. One new case
pins the §1.3 collision: an image `MediaOut` and a video `MediaOut` with the *same* `data.ndim`
route differently — the test that would have caught M-1, expressible only once modality is a
field. `tests/trainer/` gains an mp4-vs-jpg case driven by the declared modality.

**M3-M5 — contract tests.** `generate(request)` and `generate(**kwargs)` produce identical
`OmniCustomPrompt` + `OmniDiffusionSamplingParams` pairs, field by field. Per-algorithm
trajectory-key tables (FlowGRPO / NFT / DPO) asserted against all eight adapters, so a missing key
fails in CI rather than mid-run. `tests/pipelines/test_image_edit_interface_on_cpu.py` already pins
the five-candidate precedence (`:47-63`, `:84-85`): M3's compatibility test, then deliberately
**rewritten** in M5 as an explicit hunk, never a silent deletion. M5 also asserts either that a
`MediaRef(modality="video")` reaches the adapter or that `video_data` is gone.

**M6 — the declaration is the test.** `tests/pipelines/test_input_protocol_on_cpu.py`: a stub
adapter declaring `prompt_render="raw_text"` produces ids **byte-identical** to
`LTX2DiffusionSingleTurnAgentLoop.apply_chat_template`, which is what makes that override safe to
delete rather than merely equivalent-looking; `input_views={"image": ("vl_grid", "native")}` yields
exactly two `MediaRef`s sharing `source` and `role`, differing in `view`, `meta["size"]` on the
`native` one and the `vl_grid` one identical to today's `process_vision_info` output; an unknown
view name and an invalid `prompt_render` raise at construction; and no occurrence of `vae_images` /
`vae_image_sizes` survives under `verl_omni/`, asserted with the grep-style pattern
`tests/special_sanity/` already uses.

**GPU.** `qwen_image_flow_grpo` (image) and `ltx2_flow_grpo` (video+audio) re-run unchanged after
M2 and M4 — the only checks covering the `multimodal_output` merge against a real engine — plus one
`qwen_image_edit_flow_grpo` run after M3 and M5, the only live multi-input path. M6 needs one run
per half: `ltx2_flow_grpo` t2av for the text half (only a run proves the strict template is still
never probed) and `qwen_image_edit_flow_grpo` for the image half, where the VAE now reads the `native` view and the output must be no worse than the
pre-M6 baseline on the same seed.

**Every milestone.** `pre-commit run --files <changed>`; every hook must pass except
`autogen-trainer-cfg`, which fails in the current venv with a pre-existing
`ModuleNotFoundError: omegaconf` (`scripts/print_cfg.py:16`) unrelated to these files.

---

## 8. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| R1 | **M2 changes a reward silently** — the reward path is the one place a wrong refactor yields a plausible number instead of an error. | M2 keeps each scorer byte-identical **for the modality it handles today** and changes only how the modality is chosen; the existing scorer tests must pass *unchanged*. The one intended change is the M-1 fix: `http_scorer_client` **raises** on a declared non-image modality where it used to score frame 0. It gets its own test and is the only reward-path diff that may land — any other blocks the PR. |
| R2 | **A fifth representation** — adding types without deleting the old ones. | M3 and M4 are the deletions and they are in the plan. If M3 never lands, `to_generate_kwargs()` is one function and M2 replaced inference with field reads — no new representation, only accessors. |
| R3 | **`trajectory` becomes the new `extra_fields`** under a nicer name. | The per-algorithm key tables in §7 are the constraint; §5.3 states why it stays a dict and §9 q1 tracks promoting it. |
| R4 | **`final_output_type` cannot express video** on this path — the single-stage constructor only ever picks `"audio"` or `"image"` (`async_omni_engine.py:1000`), so it is confidently wrong for `wan22_dance_grpo` and never empty. | This is why §5.4 makes the **adapter** the source of truth and demotes `final_output_type` to a warn-on-disagreement cross-check. §7's M1 case pins the wan22 shape explicitly. |
| R5 | **Ten packages, one shared type** — a type fitting eight image pipelines may not fit a joint A/V one. | M1 lands with two consumers of *different* shape (one image pipeline and `ltx2_flow_grpo`), not eight of the same. G5 makes each milestone revertible. |
| R6 | **Merge conflicts with in-flight work** — anything touching `_process_output` or a rollout adapter. | M1 adds a file and two properties; M2 edits one line per site. M3/M4 touch the adapters and should be sequenced after whatever pipeline work is in flight, rather than racing it. |
| R7 | **Two views double the conditioning payload** — the same image travels twice through collate, the Ray object store and the engine prompt; for a multi-image edit request that is a real cost. | `input_views` is opt-in and defaults to today's single view, so no pipeline pays until it asks. Where the second view is large, `MediaRef.source` is cheaper — ship the origin, let the consumer decode — at the cost of moving decode onto the rollout worker (q6). M6 lands the explicit-views form first because its cost is the visible one. |
| R8 | **G7 stops halfway and the RFC reads as a false claim** — with only the request type landed, the slot for a second view exists and nothing fills it. | N8 states the boundary, and M6's success criterion is written as the artefacts that disappear (the `apply_chat_template` override, the registry key, `vae_images`/`vae_image_sizes` and their `ValueError`) rather than as "the class is gone" — so a half-landed M6 is visibly half-landed. Until M6 lands, §1.7 is a documented defect, not a solved one. |

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
6. **Second materialised view, or a source reference?** §5.6 emits both views eagerly;
   `MediaRef.source` would instead ship the origin and let the consumer decode. Eager keeps decode on
   the CPU side where it already happens; lazy halves the payload (R7) but needs the rollout worker to
   reach the origin, which fails for parquet-embedded bytes and any path not visible from the worker.
   Both fields are in §5.2 deliberately; which a pipeline should prefer is unresolved.
7. **Upstream ask: can `AgentLoopBase.__init__`'s chat-template probe be made opt-out?** It is the only
   reason `LTX2DiffusionSingleTurnAgentLoop.__init__` survives M6
   (`verl/experimental/agent_loop/agent_loop.py:239-270`); a `probe_chat_template: bool = True`
   parameter there would let the class disappear entirely. Out of scope by N1/N8 — worth raising with
   verl, not worth forking for.
8. **Does `DataProto` field projection follow?** #373 defers its real fix to per-worker field
   declarations plus a projecting dispatcher (§1.8). Out of scope here — this RFC touches neither
   `DataProto` nor the dispatcher (N3) — but `MediaOutput`'s separate `media` / `trajectory` /
   `extra` slots are its precondition: today `responses` is one flat tensor with the trajectory
   hidden in `extra_fields`, so there is no named field a worker could decline.

---

## 10. Appendix — key upstream references

Verified at vllm-omni `0.24.1.dev26+gfe478a95a`. verl-omni references are cited inline and
greppable in-tree; only the pinned upstream ones are collected here.

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

§1.6-1.7 and N8 also lean on the **other** pinned dependency, upstream `verl`, not greppable in
this repo — paths relative to the installed package. `utils/dataset/rl_dataset.py`:
`_build_messages` leaves an image element as `{"type": "image", "image": image}` (`:299-311`) and
ships `raw_prompt` **un-decoded** (`:389`, why §1.7 needs no dataset change);
`_process_multi_modal_info` is a single branch with no per-pipeline choice (`:479-500`), its own
comment calling the work *"synchronous PNG decode + smart_resize (CPU-heavy)"* (`:445-446`); the
patch size is read off the processor, not off our config (`:113`). And
`experimental/agent_loop/agent_loop.py:239-270` is the chat-template probe in
`AgentLoopBase.__init__` that LTX-2's override works around (N8, q7).

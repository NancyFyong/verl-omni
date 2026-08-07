# [RFC] Add `nemo_automodel` as an engine backend for diffusion models

Last updated: 08/04/2026

Companion RFC to [`RFC_ISSUE.md`](RFC_ISSUE.md), which covers the
`(omni_model, automodel)` track. This one covers the **second, independent track**:
`(diffusion_model, automodel)`, targeting Qwen-Image FlowGRPO.

All `nemo_automodel` findings below were read directly from source at
`NVIDIA-NeMo/Automodel` **main, rev `eca4a8e`** (2026-08-04, "fix(fsdp): resolve fp32
master-weight compute dtype per parameter (#3328)"). Line numbers in this document
track that revision. `verl-omni` line numbers track this branch.

## 1. Feature request

Register a third diffusion training engine, selected by `actor.strategy=automodel`,
that delegates **model build and parallelization** to
`nemo_automodel.NeMoAutoDiffusionPipeline` while keeping verl-omni's own forward,
sampling, and loss.

The reference task is the one the VeOmni engine already runs, so the two are directly
comparable:
[`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh`](../examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh)
— Qwen-Image OCR FlowGRPO, 4 GPUs, Qwen3-VL-8B GenRM, 512 resolution,
`layered_summon=True`.

## 2. Motivation

Three things make this worth doing beyond backend parity:

- **Context parallelism on Qwen-Image is reachable.** `VeOmniDiffusionEngineConfig`
  raises `NotImplementedError` for `ulysses_parallel_size != 1`
  (`verl_omni/workers/config/diffusion/actor.py:108`), so today no diffusion engine in
  this tree does sequence parallelism. nemo wires diffusers' native CP hooks
  (`_enable_context_parallel`, `auto_diffusion_pipeline.py:478`) and Qwen-Image's
  transformer **does** define a `_cp_plan` in diffusers 0.39.0 — verified locally
  against the installed version. Pure Ulysses (`cp_ring_degree=1`) is supported;
  ring attention is explicitly rejected for training (`:526`) because its backward is
  broken in `diffusers<=0.39`. This is the first path to longer-sequence /
  higher-resolution diffusion RL in this repo.
- **No hard dependency on a second training framework.** The VeOmni engine imports
  `veomni.trainer.dit_trainer` and drives `BaseTrainer._build_*` step by step
  (`veomni/diffusion_impl.py:255-276`). The nemo diffusion entrypoint is one
  classmethod returning a plain diffusers pipeline, so the engine surface stays
  smaller and the dependency is easier to keep optional.
- **It closes the "fully support automodel" goal.** The omni track alone leaves
  `main_diffusion` — the entrypoint most of this repo's recipes use — without an
  automodel option.

## 3. Non-goals

- Not replacing FSDP2 as the default; `actor.strategy` keeps its current default and
  existing recipes are untouched.
- Not using nemo's `components/flow_matching/` training loop — see §4.3.
- Not video (Wan2.2 / LTX2) or Flux in the first PR series. The build path is
  model-agnostic (§4.2), so those follow as recipes, not as engine work.
- Not making `nemo_automodel` a required dependency.

## 4. Architecture

### 4.1 Where the backend plugs in

Selection is the same registry mechanism the other two diffusion engines use — a
`(model_type, backend, device)` key:

| Registry key | Class | Location |
| --- | --- | --- |
| `(diffusion_model, fsdp/fsdp2, cuda/npu)` | `PPODiffusersFSDPEngine` | `fsdp/diffusers_impl.py:816` |
| `(diffusion_model, veomni, cuda)` | `VeOmniDiffusionEngine` | `veomni/diffusion_impl.py:55` |
| `(diffusion_model, automodel, cuda)` | `AutomodelDiffusionEngine` | **new** |

So the entrypoint (`main_diffusion`), trainer, dataflow, rollout, and reward loop are
all unchanged; only `actor.strategy` moves.

### 4.2 What nemo gives us, verified

| Fact | Location (`eca4a8e`) | Implication |
| --- | --- | --- |
| `NeMoAutoDiffusionPipeline` is a top-level lazy export | `nemo_automodel/__init__.py:59` | peer of `NeMoAutoModelForCausalLM` (`:43`) and `ForMultimodalLM` (`:45`), not experimental |
| `from_pretrained` returns `Tuple[DiffusionPipeline, Dict[str, ParallelManager]]` | `auto_diffusion_pipeline.py:665`, returned at `:850` | **not** an `nn.Module`; the engine must take `pipe.transformer` itself and keep the managers dict |
| Build delegates to diffusers' `DiffusionPipeline.from_pretrained` auto-detection | `:717` | **model-agnostic** — Qwen-Image loads because diffusers supports it; there is no nemo-side allowlist |
| Parallelism driven by `parallel_scheme={component: {_manager_type: 'fsdp2'\|'ddp', ...}}` | `:551` `_apply_parallelization` | a **completely different** argument shape from the LM path's `device_mesh` / `distributed_config` |
| Only components present in `parallel_scheme` are touched | `:576` (`if manager_args is None: continue`) | can shard `transformer` alone, leaving VAE / text-encoder unsharded |
| Stamps `_pre_shard_hf_state_dict_keys` just before sharding | `:581` | the checkpointer reads it (`checkpoint/checkpointing.py:1617`) to rebuild consolidated safetensors keys |
| FSDP2 accepts `offload_policy` | `:452` | CPU param offload is expressible through `parallel_scheme`, no VeOmni-style helper needed |
| Ulysses CP via diffusers `_cp_plan`; ring rejected for training | `:478`, `:526` | Qwen-Image has a `_cp_plan` in diffusers 0.39.0 (verified locally) → **SP is reachable**, unlike VeOmni (§2) |
| `peft_cfg` LoRA injection is **model-agnostic**; only `model_type is None` is rejected | `:766`, `:773` | Qwen-Image LoRA is reachable: the `{flux, flux2, wan, hunyuan, ltx2}` list in the error text is not enforced, and `model_type` is read only by a Wan-specific unfuse hook (`:790`). Injection walks `named_modules()` (`_peft/lora.py:567`, loop at `:617`) via `ModuleMatcher`; component discovery reads diffusers' `pipe.components` (`auto_diffusion_pipeline.py:161`) |
| LoRA base-weight freeze happens **after** `fully_shard` | `:843` | matches FSDP2's requirement; an engine must not pre-freeze |
| `from_config` requires a `PipelineSpec` | `:852` | random-init / pretraining path; unused here |

### 4.3 Key judgement: nemo's `flow_matching` is not reusable

`components/flow_matching/{pipeline.py,adapters/}` implements **supervised**
flow-matching finetuning. Read at this revision, `FlowMatchingPipeline.compute_loss`
(`pipeline.py:328`) is a weighted MSE between `model_pred` and a velocity target, with
`loss_weighting_scheme ∈ {bsmntw, linear}`, and `step` (`:377`) consumes
`{image_latents | video_latents, text_embeddings}` for a single supervised update.
There is **no log-prob and no reverse-sampling** anywhere in it. A Qwen-Image adapter
does exist (`adapters/qwen_image.py`), but it serves that SFT objective.

verl-omni's FlowGRPO instead needs **reverse sampling with per-step log-probs**:
`verl_omni/pipelines/utils.py:287 forward_and_sample_previous_step`, dispatched per
architecture (`model_base.py:204`) and already **shared by both existing engines**
(`fsdp/diffusers_impl.py:881`, imported in `veomni/diffusion_impl.py:43`). The
objectives differ; nemo's adapters do not apply.

**So from nemo we take only build-and-parallelize; forward, sampling and loss stay
verl-omni's own.** This is the mirror image of the omni track's tradeoff:

```text
   OMNI TRACK (RFC_ISSUE.md)             DIFFUSION TRACK (this RFC)
   inherit verl's forward,               borrow nemo's build,
   swap only the build                   keep verl-omni's forward

  ┌────────────────────────┐            ┌──────────────────────────────┐
  │ nemo                   │            │ nemo                         │
  │  NeMoAutoModelFor      │            │  NeMoAutoDiffusionPipeline   │
  │   MultimodalLM         │            │   .from_pretrained           │
  │   .from_pretrained     │            │      ↓                       │
  │      ↓                 │            │   (pipe, managers) tuple     │
  │   nn.Module            │            │      ↓ take pipe.transformer │
  └────────┬───────────────┘            └──────────────┬───────────────┘
           │                                           │
           ▼                                           ▼
  ┌────────────────────────┐            ┌──────────────────────────────┐
  │ verl                   │            │ verl-omni  (NOT nemo)        │
  │  AutomodelEngineWith   │            │  forward_and_sample_         │
  │   LMHead.forward_step  │            │   previous_step              │
  │  = INHERITED           │            │  = shared with FSDP/VeOmni   │
  └────────────────────────┘            └──────────────────────────────┘
       ~150 new lines                        400-640 new lines
```

### 4.4 Engine design — three options

`VeOmniDiffusionEngine(BaseEngine)` is **643 lines** across ~30 methods:
`initialize`, `train_mode`, `eval_mode`, `get_data_parallel_{rank,size,group}`,
`is_mp_src_rank_with_outputs`, `prepare_model_{inputs,outputs}` (`:341`, `:379`),
`forward_step` (`:388`), `forward_backward_batch` (`:442`),
`postprocess_batch_func` (`:477`), `optimizer_{zero_grad,step}`, `lr_scheduler_step`,
`to`, `{save,load}_checkpoint`, `get_per_tensor_param` (`:587`), `disable_adapter`,
plus `EngineEvalModeCtx` / `EngineTrainModeCtx` (`:615`, `:631`).

An `AutomodelDiffusionEngine` **cannot** inherit verl's `AutomodelEngineWithLMHead` —
that chain is the LM-head forward path, which is exactly what §4.3 says we are not
using. Nor can it inherit `DiffusersFSDPEngine` (`fsdp/diffusers_impl.py:79`), whose
`_build_model_optimizer` is FSDP-specific. That leaves three options:

- **D1 — `AutomodelDiffusionEngine(VeOmniDiffusionEngine)`**, overriding only
  `_build_model_optimizer`. **Rejected**: the VeOmni engine's `to()` (`:532`) and both
  checkpoint methods call `veomni.distributed.offloading` helpers imported at module
  scope (`:26`), which would make VeOmni a hard dependency of the automodel path.
- **D2 (recommended) — extract a `DiffusionEngineMixin`.** Move the VeOmni-agnostic
  parts (`forward_step`, `forward_backward_batch`, `postprocess_batch_func`,
  `prepare_model_{inputs,outputs}`, both Ctx classes) into a mixin; VeOmni and
  automodel then each implement only `_build_model_optimizer` + offload + checkpoint.
  This is exactly `code-style.md`'s "Reuse over duplication" mechanism 3 (the
  `NPUColocateWorkerMixin` precedent, #82), and bundling a refactor with the feature is
  encouraged by `CLAUDE.md` §1. Cost: touches already-merged VeOmni code, widening
  review scope.
- **D3 — write fresh from `BaseEngine`**, reproducing VeOmni's ~640 lines. Fastest to
  merge, highest duplication; the two files then drift independently.

**Recommendation: D2**, with the mixin extraction as its own commit inside PR1 so the
diff is reviewable as "no behaviour change" + "new engine".

### 4.5 Configuration

Unlike the omni tree, the diffusion tree **already has** a `model_engine` Hydra group,
so this track adds files rather than inventing a mechanism:

| Existing (VeOmni) | New (automodel) |
| --- | --- |
| `diffusion/model_engine/veomni_diffusion.yaml` | `automodel_diffusion.yaml` |
| `diffusion/actor/veomni_diffusion_actor.yaml` | `automodel_diffusion_actor.yaml` |
| `diffusion/ref/veomni_diffusion_ref.yaml` | `automodel_diffusion_ref.yaml` |
| `diffusion/engine/diffusion_veomni.yaml` | `diffusion_automodel.yaml` |

`diffusion_trainer.yaml:14,21` interpolate `${diffusion/model_engine}_actor` / `_ref`,
so the group name drives actor and ref selection together.

Config dataclasses follow `VeOmniDiffusionActorConfig`
(`workers/config/diffusion/actor.py:200`) exactly: a new
`AutomodelDiffusionEngineConfig(EngineConfig)` with `strategy="automodel"` plus the
`parallel_scheme` knobs (`fsdp_size`, `cp_size`, `param_offload`, `activation_checkpointing`),
`AutomodelDiffusionOptimizerConfig(OptimizerConfig)`, and
`AutomodelDiffusionActorConfig(DiffusionActorConfig)` whose `__post_init__` sets
`self.engine = self.automodel_config` — mirroring `:207`. Validation goes in
`__post_init__`, per repo convention.

A fourth generated reference YAML is then required. `scripts/generate_trainer_config.sh`
has a `CONFIG_SPECS` array; add
`"diffusion_trainer:_generated_diffusion_automodel_trainer.yaml:--config-name=diffusion_trainer.yaml diffusion/model_engine=automodel_diffusion"`
and regenerate — **never hand-edit** the `_generated_*.yaml` (the `autogen-trainer-cfg`
hook fails the commit otherwise). Expect ~600 generated lines, comparable to
`_generated_diffusion_veomni_trainer.yaml` (619).

### 4.6 Weight sync is mandatory in-PR

This is the sharpest difference from the omni track. Qwen-Image FlowGRPO is **online**
RL: after every update the actor's weights must reach the vLLM-Omni rollout. The omni
track inherits `get_per_tensor_param` from verl's automodel engine; here it must be
written, following `veomni/diffusion_impl.py:587`:

- `state_dict()` → `convert_weight_keys` → per-tensor `full_tensor()` on `DTensor`
- cast to `engine_config.model_dtype`, move to device, yield as `f"transformer.{name}"`
  (the `transformer.` prefix is what the rollout side expects)
- LoRA export raises `NotImplementedError` in the VeOmni engine (`:588`); PR1 should
  match that and defer LoRA to a follow-up (§5).

### 4.7 Principal unknowns (GPU-only)

Three seams cannot be settled by reading source; each needs a 4-GPU run:

1. **Offload parity.** VeOmni's `to()` moves params and optimizer state explicitly via
   its own helpers. nemo expresses offload as FSDP2 `offload_policy` (`:452`), which is
   a *sharding-time* decision, not a callable toggle. If FlowGRPO's colocated rollout
   needs `to("cpu")` between phases, the automodel engine may have to implement the
   move itself against the sharded module rather than delegating.
2. **`layered_summon=True` interaction.** The reference recipe sets it
   (`run_qwen_image_ocr_veomni.sh:63`). Whether per-tensor `full_tensor()` over an
   FSDP2-sharded nemo module composes with layered summon at 4-GPU scale is unverified.
3. **Checkpoint round-trip.** nemo's `Checkpointer` relies on
   `_pre_shard_hf_state_dict_keys`; verl-omni's resume flow expects its own layout. A
   save/load round-trip is acceptance criteria, not an assumption.

## 5. Implementation plan

### PR1 — mixin extraction + engine + config (online FlowGRPO runnable)

1. `refactor`: extract `DiffusionEngineMixin` from `VeOmniDiffusionEngine` (D2), no
   behaviour change; VeOmni engine re-derives from it.
2. `verl_omni/workers/engine/automodel/diffusion_impl.py` —
   `AutomodelDiffusionEngine`, registered `(diffusion_model, ["automodel"], ["cuda"])`.
   Build: `NeMoAutoDiffusionPipeline.from_pretrained(..., parallel_scheme={"transformer": {...}})`,
   take `pipe.transformer` as `self.module`, keep `managers`, build optimizer +
   LR schedule, wire `forward_and_sample_previous_step`.
3. Config dataclasses + four Hydra YAMLs + regenerated
   `_generated_diffusion_automodel_trainer.yaml` (§4.5).
4. `get_per_tensor_param` (§4.6) with `NotImplementedError` for LoRA.
5. `examples/automodel_trainer/qwen_image/run_qwen_image_ocr_automodel.sh`, mirroring
   the VeOmni recipe's override block.
6. CPU tests (`test_*_on_cpu.py`) + a GPU smoke test beside them using
   `importorskip("nemo_automodel")`, registered in
   `tests/gpu_smoke/run_gpu_smoke_core.sh` — the pattern PR1 of the omni track already
   established.
7. `docs/start/install.md`: extend the existing nemo_automodel subsection to note the
   diffusion path.

**Acceptance**: OCR reward curve within noise of the VeOmni run over a few hundred
steps; save/load round-trip; weight sync verified by a rollout-output diff before and
after one update.

### PR2 — Ulysses context parallelism

The capability that motivates the track (§2). `cp_size > 1` through `parallel_scheme`,
pure Ulysses only (`cp_ring_degree=1`, enforced by nemo at `:526`). Needs a
resolution/sequence-length benchmark against `cp_size=1` to justify itself.

### PR3 — LoRA + sync parity

Enable `peft_cfg` (§4.2), drop the `NotImplementedError`, and reach parity with the
FSDP engine's LoRA weight export. Note nemo freezes base weights *after* sharding
(`:843`) — the engine must not pre-freeze.

### PR4 (roadmap) — more architectures

Wan2.2 (`active_transformer` selection for the two-transformer case), Flux, LTX2. Build
is model-agnostic, so these are recipe + adapter work, not engine work.

### Effort summary

| PR | Hand-written | Notes |
| --- | --- | --- |
| PR1 | 900-1,200 (+ ~600 generated YAML) | ~400 of it is the mixin extraction, mostly moved lines |
| PR2 | 150-250 | plus benchmark |
| PR3 | 200-300 | |
| PR4 | per-architecture | |

PR1 is a 2-3 week item with 4-GPU debugging — #104's scale, because online RL forces
weight sync into the first PR.

## 6. Relationship to the omni track

The two tracks are **independent**, sharing only the "`nemo_automodel` is an optional
dependency" convention and the install docs:

| | Omni track ([`RFC_ISSUE.md`](RFC_ISSUE.md)) | Diffusion track (this RFC) |
| --- | --- | --- |
| Registry key | `(omni_model, automodel)` | `(diffusion_model, automodel)` |
| Entrypoint | `main_omni` | `main_diffusion` |
| nemo class | `NeMoAutoModelForMultimodalLM` | `NeMoAutoDiffusionPipeline` |
| nemo returns | `nn.Module` | `(pipe, managers)` tuple |
| Parallelism arg | `device_mesh` / `distributed_config` | `parallel_scheme` dict |
| Inheritance | verl `AutomodelEngineWithLMHead` — forward / optim / ckpt inherited | no inheritable base; see §4.4 |
| Forward | inherited from verl | verl-omni's `forward_and_sample_previous_step` |
| Example task | Qwen3-Omni Thinker offline DPO | Qwen-Image OCR FlowGRPO (online) |
| `model_engine` group | had to be created | **already exists, reusable** |
| Weight sync | inherited | must be written, required in-PR (§4.6) |
| Sequence parallelism | n/a | **new capability** (§2) |
| New code | ≈ 1,100 lines total | 900-1,200 in PR1 (excl. generated YAML) |
| Duration | ~1 week, GPU-bound | 2-3 weeks, 4-GPU debugging |

**Recommended order unchanged**: land the omni track first (its PR1 is written and its
risk is contained), then this one as a separate PR series.

## 7. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| nemo offload is sharding-time, not a toggle (§4.7.1) | colocated rollout OOM | implement `to()` against the sharded module directly; fall back to no offload with a smaller micro-batch |
| `layered_summon` × FSDP2 `full_tensor()` (§4.7.2) | weight-sync stall or OOM | bucketing already exists sender-side; bf16 cast + grouped gather if measurement shows a gap |
| nemo `Checkpointer` layout vs verl-omni resume (§4.7.3) | resumption | save/load round-trip in PR1 acceptance |
| D2 mixin extraction regresses the merged VeOmni engine | breaks a shipped backend | extraction is its own no-behaviour-change commit; existing VeOmni GPU smoke test is the gate |
| diffusers pins CP-plan availability per architecture | PR2 scope | Qwen-Image `_cp_plan` verified on diffusers 0.39.0; assert presence and error clearly if absent |
| `nemo_automodel` main moves under us | findings drift | this RFC pins `eca4a8e`; CI installs from a pin file, as the omni track does |

## 8. Verification

- **CPU tests** (`test_*_on_cpu.py` — CI selects on the suffix):
  - AST tests, runnable **without** verl or nemo_automodel installed: assert the
    `@EngineRegistry.register` key is `(diffusion_model, ["automodel"], ["cuda"])`, that
    the builder targets `NeMoAutoDiffusionPipeline` rather than a CausalLM class, that
    it reads `pipe.transformer`, and that no device string is hard-coded;
  - registry resolution (`importorskip`): `(diffusion_model, automodel, cuda)` →
    `AutomodelDiffusionEngine`;
  - config assembly (`importorskip("verl")`): `strategy == "automodel"`,
    `engine is automodel_config`, optimizer type, and `parallel_scheme` construction
    including the `cp_ring_degree > 1` rejection.
- **GPU smoke**: a test beside the CPU tests guarded by
  `importorskip("nemo_automodel")`, added to `tests/gpu_smoke/run_gpu_smoke_core.sh`;
  CI installs nemo via `.github/actions/gpu-smoke-prepare/action.yml`, which the omni
  track already extended.
- **Local limitation, stated plainly**: this worktree has `diffusers 0.39.0` but
  **neither `verl` nor `nemo_automodel` installed**, and no ruff. Everything in §4 was
  verified by reading a fresh clone of Automodel `main` at `eca4a8e` plus `python3 -c`
  probes against the installed diffusers; nothing here was verified by running nemo.
  `pre-commit` is the local gate.
- **GPU (manual)**: 4-GPU Qwen-Image OCR FlowGRPO, compared against the VeOmni recipe.

## 9. Your contribution

Nothing is implemented for this track yet — the omni track's PR1 is written and this
RFC is the design step that precedes any diffusion-track code. Before opening a PR,
per `CLAUDE.md`:

- run the duplicate-work checks: `gh issue view <n> --comments`,
  `gh pr list --state open --search "automodel"` / `"diffusion engine"`;
- every PR body must state why it does not duplicate existing work, the test commands
  run and their results, that AI assistance was used, and that a human reviewed every
  line;
- PR title format `[{modules}] {type}: {description}` (modules and types per
  `tests/special_sanity/check_pr_title.py`);
- commit trailers: `Co-authored-by:` and `Signed-off-by:`.

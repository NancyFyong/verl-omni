# [RFC] Add `nemo_automodel` as an engine backend for omni models

> Design draft, archived for reference.
> Code references are pinned to verl `8a694930275061f52ebd538c906ef8819af56dbd`
> and `nemo_automodel` `0.5.0+a3aa09bcc` (upstream rev `a3aa09bcc`, 2026-07-25).
>
> Structured against `.github/ISSUE_TEMPLATE/feature-request.yml` (the repo has no
> dedicated RFC template): §1 Feature request, §2 Motivation, §10 Your contribution.
> §3–§9 are the technical body.
>
> Scope: the **omni** track only. The `(diffusion_model, automodel)` track is a
> separate, independent RFC — [`RFC_ISSUE_DIFFUSION.md`](RFC_ISSUE_DIFFUSION.md).

## 1. Feature request

Make NVIDIA's [`nemo_automodel`](https://github.com/NVIDIA-NeMo/Automodel) a
selectable **engine backend** for omni models (`model_type="omni_model"`), chosen via
`actor.strategy="automodel"`, alongside the existing FSDP path.

Concretely: register a `(omni_model, automodel, cuda)` engine that subclasses verl's
`AutomodelEngineWithLMHead`, replacing **only** the model-construction step with
`NeMoAutoModelForMultimodalLM` + verl-omni's existing omni adapter
(`configure_model`). Everything else — optimizer, LR schedule, grad-clip, forward,
loss, checkpointing, offload, full-parameter weight sync — is inherited.

Scope of v1: **image + text** inputs, validated end-to-end with **offline DPO**.
`nemo_automodel` stays an **optional** dependency and the default FSDP composition is
untouched.

## 2. Motivation

verl-omni currently has exactly one engine path for omni multimodal LLMs:
`OmniFSDPEngine` (`backend=["fsdp","fsdp2"]`). Upstream verl ships a second backend,
`automodel` (`verl/workers/engine/automodel/`), which delegates model build,
parallelism, optimizer, LR schedule, grad-clip and checkpointing to `nemo_automodel`
while keeping verl's own training loop, data and loss. It is currently unusable for
omni models because:

- it is registered only for `model_type="language_model"` and hard-codes
  `NeMoAutoModelForCausalLM` (`automodel/utils.py::build_automodel_model`), so it
  cannot construct an omni/multimodal model;
- it is wired only into verl's SFT trainer — there is **no** `AutomodelActorConfig`
  for the PPO/DPO actor path the omni trainer uses;
- `_build_checkpointer` hard-codes `is_peft=False` (`transformer_impl.py:365`), i.e.
  LoRA is not plumbed on this backend.

**Benefit:** a second parallelism/optimizer stack for omni training (FSDP2 /
Megatron-FSDP, MoE expert parallelism, nemo's checkpointing, FP8 and `torch.compile`
paths), and an interface toward the NeMo ecosystem. This mirrors PR #104, which added
VeOmni as a second engine for diffusion models — the precedent this RFC deliberately
follows (§7).

## 3. Non-goals

- Modifying upstream verl. Everything is done by subclassing plus registering on the
  verl-omni side.
- Performance parity/tuning against `OmniFSDPEngine` (deferred to the §6 PR4 roadmap).
- Training the non-thinker stages (talker / code2wav). The existing stage-split
  convention is kept: only the LM head is trained.
- Replacing or deprecating the FSDP path. `automodel` is **opt-in** throughout; the
  default composition does not change.

## 4. Architecture

### 4.1 Where the backend plugs in

Engines are keyed `(model_type, backend, device)` in verl's `EngineRegistry`. The
selection seam is `verl_omni/workers/engine_workers.py:154`:

```python
EngineRegistry.new(
    model_type=self.config.model_type,     # → "omni_model"
    backend=self.engine_config.strategy,   # → "automodel"
    model_config=..., engine_config=..., optimizer_config=..., checkpoint_config=...,
)
```

The actor branch (`engine_workers.py:632-645`) builds `TrainingWorkerConfig` from
`actor_config.model_config["model_type"]`, `.engine`, `.optim` and `.checkpoint` — it
**never** reads `fsdp_config`. So routing to `(omni_model, automodel)` requires
exactly two things: an engine registered under that key, and an actor config whose
`.engine.strategy == "automodel"`. `.new()` does not pass `device`, so it falls back
to `cuda`, consistent with the existing FSDP omni path.

```text
                      actor_rollout_ref.actor.strategy
                                    │
              ┌─────────────────────┴─────────────────────┐
              │ "fsdp" / "fsdp2"            "automodel"   │
              ▼                                           ▼
  ┌───────────────────────────┐          ┌───────────────────────────────┐
  │ OmniActorConfig           │          │ OmniAutomodelActorConfig      │
  │  (FSDPActorConfig)        │          │  (verl ActorConfig)           │
  │  .engine = .fsdp_config   │          │  .engine = .automodel         │
  └────────────┬──────────────┘          └───────────────┬───────────────┘
               │                                         │
               │      EngineRegistry.new(model_type="omni_model",
               │                         backend=<strategy>)
               ▼                                         ▼
  ┌───────────────────────────┐          ┌───────────────────────────────┐
  │ OmniFSDPEngine            │          │ OmniAutomodelEngine           │
  │  (FSDPEngineWithLMHead)   │          │  (AutomodelEngineWithLMHead)  │
  │  device=[cuda, npu]       │          │  device=[cuda]                │
  └────────────┬──────────────┘          └───────────────┬───────────────┘
               │                                         │
               ▼                                         ▼
  AutoModelForMultimodalLM              NeMoAutoModelForMultimodalLM
        .from_pretrained                      .from_pretrained
               │                                         │
               └──────────────────┬──────────────────────┘
                                  ▼
             OmniModelBase.get_class_by_name(architecture, stage)
                          .configure_model(module, cfg)
                    ── the SAME omni adapter on both paths ──
```

### 4.2 Inheritance surface — what is actually written

`OmniAutomodelEngine → AutomodelEngineWithLMHead → AutomodelEngine → BaseEngine`.
The table below marks **real new code** versus inherited behaviour; it is what drives
the PR split in §6.

| Capability | Source | Notes | Phase |
| --- | --- | --- | --- |
| Registration / routing | **new** (1 decorator) | `@EngineRegistry.register(model_type="omni_model", backend=["automodel"], device=["cuda"])` | PR1 |
| Model construction | **new** (the only substantive change) | `NeMoAutoModelForMultimodalLM.from_pretrained` + `configure_model` | PR1 |
| Actor config | **new** (dataclass) | `OmniAutomodelActorConfig`, `engine = AutomodelEngineConfig` | PR1 |
| Optimizer / LR / grad-clip | inherited | `_build_optimizer`, `_build_lr_scheduler` (nemo `OptimizerParamScheduler`) | PR1 |
| forward / loss / train step | inherited | `AutomodelEngineWithLMHead.forward_step`; `prepare_model_inputs` already merges `multi_modal_inputs` and handles 3-D mrope | PR1 |
| Checkpoint save / load | inherited | nemo `Checkpointer` (`transformer_impl.py:357-414`) | PR1 |
| Param / optimizer offload | inherited | `load/offload_automodel_model_to_{gpu,cpu}` | PR1 |
| image + text inputs | inherited | LM-head forward path, no override needed | PR1 |
| Offline DPO | inherited | does not trigger rollout weight sync → end-to-end in PR1 | PR1 |
| Full-parameter weight sync | inherited (§5) | `AutomodelEngine.get_per_tensor_param` (`transformer_impl.py:423`) | PR2 |
| Online RL (GSPO / GRPO) | inherited + validation | depends on the above; the work is GPU validation, not implementation | PR2 |
| bf16 export cast | **new** (thin override, optional) | base class does not cast → larger sync payload | PR2 |
| LoRA | **new** (the real gap) | base returns `peft_config=None`; `is_peft=False` hard-coded | PR3 |
| Megatron-FSDP / EP>1 / audio·video | **new** | expressible in config, unvalidated | PR4 |

In one line: **PR1's substantive new code is "swap the model class + assemble the
config"; PR2 is mostly validation; PR3 is the second real implementation task.**

### 4.3 Engine

`verl_omni/workers/engine/automodel/omni_impl.py`:

```python
@EngineRegistry.register(model_type="omni_model", backend=["automodel"], device=["cuda"])
class OmniAutomodelEngine(AutomodelEngineWithLMHead):
    def initialize(self):
        # line-for-line equivalent to AutomodelEngine.initialize, except the build step
        self.module = build_automodel_omni_model(...)
        # then: _build_optimizer / maybe_fully_shard_optimizer /
        #       _build_lr_scheduler / _build_checkpointer / to(cpu, offload...)
```

`build_automodel_omni_model()` mirrors verl's `build_automodel_model` with two
differences: the model class becomes `NeMoAutoModelForMultimodalLM`, and the omni
adapter is applied after construction (isomorphic to `OmniFSDPEngine._build_module`):

```python
adapter_cls = OmniModelBase.get_class_by_name(architecture, model_stage, external_lib)
return adapter_cls.configure_model(module, model_config)
```

Remaining kwargs (`force_hf`, `BackendConfig`, `MoEParallelizerConfig`, FP8, compile,
`attn_implementation`, `torch_dtype`) match upstream so behaviour stays predictable.
verl's `build_distributed_config_from_engine_config(...)` is reused as-is —
`from_pretrained` lives on the shared `_BaseNeMoAutoModelClass`, so the multimodal
class accepts the same `device_mesh` / `moe_mesh` / `distributed_config` kwargs as the
CausalLM class.

### 4.4 Configuration

`OmniAutomodelActorConfig` must subclass verl's **base** `ActorConfig`, **not**
`OmniActorConfig` — the latter's parent `FSDPActorConfig.__post_init__` sets
`self.engine = self.fsdp_config` and asserts an FSDP strategy.

```python
@dataclass
class OmniAutomodelActorConfig(ActorConfig):
    strategy: str = "automodel"
    automodel: AutomodelEngineConfig = field(default_factory=AutomodelEngineConfig)
    optim: AutomodelOptimizerConfig = field(default_factory=AutomodelOptimizerConfig)
    trainer_type: str = "direct_preference"
    omni_loss: OmniLossConfig = field(default_factory=OmniLossConfig)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.automodel
        object.__setattr__(self.engine, "strategy", self.strategy)
        ...  # trainer_type / omni_loss validation, same as OmniActorConfig
```

`trainer_type` and `omni_loss` duplicate `OmniActorConfig`. v1 deliberately does not
extract a mixin: dataclass field ordering across a mixin plus two different bases is
error-prone and the benefit does not justify the risk.

### 4.5 Optional dependency and reachability

- **Registration**: `verl_omni/workers/engine/__init__.py` registers under
  `try/except ImportError` (same pattern as VeOmni diffusion). When `nemo_automodel`
  is absent the engine is simply not registered and no existing path is affected.
- **Selection**: a Hydra **group override**, not a CLI override of `actor._target_`.
  Reason: `omega_conf_to_dataclass` goes through `hydra.utils.instantiate`, and
  `BaseConfig` is a plain dataclass that rejects unknown kwargs. Swapping only
  `_target_` leaks the FSDP keys inherited from `dp_actor` (`fsdp_config`,
  `grad_clip`, `ulysses_*`) into the constructor and raises `TypeError`. A group
  override **replaces the whole actor node**, so the example ships a self-contained
  actor group file.
- **No autogen churn**: `omni_trainer.yaml`'s `defaults:` is untouched, so
  `_generated_omni_trainer.yaml` does not change.

### 4.6 Principal unknown (GPU-only)

nemo's `from_pretrained` performs FSDP2 parallelization internally, while the omni
adapter's `configure_model` does module surgery: re-registering under
`AutoModelForCausalLM`, redirecting `forward` to the thinker, forcing
`tie_word_embeddings=False`, and unfusing MoE. **Their ordering and mutual
compatibility is the one hard unknown** in this integration. Build logic is
deliberately confined to the single `build_automodel_omni_model` seam so it can be
adjusted per actual nemo/verl version (if needed: build bare model → `configure_model`
→ parallelize).

## 5. Data flow

### 5.1 Offline DPO (PR1 scope — no rollout, no weight sync)

This is why the v1 example is offline DPO: it exercises build → forward → loss →
optimizer → checkpoint without touching `get_per_tensor_param`, so PR1 is verifiable
end-to-end without depending on PR2.

```text
  Omni-Preference parquet (image split)
  chosen / rejected preference pairs
             │
             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ OmniDirectPreferenceRayTrainer      (verl-omni, unchanged)   │
  └─────────────────────────┬───────────────────────────────────┘
                            │ TensorDict micro-batch
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ OmniAutomodelEngine.forward_step        [INHERITED]          │
  │                                                              │
  │  prepare_model_inputs ── merges multi_modal_inputs           │
  │        │                  handles 3-D mrope position_ids     │
  │        ▼                                                     │
  │  self.module(**inputs) ── nemo-built, FSDP2-sharded          │
  │        │                  forward → thinker (via adapter)    │
  │        ▼                                                     │
  │  prepare_model_outputs ── logits → per-token log-probs       │
  └─────────────────────────┬───────────────────────────────────┘
                            │ log_probs (policy + frozen ref)
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ OmniDPOLoss   (verl-omni, unchanged)                         │
  │   beta / label_smoothing / sigmoid, ref built in-engine      │
  │   per omni_loss.refer_model_precision                        │
  └─────────────────────────┬───────────────────────────────────┘
                            │ loss → backward
                            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ optimizer_step / lr_scheduler_step / grad-clip [INHERITED]   │
  │   nemo OptimizerParamScheduler; offload to CPU between steps │
  └─────────────────────────┬───────────────────────────────────┘
                            ▼
              nemo Checkpointer  (save / load round-trip)
```

Only the shaded-in-prose `self.module` construction is new code. Every other box is
inherited from verl's automodel engine or is existing verl-omni code.

### 5.2 Online RL weight sync (PR2 scope)

After each training step the updated policy weights are pushed to the vLLM-Omni
rollout, driven from `engine_workers.py::update_weights`:

```text
  ┌───────────────────────────────────────────────────────────────┐
  │ OmniAutomodelEngine.get_per_tensor_param()      [INHERITED]   │
  │   defined on AutomodelEngine (transformer_impl.py:423)        │
  │                                                                │
  │   load params to GPU                                           │
  │        ▼                                                       │
  │   state_dict()                                                 │
  │        ▼                                                       │
  │   DTensor? ── yes ──▶ .full_tensor()   (all-gather → unsharded) │
  │        ▼                                                       │
  │   convert_weight_keys()   → HF key names vLLM expects          │
  │        ▼                                                       │
  │   offload back to CPU                                          │
  │        ▼                                                       │
  │   yield (name, tensor)  ─┬─▶  peft_config = None               │
  └──────────────────────────┼────────────────────────────────────┘
                             ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ rollout.update_weights(per_tensor_param, peft_config)          │
  │        ▼                                                       │
  │ BucketedWeightSender ── buckets by MB, ZMQ / IPC (opt. shm)    │
  │        ▼                                                       │
  │ vLLM-Omni: update_weights_from_ipc()                           │
  └───────────────────────────────────────────────────────────────┘
```

Two invariants: parameter names must match the HF keys vLLM expects (handled by
`convert_weight_keys`), and tensors must be **fully unsharded** (`DTensor` requires
`.full_tensor()`).

### 5.3 Comparison against the three existing implementations

| | verl `AutomodelEngine`<br>(`transformer_impl.py:423`) | this repo, VeOmni<br>(`veomni/diffusion_impl.py:587`) | this repo, `OmniFSDPEngine`<br>(`fsdp/omni_impl.py:46`) |
| --- | --- | --- | --- |
| Skeleton | load → `state_dict` → `convert_weight_keys` → offload → generator | same | same |
| DTensor | `.full_tensor()` | `.full_tensor()` | `.full_tensor()` |
| dtype cast | **none** | casts to `engine_config.model_dtype` | casts bf16 (with a TODO: MoE gate needs finer granularity) |
| Key handling | `convert_weight_keys` only | additionally prefixes `transformer.` | `replace_lora_wrapper` / `normalize_peft_param_name` |
| LoRA | **unsupported**, `peft_config=None` | explicit `NotImplementedError` | full: `collect_lora_params` + merged-LoRA + layered summon |
| Returns | `(gen, None)` | `(gen, None)` | `(gen, peft_config_dict)` |

`OmniFSDPEngine` additionally supports `base_sync_done` / `layered_summon` /
`adapter_name` (two-stage base + adapter sync for LoRA) and hooks up QAT quantization.

**Conclusion for automodel-omni: full-parameter sync is inherited, not written.**
`get_per_tensor_param` is defined on `AutomodelEngine`, so `OmniAutomodelEngine` has
it for free. The `layered_summon` / `base_sync_done` / `adapter_name` arguments passed
by `engine_workers.py` are absorbed by the base class's `**kwargs`; on the non-LoRA
path `peft_config=None` → `do_lora_base_sync=False` → a single ordinary
`update_weights`.

PR2 is therefore **validation plus a thin override**, not an implementation:

- Structurally usable as-is.
- Requires GPU verification: (a) nemo's FSDP2 sharding yields standard `DTensor` and
  `.full_tensor()` all-gathers correctly; (b) after the omni adapter has rewritten
  forward/embeddings, `state_dict()` keys still match what vLLM-Omni expects (the base
  class does call `convert_weight_keys`, so this is likely fine — but must be measured).
- Optional thin override: bf16 cast to shrink the sync payload (performance, not
  correctness).
- LoRA is a genuine gap → PR3.

## 6. Implementation plan

Three PRs to feature-complete, plus one roadmap item. Each is independently
reviewable, testable and revertible.

### PR1 — engine + config + registration (v1: image+text, offline DPO runnable)

| File | Action | Lines |
| --- | --- | --- |
| `verl_omni/workers/engine/automodel/omni_impl.py` | ADD `OmniAutomodelEngine` + `build_automodel_omni_model` | 151 |
| `verl_omni/workers/engine/automodel/__init__.py` | ADD exports | 17 |
| `verl_omni/workers/engine/__init__.py` | EDIT conditional `try/except ImportError` registration | +8 |
| `verl_omni/workers/config/omni/actor.py` | EDIT `OmniAutomodelActorConfig` + `__all__` | +35 |
| `verl_omni/workers/config/omni/__init__.py` | EDIT explicit re-export | +4/-2 |
| `tests/special_sanity/check_device_api_usage.py` | EDIT whitelist the `device=[...]` declaration line | +1 |
| `examples/automodel_trainer/qwen3_omni/**` | ADD offline-DPO recipe (trainer YAML + actor group + run.sh) | 213 |
| `tests/workers/test_omni_automodel_engine_on_cpu.py` | ADD AST registration test (runs without deps) + registry resolution | 146 |
| `tests/workers/config/test_omni_automodel_actor_config_on_cpu.py` | ADD config assembly test | 65 |
| `docs/contributing/integrating_an_omni_model.md` | EDIT new "Engine backend selection" section | +28 |
| **Written so far** | | **≈ +660 / 12 files** |
| `docs/start/install.md` | EDIT "Optional engine backends": `nemo_automodel` install | ~25 |
| `tests/workers/test_omni_automodel_engine.py` | ADD GPU smoke (`importorskip` self-skips) | ~150 |
| `tests/gpu_smoke/run_gpu_smoke_tests.sh`, `.github/workflows/gpu_smoke.yml` | EDIT register the smoke item + install the optional dep | ~13 |
| **Remaining** | | **≈ +190 / 4 files** |

**Acceptance**: `pre-commit` green; CPU tests pass; offline DPO trains for several
steps on GPU with decreasing loss and a successful checkpoint save/load round-trip;
the GPU smoke item passes with `nemo_automodel` installed and skips without it.
**Depends on**: nothing.
**Risk**: the §4.6 ordering problem surfaces here first. That is deliberate — the
smallest PR carries the largest unknown.

**Why the example is offline DPO**: `omni_trainer.yaml` already defaults to
`trainer_type=direct_preference` / `sample_source=offline` / `paired_preference=true` /
`adv_estimator=dpo`, so the example needs **no** `algorithm` overrides; and offline DPO
does **not** trigger `get_per_tensor_param`, so build / forward / loss / optimizer /
checkpoint are validated end-to-end without depending on PR2. Data reuses the existing
Omni-Preference pipeline
(`examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py --modalities image`
→ `parquet_dpo/image/{train,test}.parquet`) — no new data script.

### PR2 — online RL weight sync (validation + thin override)

- GPU-verify the inherited `get_per_tensor_param`: DTensor all-gather, key-name match,
  memory and wall-clock cost.
- Add the bf16 cast thin override if measurements justify it (including the MoE-gate
  dtype exception).
- Bring up online RL (GSPO / GRPO); add an online GSPO recipe.
- Add a CPU test asserting that `OmniAutomodelEngine`'s resolved
  `get_per_tensor_param` comes from the expected implementation, so an upstream
  refactor cannot silently change behaviour.

**Acceptance**: short online GSPO run with rising reward; rollout output consistent
with actor weights (sync effective). **Depends on**: PR1. **Size**: ≈ +80 lines — the
cost here is GPU time, not code.

### PR3 — LoRA support and sync parity

- LoRA injection (mirroring `OmniFSDPEngine._build_lora_module`, including
  `lora_dtype` conversion).
- Override `get_per_tensor_param`: `collect_lora_params` + `replace_lora_wrapper`,
  merged-LoRA generator, `layered_summon` / `base_sync_done` / `adapter_name` semantics.
- Pass a real `is_peft` to `_build_checkpointer` (upstream hard-codes `False`).

**Acceptance**: LoRA online RL runs; two-stage base+adapter sync is correct; both
merged and non-merged paths tested. **Depends on**: PR2. **Size**: ≈ +250 lines plus
tests.

### PR4 (roadmap) — scale and modality

`distributed_strategy=megatron_fsdp`, `ep_size>1` (MoE EP), audio/video modalities,
GPU benchmarks and performance comparison. May be split further.

### Effort summary

| Phase | New code | Bottleneck |
| --- | --- | --- |
| PR1 remaining | ≈ +190 | writing; ~1 day |
| PR2 | ≈ +80 | 4-GPU validation, not code |
| PR3 | ≈ +250 + tests | implementation; 2-3 days |
| **omni track total** | **≈ 1,100 lines** | ~1 week, GPU-bound |

## 7. Precedent: how the VeOmni engine was merged (PR #104)

This repo already has one complete precedent for integrating a third-party engine
backend, worth both aligning with and consciously diverging from.

**PR #104 `[trainer] feat: Support VeOmni as actor engine`** — 28 files, +1983/-79,
merged as a **single** PR (opened 2026-05-21, merged 2026-06-05). It runs **Qwen-Image
OCR FlowGRPO** (online RL) on 4 GPUs, reward via Qwen3-VL-8B-Instruct as GenRM, 512
resolution, rollout through `vllm_omni`, `layered_summon=True`.

Its composition, by measured size:

| Component | Lines |
| --- | --- |
| `verl_omni/workers/engine/veomni/diffusion_impl.py` (from `BaseEngine`, **not** a subclass of the FSDP engine) | 640 |
| `_generated_diffusion_veomni_trainer.yaml` (**generated**, not hand-written) | 540 |
| `tests/workers/test_diffusers_veomni_engine.py` (GPU smoke, `importorskip`) | 203 |
| Main-tree config YAMLs (`optim/`, `engine/`, `actor/`, `ref/`, `model_engine/`, `model/`) | 197 |
| Config dataclasses (`config/diffusion/{actor,model}.py`) | 82 |
| Docs (`install.md`, `performance.md`, `integrating_a_diffusion_model.md`) | 97 |
| `run_qwen_image_ocr_veomni.sh` | 93 |
| CI wiring (`gpu_smoke.yml`, `run_gpu_smoke_tests.sh`, `generate_trainer_config.sh`) | 14 |

**Selection mechanism**: the diffusion tree has a `model_engine` group —
`diffusion_trainer.yaml` declares `- diffusion/model_engine: dp_diffusion`, and
actor/ref derive from it by interpolation (`${diffusion/model_engine}_actor`).
Switching backend is one flag, `diffusion/model_engine=veomni_diffusion`, and it flips
actor + ref + optim + engine together. CI installs `veomni==0.1.11 --no-deps` to avoid
clashing with the vllm/torch stack.

### Deliberate divergences

| | VeOmni (#104) | This RFC |
| --- | --- | --- |
| Engine starting point | fresh from `BaseEngine` (640 lines) | subclass `AutomodelEngineWithLMHead`, swap the build step only |
| Config location | main config tree + new `_generated_*veomni*.yaml` | examples dir, deliberately avoiding autogen churn |
| Selection | `model_engine` group (one flag, 4 nodes) | `override /actor@...` actor group |
| Weight sync | delivered in the same PR | deferred to PR2 (mostly inheritable, §5.3) |
| Shape | single PR, 28 files | 3 PRs |

**Why diverge**: (a) verl's automodel engine already provides the vast majority of the
capability including full-parameter weight sync, so rewriting from `BaseEngine` would
be waste; (b) the omni tree has **no** `model_engine` group — that is diffusion-specific
— and introducing one requires editing `omni_trainer.yaml`'s `defaults:` plus adding a
`_generated_omni_automodel_trainer.yaml` target, a larger blast radius best done after
the automodel path is GPU-validated.

Two #104 practices this RFC does adopt: a GPU smoke item that self-skips via
`importorskip("nemo_automodel")` with on-demand install in `gpu_smoke.yml` (watch for
torch/vllm conflicts — may likewise need `--no-deps`), and an "Optional engine
backends" entry in `docs/start/install.md`.

### Open question: introduce an `omni/model_engine` group?

The `model_engine` mechanism is the nicer UX (`omni/model_engine=automodel` flips
actor / ref / optim / engine at once), at the cost of touching the default composition
and adding an autogen target.

- **Option A (current)**: actor group override inside `examples/`. Zero autogen churn,
  smallest PR1.
- **Option B**: introduce `omni/model_engine` in PR1, fully aligned with #104.
- **Option C (recommended)**: ship PR1 as option A to get the engine working, then
  after PR2/PR3 do a separate config-refactor PR to upgrade to a `model_engine` group
  — by then automodel is validated and changing the default composition is lowest-risk.

Note the diffusion tree's `model_engine` group **already exists**
(`verl_omni/trainer/config/diffusion/model_engine/{dp_diffusion,veomni_diffusion}.yaml`);
the omni tree's does not. So the diffusion track can **reuse** the mechanism
outright — that path actually hews closer to the VeOmni precedent than this one does.

## 8. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| nemo `from_pretrained` vs `configure_model` ordering conflict (§4.6) | blocks PR1 | confined to a single seam; can switch to bare-build → configure → parallelize |
| `state_dict()` keys mismatch vLLM-Omni | blocks PR2 | base class already calls `convert_weight_keys`; add key mapping if measurement shows a gap |
| Per-tensor `full_tensor()` memory/latency under MoE + FSDP2 | online RL performance | bucketing already exists sender-side; bf16 cast + grouped gather if needed |
| nemo `Checkpointer` format vs existing omni resume flow | resumption | PR1 acceptance includes a save/load round-trip |
| Upstream verl refactors the automodel engine | inherited behaviour drifts | PR2's inheritance-source assertion test |

## 9. Verification

- **CPU tests** (`test_*_on_cpu.py` — CI selects on the suffix):
  - AST tests (**run without verl or nemo_automodel installed**): validate the
    `@EngineRegistry.register` `model_type` / `backend` / `device`, the base class, that
    the builder targets `NeMoAutoModelForMultimodalLM` rather than CausalLM, and that
    no device is hard-coded;
  - registry resolution (`importorskip`): `(omni_model, automodel, cuda)` →
    `OmniAutomodelEngine`;
  - config assembly (`importorskip("verl")`): `strategy == "automodel"`,
    `engine is automodel`, `engine.strategy == "automodel"`, `optim` type, and
    `trainer_type` / `omni_loss` validation.
- **Local limitation (stated plainly)**: this worktree has neither `verl` nor
  `nemo_automodel` installed, so tests needing them skip; `pre-commit` (12 hooks: ruff,
  format, mypy, license, naming, compileall, autogen, …) is the local gate.
- **GPU (manual; per-PR acceptance in §6)**: short runs on a small Qwen3-Omni thinker
  checkpoint.

## 10. Your contribution

Implementation is underway on a fork; PR1's engine, config, registration, example
recipe and CPU tests are written (≈ +660 lines / 12 files, §6). Before opening any PR,
per `CLAUDE.md`:

- run the duplicate-work checks: `gh issue view <n> --comments`,
  `gh pr list --state open --search "automodel"` / `"omni engine"`;
- every PR body must state: why it does not duplicate existing work, the test commands
  run and their results, that AI assistance was used, and that a human reviewed every
  line;
- PR title format `[{modules}] {type}: {description}` (modules and types per
  `tests/special_sanity/check_pr_title.py`);
- commit trailers: `Co-authored-by:` and `Signed-off-by:`.

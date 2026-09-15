# Offline Diffusers model publishing

Last updated: 09/15/2026

`verl_omni.model_merge` converts an existing FSDP actor checkpoint into a local,
Diffusers transformer or self-contained Diffusers pipeline. It follows verl's `ModelMergerConfig`,
`merge_and_save()` and `cleanup()` lifecycle without routing Diffusers configs
through Transformers `AutoConfig`.

## Supported architectures

One exporter handles every Diffusers-style training architecture currently
registered in the repository, independently of the training algorithm:

| Architecture | Canonical transformer | `transformer` output | `pipeline` output |
| --- | --- | --- | --- |
| Qwen-Image | QwenImageTransformer2DModel | Yes | Yes |
| Qwen-Image-Edit Plus | QwenImageTransformer2DModel | Yes | Yes, including processor |
| SD3 / SD3.5 | SD3Transformer2DModel | Yes | Yes, including all three text encoders/tokenizers |
| Flux | FluxTransformer2DModel | Yes | Yes |
| Wan / Wan2.2 | WanTransformer3DModel | Yes | Yes, select the trained transformer |
| LTX-2 | LTX2VideoTransformer3DModel | Yes | Yes, including audio VAE, connectors and vocoder |
| MiniMax H3 | MiniMaxH3Transformer3DModel | Yes | No: native inference layout differs |
| Boogu-Image | BooguImageTransformer2DModel (`boogu-image`) | Yes | No: external custom pipeline |

`--output-format transformer` writes a standalone component loadable with its
canonical class's `from_pretrained()`. `--output-format pipeline` (default)
replaces the selected complete transformer and preserves all other base
components and assets. Missing actor parameters are **never** filled from the base.

MiniMax H3 requires the **Diffusers actor transformer** as its base, not the
native/fused `MiniMaxH3DiTModel` used by vLLM-Omni. Its component export does not
convert parameter names into a native vLLM-Omni inference package. Boogu resolves
the installed canonical class, never a Python shim inside the checkpoint.
BAGEL is deliberately excluded: its training class derives from
`NonDiffusersModelBase` and needs native publishing.

Supported checkpoint representations:

- One-dimensional FSDP2 DTensor `Shard(d)` and `Replicate`, including uneven and
  empty shards. All mesh coordinates, shapes, placements and dtypes are checked.
- Ordinary full tensors in a one-rank checkpoint.
- Ordinary full-shaped tensors in a multi-rank checkpoint only when every replica
  is exactly equal. Plain dim-0 shards are not guessed or concatenated.

Not supported yet: adapter-bearing/LoRA checkpoints, FSDP1 `ShardedTensor`,
HSDP/FSDP+TP, quantized weights, remote/custom full pipelines, architectures
outside the audited table, BAGEL and Omni publishing. Standard Transformers can
continue using `python -m verl.model_merger`; delegation through this entrypoint
is a follow-up. A training engine named `diffusers` does not establish that its
custom model uses a Diffusers publishing layout.

## Inputs

Prepare three non-overlapping local directories: the saved actor, the compatible
base pipeline (or standalone transformer for component output), and an **absent**
output directory whose parent already exists.

```text
actor/
  fsdp_config.json
  model_world_size_2_rank_0.pt
  model_world_size_2_rank_1.pt
  huggingface/config.json

base/
  model_index.json
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors[.index.json]
  vae/...
  text_encoder/...
  tokenizer/...
  scheduler/...
```

The transformer config saved during training must agree with the base's
behavior-affecting config. The exporter constructs a tiny-memory meta-device
schema and checks exact trained keys and shapes. The base must contain standard
safetensors files/indexes. Frozen assets are copied, not downloaded. Hub-cache
file symlinks are dereferenced into regular output files; directory symlinks and
custom Python files are rejected for full pipeline output. Component export reads
only the selected config and safetensors, omitting unused assets/Python shims.

Only trusted checkpoint files may be opened: torch checkpoint deserialization is
pickle-based. `--trust-checkpoint` explicitly acknowledges this; it is **not a
sandbox**. Do not modify source/base files while exporting. Source fingerprints
are checked again before publication.

## CLI and Python

```bash
python -m verl_omni.model_merge merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base-model "$BASE_PIPELINE" \
  --architecture QwenImagePipeline \
  --trust-checkpoint

python -m verl_omni.model_merge validate --target_dir "$OUTPUT"
```

For a standalone component (including MiniMax H3 and Boogu):

```bash
python -m verl_omni.model_merge merge \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base-model "$DIFFUSERS_TRANSFORMER" \
  --architecture MiniMaxH3Pipeline \
  --output-format transformer \
  --trust-checkpoint
```

Component output may also select a component from a full base pipeline. The
output contains `config.json`, standard Diffusers safetensors and the manifest,
not a misleading `model_index.json`.

`--architecture` is optional; when supplied it must agree with the pipeline index
and canonical transformer class. A standalone Qwen transformer cannot distinguish
T2I from Edit, so its inferred label is QwenImagePipeline; supply the Edit label
explicitly when that distinction matters.

Wan pipelines containing both `transformer` and `transformer_2` **require** an
explicit `--component transformer` or `--component transformer_2`. Only that
component is replaced. The other model and `boundary_ratio` / `expand_timesteps`
are preserved. To publish two independently trained checkpoints, export them in
two passes into distinct output directories, using the first output as the second
base. One actor checkpoint is never duplicated into both slots.

The CLI does not expose an algorithm choice. FlowGRPO, NFT and distribution
matching use the same publisher when they save the same complete transformer.

```python
from verl_omni.model_merge import ModelMergerConfig, merge_model, validate_artifact

result = merge_model(ModelMergerConfig(
    local_dir=checkpoint_dir,
    target_dir=output_dir,
    base_model=base_pipeline_dir,
    trust_checkpoint=True,
))
validate_artifact(result.output_dir)
```

The default dtype is **preserve**. `--dtype float32`, `float16` or `bfloat16`
explicitly casts checkpoint-derived floating tensors, except declared fp32
islands. Integer/bool buffers and copied frozen base weights are never cast.
Non-finite source values and overflow during casting fail export.

`--max-shard-size` is an output safetensors accumulation budget **in bytes**
(default 2 GiB). An individual larger tensor gets its own shard. Rank archives
are mmap-loaded on CPU and tensors are reconstructed one at a time; this avoids
retaining a complete merged transformer, but it is **not a hard RSS limit**.
Mapped-page residency, reconstruction, serializer/verification copies and source
metadata consume additional memory. No fallback to eager loading is performed
for unsupported serialization.

## Verification and failure semantics

Before publication the exporter:

1. Verifies pipeline components, transformer configs, base headers and exact
   trained tensor coverage.
2. Reconstructs DTensors and checks their slices against the original shards;
   replica copies must agree exactly.
3. Reopens every written safetensors shard and checks exact post-cast values,
   shapes and dtypes against its intended tensors.
4. Checks unchanged copied assets, input immutability, output indexes and SHA256
   inventories.

The portable `merge_manifest.json` records the selected architecture, source/base
fingerprints, output tensor/file inventories, dtype and verification outcomes.
Known location-only config metadata (`_name_or_path`, `name_or_path`) is removed
with a recorded deterministic transform. No source directory is recorded in the
manifest. SHA256 validates consistency, not publisher authenticity.

`validate` checks the published files, checksums, tensor metadata and index
without loading source rank checkpoints. It does not rerun training or certify
past source round-trip claims if the artifact and manifest are both replaced.

Publication uses an exclusive lock and owned sibling staging. Linux
`renameat2(RENAME_NOREPLACE)` prevents replacing even an empty directory created
concurrently. Other platforms fail explicitly. Errors clean only this export's
staging/lock, never the source or another exporter's files. Existing targets are
not overwritten; automatic crash recovery/resume is not implemented.

Runtime validation is recorded as `not_run`: there is no implicit generation or
GPU use during export. CPU tests exercise genuine two-rank Gloo DTensor
serialization, fresh-process CLI export, tiny complete pipeline reload and
transformer-forward parity across the eight architecture identities above.
All eight also have dtype/fp32-island and incomplete-state checks; the six standard
pipelines exercise their real complete `from_pretrained()` loader. Wan additionally
covers its second transformer and non-component options. A registry coverage test
fails if a new Diffusers training architecture is added without exporter coverage.

Run the matrix with the project's venv and optional `boogu-image` dependency:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/model_merge/test_architectures_on_cpu.py -v
```

Without `boogu-image`, only its cases are skipped; this is not evidence that Boogu
was tested. Local verification used Diffusers 0.40.0 and canonical Boogu source
revision `25f8f888298224a94e5ec2abafb98abea9031a0d` on PYTHONPATH, without changing
the shared venv. Boogu's declared dependency ranges differ from that venv; source
execution is not dependency-resolver or clean-install validation.

These tests are not real-weight/GPU FSDPCheckpointManager
or production image-quality evidence. Real checkpoint save/export/load remains
a release acceptance gate.

For a manual loader check after export:

```python
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(output_dir, local_files_only=True)
```

See [RFC #596](https://github.com/verl-project/verl-omni/issues/596) for the
subsequent LoRA, Transformers reuse, BAGEL and stage-specific work packages.

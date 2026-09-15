# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Publish complete, audited Diffusers transformers with unchanged base components."""

import inspect
import os
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.tensor import DTensor, Replicate, Shard

from .base_model_merger import BaseModelMerger, MergeResult
from .utils import (
    MANIFEST_NAME,
    fingerprint,
    inventory,
    publication_directory,
    read_json,
    tree_files,
    weight_files,
    write_json,
    write_weights,
)

_TRANSFORMERS = {
    "QwenImagePipeline": "QwenImageTransformer2DModel",
    "QwenImageEditPlusPipeline": "QwenImageTransformer2DModel",
    "StableDiffusion3Pipeline": "SD3Transformer2DModel",
    "FluxPipeline": "FluxTransformer2DModel",
    "WanPipeline": "WanTransformer3DModel",
    "LTX2Pipeline": "LTX2VideoTransformer3DModel",
    "MiniMaxH3Pipeline": "MiniMaxH3Transformer3DModel",
    "BooguImagePipeline": "BooguImageTransformer2DModel",
}
_PIPELINES = frozenset(_TRANSFORMERS) - {"MiniMaxH3Pipeline", "BooguImagePipeline"}


def _transformer_class(architecture: str):
    if architecture not in _TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    module = "boogu.models.transformers.transformer_boogu" if architecture == "BooguImagePipeline" else "diffusers"
    from importlib import import_module

    return getattr(import_module(module), _TRANSFORMERS[architecture])


def _resolve_architecture(index: dict | None, config: dict, explicit: str | None) -> str:
    inferred = index.get("_class_name") if index is not None else None
    if inferred is not None and explicit is not None and inferred != explicit:
        raise ValueError("Explicit architecture conflicts with base model_index.json")
    architecture = explicit or inferred
    if architecture is None:
        architecture = next((key for key, value in _TRANSFORMERS.items() if value == config.get("_class_name")), None)
    if architecture not in _TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    if config.get("_class_name") != _TRANSFORMERS[architecture]:
        raise ValueError("Base must contain the canonical Diffusers transformer, not native/fused inference weights")
    return architecture


def model_rank_files(root: Path) -> list[Path]:
    """Require exactly one model shard per declared rank, ignoring optimizer state."""
    metadata = read_json(root / "fsdp_config.json")
    world_size = metadata.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("fsdp_config.json must declare a positive integer world_size")
    if type(metadata.get("FSDP_version")) is not int or metadata["FSDP_version"] not in (1, 2):
        raise ValueError("Unsupported FSDP checkpoint version")
    expected = [root / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    if set(root.glob("model_world_size_*_rank_*.pt")) != set(expected) or not all(p.is_file() for p in expected):
        raise ValueError("Missing or unexpected model rank files")
    return expected


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    local = value.to_local() if isinstance(value, DTensor) else value
    if type(local) is not torch.Tensor:
        raise ValueError("Only plain tensors and one-dimensional DTensors are supported; ShardedTensor is unsupported")
    if local.device.type != "cpu" or local.layout != torch.strided or local.is_quantized or local.is_complex():
        raise ValueError("Expected a dense, real, non-quantized CPU tensor")
    if local.is_floating_point() and not torch.isfinite(local).all():
        raise ValueError("Checkpoint contains non-finite weights")
    return local


def reconstruct_tensor(values: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    """Reconstruct a tensor and verify exact pre-cast round-trip to every local shard."""
    local = [_local_tensor(value) for value in values]
    if len({tensor.dtype for tensor in local}) != 1:
        raise ValueError("Dtype disagreement across ranks")
    distributed = [isinstance(value, DTensor) for value in values]
    if any(distributed) and not all(distributed):
        raise ValueError("Mixed DTensor/plain representation across ranks")
    if not any(distributed):
        if any(tuple(tensor.shape) != shape for tensor in local):
            raise ValueError("Plain tensors must have the full schema shape; ambiguous plain sharding is unsupported")
        if any(not torch.equal(local[0], tensor) for tensor in local[1:]):
            raise ValueError("Plain tensor replicas disagree")
        return local[0].clone().contiguous()

    first = values[0]
    mesh = first.device_mesh.mesh
    placements = first.placements
    if mesh.ndim != 1 or len(placements) != 1 or sorted(mesh.tolist()) != list(range(len(values))):
        raise ValueError("Only a one-dimensional FSDP mesh covering every rank is supported")
    if first.device_mesh.mesh_dim_names and "tp" in first.device_mesh.mesh_dim_names:
        raise ValueError("Tensor-parallel checkpoints are unsupported")
    placement = placements[0]
    if type(placement) not in (Shard, Replicate):
        raise ValueError(f"Unsupported DTensor placement: {placement}")
    for value in values:
        if (
            tuple(value.shape) != shape
            or value.dtype != local[0].dtype
            or value.placements != placements
            or not torch.equal(value.device_mesh.mesh, mesh)
            or value.device_mesh.mesh_dim_names != first.device_mesh.mesh_dim_names
            or value.device_mesh.device_type != first.device_mesh.device_type
            or value.stride() != first.stride()
        ):
            raise ValueError("DTensor shape, mesh or placement disagreement")

    if isinstance(placement, Replicate):
        if any(tuple(tensor.shape) != shape or not torch.equal(local[0], tensor) for tensor in local):
            raise ValueError("DTensor replicas disagree with shape or values")
        return local[0].clone().contiguous()

    dim = placement.dim
    if not 0 <= dim < len(shape):
        raise ValueError("Invalid sharding dimension")
    ordered = [local[rank] for rank in mesh.tolist()]
    chunk = (shape[dim] + len(values) - 1) // len(values)
    extents = []
    for coordinate, tensor in enumerate(ordered):
        expected = list(shape)
        expected[dim] = max(0, min(chunk, shape[dim] - coordinate * chunk))
        if tuple(tensor.shape) != tuple(expected):
            raise ValueError("Invalid local shard extent (including uneven/empty shards)")
        extents.append(expected[dim])
    merged = torch.cat(ordered, dim=dim).contiguous()
    offset = 0
    for tensor, extent in zip(ordered, extents, strict=True):
        if not torch.equal(merged.narrow(dim, offset, extent), tensor):
            raise ValueError("Source shard round-trip failed")
        offset += extent
    return merged


def _portable_config(config: dict) -> dict:
    """Remove only location metadata; never rewrite behavior-affecting values."""
    result = {}
    for key, value in config.items():
        if key in {"_name_or_path", "name_or_path"}:
            continue
        result[key] = _portable_config(value) if isinstance(value, dict) else value
    return result


def _transformer_schema(base: Path, source: Path, architecture: str):
    cls = _transformer_class(architecture)
    base_config = read_json(base / "config.json")
    source_config = read_json(source / "huggingface/config.json")
    fields = set(inspect.signature(cls.__init__).parameters) - {"self"}
    for config in (base_config, source_config):
        if config.get("_class_name") != cls.__name__:
            raise ValueError(f"Expected {cls.__name__} config")
        if any(key not in fields and not key.startswith("_") for key in config):
            raise ValueError("Unrecognized transformer config fields")
    with torch.device("meta"):
        base_module = cls.from_config(base_config)
        source_module = cls.from_config(source_config)
    if any(base_module.config[key] != source_module.config[key] for key in fields):
        raise ValueError("Checkpoint/base transformer configuration mismatch")
    shapes = {key: tuple(value.shape) for key, value in base_module.state_dict().items()}
    keep_fp32 = tuple(getattr(base_module, "_keep_in_fp32_modules", None) or ())
    del base_module, source_module
    mapping = weight_files(base)
    if set(mapping) != set(shapes):
        raise ValueError("Base transformer weights do not match its config schema")
    for path in set(mapping.values()):
        with safe_open(path, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if tuple(archive.get_slice(key).get_shape()) != shapes[key]:
                    raise ValueError(f"Base tensor shape mismatch: {key}")
    return shapes, keep_fp32


def _check_pipeline(base: Path, architecture: str, component: str) -> None:
    import diffusers
    import transformers
    from diffusers import ModelMixin, SchedulerMixin
    from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

    if architecture not in _PIPELINES:
        raise ValueError(
            f"{architecture} supports output_format=transformer only; native pipeline conversion is separate"
        )
    config = read_json(base / "model_index.json")
    cls = getattr(diffusers, architecture)
    signature = inspect.signature(cls.__init__).parameters
    types = cls._get_signature_types()
    optional = cls._optional_components
    if config.get("_module") or any(not key.startswith("_") and key not in signature for key in config):
        raise ValueError("Unknown pipeline components or custom pipeline module")
    for key, parameter in signature.items():
        if key == "self":
            continue
        if key not in config:
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(f"Missing pipeline component: {key}")
            continue
        value = config[key]
        expected = types[key]
        # Pipeline options (e.g. Wan boundary_ratio) are not component descriptors.
        if all(kind in (bool, int, float, str, type(None)) for kind in expected):
            if not any(type(value) is kind for kind in expected):
                raise ValueError(f"Invalid pipeline option: {key}")
            continue
        if value == [None, None] and key in optional and key != component:
            continue
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, str) for v in value):
            raise ValueError(f"Invalid pipeline component: {key}")
        library, name = value
        modules = {"diffusers": diffusers, "transformers": transformers, "ltx2": diffusers.pipelines.ltx2}
        if library not in modules or not isinstance(actual := getattr(modules[library], name, None), type):
            raise ValueError(f"Unsupported component class: {key}")
        tokenizer_match = issubclass(actual, PreTrainedTokenizerBase) and (
            AutoTokenizer in expected
            or any(name.removesuffix("Fast") == kind.__name__.removesuffix("Fast") for kind in expected)
        )
        if not issubclass(actual, expected) and not tokenizer_match:
            raise ValueError(f"Component class conflicts with pipeline: {key}")
        root = base / key
        if issubclass(actual, ModelMixin | PreTrainedModel):
            model_config = read_json(root / "config.json")
            if model_config.get("quantization_config") or model_config.get("auto_map"):
                raise ValueError("Quantized/custom-code components are unsupported")
            weight_files(root, "model.safetensors" if issubclass(actual, PreTrainedModel) else None)
        elif issubclass(actual, SchedulerMixin):
            if read_json(root / "scheduler_config.json").get("_class_name") != name:
                raise ValueError("Scheduler config conflicts with pipeline index")
        elif issubclass(actual, PreTrainedTokenizerBase):
            read_json(root / "tokenizer_config.json")
            if not any(
                (root / asset).is_file() for asset in ("tokenizer.json", "spiece.model", "tokenizer.model")
            ) and not all((root / asset).is_file() for asset in ("vocab.json", "merges.txt")):
                raise ValueError("Missing tokenizer vocabulary assets")
        else:
            # Processor / image processor assets are copied unchanged, never reconstructed.
            assets = [root / name for name in ("processor_config.json", "preprocessor_config.json")]
            if not any(path.is_file() for path in assets):
                raise ValueError(f"Missing processor config: {key}")
            for path in assets:
                if path.is_file():
                    read_json(path)
    if config.get(component) in (None, [None, None]):
        raise ValueError(f"Selected trained component is absent: {component}")
    if architecture == "WanPipeline" and config.get("transformer_2") not in (None, [None, None]):
        ratio = config.get("boundary_ratio")
        if not isinstance(ratio, float | int) or isinstance(ratio, bool) or not 0 < ratio < 1:
            raise ValueError("Dual-transformer Wan requires a boundary_ratio in (0, 1)")


class FSDPModelMerger(BaseModelMerger):
    """Recover FSDP checkpoints and publish canonical Diffusers artifacts."""

    def iter_merged_weights(self, expected_shapes: Mapping[str, tuple[int, ...]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Mmap rank files and yield complete schema-checked weights without initializing distributed."""
        states = []
        try:
            for path in model_rank_files(Path(self.config.local_dir)):
                # Pickle is explicitly trusted by ModelMergerConfig. Do not silently fall back
                # to eager loading if mmap or this installed torch's decoder is unsupported.
                state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
                    raise ValueError("Expected a flat state dict in each model rank file")
                if any("lora_" in key or ".base_layer." in key for key in state):
                    raise ValueError("Adapter-bearing checkpoints require the future LoRA export mode")
                if set(state) != set(expected_shapes):
                    raise ValueError(
                        f"Incomplete transformer state: missing={sorted(set(expected_shapes) - set(state))[:8]}, "
                        f"unexpected={sorted(set(state) - set(expected_shapes))[:8]}"
                    )
                states.append(state)
            for key, shape in sorted(expected_shapes.items()):
                try:
                    yield key, reconstruct_tensor([state[key] for state in states], shape)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Cannot reconstruct {key}: {exc}") from exc
        finally:
            states.clear()

    def merge_and_save(self) -> MergeResult:
        source = Path(self.config.local_dir).resolve(strict=True)
        base = Path(self.config.base_model).resolve(strict=True)
        raw_target = Path(self.config.target_dir)
        if os.path.lexists(raw_target):
            raise FileExistsError(raw_target)
        target = raw_target.resolve()
        if not target.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        roots = (source, base, target)
        for i, left in enumerate(roots):
            for right in roots[i + 1 :]:
                if left.is_relative_to(right) or right.is_relative_to(left):
                    raise ValueError("Source, base and target directories must not overlap")
        if (source / "merge_source.json").exists():
            raise ValueError("Save-time merge_source schemas are not supported by this initial exporter")
        if (source / "lora_train_meta.json").exists():
            raise ValueError("LoRA checkpoint metadata requires the future adapter export mode")
        source_files = model_rank_files(source) + [source / "fsdp_config.json", source / "huggingface/config.json"]
        component = self.config.component
        index_path = base / "model_index.json"
        index = read_json(index_path) if index_path.is_file() else None
        if component is None:
            slots = [
                key for key in ("transformer", "transformer_2") if index and index.get(key) not in (None, [None, None])
            ]
            if len(slots) > 1:
                raise ValueError("Multiple transformers: explicitly select --component transformer or transformer_2")
            component = slots[0] if slots else "transformer"
        model_root = base / component if index is not None else base
        if model_root.is_symlink():
            raise ValueError("Component directory symlinks are unsupported")
        component_config = read_json(model_root / "config.json")
        architecture = _resolve_architecture(index, component_config, self.config.architecture)
        if component != "transformer" and architecture != "WanPipeline":
            raise ValueError("Only Wan supports selecting transformer_2")
        pipeline_output = self.config.output_format == "pipeline"
        if pipeline_output:
            if index is None:
                raise ValueError("Pipeline export requires a complete base pipeline; use output_format=transformer")
            base_files = tree_files(base)
            if any(path.suffix == ".py" for path in base_files):
                raise ValueError("Custom-code pipeline assets are unsupported")
        else:
            # Never copy native/custom-code assets into a canonical Diffusers component.
            base_files = sorted(set(weight_files(model_root).values()) | {model_root / "config.json"})
            if index is not None:
                base_files.append(index_path)
        source_inventory = inventory(source, source_files)
        base_inventory = inventory(base, base_files)
        if pipeline_output:
            _check_pipeline(base, architecture, component)
        shapes, keep_fp32 = _transformer_schema(model_root, source, architecture)
        dtype = None if self.config.dtype == "preserve" else getattr(torch, self.config.dtype)

        def weights():
            for key, value in self.iter_merged_weights(shapes):
                if dtype is not None and value.is_floating_point():
                    effective_dtype = torch.float32 if any(part in key.split(".") for part in keep_fp32) else dtype
                    value = value.to(effective_dtype)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Requested cast produced non-finite weights: {key}")
                yield key, value

        with publication_directory(target) as staging:
            tensor_directory = component if pipeline_output else "."
            specs = write_weights(staging / tensor_directory, weights(), self.config.max_shard_size)
            rewritten = []
            copy_files = base_files if pipeline_output else [model_root / "config.json"]
            copied = {}
            for path in copy_files:
                relative = path.relative_to(base) if pipeline_output else Path("config.json")
                # No pretrained transformer tensor can fill a missing checkpoint key.
                if pipeline_output and relative.parts[0] == component and relative.name != "config.json":
                    continue
                copied[relative.as_posix()] = base_inventory[path.relative_to(base).as_posix()]
                destination = staging / relative
                if destination.exists():
                    raise ValueError(f"Output path collision: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json" and ("config" in path.name or path.name == "model_index.json"):
                    original = read_json(path)
                    portable = _portable_config(original)
                    if portable != original:
                        write_json(destination, portable)
                        rewritten.append(relative.as_posix())
                        continue
                shutil.copyfile(path, destination)
            output_inventory = inventory(staging, tree_files(staging))
            for name, digest in copied.items():
                if name not in rewritten and output_inventory.get(name) != digest:
                    raise ValueError(f"Copied base asset verification failed: {name}")
            if (
                inventory(source, source_files) != source_inventory
                or inventory(base, tree_files(base) if pipeline_output else base_files) != base_inventory
                or read_json(model_root / "config.json") != component_config
                or (index is not None and read_json(index_path) != index)
            ):
                raise ValueError("Source/base inputs changed during export")
            if model_rank_files(source) != source_files[:-2]:
                raise ValueError("Model rank inventory changed during export")
            import diffusers

            manifest = {
                "schema_version": 1,
                "artifact_type": "diffusers_pipeline" if pipeline_output else "diffusers_transformer",
                "architecture": architecture,
                "backend": "fsdp",
                "trained_components": [component],
                "tensor_directory": tensor_directory,
                "dtype": self.config.dtype,
                "max_shard_size_bytes": self.config.max_shard_size,
                "source_fingerprint": fingerprint(source_inventory),
                "base_fingerprint": fingerprint(base_inventory),
                "source_files": source_inventory,
                "base_files": base_inventory,
                "files": output_inventory,
                "tensors": specs,
                "config_transform": {"id": "remove_location_metadata_v1", "files": rewritten},
                "producer": {"torch": torch.__version__, "diffusers": diffusers.__version__, "training": "unknown"},
                "verification": {
                    "structure": "passed",
                    "source_round_trip": "passed",
                    "artifact_round_trip": "passed",
                    "integrity": "passed",
                    "runtime": "not_run",
                },
            }
            write_json(staging / MANIFEST_NAME, manifest)
            from .output_validation import validate_artifact

            validate_artifact(staging)
        return MergeResult(target, target / MANIFEST_NAME)

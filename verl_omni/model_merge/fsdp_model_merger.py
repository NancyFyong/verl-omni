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
"""Strict per-key CPU reconstruction of one-dimensional FSDP checkpoints."""

from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

from .base_model_merger import BaseModelMerger
from .utils import read_json


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


class FSDPModelMerger(BaseModelMerger):
    """Shared reconstruction for non-HF mergers; subclasses own model packaging."""

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

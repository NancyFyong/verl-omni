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
"""Shared tensor and provenance helpers for offline distillation datasets."""

import hashlib
import io
import json
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

__all__ = ["canonical_manifest_sha256", "is_present", "load_float_tensor"]


def load_float_tensor(value: Any, field: str) -> torch.Tensor:
    """Load a non-empty finite detached fp32 tensor without arbitrary pickle execution."""
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, bytes | bytearray | memoryview):
        tensor = torch.load(io.BytesIO(bytes(value)), map_location="cpu", weights_only=True)
    elif isinstance(value, str):
        path = os.path.expanduser(value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Distillation tensor path for {field!r} does not exist: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Distillation field {field!r} must resolve to a tensor, got {type(tensor)}.")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"Distillation field {field!r} must be non-empty and finite.")
    return tensor.detach().float()


def is_present(value: Any) -> bool:
    """Treat missing parquet cells and NaN placeholders as absent values."""
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible manifest independently of mapping insertion order."""
    payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()

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
"""Dataset adapter for original-DMD Qwen-Image regression pairs."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

__all__ = ["QwenImageDMDPairDataset"]


def _load_float_tensor(value: Any, field: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, bytes | bytearray | memoryview):
        buffer = io.BytesIO(bytes(value))
        try:
            tensor = torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:
            buffer.seek(0)
            tensor = torch.load(buffer, map_location="cpu")
    elif isinstance(value, str):
        path = os.path.expanduser(value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DMD tensor path for {field!r} does not exist: {path}")
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            tensor = torch.load(path, map_location="cpu")
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"DMD field {field!r} must resolve to a tensor, got {type(tensor)}.")
    if tensor.numel() == 0:
        raise ValueError(f"DMD field {field!r} must not be empty.")
    return tensor.detach().float()


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


class QwenImageDMDPairDataset(RLHFDataset):
    """Load prompt, reference-noise, target, and provenance fields for original DMD."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        if "reference_noise" not in row:
            raise ValueError("Original-DMD rows require reference_noise.")
        has_latents = _is_present(row.get("teacher_target_latents"))
        has_pixels = _is_present(row.get("teacher_target_pixels"))
        if has_latents == has_pixels:
            raise ValueError("Original-DMD rows require exactly one teacher target: latents or pixels.")
        manifest = row.get("teacher_sampling_manifest")
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Original-DMD rows require a non-empty teacher_sampling_manifest mapping.")

        row["reference_noise"] = _load_float_tensor(row["reference_noise"], "reference_noise")
        target_key = "teacher_target_latents" if has_latents else "teacher_target_pixels"
        row[target_key] = _load_float_tensor(row[target_key], target_key)
        row["pair_id"] = str(row.get("pair_id", row.get("index", item)))
        return row

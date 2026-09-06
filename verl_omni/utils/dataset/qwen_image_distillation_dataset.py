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
"""Qwen-Image datasets for original-DMD pairs and DMD2 adversarial real samples."""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

__all__ = ["QwenImageDMDPairDataset", "QwenImageDMDRealDataset"]


def load_float_tensor(value: Any, field: str) -> torch.Tensor:
    """Load a non-empty detached fp32 tensor without arbitrary pickle execution."""
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, bytes | bytearray | memoryview):
        buffer = io.BytesIO(bytes(value))
        tensor = torch.load(buffer, map_location="cpu", weights_only=True)
    elif isinstance(value, str):
        path = os.path.expanduser(value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DMD tensor path for {field!r} does not exist: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"DMD field {field!r} must resolve to a tensor, got {type(tensor)}.")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError(f"DMD field {field!r} must be non-empty and finite.")
    return tensor.detach().float()


def is_present(value: Any) -> bool:
    """Treat missing parquet cells and NaN placeholders as absent targets."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


class QwenImageDMDRealDataset(RLHFDataset):
    """Read real RGB [C, H, W] in [0, 1] or provenance-tracked normalized latents."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        has_latents = is_present(row.get("real_latents"))
        has_pixels = is_present(row.get("real_pixels"))
        if has_latents == has_pixels:
            raise ValueError("DMD2 adversarial rows require exactly one of real_latents or real_pixels.")
        field = "real_latents" if has_latents else "real_pixels"
        value = load_float_tensor(row[field], field)
        if has_latents:
            manifest = row.get("real_latent_manifest")
            if (
                not isinstance(manifest, dict)
                or manifest.get("normalization") != "qwen_image"
                or not isinstance(manifest.get("vae_config_sha256"), str)
                or not manifest["vae_config_sha256"]
            ):
                raise ValueError("real_latent_manifest requires normalization=qwen_image and vae_config_sha256.")
        elif value.ndim != 3 or value.shape[0] != 3 or torch.any((value < 0) | (value > 1)):
            raise ValueError("real_pixels must have shape [3, H, W] and values in [0, 1].")
        row[field] = value
        return row


class QwenImageDMDPairDataset(RLHFDataset):
    """Load prompt, reference-noise, target, and provenance fields for original DMD."""

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        if "reference_noise" not in row:
            raise ValueError("Original-DMD rows require reference_noise.")
        has_latents = is_present(row.get("teacher_target_latents"))
        has_pixels = is_present(row.get("teacher_target_pixels"))
        if has_latents == has_pixels:
            raise ValueError("Original-DMD rows require exactly one teacher target: latents or pixels.")
        manifest = row.get("teacher_sampling_manifest")
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Original-DMD rows require a non-empty teacher_sampling_manifest mapping.")

        row["reference_noise"] = load_float_tensor(row["reference_noise"], "reference_noise")
        target_key = "teacher_target_latents" if has_latents else "teacher_target_pixels"
        row[target_key] = load_float_tensor(row[target_key], target_key)
        row["pair_id"] = str(row.get("pair_id", row.get("index", item)))
        return row

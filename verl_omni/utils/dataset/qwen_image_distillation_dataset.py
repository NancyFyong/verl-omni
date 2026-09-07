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

from typing import Any

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_omni.utils.dataset.distillation import is_present, load_float_tensor

__all__ = ["QwenImageDMDPairDataset", "QwenImageDMDRealDataset", "is_present", "load_float_tensor"]


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
        row.pop("real_pixels" if has_latents else "real_latents", None)
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
        row.pop("teacher_target_pixels" if has_latents else "teacher_target_latents", None)
        row["pair_id"] = str(row.get("pair_id", row.get("index", item)))
        return row

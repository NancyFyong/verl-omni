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
"""Dataset for provenance-checked Wan ODE trajectories."""

from collections.abc import Mapping
from typing import Any

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_omni.utils.dataset.distillation import canonical_manifest_sha256, load_float_tensor

__all__ = ["WanODETrajectoryDataset"]


class WanODETrajectoryDataset(RLHFDataset):
    """Load `[steps, frames, channels, height, width]` ODE trajectories."""

    required_manifest_fields = frozenset(
        {
            "teacher_model",
            "teacher_revision",
            "scheduler_class",
            "scheduler_config",
            "guidance_scale",
            "negative_prompt",
            "timesteps",
            "vae",
            "latent_layout",
            "dtype",
            "height",
            "width",
            "num_frames",
            "prompt_tokenizer",
            "seed_policy",
        }
    )

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = super().__getitem__(item)
        for field in ("ode_latents", "ode_timesteps", "final_clean_latent"):
            if field not in row:
                raise ValueError(f"Wan ODE rows require {field}.")
        trajectory = load_float_tensor(row["ode_latents"], "ode_latents")
        timesteps = load_float_tensor(row["ode_timesteps"], "ode_timesteps")
        clean = load_float_tensor(row["final_clean_latent"], "final_clean_latent")
        for key in ("prompt_embeds", "negative_prompt_embeds"):
            if row.get(key) is not None:
                embeds = load_float_tensor(row[key], key)
                if embeds.ndim != 2 or embeds.shape[0] == 0:
                    raise ValueError(f"{key} must have shape [sequence, hidden_size].")
                row[key] = embeds
        manifest = row.get("trajectory_manifest")
        if trajectory.ndim != 5:
            raise ValueError("ode_latents must have shape [steps, frames, channels, height, width].")
        if timesteps.ndim != 1 or timesteps.shape[0] != trajectory.shape[0]:
            raise ValueError("ode_timesteps must be one-dimensional and match the trajectory step count.")
        if timesteps[-1].item() != 0 or torch.any(timesteps[:-1] <= timesteps[1:]):
            raise ValueError("ode_timesteps must be strictly descending and end at zero.")
        if clean.shape != trajectory.shape[1:]:
            raise ValueError("final_clean_latent must match one trajectory state's [frames, channels, height, width].")
        if not isinstance(manifest, Mapping):
            raise ValueError("trajectory_manifest must be a mapping.")
        missing = self.required_manifest_fields - set(manifest)
        if missing:
            raise ValueError(f"trajectory_manifest is missing required fields: {sorted(missing)}.")
        if manifest.get("latent_layout") != "SFCHW":
            raise ValueError("trajectory_manifest.latent_layout must be 'SFCHW'.")
        if int(manifest.get("num_frames")) != trajectory.shape[1]:
            raise ValueError("trajectory_manifest.num_frames does not match ode_latents.")
        if int(manifest.get("height")) != trajectory.shape[3] or int(manifest.get("width")) != trajectory.shape[4]:
            raise ValueError("trajectory_manifest spatial shape does not match ode_latents.")
        if not torch.equal(trajectory[-1], clean):
            raise ValueError("final_clean_latent must exactly match the final ODE trajectory state.")
        if list(map(float, manifest.get("timesteps", []))) != timesteps.tolist():
            raise ValueError("trajectory_manifest.timesteps does not match ode_timesteps.")
        row["ode_latents"] = trajectory
        row["ode_timesteps"] = timesteps
        row["final_clean_latent"] = clean
        row["trajectory_manifest"] = dict(manifest)
        row["trajectory_manifest_sha256"] = canonical_manifest_sha256(manifest)
        return row

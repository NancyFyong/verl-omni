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
"""CPU tests for the Wan ODE trajectory dataset contract."""

import pytest
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from verl_omni.utils.dataset.distillation import canonical_manifest_sha256
from verl_omni.utils.dataset.wan_ode_dataset import WanODETrajectoryDataset


def manifest(frames=4):
    """Return a complete synthetic trajectory manifest."""
    return {
        "teacher_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "teacher_revision": "revision",
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "scheduler_config": {"shift": 8.0},
        "guidance_scale": 6.0,
        "negative_prompt": "",
        "timesteps": [1000.0, 500.0, 0.0],
        "vae": "Wan2.1_VAE",
        "latent_layout": "SFCHW",
        "dtype": "float32",
        "height": 2,
        "width": 2,
        "num_frames": frames,
        "prompt_tokenizer": "umt5-xxl",
        "seed_policy": "per-sample",
    }


def valid_row():
    """Return one valid trajectory row."""
    trajectory = torch.randn(3, 4, 5, 2, 2)
    return {
        "ode_latents": trajectory,
        "ode_timesteps": torch.tensor([1000.0, 500.0, 0.0]),
        "final_clean_latent": trajectory[-1].clone(),
        "prompt_embeds": torch.randn(3, 8).tolist(),
        "trajectory_manifest": manifest(),
    }


class TestWanODETrajectoryDataset:
    @staticmethod
    def make_dataset():
        return object.__new__(WanODETrajectoryDataset)

    def test_loads_trajectory_and_adds_canonical_fingerprint(self, monkeypatch):
        row = valid_row()
        monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: dict(row))
        output = self.make_dataset()[0]
        assert output["ode_latents"].shape == (3, 4, 5, 2, 2)
        assert output["prompt_embeds"].shape == (3, 8)
        assert output["trajectory_manifest_sha256"] == canonical_manifest_sha256(row["trajectory_manifest"])

    def test_default_collate_stacks_precomputed_conditioning(self, monkeypatch):
        row = valid_row()
        monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: dict(row))
        dataset = self.make_dataset()
        batch = collate_fn([dataset[0], dataset[1]])
        assert batch["prompt_embeds"].shape == (2, 3, 8)
        assert batch["ode_latents"].shape == (2, 3, 4, 5, 2, 2)

    @pytest.mark.parametrize(
        "mutate,error",
        [
            (lambda row: row.pop("ode_latents"), "ode_latents"),
            (lambda row: row.__setitem__("ode_timesteps", [1000.0]), "step count"),
            (lambda row: row.__setitem__("ode_timesteps", [1000.0, 500.0, 250.0]), "end at zero"),
            (lambda row: row.__setitem__("prompt_embeds", [1.0, 2.0]), "prompt_embeds"),
            (lambda row: row.__setitem__("final_clean_latent", torch.zeros(4, 3, 2, 2)), "must match"),
            (lambda row: row.__setitem__("trajectory_manifest", {}), "missing required"),
            (
                lambda row: row["trajectory_manifest"].__setitem__("latent_layout", "SCFHW"),
                "latent_layout",
            ),
        ],
    )
    def test_rejects_invalid_rows(self, monkeypatch, mutate, error):
        row = valid_row()
        mutate(row)
        monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: dict(row))
        with pytest.raises((TypeError, ValueError), match=error):
            self.make_dataset()[0]

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
"""Create deterministic synthetic Wan ODE trajectories for GPU smoke tests."""

import argparse
import json
import os

import pandas as pd
import torch
from safetensors import safe_open

from verl_omni.utils.dataset.distillation import canonical_manifest_sha256


def checkpoint_dtype(model_path: str) -> str:
    """Read the stored transformer dtype without loading model tensors."""
    index_path = os.path.join(model_path, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        weight_file = next(iter(index["weight_map"].values()))
    else:
        weight_file = "diffusion_pytorch_model.safetensors"
    with safe_open(os.path.join(model_path, "transformer", weight_file), framework="pt", device="cpu") as file:
        return str(file.get_tensor(next(iter(file.keys()))).dtype).removeprefix("torch.")


def make_manifest(model_path: str, latent_frames: int, latent_height: int, latent_width: int) -> dict:
    """Build the complete provenance manifest consumed by WanODETrajectoryDataset."""
    with open(os.path.join(model_path, "scheduler", "scheduler_config.json"), encoding="utf-8") as file:
        scheduler_config = json.load(file)
    return {
        "teacher_model": os.path.basename(model_path.rstrip("/")),
        "teacher_revision": "local-smoke-fixture",
        "scheduler_class": scheduler_config.get("_class_name"),
        "scheduler_config": scheduler_config,
        "guidance_scale": 6.0,
        "negative_prompt": "",
        "timesteps": [1000.0, 500.0, 0.0],
        "vae": "AutoencoderKLWan",
        "latent_layout": "SFCHW",
        "dtype": checkpoint_dtype(model_path),
        "height": latent_height,
        "width": latent_width,
        "num_frames": latent_frames,
        "prompt_tokenizer": "umt5-xxl",
        "seed_policy": "deterministic-synthetic-smoke",
    }


def build_rows(size: int, model_path: str, latent_frames: int, latent_height: int, latent_width: int):
    """Build synthetic, shape-valid ODE rows and their shared manifest digest."""
    with open(os.path.join(model_path, "transformer", "config.json"), encoding="utf-8") as file:
        transformer_config = json.load(file)
    channels = int(transformer_config["in_channels"])
    text_dim = int(transformer_config["text_dim"])
    manifest = make_manifest(model_path, latent_frames, latent_height, latent_width)
    digest = canonical_manifest_sha256(manifest)
    rows = []
    for index in range(size):
        generator = torch.Generator().manual_seed(index)
        noise = torch.randn(latent_frames, channels, latent_height, latent_width, generator=generator)
        clean = torch.randn(latent_frames, channels, latent_height, latent_width, generator=generator)
        middle = 0.5 * noise + 0.5 * clean
        rows.append(
            {
                "data_source": "wan_ode_smoke",
                "prompt": [{"role": "user", "content": f"Synthetic video {index}"}],
                "prompt_embeds": torch.zeros(4, text_dim).tolist(),
                "ode_latents": torch.stack((noise, middle, clean)).tolist(),
                "ode_timesteps": manifest["timesteps"],
                "final_clean_latent": clean.tolist(),
                "trajectory_manifest": manifest,
                "extra_info": {"index": index},
            }
        )
    return rows, digest


def main() -> None:
    """Write train/test parquet and the canonical manifest digest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=8)
    parser.add_argument("--latent_frames", type=int, default=3)
    parser.add_argument("--latent_height", type=int, default=8)
    parser.add_argument("--latent_width", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_rows, digest = build_rows(
        args.train_size, args.model_path, args.latent_frames, args.latent_height, args.latent_width
    )
    validation_rows, validation_digest = build_rows(
        args.val_size, args.model_path, args.latent_frames, args.latent_height, args.latent_width
    )
    if validation_digest != digest:
        raise RuntimeError("Synthetic train and validation manifests diverged.")
    pd.DataFrame(train_rows).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    pd.DataFrame(validation_rows).to_parquet(os.path.join(args.output_dir, "test.parquet"))
    with open(os.path.join(args.output_dir, "manifest.sha256"), "w", encoding="utf-8") as file:
        file.write(f"{digest}\n")


if __name__ == "__main__":
    main()

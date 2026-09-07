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

"""Generate resumable Wan 2.1 teacher trajectories for causal ODE training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
SNAPSHOT_STATE_INDICES = (0, 36, 44, 50)


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_revision(model_path: Path) -> str:
    metadata = model_path / ".cache/huggingface/download/model_index.json.metadata"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing Hugging Face revision metadata: {metadata}")
    revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    if len(revision) != 40:
        raise ValueError(f"Invalid checkpoint revision in {metadata}: {revision!r}")
    return revision


class CausVidFlowMatchEulerScheduler(FlowMatchEulerDiscreteScheduler):
    """Reference 50-step shift-8 Euler grid with a removed terminal zero."""

    reference_sigma_min = 0.0

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        timesteps: list[float] | None = None,
    ) -> None:
        if sigmas is None and timesteps is None:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required for the CausVid schedule.")
            sigmas = np.linspace(
                1.0,
                self.reference_sigma_min,
                num_inference_steps + 1,
                dtype=np.float32,
            )[:-1].tolist()
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )


class TrajectoryCollector:
    """Collect the two intermediate states selected by the CausVid recipe."""

    capture_after_steps = frozenset({35, 43})

    def __init__(self, initial_latents: torch.Tensor, index: int) -> None:
        self.states = [self.to_storage(initial_latents)]
        self.index = index
        self.started = time.perf_counter()

    @staticmethod
    def to_storage(latents: torch.Tensor) -> torch.Tensor:
        return latents.detach().to(device="cpu", dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous()

    def __call__(self, pipeline, step: int, timestep: torch.Tensor, callback_kwargs: dict[str, torch.Tensor]):
        del pipeline, timestep
        if step == 0 or (step + 1) % 5 == 0:
            print(
                json.dumps(
                    {
                        "event": "denoise",
                        "sample": self.index,
                        "rank": int(os.environ.get("RANK", 0)),
                        "step": step + 1,
                        "total": 50,
                        "elapsed_s": round(time.perf_counter() - self.started, 2),
                    }
                ),
                flush=True,
            )
        if step in self.capture_after_steps:
            self.states.append(self.to_storage(callback_kwargs["latents"]))
        return callback_kwargs

    def finish(self, final_latents: torch.Tensor) -> torch.Tensor:
        self.states.append(self.to_storage(final_latents))
        if len(self.states) != 4:
            raise RuntimeError(f"Expected four ODE states, collected {len(self.states)}.")
        return torch.stack([state.squeeze(0) for state in self.states], dim=0)


def build_scheduler(device: torch.device) -> CausVidFlowMatchEulerScheduler:
    scheduler = CausVidFlowMatchEulerScheduler(
        num_train_timesteps=1000,
        shift=8.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(50, device=device)
    return scheduler


def ode_timesteps() -> list[float]:
    scheduler = build_scheduler(torch.device("cpu"))
    values = [scheduler.timesteps[0], scheduler.timesteps[36], scheduler.timesteps[44], torch.tensor(0.0)]
    return [float(value.to(torch.float32).item()) for value in values]


def build_manifest(model_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    revision = checkpoint_revision(model_path)
    vae_config_path = model_path / "vae/config.json"
    tokenizer_config_path = model_path / "tokenizer/tokenizer_config.json"
    return {
        "teacher_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "teacher_revision": revision,
        "scheduler_class": "CausVidFlowMatchEulerScheduler",
        "scheduler_config": {
            "num_train_timesteps": 1000,
            "num_inference_steps": 50,
            "shift": 8.0,
            "sigma_min": CausVidFlowMatchEulerScheduler.reference_sigma_min,
            "extra_one_step": True,
            "snapshot_state_indices": list(SNAPSHOT_STATE_INDICES),
        },
        "guidance_scale": args.guidance_scale,
        "negative_prompt": NEGATIVE_PROMPT,
        "timesteps": ode_timesteps(),
        "vae": {
            "class": file_json(vae_config_path).get("_class_name", "AutoencoderKLWan"),
            "config_sha256": file_sha256(vae_config_path),
            "revision": revision,
        },
        "latent_layout": "SFCHW",
        "dtype": "bfloat16",
        "teacher_compute_dtype": args.teacher_dtype,
        "height": args.height // 8,
        "width": args.width // 8,
        "num_frames": (args.num_frames - 1) // 4 + 1,
        "prompt_tokenizer": {
            "class": file_json(tokenizer_config_path).get("tokenizer_class", "AutoTokenizer"),
            "config_sha256": file_sha256(tokenizer_config_path),
            "max_sequence_length": args.max_sequence_length,
            "revision": revision,
        },
        "seed_policy": {"name": "base_seed_plus_prompt_index", "base_seed": args.base_seed},
    }


def read_prompts(path: Path, max_samples: int | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples is not None:
        prompts = prompts[:max_samples]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}.")
    return prompts


def sample_paths(output_dir: Path, index: int) -> dict[str, Path]:
    shard = output_dir / "samples" / f"{index // 1000:04d}"
    return {
        "trajectory": shard / f"{index:06d}.trajectory.pt",
        "clean": shard / f"{index:06d}.clean.pt",
        "prompt_embeds": shard / f"{index:06d}.prompt_embeds.pt",
        "metadata": shard / f"{index:06d}.json",
    }


def atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sample_complete(paths: dict[str, Path], expected_prompt_sha256: str, manifest_sha256: str) -> bool:
    if not all(paths[name].is_file() for name in ("trajectory", "clean", "prompt_embeds", "metadata")):
        return False
    try:
        metadata = file_json(paths["metadata"])
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("prompt_sha256") == expected_prompt_sha256 and metadata.get("manifest_sha256") == manifest_sha256
    )


def load_pipeline(model_path: Path, device: torch.device, dtype: torch.dtype) -> WanPipeline:
    scheduler = build_scheduler(device)
    pipeline = WanPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        vae=None,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.transformer.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    return pipeline


def generate_sample(
    pipeline: WanPipeline,
    prompt: str,
    index: int,
    args: argparse.Namespace,
    paths: dict[str, Path],
    negative_prompt_embeds: torch.Tensor,
    manifest_sha256: str,
) -> None:
    device = pipeline._execution_device
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
        max_sequence_length=args.max_sequence_length,
        device=device,
        dtype=pipeline.transformer.dtype,
    )
    generator = torch.Generator(device=device).manual_seed(args.base_seed + index)
    initial_latents = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=pipeline.transformer.config.in_channels,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )
    collector = TrajectoryCollector(initial_latents, index)
    result = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=50,
        guidance_scale=args.guidance_scale,
        generator=generator,
        latents=initial_latents,
        output_type="latent",
        callback_on_step_end=collector,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=args.max_sequence_length,
    )
    trajectory = collector.finish(result.frames)
    clean = trajectory[-1].clone()
    stored_prompt_embeds = prompt_embeds.squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(trajectory.float()).all() or not torch.isfinite(stored_prompt_embeds.float()).all():
        raise FloatingPointError(f"Non-finite trajectory or conditioning for sample {index}.")
    atomic_torch_save(trajectory, paths["trajectory"])
    atomic_torch_save(clean, paths["clean"])
    atomic_torch_save(stored_prompt_embeds, paths["prompt_embeds"])
    atomic_json_save(
        {
            "index": index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "manifest_sha256": manifest_sha256,
            "seed": args.base_seed + index,
            "trajectory_shape": list(trajectory.shape),
            "prompt_embeds_shape": list(stored_prompt_embeds.shape),
        },
        paths["metadata"],
    )


def run_generation(args: argparse.Namespace) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = read_prompts(args.prompts, args.max_samples)
    manifest = build_manifest(args.model_path, args)
    manifest_sha256 = canonical_sha256(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json_save(manifest, args.output_dir / "trajectory_manifest.json")
        (args.output_dir / "manifest.sha256").write_text(f"{manifest_sha256}\n", encoding="utf-8")

    dtype = torch.float32 if args.teacher_dtype == "float32" else torch.bfloat16
    pipeline = load_pipeline(args.model_path, device, dtype)
    negative_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=NEGATIVE_PROMPT,
        do_classifier_free_guidance=False,
        max_sequence_length=args.max_sequence_length,
        device=device,
        dtype=pipeline.transformer.dtype,
    )
    atomic_torch_save(
        negative_prompt_embeds[0].detach().to(device="cpu", dtype=torch.bfloat16),
        args.output_dir / f"negative_prompt_embeds_rank{rank}.pt",
    )
    assigned = range(rank, len(prompts), world_size)
    for completed, index in enumerate(assigned, start=1):
        prompt = prompts[index]
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        paths = sample_paths(args.output_dir, index)
        if sample_complete(paths, prompt_sha256, manifest_sha256):
            continue
        started = time.perf_counter()
        with torch.inference_mode():
            generate_sample(pipeline, prompt, index, args, paths, negative_prompt_embeds, manifest_sha256)
        duration = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "rank": rank,
                    "index": index,
                    "completed_on_rank": completed,
                    "assigned_on_rank": len(range(rank, len(prompts), world_size)),
                    "seconds": duration,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def finalize_dataset(args: argparse.Namespace) -> None:
    prompts = read_prompts(args.prompts, args.max_samples)
    manifest = build_manifest(args.model_path, args)
    manifest_sha256 = canonical_sha256(manifest)
    rows = []
    for index, prompt in enumerate(prompts):
        paths = sample_paths(args.output_dir, index)
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        if not sample_complete(paths, prompt_sha256, manifest_sha256):
            raise RuntimeError(f"Sample {index} is incomplete or stale: {paths['metadata']}")
        rows.append(
            {
                "data_source": "wan21_causvid_teacher_ode",
                "prompt": [{"role": "user", "content": prompt}],
                "prompt_embeds": str(paths["prompt_embeds"]),
                "negative_prompt_embeds": str(args.output_dir / "negative_prompt_embeds_rank0.pt"),
                "ode_latents": str(paths["trajectory"]),
                "ode_timesteps": manifest["timesteps"],
                "final_clean_latent": str(paths["clean"]),
                "trajectory_manifest": manifest,
                "extra_info": {
                    "index": index,
                    "seed": args.base_seed + index,
                    "prompt_sha256": prompt_sha256,
                },
            }
        )
    validation_size = min(args.validation_size, len(rows) - 1)
    if len(rows) <= validation_size:
        raise ValueError("The generated dataset is too small for the requested validation split.")
    train_rows = rows[:-validation_size]
    validation_rows = rows[-validation_size:]
    pd.DataFrame(train_rows).to_parquet(args.output_dir / "train.parquet")
    pd.DataFrame(validation_rows).to_parquet(args.output_dir / "test.parquet")
    summary = {
        "manifest_sha256": manifest_sha256,
        "prompt_count": len(rows),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "prompt_source_sha256": file_sha256(args.prompts),
    }
    atomic_json_save(summary, args.output_dir / "dataset_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--base-seed", type=int, default=20260906)
    parser.add_argument("--teacher-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--validation-size", type=int, default=83)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.height % 16 or args.width % 16:
        parser.error("height and width must be divisible by 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("num_frames must satisfy (num_frames - 1) % 4 == 0")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max_samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.finalize:
        finalize_dataset(args)
    else:
        run_generation(args)


if __name__ == "__main__":
    main()

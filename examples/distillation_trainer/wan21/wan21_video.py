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
"""Export causal Wan adapters, generate/decode MP4 files, and make a video gallery."""

import argparse
import hashlib
import html
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from peft import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from verl.utils.device import get_device_name

from verl_omni.pipelines.wan21_distillation.causal_attention import configure_causal_wan
from verl_omni.pipelines.wan21_distillation.inference import (
    decode_wan_latents,
    sample_wan_causal,
    validate_wan_sampling_timesteps,
)
from verl_omni.utils.fsdp_utils import export_fsdp_lora_adapter


def read_json(path: Path) -> dict:
    """Read a JSON manifest."""
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    """Hash artifact bytes without materializing them twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_manifest(trajectory_manifest: dict, frames_per_block: int) -> dict:
    """Use actual trajectory timesteps, never shift already shifted metadata again."""
    timesteps = trajectory_manifest["timesteps"]
    if timesteps[-1] != 0:
        raise ValueError("Trajectory schedule must end at zero.")
    total = trajectory_manifest["scheduler_config"]["num_train_timesteps"]
    resolved = validate_wan_sampling_timesteps(timesteps[:-1], total)
    if frames_per_block <= 0 or trajectory_manifest["num_frames"] % frames_per_block:
        raise ValueError("Trajectory frames must be divisible by frames_per_block.")
    return {
        "version": 1,
        "architecture": "WanPipeline",
        "base_model": trajectory_manifest["teacher_model"],
        "base_revision": trajectory_manifest["teacher_revision"],
        "transition": "consistency_renoise",
        "denoising_timesteps": resolved,
        "num_train_timesteps": total,
        "frames_per_block": frames_per_block,
        "latent_frames": trajectory_manifest["num_frames"],
        "latent_height": trajectory_manifest["height"],
        "latent_width": trajectory_manifest["width"],
        "max_sequence_length": trajectory_manifest["prompt_tokenizer"]["max_sequence_length"],
    }


def export_adapter(args) -> None:
    """Export only a selected semantic role from a trusted complete FSDP checkpoint."""
    if not args.trust_checkpoint:
        raise ValueError("FSDP pickle deserialization requires explicit --trust-checkpoint for your own checkpoint.")
    checkpoint_manifest = read_json(args.checkpoint / "manifest.json")
    groups = args.checkpoint / "workers/role_groups"
    selected = []
    for metadata_path in groups.glob("*/role_group.json"):
        metadata = read_json(metadata_path)
        selected.extend(
            (metadata_path.parent, binding) for binding in metadata["bindings"] if binding["role"] == args.role
        )
    if len(selected) != 1 or not selected[0][1]["adapter"]:
        raise ValueError("Checkpoint must bind the selected role to exactly one LoRA adapter.")
    manifest = sampling_manifest(read_json(args.manifest), args.frames_per_block)
    if checkpoint_manifest["plan_name"] == "causvid":
        rollout = checkpoint_manifest["rollout"]
        manifest["denoising_timesteps"] = validate_wan_sampling_timesteps(
            rollout["denoising_timesteps"][:-1], rollout["num_train_timesteps"]
        )
        manifest["frames_per_block"] = rollout["frames_per_block"]
        manifest["reference_grid_shift"] = rollout["scheduler_shift"]
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".wan-export-", dir=args.output.parent))
    try:
        export_fsdp_lora_adapter(
            selected[0][0], temporary, manifest["base_model"], adapter_name=selected[0][1]["adapter"]
        )
        weights = load_file(temporary / "adapter_model.safetensors")
        weights = {"transformer." + name.removeprefix("base_model.model."): value for name, value in weights.items()}
        save_file(weights, temporary / "adapter_model.safetensors")
        manifest.update(
            role=args.role,
            global_step=checkpoint_manifest["global_step"],
            recipe=checkpoint_manifest["plan_name"],
            weights_sha256=sha256_file(temporary / "adapter_model.safetensors"),
        )
        (temporary / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


def load_adapter(transformer, adapter: Path) -> dict:
    """Load a single verified Diffusers-native adapter and reject missing/extra tensors."""
    manifest = read_json(adapter / "inference_manifest.json")
    if sha256_file(adapter / "adapter_model.safetensors") != manifest["weights_sha256"]:
        raise ValueError("Inference adapter hash does not match its manifest.")
    state = load_file(adapter / "adapter_model.safetensors")
    state = {
        name.removeprefix("base_model.model.").removeprefix("transformer."): value for name, value in state.items()
    }
    transformer.add_adapter(LoraConfig.from_pretrained(adapter), adapter_name="inference")
    expected = get_peft_model_state_dict(transformer, adapter_name="inference")
    if set(state) != set(expected):
        raise ValueError(
            f"Adapter keys differ: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}"
        )
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid adapter tensor {name}.")
    incompatible = set_peft_model_state_dict(transformer, state, adapter_name="inference")
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {incompatible.unexpected_keys}")
    transformer.set_adapter("inference")
    return manifest


def log_progress(block: int, blocks: int, step: int, steps: int) -> None:
    """Flush block/denoising progress even when redirected to a log file."""
    print(f"block {block}/{blocks} | denoise {step}/{steps}", flush=True)


def save_video(vae, latents: torch.Tensor, args, metadata: dict) -> None:
    """Write one playable MP4, thumbnail and human-readable metadata sidecar."""
    from PIL import Image

    print("VAE decode started", flush=True)
    pixels = decode_wan_latents(vae, latents)[0].cpu()
    frames = pixels.permute(0, 2, 3, 1).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=args.fps)
    Image.fromarray((frames[len(frames) // 2] * 255).round().astype("uint8")).save(args.output.with_suffix(".jpg"))
    metadata.update(prompt=args.prompt, fps=args.fps, frames=len(frames), video=args.output.name)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"MP4 ready: {args.output}", flush=True)


def decode(args) -> None:
    """Decode a final FCHW teacher target, not model weights."""
    latent = torch.load(args.latent, map_location="cpu", weights_only=True)
    if latent.ndim != 4:
        raise ValueError("--latent must contain final clean [F,C,H,W] latents.")
    vae = AutoencoderKLWan.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32).to(args.device)
    vae.enable_tiling()
    save_video(vae, latent.permute(1, 0, 2, 3).unsqueeze(0), args, {"variant": "teacher_target", "causal": False})


@torch.inference_mode()
def generate(args) -> None:
    """Generate a conditional-only causal student video using a fresh batch-local cache."""
    manifest = (
        read_json(args.adapter / "inference_manifest.json")
        if args.adapter
        else sampling_manifest(read_json(args.manifest), args.frames_per_block)
    )
    if manifest["version"] != 1 or manifest["transition"] != "consistency_renoise":
        raise ValueError("Unsupported causal inference manifest.")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.base_model, subfolder="vae", torch_dtype=torch.float32)
    vae.enable_tiling()
    pipeline = WanPipeline.from_pretrained(args.base_model, vae=vae, torch_dtype=dtype).to(args.device)
    if args.adapter:
        load_adapter(pipeline.transformer, args.adapter)
    configure_causal_wan(pipeline.transformer)
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt=args.prompt,
        do_classifier_free_guidance=False,
        max_sequence_length=manifest["max_sequence_length"],
        device=args.device,
        dtype=dtype,
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    shape = (
        1,
        pipeline.transformer.config.in_channels,
        manifest["latent_frames"],
        manifest["latent_height"],
        manifest["latent_width"],
    )
    noise = torch.randn(shape, device=args.device, dtype=torch.float32, generator=generator)
    if args.trajectory:
        trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=True)
        noise = trajectory[0].permute(1, 0, 2, 3).unsqueeze(0).to(device=args.device, dtype=torch.float32)
        if tuple(noise.shape) != shape:
            raise ValueError("Initial trajectory noise does not match inference geometry.")
    started = time.perf_counter()
    latents = sample_wan_causal(
        pipeline.transformer,
        noise,
        prompt_embeds,
        timesteps=manifest["denoising_timesteps"],
        frames_per_block=manifest["frames_per_block"],
        num_train_timesteps=manifest["num_train_timesteps"],
        reference_grid_shift=manifest.get("reference_grid_shift"),
        generator=generator,
        callback=log_progress,
    )
    elapsed = time.perf_counter() - started
    save_video(
        pipeline.vae,
        latents,
        args,
        {
            "variant": "causal_untrained" if args.adapter is None else manifest["role"],
            "seed": args.seed,
            "causal": True,
            "sampling": manifest,
            "sampling_seconds": elapsed,
            "quality_validated": False,
        },
    )


def gallery(args) -> None:
    """Make a static prompt/video page without exposing tensor/checkpoint files."""
    cards = []
    for video in sorted(args.directory.glob("*.mp4")):
        metadata = read_json(video.with_suffix(".json"))
        cards.append(
            f"<article><h2>{html.escape(video.stem)}</h2><p>{html.escape(metadata['prompt'])}</p>"
            f"<p>{html.escape(metadata['variant'])}</p>"
            f'<video controls preload="metadata" poster="{html.escape(video.with_suffix(".jpg").name)}" '
            f'src="{html.escape(video.name)}"></video></article>'
        )
    document = '<!doctype html><meta charset="utf-8"><title>Wan causal video evaluation</title>'
    document += (
        "<style>body{max-width:1400px;margin:2em auto;font-family:sans-serif}"
        "main{display:grid;grid-template-columns:repeat(2,1fr);gap:2em}video{width:100%}"
        "article{border:1px solid #bbb;padding:1em}</style>"
    )
    document += (
        "<h1>Wan 视频评估</h1><p>teacher_target 是双向教师目标，不代表学生效果；"
        "causal_untrained 是未经训练的因果基线。执行成功不等于质量改善。</p><main>"
    )
    document += "\n".join(cards) + "</main>"
    (args.directory / "index.html").write_text(document)
    print(f"Gallery: {args.directory / 'index.html'} ({len(cards)} videos)", flush=True)


def main() -> None:
    """Dispatch the user-facing export, generation, decoding and gallery commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export")
    exporter.add_argument("--checkpoint", type=Path, required=True)
    exporter.add_argument("--role", choices=("student", "student_ema"), default="student")
    exporter.add_argument("--manifest", type=Path, required=True)
    exporter.add_argument("--frames-per-block", type=int, default=3)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--trust-checkpoint", action="store_true")
    for command in ("generate", "decode"):
        child = subparsers.add_parser(command)
        child.add_argument("--base-model", required=True)
        child.add_argument("--prompt", required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--device", default=get_device_name())
        child.add_argument("--fps", type=int, default=16)
        if command == "decode":
            child.add_argument("--latent", type=Path, required=True)
        else:
            child.add_argument("--adapter", type=Path)
            child.add_argument("--manifest", type=Path)
            child.add_argument("--frames-per-block", type=int, default=3)
            child.add_argument("--seed", type=int, default=42)
            child.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
            child.add_argument("--trajectory", type=Path)
    page = subparsers.add_parser("gallery")
    page.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate" and not args.adapter and not args.manifest:
        parser.error("generate requires --adapter, or --manifest for the untrained causal baseline")
    {"export": export_adapter, "generate": generate, "decode": decode, "gallery": gallery}[args.command](args)


if __name__ == "__main__":
    main()

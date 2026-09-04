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

"""Shared LTX-2.x FlowGRPO constants and numerical helpers."""

import inspect
import math
from typing import Any

import torch

LTX2_LORA_TARGET_MODULES = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
]


def normalize_ltx_output_type(output_type: str | None) -> str | None:
    """Translate the generic image default to an LTX video tensor output."""
    return "pt" if output_type == "image" else output_type


def calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int,
    max_image_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """Calculate the flow scheduler's dynamic timestep shift."""
    slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    intercept = base_shift - slope * base_image_seq_len
    return image_seq_len * slope + intercept


def is_ltx25_transformer_config(config: Any) -> bool:
    """Return whether a transformer config uses the LTX-2.5 parameter layout."""
    if config is None:
        return False
    if hasattr(config, "get"):
        return config.get("ff_bias") is False
    return getattr(config, "ff_bias", None) is False


def set_ltx25_timesteps(scheduler: Any, num_inference_steps: int, device: str | torch.device) -> None:
    """Configure the official LTX-2.5 Full/SFT one-stage sigma schedule."""
    if num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}.")

    config = scheduler.config
    base_anchor = config.get("base_image_seq_len", 1024)
    max_anchor = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.95)
    max_shift = config.get("max_shift", 2.05)
    sigma_shift = calculate_shift(max_anchor, base_anchor, max_anchor, base_shift, max_shift)

    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
    exp_shift = math.exp(sigma_shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

    terminal = config.get("shift_terminal")
    terminal = 0.1 if terminal is None else terminal
    non_zero = sigmas != 0
    one_minus_sigmas = 1.0 - sigmas[non_zero]
    scale = one_minus_sigmas[-1] / (1.0 - terminal)
    sigmas[non_zero] = 1.0 - one_minus_sigmas / scale

    scheduler.sigmas = sigmas.to(device=device)
    scheduler.timesteps = scheduler.sigmas[:-1] * config.get("num_train_timesteps", 1000)
    scheduler.num_inference_steps = num_inference_steps
    scheduler._step_index = None
    scheduler._begin_index = None


def _first_frame_mask(hidden_states: torch.Tensor, num_frames: int) -> torch.Tensor:
    tokens_per_frame, remainder = divmod(hidden_states.shape[1], num_frames)
    if remainder:
        raise ValueError(
            f"LTX video token count {hidden_states.shape[1]} is not divisible by {num_frames} latent frames."
        )
    mask = hidden_states.new_zeros((hidden_states.shape[0], hidden_states.shape[1], 1))
    mask[:, :tokens_per_frame] = 1
    return mask


def forward_ltx_transformer(module: torch.nn.Module, model_inputs: dict[str, Any]) -> Any:
    """Run the transformer with the LTX-2.5 first-frame embedding when required."""
    config = getattr(module, "config", None)
    use_keyframes = (
        config.get("use_keyframes_abs_pos_embedding", False)
        if hasattr(config, "get")
        else getattr(config, "use_keyframes_abs_pos_embedding", False)
    )
    if not use_keyframes or getattr(module, "keyframes_abs_pos_embedding", None) is None:
        return module(**model_inputs)

    keyframes_mask = _first_frame_mask(model_inputs["hidden_states"], int(model_inputs["num_frames"]))
    if "keyframes_mask" in inspect.signature(module.forward).parameters:
        return module(**model_inputs, keyframes_mask=keyframes_mask)

    def add_keyframe_embedding(
        _projection: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        embedding = module.keyframes_abs_pos_embedding
        return output + keyframes_mask.to(device=output.device, dtype=output.dtype) * embedding.to(
            device=output.device, dtype=output.dtype
        )

    handle = module.proj_in.register_forward_hook(add_keyframe_embedding)
    try:
        return module(**model_inputs)
    finally:
        handle.remove()


def apply_x0_cfg(
    sample: torch.Tensor,
    positive_velocity: torch.Tensor,
    negative_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply LTX classifier-free guidance in clean-sample space."""
    positive_x0 = sample - sigma * positive_velocity
    negative_x0 = sample - sigma * negative_velocity
    guided_x0 = positive_x0 + (guidance_scale - 1.0) * (positive_x0 - negative_x0)
    return (sample - guided_x0) / sigma

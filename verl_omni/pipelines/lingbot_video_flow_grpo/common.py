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

"""Small dependency-free helpers shared by the LingBot dense adapters."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

# Kept in sync with Robbyant/lingbot-video at a638721.  Keeping the template
# here lets the dataset-side agent loop run before the optional model package is
# imported by a rollout worker.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative":{"visual_quality":["low quality","worst quality","blurry",'
    '"pixelated","jpeg artifacts","low resolution","unstable color","color flicker",'
    '"underexposed","overexposed","invisible subject","subject hidden in darkness"],'
    '"artistic_style":["painting","illustration","drawing","cartoon","3d render",'
    '"cgi","sketch","digital art"],"composition_and_content":["text","watermark",'
    '"signature","logo","subtitles","pillarboxed","side bars","portrait image in landscape frame"],'
    '"temporal_and_motion_stability":["flickering","jittery","motion blur",'
    '"temporal inconsistency","warping","morphing","incoherent motion","unnatural movement",'
    '"static object with sudden jump","frame-to-frame inconsistency"],"material_and_structure":'
    '["plastic-like glass","unrealistic texture","deformed bottle","liquid freezing improperly",'
    '"distorted reflections"]}}'
)


def caption_to_json(caption: Any) -> str:
    """Serialize a structured LingBot caption with the official compact JSON form.

    The model is trained on structured captions.  Rejecting plain text here is
    intentional: silently accepting it would make an incorrectly preprocessed
    dataset look like a valid LingBot rollout.
    """

    if isinstance(caption, str):
        try:
            caption = json.loads(caption)
        except json.JSONDecodeError as exc:
            raise ValueError("LingBot T2V prompts must be valid structured JSON.") from exc
    if not isinstance(caption, dict | list):
        raise TypeError(f"LingBot T2V prompts must be a mapping, list, or JSON string; got {type(caption).__name__}.")
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":"))


def apply_prompt_template(caption: str) -> str:
    """Apply the official LingBot text-only T2V template."""

    return PROMPT_TEMPLATE.format(caption)


def validate_t2v_dimensions(height: int, width: int, num_frames: int) -> None:
    """Validate the official Dense T2V spatial and temporal constraints."""

    if num_frames != 1 and (num_frames - 1) % 4 != 0:
        raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")


def shifted_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
    """Return the LingBot/FlowMatch shifted sigma schedule without terminal zero."""

    if num_inference_steps <= 0:
        raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}.")
    if shift <= 0:
        raise ValueError(f"`shift` must be positive, got {shift}.")
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    return sigmas[:-1].cpu().numpy()


def apply_cfg(noise_pred: torch.Tensor, negative_noise_pred: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Apply LingBot's standard classifier-free guidance rule."""

    return negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)


def guidance_scale(pipeline_config: Any) -> float:
    """Resolve the user-facing LingBot guidance value from shared pipeline config."""

    value = getattr(pipeline_config, "guidance_scale", None)
    if value is None and hasattr(pipeline_config, "get"):
        value = pipeline_config.get("guidance_scale", None)
    if value is None:
        value = getattr(pipeline_config, "true_cfg_scale", 1.0)
        if hasattr(pipeline_config, "get"):
            value = pipeline_config.get("true_cfg_scale", value)
    return float(value)

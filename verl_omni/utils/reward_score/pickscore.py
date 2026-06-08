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

"""PickScore reward function for image generation/editing."""

import asyncio
import io
import logging
import threading
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "yuvalkirstain/PickScore_v1"
SCORE_SCALE = 26.0

_lock: threading.Lock = threading.Lock()
_processor: Any = None
_model: Any = None
_device: str = "cuda"


def _load_model() -> None:
    """Thread-safe lazy loading of PickScore model and processor."""
    global _device, _model, _processor

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading PickScore processor: %s", PROCESSOR_NAME)
        _processor = CLIPProcessor.from_pretrained(PROCESSOR_NAME)
        logger.info("Loading PickScore model: %s", MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME).eval().to(_device)


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    """Extract feature tensor from transformers outputs across versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    raise TypeError(f"Unsupported PickScore feature output type: {type(output)}")


def _to_pil(image: Any) -> Image.Image:
    """Convert tensor / ndarray / PIL / parquet image dict to RGB PIL image."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])

    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        image = image.float().permute(1, 2, 0).cpu().numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        assert image.shape[-1] == 3, "PickScore expects RGB images in HWC format"
        if image.dtype != np.uint8:
            image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    assert isinstance(image, Image.Image)
    return image.convert("RGB")


async def compute_score(
    data_source: str,
    solution_image: torch.Tensor | np.ndarray | Image.Image | dict,
    ground_truth: str,
    extra_info: dict,
) -> dict[str, float]:
    """Compute PickScore reward.

    This function is async-compatible: VisualRewardManager calls it via ``await``
    inside the main event loop rather than ``run_in_executor``, so the PickScore
    model stays loaded on the reward worker's GPU across samples.

    Args:
        data_source: Dataset name, kept for verl-omni reward interface compatibility.
        solution_image: Generated image in CHW / NCHW tensor, HWC / NHWC ndarray,
            PIL image, or parquet image dict format.
        ground_truth: Text prompt or edit instruction.
        extra_info: Extra sample metadata, kept for interface compatibility.

    Returns:
        A dict containing normalized ``score`` and unnormalized ``pickscore_raw``.
    """
    _load_model()
    loop = asyncio.get_event_loop()

    image = await loop.run_in_executor(None, _to_pil, solution_image)

    image_inputs = _processor(
        images=[image],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(_device)
    text_inputs = _processor(
        text=[ground_truth],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        image_embs = _extract_feature_tensor(_model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = _extract_feature_tensor(_model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        raw_score = _model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        raw_score = raw_score[0]
        score = raw_score / SCORE_SCALE

    return {"score": float(score.cpu()), "pickscore_raw": float(raw_score.cpu())}

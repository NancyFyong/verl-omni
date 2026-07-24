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

"""Media conversion helpers shared by trainer output paths."""

import torch
from PIL import Image


def video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert an RGB ``[T, C, H, W]`` tensor in ``[0, 1]`` to PIL frames.

    PIL frames are intentional here. ``diffusers.utils.export_to_video`` scales
    NumPy inputs by 255 even when they are already ``uint8``; passing PIL images
    prevents that second scaling and the resulting modulo-256 color inversion.
    Values outside the display range, including non-finite decoder output, are
    sanitized before quantization.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"Expected an RGB video tensor with shape [T, 3, H, W], got {tuple(video.shape)}")

    frames = []
    for frame in video:
        frame = frame.detach().permute(1, 2, 0).to(dtype=torch.float32)
        frame = torch.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
        frame = frame.mul_(255).round_().to(dtype=torch.uint8, device="cpu").contiguous().numpy()
        frames.append(Image.fromarray(frame))
    return frames

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

"""CPU tests for media conversion helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_MEDIA_PATH = Path(__file__).resolve().parents[2] / "verl_omni" / "utils" / "media.py"
_MEDIA_SPEC = importlib.util.spec_from_file_location("media_under_test", _MEDIA_PATH)
assert _MEDIA_SPEC is not None
assert _MEDIA_SPEC.loader is not None
_MEDIA_MODULE = importlib.util.module_from_spec(_MEDIA_SPEC)
_MEDIA_SPEC.loader.exec_module(_MEDIA_MODULE)
video_tensor_to_pil_frames = _MEDIA_MODULE.video_tensor_to_pil_frames


def test_video_tensor_to_pil_frames_clips_before_uint8_quantization():
    video = torch.tensor(
        [
            [
                [[-0.1, 0.0, 1.0, 1.1]],
                [[float("nan"), 0.5, float("inf"), -float("inf")]],
                [[0.25, 0.75, 0.999, 0.001]],
            ]
        ]
    )

    frames = video_tensor_to_pil_frames(video)

    assert len(frames) == 1
    assert frames[0].mode == "RGB"
    pixels = np.asarray(frames[0])
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[[0, 0, 64], [0, 128, 191], [255, 255, 255], [255, 0, 0]]]


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 4, 4, 5), (2, 3, 4, 5, 1)])
def test_video_tensor_to_pil_frames_rejects_non_tchw_rgb(shape):
    with pytest.raises(ValueError, match="Expected an RGB video tensor"):
        video_tensor_to_pil_frames(torch.zeros(shape))

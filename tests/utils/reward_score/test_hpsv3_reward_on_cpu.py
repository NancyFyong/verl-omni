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
"""CPU tests for HPSv3 reward frame extraction."""

import numpy as np
import torch

from verl_omni.utils.reward_score.hpsv3_reward import _extract_frames


def _rgb_at(image):
    return tuple(np.asarray(image)[0, 0].tolist())


def _fill_tchw(video, b=None):
    if b is None:
        video[0, :, 0, 0] = torch.tensor([0.1, 0.2, 0.3])
        video[2, :, 0, 0] = torch.tensor([0.5, 0.6, 0.7])
    else:
        video[b, 0, :, 0, 0] = torch.tensor([0.1 + b * 0.1, 0.2, 0.3])
        video[b, 3, :, 0, 0] = torch.tensor([0.5 + b * 0.1, 0.6, 0.7])


def _fill_thwc(video, b=None):
    if b is None:
        video[0, 0, 0, :] = torch.tensor([0.1, 0.2, 0.3])
        video[2, 0, 0, :] = torch.tensor([0.5, 0.6, 0.7])
    else:
        video[b, 0, 0, 0, :] = torch.tensor([0.1 + b * 0.1, 0.2, 0.3])
        video[b, 3, 0, 0, :] = torch.tensor([0.5 + b * 0.1, 0.6, 0.7])


def test_extract_frames_subsamples_per_sample_tchw_video_on_time_axis():
    video = torch.zeros(4, 3, 2, 2)
    _fill_tchw(video)

    frames = _extract_frames(video, frame_interval=2)

    assert len(frames) == 2
    assert frames[0].size == (2, 2)
    assert _rgb_at(frames[0]) == (26, 51, 76)
    assert _rgb_at(frames[1]) == (128, 153, 178)


def test_extract_frames_subsamples_per_sample_thwc_video_on_time_axis():
    video = torch.zeros(4, 3, 4, 3)
    _fill_thwc(video)

    frames = _extract_frames(video, frame_interval=2)

    assert len(frames) == 2
    assert frames[0].size == (4, 3)
    assert _rgb_at(frames[0]) == (26, 51, 76)
    assert _rgb_at(frames[1]) == (128, 153, 178)


def test_extract_frames_flattens_batched_btchw_video_after_time_subsample():
    video = torch.zeros(2, 4, 3, 2, 2)
    _fill_tchw(video, b=0)
    _fill_tchw(video, b=1)

    frames = _extract_frames(video, frame_interval=3)

    assert len(frames) == 4
    assert [_rgb_at(frame) for frame in frames] == [
        (26, 51, 76),
        (128, 153, 178),
        (51, 51, 76),
        (153, 153, 178),
    ]


def test_extract_frames_flattens_batched_bthwc_video_after_time_subsample():
    video = torch.zeros(2, 4, 3, 4, 3)
    _fill_thwc(video, b=0)
    _fill_thwc(video, b=1)

    frames = _extract_frames(video, frame_interval=3)

    assert len(frames) == 4
    assert frames[0].size == (4, 3)
    assert [_rgb_at(frame) for frame in frames] == [
        (26, 51, 76),
        (128, 153, 178),
        (51, 51, 76),
        (153, 153, 178),
    ]

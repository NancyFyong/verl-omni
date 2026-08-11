# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace

from verl_omni.utils.reward_score import hpsv3_reward


def test_hpsv3_default_npu_falls_back_to_cuda_when_npu_unavailable(monkeypatch):
    monkeypatch.delenv("custom_reward_device", raising=False)
    monkeypatch.setattr(hpsv3_reward.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hpsv3_reward.torch, "npu", None, raising=False)

    assert hpsv3_reward._resolve_device("npu") == "cuda"


def test_hpsv3_custom_reward_device_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("custom_reward_device", "cpu")
    monkeypatch.setattr(hpsv3_reward.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hpsv3_reward.torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False)

    assert hpsv3_reward._resolve_device("npu") == "cpu"


def test_hpsv3_extract_frames_accepts_tchw_video():
    import torch

    video = torch.zeros(5, 3, 4, 6)
    frames = hpsv3_reward._extract_frames(video, frame_interval=2)

    assert len(frames) == 3
    assert frames[0].size == (6, 4)


def test_hpsv3_extract_frames_accepts_cthw_video():
    import torch

    video = torch.zeros(3, 5, 4, 6)
    frames = hpsv3_reward._extract_frames(video, frame_interval=2)

    assert len(frames) == 3
    assert frames[0].size == (6, 4)


def test_hpsv3_extract_frames_accepts_batched_btchw_video():
    import torch

    video = torch.zeros(2, 5, 3, 4, 6)
    frames = hpsv3_reward._extract_frames(video, frame_interval=2)

    assert len(frames) == 6
    assert frames[0].size == (6, 4)

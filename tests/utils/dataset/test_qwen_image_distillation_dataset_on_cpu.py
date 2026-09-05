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
"""CPU tests for Qwen-Image original-DMD regression-pair data."""

import io

import pytest
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_omni.utils.dataset.qwen_image_distillation_dataset import QwenImageDMDPairDataset, load_float_tensor


class TestDMDTensorLoading:
    def test_loads_nested_values_and_serialized_tensors(self):
        nested = load_float_tensor([[1, 2], [3, 4]], "value")
        buffer = io.BytesIO()
        torch.save(torch.tensor([5.0]), buffer)
        serialized = load_float_tensor(buffer.getvalue(), "value")

        assert nested.dtype == torch.float32
        torch.testing.assert_close(nested, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        torch.testing.assert_close(serialized, torch.tensor([5.0]))

    def test_rejects_empty_tensor(self):
        with pytest.raises(ValueError, match="must not be empty"):
            load_float_tensor([], "value")


class TestQwenImageDMDPairDataset:
    @staticmethod
    def make_dataset():
        return object.__new__(QwenImageDMDPairDataset)

    def test_converts_regression_pair_and_preserves_manifest(self, monkeypatch):
        row = {
            "reference_noise": [[[1.0]]],
            "teacher_target_latents": [[[2.0]]],
            "teacher_sampling_manifest": {"model": "teacher"},
            "index": 7,
        }
        monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: dict(row))

        output = self.make_dataset()[0]

        assert output["reference_noise"].dtype == torch.float32
        assert output["teacher_target_latents"].dtype == torch.float32
        assert output["pair_id"] == "7"
        assert output["teacher_sampling_manifest"] == {"model": "teacher"}

    @pytest.mark.parametrize(
        "row,error",
        [
            ({"teacher_target_latents": [1], "teacher_sampling_manifest": {"x": 1}}, "reference_noise"),
            ({"reference_noise": [1], "teacher_sampling_manifest": {"x": 1}}, "exactly one teacher target"),
            (
                {
                    "reference_noise": [1],
                    "teacher_target_latents": [1],
                    "teacher_target_pixels": [1],
                    "teacher_sampling_manifest": {"x": 1},
                },
                "exactly one teacher target",
            ),
            ({"reference_noise": [1], "teacher_target_latents": [1]}, "teacher_sampling_manifest"),
        ],
    )
    def test_rejects_incomplete_rows(self, monkeypatch, row, error):
        monkeypatch.setattr(RLHFDataset, "__getitem__", lambda self, item: dict(row))
        with pytest.raises(ValueError, match=error):
            self.make_dataset()[0]

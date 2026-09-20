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
"""Tiny-run evidence hooks must distinguish packing and restore temporary patches."""

import json
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _STEP_TRANSFORMER

from tests.special_e2e.minimax_h3_tiny_patch.smoke_trace import instrument_pipeline


@pytest.mark.parametrize("fail", [False, True])
def test_trace_records_kernel_calls_and_restores_hooks(tmp_path, monkeypatch, fail):
    def fake_flash(impl, query, key, value, **metadata):
        return query + value

    monkeypatch.setattr(FlashAttentionImpl, "_forward_varlen_packed", fake_flash)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", fake_flash)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata = {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": 2, "max_seqlen_k": 2}

    class Model(torch.nn.Module):
        def forward(self, *, packed_seq_params):
            if fail:
                raise RuntimeError("forward failed")
            q = torch.ones(1, 4, 1, 8)
            return FlashAttentionImpl._forward_varlen_packed(
                SimpleNamespace(softmax_scale=8**-0.5, causal=False), q, q, q, **metadata
            )

    model = Model()

    class Pipeline:
        def forward(self, arg):
            raise NotImplementedError

        def post_decode(self, arg):
            raise NotImplementedError

        def denoise_step(self, arg):
            return model(packed_seq_params={"num_requests": 2, **metadata})

    instrument_pipeline(Pipeline, tmp_path / "FL2VA" / "text_encoder" / "config.json")
    arg = SimpleNamespace(states=[SimpleNamespace(step_index=0, extra={_STEP_TRANSFORMER: model}) for _ in range(2)])
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            Pipeline().denoise_step(arg)
    else:
        Pipeline().denoise_step(arg)
        path = next((tmp_path / "smoke_traces").glob("*/events.jsonl"))
        event = json.loads(path.read_text())
        assert event["dit_forwards"] == [{"num_requests": 2, "cu_seqlens_q": [0, 2, 4]}]
        assert len(event["flash_packed_calls"]) == 1
        assert (path.parent / event["flash_packed_calls"][0]["file"]).is_file()
    assert FlashAttentionImpl._forward_varlen_packed is fake_flash
    assert not model._forward_pre_hooks

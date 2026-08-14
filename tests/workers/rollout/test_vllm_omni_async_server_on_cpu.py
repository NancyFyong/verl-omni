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
"""CPU contract tests for vLLM-Omni diffusion output compatibility."""

from types import SimpleNamespace

import numpy as np
import pytest

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")


def _server():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server._ar_mode = False
    server.global_steps = 3
    return server


def test_diffusion_output_without_legacy_request_output():
    final = SimpleNamespace(
        images=[np.zeros((2, 2, 3), dtype=np.uint8)],
        custom_output={},
        multimodal_output={},
    )

    output = _server()._process_output(final, params=None, sampling_params={})

    assert output.stop_reason == "completed"
    assert tuple(output.diffusion_output.shape) == (2, 2, 3)
    assert output.extra_fields["global_steps"] == 3


def test_aborted_output_without_legacy_request_output():
    final = SimpleNamespace(images=[], custom_output={}, multimodal_output={})

    output = _server()._process_output(final, params=None, sampling_params={})

    assert output.stop_reason == "aborted"
    assert output.diffusion_output.numel() == 0

# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright contributors to the vLLM-Omni project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3-Omni compatibility with vLLM 0.30's CUDA Graph bytecode guard."""

from collections.abc import Sequence

import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3MoeLLMModel

_ORIGINAL_THINKER_FORWARD = Qwen3MoeLLMModel.forward


def patch_qwen3_omni_thinker_forward() -> None:
    """Wrap the pinned Thinker forward before model construction in each worker."""
    if "update" in Qwen3MoeLLMModel.forward.__code__.co_names:
        Qwen3MoeLLMModel.forward = _qwen3_omni_thinker_forward


# Dynamo inlines the upstream call, consuming its local dict.update during tracing.
# Real buffer mutations still emit update in transformed bytecode and retain vLLM's guard.
# TODO: Remove when the pin fixes the local dict.update introduced by
# https://github.com/vllm-project/vllm-omni/pull/7345 and the unwrapped forward compiles.
def _qwen3_omni_thinker_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    capture_layer_indices: Sequence[int] | None = None,
    return_hidden_states: bool = False,
    deepstack_input_embeds: IntermediateTensors | None = None,
) -> torch.Tensor | IntermediateTensors:
    return _ORIGINAL_THINKER_FORWARD(
        self,
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        capture_layer_indices=capture_layer_indices,
        return_hidden_states=return_hidden_states,
        deepstack_input_embeds=deepstack_input_embeds,
    )

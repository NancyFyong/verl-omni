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
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors


def patch_qwen3_omni_thinker_forward() -> None:
    """Replace the pinned Thinker forward before model construction in each worker."""
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3MoeLLMModel

    if "update" in Qwen3MoeLLMModel.forward.__code__.co_names:
        # Preserve the class's support_torch_compile decorator and its dynamic argument dimensions.
        Qwen3MoeLLMModel.forward = _qwen3_omni_thinker_forward


# Copied from vllm-omni 12e9280 Qwen3MoeLLMModel.forward (Apache-2.0), changing only the local dict update.
# TODO: Remove when the pin includes a CUDA Graph-safe replacement for the dict.update introduced by
# https://github.com/vllm-project/vllm-omni/pull/7345. vLLM 0.30 rejects any "update" in forward.co_names,
# even on PP=1 where Dynamo eliminates the branch; this is not an nn.Module buffer mutation.
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
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import PP_CAPTURE_PREFIX

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
    capture_set = set(capture_layer_indices) if capture_layer_indices else None
    captured_hidden_states = {} if return_hidden_states else None

    if captured_hidden_states is not None and capture_set and intermediate_tensors is not None:
        for layer_idx in capture_set:
            if layer_idx < self.start_layer:
                hs = captured_hidden_states.setdefault("hidden_states", {})
                layers = hs.setdefault("layers", {})
                # Receive buffers are reused on the next step; retain an independent snapshot.
                layers[layer_idx] = intermediate_tensors[f"{PP_CAPTURE_PREFIX}{layer_idx}"].clone()

    for layer_idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
        layer_idx = layer_idx + self.start_layer

        if captured_hidden_states is not None and capture_set is not None:
            if layer_idx in capture_set:
                hs = captured_hidden_states.setdefault("hidden_states", {})
                layers = hs.setdefault("layers", {})
                # vLLM defers the residual addition until the next RMSNorm.
                # Reconstruct the logical decoder state before capturing it.
                captured = hidden_states.clone() if residual is None else hidden_states + residual
                layers[layer_idx] = captured.view(-1, captured.shape[-1])

        hidden_states, residual = layer(
            positions,
            hidden_states,
            residual,
        )

        if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
            hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]

    if not get_pp_group().is_last_rank:
        tensors = {"hidden_states": hidden_states, "residual": residual}
        if captured_hidden_states:
            for index, value in captured_hidden_states["hidden_states"]["layers"].items():
                tensors[f"{PP_CAPTURE_PREFIX}{index}"] = value
        return IntermediateTensors(tensors)
    hidden_states, _ = self.norm(hidden_states, residual)
    if captured_hidden_states is not None:
        return hidden_states, captured_hidden_states
    else:
        return hidden_states, None

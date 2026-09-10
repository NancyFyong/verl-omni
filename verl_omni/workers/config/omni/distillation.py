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
"""Omni teacher configuration, adapted from the shared OPD integration in PR #375."""

from dataclasses import dataclass
from typing import Optional

from verl.workers.config import DistillationTeacherModelConfig

__all__ = ["OmniDistillationTeacherModelConfig"]


@dataclass
class OmniDistillationTeacherModelConfig(DistillationTeacherModelConfig):
    """Extend the upstream teacher's log-probability budget to vLLM-Omni."""

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if self.inference.name != "vllm_omni" or not use_topk:
            return super()._validate_topk_logprobs(use_topk, topk)
        if topk is None or topk <= 0:
            raise ValueError("topk must be positive when requesting teacher top-k log probabilities.")
        engine_kwargs = dict(self.inference.engine_kwargs.get("vllm_omni", {}))
        max_logprobs = engine_kwargs.get("max_logprobs")
        if max_logprobs is None:
            engine_kwargs["max_logprobs"] = topk
        elif max_logprobs < topk:
            raise ValueError(f"vllm_omni max_logprobs ({max_logprobs}) must be >= distillation topk ({topk}).")
        self.inference.engine_kwargs["vllm_omni"] = engine_kwargs

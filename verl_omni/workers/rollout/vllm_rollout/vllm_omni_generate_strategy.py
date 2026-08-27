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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch
from verl.utils.tokenizer import normalize_token_ids
from vllm_omni.lora.request import LoRARequest

if TYPE_CHECKING:
    from argparse import Namespace

    from verl.workers.rollout.replica import TokenOutput

    from verl_omni.workers.rollout.replica import DiffusionOutput
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


class _OmniGenerateStrategy(ABC):
    """Mode-specific vLLM-Omni configuration and generation behavior."""

    def __init__(self, server: vLLMOmniHttpServer) -> None:
        self.server = server

    @abstractmethod
    def init_config(self, config: Any) -> Any:
        pass

    @abstractmethod
    def init_model_config(self, model_config: Any) -> Any:
        pass

    def validate_configs(self) -> None:
        return None

    def post_init(self, cuda_visible_devices: str) -> None:
        return None

    def apply_quantization(self) -> tuple[str | None, dict[str, Any]]:
        return None, {}

    def override_generation_config(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def worker_extension_cls(self, device_type: str) -> str:
        pass

    def preprocess_engine_kwargs(self, engine_kwargs: dict[str, Any]) -> None:
        engine_kwargs.pop("output_mode", None)

    @abstractmethod
    def prepare_engine_args(self, engine_args: dict[str, Any], args: Namespace) -> None:
        pass

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        prompt_ids = normalize_token_ids(prompt_ids)
        multi_modal_data = self._build_multi_modal_data(image_data, video_data, audio_data)
        lora_request = await self.server._resolve_lora_request()
        prompt, params = self.preprocess_input(
            prompt_ids,
            sampling_params,
            multi_modal_data,
            lora_request,
            negative_prompt_ids,
            prompt_mask,
            mm_processor_kwargs,
            extra_prompt_ids,
            negative_extra_prompt_ids,
        )
        final_res = await self.run_generation(prompt, params, request_id, lora_request, priority)
        return self.process_output(final_res, params, sampling_params)

    @staticmethod
    def _build_multi_modal_data(
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        audio_data: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Assemble vLLM multimodal inputs from the public rollout arguments."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data
        return multi_modal_data

    @staticmethod
    def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
        """Map a vLLM finish reason to verl's stop-reason vocabulary."""
        if finish_reason == "abort":
            return "aborted"
        if finish_reason in ("stop", "length"):
            return "completed"
        return finish_reason

    @abstractmethod
    def preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        lora_request: Optional[LoRARequest],
        negative_prompt_ids: Optional[list[int]],
        prompt_mask: torch.BoolTensor | None = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        extra_prompt_ids: Optional[dict[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[dict[str, list[int]]] = None,
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    async def run_generation(
        self,
        prompt: Any,
        params: Any,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
    ) -> Any:
        pass

    @abstractmethod
    def process_output(self, final_res: Any, params: Any, sampling_params: dict[str, Any]) -> Any:
        pass

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
"""NFT compatibility exports and its existing H3 prompt/sync policies."""

from typing import Any

import torch

from verl_omni.pipelines.minimax_h3_shared.common import *  # noqa: F403
from verl_omni.pipelines.minimax_h3_shared.common import (
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    _PromptTokenOverride,
)
from verl_omni.pipelines.minimax_h3_shared.common import __all__ as _COMMON_EXPORTS
from verl_omni.pipelines.minimax_h3_shared.weight_sync import (
    _LORA_STACKED_PARAMS_MAPPING,
    _LORA_VLLM_TARGET_MODULES,
    MiniMaxH3WeightSyncBase,
    map_lora_tensors,
    validate_lora_target_modules,
)
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload

__all__ = [
    *_COMMON_EXPORTS,
    "MiniMaxH3RolloutWeightSyncMixin",
    "validate_lora_target_modules",
    "_LORA_STACKED_PARAMS_MAPPING",
    "_LORA_VLLM_TARGET_MODULES",
]


class MiniMaxH3RolloutWeightSyncMixin(MiniMaxH3WeightSyncBase):
    """Keep NFT's prompt encoding, transformer selection and first-sync RoPE policy."""

    _h3_sync_components = ("transformer",)
    _h3_initialize_rope = True

    def encode_prompt(self, *, task: str, prompt: str, image=None, images=None, **kwargs):
        """Encode Agent Loop IDs while letting vLLM-Omni build reference vision spans."""
        prompt_ids = getattr(self, "_h3_prompt_ids", None)
        if prompt_ids is None or task not in {"t2va", "fl2va", "ref2va"}:
            return super().encode_prompt(task=task, prompt=prompt, image=image, images=images, **kwargs)

        if task == "ref2va":
            # Let the upstream pipeline build every reference span; the override keeps the Agent Loop text token IDs.
            tokenizer = self.tokenizer
            self.tokenizer = _PromptTokenOverride(tokenizer, prompt, prompt_ids)
            try:
                return super().encode_prompt(task=task, prompt=prompt, image=image, images=images, **kwargs)
            finally:
                self.tokenizer = tokenizer

        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            _broadcast_tensor,
            _dit_rank_world,
            minimax_h3_multi_image_presentation,
        )

        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        condition_images = list(images) if images is not None else ([image] if image is not None else [])
        if rank == 0:
            if task == "t2va":
                ids = prompt_ids
                tags = torch.ones(ids.shape[0], dtype=torch.long)
            else:
                if not condition_images:
                    raise ValueError(f"MiniMax H3 {task} requires at least one condition image.")
                vision = self.processor.image_processor(images=condition_images, return_tensors="pt")
                image_grid = vision["image_grid_thw"]
                merge = int(self.processor.image_processor.merge_size) ** 2
                image_token_counts = [int(grid.prod().item()) // merge for grid in image_grid]
                prefix_ids, prefix_tags = minimax_h3_multi_image_presentation(
                    self.tokenizer, prompt="", image_token_counts=image_token_counts
                )
                ids = torch.cat([prefix_ids, prompt_ids])
                tags = torch.cat([prefix_tags, torch.ones(prompt_ids.shape[0], dtype=torch.long)])
                vision_kwargs = {
                    "pixel_values": vision["pixel_values"],
                    "image_grid_thw": image_grid,
                }

        if rank < self.text_encoder_tp_size:
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)
        hidden = _broadcast_tensor(hidden, dtype=torch.bfloat16, device=self.device)
        tags = _broadcast_tensor(tags, dtype=torch.long, device=self.device)
        return hidden, tags

    def _install_lora_layout(self) -> None:
        self.install_h3_lora_layout()

    def map_lora_update_to_engine(
        self, tensors: dict[str, torch.Tensor], peft_config: dict
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Retain NFT's target expansion and permissive payload handling."""
        target_modules = peft_config.get("target_modules") if peft_config is not None else None
        validate_lora_target_modules(target_modules)
        mapped = map_lora_tensors(tensors, "transformer", self.transformer.arch.ffn_hidden_size, strict=False)
        new_config = dict(peft_config) if peft_config is not None else {}
        new_config["target_modules"] = list(_LORA_VLLM_TARGET_MODULES)
        return mapped, new_config

    def _ensure_prompt_text(self, request: Any) -> None:
        """Expose pre-tokenized IDs and satisfy the upstream non-empty-text check."""
        self._h3_prompt_ids = None
        prompts = getattr(request, "prompts", None)
        if not prompts or not isinstance(prompts[0], dict):
            return
        custom_prompt = prompts[0]
        token_ids = prompt_ids_from_payload(custom_prompt)
        if token_ids is None:
            return
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        if extra_args.get(MINIMAX_H3_TOKEN_ID_NATIVE_KEY) is not True:
            raise ValueError(
                "MiniMax H3 token-ID-native rollout requires "
                "actor_rollout_ref.rollout.agent.default_agent_loop="
                "minimax_h3_diffusion_single_turn_agent."
            )
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        self._h3_prompt_ids = torch.as_tensor([int(token) for token in token_ids], dtype=torch.long)
        if self._h3_prompt_ids.numel() == 0:
            raise ValueError("MiniMax H3 requires non-empty prompt_ids.")
        custom_prompt["prompt"] = "[pretokenized]"

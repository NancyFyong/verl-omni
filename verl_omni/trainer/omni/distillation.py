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
"""Token and padding contracts for the existing omni OPD trainer."""

import torch
from transformers import AutoTokenizer
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.utils.fs import resolve_model_local_dir


def validate_teacher_tokenizers(tokenizer, config):
    """Require compatible vocabulary and special-token mappings before allocating workers."""
    distillation = omega_conf_to_dataclass(config.distillation)
    for teacher in distillation.teacher_models.values():
        kwargs = teacher.inference.engine_kwargs.get("vllm_omni", {})
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            resolve_model_local_dir(teacher.model_path), trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or any(
            getattr(tokenizer, field) != getattr(teacher_tokenizer, field)
            for field in ("bos_token_id", "eos_token_id", "pad_token_id", "all_special_ids")
        ):
            raise ValueError(
                f"Omni OPD teacher {teacher.key!r} must share the student's tokenizer and special-token IDs."
            )


def install_teacher_padding():
    """Pad teacher sequence fields in verl's synthetic zero-loss samples (PR #375)."""
    from verl.trainer.ppo import padding_utils

    original = padding_utils.construct_minimal_padding_template
    if getattr(original, "_omni_teacher_padding", False):
        return

    def construct_minimal_padding_template(source_td, source_tag, eos_token_id):
        sample, tag = original(source_td, source_tag, eos_token_id)
        media = source_td.get("multi_modal_inputs")
        media = getattr(media, "data", media)
        if isinstance(media, dict) and "minicpm_duplex_replay" in media:
            sample["multi_modal_inputs"] = {"image_bound": [], "minicpm_duplex_replay": {}}
        length = sample["input_ids"].shape[0]
        for key, fill in (("teacher_ids", eos_token_id), ("teacher_logprobs", 0.0)):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                sample[key] = value.new_full((length, *value.shape[1:]), fill)
        return sample, tag

    construct_minimal_padding_template._omni_teacher_padding = True
    padding_utils.construct_minimal_padding_template = construct_minimal_padding_template

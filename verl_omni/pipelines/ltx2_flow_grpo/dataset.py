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

"""Dataset adapter for LTX media transport with a text-only prompt encoder."""

from typing import cast

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from verl.utils.dataset.rl_dataset import RLHFDataset


class _MediaTransportProcessor:
    pass


def ensure_ltx_media_processor(processor: ProcessorMixin | None) -> ProcessorMixin:
    """Provide the non-rendering marker needed to extract separately transported images."""
    return processor if processor is not None else cast(ProcessorMixin, _MediaTransportProcessor())


class LTX2TI2VADataset(RLHFDataset):
    """Allow image placeholders when the LTX Gemma-3 processor is unavailable."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizerBase,
        config: DictConfig,
        processor: ProcessorMixin | None = None,
        max_samples: int = -1,
    ) -> None:
        super().__init__(data_files, tokenizer, config, processor=processor, max_samples=max_samples)
        self.processor = ensure_ltx_media_processor(self.processor)

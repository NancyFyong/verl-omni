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
"""Model-publishing lifecycle, following verl.model_merger without HF initialization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMergerConfig:
    """Offline full-component export configuration; all paths must already be local."""

    local_dir: str
    target_dir: str
    base_model: str
    backend: str = "fsdp"
    architecture: str | None = None
    dtype: str = "preserve"
    max_shard_size: int = 2 * 1024**3
    trust_checkpoint: bool = False
    output_format: str = "pipeline"
    component: str | None = None

    def __post_init__(self):
        if self.output_format not in {"pipeline", "transformer"}:
            raise ValueError("output_format must be pipeline or transformer")
        if self.component not in {None, "transformer", "transformer_2"}:
            raise ValueError("component must be transformer or transformer_2")
        if self.backend != "fsdp":
            raise ValueError("Only the fsdp checkpoint backend is supported")
        if self.dtype not in {"preserve", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported output dtype: {self.dtype}")
        if type(self.max_shard_size) is not int or self.max_shard_size <= 0:
            raise ValueError("max_shard_size must be a positive byte count")
        if self.trust_checkpoint is not True:
            raise ValueError("Pickled rank checkpoints require explicit trust_checkpoint=True / --trust-checkpoint")


@dataclass(frozen=True)
class MergeResult:
    """Published artifact locations; detailed verification is recorded in the manifest."""

    output_dir: Path
    manifest_path: Path


class BaseModelMerger(ABC):
    """Common config, merge_and_save and cleanup lifecycle, without model loading."""

    def __init__(self, config: ModelMergerConfig):
        self.config = config

    @abstractmethod
    def merge_and_save(self) -> MergeResult:
        """Verify, reconstruct and publish the selected model."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release merger resources; publication transactions own their staging paths."""
        return None


def merge_model(config: ModelMergerConfig) -> MergeResult:
    """Run the audited Diffusers implementation through the common merger lifecycle."""
    from .diffusers_model_merger import DiffusersFSDPModelMerger

    merger = DiffusersFSDPModelMerger(config)
    try:
        return merger.merge_and_save()
    finally:
        merger.cleanup()

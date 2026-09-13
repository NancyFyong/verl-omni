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

from dataclasses import dataclass

from .model import DiffusionModelConfig

__all__ = ["MiniMaxH3ModelConfig"]


@dataclass
class MiniMaxH3ModelConfig(DiffusionModelConfig):
    """MiniMax-H3 Actor options shared by DiffusionNFT and FlowGRPO."""

    # Pack fixed Actor micro-batches; keep serial NFT / dense FlowGRPO by default.
    use_packed_batch: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.architecture != "MiniMaxH3Pipeline":
            raise ValueError("MiniMaxH3ModelConfig requires architecture=MiniMaxH3Pipeline.")
        if self.use_packed_batch and self.attn_backend not in {"native", "_flash_3_varlen_hub"}:
            raise ValueError("Packed MiniMax-H3 requires attn_backend=native or _flash_3_varlen_hub.")

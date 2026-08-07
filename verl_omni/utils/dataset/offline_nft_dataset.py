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

"""Offline diffusion NFT dataset utilities.

The on-policy DiffusionNFT path scores rollouts and then derives ``reward_prob``
(the optimality weight) group-wise in ``DiffusionNFTLoss.prepare_actor_batch``.
The offline fit loop never calls that helper, so each parquet row must already
carry the tensors the engine reads directly: the clean latent, the per-step
``train_timesteps`` pool, and the ``reward_prob`` weight. ``reward_prob`` is
expected to have been baked at prep time via the same
``DiffusionNFTLoss._compute_group_advantages`` / ``_advantage_to_reward_prob``
functions, so offline training is numerically faithful to the online path.

Big tensors are stored as ``torch.save`` bytes per cell and decoded with the DPO
sibling's ``_tensor_from_column``; there is no pairing, so every row is one sample.
"""

import uuid
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn

from verl_omni.utils.dataset.offline_dpo_dataset import _read_dataframe, _tensor_from_column


class OfflineNFTDataset(Dataset):
    """Dataset for pre-generated DiffusionNFT rollout samples with baked ``reward_prob``."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        del tokenizer, processor  # H3 offline NFT uses precomputed ``prompt_embeds``.
        if config is None:
            raise ValueError("OfflineNFTDataset requires a data config.")
        self.data_files = [data_files] if isinstance(data_files, str) else list(data_files)
        self.dataframe = _read_dataframe(self.data_files)
        if max_samples is not None and max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        self.config = config
        self.data_source = config.get("data_source", "offline_nft")

        required = {
            "latents_clean",
            "train_timesteps",
            "reward_prob",
            "latent_meta",
            "prompt_embeds",
            "prompt_embeds_mask",
            "sample_level_scores",
        }
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Offline NFT data is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()

        train_timesteps = _tensor_from_column(row["train_timesteps"], dtype=torch.float32)
        reward_prob = _tensor_from_column(row["reward_prob"], dtype=torch.float32)
        if train_timesteps.shape[0] != reward_prob.shape[0]:
            raise ValueError(
                f"Offline NFT row {item} has mismatched step counts: "
                f"train_timesteps={train_timesteps.shape[0]} vs reward_prob={reward_prob.shape[0]}."
            )

        raw_prompt = str(row.get("raw_prompt", ""))
        return {
            "uid": str(row.get("uid") or uuid.uuid4()),
            "latents_clean": _tensor_from_column(row["latents_clean"], dtype=torch.float32),
            "train_timesteps": train_timesteps,
            "reward_prob": reward_prob,
            "latent_meta": _tensor_from_column(row["latent_meta"], dtype=torch.long),
            "prompt_embeds": _tensor_from_column(row["prompt_embeds"], dtype=torch.float32),
            "prompt_embeds_mask": _tensor_from_column(row["prompt_embeds_mask"], dtype=torch.int32),
            "sample_level_scores": _tensor_from_column(row["sample_level_scores"], dtype=torch.float32),
            "raw_prompt": raw_prompt,
            "data_source": row.get("data_source", self.data_source),
            "reward_model": row.get("reward_model", {"style": "model", "ground_truth": raw_prompt}),
            "extra_info": {"index": int(item)},
        }


def offline_nft_collate_fn(features):
    """Collate pre-generated NFT samples (no pairing; one row is one sample)."""
    return _upstream_collate_fn(features)

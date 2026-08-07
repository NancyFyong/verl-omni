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
"""CPU tests for offline diffusion NFT dataset utilities."""

import io

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from verl_omni.utils.dataset.offline_nft_dataset import OfflineNFTDataset, offline_nft_collate_fn

# Tiny H3 layout mirroring the B3a smoke: meta [Nv, Na, latent_t, latent_h, latent_w, audio_t].
META = [4, 6, 1, 4, 4, 3]
VIDEO_ROW_WIDTH, AUDIO_ROW_WIDTH = 96, 32
PACKED_W = META[0] * VIDEO_ROW_WIDTH + META[1] * AUDIO_ROW_WIDTH
TEXT_DIM, TEXT_LEN, NUM_STEPS = 32, 5, 2


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _row(uid: str, score: float, reward_prob_val: float, **overrides) -> dict:
    row = {
        "uid": uid,
        "latents_clean": _tensor_bytes(torch.randn(PACKED_W)),
        "train_timesteps": _tensor_bytes(torch.tensor([750.0, 250.0])),
        "reward_prob": _tensor_bytes(torch.full((NUM_STEPS,), reward_prob_val)),
        "latent_meta": _tensor_bytes(torch.tensor(META, dtype=torch.long)),
        "prompt_embeds": _tensor_bytes(torch.randn(TEXT_LEN, TEXT_DIM)),
        "prompt_embeds_mask": _tensor_bytes(torch.ones(TEXT_LEN, dtype=torch.int32)),
        "sample_level_scores": _tensor_bytes(torch.tensor([score])),
        "raw_prompt": "a person speaking",
    }
    row.update(overrides)
    return row


def _parquet(tmp_path, rows) -> str:
    path = tmp_path / "nft.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _config():
    return OmegaConf.create({"data_source": "offline_nft"})


def test_dataset_item_decodes_row(tmp_path):
    data_file = _parquet(tmp_path, [_row("uid-0", 0.8, 0.53)])
    dataset = OfflineNFTDataset(data_file, tokenizer=None, config=_config())

    item = dataset[0]
    assert item["uid"] == "uid-0"
    assert item["latents_clean"].shape == (PACKED_W,)
    assert item["latents_clean"].dtype == torch.float32
    assert item["train_timesteps"].shape == (NUM_STEPS,)
    assert item["reward_prob"].shape == (NUM_STEPS,)
    assert item["latent_meta"].tolist() == META
    assert item["latent_meta"].dtype == torch.long
    assert item["prompt_embeds"].shape == (TEXT_LEN, TEXT_DIM)
    assert item["prompt_embeds_mask"].dtype == torch.int32
    torch.testing.assert_close(item["sample_level_scores"], torch.tensor([0.8]))
    assert item["data_source"] == "offline_nft"


def test_collate_stacks_batch(tmp_path):
    rows = [_row("uid-0", 0.8, 0.53), _row("uid-0", 0.2, 0.47)]
    dataset = OfflineNFTDataset(_parquet(tmp_path, rows), tokenizer=None, config=_config())

    batch = offline_nft_collate_fn([dataset[0], dataset[1]])
    assert batch["latents_clean"].shape == (2, PACKED_W)
    assert batch["train_timesteps"].shape == (2, NUM_STEPS)
    assert batch["reward_prob"].shape == (2, NUM_STEPS)
    assert batch["latent_meta"].shape == (2, len(META))
    assert batch["prompt_embeds"].shape == (2, TEXT_LEN, TEXT_DIM)
    assert batch["sample_level_scores"].shape == (2, 1)
    assert list(batch["uid"]) == ["uid-0", "uid-0"]


def test_missing_column_raises(tmp_path):
    row = _row("uid-0", 0.8, 0.53)
    del row["reward_prob"]
    with pytest.raises(ValueError, match="missing required columns"):
        OfflineNFTDataset(_parquet(tmp_path, [row]), tokenizer=None, config=_config())


def test_step_count_mismatch_raises(tmp_path):
    row = _row("uid-0", 0.8, 0.53, reward_prob=_tensor_bytes(torch.full((NUM_STEPS + 1,), 0.5)))
    dataset = OfflineNFTDataset(_parquet(tmp_path, [row]), tokenizer=None, config=_config())
    with pytest.raises(ValueError, match="mismatched step counts"):
        dataset[0]


def test_requires_config(tmp_path):
    with pytest.raises(ValueError, match="requires a data config"):
        OfflineNFTDataset(_parquet(tmp_path, [_row("uid-0", 0.8, 0.53)]), tokenizer=None, config=None)

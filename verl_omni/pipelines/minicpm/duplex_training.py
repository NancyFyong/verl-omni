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
"""Actor replay with frozen native input spans and emitted-action probabilities."""

import torch

from verl_omni.utils.dataset.duplex.contracts import prefix_fingerprint

from .duplex_sampling import duplex_log_distribution


def forward_duplex(self, input_ids, attention_mask, position_ids, replays, **kwargs):
    """Replay padded (B,S) prefixes without re-encoding or observing future input."""
    if input_ids.ndim != 2 or len(replays) != input_ids.shape[0]:
        raise ValueError("Duplex replay requires one native window per padded sample.")
    embedding_layer = self.llm.get_input_embeddings()
    if embedding_layer.weight.requires_grad:
        raise ValueError("Duplex cached native spans require frozen token embeddings and media encoders.")
    embeddings = embedding_layer(input_ids)
    embeddings = embeddings * getattr(self.llm.config, "scale_emb", 1.0)
    rows, target_positions = [], []
    for index, replay in enumerate(replays):
        row = embeddings[index]
        if not replay:  # Synthetic zero-loss padding sample.
            rows.append(row)
            target_positions.append(None)
            continue
        action = replay["action"]
        prefix = action["prefix_ids"]
        valid = (
            attention_mask[index].nonzero().flatten()
            if attention_mask is not None
            else torch.arange(len(row), device=row.device)
        )
        if len(valid) != len(prefix) + 1:
            raise ValueError("Duplex actor window is truncated or has extra tokens.")
        offset = int(valid[0])
        expected = torch.tensor([*prefix, action["token_id"]], device=input_ids.device)
        if not torch.equal(input_ids[index, valid], expected):
            raise ValueError("Duplex actor IDs differ from the actual sampled prefix/action.")
        if (
            prefix_fingerprint(replay["identity"], prefix, replay["spans"], action["policy"])
            != action["prefix_fingerprint"]
        ):
            raise ValueError("Duplex actor received a mismatched causal prefix fingerprint.")
        for span in replay["spans"]:
            start, end = span["offset"], span["offset"] + len(span["ids"])
            if end > len(prefix):
                raise ValueError("Duplex actor span includes future input.")
            positions = torch.arange(start + offset, end + offset, device=row.device)
            row = row.index_copy(0, positions, span["embeddings"].to(row))
        rows.append(row)
        target_positions.append(offset + len(prefix) - 1)
    kwargs.pop("inputs_embeds", None)
    kwargs.pop("use_cache", None)
    output = self.llm(
        inputs_embeds=torch.stack(rows),
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        **kwargs,
    )
    logits = output.logits.float()
    for index, position in enumerate(target_positions):
        if position is not None:
            policy_logits = duplex_log_distribution(logits[index, position], **replays[index]["action"]["policy"])
            # Functional replacement keeps the gradient through the native policy transform.
            logits = logits.clone()
            logits[index, position] = policy_logits
    output.logits = logits
    return output

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
"""The emitted-action distribution of MiniCPM's two-pass duplex decoder."""

import torch


def duplex_log_distribution(logits, *, chunk_eos, listen, speak, forbidden, alias_listen=False, forced_token=None):
    """Return fp32 log probabilities, retaining the native chunk-boundary mixture.

    Training disables optional repetition, character caps, temperature and top-k/p
    processing. The native boundary draw and listen-to-speak projection remain.
    Forced structural actions have zero loss, not an invented model probability.
    """
    logits = logits.float()
    vocab = logits.shape[-1]
    if any(not 0 <= token < vocab for token in (chunk_eos, listen, speak)):
        raise ValueError("Duplex control tokens must belong to the policy vocabulary.")
    floor = torch.finfo(logits.dtype).min
    if forced_token is not None:
        if not 0 <= forced_token < vocab:
            raise ValueError("Invalid forced duplex token.")
        result = torch.full_like(logits, floor)
        result[..., forced_token] = 0.0
        return result
    if not torch.isfinite(logits).all():
        raise ValueError("Duplex policy requires finite unprocessed logits.")

    boundary_mask = torch.arange(vocab, device=logits.device) == chunk_eos
    without_boundary = logits.masked_fill(boundary_mask, float("-inf"))
    log_remaining = torch.logsumexp(without_boundary, dim=-1, keepdim=True) - torch.logsumexp(
        logits, dim=-1, keepdim=True
    )
    banned = sorted({chunk_eos, *(int(t) for t in forbidden if 0 <= t < vocab)})
    mask = torch.zeros(vocab, dtype=torch.bool, device=logits.device)
    mask[banned] = True
    if bool(mask.all()):
        raise ValueError("Duplex policy has no legal second-pass action.")
    conditional = logits.masked_fill(mask, float("-inf")).log_softmax(dim=-1)
    result = conditional + log_remaining
    result = torch.where(boundary_mask, logits.log_softmax(dim=-1), result)
    if alias_listen:
        merged = torch.logaddexp(result[..., speak], result[..., listen])
        result = result.clone()
        result[..., speak] = merged
        result[..., listen] = float("-inf")
    # Finite zero-mass logits also keep the generic entropy implementation safe.
    return result.clamp_min(floor)


def validate_duplex_sampling(params):
    """Reject transformations that are not represented in the replay objective."""
    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    for key, value in expected.items():
        if params.get(key, value) != value:
            raise ValueError(f"Duplex OPD requires {key}={value}.")
    if params.get("top_k", -1) not in (-1, 0):
        raise ValueError("Duplex OPD requires top_k=-1 (no truncation).")
    if params.get("logit_bias") or params.get("allowed_token_ids") or params.get("min_tokens", 0):
        raise ValueError("Duplex OPD does not support extra token constraints.")

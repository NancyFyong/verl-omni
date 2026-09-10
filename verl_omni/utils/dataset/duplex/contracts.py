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
"""Bounded, fenced native-input traces shared by duplex training adapters."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field

import torch


def prefix_fingerprint(identity, token_ids, spans, policy):
    """Bind scores to one native causal prefix, including its media provenance."""
    payload = {
        "identity": identity,
        "ids": token_ids,
        "spans": [span["fingerprint"] for span in spans],
        "policy": policy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class DuplexTrace:
    """Store input spans once; each action references only its available prefix."""

    session_id: str
    incarnation: int
    epoch: int
    policy_version: int
    max_context_tokens: int
    max_actions: int
    max_trace_bytes: int = 256 * 1024 * 1024
    spans: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    prefix_ids: list = field(default_factory=list)
    closed: bool = False
    _bytes: int = 0

    @property
    def identity(self):
        """Session/epoch/policy fence carried by every score request."""
        return {
            "session_id": self.session_id,
            "incarnation": self.incarnation,
            "epoch": self.epoch,
            "policy_version": self.policy_version,
            "schema_version": 1,
            "stage": "thinker",
        }

    def check_fence(self, *, incarnation, epoch, policy_version):
        """Refuse stale callbacks and updates underneath a live session."""
        if self.closed or (incarnation, epoch, policy_version) != (self.incarnation, self.epoch, self.policy_version):
            raise ValueError("Stale or closed duplex session/policy fence.")

    def input_span(self, offset, ids, embeddings, metadata):
        """Capture the real runner span, never the scheduler reservation IDs."""
        ids = list(ids)
        if offset < 0 or offset > len(self.prefix_ids) or offset + len(ids) > self.max_context_tokens:
            raise ValueError("Duplex input has an unavailable prefix or exceeds the context budget.")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or not torch.isfinite(embeddings).all():
            raise ValueError("Duplex native input IDs/features are not aligned and finite.")
        snapshot = embeddings.detach().to("cpu").clone()
        size = snapshot.numel() * snapshot.element_size()
        if self._bytes + size > self.max_trace_bytes:
            raise ValueError("Duplex trace exceeds its byte budget.")
        self._bytes += size
        info = copy.deepcopy(metadata)
        fingerprint = hashlib.sha256(json.dumps(info, sort_keys=True, default=str).encode())
        fingerprint.update(snapshot.contiguous().view(torch.uint8).numpy().tobytes())
        self.spans.append(
            {
                "offset": offset,
                "ids": ids,
                "embeddings": snapshot,
                "metadata": info,
                "fingerprint": fingerprint.hexdigest(),
            }
        )
        self.prefix_ids[offset:] = ids

    def action(self, token_id, logprob, *, policy, origin, seq, timestamp_ns):
        """Record a sampled or forced action before any serving projection."""
        if self.closed or not self.prefix_ids or len(self.actions) >= self.max_actions:
            raise ValueError("Duplex action has no prefix, a closed fence, or exceeds its action budget.")
        if not math.isfinite(logprob) or logprob > 1e-6 or origin not in {"sampled", "forced"}:
            raise ValueError("Invalid duplex behavior probability/action origin.")
        if len(self.prefix_ids) >= self.max_context_tokens:
            raise ValueError("Duplex action exceeds the context budget.")
        policy = copy.deepcopy(policy)
        self.actions.append(
            {
                "prefix_ids": list(self.prefix_ids),
                "span_count": len(self.spans),
                "token_id": int(token_id),
                "logprob": float(logprob),
                "policy": policy,
                "origin": origin,
                "seq": seq,
                "timestamp_ns": timestamp_ns,
                "context_mask": True,
                "loss_mask": origin == "sampled",
                "playback_mask": False,
                "prefix_fingerprint": prefix_fingerprint(self.identity, self.prefix_ids, self.spans, policy),
            }
        )
        self.prefix_ids.append(int(token_id))

    def cancel(self, retained_tokens, played_actions=()):
        """Apply an explicit runtime rollback boundary, not an audible-text guess."""
        if not 0 <= retained_tokens <= len(self.prefix_ids):
            raise ValueError("Invalid duplex retained-prefix boundary.")
        for index, action in enumerate(self.actions):
            action["context_mask"] = len(action["prefix_ids"]) < retained_tokens
            action["loss_mask"] = action["loss_mask"] and action["context_mask"]
            action["playback_mask"] = index in played_actions
        self.prefix_ids = self.prefix_ids[:retained_tokens]
        self.closed = True

    def windows(self):
        """Build bounded one-action windows; every learned action occurs once."""
        result = []
        for action in self.actions:
            spans = self.spans[: action["span_count"]]
            if (
                prefix_fingerprint(self.identity, action["prefix_ids"], spans, action["policy"])
                != action["prefix_fingerprint"]
            ):
                raise ValueError("Duplex prefix fingerprint changed after collection.")
            result.append({"identity": self.identity, "action": copy.deepcopy(action), "spans": spans})
        return result

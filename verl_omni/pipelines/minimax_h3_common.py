# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
"""Packed training forward, adapted from diffusers' transformer_minimax_h3.py.

Only sequence-boundary plumbing differs from the upstream transformer. All
parameters keep their original names for PEFT, FSDP, and rollout weight sync.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
from itertools import accumulate

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
    _apply_rotary_emb,
)
from diffusers.utils import apply_lora_scale
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _get_fa3_varlen():
    """Load the same autograd-enabled FA3 kernel version as diffusers."""
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", version=1).flash_attn_varlen_func


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Immutable per-forward boundaries, including indices for the native reference backend."""

    cu_seqlens: torch.Tensor
    max_seqlen: int
    lengths: tuple[int, ...]
    total_tokens: int

    @classmethod
    def from_lengths(cls, lengths: list[int], device: torch.device):
        """Build boundaries for nonempty sequences without allocating a quadratic mask."""
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("Packed H3 sequences must have positive lengths.")
        return cls(
            torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device),
            max(lengths),
            tuple(lengths),
            sum(lengths),
        )

    @cached_property
    def valid_mask(self):
        """Materialize padding only for the native SDPA backend."""
        device = self.cu_seqlens.device
        return torch.arange(self.max_seqlen, device=device)[None] < torch.tensor(self.lengths, device=device)[:, None]

    @cached_property
    def padded_indices(self):
        return self.valid_mask.flatten().nonzero().flatten()

    def attention(self, query, key, value, backend):
        """Attend independently to each document in packed ``(1, total, heads, dim)`` tensors."""
        if backend == "_flash_3_varlen_hub":
            return _get_fa3_varlen()(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                cu_seqlens_q=self.cu_seqlens,
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=self.max_seqlen,
                max_seqlen_k=self.max_seqlen,
                causal=False,
            ).unsqueeze(0)
        if backend == "torch_varlen":
            from torch.nn.attention.varlen import varlen_attn

            return varlen_attn(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            ).unsqueeze(0)
        if backend != "native":
            raise ValueError(f"Unsupported packed H3 attention backend: {backend!r}.")

        batch = self.valid_mask.shape[0]

        def pad(tensor):
            padded = tensor.new_zeros((batch * self.max_seqlen, *tensor.shape[2:]))
            return (
                padded.index_copy(0, self.padded_indices, tensor.squeeze(0))
                .view(batch, self.max_seqlen, *tensor.shape[2:])
                .transpose(1, 2)
            )

        output = F.scaled_dot_product_attention(
            pad(query),
            pad(key),
            pad(value),
            attn_mask=self.valid_mask[:, None, None, :],
            dropout_p=0.0,
        )
        return output.transpose(1, 2).flatten(0, 1).index_select(0, self.padded_indices).unsqueeze(0)


def pack_model_inputs(samples: list[dict]) -> dict:
    """Merge single-sample H3 kwargs, preserving positions and distinct noise levels."""
    if not samples:
        raise ValueError("Cannot pack an empty H3 batch.")
    device = samples[0]["hidden_states"].device
    lengths = [sample["token_tags"].numel() for sample in samples]
    text_lengths = [sample["text_indices"].numel() for sample in samples]
    offsets = [0, *accumulate(lengths[:-1])]
    packed = {
        key: torch.cat([sample[key] for sample in samples], dim=1)
        for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    }
    for key in ("video_indices", "audio_indices", "text_indices"):
        packed[key] = torch.cat([sample[key] + offset for sample, offset in zip(samples, offsets, strict=True)]).to(
            device
        )
    for key in ("position_ids", "token_tags"):
        packed[key] = torch.cat([sample[key] for sample in samples]).to(device)
    timesteps, table_indices = torch.unique(
        torch.cat([sample["timestep"] for sample in samples]), sorted=True, return_inverse=True
    )
    tables = table_indices.split([sample["timestep"].numel() for sample in samples])
    indices = torch.cat([table[sample["timestep_indices"]] for table, sample in zip(tables, samples, strict=True)])
    packed["timestep"], packed["timestep_indices"] = timesteps.to(device), indices.to(device)
    packed["sequence_layout"] = PackedSequenceLayout.from_lengths(lengths, device)
    packed["text_sequence_layout"] = PackedSequenceLayout.from_lengths(text_lengths, device)
    packed["return_dict"] = False
    return packed


class MiniMaxH3PackedAttnProcessor(MiniMaxH3AttnProcessor):
    """Apply upstream QKV, QK norm and RoPE with explicit per-call sample boundaries."""

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if not isinstance(attention_mask, PackedSequenceLayout):
            raise ValueError("Packed H3 attention requires explicit sample boundaries.")
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query, key = _apply_rotary_emb(query, *rotary_emb), _apply_rotary_emb(key, *rotary_emb)
        output = attention_mask.attention(query, key, value, self._attention_backend)
        return attn.to_out[1](attn.to_out[0](output.flatten(2, 3).type_as(query)))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Keep the upstream FSDP selector name while forwarding explicit attention boundaries."""

    def __init__(self, block):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.ff = block.norm1, block.attn, block.norm2, block.ff

    def forward(self, hidden_states, sequence_layout):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=sequence_layout)
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3PackedTokenRefiner(nn.Module):
    """Refine packed text without allowing attention across sample boundaries."""

    def __init__(self, refiner):
        super().__init__()
        self.refiner_blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(b) for b in refiner.refiner_blocks])
        self.final_norm = refiner.final_norm
        self.gradient_checkpointing = False

    def forward(self, hidden_states, sequence_layout):
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, sequence_layout)
            else:
                hidden_states = block(hidden_states, sequence_layout)
        return self.final_norm(hidden_states)


class MiniMaxH3PackedTransformer3DModel(MiniMaxH3Transformer3DModel):
    """Checkpoint-compatible H3 transformer executing a packed micro-batch in one forward."""

    supports_packed_batch = True

    @wraps(MiniMaxH3Transformer3DModel.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_refiner = MiniMaxH3PackedTokenRefiner(self.token_refiner)
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.set_processor(MiniMaxH3PackedAttnProcessor())
        self.set_attention_backend("native")

    def set_attention_backend(self, backend):
        """Select FA3, PyTorch varlen or native batched SDPA; never silently fall back."""
        if backend not in {"native", "torch_varlen", "_flash_3_varlen_hub"}:
            raise ValueError("Packed H3 requires attn_backend=_flash_3_varlen_hub, torch_varlen or native.")
        if backend == "_flash_3_varlen_hub":
            _get_fa3_varlen()
        elif backend == "torch_varlen":
            from torch.nn.attention.varlen import varlen_attn  # noqa: F401
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.processor._attention_backend = backend

    def enable_parallelism(self, *args, **kwargs):
        """Reject unimplemented packed sequence-parallel semantics."""
        raise NotImplementedError("Packed H3 batch forward does not yet support context/sequence parallelism.")

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        sequence_layout,
        text_sequence_layout,
        attention_kwargs=None,
        return_dict=True,
    ):
        """Run projections, the text refiner and DiT once over isolated packed samples."""
        if hidden_states.shape[0] != 1 or audio_hidden_states.shape[0] != 1 or encoder_hidden_states.shape[0] != 1:
            raise ValueError("Packed H3 expects a singleton outer dimension and concatenated sequence rows.")
        sequence_length = position_ids.shape[0]
        if (
            position_ids.shape != (sequence_length, 3)
            or token_tags.shape != (sequence_length,)
            or timestep_indices.shape != (sequence_length,)
        ):
            raise ValueError("Packed H3 positions, token tags and timesteps must describe the same sequence.")

        if sequence_layout.total_tokens != sequence_length:
            raise ValueError("Packed H3 attention boundaries do not match the sequence length.")
        if text_sequence_layout.total_tokens != encoder_hidden_states.shape[1]:
            raise ValueError("Packed H3 text attention boundaries do not match the text length.")
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds, text_sequence_layout)
        hidden_states = text_embeds.new_zeros((1, sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    sequence_layout,
                )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, sequence_layout)
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

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
"""Block-causal attention and incremental KV caching for Diffusers Wan."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttnProcessor, WanTransformer3DModel

__all__ = [
    "WanCausalCache",
    "WanCausalCrossAttentionProcessor",
    "WanCausalSelfAttentionProcessor",
    "allocate_wan_cache",
    "configure_causal_wan",
    "wan_causal_forward",
]


def apply_wan_rotary(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the real-valued rotary transform used by Diffusers Wan attention."""
    first, second = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = first * cos - second * sin
    output[..., 1::2] = first * sin + second * cos
    return output.type_as(hidden_states)


def combine_wan_rope_grid(
    temporal: torch.Tensor,
    vertical: torch.Tensor,
    horizontal: torch.Tensor,
    *,
    start_frame: int,
    end_frame: int,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """Combine separate temporal and spatial frequencies into Wan token order."""
    frame_count = end_frame - start_frame
    temporal = temporal[start_frame:end_frame].view(frame_count, 1, 1, -1)
    vertical = vertical[:patch_height].view(1, patch_height, 1, -1)
    horizontal = horizontal[:patch_width].view(1, 1, patch_width, -1)
    return torch.cat(
        [
            temporal.expand(frame_count, patch_height, patch_width, -1),
            vertical.expand(frame_count, patch_height, patch_width, -1),
            horizontal.expand(frame_count, patch_height, patch_width, -1),
        ],
        dim=-1,
    ).reshape(1, frame_count * patch_height * patch_width, 1, -1)


class WanOffsetRotaryEmbedding(torch.nn.Module):
    """Wan rotary embedding with an explicit latent-frame offset."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        self.start_frame = 0

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build spatial-temporal rotary values for the configured frame range."""
        _, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.base.patch_size
        if patch_t != 1:
            raise ValueError(f"Causal Wan requires temporal patch size 1, got {patch_t}.")
        frame_count = num_frames // patch_t
        patch_height = height // patch_h
        patch_width = width // patch_w
        split_sizes = [self.base.t_dim, self.base.h_dim, self.base.w_dim]
        cos_t, cos_h, cos_w = self.base.freqs_cos.split(split_sizes, dim=1)
        sin_t, sin_h, sin_w = self.base.freqs_sin.split(split_sizes, dim=1)
        end_frame = self.start_frame + frame_count
        if end_frame > cos_t.shape[0]:
            raise ValueError(f"Wan rotary range [{self.start_frame}, {end_frame}) exceeds {cos_t.shape[0]} frames.")

        rope_kwargs = {
            "start_frame": self.start_frame,
            "end_frame": end_frame,
            "patch_height": patch_height,
            "patch_width": patch_width,
        }
        return combine_wan_rope_grid(cos_t, cos_h, cos_w, **rope_kwargs), combine_wan_rope_grid(
            sin_t, sin_h, sin_w, **rope_kwargs
        )


@dataclass
class WanCausalCache:
    """Per-layer self/cross-attention cache committed atomically by latent-frame block."""

    layer_count: int
    batch_size: int
    tokens_per_frame: int
    max_frames: int
    key_values: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = field(init=False)
    cross_key_values: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = field(init=False)
    committed_frames: int = 0

    def __post_init__(self) -> None:
        if min(self.layer_count, self.batch_size, self.tokens_per_frame, self.max_frames) <= 0:
            raise ValueError("Wan cache dimensions must all be positive.")
        self.key_values = [None] * self.layer_count
        self.cross_key_values = [None] * self.layer_count

    def commit(
        self,
        pending: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
        pending_cross: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
        frame_count: int,
    ) -> None:
        """Commit every layer's self/cross K/V together, rejecting partial updates."""
        if len(pending) != self.layer_count or any(value is None for value in pending):
            raise ValueError("Every Wan layer must produce self-attention K/V before a cache block can be committed.")
        if len(pending_cross) != self.layer_count:
            raise ValueError("Every Wan layer must report cross-attention cache state before commit.")
        if frame_count <= 0 or self.committed_frames + frame_count > self.max_frames:
            raise ValueError("Wan cache commit exceeds its configured frame capacity.")
        expected_tokens = frame_count * self.tokens_per_frame
        updated = []
        for layer, current in enumerate(pending):
            assert current is not None
            key, value = current
            if key.shape[0] != self.batch_size or key.shape[1] != expected_tokens or value.shape != key.shape:
                raise ValueError(f"Invalid Wan K/V shape for layer {layer}: {tuple(key.shape)}, {tuple(value.shape)}.")
            previous = self.key_values[layer]
            if previous is not None:
                key = torch.cat((previous[0], key), dim=1)
                value = torch.cat((previous[1], value), dim=1)
            updated.append((key.detach(), value.detach()))
        updated_cross = []
        for layer, pending_value in enumerate(pending_cross):
            previous = self.cross_key_values[layer]
            if previous is None:
                if pending_value is None:
                    raise ValueError(f"Wan layer {layer} did not produce cross-attention K/V before first commit.")
                key, value = pending_value
                if key.shape[0] != self.batch_size or key.shape != value.shape:
                    raise ValueError(
                        f"Invalid Wan cross-attention K/V shape for layer {layer}: "
                        f"{tuple(key.shape)}, {tuple(value.shape)}."
                    )
                updated_cross.append((key.detach(), value.detach()))
            else:
                updated_cross.append(previous)
        self.key_values = updated
        self.cross_key_values = updated_cross
        self.committed_frames += frame_count

    def reset(self) -> None:
        """Discard all batch-local cache contents."""
        self.key_values = [None] * self.layer_count
        self.cross_key_values = [None] * self.layer_count
        self.committed_frames = 0


class WanCausalCrossAttentionProcessor(WanAttnProcessor):
    """Wan cross-attention with batch-local encoder K/V reuse."""

    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.cache: Optional[WanCausalCache] = None
        self.commit_cache = False
        self.pending: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def configure(self, *, cache: Optional[WanCausalCache], commit_cache: bool) -> None:
        """Configure one forward without mutating committed cross-attention state."""
        self.cache = cache
        self.commit_cache = commit_cache
        self.pending = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Project query tokens and attend to cached or newly projected text K/V."""
        if encoder_hidden_states is None or attention_mask is not None or rotary_emb is not None:
            raise ValueError("Wan causal cross-attention requires encoder states without a mask or rotary embedding.")
        if attn.add_k_proj is not None:
            raise NotImplementedError("Causal Wan currently supports T2V text conditioning only.")
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        previous = None if self.cache is None else self.cache.cross_key_values[self.layer_index]
        if previous is None:
            key = attn.norm_k(attn.to_k(encoder_hidden_states)).unflatten(2, (attn.heads, -1))
            value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            if self.cache is not None and self.commit_cache:
                self.pending = (key, value)
        else:
            key, value = previous
        if self._parallel_config is not None:
            raise NotImplementedError("Causal Wan attention does not yet support context parallelism.")
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=None,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


class WanCausalSelfAttentionProcessor(WanAttnProcessor):
    """Wan self-attention using block-prefix visibility or committed K/V state."""

    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.num_frames = 0
        self.frames_per_block = 1
        self.cache: Optional[WanCausalCache] = None
        self.commit_cache = False
        self.pending: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def configure(
        self,
        *,
        num_frames: int,
        frames_per_block: int,
        cache: Optional[WanCausalCache],
        commit_cache: bool,
    ) -> None:
        """Configure one forward without changing learned state."""
        if num_frames <= 0 or frames_per_block <= 0 or num_frames % frames_per_block:
            raise ValueError("Wan frames must be positive and divisible by frames_per_block.")
        self.num_frames = num_frames
        self.frames_per_block = frames_per_block
        self.cache = cache
        self.commit_cache = commit_cache
        self.pending = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply block-causal self-attention, optionally reading a committed prefix."""
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("Wan causal self-attention accepts neither cross inputs nor an external mask.")
        if self.num_frames <= 0 or hidden_states.shape[1] % self.num_frames:
            raise ValueError("Wan causal attention was not configured for the current token geometry.")
        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = apply_wan_rotary(query, *rotary_emb)
            key = apply_wan_rotary(key, *rotary_emb)

        if self.cache is None:
            tokens_per_frame = hidden_states.shape[1] // self.num_frames
            block_tokens = self.frames_per_block * tokens_per_frame
            outputs = []
            for start in range(0, hidden_states.shape[1], block_tokens):
                end = min(start + block_tokens, hidden_states.shape[1])
                outputs.append(self.attend(attn, query[:, start:end], key[:, :end], value[:, :end]))
            hidden_states = torch.cat(outputs, dim=1)
        else:
            previous = self.cache.key_values[self.layer_index]
            if previous is None:
                attended_key, attended_value = key, value
            else:
                attended_key = torch.cat((previous[0], key), dim=1)
                attended_value = torch.cat((previous[1], value), dim=1)
            hidden_states = self.attend(attn, query, attended_key, attended_value)
            if self.commit_cache:
                self.pending = (key, value)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))

    def attend(self, attn, query, key, value):
        """Dispatch through the selected Diffusers attention backend."""
        if self._parallel_config is not None:
            raise NotImplementedError("Causal Wan attention does not yet support context parallelism.")
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )


def unwrap_wan(module: torch.nn.Module) -> WanTransformer3DModel:
    """Return the underlying Wan transformer through an optional FSDP wrapper."""
    return getattr(module, "_fsdp_wrapped_module", module)


def allocate_wan_cache(
    module: torch.nn.Module,
    *,
    batch_size: int,
    latent_height: int,
    latent_width: int,
    max_frames: int,
) -> WanCausalCache:
    """Allocate shape metadata and empty per-layer caches without model-size constants."""
    transformer = unwrap_wan(module)
    patch_t, patch_h, patch_w = transformer.config.patch_size
    if patch_t != 1:
        raise ValueError(f"Causal Wan requires temporal patch size 1, got {patch_t}.")
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError("Wan latent height and width must be divisible by the spatial patch size.")
    return WanCausalCache(
        layer_count=len(transformer.blocks),
        batch_size=batch_size,
        tokens_per_frame=(latent_height // patch_h) * (latent_width // patch_w),
        max_frames=max_frames,
    )


def configure_causal_wan(module: torch.nn.Module) -> torch.nn.Module:
    """Install parameter-free causal attention and offset-aware RoPE adapters."""
    transformer = unwrap_wan(module)
    if not isinstance(transformer, WanTransformer3DModel):
        raise TypeError(f"Causal Wan requires WanTransformer3DModel, got {type(transformer)}.")
    if not isinstance(transformer.rope, WanOffsetRotaryEmbedding):
        transformer.rope = WanOffsetRotaryEmbedding(transformer.rope)
    for index, block in enumerate(transformer.blocks):
        for attention, processor_type in (
            (block.attn1, WanCausalSelfAttentionProcessor),
            (block.attn2, WanCausalCrossAttentionProcessor),
        ):
            if not isinstance(attention.processor, processor_type):
                processor = processor_type(index)
                processor._attention_backend = attention.processor._attention_backend
                processor._parallel_config = attention.processor._parallel_config
                attention.set_processor(processor)
    return module


@contextmanager
def wan_causal_forward(
    module: torch.nn.Module,
    *,
    num_frames: int,
    frames_per_block: int,
    cache: Optional[WanCausalCache] = None,
    commit_cache: bool = False,
):
    """Configure a full forward or exactly one cached block, atomically committing K/V."""
    transformer = unwrap_wan(module)
    configure_causal_wan(transformer)
    if not isinstance(transformer.rope, WanOffsetRotaryEmbedding):
        raise TypeError("Causal Wan RoPE adapter was not installed.")
    if cache is not None:
        if num_frames != frames_per_block:
            raise ValueError("Cached Wan forward requires exactly one temporal block.")
        if cache.layer_count != len(transformer.blocks):
            raise ValueError("Wan cache layer count does not match the transformer.")
        if cache.committed_frames + num_frames > cache.max_frames:
            raise ValueError("Wan incremental forward exceeds cache capacity.")
        transformer.rope.start_frame = cache.committed_frames
    else:
        transformer.rope.start_frame = 0
    self_processors = [block.attn1.processor for block in transformer.blocks]
    cross_processors = [block.attn2.processor for block in transformer.blocks]
    for processor in self_processors:
        if not isinstance(processor, WanCausalSelfAttentionProcessor):
            raise TypeError("Causal Wan self-attention processor was not installed.")
        processor.configure(
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            cache=cache,
            commit_cache=commit_cache,
        )
    for processor in cross_processors:
        if not isinstance(processor, WanCausalCrossAttentionProcessor):
            raise TypeError("Causal Wan cross-attention processor was not installed.")
        processor.configure(cache=cache, commit_cache=commit_cache)
    try:
        yield module
    except Exception:
        for processor in (*self_processors, *cross_processors):
            processor.pending = None
        raise
    else:
        if cache is not None and commit_cache:
            cache.commit(
                [processor.pending for processor in self_processors],
                [processor.pending for processor in cross_processors],
                num_frames,
            )
    finally:
        transformer.rope.start_frame = 0
        for processor in (*self_processors, *cross_processors):
            processor.cache = None
            processor.commit_cache = False
            processor.pending = None

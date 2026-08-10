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
"""Shared dual-stream helpers for the MiniMax H3 pipelines (DiffusionNFT + FlowGRPO).

MiniMax H3 is genuinely dual-stream: the transformer consumes and produces
separate video token rows (width 96) and audio token rows (width 32). The shared
diffusion engine and losses operate on a single latent tensor, so rollout packs
both row streams into one flat vector and the training adapter inverts it, then
splits the transformer's ``(v_video, v_audio)`` output back apart. Keeping the
pack/unpack/split here makes rollout and training layouts consistent by
construction and CPU-testable without diffusers or vllm_omni, and lets the
DiffusionNFT and FlowGRPO adapters share one layout contract.

The transformer forward reads the packed sequence through a set of static
structural tensors (``token_tags``, ``position_ids`` and the three row-index
tensors). :func:`build_packed_sequence` builds them; it is ported verbatim from
the diffusers ``minimax-h3`` branch (``MiniMaxH3PrepareLayoutStep``) so training
lays out byte-identical sequences to rollout without importing diffusers. See
``docs/rfcs/rfc-0001-minimax-h3-fl2va.md``.

:class:`MiniMaxH3RolloutWeightSyncMixin` is shared here for the same reason: both
rollout adapters receive the trainer's diffusers-named weights and have to translate
them into the fused vllm layout identically, or the DiT generates from dummy weights.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

# DiT row widths: patchified 24-channel video latent with patch (1, 2, 2) -> 96; audio -> 32.
VIDEO_ROW_WIDTH = 96
AUDIO_ROW_WIDTH = 32

# Number of leading columns of ``latent_meta`` consumed by the training-side unpack:
# ``[Nv, Na, latent_t, latent_h, latent_w, audio_t]``.
LATENT_META_WIDTH = 6

# Packed-sequence modality tags, per the transformer's ``token_tags`` contract.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

# Rotary-time constants, ported verbatim from the diffusers minimax-h3 layout builder so training and
# rollout lay out identical sequences. One latent frame spans 5/3 * frames_per_latent rotary units; the
# (1, 4, 4, 4, 4) pattern mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping, and the spatial
# axes are normalized by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

__all__ = [
    "VIDEO_ROW_WIDTH",
    "AUDIO_ROW_WIDTH",
    "LATENT_META_WIDTH",
    "VIDEO_TAG",
    "TEXT_TAG",
    "AUDIO_TAG",
    "pack_video_audio_rows",
    "unpack_video_audio_rows",
    "split_dual_velocity",
    "h3_dit_timestep",
    "h3_velocity_to_flow_match",
    "build_packed_sequence",
    "build_layout_from_meta",
    "build_row_timesteps",
    "MiniMaxH3RolloutWeightSyncMixin",
]


def h3_dit_timestep(timesteps: torch.Tensor) -> torch.Tensor:
    """Convert diffusers-style timesteps (``sigma * 1000``) to the DiT's own convention.

    MiniMax H3's DiT consumes ``t`` in ``[0, 1]`` as a *data* fraction, not a noise
    fraction: vllm-omni feeds it ``t = 1 - sigma`` (``denoise_loop.py``, and
    ``scheduling_minimax_h3_euler_ancestral._validate_unit_timestep`` rejects any pair
    where ``sigma_curr != 1 - timestep``). Sigmas descend ``1.0 -> 0.0``, so the noisiest
    step is ``sigma=1`` / ``t=0``. Passing ``sigma`` straight through tells the model it is
    looking at clean data when it is looking at pure noise.

    Args:
        timesteps: Timesteps on the diffusers ``[0, 1000]`` scale.

    Returns:
        The same tensor as DiT timesteps in ``[0, 1]``.
    """
    return 1.0 - timesteps / 1000.0


def h3_velocity_to_flow_match(velocity: torch.Tensor) -> torch.Tensor:
    """Flip a MiniMax H3 velocity into the diffusers flow-match sign convention.

    vllm-omni defines ``x0 = x_t + sigma * v`` (``minimax_h3_rf_v_to_x0``), so H3's
    velocity is ``x0 - noise``. Every consumer in this tree assumes the opposite
    ``noise - x0``: the SDE scheduler recovers ``x0 = sample - sigma * model_output``
    (``schedulers/flow_match_sde.py``) and the DiffusionNFT loss uses
    ``x0_prediction = xt - t * prediction`` (``trainer/diffusion/diffusion_algos.py``).
    Feeding the raw velocity therefore steps *away* from the data.

    Args:
        velocity: A raw DiT velocity row tensor.

    Returns:
        The negated velocity.
    """
    return -velocity


def pack_video_audio_rows(video_rows: torch.Tensor, audio_rows: torch.Tensor) -> torch.Tensor:
    """Flatten and concatenate video + audio DiT rows into one packed vector.

    Args:
        video_rows: video token rows, shape ``(B, Nv, 96)`` or ``(Nv, 96)``.
        audio_rows: audio token rows, shape ``(B, Na, 32)`` or ``(Na, 32)``.

    Returns:
        Packed tensor of shape ``(B, Nv * 96 + Na * 32)``. A leading batch dim of
        1 is added when the inputs are unbatched (the per-request rollout case).
    """
    if video_rows.ndim == 2:
        video_rows = video_rows.unsqueeze(0)
    if audio_rows.ndim == 2:
        audio_rows = audio_rows.unsqueeze(0)
    batch = video_rows.shape[0]
    return torch.cat([video_rows.reshape(batch, -1), audio_rows.reshape(batch, -1)], dim=1)


def unpack_video_audio_rows(
    packed: torch.Tensor,
    num_video_rows: int,
    num_audio_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`pack_video_audio_rows`.

    Args:
        packed: packed tensor of shape ``(B, Nv * 96 + Na * 32)``.
        num_video_rows: ``Nv``, the number of video token rows.
        num_audio_rows: ``Na``, the number of audio token rows.

    Returns:
        A pair ``(video_rows, audio_rows)`` of shapes ``(B, Nv, 96)`` and
        ``(B, Na, 32)``.
    """
    batch = packed.shape[0]
    split = num_video_rows * VIDEO_ROW_WIDTH
    video_rows = packed[:, :split].reshape(batch, num_video_rows, VIDEO_ROW_WIDTH)
    audio_rows = packed[:, split:].reshape(batch, num_audio_rows, AUDIO_ROW_WIDTH)
    return video_rows, audio_rows


def split_dual_velocity(result) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a MiniMax H3 transformer output into ``(v_video, v_audio)`` rows.

    Args:
        result: The transformer forward output. With ``return_dict=False`` it is a
            2-tuple/list ``(v_video, v_audio)``; with ``return_dict=True`` it is a
            ``MiniMaxH3TransformerOutput`` exposing ``sample`` / ``audio_sample``.

    Returns:
        A pair ``(v_video, v_audio)`` of the video and audio velocity tensors.

    Raises:
        TypeError: If *result* is neither a tuple/list nor a sample/audio_sample container.
    """
    if isinstance(result, tuple | list):
        return result[0], result[1]
    if hasattr(result, "sample") and hasattr(result, "audio_sample"):
        return result.sample, result.audio_sample
    raise TypeError(f"Unexpected MiniMax H3 transformer output type: {type(result).__name__}")


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis: ``dim // patch`` coords on ``[0, 32)`` for a square canvas.

    Built with numpy because ``np.linspace(..., endpoint=False)`` is
    ``start + arange(num) * (stop - start) / num`` (not ``torch.linspace``), and the
    float64 grid has to be reproduced exactly.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame, starting at ``origin``. Spacing is ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int = AUDIO_TAG,
    video_tag: int = VIDEO_TAG,
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the ``[text | keyframe conditions | target audio | target video]`` packed layout.

    Ported verbatim from the diffusers ``minimax-h3`` branch ``MiniMaxH3PrepareLayoutStep`` so training
    and rollout produce byte-identical structural tensors. This is the ``t2va`` / ``fl2va`` layout.

    Args:
        text_token_tags: modality tag of every text row, shape ``(num_text_tokens,)``. Text is ``1``
            except the rows of a keyframe's vision block, which are tagged ``0`` (video).
        num_latent_frames: number of target latent frames.
        latent_height: target latent height.
        latent_width: target latent width.
        num_audio_latents: number of target audio latents per channel.
        patch_size: the transformer's ``(t, h, w)`` patch.
        audio_channels: channels the soundtrack is packed channel-major over.
        audio_tag: modality tag an audio row carries.
        video_tag: modality tag a video row carries.
        keyframe_anchors: one entry per keyframe conditioning block, in packed order — ``"first"`` anchors
            it at the first latent frame, ``"last"`` at the last.

    Returns:
        ``position_ids`` ``(seq_len, 3)`` float64, ``token_tags`` ``(seq_len,)``, ``video_indices``,
        ``audio_indices``, ``text_indices``, and the number of leading video and audio rows that are
        conditioning rather than generated.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows
    # continue the time axis from there, so text length shifts the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            # The rotary time the generated frames span, summed by numpy's pairwise summation because that
            # is how the reference computes this anchor.
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s
    # equals 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0


def build_layout_from_meta(
    meta: Sequence[int],
    num_text_tokens: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Derive :func:`build_packed_sequence` arguments from a ``latent_meta`` row and call it.

    The training adapter only carries the per-sample grid dims through ``latent_meta``
    ``[Nv, Na, latent_t, latent_h, latent_w, audio_t]`` and the text length; this rebuilds the full
    static layout the transformer forward reads by name.

    Args:
        meta: one ``latent_meta`` row ``[Nv, Na, latent_t, latent_h, latent_w, audio_t]``.
        num_text_tokens: the sample's true (unpadded) text length; ``t2va`` tags them all ``TEXT_TAG``.
        patch_size: the transformer's ``(t, h, w)`` patch (video defaults to ``(1, 2, 2)``).
        keyframe_anchors: keyframe conditioning blocks (``fl2va``); empty for ``t2va``.

    Returns:
        The 7-tuple returned by :func:`build_packed_sequence`.

    Raises:
        ValueError: if ``audio_t`` is non-positive, or the derived layout row counts disagree with ``meta``.
    """
    num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
    num_latent_frames, latent_height, latent_width = int(meta[2]), int(meta[3]), int(meta[4])
    num_audio_latents = int(meta[5])
    if num_audio_latents <= 0:
        raise ValueError(f"latent_meta audio_t must be positive, got {num_audio_latents}.")
    audio_channels = num_audio_rows // num_audio_latents

    layout = build_packed_sequence(
        text_token_tags=torch.full((num_text_tokens,), TEXT_TAG, dtype=torch.long),
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        keyframe_anchors=keyframe_anchors,
    )
    _, _, video_indices, audio_indices, *_ = layout
    if audio_indices.shape[0] != num_audio_rows:
        raise ValueError(f"Derived {audio_indices.shape[0]} audio rows, latent_meta says {num_audio_rows}.")
    if not keyframe_anchors and video_indices.shape[0] != num_video_rows:
        raise ValueError(f"Derived {video_indices.shape[0]} video rows, latent_meta says {num_video_rows}.")
    return layout


def build_row_timesteps(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign a timestep to every packed row, then reduce to the transformer's ``(timestep, timestep_indices)``.

    Ported verbatim from the diffusers ``minimax-h3`` branch. One forward serves rows sitting at different
    noise levels: generated video and audio rows step down their own schedules (so are at distinct levels),
    conditioning rows stay pinned at their augmentation level, and text rows inherit the video timestep (the
    default fill — they never reach an output head). The transformer only needs the *distinct* timesteps plus
    a per-row index into them, which is what ``torch.unique(..., return_inverse=True)`` yields.

    ``diffusion_nft`` (Option C) noises the whole packed latent at one level, so ``video_timestep`` and
    ``audio_timestep`` coincide and this collapses to one distinct value with all indices ``0``. ``flow_grpo``
    reverses two schedules, so the two differ and this returns two distinct values with per-modality routing.

    Args:
        video_indices: packed row positions of the video rows, shape ``(Nv,)``.
        audio_indices: packed row positions of the audio rows, shape ``(Na,)``.
        num_condition_video_rows: leading video rows that are keyframe conditioning (``0`` for ``t2va``).
        num_condition_audio_rows: leading audio rows that are audio-reference conditioning (``0`` for ``t2va``).
        num_text_tokens: number of text rows (they inherit ``video_timestep``).
        video_timestep: noise level of the generated video rows and text rows.
        audio_timestep: noise level of the generated audio rows.
        condition_video_timestep: noise level of the conditioning video rows (unused when there are none).
        condition_audio_timestep: noise level of the conditioning audio rows (unused when there are none).

    Returns:
        ``timestep`` ``(num_distinct,)`` — the sorted distinct timesteps — and ``timestep_indices``
        ``(seq_len,)`` — the index of every packed row into ``timestep``.
    """
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


# The trainer holds diffusers' MiniMaxH3Transformer3DModel; vllm serves the fused
# MiniMaxH3DiTModel. Their weights are numerically identical, but four structural
# differences need bridging when the base weight sync streams diffusers-named params:
# a set of pure renames, the two GEGLU halves of ff.net.0.proj swapped, separate
# q/k/v projections packed per-head-interleaved into one qkv_proj, and the fixed
# rope.inv_freq buffer -- which the vllm DiT carries but diffusers computes on the
# fly, so it is absent from the stream and must be synthesized. Verified maxdiff=0
# against both on-disk checkpoints.
_TOPLEVEL_RENAMES = (
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("context_embedder", "condition_proj"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
)


def _diffusers_to_vllm_name(name: str) -> str:
    """Rename a diffusers transformer param to its fused-vllm counterpart (no reshape)."""
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    name = name.replace("transformer_blocks.", "blocks.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    for old, new in _TOPLEVEL_RENAMES:  # audio_* listed first so they win over proj_in/out
        if name.startswith(old + "."):
            return new + name[len(old) :]
    return name


class MiniMaxH3RolloutWeightSyncMixin:
    """Bridge the trainer's diffusers weight names and prompt token ids to the vllm H3 pipeline.

    Mix in ahead of ``MiniMaxH3Pipeline`` so ``load_weights`` intercepts the base weight
    sync. Carries no constructor state -- the q/k/v partial buffer is created on first use
    -- so the rollout adapters need no cooperative ``__init__``.
    """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Translate diffusers-named base weights into fused-vllm names, then load.

        The base weight sync streams the trainer's diffusers checkpoint prefixed with
        ``transformer.``, in buckets that may split one block's q/k/v across calls.
        Rename, GEGLU-swap ``ff.net.0.proj``, and per-head-interleave q/k/v into
        ``qkv_proj`` before delegating to the parent's exact-name loader. LoRA deltas
        (``lora_`` names) arrive through a separate ``add_lora`` path.
        """
        arch = self.transformer.arch
        heads, head_dim, ff_half = arch.num_attention_heads, arch.attention_head_dim, arch.ffn_hidden_size
        partials = getattr(self, "_qkv_buffer", None)
        if partials is None:
            partials = self._qkv_buffer = {}
        translated: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if not name.startswith("transformer."):
                translated.append((name, tensor))
                continue
            inner = name[len("transformer.") :].replace(".base_layer", "")
            if "lora_" in inner:
                continue
            if inner.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
                block, comp = inner.rsplit(".attn.to_", 1)
                slot = partials.setdefault(block, {})
                slot[comp[0]] = tensor
                if len(slot) == 3:
                    heads_qkv = [slot[c].view(heads, head_dim, -1) for c in ("q", "k", "v")]
                    qkv = torch.stack(heads_qkv, dim=1).reshape(heads * 3 * head_dim, -1)
                    translated.append((f"transformer.{_diffusers_to_vllm_name(block)}.attn.qkv_proj.weight", qkv))
                    del partials[block]
                continue
            if inner.endswith(".ff.net.0.proj.weight"):
                swapped = torch.cat([tensor[ff_half:], tensor[:ff_half]], dim=0)
                vname = _diffusers_to_vllm_name(inner).replace(".ff.net.0.proj.", ".mlp.fc1.")
                translated.append((f"transformer.{vname}", swapped))
                continue
            translated.append((f"transformer.{_diffusers_to_vllm_name(inner)}", tensor))
        # Synthesize the fixed 3D-RoPE frequency table. diffusers computes it on the fly so
        # it never reaches this stream, and the vllm buffer is registered uninitialized, so
        # skipping it silently scrambles RoPE. Matches the on-disk vllm checkpoint's
        # 10000**-(arange(0, 2L, 2) / 2L) to 3.7e-09.
        rope_len = arch.rope_inv_freq_len
        inv_freq = 10000.0 ** (-(torch.arange(0, 2 * rope_len, 2, dtype=torch.float32) / (2 * rope_len)))
        translated.append(("transformer.rope.inv_freq", inv_freq))
        return super().load_weights(translated)

    def _ensure_prompt_text(self, request: Any) -> None:
        """Fill ``prompts[0]["prompt"]`` from the request's token ids.

        The agent loop renders the caption via the custom chat template, tokenizes
        it, and the server forwards only ``prompt_token_ids`` (no decoded text) so
        pipelines never re-encode. But the H3 pipeline's text encoder consumes the
        caption *string* (``encode_prompt`` re-tokenizes with the same
        ``self.tokenizer``), so decode the ids back here. H3 is CFG-distilled, so
        there is no negative branch to decode.
        """
        prompts = getattr(request, "prompts", None)
        if not prompts or not isinstance(prompts[0], dict):
            return
        custom_prompt = prompts[0]
        if custom_prompt.get("prompt"):
            return
        token_ids = custom_prompt.get("prompt_token_ids")
        if token_ids is None:
            return
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        text = self.tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=True).strip()
        if text:
            custom_prompt["prompt"] = text

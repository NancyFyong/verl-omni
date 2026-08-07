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
"""CPU tests for the MiniMax H3 DiffusionNFT training adapter.

MiniMax H3 is dual-stream but DiffusionNFT's engine and loss are single-tensor,
so the rollout packs video rows (width 96) and audio rows (width 32) into one
flat vector and the adapter unpacks it, runs the transformer per micro-batch
sample on its real packed-sequence interface, and re-packs the
``(v_video, v_audio)`` velocity. These tests pin the pack/unpack round trip, the
static layout the forward derives from ``latent_meta``, and the per-sample
forward loop with a mocked transformer -- no diffusers weights, no vllm_omni.
"""

from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_ROW_WIDTH,
    VIDEO_TAG,
    build_layout_from_meta,
    pack_video_audio_rows,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig

# latent_meta = [Nv, Na, latent_t, latent_h, latent_w, audio_t], internally consistent with the layout
# builder: Nv = latent_t * (latent_h // 2) * (latent_w // 2) = 1*2*2 = 4; Na = audio_t * audio_ch = 3*2 = 6.
_META = [4, 6, 1, 4, 4, 3]
_NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS = 4, 6
_BATCH = 2
_TEXT_LEN = 12
_TEXT_DIM = 64


def _rows(batch=_BATCH, num_video_rows=_NUM_VIDEO_ROWS, num_audio_rows=_NUM_AUDIO_ROWS):
    video_rows = torch.randn(batch, num_video_rows, VIDEO_ROW_WIDTH)
    audio_rows = torch.randn(batch, num_audio_rows, AUDIO_ROW_WIDTH)
    return video_rows, audio_rows


def _micro_batch(batch=_BATCH):
    return TensorDict({"latent_meta": torch.tensor([_META] * batch, dtype=torch.long)}, batch_size=batch)


def _module(side_effect):
    """A mock transformer with the given per-call behavior and no real ``config`` (patch falls back)."""
    module = MagicMock(side_effect=side_effect)
    module.config = None
    return module


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def _prepared_inputs(video_rows, audio_rows, timesteps, mask=None):
    packed = pack_video_audio_rows(video_rows, audio_rows)
    batch = video_rows.shape[0]
    if mask is None:
        mask = torch.ones(batch, _TEXT_LEN, dtype=torch.int32)
    return MiniMaxH3DiffusionNFT.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=packed,
        timesteps=timesteps,
        prompt_embeds=torch.randn(batch, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=mask,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=_micro_batch(batch),
        step=0,
    )


class TestMiniMaxH3DiffusionNFTRegistry:
    def test_registered_for_minimax_h3_diffusion_nft(self):
        resolved = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", "diffusion_nft")
        assert resolved is MiniMaxH3DiffusionNFT


class TestMiniMaxH3PackUnpack:
    def test_batched_round_trip_inverts_exactly(self):
        video_rows, audio_rows = _rows()
        packed = pack_video_audio_rows(video_rows, audio_rows)
        expected_width = _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH
        assert packed.shape == (_BATCH, expected_width)

        out_video, out_audio = unpack_video_audio_rows(packed, _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        assert torch.equal(out_video, video_rows)
        assert torch.equal(out_audio, audio_rows)

    def test_unbatched_inputs_gain_leading_batch_dim(self):
        video_rows = torch.randn(_NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH)
        audio_rows = torch.randn(_NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH)
        packed = pack_video_audio_rows(video_rows, audio_rows)
        assert packed.shape == (1, _NUM_VIDEO_ROWS * VIDEO_ROW_WIDTH + _NUM_AUDIO_ROWS * AUDIO_ROW_WIDTH)

        out_video, out_audio = unpack_video_audio_rows(packed, _NUM_VIDEO_ROWS, _NUM_AUDIO_ROWS)
        assert torch.equal(out_video, video_rows.unsqueeze(0))
        assert torch.equal(out_audio, audio_rows.unsqueeze(0))


class TestMiniMaxH3BuildLayoutFromMeta:
    def test_row_counts_and_tags_match_meta(self):
        position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = (
            build_layout_from_meta(_META, _TEXT_LEN)
        )
        seq_len = _TEXT_LEN + _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS
        assert position_ids.shape == (seq_len, 3)
        assert video_indices.shape[0] == _NUM_VIDEO_ROWS
        assert audio_indices.shape[0] == _NUM_AUDIO_ROWS
        assert text_indices.shape[0] == _TEXT_LEN
        assert (num_cond_video, num_cond_audio) == (0, 0)
        # t2va tags: text rows 1, audio rows 2, video rows 0.
        assert torch.equal(token_tags[text_indices], torch.full((_TEXT_LEN,), TEXT_TAG))
        assert torch.equal(token_tags[audio_indices], torch.full((_NUM_AUDIO_ROWS,), AUDIO_TAG))
        assert torch.equal(token_tags[video_indices], torch.full((_NUM_VIDEO_ROWS,), VIDEO_TAG))

    def test_inconsistent_meta_is_rejected(self):
        # audio_t=4 does not divide Na=6 evenly -> derived rows disagree with meta.
        with pytest.raises(ValueError, match="audio rows"):
            build_layout_from_meta([4, 6, 1, 4, 4, 4], _TEXT_LEN)


class TestMiniMaxH3PrepareModelInputs:
    def test_unpacks_latents_into_video_and_audio_rows(self):
        video_rows, audio_rows = _rows()
        model_inputs, negative_model_inputs = _prepared_inputs(
            video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0])
        )

        assert negative_model_inputs is None
        assert torch.equal(model_inputs["video_rows"], video_rows)
        assert torch.equal(model_inputs["audio_rows"], audio_rows)
        assert model_inputs["latent_meta"] == _META

    def test_timestep_is_scaled_to_unit_interval(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))
        torch.testing.assert_close(model_inputs["timestep"], torch.tensor([0.5, 0.25]))


class TestMiniMaxH3Forward:
    def test_identity_transformer_repacks_to_input_layout(self):
        video_rows, audio_rows = _rows()
        packed = pack_video_audio_rows(video_rows, audio_rows)
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        # Echoing the per-sample rows back and re-packing must reproduce the flat ``xt`` exactly.
        out = MiniMaxH3DiffusionNFT.forward(
            module=_module(_identity), model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert out.shape == packed.shape
        torch.testing.assert_close(out, packed)

    def test_per_sample_loop_stacks_scaled_velocity_video_then_audio(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        module = _module(lambda **kw: (kw["hidden_states"] * 2.0, kw["audio_hidden_states"] * 3.0))
        out = MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert module.call_count == _BATCH
        torch.testing.assert_close(out, pack_video_audio_rows(video_rows * 2.0, audio_rows * 3.0))

    def test_forward_calls_module_with_real_packed_sequence_kwargs(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))

        module = _module(_identity)
        MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        kwargs = module.call_args_list[0].kwargs
        seq_len = _TEXT_LEN + _NUM_VIDEO_ROWS + _NUM_AUDIO_ROWS
        assert kwargs["hidden_states"].shape == (1, _NUM_VIDEO_ROWS, VIDEO_ROW_WIDTH)
        assert kwargs["audio_hidden_states"].shape == (1, _NUM_AUDIO_ROWS, AUDIO_ROW_WIDTH)
        assert kwargs["encoder_hidden_states"].shape == (1, _TEXT_LEN, _TEXT_DIM)
        assert kwargs["timestep"].shape == (1,)
        assert kwargs["token_tags"].shape == (seq_len,)
        assert kwargs["position_ids"].shape == (seq_len, 3)
        assert kwargs["return_dict"] is False
        # Option C: the engine noised the whole packed latent at one level, so every row shares it.
        assert kwargs["timestep_indices"].shape == (seq_len,)
        assert torch.equal(kwargs["timestep_indices"], torch.zeros(seq_len, dtype=torch.long))

    def test_forward_slices_encoder_to_true_text_length(self):
        video_rows, audio_rows = _rows()
        mask = torch.zeros(_BATCH, _TEXT_LEN, dtype=torch.int32)
        mask[0, :5] = 1  # first prompt is 5 tokens, second is full length
        mask[1, :] = 1
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]), mask=mask)

        module = _module(_identity)
        MiniMaxH3DiffusionNFT.forward(
            module=module, model_config=MagicMock(), model_inputs=model_inputs, negative_model_inputs=None
        )
        assert module.call_args_list[0].kwargs["encoder_hidden_states"].shape == (1, 5, _TEXT_DIM)
        assert module.call_args_list[1].kwargs["encoder_hidden_states"].shape == (1, _TEXT_LEN, _TEXT_DIM)

    def test_non_tuple_output_raises(self):
        video_rows, audio_rows = _rows()
        model_inputs, _ = _prepared_inputs(video_rows, audio_rows, timesteps=torch.tensor([500.0, 250.0]))
        with pytest.raises(TypeError, match="Unexpected MiniMax H3 transformer output"):
            MiniMaxH3DiffusionNFT.forward(
                module=_module(lambda **kw: torch.randn(1, 8)),
                model_config=MagicMock(),
                model_inputs=model_inputs,
                negative_model_inputs=None,
            )


class TestMiniMaxH3ForwardAndSamplePreviousStep:
    def test_reverse_sampling_is_not_implemented(self):
        with pytest.raises(NotImplementedError, match="forward-process objective"):
            MiniMaxH3DiffusionNFT.forward_and_sample_previous_step(
                module=MagicMock(),
                scheduler=MagicMock(),
                model_config=MagicMock(),
                model_inputs={},
                negative_model_inputs=None,
                scheduler_inputs=None,
                step=0,
            )


class TestMiniMaxH3BuildScheduler:
    def test_build_scheduler_sets_video_timesteps(self):
        cfg = object.__new__(DiffusionModelConfig)
        object.__setattr__(cfg, "architecture", "MiniMaxH3Pipeline")
        object.__setattr__(cfg, "algorithm", "diffusion_nft")
        object.__setattr__(cfg, "pipeline", DiffusionPipelineConfig(num_inference_steps=4, video_flow_shift=12.0))

        scheduler = MiniMaxH3DiffusionNFT.build_scheduler(cfg)
        assert len(scheduler.timesteps) == 4

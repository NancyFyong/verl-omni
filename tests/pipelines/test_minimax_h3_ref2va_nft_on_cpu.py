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
"""CPU contracts for MiniMax H3 image-reference Ref2VA DiffusionNFT."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    TEXT_TAG,
    VIDEO_ROW_WIDTH,
    MiniMaxH3RolloutWeightSyncMixin,
    build_ref2va_layout_from_meta,
    pack_video_audio_rows,
    serialize_ref_blocks,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT

vllm_packed = pytest.importorskip("vllm_omni.diffusion.models.minimax_h3.packed_sequence")

_META = [4, 6, 1, 4, 4, 3]
_TEXT_LEN = 7
_TEXT_DIM = 16
_REF_BLOCKS = [{"kind": "image", "latent_h": 4, "latent_w": 4}]


def _identity(**kwargs):
    return kwargs["hidden_states"], kwargs["audio_hidden_states"]


def test_ref_block_metadata_rebuilds_the_upstream_layout():
    text_tags = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    metadata = serialize_ref_blocks(_REF_BLOCKS)

    actual = build_ref2va_layout_from_meta(
        _META,
        _TEXT_LEN,
        metadata,
        ref_block_count=1,
        text_token_tags=text_tags,
    )
    upstream = vllm_packed.minimax_h3_packed_sequence_ref2va_blocks(
        text_len=_TEXT_LEN,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        ref_blocks=_REF_BLOCKS,
    )
    expected_tags = upstream["token_tags"].clone()
    expected_tags[upstream["text_pos"]] = text_tags
    used = int(upstream["cu_seqlens"][1])

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = actual
    torch.testing.assert_close(position_ids, upstream["img_position_ids"][:used], rtol=0, atol=0)
    assert torch.equal(token_tags, expected_tags[:used])
    assert torch.equal(video_indices, upstream["img_pos"])
    assert torch.equal(audio_indices, upstream["audio_pos"])
    assert torch.equal(text_indices, upstream["text_pos"])
    assert (num_cond_video, num_cond_audio) == (4, 0)


def test_ref2va_actor_replays_fixed_image_rows_and_returns_targets_only():
    video_rows = torch.randn(1, 4, VIDEO_ROW_WIDTH)
    audio_rows = torch.randn(1, 6, AUDIO_ROW_WIDTH)
    condition_video_rows = torch.randn(1, 4, VIDEO_ROW_WIDTH)
    micro_batch = TensorDict(
        {
            "latent_meta": torch.tensor([_META], dtype=torch.long),
            "condition_video_rows": condition_video_rows,
            "condition_audio_rows": torch.empty(1, 0, AUDIO_ROW_WIDTH),
            "ref_block_meta": serialize_ref_blocks(_REF_BLOCKS).unsqueeze(0),
            "ref_block_count": torch.tensor([[1]], dtype=torch.long),
            "prompt_token_tags": torch.full((1, _TEXT_LEN), TEXT_TAG, dtype=torch.long),
        },
        batch_size=1,
    )
    model_inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
        module=MagicMock(),
        model_config=MagicMock(),
        latents=pack_video_audio_rows(video_rows, audio_rows),
        timesteps=torch.tensor([500.0]),
        prompt_embeds=torch.randn(1, _TEXT_LEN, _TEXT_DIM),
        prompt_embeds_mask=torch.ones(1, _TEXT_LEN, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )
    module = MagicMock(side_effect=_identity)
    module.config = None

    output = MiniMaxH3DiffusionNFT.forward(module, MagicMock(), model_inputs)

    call = module.call_args.kwargs
    assert call["hidden_states"].shape == (1, 8, VIDEO_ROW_WIDTH)
    assert call["audio_hidden_states"].shape == (1, 6, AUDIO_ROW_WIDTH)
    torch.testing.assert_close(call["hidden_states"][0, :4], condition_video_rows[0])
    assert call["timestep"].tolist() == pytest.approx([0.5, 0.999])
    torch.testing.assert_close(output, -pack_video_audio_rows(video_rows, audio_rows))


def test_ref2va_token_ids_append_after_the_official_image_prefix(monkeypatch):
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module

    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", lambda value, **kwargs: value)

    class Tokenizer:
        special = {"<|vision_start|>": 11, "<|image_pad|>": 12, "<|vision_end|>": 13}

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [] if not text else [20 + index for index, _ in enumerate(text)]}

        def convert_tokens_to_ids(self, token):
            return self.special[token]

    class ImageProcessor:
        merge_size = 1

        def __call__(self, images, return_tensors):
            del images, return_tensors
            return {"pixel_values": torch.ones(1, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    class Stub(MiniMaxH3RolloutWeightSyncMixin):
        pass

    stub = Stub()
    stub._h3_prompt_ids = torch.tensor([101, 102])
    stub.tokenizer = Tokenizer()
    stub.processor = SimpleNamespace(image_processor=ImageProcessor())
    stub.text_encoder_tp_size = 1
    stub.device = torch.device("cpu")
    stub._distribute_encode_inputs = lambda ids, vision_kwargs: ids
    stub._encode_text_hidden = lambda ids, vision_kwargs: ids[:, None].float()

    hidden, tags = stub.encode_prompt(
        task="ref2va",
        prompt="[pretokenized]",
        images=[object()],
        prepared_videos=None,
        condition_labels=[("image", 1)],
    )

    assert hidden[-2:, 0].tolist() == [101.0, 102.0]
    assert tags[-2:].tolist() == [TEXT_TAG, TEXT_TAG]
    assert (tags[:-2] == 0).any()


def test_ref2va_rollout_publishes_reference_replay_fields(monkeypatch):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline

    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._nft_capture = {
        "video_latent": torch.randn(1, 24, 1, 4, 4),
        "audio_latent": torch.randn(2, 32, 3),
        "condition_video_rows": torch.randn(4, VIDEO_ROW_WIDTH),
        "condition_audio_rows": torch.empty(0, AUDIO_ROW_WIDTH),
        "keyframe_frame_indices": [],
        "ref_block_meta": serialize_ref_blocks(_REF_BLOCKS),
        "task": "ref2va",
        "text_embeddings": torch.randn(_TEXT_LEN, _TEXT_DIM),
        "text_tags": torch.full((_TEXT_LEN,), TEXT_TAG, dtype=torch.long),
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 3,
        "num_steps": 3,
        "video_shift": 12.0,
        "base_schedule": None,
    }
    monkeypatch.setattr(
        MiniMaxH3Pipeline,
        "forward",
        lambda self, request: DiffusionOutput(output=(torch.zeros(1), torch.zeros(1))),
    )
    request = MagicMock(
        prompts=[{"prompt_token_ids": [1, 2]}],
        sampling_params=SimpleNamespace(
            num_outputs_per_prompt=1,
            extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True},
        ),
    )

    output = pipeline.forward(request)

    rl = output.output["metadata"]["rl"]
    assert rl["condition_video_rows"].shape == (1, 4, VIDEO_ROW_WIDTH)
    assert rl["condition_audio_rows"].shape == (1, 0, AUDIO_ROW_WIDTH)
    assert rl["ref_block_meta"].shape == (1, 1, 5)
    assert rl["ref_block_count"].tolist() == [[1]]
    assert rl["latents_clean"].shape == (1, 4 * VIDEO_ROW_WIDTH + 6 * AUDIO_ROW_WIDTH)


def test_ref2va_rollout_captures_the_official_image_anchor(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MINIMAX_H3_IMGVID_COND_TIMESTEP
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline

    video_latent = torch.randn(1, 24, 1, 4, 4)
    audio_latent = torch.randn(1, 2, 3, 16)
    monkeypatch.setattr(MiniMaxH3Pipeline, "diffuse", lambda self, **kwargs: (video_latent, audio_latent))

    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.default_video_shift = 12.0
    clean_condition = torch.randn(4, VIDEO_ROW_WIDTH)
    kwargs = {
        "task": "ref2va",
        "text_embeddings": torch.randn(_TEXT_LEN, _TEXT_DIM),
        "text_tags": torch.full((_TEXT_LEN,), TEXT_TAG, dtype=torch.long),
        "seed": 7,
        "latent_t": 1,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 3,
        "num_steps": 3,
        "video_shift": 12.0,
        "base_schedule": None,
        "visual_condition": clean_condition,
        "visual_condition_shape": (1, 4, 4),
        "visual_condition_shapes": [(1, 4, 4)],
        "audio_condition": None,
        "ref_blocks": _REF_BLOCKS,
    }

    pipeline.diffuse(**kwargs)

    expected_condition = minimax_h3_imgvid_cond_noise_aug_rows(
        clean_condition,
        condition_shapes=[(1, 4, 4)],
        target_latent_t=1,
        imgvid_cond_num_frames=1,
        seed=7,
        noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    capture = pipeline._nft_capture
    torch.testing.assert_close(capture["condition_video_rows"], expected_condition)
    assert capture["condition_audio_rows"].shape == (0, AUDIO_ROW_WIDTH)
    assert torch.equal(capture["ref_block_meta"], serialize_ref_blocks(_REF_BLOCKS))


@pytest.mark.parametrize(
    "ref_blocks",
    [
        [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "image", "latent_h": 4, "latent_w": 4}],
        [{"kind": "video", "ref_audio_t": 0, "latent_t": 1, "latent_h": 4, "latent_w": 4}],
        [{"kind": "audio", "ref_audio_t": 3}],
    ],
)
def test_ref2va_rollout_rejects_reference_modes_not_in_the_first_milestone(monkeypatch, ref_blocks):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline

    parent = MagicMock(return_value=(torch.empty(0), torch.empty(0)))
    monkeypatch.setattr(MiniMaxH3Pipeline, "diffuse", parent)
    pipeline = object.__new__(MiniMaxH3DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)

    with pytest.raises(NotImplementedError, match="one image reference"):
        pipeline.diffuse(task="ref2va", ref_blocks=ref_blocks, audio_condition=None)
    parent.assert_not_called()

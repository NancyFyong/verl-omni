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
"""Numerical contracts for a real tiny H3 transformer, without checkpoints or GPUs."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from peft import LoraConfig, get_peft_model
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import pack_video_audio_rows, serialize_ref_blocks
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import (
    MiniMaxH3DiffusionNFT,
    PackedSequenceLayout,
    enable_packed_forward,
    pack_model_inputs,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionNFTLoss
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig

_MODEL_KWARGS = dict(
    hidden_size=32,
    num_attention_heads=2,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    text_dim=16,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
)


def _models(lora=False, checkpointing=False, install_packed=True):
    torch.manual_seed(13)
    serial = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    packed = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    serial.set_attention_backend("native")
    packed.load_state_dict(serial.state_dict(), strict=True)
    if lora:
        config = LoraConfig(
            r=2, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
        )
        serial, packed = get_peft_model(serial, config), get_peft_model(packed, config)
        serial.add_adapter("old", config)
        packed.add_adapter("old", config)
        with torch.no_grad():
            for name, parameter in serial.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.02)
        packed.load_state_dict(serial.state_dict(), strict=True)
    if checkpointing:
        serial.enable_gradient_checkpointing()
        packed.enable_gradient_checkpointing()
    if install_packed:
        enable_packed_forward(packed)
    return serial, packed


def _inputs(model, lengths=(3, 5, 4), task="t2va"):
    torch.manual_seed(19)
    batch = len(lengths)
    meta = torch.tensor([4, 6, 1, 4, 4, 3]).repeat(batch, 1)
    latents = pack_video_audio_rows(torch.randn(batch, 4, 96), torch.randn(batch, 6, 32))
    micro_batch = TensorDict({"latent_meta": meta}, batch_size=[batch])
    if task == "fl2va":
        micro_batch["condition_video_rows"] = torch.randn(batch, 4, 96)
        micro_batch["condition_video_row_count"] = torch.full((batch, 1), 4)
        micro_batch["keyframe_frame_indices"] = torch.zeros(batch, 1, dtype=torch.long)
    elif task == "ref2va":
        blocks = [
            [{"kind": "image", "latent_h": 4, "latent_w": 4}] * (index + 1)
            + ([{"kind": "audio", "ref_audio_t": 2}] if index == 1 else [])
            for index in range(batch)
        ]
        metadata = [serialize_ref_blocks(refs) for refs in blocks]
        micro_batch["ref_block_meta"] = torch.stack([meta for meta, _ in metadata])
        micro_batch["ref_block_count"] = torch.tensor([[count] for _, count in metadata])
        micro_batch["condition_video_rows"] = torch.randn(batch, batch * 4, 96)
        micro_batch["condition_video_row_count"] = torch.arange(1, batch + 1)[:, None] * 4
        micro_batch["condition_audio_rows"] = torch.randn(batch, 4, 32)
        micro_batch["condition_audio_row_count"] = torch.tensor([[4 if i == 1 else 0] for i in range(batch)])
    inputs, _ = MiniMaxH3DiffusionNFT.prepare_model_inputs(
        model,
        SimpleNamespace(),
        latents,
        torch.tensor([200.0, 700.0, 500.0])[:batch],
        torch.randn(batch, max(lengths), 16),
        torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None],
        None,
        None,
        micro_batch,
        0,
    )
    return inputs


def _forward(model, inputs, packed):
    if packed:
        return MiniMaxH3DiffusionNFT.forward(model, SimpleNamespace(), inputs)
    # The former per-sample implementation is retained only as a numerical baseline.
    outputs = []
    for sample, video_count, audio_count in MiniMaxH3DiffusionNFT._iter_sample_inputs(model, inputs):
        video, audio = model(**sample)
        outputs.append(pack_video_audio_rows(-video[:, video_count:], -audio[:, audio_count:]))
    return torch.cat(outputs)


def _loss(prediction, old_prediction, ref_prediction):
    torch.manual_seed(23)
    return DiffusionNFTLoss.compute_loss(
        forward_prediction=prediction,
        old_prediction=old_prediction,
        ref_forward_prediction=ref_prediction,
        x0=torch.randn_like(prediction),
        xt=torch.randn_like(prediction),
        t_expanded=torch.tensor([0.2, 0.7, 0.5], device=prediction.device)[: len(prediction)].reshape(
            -1, *([1] * (prediction.ndim - 1))
        ),
        reward_prob=torch.tensor([0.1, 0.9, 0.6])[: len(prediction)],
        config=SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="diffusion_nft", ref_kl_coef=0.1)),
    )[0]


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_packed_nft_matches_serial_outputs_loss_and_lora_gradients(task, checkpointing):
    serial, packed = _models(lora=True, checkpointing=checkpointing)
    inputs = _inputs(serial, task=task)
    old, ref = [], []
    for model, enabled in ((serial, False), (packed, True)):
        with torch.no_grad():
            model.set_adapter("old")
            old.append(_forward(model, inputs, enabled))
            with model.disable_adapter():
                ref.append(_forward(model, inputs, enabled))
        model.set_adapter("default")
    predictions = [_forward(serial, inputs, False), _forward(packed, inputs, True)]
    for pair in (old, ref, predictions):
        torch.testing.assert_close(pair[0], pair[1], atol=2e-6, rtol=2e-5)
    losses = [
        _loss(prediction, previous, reference)
        for prediction, previous, reference in zip(predictions, old, ref, strict=True)
    ]
    torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=2e-5)
    for loss in losses:
        loss.backward()
    grads = [{name: p.grad for name, p in model.named_parameters() if p.requires_grad} for model in (serial, packed)]
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        assert grads[0][name] is not None, name
        torch.testing.assert_close(grads[0][name], grads[1][name], atol=3e-6, rtol=3e-4, msg=name)


def test_one_forward_per_micro_batch_and_no_cross_sample_attention():
    serial, packed = _models()
    inputs = _inputs(serial)
    calls = []
    hook = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    expected = _forward(packed, inputs, True)
    assert len(calls) == 1
    changed = copy.deepcopy(inputs)
    for key in ("video_rows", "audio_rows", "encoder_hidden_states"):
        changed[key][1] += 9
    changed["timestep"][1] = 0.95
    actual = _forward(packed, changed, True)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]])
    assert not torch.allclose(actual[1], expected[1])
    hook.remove()


def test_fa3_receives_separate_dit_and_text_boundaries(monkeypatch):
    from verl_omni.pipelines.minimax_h3_diffusion_nft import diffusers_training_adapter as packed_forward

    calls = []

    def fa3(query, key, value, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        assert cu_seqlens_q is cu_seqlens_k
        assert max_seqlen_q == max_seqlen_k and not causal
        boundaries = cu_seqlens_q.tolist()
        calls.append(boundaries)
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)]
        assert max_seqlen_q == max(lengths)
        layout = PackedSequenceLayout.from_lengths(lengths, query.device)
        return layout.attention(query[None], key[None], value[None], "native").squeeze(0)

    monkeypatch.setattr(packed_forward, "_get_fa3_varlen", lambda: fa3)
    serial, packed = _models(checkpointing=True)
    packed.set_attention_backend("_flash_3_varlen_hub")
    inputs = _inputs(serial)
    actual = _forward(packed, inputs, True)
    expected = _forward(serial, inputs, False)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert set(map(tuple, calls)) == {(0, 3, 8, 12), (0, 13, 28, 42)}
    for (name, parameter), (other_name, reference) in zip(
        packed.named_parameters(), serial.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_unavailable_fa3_fails_instead_of_falling_back(monkeypatch):
    from verl_omni.pipelines.minimax_h3_diffusion_nft import diffusers_training_adapter as packed_forward

    def unavailable():
        raise RuntimeError("FA3 kernel unavailable")

    monkeypatch.setattr(packed_forward, "_get_fa3_varlen", unavailable)
    _, packed = _models()
    with pytest.raises(RuntimeError, match="FA3 kernel unavailable"):
        packed.set_attention_backend("_flash_3_varlen_hub")
    assert packed.transformer_blocks[0].attn.processor._attention_backend == "native"


def test_varlen_layout_does_not_materialize_native_padding():
    layout = PackedSequenceLayout.from_lengths([2, 4], torch.device("cpu"))
    assert layout.total_tokens == 6
    assert "valid_mask" not in vars(layout) and "padded_indices" not in vars(layout)
    query = torch.randn(1, 6, 2, 16)
    assert layout.attention(query, query, query, "native").shape == query.shape
    assert "valid_mask" in vars(layout) and "padded_indices" in vars(layout)


def test_packing_preserves_row_timesteps_and_resets_positions():
    serial, _ = _models()
    inputs = _inputs(serial, task="fl2va")
    samples = [sample for sample, _, _ in MiniMaxH3DiffusionNFT._iter_sample_inputs(serial, inputs)]
    packed = pack_model_inputs(samples)
    expected = torch.cat([sample["timestep"][sample["timestep_indices"]] for sample in samples])
    torch.testing.assert_close(packed["timestep"][packed["timestep_indices"]], expected)
    torch.testing.assert_close(packed["position_ids"], torch.cat([sample["position_ids"] for sample in samples]))
    assert packed["sequence_layout"].cu_seqlens.tolist() == [0, 17, 36, 54]
    assert packed["text_sequence_layout"].cu_seqlens.tolist() == [0, 3, 8, 12]


def test_checkpoint_recompute_keeps_each_forward_boundaries():
    serial, packed = _models(checkpointing=True)
    batches = [_inputs(serial, lengths=(3, 5)), _inputs(serial, lengths=(4, 2, 3))]
    # Two forwards remain live before backward; no mutable processor-side layout is allowed.
    serial_loss = sum(_forward(serial, inputs, False).square().mean() for inputs in batches)
    packed_loss = sum(_forward(packed, inputs, True).square().mean() for inputs in batches)
    serial_loss.backward()
    packed_loss.backward()
    for (name, p), (other_name, q) in zip(serial.named_parameters(), packed.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_checkpoint_roundtrip_preserves_class_config_and_weight_names(tmp_path):
    serial, packed = _models()
    serial.save_pretrained(tmp_path)
    loaded = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path)
    enable_packed_forward(loaded)
    assert loaded.config.hidden_size == 32
    assert loaded.state_dict().keys() == serial.state_dict().keys()
    inputs = _inputs(serial)
    torch.testing.assert_close(_forward(loaded, inputs, True), _forward(packed, inputs, True))
    loaded.save_pretrained(tmp_path / "packed")
    restored = MiniMaxH3Transformer3DModel.from_pretrained(tmp_path / "packed")
    restored.set_attention_backend("native")
    assert restored.state_dict().keys() == serial.state_dict().keys()
    torch.testing.assert_close(_forward(restored, inputs, False), _forward(serial, inputs, False))


def test_fsdp_loader_keeps_attention_checkpointing_and_fp32_islands(tmp_path, monkeypatch):
    from verl_omni.workers.config import DiffusionModelConfig
    from verl_omni.workers.engine.fsdp import diffusers_impl

    serial, _ = _models()
    serial.save_pretrained(tmp_path)
    config = DiffusionModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        architecture="MiniMaxH3Pipeline",
        algorithm="diffusion_nft",
        external_lib=None,
        config_path=str(tmp_path),
        local_path=str(tmp_path),
        trust_remote_code=False,
        attn_backend="native",
        enable_gradient_checkpointing=True,
    )
    engine = SimpleNamespace(
        model_config=config,
        engine_config=SimpleNamespace(model_dtype="bf16"),
        device_mesh=None,
        _build_module_from_registry=lambda _: None,
    )
    monkeypatch.setattr(diffusers_impl, "get_init_weight_context_manager", lambda **_: nullcontext)
    model = diffusers_impl.DiffusersFSDPEngine._build_module(engine)
    assert type(model) is MiniMaxH3Transformer3DModel
    assert not getattr(model, "supports_packed_batch", False)
    enable_packed_forward(model)
    assert model.gradient_checkpointing and model.token_refiner.gradient_checkpointing
    assert model.proj_in.weight.dtype == torch.float32
    assert model.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert model.transformer_blocks[0].attn.processor._attention_backend == "native"


def test_packed_model_preserves_recipe_fsdp_wrap_targets():
    from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets

    for model in _models():
        targets = _select_fsdp2_wrap_targets(model, ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"])
        assert set(targets) == set([*model.transformer_blocks, *model.token_refiner.refiner_blocks])


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_h3_packing_is_default_with_the_generic_model_config(tmp_path, algorithm):
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    from verl_omni.workers.config import DiffusionModelConfig

    (tmp_path / "model_index.json").write_text('{"_class_name": "MiniMaxH3Pipeline"}')
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config/diffusion/model"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="diffusion_model",
            overrides=[
                f"path={tmp_path}",
                "+load_tokenizer=false",
                f"algorithm={algorithm}",
                "attn_backend=native",
            ],
        )
    model_config = omega_conf_to_dataclass(config)
    assert type(model_config) is DiffusionModelConfig
    adapter = DiffusionModelBase.get_class_by_name("MiniMaxH3Pipeline", algorithm)
    model = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    enable_packed_forward(model)
    assert getattr(model, "supports_packed_batch", False)
    assert adapter is not None
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

    rollout_model_config = DiffusionStrategy(None).init_model_config(config)
    assert type(rollout_model_config) is DiffusionModelConfig
    assert "use_packed_batch" not in config


def test_generic_diffusion_config_does_not_expose_packing():
    from dataclasses import fields

    from hydra import compose, initialize_config_dir

    from verl_omni.workers.config import DiffusionModelConfig

    assert "use_packed_batch" not in {field.name for field in fields(DiffusionModelConfig)}
    with pytest.raises(TypeError, match="use_packed_batch"):
        DiffusionModelConfig(use_packed_batch=True)
    with pytest.raises(ValueError, match="Invalid attn_backend"):
        DiffusionModelConfig(attn_backend="torch_varlen")
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer")
    assert "use_packed_batch" not in config.actor_rollout_ref.model


def test_packed_boundaries_must_match_rows():
    serial, packed = _models()
    samples = [sample for sample, _, _ in MiniMaxH3DiffusionNFT._iter_sample_inputs(serial, _inputs(serial))]
    inputs = pack_model_inputs(samples)
    inputs["text_sequence_layout"] = PackedSequenceLayout.from_lengths([1], torch.device("cpu"))
    with pytest.raises(ValueError, match="text attention boundaries"):
        packed(**inputs)
    inputs = pack_model_inputs(samples)
    inputs["sequence_layout"] = PackedSequenceLayout.from_lengths([1], torch.device("cpu"))
    with pytest.raises(ValueError, match="attention boundaries"):
        packed(**inputs)


def test_packed_input_preparation_rejects_mixed_target_geometry():
    serial, _ = _models()
    batch = TensorDict({"latent_meta": torch.tensor([[4, 6, 1, 4, 4, 3], [4, 6, 1, 2, 8, 3]])}, batch_size=[2])
    with pytest.raises(ValueError, match="shared target latent layout"):
        MiniMaxH3DiffusionNFT.prepare_model_inputs(
            serial,
            SimpleNamespace(),
            None,
            None,
            None,
            None,
            None,
            None,
            batch,
            0,
        )


def test_automodel_forward_override_and_unsupported_backend_fail_closed():
    serial, packed = _models()
    assert type(packed) is MiniMaxH3Transformer3DModel
    assert packed is enable_packed_forward(packed)
    with pytest.raises(ValueError, match="attn_backend"):
        packed.set_attention_backend("unsupported_backend")

    sequence_parallel = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    sequence_parallel.transformer_blocks[0].attn.processor._parallel_config = object()
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        enable_packed_forward(sequence_parallel)
    with pytest.raises(TypeError, match="MiniMaxH3Transformer3DModel"):
        enable_packed_forward(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("lengths", [[], [0], [3, -1]])
def test_empty_or_invalid_sequences_are_rejected(lengths):
    with pytest.raises(ValueError, match="positive lengths"):
        PackedSequenceLayout.from_lengths(lengths, torch.device("cpu"))

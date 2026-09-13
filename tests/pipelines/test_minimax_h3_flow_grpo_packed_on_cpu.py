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
"""FlowGRPO transition and gradient parity for variable-length H3 packed batches."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from tests.pipelines.test_minimax_h3_packed_forward_on_cpu import _MODEL_KWARGS, _inputs, _models
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.minimax_h3_flow_grpo.common import (
    configure_flow_scheduler,
    flatten_joint_latents,
    h3_sigma_schedules,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.diffusers_training_adapter import MiniMaxH3FlowGRPO
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


def _config(sde_type="sde"):
    return SimpleNamespace(
        algo=SimpleNamespace(noise_level=0.6, sde_type=sde_type),
        pipeline=SimpleNamespace(av_logprob_video_weight=0.25, av_logprob_audio_weight=0.75),
    )


def _schedulers(device="cpu"):
    schedulers = (FlowMatchSDEDiscreteScheduler(), FlowMatchSDEDiscreteScheduler())
    for scheduler, sigmas in zip(schedulers, h3_sigma_schedules(4), strict=True):
        configure_flow_scheduler(scheduler, sigmas, device)
    return schedulers


def _flow_batch(model, task="t2va", lengths=(3, 5, 4), shared_steps=False):
    nft = _inputs(model, task=task, lengths=lengths)
    samples = list(MiniMaxH3DiffusionNFT._iter_sample_inputs(model, nft))
    batch = len(samples)
    steps = torch.zeros(batch, dtype=torch.long) if shared_steps else torch.arange(batch)
    video_sigmas, audio_sigmas = (torch.tensor(sigmas) for sigmas in h3_sigma_schedules(4))
    currents = []
    for sample, cv, ca in samples:
        video, audio = sample["hidden_states"], sample["audio_hidden_states"]
        if task == "ref2va":
            video, audio = video[:, cv:], audio[:, ca:]
        currents.append(flatten_joint_latents(video, audio))
    current = torch.cat(currents)
    result = {
        "all_latents": current.unsqueeze(1),
        "all_next_latents": (current + torch.randn_like(current) * 0.05).unsqueeze(1),
        "all_timesteps": video_sigmas[steps, None],
        "h3_audio_timesteps": audio_sigmas[steps, None],
        "h3_step_indices": steps[:, None],
        "prompt_embeds": nft["encoder_hidden_states"],
        "prompt_embeds_mask": nft["encoder_mask"],
    }
    if task == "ref2va":
        for key in (
            "ref_block_meta",
            "ref_block_count",
            "condition_video_rows",
            "condition_audio_rows",
            "condition_video_row_count",
            "condition_audio_row_count",
        ):
            result[key] = nft[key]
        result["latent_meta"] = torch.tensor(nft["latent_meta"]).repeat(batch, 1)
        result["prompt_token_tags"] = pad_sequence(
            [sample["token_tags"][sample["text_indices"]] for sample, _, _ in samples], batch_first=True
        )
    else:
        for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
            result[f"h3_{name}"] = pad_sequence([sample[name] for sample, _, _ in samples], batch_first=True)
        result["h3_video_rows"] = torch.tensor([sample["hidden_states"].shape[1] for sample, _, _ in samples])
        result["h3_audio_rows"] = torch.tensor([sample["audio_hidden_states"].shape[1] for sample, _, _ in samples])
        result["h3_seq_len"] = torch.tensor([sample["position_ids"].shape[0] for sample, _, _ in samples])
        result["h3_video_update_mask"] = torch.stack(
            [torch.arange(sample["hidden_states"].shape[1]) >= cv for sample, cv, _ in samples]
        )
    return TensorDict(result, batch_size=[batch])


def _prepare(model, data, packed, step=0):
    if not packed:
        return MiniMaxH3FlowGRPO._prepare_batch_inputs(
            data["all_latents"], data["all_timesteps"], data["prompt_embeds"], data["prompt_embeds_mask"], data, step
        )[0]
    return MiniMaxH3FlowGRPO.prepare_model_inputs(
        model,
        _config(),
        data["all_latents"],
        data["all_timesteps"],
        data["prompt_embeds"],
        data["prompt_embeds_mask"],
        None,
        None,
        data,
        step,
    )[0]


def _run(model, data, packed, schedulers, sde_type="sde", step=0):
    if not packed:
        inputs = _prepare(model, data, False, step)
        device = data["all_latents"].device
        kwargs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
            if not key.startswith("_h3_")
        }
        video, audio = model(**kwargs)
        return MiniMaxH3FlowGRPO._sample_previous_step(schedulers, _config(sde_type), inputs, data, video, audio, step)
    return MiniMaxH3FlowGRPO.forward_and_sample_previous_step(
        model,
        schedulers,
        _config(sde_type),
        _prepare(model, data, packed, step),
        None,
        data,
        step,
    )


def _serial(model, data, schedulers, sde_type="sde"):
    outputs = [_run(model, data[i : i + 1], False, schedulers, sde_type) for i in range(data.shape[0])]
    return tuple(torch.cat(parts) for parts in zip(*outputs, strict=True))


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("sde_type", ["sde", "cps"])
def test_variable_batches_match_serial_transitions_loss_and_lora_gradients(task, sde_type):
    serial, packed = _models(lora=True, checkpointing=True)
    data = _flow_batch(serial, task)
    schedulers = _schedulers()
    serial.set_adapter("old")
    with torch.no_grad():
        old_log_probs = _serial(serial, data, schedulers, sde_type)[0]
    serial.set_adapter("default")
    packed.set_adapter("default")
    expected = _serial(serial, data, schedulers, sde_type)
    calls = []
    handle = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    actual = _run(packed, data, True, schedulers, sde_type)
    handle.remove()
    assert len(calls) == 1
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)
    losses = [
        FlowGRPOLoss.compute_loss(
            old_log_prob=old_log_probs,
            log_prob=output[0],
            advantages=torch.tensor([0.6, -0.8, 0.3]),
            config=SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2)),
        )[0]
        for output in (expected, actual)
    ]
    torch.testing.assert_close(*losses, rtol=3e-4, atol=3e-5)
    for loss in losses:
        loss.backward()
    gradients = dict(serial.named_parameters())
    norm = 0.0
    for name, parameter in packed.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and gradients[name].grad is not None
            torch.testing.assert_close(parameter.grad, gradients[name].grad, rtol=1e-3, atol=3e-5)
            norm += parameter.grad.norm().item()
    assert norm > 0


@pytest.mark.parametrize("steps", [[1, 0, 1], [1, 1, 0]])
def test_reference_layouts_batch_scheduler_replay_by_step_and_restore_order(monkeypatch, steps):
    from unittest.mock import Mock

    serial, packed = _models()
    data = _flow_batch(serial, "ref2va")
    video_sigmas, audio_sigmas = (torch.tensor(sigmas) for sigmas in h3_sigma_schedules(4))
    data["h3_step_indices"] = torch.tensor(steps)[:, None]
    data["all_timesteps"] = video_sigmas[steps, None]
    data["h3_audio_timesteps"] = audio_sigmas[steps, None]
    expected = _serial(serial, data, _schedulers())
    replay = Mock(wraps=MiniMaxH3FlowGRPO._sample_previous_step)
    monkeypatch.setattr(MiniMaxH3FlowGRPO, "_sample_previous_step", replay)
    actual = _run(packed, data, True, _schedulers())
    assert replay.call_count == 2
    assert [call.args[2]["hidden_states"].shape[0] for call in replay.call_args_list] == [2, 1]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_packed_matches_existing_dense_batch_when_layouts_are_shared():
    dense, packed = _models()
    data = _flow_batch(dense, lengths=(4, 4, 4), shared_steps=True)
    expected = _run(dense, data, False, _schedulers())
    actual = _run(packed, data, True, _schedulers())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_packed_ref2va_accepts_variable_reference_layouts_rejected_by_dense_batch():
    dense, packed = _models()
    data = _flow_batch(dense, task="ref2va", lengths=(4, 4, 4, 4), shared_steps=True)
    grouped_fields = (
        "prompt_embeds",
        "prompt_embeds_mask",
        "ref_block_meta",
        "ref_block_count",
        "condition_video_rows",
        "condition_audio_rows",
        "condition_video_row_count",
        "condition_audio_row_count",
        "prompt_token_tags",
    )
    for key in grouped_fields:
        first_prompt, second_prompt = data[key][0].clone(), data[key][1].clone()
        data[key][0:2] = first_prompt
        data[key][2:4] = second_prompt

    # Model two rollout groups with n=2: layouts match within each prompt and differ across prompts.
    assert data["condition_video_row_count"].flatten().tolist() == [4, 4, 8, 8]
    assert data["condition_audio_row_count"].flatten().tolist() == [0, 0, 4, 4]
    torch.testing.assert_close(data["prompt_embeds"][0], data["prompt_embeds"][1])
    torch.testing.assert_close(data["prompt_embeds"][2], data["prompt_embeds"][3])

    with pytest.raises(ValueError, match="shared condition video row count"):
        _prepare(dense, data, False)

    prepared = _prepare(packed, data, True)
    sequence_lengths = [sample["position_ids"].shape[0] for sample in prepared["_h3_samples"]]
    assert sequence_lengths[0] == sequence_lengths[1]
    assert sequence_lengths[2] == sequence_lengths[3]
    assert sequence_lengths[0] != sequence_lengths[2]

    expected = _serial(dense, data, _schedulers())
    calls = []
    handle = packed.register_forward_pre_hook(lambda *_: calls.append(1))
    actual = _run(packed, data, True, _schedulers())
    handle.remove()
    assert len(calls) == 1
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-5)


def test_different_text_lengths_still_fail_closed_in_dense_mode():
    dense, _ = _models()
    with pytest.raises(ValueError, match="shared text length"):
        _prepare(dense, _flow_batch(dense), False)


def test_flowgrpo_installs_packed_forward_on_automodel():
    dense = MiniMaxH3Transformer3DModel(**_MODEL_KWARGS)
    data = _flow_batch(dense)
    assert not getattr(dense, "supports_packed_batch", False)
    _run(dense, data, True, _schedulers())
    assert getattr(dense, "supports_packed_batch", False)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_engine_prepares_packed_replay_without_slicing_unrelated_nested_fields(task):
    from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine

    _, packed = _models()
    data = _flow_batch(packed, task)
    raw = data.clone()
    lengths = data["prompt_embeds_mask"].sum(dim=1).tolist()
    for key in ("prompt_embeds", "prompt_embeds_mask"):
        raw[key] = torch.nested.nested_tensor(
            [value[:length] for value, length in zip(data[key], lengths, strict=True)], layout=torch.jagged
        )
    raw["input_ids"] = torch.nested.nested_tensor(
        [torch.ones(n, dtype=torch.long) for n in lengths], layout=torch.jagged
    )
    for key in ("condition_video_rows", "condition_audio_rows"):
        if key not in raw:
            continue
        counts = data[key.replace("_rows", "_row_count")].flatten().tolist()
        raw[key] = torch.nested.nested_tensor(
            [value[:count] for value, count in zip(data[key], counts, strict=True)], layout=torch.jagged
        )
        raw[f"{key}_mask"] = torch.nested.nested_tensor(
            [torch.ones(count, dtype=torch.bool) for count in counts], layout=torch.jagged
        )
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = packed
    engine.model_config = _config()
    engine.model_config.architecture = "MiniMaxH3Pipeline"
    engine.model_config.algorithm = "flow_grpo"
    engine.model_config.external_lib = None
    engine.use_ulysses_sp = False
    inputs, _ = engine.prepare_model_inputs(raw, step=0)
    expected = _prepare(packed, data, True)
    for a, b in zip(inputs["_h3_samples"], expected["_h3_samples"], strict=True):
        for key in a:
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key])
            else:
                assert a[key] == b[key]
    assert raw["input_ids"].is_nested
    assert raw["prompt_embeds"].is_nested


def test_packed_replay_selects_the_requested_trajectory_column():
    serial, packed = _models()
    data = _flow_batch(serial)
    for key in ("all_latents", "all_next_latents", "all_timesteps", "h3_audio_timesteps", "h3_step_indices"):
        data[key] = torch.cat([data[key], data[key]], dim=1)
    data["all_next_latents"][:, 1] += 0.5
    actual = _run(packed, data, True, _schedulers(), step=1)
    expected = [_run(serial, data[i : i + 1], False, _schedulers(), step=1) for i in range(3)]
    for a, parts in zip(actual, zip(*expected, strict=True), strict=True):
        torch.testing.assert_close(a, torch.cat(parts), rtol=3e-4, atol=3e-5)
    assert not torch.allclose(actual[0], _run(packed, data, True, _schedulers(), step=0)[0])


def test_packed_rejects_different_target_row_counts():
    _, packed = _models()
    data = _flow_batch(packed)
    data["h3_video_update_mask"][1, 0] = False
    with pytest.raises(ValueError, match="shared target video/audio row counts"):
        _prepare(packed, data, True)


def test_packed_rejects_truncated_reference_rows():
    _, packed = _models()
    data = _flow_batch(packed, "ref2va")
    data["condition_video_rows"] = data["condition_video_rows"][:, :2]
    with pytest.raises(ValueError, match="counts exceed the supplied condition tensors"):
        _prepare(packed, data, True)


def test_packed_samples_cannot_attend_to_each_other():
    _, packed = _models()
    data = _flow_batch(packed)
    expected = _run(packed, data, True, _schedulers())
    modified = data.clone()
    modified["prompt_embeds"][1] += 10
    modified["all_latents"][1] += 3
    actual = _run(packed, modified, True, _schedulers())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a[[0, 2]], b[[0, 2]])
    assert not torch.allclose(actual[1][1], expected[1][1])

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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl_omni.pipelines.flux2_klein_flow_grpo.diffusers_training_adapter import Flux2KleinFlowGRPO


def _non_tensor_stack(values):
    return NonTensorStack.from_list([NonTensorData(value) for value in values])


def _model_config(true_cfg_scale: float = 1.0):
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            guidance_scale=1.0,
            true_cfg_scale=true_cfg_scale,
            height=32,
            width=48,
        ),
        algo=SimpleNamespace(noise_level=0.0, sde_type="sde"),
    )


class _EchoModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return (kwargs["hidden_states"],)


def test_prepare_model_inputs_reads_constant_latent_size_from_tensordict():
    micro_batch = TensorDict(
        {
            "latent_h": _non_tensor_stack([2, 2]),
            "latent_w": _non_tensor_stack([3, 3]),
        },
        batch_size=[2],
    )

    model_inputs, negative_model_inputs = Flux2KleinFlowGRPO.prepare_model_inputs(
        module=None,
        model_config=_model_config(),
        latents=torch.zeros(2, 1, 6, 128),
        timesteps=torch.ones(2, 1),
        prompt_embeds=torch.zeros(2, 5, 4),
        prompt_embeds_mask=torch.ones(2, 5, dtype=torch.bool),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        micro_batch=micro_batch,
        step=0,
    )

    assert negative_model_inputs is None
    assert model_inputs["img_ids"].shape == (2, 6, 4)


def test_prepare_model_inputs_rejects_mixed_latent_size():
    micro_batch = TensorDict(
        {
            "latent_h": _non_tensor_stack([2, 3]),
            "latent_w": _non_tensor_stack([3, 3]),
        },
        batch_size=[2],
    )

    with pytest.raises(ValueError, match="latent_h differs across the micro-batch"):
        Flux2KleinFlowGRPO.prepare_model_inputs(
            module=None,
            model_config=_model_config(),
            latents=torch.zeros(2, 1, 6, 128),
            timesteps=torch.ones(2, 1),
            prompt_embeds=torch.zeros(2, 5, 4),
            prompt_embeds_mask=torch.ones(2, 5, dtype=torch.bool),
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            micro_batch=micro_batch,
            step=0,
        )


def test_prepare_condition_uses_tensordict_metadata_helper():
    micro_batch = TensorDict(
        {
            "condition_image_latents": torch.zeros(2, 3, 128),
            "condition_image_latent_ids": torch.zeros(2, 3, 4),
            "sp_size": NonTensorData(2),
        },
        batch_size=[2],
    )

    condition = Flux2KleinFlowGRPO.prepare_condition(
        micro_batch,
        latents=torch.zeros(2, 1, 4, 128),
        step=0,
    )

    assert condition["image_latents"].shape == (2, 3, 128)
    assert condition["image_latent_ids"].shape == (2, 3, 4)
    assert condition["sp_size"] == 2


def test_inject_condition_concatenates_latents_and_ids():
    condition_latents = torch.ones(1, 4, 128)
    condition_ids = torch.ones(1, 4, 4)
    model_inputs = {
        "hidden_states": torch.zeros(1, 4, 128),
        "img_ids": torch.zeros(1, 4, 4),
    }

    output, _ = Flux2KleinFlowGRPO.inject_condition(
        model_inputs,
        None,
        {
            "image_latents": condition_latents,
            "image_latent_ids": condition_ids,
            "sp_size": 2,
        },
    )

    assert output["hidden_states"].shape == (1, 8, 128)
    assert output["img_ids"].shape == (1, 8, 4)
    assert output["_target_seq_len"] == 4
    torch.testing.assert_close(output["hidden_states"][:, 4:], condition_latents)
    torch.testing.assert_close(output["img_ids"][:, 4:], condition_ids)


def test_forward_uses_i2i_base_to_crop_condition_predictions():
    module = _EchoModule()

    prediction = Flux2KleinFlowGRPO.forward(
        module,
        _model_config(),
        {
            "hidden_states": torch.zeros(1, 8, 128),
            "_target_seq_len": 4,
        },
    )

    assert prediction.shape == (1, 4, 128)
    assert "_target_seq_len" not in module.last_kwargs


def test_sampling_receives_only_target_predictions():
    class _Scheduler:
        model_output = None

        def sample_previous_step(self, **kwargs):
            self.model_output = kwargs["model_output"]
            value = torch.zeros(1)
            return None, value, value, value, value

    scheduler = _Scheduler()
    module = _EchoModule()
    Flux2KleinFlowGRPO.forward_and_sample_previous_step(
        module=module,
        scheduler=scheduler,
        model_config=_model_config(),
        model_inputs={
            "hidden_states": torch.zeros(1, 8, 128),
            "_target_seq_len": 4,
        },
        negative_model_inputs=None,
        scheduler_inputs={
            "all_latents": torch.zeros(1, 2, 4, 128),
            "all_timesteps": torch.ones(1, 1),
        },
        step=0,
    )

    assert scheduler.model_output.shape == (1, 4, 128)

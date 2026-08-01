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

"""GPU smoke for the omni automodel actor engine.

Mirrors :mod:`tests.workers.test_diffusers_veomni_engine` but exercises the
``(omni_model, automodel)`` engine backend: verifies that
``OmniAutomodelEngine.initialize()`` builds a real ``NeMoAutoModelForMultimodalLM``
+ omni-adapter ``configure_model`` on GPU, and that one forward and one
train step drive the inherited LM-head path end-to-end.

Skips itself if ``nemo_automodel`` is not installed (optional backend).
"""

import os
from functools import partial

import pytest
import ray
import torch
from tensordict import TensorDict
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu
from verl.workers.config import TrainingWorkerConfig

from verl_omni.workers.config.omni import OmniAutomodelActorConfig, OmniModelConfig
from verl_omni.workers.engine_workers import TrainingWorker

from ..utils.gpu_test_topology import resolve_requested_num_gpus

pytest.importorskip("nemo_automodel")


def _create_training_config(device_count: int, model_path: str):
    """Compose the automodel actor + model config for a Qwen3-Omni tiny-random smoke."""
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    actor_dir = os.path.abspath("examples/automodel_trainer/qwen3_omni/config/actor")
    with initialize_config_dir(config_dir=actor_dir, version_base=None):
        cfg = compose(
            config_name="qwen3_omni_automodel_actor",
            overrides=[
                # trainer_type is resolved from ${algorithm.trainer_type} at trainer runtime;
                # pin it here so the actor config validates standalone.
                "trainer_type=direct_preference",
                # OmniActorConfig-style runtime injections that live in _mutable_fields.
                "+rollout_n=1",
                "+optim.total_training_steps=4",
                # Keep the smoke on a single GPU with everything on device.
                "automodel.attn_implementation=sdpa",
                "automodel.param_offload=false",
                "automodel.optimizer_offload=false",
                "automodel.activation_checkpointing=false",
                "automodel.use_dynamic_bsz=false",
                "automodel.micro_batch_size_per_gpu=2",
                "automodel.infer_micro_batch_size_per_gpu=2",
            ],
        )
    actor_config: OmniAutomodelActorConfig = omega_conf_to_dataclass(cfg)

    model_config: OmniModelConfig = omega_conf_to_dataclass(
        {
            "_target_": "verl_omni.workers.config.omni.OmniModelConfig",
            "path": model_path,
            "model_type": "omni_model",
            "trust_remote_code": False,
        },
        dataclass_type=OmniModelConfig,
    )

    training_config = TrainingWorkerConfig(
        model_type="omni_model",
        model_config=model_config,
        engine_config=actor_config.engine,
        optimizer_config=actor_config.optim,
        checkpoint_config=actor_config.checkpoint,
    )
    return training_config, actor_config


def _build_no_padding_batch(vocab_size: int, batch_size: int = 2) -> TensorDict:
    """Nested-tensor NO_PADDING batch — automodel's ``prepare_model_inputs`` only supports that mode."""
    lens = [16, 12][:batch_size]

    def _nested(rows):
        return torch.nested.nested_tensor(rows, layout=torch.jagged)

    ids_rows = [torch.randint(0, vocab_size, (n,)) for n in lens]
    data = TensorDict(
        {
            "input_ids": _nested(ids_rows),
            "position_ids": _nested([torch.arange(n) for n in lens]),
            "loss_mask": _nested([torch.ones(n, dtype=torch.long) for n in lens]),
            "responses": _nested([r[-8:] for r in ids_rows]),
            "response_mask": _nested([torch.ones(8, dtype=torch.long) for _ in lens]),
        },
        batch_size=[batch_size],
    )
    # engine_workers.TrainingWorker.train_batch/infer_batch fills these when absent,
    # but the loss/forward path also reads temperature — set it explicitly.
    tu.assign_non_tensor(data, temperature=1.0)
    return data


def _probe_loss(model_output, data, dp_group=None):
    """Minimal loss compatible with ``forward_step``'s loss_function(model_output=, data=, dp_group=) contract."""
    log_probs = model_output["log_probs"]
    loss = -log_probs.float().mean()
    return loss, {"probe_loss": loss.detach().item()}


def test_omni_automodel_engine():
    ray.init()
    try:
        visible_gpus = torch.cuda.device_count()
        device_count = resolve_requested_num_gpus(default_num_gpus=max(1, visible_gpus))

        model_path = os.path.expanduser("~/models/tiny-random/Qwen3-Omni")
        training_config, actor_config = _create_training_config(device_count=device_count, model_path=model_path)

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=training_config)
        resource_pool = RayResourcePool(process_on_nodes=[device_count])
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.reset()

        # Read vocab_size straight from the HF config (no worker roundtrip).
        vocab_size = training_config.model_config.hf_config.thinker_config.text_config.vocab_size

        # ---- infer path ----
        data = _build_no_padding_batch(vocab_size=vocab_size, batch_size=2)
        tu.assign_non_tensor(data, compute_loss=False)
        output = wg.infer_batch(data).get()
        assert "metrics" in output.keys(), f"expected 'metrics' in infer output; got {sorted(output.keys())}"

        # ---- train path ----
        wg.set_loss_fn(partial(_probe_loss))
        data = _build_no_padding_batch(vocab_size=vocab_size, batch_size=2)
        output = wg.train_batch(data).get()
        assert "metrics" in output.keys(), f"expected 'metrics' in train output; got {sorted(output.keys())}"
    finally:
        ray.shutdown()

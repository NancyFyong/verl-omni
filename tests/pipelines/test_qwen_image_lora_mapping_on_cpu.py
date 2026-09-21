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
"""CPU regressions for Qwen-Image live LoRA mapping and binding validation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_qwen_lora_maps_output_projection_keys_and_targets_without_mutating_inputs():
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    tensor = torch.ones(2, 3)
    name = "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"
    config = {"target_modules": ["to_out.0", "to_q", "transformer_blocks.0.attn.to_out.0"]}
    mapped, mapped_config = QwenImagePipelineWithLogProb.map_lora_update_to_engine({name: tensor}, config)
    assert list(mapped) == ["transformer.transformer_blocks.0.attn.to_out.lora_A.weight"]
    assert next(iter(mapped.values())) is tensor
    assert mapped_config["target_modules"] == ["to_out", "to_q", "transformer_blocks.0.attn.to_out"]
    assert config["target_modules"][0] == "to_out.0"


def test_qwen_lora_rejects_colliding_names():
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    key = "transformer.transformer_blocks.0.attn.to_out"
    with pytest.raises(ValueError, match="Duplicate Qwen-Image LoRA tensor"):
        QwenImagePipelineWithLogProb.map_lora_update_to_engine(
            {f"{key}.0.lora_A.weight": torch.ones(4, 32), f"{key}.lora_A.weight": torch.zeros(4, 32)}, {}
        )


def test_qwen_lora_rejects_silent_partial_binding():
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    model = SimpleNamespace(loras={"q": object(), "out": object()})
    with pytest.raises(ValueError, match="1 unbound modules"):
        QwenImagePipelineWithLogProb._validate_diffusion_lora_binding(
            lora_model=model, bound_lora_names=frozenset({"q"})
        )
    QwenImagePipelineWithLogProb._validate_diffusion_lora_binding(
        lora_model=model, bound_lora_names=frozenset({"q", "out"})
    )


_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def _export_actor(monkeypatch, backend):
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=32,
        joint_attention_dim=32,
        axes_dims_rope=(8, 12, 12),
    )
    if backend == "veomni":
        lora = pytest.importorskip("veomni.lora")
        from tests.workers.test_veomni_diffusion_lora_on_cpu import _export, _make_engine

        model = lora.VeOmniLoraModel(model, lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
        engine = _make_engine(model, lora_rank=4)
    else:
        from peft import LoraConfig, get_peft_model

        from tests.workers.test_diffusers_fsdp_merged_lora_on_cpu import _make_engine, _patch_sync_helpers

        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
        _patch_sync_helpers(monkeypatch)
        engine = _make_engine(model, lora_config={})

    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters()):
            if ".lora_" in name:
                param.fill_((index + 1) / 128)
    if backend == "veomni":
        return _export(engine, monkeypatch, base_sync_done=True)
    params, config = engine.get_per_tensor_param(base_sync_done=True)
    return dict(params), config


@pytest.fixture
def runtime_manager(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen

    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(qwen, "Attention", lambda **_: torch.nn.Identity())
    # Restore the global hijack when leaving this test, including on failure.
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        transformer = qwen.QwenImageTransformer2DModel(
            OmniDiffusionConfig(),
            num_layers=1,
            num_attention_heads=1,
            attention_head_dim=32,
            joint_attention_dim=32,
            axes_dims_rope=(8, 12, 12),
        )
        transformer.load_weights([])  # Installs the real QKV stacked-parameter mapping.
        pipeline = object.__new__(QwenImagePipelineWithLogProb)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        assert isinstance(transformer.transformer_blocks[0].attn.to_out, qwen.RowParallelLinear)
        pipeline._validate_diffusion_lora_binding = Mock(wraps=pipeline._validate_diffusion_lora_binding)
        yield DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize("backend", ["fsdp2", "veomni"])
@pytest.mark.parametrize("damage", [None, "partial", "empty"])
def test_export_load_bind_activate_contract(monkeypatch, runtime_manager, backend, damage):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    params, config = _export_actor(monkeypatch, backend)
    assert len(params) == 24
    if damage == "partial":
        params["transformer.missing.to_out.lora_A.weight"] = torch.ones(4, 32)
        params["transformer.missing.to_out.lora_B.weight"] = torch.ones(32, 4)
    elif damage == "empty":
        params = {}
    request = OmniTensorLoRARequest(
        lora_name="actor",
        lora_int_id=1,
        lora_path="in-memory",
        lora_tensors=params,
        peft_config=config,
    )
    manager = runtime_manager
    if damage:
        with pytest.raises(ValueError, match="no-op sync"):
            manager.set_active_adapter(request)
        assert manager._active_adapter_id is None
        for layer in manager._lora_modules.values():
            for tensor in (*layer.lora_a_stacked, *layer.lora_b_stacked):
                assert torch.count_nonzero(tensor) == 0
        return

    manager.set_active_adapter(request)
    assert manager._active_adapter_id == 1
    validator = manager.pipeline._validate_diffusion_lora_binding
    validator.assert_called_once()
    assert len(validator.call_args.kwargs["bound_lora_names"]) == 12
    mapped, _ = manager.pipeline.map_lora_update_to_engine(params, config)
    for name, layer in manager._lora_modules.items():
        prefix, _, suffix = name.rpartition(".")
        sublayers = manager._packed_modules_mapping.get(suffix, [suffix])
        for index, sublayer in enumerate(sublayers):
            key = f"{prefix}.{sublayer}"
            expected_a = mapped[f"{key}.lora_A.weight"].float()
            expected_b = mapped[f"{key}.lora_B.weight"].float() * 2
            actual_a = layer.lora_a_stacked[index][0, 0, :4]
            actual_b = layer.lora_b_stacked[index][0, 0, :, :4]
            torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
            torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
            torch.testing.assert_close(actual_b @ actual_a, expected_b @ expected_a, rtol=0, atol=0)

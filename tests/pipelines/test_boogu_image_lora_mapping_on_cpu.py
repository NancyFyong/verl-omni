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
"""CPU regressions for Boogu-Image live LoRA name mapping."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("container", [list, tuple, set])
def test_boogu_maps_keys_and_targets_without_mutating_inputs(container):
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    pipeline = VllmOmniPipelineBase.get_class("BooguImagePipeline", "flow_grpo")
    tensor = torch.ones(4, 32)
    prefix = "transformer.double_stream_layers.0.img_instruct_attn."
    tensors = {f"{prefix}to_out.0.lora_A.weight": tensor, f"{prefix}img_out.lora_A.weight": tensor}
    targets = container(["to_out.0", "img_out", "double_stream_layers.0.img_instruct_attn.to_out.0"])
    config = {"target_modules": targets, "r": 4, "lora_alpha": 8}
    mapped, mapped_config = pipeline.map_lora_update_to_engine(tensors, config)
    assert list(mapped) == [f"{prefix}to_out.lora_A.weight", f"{prefix}img_out.lora_A.weight"]
    assert all(value is tensor for value in mapped.values())
    assert set(mapped_config["target_modules"]) == {
        "to_out",
        "img_out",
        "double_stream_layers.0.img_instruct_attn.to_out",
    }
    assert config["target_modules"] is targets and "to_out.0" in targets
    assert f"{prefix}to_out.0.lora_A.weight" in tensors
    assert mapped_config["r"] == 4 and mapped_config["lora_alpha"] == 8


@pytest.fixture(params=["self", "joint"])
def boogu_manager(monkeypatch, request):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer as boogu

    from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(boogu, "Attention", lambda **_: torch.nn.Identity())
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        attention_class = boogu.BooguImageSelfAttention if request.param == "self" else boogu.BooguImageJointAttention
        attention = attention_class(dim=32, num_attention_heads=1, num_kv_heads=1)
        # Exercise both real Boogu output layouts without constructing the full model.
        output_class = boogu.RowParallelLinear if request.param == "self" else boogu.ReplicatedLinear
        assert isinstance(attention.to_out, output_class)
        pipeline = object.__new__(BooguImagePipelineWithLogProb)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = torch.nn.ModuleDict({"attn": attention})
        targets = ["to_out.0", "to_q" if request.param == "self" else "img_to_q"]
        params = {}
        for target in targets:
            prefix = f"transformer.attn.{target}"
            params[f"{prefix}.lora_A.weight"] = torch.full((4, 32), 0.125)
            params[f"{prefix}.lora_B.weight"] = torch.full((32, 4), 0.25)
        config = {"r": 4, "lora_alpha": 8, "target_modules": targets}
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        yield manager, params, config


@pytest.mark.parametrize("case", ["unmapped", "full", "zero_init", "output_only"])
def test_boogu_load_bind_activate_contract(boogu_manager, monkeypatch, case):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = boogu_manager
    if case == "unmapped":
        monkeypatch.setattr(manager.pipeline, "map_lora_update_to_engine", lambda tensors, config: (tensors, config))
    elif case == "zero_init":
        params = {name: torch.zeros_like(tensor) if ".lora_B." in name else tensor for name, tensor in params.items()}
    elif case == "output_only":
        params = {name: tensor for name, tensor in params.items() if ".to_out.0." in name}
        config = {**config, "target_modules": ["to_out.0"]}
    manager.set_active_adapter(
        OmniTensorLoRARequest(
            lora_name="boogu", lora_int_id=1, lora_path="in-memory", lora_tensors=params, peft_config=config
        )
    )
    assert manager._active_adapter_id == 1
    if case == "unmapped":
        assert "transformer.attn.to_out.0" in manager._registered_adapters[1].loras
        assert manager._lora_modules and "transformer.attn.to_out" not in manager._lora_modules
        return

    mapped, _ = manager.pipeline.map_lora_update_to_engine(params, config)
    assert "transformer.attn.to_out" in manager._lora_modules
    assert len(manager._lora_modules) == len(params) // 2
    for name, layer in manager._lora_modules.items():
        expected_a = mapped[f"{name}.lora_A.weight"]
        expected_b = mapped[f"{name}.lora_B.weight"] * 2
        torch.testing.assert_close(layer.lora_a_stacked[0][0, 0, :4], expected_a, rtol=0, atol=0)
        torch.testing.assert_close(layer.lora_b_stacked[0][0, 0, :, :4], expected_b, rtol=0, atol=0)

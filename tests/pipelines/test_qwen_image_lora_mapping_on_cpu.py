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
"""CPU regressions for Qwen-Image and Boogu-Image live LoRA binding."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(
    ("architecture", "algorithm"),
    [
        ("QwenImagePipeline", "flow_grpo"),
        ("QwenImagePipeline", "dual_grpo"),
        ("QwenImagePipeline", "mix_grpo"),
        ("QwenImagePipeline", "diffusion_nft"),
        ("QwenImagePipeline", "dpo"),
        ("QwenImageEditPlusPipeline", "flow_grpo"),
        ("BooguImagePipeline", "flow_grpo"),
    ],
)
@pytest.mark.parametrize("container", [list, tuple, set])
def test_registered_pipelines_map_keys_and_targets_without_mutating_inputs(architecture, algorithm, container):
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    pipeline = VllmOmniPipelineBase.get_class(architecture, algorithm)
    tensor = torch.ones(2, 3)
    prefix = "transformer.transformer_blocks.0.attn."
    tensors = {f"{prefix}to_out.0.lora_A.weight": tensor, f"{prefix}to_q.lora_B.weight": tensor}
    targets = container(["to_out.0", "to_q", "transformer_blocks.0.attn.to_out.0", "img_mlp.net.0.proj"])
    config = {"target_modules": targets, "r": 4, "lora_alpha": 8}
    mapped, mapped_config = pipeline.map_lora_update_to_engine(tensors, config)
    assert list(mapped) == [f"{prefix}to_out.lora_A.weight", f"{prefix}to_q.lora_B.weight"]
    assert all(value is tensor for value in mapped.values())
    assert set(mapped_config["target_modules"]) == {
        "to_out",
        "to_q",
        "transformer_blocks.0.attn.to_out",
        "img_mlp.net.0.proj",
    }
    assert config["target_modules"] is targets
    assert "to_out.0" in targets
    assert f"{prefix}to_out.0.lora_A.weight" in tensors
    assert mapped_config["r"] == 4 and mapped_config["lora_alpha"] == 8
    remapped, reconfig = pipeline.map_lora_update_to_engine(mapped, mapped_config)
    assert list(remapped) == list(mapped)
    assert reconfig == mapped_config


@pytest.mark.parametrize("wrapped", [False, True])
def test_lora_rejects_colliding_names(wrapped):
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    key = "transformer.transformer_blocks.0.attn.to_out"
    other = key.replace("transformer.", "transformer.base_model.model.", 1) if wrapped else key + ".0"
    with pytest.raises(ValueError, match="Duplicate .*LoRA tensor"):
        QwenImagePipelineWithLogProb.map_lora_update_to_engine(
            {f"{other}.lora_A.weight": torch.ones(4, 32), f"{key}.lora_A.weight": torch.zeros(4, 32)}, {}
        )


_QWEN_TARGETS = [
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
_BOOGU_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "img_to_q",
    "img_to_k",
    "img_to_v",
    "img_out",
    "instruct_to_q",
    "instruct_to_k",
    "instruct_to_v",
    "instruct_out",
    "feed_forward.linear_1",
    "feed_forward.linear_2",
    "feed_forward.linear_3",
    "img_feed_forward.linear_1",
    "img_feed_forward.linear_2",
    "img_feed_forward.linear_3",
]


def _export_qwen_actor(monkeypatch):
    from diffusers import QwenImageTransformer2DModel
    from peft import LoraConfig, get_peft_model

    from tests.workers.test_diffusers_fsdp_merged_lora_on_cpu import _make_engine, _patch_sync_helpers

    model = QwenImageTransformer2DModel(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=32,
        joint_attention_dim=32,
        axes_dims_rope=(8, 12, 12),
    )
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=_QWEN_TARGETS))
    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters()):
            if ".lora_" in name:
                param.fill_((index + 1) / 128)
    _patch_sync_helpers(monkeypatch)
    engine = _make_engine(model, lora_config={})
    params, config = engine.get_per_tensor_param(base_sync_done=True)
    params = dict(params)
    assert len(params) == 24
    return params, config


def _boogu_actor_tensors(transformer):
    from vllm_omni.diffusion.lora.utils import _match_target_modules

    tensors = {}
    # Use real runtime projection sizes, but the actor's diffusers module names.
    for name, module in transformer.named_modules():
        actor_name = name + ".0" if name.endswith(".to_out") else name
        if _match_target_modules(actor_name, _BOOGU_TARGETS):
            out_features, in_features = module.weight.shape
            prefix = f"transformer.{actor_name}"
            tensors[f"{prefix}.lora_A.weight"] = torch.full((4, in_features), 0.125)
            tensors[f"{prefix}.lora_B.weight"] = torch.full((out_features, 4), 0.25)
    assert len(tensors) == 88
    assert sum(".to_out.0.lora_A." in name for name in tensors) == 6
    return tensors, {"r": 4, "lora_alpha": 8, "target_modules": _BOOGU_TARGETS}


@pytest.fixture(params=["qwen", "boogu"])
def runtime_manager(monkeypatch, request):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer as boogu
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen

    from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(qwen, "Attention", lambda **_: torch.nn.Identity())
    monkeypatch.setattr(boogu, "Attention", lambda **_: torch.nn.Identity())
    # Restore the global hijack even when a test fails.
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        if request.param == "qwen":
            transformer = qwen.QwenImageTransformer2DModel(
                OmniDiffusionConfig(),
                num_layers=1,
                num_attention_heads=1,
                attention_head_dim=32,
                joint_attention_dim=32,
                axes_dims_rope=(8, 12, 12),
            )
            transformer.load_weights([])  # Installs the real QKV stacked-parameter mapping.
            pipeline_class = QwenImagePipelineWithLogProb
            tensors, config = _export_qwen_actor(monkeypatch)
            assert isinstance(transformer.transformer_blocks[0].attn.to_out, qwen.RowParallelLinear)
        else:
            transformer = boogu.BooguImageTransformer2DModel(
                OmniDiffusionConfig(
                    tf_model_config=TransformerConfig.from_dict(
                        {
                            "hidden_size": 32,
                            "num_layers": 2,
                            "num_double_stream_layers": 1,
                            "num_refiner_layers": 1,
                            "num_attention_heads": 1,
                            "num_kv_heads": 1,
                            "multiple_of": 32,
                            "axes_dim_rope": (8, 12, 12),
                            "axes_lens": (16, 16, 16),
                            "instruction_feature_configs": {
                                "instruction_feat_dim": 32,
                                "reduce_type": "mean",
                                "num_instruction_feature_layers": 1,
                            },
                        }
                    )
                )
            )
            pipeline_class = BooguImagePipelineWithLogProb
            tensors, config = _boogu_actor_tensors(transformer)
            assert isinstance(transformer.single_stream_layers[0].attn.to_out, boogu.RowParallelLinear)
            assert isinstance(transformer.double_stream_layers[0].img_instruct_attn.to_out, boogu.ReplicatedLinear)
        pipeline = object.__new__(pipeline_class)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        pipeline._validate_diffusion_lora_binding = Mock(wraps=pipeline._validate_diffusion_lora_binding)
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        yield manager, tensors, config


def test_unmapped_output_projection_is_silently_unbound(runtime_manager, monkeypatch):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = runtime_manager
    # Isolate the output-projection mismatch from PEFT wrapper prefixes.
    params = {name.replace("transformer.base_model.model.", "transformer.", 1): t for name, t in params.items()}
    monkeypatch.setattr(manager.pipeline, "map_lora_update_to_engine", lambda tensors, config: (tensors, config))
    validator = Mock()
    monkeypatch.setattr(manager.pipeline, "_validate_diffusion_lora_binding", validator)
    manager.set_active_adapter(
        OmniTensorLoRARequest(
            lora_name="unmapped",
            lora_int_id=1,
            lora_path="in-memory",
            lora_tensors=params,
            peft_config=config,
        )
    )
    assert manager._active_adapter_id == 1
    missing = {name.removesuffix(".lora_A.weight") for name in params if ".to_out.0.lora_A." in name}
    loaded = manager._registered_adapters[1].loras
    bound = validator.call_args.kwargs["bound_lora_names"]
    assert set(loaded) - bound == missing
    assert bound and missing
    for name, layer in manager._lora_modules.items():
        if name.endswith(".to_out"):
            for tensor in (*layer.lora_a_stacked, *layer.lora_b_stacked):
                assert torch.count_nonzero(tensor) == 0


@pytest.mark.parametrize("case", ["full", "partial", "empty", "zero_init", "output_only"])
def test_export_load_bind_activate_contract(runtime_manager, case):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = runtime_manager
    if case == "partial":
        params["transformer.missing.to_out.lora_A.weight"] = torch.ones(4, 32)
        params["transformer.missing.to_out.lora_B.weight"] = torch.ones(32, 4)
    elif case == "empty":
        params = {}
    elif case == "zero_init":
        params = {name: torch.zeros_like(tensor) if ".lora_B." in name else tensor for name, tensor in params.items()}
    elif case == "output_only":
        params = {name: tensor for name, tensor in params.items() if ".to_out.0." in name}
        config = {**config, "target_modules": ["to_out.0"]}
    request = OmniTensorLoRARequest(
        lora_name="actor",
        lora_int_id=1,
        lora_path="in-memory",
        lora_tensors=params,
        peft_config=config,
    )
    if case in {"partial", "empty"}:
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
    assert len(validator.call_args.kwargs["bound_lora_names"]) == len(params) // 2
    mapped, _ = manager.pipeline.map_lora_update_to_engine(params, config)
    assert any(name.endswith(".to_out") for name in manager._lora_modules)
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

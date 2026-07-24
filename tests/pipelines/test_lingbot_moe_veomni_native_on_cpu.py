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

"""CPU contract tests for the VeOmni-native LingBot-Video MoE transformer.

The tiny model mirrors ``robbyant/lingbot-video-moe-30b-a3b``'s real
``transformer/config.json`` exactly, shrinking only size dimensions.  All MoE
routing semantics stay at production values: 128 routed experts, top-8,
sigmoid scores, group-limited top-k (n_group=4, topk_group=2),
norm_topk_prob, routed_scaling_factor=2.5, one shared expert, and every
layer sparse (decoder_sparse_step=1, no mlp_only_layers).  head_dim stays
128 = sum(axes_dims), so the real 30B rope axes are kept verbatim.
"""

import json
import os

import pytest
import torch

pytest.importorskip("veomni", reason="VeOmni engine backend not installed")
pytest.importorskip("lingbot_video", reason="optional lingbot-video dependency not installed")

from verl_omni.models.veomni import register_veomni_models  # noqa: E402

register_veomni_models("LingBotVideoTransformer3DModel")

from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY  # noqa: E402

from verl_omni.models.veomni.lingbot_video.configuration_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModelConfig,
)
from verl_omni.models.veomni.lingbot_video.modeling_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)

_ARCH = "LingBotVideoTransformer3DModel"

# The real robbyant/lingbot-video-moe-30b-a3b transformer/config.json
# (_class_name/_diffusers_version dropped).  Source of truth for the
# non-shrunk fields below and for the config round-trip test.
MOE_30B_CONFIG = {
    "axes_dims": [32, 48, 48],
    "axes_lens": [4096, 512, 512],
    "decoder_sparse_step": 1,
    "depth": 48,
    "freq_dim": 256,
    "hidden_size": 2048,
    "in_channels": 16,
    "intermediate_size": 6144,
    "mlp_only_layers": [],
    "moe_intermediate_size": 768,
    "n_group": 4,
    "n_shared_experts": 1,
    "norm_eps": 1e-06,
    "norm_topk_prob": True,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "out_bias": True,
    "out_channels": 16,
    "patch_embed_bias": True,
    "patch_size": [1, 2, 2],
    "qkv_bias": False,
    "rope_theta": 256.0,
    "routed_scaling_factor": 2.5,
    "score_func": "sigmoid",
    "text_dim": 2560,
    "timestep_mlp_bias": True,
    "topk_group": 2,
}

# Only size dims are shrunk; every routing/moe semantic field is the 30B value.
# head_dim = hidden_size / num_attention_heads = 128 = sum(axes_dims) — the
# real 30B head geometry, so rope axes stay verbatim.
_TINY_OVERRIDES = {
    "hidden_size": 128,
    "num_attention_heads": 1,
    "depth": 2,
    "intermediate_size": 96,
    "text_dim": 48,
    "freq_dim": 32,
    "moe_intermediate_size": 16,
}
_SIZE_ONLY_FIELDS = set(_TINY_OVERRIDES)


def tiny_moe_config(**overrides) -> LingBotVideoTransformer3DModelConfig:
    kwargs = {**MOE_30B_CONFIG, **_TINY_OVERRIDES, **overrides}
    return LingBotVideoTransformer3DModelConfig(**kwargs)


def build_tiny_moe_model(seed: int = 0) -> LingBotVideoTransformer3DModel:
    torch.manual_seed(seed)
    model = LingBotVideoTransformer3DModel(tiny_moe_config())
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    # Non-trivial correction bias so selection (bias-added) and gating
    # (bias-free) actually diverge, covering the DeepSeek-V3 asymmetry.
    for block in model.blocks:
        block.ffn.router.e_score_correction_bias.uniform_(-0.05, 0.05)
    return model


def tiny_inputs(batch_size: int = 1, seed: int = 7):
    torch.manual_seed(seed)
    x = torch.randn(batch_size, 16, 3, 8, 8)
    timestep = torch.full((batch_size,), 500.0)
    text = torch.randn(batch_size, 7, _TINY_OVERRIDES["text_dim"])
    mask = torch.ones(batch_size, 7)
    return {
        "hidden_states": x,
        "timestep": timestep,
        "encoder_hidden_states": text,
        "encoder_attention_mask": mask,
        "return_dict": False,
    }


def test_tiny_config_preserves_all_30b_moe_semantics():
    config = tiny_moe_config()
    for field, value in MOE_30B_CONFIG.items():
        if field in _SIZE_ONLY_FIELDS:
            continue
        actual = getattr(config, field)
        actual = list(actual) if isinstance(actual, tuple) else actual
        assert actual == value, f"{field}: tiny={actual!r} != 30b={value!r}"
    head_dim = config.hidden_size // config.num_attention_heads
    assert head_dim == sum(config.axes_dims) == 128  # real 30B head geometry


def test_veomni_registry_resolves_arch_from_diffusers_class_name(tmp_path):
    assert _ARCH in MODEL_CONFIG_REGISTRY.valid_keys()
    assert _ARCH in MODELING_REGISTRY.valid_keys()
    assert MODELING_REGISTRY[_ARCH](None) is LingBotVideoTransformer3DModel

    # A diffusers-style config.json (``_class_name``, no ``model_type``) must
    # resolve through veomni.build_config to our config class — this is the
    # exact path BaseTrainer._build_model takes for the real checkpoint.
    from veomni.models.auto import build_config

    config_dir = tmp_path / "transformer"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps({"_class_name": _ARCH, "_diffusers_version": "0.37.0", **MOE_30B_CONFIG, **_TINY_OVERRIDES})
    )
    config = build_config(os.fspath(config_dir))
    assert isinstance(config, LingBotVideoTransformer3DModelConfig)
    assert config.num_experts == 128 and config.num_experts_per_tok == 8
    assert config.n_group == 4 and config.topk_group == 2
    assert config.routed_scaling_factor == 2.5 and config.n_shared_experts == 1


def test_config_to_dict_round_trips_class_name():
    serialized = tiny_moe_config().to_dict()
    assert serialized["_class_name"] == _ARCH
    assert "dtype" not in serialized


def test_state_dict_keys_match_30b_checkpoint_layout():
    model = build_tiny_moe_model()
    keys = set(model.state_dict().keys())

    config = model.config
    for layer in range(config.depth):
        # decoder_sparse_step=1 + empty mlp_only_layers → every block is MoE.
        prefix = f"blocks.{layer}.ffn"
        for expected in (
            f"{prefix}.router.weight",
            f"{prefix}.router.e_score_correction_bias",
            f"{prefix}.experts.w1",
            f"{prefix}.experts.w2",
            f"{prefix}.experts.w3",
            f"{prefix}.shared_experts.gate_proj.weight",
            f"{prefix}.shared_experts.up_proj.weight",
            f"{prefix}.shared_experts.down_proj.weight",
        ):
            assert expected in keys, expected
        # MoE blocks must not carry a dense FFN.
        assert f"{prefix}.gate_proj.weight" not in keys

    experts = model.blocks[0].ffn.experts
    E, MI, H = config.num_experts, config.moe_intermediate_size, config.hidden_size
    assert experts.w1.shape == (E, MI, H)
    assert experts.w2.shape == (E, H, MI)
    assert experts.w3.shape == (E, MI, H)
    router = model.blocks[0].ffn.router
    assert router.weight.shape == (E, H)
    assert router.e_score_correction_bias.shape == (E,)


def test_forward_matches_pip_reference_exactly():
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel as PipModel

    wrapper = build_tiny_moe_model()
    pip = PipModel(**wrapper.config.to_diffuser_dict())
    pip.load_state_dict(wrapper.state_dict(), strict=True)

    inputs = tiny_inputs()
    wrapper.eval()
    pip.eval()
    with torch.no_grad():
        out_wrapper = wrapper(**inputs)[0]
        out_pip = pip(**inputs)[0]

    assert out_wrapper.shape == (1, 16, 3, 8, 8)
    assert torch.equal(out_wrapper, out_pip), "wrapper forward diverges from the pip reference"


def test_router_math_follows_deepseek_v3_semantics():
    """Sigmoid + bias-added selection / bias-free gating + group top-k + scale."""
    model = build_tiny_moe_model()
    router = model.blocks[0].ffn.router
    config = model.config

    tokens = torch.randn(11, config.hidden_size)
    top_indices, top_scores, logits, scores, scores_for_choice = router(tokens)

    assert top_indices.shape == (11, config.num_experts_per_tok)
    assert torch.equal(scores, logits.sigmoid())
    assert torch.allclose(scores_for_choice, scores + router.e_score_correction_bias.unsqueeze(0))

    # Selected experts must come only from the topk_group best groups
    # (group score = sum of that group's top-2 bias-added scores).
    experts_per_group = config.num_experts // config.n_group
    grouped = scores_for_choice.view(11, config.n_group, experts_per_group)
    group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
    allowed_groups = group_scores.topk(config.topk_group, dim=-1)[1]
    selected_groups = top_indices // experts_per_group
    for token in range(11):
        assert set(selected_groups[token].tolist()) <= set(allowed_groups[token].tolist())

    # Gating weights gather from the bias-FREE scores, then normalize + scale.
    gathered = scores.gather(1, top_indices)
    expected = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) * config.routed_scaling_factor
    assert torch.allclose(top_scores, expected.to(top_scores.dtype))
    assert top_scores.sum(dim=-1).allclose(
        torch.full((11,), config.routed_scaling_factor), atol=1e-5
    )  # norm_topk_prob=True


def test_parallel_plan_covers_exactly_the_grouped_experts():
    from torch.distributed._tensor import Shard
    from veomni.distributed.utils import check_fqn_match

    model = build_tiny_moe_model()
    plan = model.get_parallel_plan()
    assert set(plan.extra_parallel_plan.keys()) == {"ep"}
    ep_plan = plan.extra_parallel_plan["ep"]
    assert all(isinstance(shard, Shard) and shard.dim == 0 for shard in ep_plan.values())

    matched = {fqn for fqn, _ in model.named_parameters() if any(check_fqn_match(pattern, fqn) for pattern in ep_plan)}
    expected = {f"blocks.{layer}.ffn.experts.{w}" for layer in range(model.config.depth) for w in ("w1", "w2", "w3")}
    assert matched == expected, "EP plan must cover all grouped experts and nothing else"

    # Router / shared experts stay replicated; 128 experts divide any sane EP size.
    assert plan.extra_parallel_fsdp_no_shard_module["ep"] == {"blocks.*.ffn.experts"}
    for ep_size in (2, 4, 8, 16):
        assert model.config.num_experts % ep_size == 0


def test_fp32_keep_list_covers_router_and_export_exemption():
    model = build_tiny_moe_model()
    fp32_fragments = frozenset(list(model._keep_in_fp32_modules or []) + list(model._keep_in_fp32_modules_strict or []))

    def keeps_source_dtype(name: str) -> bool:
        # Mirrors VeOmniDiffusionEngine.get_per_tensor_param's exemption rule.
        return any(fragment in name.split(".") for fragment in fp32_fragments)

    assert keeps_source_dtype("blocks.0.ffn.router.weight")
    assert keeps_source_dtype("blocks.0.ffn.router.e_score_correction_bias")
    assert keeps_source_dtype("blocks.0.norm1.weight")
    assert keeps_source_dtype("time_embedder.linear_1.weight")
    assert not keeps_source_dtype("blocks.0.ffn.experts.w1")
    assert not keeps_source_dtype("blocks.0.attn.to_q.weight")

    from verl_omni.models.veomni.lingbot_video.modeling_lingbot_video import LingBotVideoRouter

    assert model.get_ignore_modules_in_mixed_precision() == (LingBotVideoRouter,)


def test_bf16_cast_preserves_fp32_router_via_custom_to():
    model = build_tiny_moe_model().to(torch.bfloat16)
    router = model.blocks[0].ffn.router
    assert router.weight.dtype == torch.float32
    assert router.e_score_correction_bias.dtype == torch.float32
    assert model.blocks[0].ffn.experts.w1.dtype == torch.bfloat16
    assert model.blocks[0].attn.to_q.weight.dtype == torch.bfloat16

    # In training FSDP2's ``cast_forward_inputs=True`` lowers activations to
    # the bf16 param dtype before the model sees them; mirror that here.
    # ``timestep`` stays fp32 — the fp32 keep-list retains the time embedder.
    inputs = tiny_inputs()
    inputs["hidden_states"] = inputs["hidden_states"].to(torch.bfloat16)
    inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].to(torch.bfloat16)
    out = model(**inputs)[0]
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_backward_reaches_experts_router_and_shared_experts():
    model = build_tiny_moe_model()
    model.train()
    out = model(**tiny_inputs())[0]
    out.float().pow(2).mean().backward()

    for layer in range(model.config.depth):
        ffn = model.blocks[layer].ffn
        for name, grad in (
            (f"blocks.{layer}.experts.w1", ffn.experts.w1.grad),
            (f"blocks.{layer}.experts.w2", ffn.experts.w2.grad),
            (f"blocks.{layer}.experts.w3", ffn.experts.w3.grad),
            (f"blocks.{layer}.router.weight", ffn.router.weight.grad),
            (f"blocks.{layer}.shared_experts.gate", ffn.shared_experts.gate_proj.weight.grad),
        ):
            assert grad is not None, f"{name} got no grad"
            assert torch.isfinite(grad).all(), f"{name} grad not finite"
        # top-8 of 128 with random routing: some experts idle, but not all.
        assert ffn.experts.w1.grad.abs().sum() > 0
    # The correction bias is a buffer: never trained, exactly like DeepSeek-V3.
    assert not model.blocks[0].ffn.router.e_score_correction_bias.requires_grad


def test_meta_init_then_materialize_matches_checkpoint_loading_path():
    """VeOmni builds on meta then loads weights; _init_weights must cover every param."""
    config = tiny_moe_config()
    with torch.device("meta"):
        model = LingBotVideoTransformer3DModel(config)
    assert next(model.parameters()).is_meta

    model.to_empty(device="cpu")
    for module in model.modules():
        model._init_weights(module)
    for name, parameter in model.named_parameters():
        assert not parameter.is_meta, name
        assert torch.isfinite(parameter).all(), name

    out = model(**tiny_inputs())[0]
    assert out.shape == (1, 16, 3, 8, 8)
    assert torch.isfinite(out).all()


def test_save_pretrained_round_trips_through_pip_loader(tmp_path):
    model = build_tiny_moe_model()
    save_dir = tmp_path / "transformer"
    model.save_pretrained(os.fspath(save_dir))

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["_class_name"] == _ARCH
    assert saved_config["num_experts"] == 128

    reloaded = LingBotVideoTransformer3DModel.from_pretrained(os.fspath(save_dir))
    assert type(reloaded).__name__ == "LingBotVideoTransformer3DModel"
    original_state = model.state_dict()
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, original_state[name]), name

    inputs = tiny_inputs()
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(model(**inputs)[0], reloaded(**inputs)[0])


def test_weight_export_names_and_dtypes_match_rollout_contract():
    """Emulate get_per_tensor_param on CPU: names, shapes, and dtype policy."""
    model = build_tiny_moe_model().to(torch.bfloat16)
    export_dtype = torch.bfloat16
    fp32_fragments = frozenset(list(model._keep_in_fp32_modules or []) + list(model._keep_in_fp32_modules_strict or []))

    exported = {}
    for name, param in model.state_dict().items():
        tensor = param
        if (
            tensor.is_floating_point()
            and tensor.dtype != export_dtype
            and not any(fragment in name.split(".") for fragment in fp32_fragments)
        ):
            tensor = tensor.to(export_dtype)
        exported[f"transformer.{name}"] = tensor

    # Rollout consumes `transformer.<bare-name>` 1:1 (vllm_omni_rollout_adapter.load_weights).
    assert "transformer.blocks.0.ffn.experts.w1" in exported
    assert "transformer.blocks.0.ffn.router.weight" in exported
    assert exported["transformer.blocks.0.ffn.experts.w1"].dtype == torch.bfloat16
    # fp32-sensitive tensors ride through the sync uncast.
    assert exported["transformer.blocks.0.ffn.router.weight"].dtype == torch.float32
    assert exported["transformer.blocks.0.ffn.router.e_score_correction_bias"].dtype == torch.float32
    assert exported["transformer.blocks.0.norm1.weight"].dtype == torch.float32


def test_moe_engine_config_guard_rejects_eager_ep():
    from verl_omni.workers.config import VeOmniDiffusionEngineConfig

    with pytest.raises(ValueError, match="fused MoE kernel"):
        VeOmniDiffusionEngineConfig(expert_parallel_size=2, moe_implementation="eager")
    VeOmniDiffusionEngineConfig(expert_parallel_size=2, moe_implementation="fused_triton")
    VeOmniDiffusionEngineConfig(expert_parallel_size=1, moe_implementation="eager")

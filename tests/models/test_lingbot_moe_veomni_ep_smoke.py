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

"""2-GPU expert-parallel smoke test for the VeOmni-native LingBot MoE DiT.

Run:
    torchrun --nproc_per_node=2 tests/gpu_smoke/test_lingbot_moe_veomni_ep_smoke.py

Covers, on a tiny MoE config:
1. VeOmni parallel-state init with ep_size=2 + FSDP2 parallelization via the
   real ``build_parallelize_model`` path (parallel plan applied).
2. Expert tensors sliced to [E/ep, ...] locally, non-expert params replicated.
3. Forward through the fused_triton EP path (token all-to-all) matches a
   single-process eager reference bitwise-tolerantly.
4. Weight export: restoring the EP dim + ``full_tensor()`` reproduces the
   original full [E, ...] expert tensors (the ``get_per_tensor_param`` path).
"""

import os

import torch
import torch.distributed as dist


def log(rank, *args):
    print(f"[rank{rank}]", *args, flush=True)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "run with --nproc_per_node=2"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")

    from veomni.arguments import OpsImplementationConfig
    from veomni.distributed import parallel_state
    from veomni.ops import apply_ops_config

    from verl_omni.models.veomni import register_veomni_models

    register_veomni_models("LingBotVideoTransformer3DModel")
    from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="fused_triton",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )
    apply_ops_config(ops_config)

    config_cls = MODEL_CONFIG_REGISTRY["LingBotVideoTransformer3DModel"]()
    model_cls = MODELING_REGISTRY["LingBotVideoTransformer3DModel"](None)

    # Bind the fused MoE kernel exactly like build_foundation_model does.
    import sys

    from veomni.models.auto import _bind_veomni_ops

    _bind_veomni_ops(sys.modules[model_cls.__module__], ops_config)

    num_experts = 8
    tiny = config_cls(
        hidden_size=128,
        num_attention_heads=1,
        axes_dims=(64, 32, 32),
        axes_lens=(64, 16, 16),
        depth=2,
        intermediate_size=256,
        text_dim=64,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=2.5,
        n_shared_experts=1,
    )

    # Reference model with deterministic weights, identical on both ranks.
    # bf16 like real training: the fused MoE kernel requires bf16/fp16
    # activations.  Force ALL params/buffers to bf16 (bypassing the pip
    # custom ``.to`` fp32 keep-list) so the reference and the meta-built EP
    # model hold bit-identical weights and routing cannot diverge.
    torch.manual_seed(1234)
    reference = model_cls(tiny).cuda()
    for p in reference.parameters():
        torch.nn.init.normal_(p, std=0.02)
    for p in reference.parameters():
        p.data = p.data.to(torch.bfloat16)
    for _, buf in reference.named_buffers():
        buf.data = buf.data.to(torch.bfloat16)
    full_state = {k: v.detach().clone() for k, v in reference.state_dict().items()}

    B, C, T, H, W = 1, 16, 3, 8, 8
    torch.manual_seed(7)
    x = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
    t = torch.full((B,), 500.0, device="cuda")
    txt = torch.randn(B, 7, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, 7, device="cuda")

    # Reference forward through the pip eager grouped_mm path (full weights,
    # no parallel-state involvement).  EP is process-global once initialized,
    # so the fused kernel would otherwise take the all-to-all branch even for
    # this unsharded model.
    from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

    import verl_omni.models.veomni.lingbot_video.modeling_lingbot_video as lingbot_modeling

    reference.eval()
    LingBotVideoSparseMoeBlock._run_selected_experts = lingbot_modeling._original_run_selected_experts
    try:
        with torch.no_grad():
            ref_out = reference(
                hidden_states=x, timestep=t, encoder_hidden_states=txt, encoder_attention_mask=mask, return_dict=False
            )[0]
    finally:
        lingbot_modeling.apply_veomni_lingbot_video_moe_patch()
    log(rank, "reference forward done (pip eager grouped_mm)", tuple(ref_out.shape))

    parallel_state.init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=1,
        dp_shard_size=world_size,
        extra_parallel_sizes=(2,),
        ulysses_size=1,
        dp_mode="fsdp2",
    )
    ps = parallel_state.get_parallel_state()
    assert ps.ep_enabled and ps.ep_size == 2, (ps.ep_enabled, ps.ep_size)

    # Parallelized model through the real VeOmni path: meta init + weight
    # loading from disk, which exercises ``parallel_plan.shard_tensor``'s EP
    # slicing exactly as production ``BaseTrainer._build_model`` does.
    import tempfile

    from safetensors.torch import save_file

    shared_dir = os.environ.get("LINGBOT_EP_SMOKE_TMPDIR", tempfile.gettempdir())
    weights_dir = os.path.join(shared_dir, "lingbot_ep_smoke_weights")
    if rank == 0:
        os.makedirs(weights_dir, exist_ok=True)
        save_file(
            {k: v.cpu().contiguous() for k, v in full_state.items()}, os.path.join(weights_dir, "model.safetensors")
        )
    dist.barrier()

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = model_cls(tiny)
    finally:
        torch.set_default_dtype(default_dtype)

    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model

    # Same mixed-precision layout as the production engine: fp32 master
    # weights (upcast), bf16 param casts at forward, fp32 reduce.
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=weights_dir,
        mixed_precision=MixedPrecisionConfig(
            enable=True,
            param_dtype="bfloat16",
            reduce_dtype="float32",
            cast_forward_inputs=True,
        ),
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=[],
        enable_reshard_after_forward=True,
        broadcast_model_weights_from_rank0=False,
    )

    # 1) expert slicing check
    experts = model.blocks[0].ffn.experts
    local_w1 = experts.w1
    local_shape = local_w1.to_local().shape if hasattr(local_w1, "to_local") else local_w1.shape
    log(rank, "local w1 param type", type(local_w1).__name__, "shape", tuple(local_shape))

    fqn2spec = model._fqn2spec_info
    w1_spec = fqn2spec["blocks.0.ffn.experts.w1"]
    from torch.distributed._tensor import Shard

    assert isinstance(w1_spec.placement, Shard) and w1_spec.placement.dim == 0, w1_spec
    router_spec = fqn2spec["blocks.0.ffn.router.weight"]
    assert not isinstance(router_spec.placement, Shard), router_spec

    # 2) EP forward vs reference
    model.eval()
    with torch.no_grad():
        ep_out = model(
            hidden_states=x, timestep=t, encoder_hidden_states=txt, encoder_attention_mask=mask, return_dict=False
        )[0]
    diff = (ep_out.float() - ref_out.float()).abs().max().item()
    log(rank, f"EP forward vs single-process reference: max abs diff {diff:.3e}")
    assert torch.isfinite(ep_out).all()
    assert diff < 5e-2, f"EP forward diverges: {diff}"

    # 3) weight export path: restore EP dim then full_tensor
    from torch.distributed.tensor import DTensor, Replicate
    from veomni.checkpoint.dcp_checkpointer import restore_extra_parallel_dim

    params = model.state_dict()
    checked = 0
    for name in (
        "blocks.0.ffn.experts.w1",
        "blocks.0.ffn.experts.w2",
        "blocks.1.ffn.experts.w3",
        "blocks.0.ffn.router.weight",
        "blocks.0.attn.to_q.weight",
    ):
        param = params[name]
        spec = fqn2spec.get(name)
        if spec is not None and not isinstance(spec.placement, Replicate):
            param = restore_extra_parallel_dim(
                param, spec.para_fsdp_mesh, spec.para_fsdp_mesh[f"{spec.para_name}_fsdp"]
            )
        tensor = param.full_tensor() if isinstance(param, DTensor) else param
        expected = full_state[name]
        assert tensor.shape == expected.shape, (name, tensor.shape, expected.shape)
        max_diff = (tensor.float().cuda() - expected.float()).abs().max().item()
        assert max_diff == 0.0, (name, max_diff)
        checked += 1
    log(rank, f"weight export round-trip exact for {checked} params (incl. EP-restored experts)")

    dist.barrier()
    if rank == 0:
        print("EP SMOKE TEST PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

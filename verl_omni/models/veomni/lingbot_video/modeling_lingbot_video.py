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

"""VeOmni-native LingBot-Video transformer (Dense and MoE).

This wraps the reference pip ``lingbot_video`` transformer in a
``transformers.PreTrainedModel`` shell so VeOmni's foundation-model loader,
FSDP2 parallelizer, and expert-parallel machinery can drive it, mirroring the
``veomni.models.diffusers.wan_t2v`` wrapper pattern.  The module tree, the
state-dict keys (``blocks.*`` etc.), and the numerics are the pip
implementation's, so checkpoint loading and rollout weight sync need no key
remapping.

Expert parallelism: the pip eager expert paths index the grouped expert
tensors with *global* expert ids, which is wrong once ``ParallelPlan`` slices
``w1/w2/w3`` to ``[E/ep, ...]``.  ``apply_veomni_lingbot_video_moe_patch``
therefore reroutes ``LingBotVideoSparseMoeBlock._run_selected_experts``
through VeOmni's fused grouped-GEMM MoE kernel, whose EP branch performs the
token all-to-all dispatch/combine internally (``group_gemm.py``,
``distributed/moe/moe_layer.py``).  LingBot keeps split ``w1``/``w3`` (gate /
up) tensors, which map onto the kernel's split ``fc1_1_weight`` /
``fc1_2_weight`` inputs — no gate/up fusion, so training, checkpoint, and
rollout keep identical parameter layouts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from veomni.distributed.parallel_plan import ParallelPlan
from veomni.ops.dispatch import OpSlot
from veomni.utils import logging

from .configuration_lingbot_video import LingBotVideoTransformer3DModelConfig

# The reference implementation.  Kept as the single source of truth for the
# architecture; this module adds only the VeOmni-facing shell and EP dispatch.
from lingbot_video.transformer_lingbot_video import (  # isort: skip
    LINGBOT_VIDEO_FP32_MODULES,
    LingBotVideoBlock,
    LingBotVideoGroupedExperts,
    LingBotVideoRouter,
    LingBotVideoSparseMoeBlock,
    LingBotVideoTransformer3DModel as _LingBotVideoTransformer3DModel,
)

logger = logging.get_logger(__name__)

# Bound by ``veomni.models.auto._bind_veomni_ops`` from
# ``ops_implementation.moe_implementation`` when this modeling module is
# resolved through MODELING_REGISTRY.  ``eager`` leaves the slot unbound and
# the pip expert paths untouched; any ``fused_*`` value also installs the
# module-level ``veomni.ops.kernels.moe._fused_moe_forward`` pointer that the
# patched expert dispatch below relies on.
veomni_moe_experts_forward = OpSlot("moe_experts", "standard")

_original_run_selected_experts = LingBotVideoSparseMoeBlock._run_selected_experts


def _veomni_grouped_experts_forward(
    self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
):
    """Fused grouped-GEMM forward for ``LingBotVideoGroupedExperts``.

    LingBot's SwiGLU maps onto the kernel's split fc1 inputs as ``fc1_1 = w1``
    (gate, silu), ``fc1_2 = w3`` (up), ``fc2 = w2`` (down).  ``num_experts``
    stays the GLOBAL count — under EP the kernel derives the local count from
    the EP group size and runs the token all-to-all internally.  Routing
    weights are applied in the kernel's combine step, matching the pip
    restore-path semantics (unweighted expert outputs, weighted top-k sum).
    """
    from veomni.ops.kernels.moe import fused_moe_forward

    return fused_moe_forward(
        num_experts=self.num_experts,
        routing_weights=top_k_weights.to(hidden_states.dtype),
        selected_experts=top_k_index,
        hidden_states=hidden_states,
        fc1_1_weight=self.w1,
        fc1_2_weight=self.w3,
        fc2_weight=self.w2,
    )


def _veomni_run_selected_experts(self, tokens: torch.Tensor, top_scores: torch.Tensor, top_indices: torch.Tensor):
    """EP-capable replacement for ``LingBotVideoSparseMoeBlock._run_selected_experts``.

    With a fused kernel bound, dispatch through ``self.experts(...)`` — the
    pip experts module is a bare parameter container, but VeOmni FSDP2-wraps
    it separately (EP no-shard module), and only a real ``__call__`` triggers
    the FSDP2 pre-forward hook that unshards the expert weights and applies
    the bf16 mixed-precision cast.  Reading ``self.experts.w1`` directly
    would hand fp32 master shards to the kernel.
    """
    fused_bound = veomni_moe_experts_forward.use_non_eager_impl
    fused_dtype_ok = tokens.dtype in (torch.bfloat16, torch.float16)
    if fused_bound and fused_dtype_ok:
        return self.experts(tokens, top_indices, top_scores)

    from veomni.distributed.parallel_state import get_parallel_state

    parallel_state = get_parallel_state()
    if parallel_state is not None and getattr(parallel_state, "ep_enabled", False):
        if not fused_bound:
            raise RuntimeError(
                "Expert parallelism requires a fused MoE kernel: the eager LingBot expert paths index "
                "EP-sharded [E/ep, ...] tensors with global expert ids. Set "
                "`moe_implementation=fused_triton` (engine config) when `expert_parallel_size > 1`."
            )
        raise RuntimeError(
            f"Fused MoE kernel requires bf16/fp16 activations under expert parallelism, got {tokens.dtype}. "
            "Enable mixed precision (bf16 params) or disable EP."
        )
    return _original_run_selected_experts(self, tokens, top_scores, top_indices)


def apply_veomni_lingbot_video_moe_patch() -> None:
    """Patch the pip MoE block's expert dispatch with the EP-capable variant (idempotent)."""
    if LingBotVideoSparseMoeBlock._run_selected_experts is _veomni_run_selected_experts:
        return
    LingBotVideoGroupedExperts.forward = _veomni_grouped_experts_forward
    LingBotVideoSparseMoeBlock._run_selected_experts = _veomni_run_selected_experts
    logger.info_rank0("Applied VeOmni EP-capable expert dispatch to LingBotVideoSparseMoeBlock.")


class _LingBotVideoInitShim(_LingBotVideoTransformer3DModel):
    """Breaks the init chain from PreTrainedModel to the pip diffusers model.

    ``PreTrainedModel.__init__``'s ``super().__init__()`` would otherwise
    invoke the pip ``__init__`` with default arguments and build a throwaway
    default model.  The shim intercepts that call; the real model is built
    once, with the checkpoint config, in ``LingBotVideoTransformer3DModel.__init__``.
    """

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)


class LingBotVideoTransformer3DModel(PreTrainedModel, _LingBotVideoInitShim):
    config_class = LingBotVideoTransformer3DModelConfig
    # The pip implementation deliberately does not implement gradient
    # checkpointing (`_supports_gradient_checkpointing = False`); keep the
    # transformers-side flag consistent so `gradient_checkpointing_enable`
    # fails loudly instead of silently doing nothing.
    supports_gradient_checkpointing = False
    _no_split_modules = ["LingBotVideoBlock"]
    _keep_in_fp32_modules = list(LINGBOT_VIDEO_FP32_MODULES)
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    def __init__(self, config: LingBotVideoTransformer3DModelConfig, **kwargs):
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        # Remove VeOmni/transformers-specific kwargs before the diffusers init.
        kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)
        _LingBotVideoTransformer3DModel.__init__(self, **config.to_diffuser_dict())
        self.config: LingBotVideoTransformer3DModelConfig = config
        self.config.tie_word_embeddings = False
        self._install_time_embedder_dtype_hook()

    def _install_time_embedder_dtype_hook(self) -> None:
        """Cast the timestep projection to the embedder weight dtype.

        The pip forward computes ``timestep.float()`` *inside* forward, past
        any FSDP2 ``cast_forward_inputs`` boundary, so under bf16 mixed
        precision the fp32 projection meets bf16 ``time_embedder`` weights and
        ``F.linear`` raises.  Same fix as the FSDP training adapter's
        ``_patch_time_embedder_input_dtype``; a no-op in fp32 runs.
        """

        def _cast_to_weight_dtype(embedder: torch.nn.Module, args: tuple):
            if not args or not isinstance(args[0], torch.Tensor):
                return args
            weight = getattr(getattr(embedder, "linear_1", None), "weight", None)
            if weight is None or args[0].dtype == weight.dtype:
                return args
            return (args[0].to(dtype=weight.dtype), *args[1:])

        self.time_embedder.register_forward_pre_hook(_cast_to_weight_dtype)

    # diffusers ConfigMixin reads/writes ``_internal_dict`` through the
    # ``config`` property; alias it so both the transformers and diffusers
    # halves of the MRO agree on a single PretrainedConfig object.
    @property
    def config(self):
        return self._internal_dict

    @config.setter
    def config(self, value):
        self._internal_dict = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        # verl-omni's FlowGRPO adapter drives this exactly like the pip
        # transformer: kwargs in, `(sample,)` out with `return_dict=False`.
        return _LingBotVideoTransformer3DModel.forward(
            self,
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )

    def get_parallel_plan(self) -> ParallelPlan:
        """Expert-parallel plan: slice the grouped expert tensors on the expert dim.

        Matches the real checkpoint FQNs (``blocks.N.ffn.experts.w{1,2,3}``).
        The router (``weight``/``e_score_correction_bias``) and the shared
        expert stay replicated.  ``ParallelPlan`` derives the FSDP no-shard
        module (``blocks.*.ffn.experts``) from these patterns, and the
        parallelizer then FSDP-shards that module on the ``ep_fsdp`` sub-mesh.
        """
        from torch.distributed._tensor import Shard

        ep_plan = {
            "blocks.*.ffn.experts.w1": Shard(0),
            "blocks.*.ffn.experts.w2": Shard(0),
            "blocks.*.ffn.experts.w3": Shard(0),
        }
        return ParallelPlan(extra_parallel_plan={"ep": ep_plan})

    def get_ignore_modules_in_mixed_precision(self) -> tuple:
        """Keep the MoE router out of the bf16 mixed-precision cast.

        Selection runs sigmoid + grouped top-k on ``F.linear(tokens.float(),
        weight.float())``; a bf16-truncated router weight can flip borderline
        top-k picks relative to the fp32 rollout router, silently biasing the
        FlowGRPO importance ratio.  Router params are tiny (E x H per MoE
        layer), so the fp32 FSDP group is cheap.
        """
        return (LingBotVideoRouter,)

    def _init_weights(self, module):
        """Init for params missing from the checkpoint / scratch meta-init."""
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, LingBotVideoRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.zero_()
        elif isinstance(module, LingBotVideoGroupedExperts):
            module.w1.data.normal_(mean=0.0, std=std)
            module.w2.data.normal_(mean=0.0, std=std)
            module.w3.data.normal_(mean=0.0, std=std)
        elif isinstance(module, LingBotVideoBlock):
            module.scale_shift_table.data.normal_(mean=0.0, std=std)
        elif hasattr(module, "weight") and isinstance(getattr(module, "weight", None), nn.Parameter):
            # RMSNorm-style single-weight norms.
            if module.weight.dim() == 1:
                module.weight.data.fill_(1.0)

    def save_pretrained(self, path, **kwargs):
        import copy

        hf_config = copy.deepcopy(self.config)
        # diffusers ConfigMixin serializes ``_internal_dict``; hand it the
        # diffusers-style kwargs so the saved config round-trips with the pip
        # ``from_pretrained`` (same swap as VeOmni's Wan wrapper).
        self.config = self.config.to_diffuser_dict()
        try:
            _LingBotVideoTransformer3DModel.save_pretrained(self, path, **kwargs)
        finally:
            self.config = hf_config

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return _LingBotVideoTransformer3DModel.from_pretrained(path, **kwargs)


__all__ = [
    "LingBotVideoTransformer3DModel",
    "apply_veomni_lingbot_video_moe_patch",
    "veomni_moe_experts_forward",
]

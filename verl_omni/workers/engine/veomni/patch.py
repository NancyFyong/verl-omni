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

from contextlib import contextmanager
from dataclasses import replace
from importlib.metadata import version
from threading import Lock, get_ident
from types import MethodType

import torch

# TODO: Remove this shim, its call sites and tests once the required VeOmni includes
# ByteDance-Seed/VeOmni#1214 and #1236. VeOmni 0.1.12 rejects Hub attention names and Qwen-Image
# ignores attn_implementation. Only Hub varlen backends handle Qwen-Image's text padding correctly,
# so this table is the allowlist except for H3's instance-local attention bridge.
_DIFFUSERS_ATTENTION_BACKENDS = {
    "eager": "native",
    "flash_attention_2_hub": "flash_varlen_hub",
    "flash_attention_3_hub": "_flash_3_varlen_hub",
}


# TODO: Remove this bridge and its call site once the required VeOmni includes
# ByteDance-Seed/VeOmni#1239 and this integration uses its native attention setup.
_MINIMAX_H3_ATTENTION_BACKENDS = {
    "eager": "native",
    "flash_attention_2": "flash",
    "flash_attention_3": "_flash_3",
    "flash_attention_2_hub": "flash_hub",
    "flash_attention_3_hub": "_flash_3_hub",
}


def _veomni_attn_implementation(attn_implementation: str) -> str:
    """Build with eager when the installed VeOmni cannot parse Hub attention names."""
    from veomni.arguments import OpsImplementationConfig

    if attn_implementation.endswith("_hub") and not hasattr(OpsImplementationConfig, "normalize_hub_attention_backend"):
        return "eager"  # Qwen-Image and H3 get the Hub kernel in _apply_attention_backend
    return attn_implementation


def _apply_attention_backend(model: torch.nn.Module, attn_implementation: str) -> None:
    """Apply model-specific attention compatibility while retaining backend validation."""
    if getattr(getattr(model, "config", None), "model_type", None) == "MiniMaxH3DiTModel":
        _apply_minimax_h3_attention_backend(model, attn_implementation)
        return
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageSPAttnProcessor,
    )

    backend = _DIFFUSERS_ATTENTION_BACKENDS.get(attn_implementation)
    if backend is None:
        raise ValueError(
            f"veomni_config.attn_implementation={attn_implementation!r} is not supported by the VeOmni "
            f"diffusion engine; use one of {sorted(_DIFFUSERS_ATTENTION_BACKENDS)}. Local flash_attention_2/3 "
            "map to diffusers varlen kernels that mishandle Qwen-Image's text padding."
        )
    if not any(isinstance(getattr(m, "processor", None), QwenImageSPAttnProcessor) for m in model.modules()):
        # e.g. Wan / LTX select attention inside VeOmni.
        if _veomni_attn_implementation(attn_implementation) != attn_implementation:
            raise ValueError(
                f"veomni_config.attn_implementation={attn_implementation!r} requires a VeOmni release "
                "with Hub attention support; verl-omni only supplies it for Qwen-Image and MiniMax H3."
            )
        return
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    # set_attention_backend also switches diffusers' process-wide default; keep it for other models.
    active_backend = _AttentionBackendRegistry._active_backend
    model.set_attention_backend(backend)
    _AttentionBackendRegistry.set_active_backend(active_backend)


def _minimax_h3_attention_forward(self, x, *, rope_cos, rope_sin, cu_seqlens, max_seqlen=None, use_ulysses=False):
    """Single-sample forward adapted from VeOmni's MiniMaxH3Attention."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import _apply_rope

    if use_ulysses or len(cu_seqlens) != 2:
        raise ValueError("The H3 VeOmni attention bridge requires one sample and Ulysses SP=1.")
    total = x.shape[0]
    q, k, v = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    if rope_cos is not None:
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
    out = dispatch_attention_fn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        scale=self.softmax_scale,
        backend=self._h3_attention_backend,
    )
    return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


def _apply_minimax_h3_attention_backend(module: torch.nn.Module, implementation: str) -> None:
    """Select H3 attention per instance without changing parameters or global dispatch."""
    if hasattr(module, "_load_attention_kernel"):
        return
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _check_attention_backend_requirements,
        _maybe_download_kernel_for_backend,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3Attention

    if implementation not in _MINIMAX_H3_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported H3 VeOmni attention {implementation!r}; use one of {sorted(_MINIMAX_H3_ATTENTION_BACKENDS)}."
        )
    backend = AttentionBackendName(_MINIMAX_H3_ATTENTION_BACKENDS[implementation])
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)
    for layer in module.modules():
        if isinstance(layer, MiniMaxH3Attention):
            layer._h3_attention_backend = backend
            layer.forward = MethodType(_minimax_h3_attention_forward, layer)


# Adapted from ByteDance-Seed/VeOmni#1260. Remove this compatibility path once the
# required VeOmni and this engine use the validated native H3 precision policies.
_MINIMAX_H3_FP32_PROJECTIONS = (
    "video_patch_proj",
    "audio_patch_proj",
    "time_embedder.proj_in",
    "time_embedder.proj_out",
    "final_layer.video_out",
    "final_layer.audio_out",
)
_MINIMAX_H3_PRECISION_LOCK = Lock()


def _minimax_h3_fp32_linear_forward(self, inputs):
    if self.weight.dtype != torch.float32:
        raise RuntimeError("H3 FP32 projection was downcast after precision setup.")
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return torch.nn.functional.linear(inputs.to(self.weight.dtype), self.weight, self.bias)


def _minimax_h3_time_forward(self, timesteps, *, dtype):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3TimeEmbedder

    return MiniMaxH3TimeEmbedder.forward(self, timesteps, dtype=torch.float32)


def _minimax_h3_cast_projection_input(module, args):
    # FSDP's prepended pre-hook has already materialized the compute weights.
    return (args[0].to(module.weight.dtype), *args[1:])


def _minimax_h3_precision_embed(
    self,
    *,
    x,
    audio_x,
    text_embeddings_selected,
    unique_timesteps,
    img_pos,
    audio_pos,
    text_pos,
    refiner_cu_seqlens,
    refiner_max_seqlen,
    seq_len,
    device,
):
    video_embed = self.video_patch_proj(x.view(-1, x.shape[-1]).index_select(0, img_pos))
    audio_embed = self.audio_patch_proj(audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos))
    text_embed = self.condition_proj(text_embeddings_selected.to(device=device))
    text_embed = self.token_refiner(text_embed, cu_seqlens=refiner_cu_seqlens, max_seqlen=refiner_max_seqlen)
    dtype = text_embed.dtype
    embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
    embeddings[text_pos] = text_embed.to(dtype)[: text_pos.shape[0]]
    embeddings[img_pos] = video_embed.to(dtype)[: img_pos.shape[0]]
    embeddings[audio_pos] = audio_embed.to(dtype)[: audio_pos.shape[0]]
    return embeddings, self.time_embedder(unique_timesteps, dtype=dtype)


def _prepare_minimax_h3_precision(model: torch.nn.Module) -> None:
    """Repair H3 precision on the FP32 model before LoRA injection and FSDP2 loading."""
    if getattr(getattr(model, "config", None), "model_type", None) != "MiniMaxH3DiTModel":
        return
    if getattr(model, "_h3_precision_patched", False):
        return
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3AdalnProj

    # Attention support does not imply that the separate precision fix is present.
    # Do not combine a partially backported model with the 0.1.12 FSDP shim.
    native_precision = (
        hasattr(model, "get_fsdp_mixed_precision_policy")
        or hasattr(model, "get_ignore_modules_in_mixed_precision")
        or bool(getattr(model, "_keep_in_fp32_modules", None))
        or bool(getattr(model, "_preserve_fp32_modules_on_export", False))
    )
    if version("veomni") != "0.1.12" or native_precision:
        raise NotImplementedError(
            "The H3 precision shim targets VeOmni 0.1.12; validate the native VeOmni#1260 "
            "model and FSDP2 integration before using a different precision implementation."
        )
    projections = [model.dit.get_submodule(name) for name in _MINIMAX_H3_FP32_PROJECTIONS]
    ordinary_projections = [model.dit.condition_proj]
    ordinary_projections.extend(layer.linear for layer in model.dit.modules() if isinstance(layer, MiniMaxH3AdalnProj))
    for layer in projections + ordinary_projections:
        if type(layer) is not torch.nn.Linear or getattr(layer, "_veomni_keep_in_fp32", False):
            raise NotImplementedError("H3 precision setup requires the original Linear modules, before LoRA/FSDP2.")
    for layer in projections:
        if any(param.dtype != torch.float32 for param in layer.parameters()):
            raise ValueError(
                "H3 precision setup requires FP32 projections before checkpoint loading, not BF16 upcasts."
            )
    for layer in projections:
        layer.forward = MethodType(_minimax_h3_fp32_linear_forward, layer)
    for layer in ordinary_projections:
        layer.register_forward_pre_hook(_minimax_h3_cast_projection_input)
    model.dit.time_embedder.forward = MethodType(_minimax_h3_time_forward, model.dit.time_embedder)
    model.dit._embed = MethodType(_minimax_h3_precision_embed, model.dit)
    model._h3_precision_patched = True


@contextmanager
def _minimax_h3_fsdp_precision(model: torch.nn.Module):
    """Apply 0.1.12 H3 precision policies only during this model's FSDP2 construction."""
    if getattr(getattr(model, "config", None), "model_type", None) != "MiniMaxH3DiTModel":
        yield
        return
    if not getattr(model, "_h3_precision_patched", False):
        raise RuntimeError("Prepare H3 precision before LoRA injection and FSDP2 construction.")
    from torch.distributed.fsdp import FSDPModule
    from veomni.distributed import torch_parallelize
    from veomni.lora.layers import LoraLinear

    target_modules = set(model.modules())
    fp32_projections = tuple(model.dit.get_submodule(name) for name in _MINIMAX_H3_FP32_PROJECTIONS)
    for name, layer in model.named_modules():
        if isinstance(layer, FSDPModule):
            raise RuntimeError("H3 precision policies must be installed before any FSDP2 wrapping.")
        if isinstance(layer, LoraLinear) and name.rsplit(".", 1)[-1] not in {"qkv_proj", "out_proj", "fc1", "fc2"}:
            raise NotImplementedError(
                f"H3 precision compatibility supports LoRA on qkv_proj/out_proj/fc1/fc2 only; got {name!r}."
            )
    if not _MINIMAX_H3_PRECISION_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent or nested H3 FSDP2 precision setup is not supported.")
    original_fully_shard = torch_parallelize.fully_shard
    owner_thread = get_ident()
    sharded_projections = set()
    root_sharded = False

    def shard_fp32(layer, kwargs):
        if layer in sharded_projections:
            return layer
        precision_kwargs = dict(kwargs)
        precision_kwargs.pop("mp_policy", None)
        precision_kwargs["reshard_after_forward"] = False
        result = original_fully_shard(layer, **precision_kwargs)
        sharded_projections.add(layer)
        return result

    def fully_shard_with_precision(layer, **kwargs):
        nonlocal root_sharded
        if get_ident() != owner_thread or layer not in target_modules:
            return original_fully_shard(layer, **kwargs)
        policy = kwargs.get("mp_policy")
        if policy is not None and policy.output_dtype not in (None, policy.param_dtype or torch.float32):
            raise NotImplementedError(
                "H3 precision compatibility requires output_dtype=None or the ordinary compute dtype; "
                "a different output dtype would change the next block's hidden-state precision."
            )
        if layer in fp32_projections:
            return shard_fp32(layer, kwargs)
        # Also handle an explicitly selected parent of an FP32 projection, not
        # only the default root, so it cannot take ownership of those parameters.
        descendants = set(layer.modules())
        for projection in fp32_projections:
            if projection in descendants:
                shard_fp32(projection, kwargs)
        if policy is not None:
            ordinary_projection = isinstance(layer, torch.nn.Linear | LoraLinear)
            kwargs = {**kwargs, "mp_policy": replace(policy, cast_forward_inputs=ordinary_projection)}
        result = original_fully_shard(layer, **kwargs)
        if layer is model:
            root_sharded = True
        return result

    try:
        torch_parallelize.fully_shard = fully_shard_with_precision
        yield
        if not root_sharded or len(sharded_projections) != len(fp32_projections):
            raise RuntimeError("H3 precision setup did not wrap the root and all six FP32 projections with FSDP2.")
        for layer in fp32_projections:
            if any(param.dtype != torch.float32 for param in layer.parameters()):
                raise RuntimeError("H3 checkpoint loading did not preserve the FP32 projection parameters.")
    finally:
        torch_parallelize.fully_shard = original_fully_shard
        _MINIMAX_H3_PRECISION_LOCK.release()


def _minimax_h3_export_dtype(model: torch.nn.Module, name: str, default_dtype: torch.dtype) -> torch.dtype:
    """Preserve H3's six FP32 projections on the engine's existing weight-sync path."""
    if getattr(getattr(model, "config", None), "model_type", None) == "MiniMaxH3DiTModel":
        key = name.removeprefix("base_model.model.").removeprefix("dit.")
        module_name, _, param_name = key.rpartition(".")
        if module_name in _MINIMAX_H3_FP32_PROJECTIONS and param_name in {"weight", "bias"}:
            return torch.float32
    return default_dtype

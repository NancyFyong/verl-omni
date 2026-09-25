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

"""Single-sample attention bridge adapted from VeOmni's MiniMaxH3Attention."""

from types import MethodType

# TODO: Remove this bridge and its call site once the required VeOmni includes
# ByteDance-Seed/VeOmni#1239 and this integration uses its native attention setup.
_BACKENDS = {
    "eager": "native",
    "flash_attention_2": "flash",
    "flash_attention_3": "_flash_3",
    "flash_attention_2_hub": "flash_hub",
    "flash_attention_3_hub": "_flash_3_hub",
}


def _attention_forward(self, x, *, rope_cos, rope_sin, cu_seqlens, max_seqlen=None, use_ulysses=False):
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


def configure_veomni_attention(module, implementation: str) -> None:
    """Select H3 attention per instance without changing parameters or global dispatch."""
    if hasattr(module, "_load_attention_kernel"):
        return
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _check_attention_backend_requirements,
        _maybe_download_kernel_for_backend,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3Attention

    if implementation not in _BACKENDS:
        raise ValueError(f"Unsupported H3 VeOmni attention {implementation!r}; use one of {sorted(_BACKENDS)}.")
    backend = AttentionBackendName(_BACKENDS[implementation])
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)
    for layer in module.modules():
        if isinstance(layer, MiniMaxH3Attention):
            layer._h3_attention_backend = backend
            layer.forward = MethodType(_attention_forward, layer)

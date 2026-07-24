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

"""VeOmni-facing config for the LingBot-Video transformer.

The checkpoint's ``transformer/config.json`` is diffusers-style: it carries
``_class_name`` and no ``model_type``, so VeOmni's ``get_model_config``
resolves the registry key from ``_class_name``.  This class re-declares that
string as ``model_type`` so the same key round-trips through the HF config
machinery, mirroring ``veomni``'s Wan/Qwen-Image diffusers configs.
"""

from typing import Optional

from transformers import PretrainedConfig


class LingBotVideoTransformer3DModelConfig(PretrainedConfig):
    model_type = "LingBotVideoTransformer3DModel"

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] = (32, 48, 48),
        axes_lens: tuple[int, int, int] = (8192, 1024, 1024),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        decoder_sparse_step: int = 1,
        mlp_only_layers: tuple[int, ...] = (),
        n_shared_experts: Optional[int] = None,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: float = 1.0,
        **kwargs,
    ):
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.depth = depth
        self.intermediate_size = intermediate_size
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = tuple(axes_lens)
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias
        self.patch_embed_bias = patch_embed_bias
        self.timestep_mlp_bias = timestep_mlp_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = tuple(mlp_only_layers)
        self.n_shared_experts = n_shared_experts
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        super().__init__(**kwargs)

    def to_diffuser_dict(self) -> dict:
        """Kwargs for the pip ``LingBotVideoTransformer3DModel.__init__``.

        Computed from the pip signature (imported lazily so this config module
        stays importable without the optional ``lingbot-video`` dependency).
        """
        import inspect

        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

        signature = inspect.signature(LingBotVideoTransformer3DModel.__init__)
        return {key: getattr(self, key) for key in signature.parameters.keys() if key != "self"}

    def to_dict(self):
        return_dict = super().to_dict()
        return_dict["_class_name"] = "LingBotVideoTransformer3DModel"
        return_dict.pop("dtype", None)
        return return_dict

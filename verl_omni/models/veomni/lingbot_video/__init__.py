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

"""Registers the LingBot-Video transformer into VeOmni's model registries.

VeOmni resolves the transformer class from the checkpoint's
``transformer/config.json`` ``_class_name`` (diffusers configs have no
``model_type``; ``veomni.models.loader.get_model_config`` falls back to
``_class_name``).  Importing this module makes ``build_foundation_model``
able to construct the LingBot DiT, exactly like VeOmni's in-tree
``wan_t2v``/``qwen_image`` diffusers registrations.
"""

from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

_LINGBOT_ARCH = "LingBotVideoTransformer3DModel"


@MODEL_CONFIG_REGISTRY.register(_LINGBOT_ARCH)
def register_lingbot_video_transformer_config():
    from .configuration_lingbot_video import LingBotVideoTransformer3DModelConfig

    return LingBotVideoTransformer3DModelConfig


@MODELING_REGISTRY.register(_LINGBOT_ARCH)
def register_lingbot_video_transformer_modeling(architecture: str = None):
    del architecture  # diffusers configs carry no `architectures` list
    from .modeling_lingbot_video import (
        LingBotVideoTransformer3DModel,
        apply_veomni_lingbot_video_moe_patch,
    )

    apply_veomni_lingbot_video_moe_patch()
    return LingBotVideoTransformer3DModel


__all__ = [
    "register_lingbot_video_transformer_config",
    "register_lingbot_video_transformer_modeling",
]

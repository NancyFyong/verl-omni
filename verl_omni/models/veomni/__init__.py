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

"""VeOmni-native model registrations shipped by verl-omni.

Importing a submodule registers its architecture into VeOmni's
``MODEL_CONFIG_REGISTRY``/``MODELING_REGISTRY``.  Registration is on-demand
(``register_veomni_models``) rather than at package import: both ``veomni``
and the model's own optional dependency (e.g. ``lingbot-video``) must be
installed, and only the VeOmni engine needs these entries.
"""


def register_veomni_models(architecture: str | None = None) -> None:
    """Idempotently register verl-omni's VeOmni-native models.

    Args:
        architecture: When given, only the registration covering this
            architecture is imported; unknown names are ignored so stock
            VeOmni models (e.g. Qwen-Image) keep working unchanged.
    """
    if architecture is not None and architecture not in _ARCH_TO_MODULE:
        return
    modules = {_ARCH_TO_MODULE[architecture]} if architecture is not None else set(_ARCH_TO_MODULE.values())
    import importlib

    for module in modules:
        importlib.import_module(f"{__name__}.{module}")


_ARCH_TO_MODULE = {
    "LingBotVideoTransformer3DModel": "lingbot_video",
}

__all__ = ["register_veomni_models"]

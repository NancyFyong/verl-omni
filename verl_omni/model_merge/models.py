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
"""Closed architecture mapping, shared across algorithms; not a plugin registry."""

from importlib import import_module

TRANSFORMERS = {
    "QwenImagePipeline": "QwenImageTransformer2DModel",
    "QwenImageEditPlusPipeline": "QwenImageTransformer2DModel",
    "StableDiffusion3Pipeline": "SD3Transformer2DModel",
    "FluxPipeline": "FluxTransformer2DModel",
    "WanPipeline": "WanTransformer3DModel",
    "LTX2Pipeline": "LTX2VideoTransformer3DModel",
    "MiniMaxH3Pipeline": "MiniMaxH3Transformer3DModel",
    "BooguImagePipeline": "BooguImageTransformer2DModel",
}
PIPELINES = frozenset(TRANSFORMERS) - {"MiniMaxH3Pipeline", "BooguImagePipeline"}


def transformer_class(architecture: str):
    """Resolve only repository-supported canonical classes, never checkpoint Python code."""
    if architecture not in TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    library = "boogu.models.transformers.transformer_boogu" if architecture == "BooguImagePipeline" else "diffusers"
    return getattr(import_module(library), TRANSFORMERS[architecture])


def resolve_architecture(index: dict | None, config: dict, explicit: str | None) -> str:
    """Use the pipeline identity or the canonical component class; reject conflicting claims."""
    inferred = index.get("_class_name") if index is not None else None
    if inferred is not None and explicit is not None and inferred != explicit:
        raise ValueError("Explicit architecture conflicts with base model_index.json")
    architecture = explicit or inferred
    if architecture is None:
        architecture = next((key for key, value in TRANSFORMERS.items() if value == config.get("_class_name")), None)
    if architecture not in TRANSFORMERS:
        raise ValueError(f"Unsupported publishing architecture: {architecture}")
    if config.get("_class_name") != TRANSFORMERS[architecture]:
        raise ValueError("Base must contain the canonical Diffusers transformer, not native/fused inference weights")
    return architecture

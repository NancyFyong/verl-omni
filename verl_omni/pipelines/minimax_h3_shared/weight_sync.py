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

"""Shared Diffusers-to-native H3 weight layout; adapters retain their policies."""

from collections.abc import Iterable

import torch

_TOPLEVEL_RENAMES = (
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("context_embedder", "condition_proj"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
)

_LORA_STACKED_PARAMS_MAPPING = [
    (".qkv_proj", ".to_q", "q"),
    (".qkv_proj", ".to_k", "k"),
    (".qkv_proj", ".to_v", "v"),
    (".fc1", ".fc1_0", "0"),
    (".fc1", ".fc1_1", "1"),
]
_LORA_TARGET_MAPPING = {
    "to_q": ("to_q",),
    "to_k": ("to_k",),
    "to_v": ("to_v",),
    "to_out.0": ("out_proj",),
    "ff.net.0.proj": ("fc1_0", "fc1_1"),
    "ff.net.2": ("fc2",),
}
H3_LORA_TARGETS = frozenset(_LORA_TARGET_MAPPING)
_LORA_VLLM_TARGET_MODULES = [target for targets in _LORA_TARGET_MAPPING.values() for target in targets]


def _diffusers_to_vllm_name(name: str) -> str:
    """Rename an unfused Diffusers H3 parameter without changing its tensor."""
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    name = name.replace("transformer_blocks.", "blocks.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    for source, target in _TOPLEVEL_RENAMES:
        if name.startswith(source + "."):
            return target + name[len(source) :]
    return name


def validate_lora_target_modules(target_modules) -> set[str]:
    """Retain NFT's exact-target whitelist, also used by its Actor validation."""
    if isinstance(target_modules, str):
        requested = {target_modules}
    elif isinstance(target_modules, list | tuple | set | frozenset):
        requested = {str(target) for target in target_modules}
    else:
        raise ValueError(f"MiniMax H3 LoRA requires an explicit target_modules list; got {target_modules!r}.")
    unsupported = requested - H3_LORA_TARGETS
    if not requested or unsupported:
        raise ValueError(
            "MiniMax H3 LoRA supports only transformer/refiner block targets "
            f"{sorted(H3_LORA_TARGETS)}, got {sorted(requested)}. "
            "`all-linear` and other top-level modules are not synced to rollout "
            "(FSDP layered-summon does not transport them)."
        )
    return requested


def map_lora_tensors(
    tensors: dict[str, torch.Tensor], component: str, ff_half: int, *, strict: bool
) -> dict[str, torch.Tensor]:
    """Share the tensor mapping while preserving the adapters' payload validation."""
    mapped: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        is_lora_a = name.endswith(".lora_A.weight")
        is_lora_b = name.endswith(".lora_B.weight")
        if not (is_lora_a or is_lora_b):
            mapped[name] = tensor
            continue
        suffix = ".lora_A.weight" if is_lora_a else ".lora_B.weight"
        module = name[: -len(suffix)]
        anchors = [
            offset
            for offset in (module.find("transformer_blocks."), module.find("token_refiner.refiner_blocks."))
            if offset >= 0
        ]
        if not anchors:
            if strict:
                raise ValueError(f"MiniMax H3 cannot map LoRA tensor outside supported DiT blocks: {name}.")
            mapped[name] = tensor
            continue
        module = module[min(anchors) :]
        vllm_module = _diffusers_to_vllm_name(module + ".")[:-1]
        if ".ff.net.0.proj" in module:
            base = vllm_module.replace(".ff.net.0.proj", ".mlp.fc1")
            if is_lora_b:
                if strict and tensor.shape[0] != 2 * ff_half:
                    raise ValueError(
                        f"MiniMax H3 fc1 LoRA B rows must be {2 * ff_half}, got {tensor.shape[0]} for {name}."
                    )
                # Diffusers stores [up, gate]; native logical slices are [gate, up].
                swapped = torch.cat([tensor[ff_half:], tensor[:ff_half]], dim=0)
                mapped[f"{component}.{base}_0{suffix}"] = swapped[:ff_half].contiguous()
                mapped[f"{component}.{base}_1{suffix}"] = swapped[ff_half:].contiguous()
            else:
                mapped[f"{component}.{base}_0{suffix}"] = tensor
                mapped[f"{component}.{base}_1{suffix}"] = tensor
            continue
        mapped[f"{component}.{vllm_module}{suffix}"] = tensor
    return mapped


class MiniMaxH3WeightSyncBase:
    """Load fused projections through native TP-aware loaders for both algorithms.

    The two class attributes preserve existing NFT/FlowGRPO component and RoPE
    policies; they are not runtime configuration options.
    """

    _h3_sync_components = ("transformer", "transformers_ref")
    _h3_initialize_rope = False

    def _h3_weight_component_name(self) -> str:
        return "transformer"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Translate names and load logical QKV/GEGLU shards without retaining buckets."""
        translated: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        component_params: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            component, separator, inner = name.partition(".")
            if separator != "." or component not in self._h3_sync_components:
                translated.append((name, tensor))
                continue
            target_component = self._h3_weight_component_name() if component == "transformer" else component
            inner = inner.replace(".base_layer", "")
            if "lora_" in inner:
                continue

            is_qkv = inner.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight"))
            is_fc1 = inner.endswith(".ff.net.0.proj.weight")
            if is_qkv or is_fc1:
                if is_qkv:
                    block, projection = inner.rsplit(".attn.to_", 1)
                    target_name = f"{_diffusers_to_vllm_name(block)}.attn.qkv_proj.weight"
                else:
                    target_name = _diffusers_to_vllm_name(inner).replace(".ff.net.0.proj.", ".mlp.fc1.")
                if target_component not in component_params:
                    component_params[target_component] = dict(getattr(self, target_component).named_parameters())
                param = component_params[target_component][target_name]
                if is_qkv:
                    param.weight_loader(param, tensor, projection[0])
                else:
                    up, gate = tensor.chunk(2, dim=0)
                    param.weight_loader(param, gate, 0)
                    param.weight_loader(param, up, 1)
                loaded.add(f"{component}.{target_name}")
                continue
            translated.append((f"{target_component}.{_diffusers_to_vllm_name(inner)}", tensor))

        needs_rope = self._h3_initialize_rope and not getattr(self, "_rope_inv_freq_loaded", False)
        if needs_rope:
            rope_len = self.transformer.arch.rope_inv_freq_len
            inv_freq = 10000.0 ** (-(torch.arange(0, 2 * rope_len, 2, dtype=torch.float32) / (2 * rope_len)))
            # Keep prefix groups contiguous for the native pipeline loader.
            insert_at = next(
                (i for i, (name, _) in enumerate(translated) if not name.startswith("transformer.")), len(translated)
            )
            translated.insert(insert_at, ("transformer.rope.inv_freq", inv_freq))
        if translated or self._h3_initialize_rope:
            loaded.update(super().load_weights(translated))
        if needs_rope and "transformer.rope.inv_freq" in loaded:
            self._rope_inv_freq_loaded = True
        return loaded

    def install_h3_lora_layout(self) -> None:
        """Complete native QKV/GEGLU metadata without discarding existing mappings."""
        transformer = getattr(self, self._h3_weight_component_name(), None)
        if transformer is None:
            return
        existing = getattr(transformer, "stacked_params_mapping", None) or ()
        # Native H3 already declares scoped QKV entries, but not the FC1 slices.
        # The manager matches leaf names, so scoped and unscoped entries are equivalent.
        present = {(packed.rsplit(".", 1)[-1], sub.rsplit(".", 1)[-1], str(shard)) for packed, sub, shard in existing}
        missing = [
            (packed, sub, shard)
            for packed, sub, shard in _LORA_STACKED_PARAMS_MAPPING
            if (packed.rsplit(".", 1)[-1], sub.rsplit(".", 1)[-1], str(shard)) not in present
        ]
        if missing:
            transformer.stacked_params_mapping = [*existing, *missing]

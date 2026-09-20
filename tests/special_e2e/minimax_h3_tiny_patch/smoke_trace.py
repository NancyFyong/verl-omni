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
"""Test-local evidence that tiny H3 smoke actually exercises multi-request paths."""

import functools
import json
import os
from pathlib import Path


def instrument_pipeline(cls, config_path):
    """Record batch admission and CPU training payloads, never model weights."""
    import torch

    root = Path(config_path).parents[2] / "smoke_traces" / f"{cls.__name__}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    counter = 0
    captured_flash = False

    def record(event):
        with (root / "events.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")

    def save(output, request, source):
        nonlocal counter
        metadata = output.output.get("metadata", {}) if isinstance(output.output, dict) else {}
        fields = dict(metadata.get("rl", {})) | dict(metadata.get("prompt_embeddings", {}))
        for key in ("trajectory_latents", "trajectory_timesteps", "trajectory_log_probs"):
            value = getattr(output, key, None)
            if value is not None:
                fields[key] = value
        fields = {key: value.detach().cpu() for key, value in fields.items() if isinstance(value, torch.Tensor)}
        sampling = getattr(request, "sampling", None) or getattr(request, "sampling_params", None)
        event = {
            "event": source,
            "request_id": str(getattr(request, "request_id", "")),
            "seed": getattr(sampling, "seed", None),
            "policy": (getattr(sampling, "extra_args", None) or {}).get("global_steps"),
            "file": f"output_{counter}.pt",
            "finite": all(torch.isfinite(value).all().item() for value in fields.values()),
            "shapes": {key: list(value.shape) for key, value in fields.items()},
        }
        torch.save(fields, root / event["file"])
        counter += 1
        record(event)

    def denoise(self, original, arg, args, kwargs):
        # Scoped to this synchronous tiny forward; never patch installed files.
        from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
        from vllm_omni.diffusion.attention.backends.utils import fa
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _STEP_TRANSFORMER

        nonlocal captured_flash
        states = kwargs.get("states") or arg.states
        event = {
            "event": "denoise_step",
            "batch_size": len(states),
            "step_indices": [state.step_index for state in states],
            "dit_forwards": [],
            "flash_packed_calls": [],
        }
        capture = len(states) > 1 and not captured_flash

        def before_dit(module, inputs, keywords):
            packed = keywords.get("packed_seq_params") or {}
            event["dit_forwards"].append(
                {
                    "num_requests": packed.get("num_requests"),
                    "cu_seqlens_q": packed["cu_seqlens_q"].tolist(),
                }
            )

        original_flash = FlashAttentionImpl._forward_varlen_packed

        def flash(impl, query, key, value, **metadata):
            result = original_flash(impl, query, key, value, **metadata)
            call = {
                "kernel": fa.flash_attn_varlen_func.__module__,
                "shape": list(query.shape),
                "dtype": str(query.dtype),
                "cu_seqlens_q": metadata["cu_seqlens_q"].tolist(),
                "cu_seqlens_k": metadata["cu_seqlens_k"].tolist(),
            }
            if capture:
                filename = f"flash_packed_{len(event['flash_packed_calls'])}.pt"
                torch.save(
                    {
                        "q": query.cpu(),
                        "k": key.cpu(),
                        "v": value.cpu(),
                        "output": result.cpu(),
                        "scale": impl.softmax_scale,
                        "causal": impl.causal,
                        "metadata": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in metadata.items()},
                    },
                    root / filename,
                )
                call["file"] = filename
            event["flash_packed_calls"].append(call)
            return result

        transformers = {id(s.extra[_STEP_TRANSFORMER]): s.extra[_STEP_TRANSFORMER] for s in states}
        hooks = [model.register_forward_pre_hook(before_dit, with_kwargs=True) for model in transformers.values()]
        FlashAttentionImpl._forward_varlen_packed = flash
        try:
            result = original(self, arg, *args, **kwargs)
            if capture and event["flash_packed_calls"]:
                captured_flash = True
            record(event)
            return result
        finally:
            FlashAttentionImpl._forward_varlen_packed = original_flash
            for hook in hooks:
                hook.remove()

    for name in ("forward", "denoise_step", "post_decode"):
        original = getattr(cls, name)

        def wrap(original=original, name=name):
            @functools.wraps(original)
            def traced(self, arg, *args, **kwargs):
                if name == "denoise_step":
                    return denoise(self, original, arg, args, kwargs)
                result = original(self, arg, *args, **kwargs)
                if name == "post_decode":
                    save(result, arg, name)
                elif name == "forward":
                    requests = getattr(arg, "requests", [arg])
                    outputs = result if isinstance(result, list) else [result]
                    record({"event": "forward_batch", "batch_size": len(requests)})
                    for output, request in zip(outputs, requests, strict=True):
                        save(output, request, name)
                return result

            return traced

        setattr(cls, name, wrap())

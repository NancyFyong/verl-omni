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
"""Compatibility shim restoring vllm-omni's pre-0.26 diffusion ``custom_output`` channel.

vllm-omni 0.26 (PR #4922, "Refactor diffusion outputs to payload metadata") dropped the
free-form ``custom_output`` channel that verl-omni's diffusion-RL rollout relies on: the
``DiffusionOutput.custom_output`` field was removed and the engine bridge that threaded it
onto ``OmniRequestOutput`` was deleted, so arbitrary training tensors (``all_latents`` /
``all_log_probs`` / ``all_timesteps`` / H3's ``audio_all_timesteps`` / ``latent_meta`` /
``prompt_embeds`` / ``negative_*``) no longer reach the trainer.

``OmniRequestOutput.custom_output`` itself still exists in 0.26 -- only the diffusion path
stopped populating it -- so this module re-threads the channel from verl-omni's side with
two idempotent monkeypatches, applied at import:

* ``DiffusionOutput.__init__`` regains a ``custom_output`` keyword, stored as an attribute
  and CPU-moved under ``to_cpu``. Restores both construction (the rollout adapters) and
  ``result.custom_output`` reads (``split_diffusion_output_by_request``).
* ``format_diffusion_outputs`` copies the raw output's ``custom_output`` onto each returned
  ``OmniRequestOutput``. Restores the async server's ``final_res.custom_output`` read; the
  value then rides the existing request-output transfer to the client, exactly as on 0.24.

The patch runs in the vllm-omni engine-worker process (imported via ``verl_omni.pipelines``)
and is a silent no-op wherever vllm_omni is absent (CPU) or still carries a native
``custom_output`` field (< 0.26), so every rollout adapter and read-site keeps working
unchanged across the 0.24 -> 0.26 bump.
"""

from typing import Any, Callable

_APPLIED = False
_SHIM_FLAG = "_verl_custom_output_shim"


def _store_custom_output(
    instance: Any,
    custom_output: dict[str, Any] | None,
    to_cpu: bool,
    cpu_mover: Callable[[Any], Any],
) -> None:
    """Set ``instance.custom_output`` (default ``{}``), CPU-moving the tree when ``to_cpu``."""
    value: Any = {} if custom_output is None else custom_output
    if to_cpu:
        value = cpu_mover(value)
    instance.custom_output = value


def _attach_custom_output(diffusion_output: Any, outputs: list) -> list:
    """Copy ``diffusion_output.custom_output`` onto every request output, when non-empty."""
    custom_output = getattr(diffusion_output, "custom_output", None)
    if custom_output:
        for output in outputs:
            output.custom_output = custom_output
    return outputs


def _has_native_custom_output(diffusion_output_cls: Any) -> bool:
    """True when ``DiffusionOutput`` still declares ``custom_output`` as a dataclass field.

    Pre-0.26 vllm-omni keeps the field and its engine bridge, so the channel already works
    and the shim must stay out of the way; 0.26 dropped it, which is what needs the rescue.
    """
    return "custom_output" in getattr(diffusion_output_cls, "__dataclass_fields__", {})


def ensure_custom_output_support() -> None:
    """Idempotently restore the diffusion ``custom_output`` channel on vllm-omni >= 0.26.

    A silent no-op when vllm_omni is unavailable (CPU-only environments), when the native
    ``custom_output`` field is still present (< 0.26), or when already patched.
    """
    global _APPLIED
    if _APPLIED:
        return
    try:
        from vllm_omni.diffusion import data, diffusion_engine, output_formatter
    except Exception:
        return

    diffusion_output_cls = data.DiffusionOutput
    if getattr(diffusion_output_cls, _SHIM_FLAG, False) or _has_native_custom_output(diffusion_output_cls):
        _APPLIED = True
        return

    cpu_mover = diffusion_engine._move_tensor_tree_to_cpu
    original_init = diffusion_output_cls.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        custom_output = kwargs.pop("custom_output", None)
        original_init(self, *args, **kwargs)
        _store_custom_output(self, custom_output, self.to_cpu, cpu_mover)

    diffusion_output_cls.__init__ = _patched_init
    # Fallback so ``result.custom_output`` never raises on an instance built without __init__.
    diffusion_output_cls.custom_output = None
    setattr(diffusion_output_cls, _SHIM_FLAG, True)

    original_format = output_formatter.format_diffusion_outputs

    def _patched_format(**kwargs: Any) -> list:
        outputs = original_format(**kwargs)
        return _attach_custom_output(kwargs.get("diffusion_output"), outputs)

    # The engine calls its own by-name import (diffusion_engine.py imports the symbol), so
    # patch that binding too -- rebinding output_formatter alone would not reach the caller.
    output_formatter.format_diffusion_outputs = _patched_format
    diffusion_engine.format_diffusion_outputs = _patched_format

    _APPLIED = True


ensure_custom_output_support()

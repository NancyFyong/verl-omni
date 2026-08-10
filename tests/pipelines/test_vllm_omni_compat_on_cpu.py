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
"""CPU tests for the vllm-omni 0.26 ``custom_output`` compatibility shim.

The shim re-threads the diffusion ``custom_output`` channel that vllm-omni 0.26 dropped.
Its two monkeypatches wrap real vllm_omni classes (only exercised on GPU / in an isolated
0.26 venv), but their bodies delegate to two pure helpers that take injected objects. These
tests pin those helpers with fakes -- no vllm_omni -- and assert that ``ensure_...`` is a
silent, idempotent no-op wherever vllm_omni is unavailable.
"""

from types import SimpleNamespace

import pytest

from verl_omni.pipelines._vllm_omni_compat import (
    _attach_custom_output,
    _has_native_custom_output,
    _store_custom_output,
    ensure_custom_output_support,
)


class _FakeRequestOutput:
    """Mirrors ``OmniRequestOutput``'s surviving ``custom_output`` property + setter."""

    def __init__(self):
        self._custom_output = None

    @property
    def custom_output(self):
        return self._custom_output

    @custom_output.setter
    def custom_output(self, value):
        self._custom_output = value


def _recording_mover():
    calls = []

    def mover(value):
        calls.append(value)
        return {"__moved__": value}

    return mover, calls


class TestStoreCustomOutput:
    def test_defaults_missing_payload_to_empty_dict(self):
        instance = SimpleNamespace()
        mover, calls = _recording_mover()
        _store_custom_output(instance, None, to_cpu=False, cpu_mover=mover)
        assert instance.custom_output == {}
        assert calls == []

    def test_passes_payload_through_untouched_when_not_to_cpu(self):
        instance = SimpleNamespace()
        mover, calls = _recording_mover()
        payload = {"all_latents": 1, "latent_meta": 2}
        _store_custom_output(instance, payload, to_cpu=False, cpu_mover=mover)
        assert instance.custom_output is payload
        assert calls == []  # mover only runs under to_cpu

    def test_applies_cpu_mover_only_when_to_cpu(self):
        instance = SimpleNamespace()
        mover, calls = _recording_mover()
        payload = {"all_latents": 1}
        _store_custom_output(instance, payload, to_cpu=True, cpu_mover=mover)
        assert instance.custom_output == {"__moved__": payload}
        assert calls == [payload]


class TestAttachCustomOutput:
    def test_copies_payload_onto_every_output(self):
        source = SimpleNamespace(custom_output={"all_log_probs": 0})
        outputs = [_FakeRequestOutput(), _FakeRequestOutput()]
        result = _attach_custom_output(source, outputs)
        assert result is outputs
        for output in outputs:
            assert output.custom_output == {"all_log_probs": 0}

    def test_noop_on_empty_payload(self):
        source = SimpleNamespace(custom_output={})
        output = _FakeRequestOutput()
        _attach_custom_output(source, [output])
        assert output.custom_output is None

    def test_noop_when_attribute_absent(self):
        source = SimpleNamespace()
        output = _FakeRequestOutput()
        _attach_custom_output(source, [output])
        assert output.custom_output is None


class TestHasNativeCustomOutput:
    def test_true_when_field_still_declared(self):
        # Pre-0.26 vllm-omni: the native field is present, so the shim must stay out.
        class Fake:
            __dataclass_fields__ = {"output": object(), "custom_output": object()}

        assert _has_native_custom_output(Fake) is True

    def test_false_when_field_removed(self):
        # 0.26 dropped the field -- this is the case that needs the rescue.
        class Fake:
            __dataclass_fields__ = {"output": object()}

        assert _has_native_custom_output(Fake) is False

    def test_false_when_not_a_dataclass(self):
        class Fake:
            pass

        assert _has_native_custom_output(Fake) is False


class TestEnsureCustomOutputSupport:
    def test_silent_idempotent_noop_without_vllm_omni(self):
        try:
            import vllm_omni  # noqa: F401
        except Exception:
            pass
        else:
            pytest.skip("vllm_omni is installed; the CPU no-op path is not exercised here")
        # Importing the module already ran ensure() once; calling again must not raise.
        ensure_custom_output_support()
        ensure_custom_output_support()

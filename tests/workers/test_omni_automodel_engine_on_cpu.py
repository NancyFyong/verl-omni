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
"""CPU tests for the omni automodel engine.

The AST-based tests parse the engine source and run without ``verl`` or
``nemo_automodel`` installed. The registry-resolution test needs ``verl`` (for
``EngineRegistry`` and the ``AutomodelEngineWithLMHead`` base) and is skipped
otherwise.
"""

import ast
import os

import pytest

OMNI_IMPL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "verl_omni",
    "workers",
    "engine",
    "automodel",
    "omni_impl.py",
)


def _parse_omni_impl_ast() -> ast.Module:
    with open(OMNI_IMPL_PATH, encoding="utf-8") as f:
        return ast.parse(f.read())


def _get_class_def(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {OMNI_IMPL_PATH}")


def _get_func_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {OMNI_IMPL_PATH}")


class TestEngineRegistrationAST:
    """Static checks on the ``@EngineRegistry.register`` decorator."""

    def test_engine_registered_via_decorator(self):
        tree = _parse_omni_impl_ast()
        cls = _get_class_def(tree, "OmniAutomodelEngine")

        register_call = None
        for dec in cls.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "register"
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "EngineRegistry"
            ):
                register_call = dec
                break
        assert register_call is not None, "OmniAutomodelEngine is not decorated with @EngineRegistry.register"

        kwargs = {kw.arg: kw.value for kw in register_call.keywords}
        assert isinstance(kwargs["model_type"], ast.Constant)
        assert kwargs["model_type"].value == "omni_model"
        assert [e.value for e in kwargs["backend"].elts] == ["automodel"]
        assert [e.value for e in kwargs["device"].elts] == ["cuda"]

    def test_engine_subclasses_lm_head_engine(self):
        tree = _parse_omni_impl_ast()
        cls = _get_class_def(tree, "OmniAutomodelEngine")
        base_names = [b.id for b in cls.bases if isinstance(b, ast.Name)]
        assert "AutomodelEngineWithLMHead" in base_names


class TestBuildHelperAST:
    """Static checks that the build helper targets the multimodal model + adapter."""

    def test_builder_uses_multimodal_model_not_causal_lm(self):
        source = open(OMNI_IMPL_PATH, encoding="utf-8").read()
        assert "NeMoAutoModelForMultimodalLM" in source
        assert "NeMoAutoModelForCausalLM" not in source

    def test_builder_applies_omni_adapter_configure_model(self):
        tree = _parse_omni_impl_ast()
        func = _get_func_def(tree, "build_automodel_omni_model")
        func_src = ast.get_source_segment(open(OMNI_IMPL_PATH, encoding="utf-8").read(), func)
        assert "OmniModelBase.get_class_by_name" in func_src
        assert "configure_model" in func_src

    def test_no_hardcoded_cuda_device_string(self):
        # The engine uses get_torch_device().empty_cache(), not torch.cuda directly.
        source = open(OMNI_IMPL_PATH, encoding="utf-8").read()
        assert "torch.cuda" not in source
        assert "get_torch_device()" in source


class TestEngineRegistryResolution:
    """Live registry resolution; needs verl (base engine + registry) and nemo_automodel."""

    def test_engine_registered_for_automodel_backend(self):
        pytest.importorskip("verl")
        pytest.importorskip("nemo_automodel")

        from verl.workers.engine.base import EngineRegistry

        import verl_omni.workers.engine.automodel  # noqa: F401  (registers the engine)

        engines = EngineRegistry._engines
        assert "omni_model" in engines, f"'omni_model' not in EngineRegistry._engines. Keys: {list(engines)}"
        omni_registry = engines["omni_model"]
        assert "automodel" in omni_registry, (
            f"'automodel' not registered for omni_model; registered: {list(omni_registry)}"
        )
        entry = omni_registry["automodel"].get("cuda")
        assert entry is not None, "No engine for omni_model/automodel/cuda"
        assert entry.__name__ == "OmniAutomodelEngine"

    def test_get_engine_cls_resolves_to_omni_automodel_engine(self):
        pytest.importorskip("verl")
        pytest.importorskip("nemo_automodel")

        from verl.workers.engine.base import EngineRegistry

        import verl_omni.workers.engine.automodel  # noqa: F401  (registers the engine)
        from verl_omni.workers.engine.automodel import OmniAutomodelEngine

        os.environ["VERL_ENGINE_DEVICE"] = "cuda"
        try:
            cls = EngineRegistry.get_engine_cls("omni_model", "automodel")
        finally:
            os.environ.pop("VERL_ENGINE_DEVICE", None)
        assert cls is OmniAutomodelEngine

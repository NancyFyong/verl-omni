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
"""Every repository Diffusers architecture: real component and complete-pipeline round trips."""

import ast
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import diffusers
import pytest
import torch
from model_fixtures import run_forward, tiny_pipeline, tiny_transformer

from verl_omni.model_merge import ModelMergerConfig, merge_model, validate_artifact
from verl_omni.model_merge.fsdp_model_merger import _PIPELINES, _TRANSFORMERS
from verl_omni.model_merge.utils import inventory, read_json, tree_files, weight_files, write_json


def _case(tmp_path, architecture, pipeline=False, component="transformer"):
    if architecture == "BooguImagePipeline":
        pytest.importorskip("boogu", reason="Install the optional boogu-image package to test its canonical class")
    model = tiny_transformer(architecture)
    base = tmp_path / "base"
    if pipeline:
        pipe = tiny_pipeline(architecture, model)
        if component == "transformer_2":
            pipe.register_modules(transformer_2=tiny_transformer(architecture))
            pipe.register_to_config(boundary_ratio=0.875, expand_timesteps=True)
            model = pipe.transformer_2
        pipe.save_pretrained(base)
    else:
        model.save_pretrained(base)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.005)
    source = tmp_path / "actor"
    source.mkdir()
    model.save_config(source / "huggingface")
    torch.save(model.state_dict(), source / "model_world_size_1_rank_0.pt")
    write_json(source / "fsdp_config.json", {"FSDP_version": 2, "world_size": 1})
    return ModelMergerConfig(
        str(source),
        str(tmp_path / "output"),
        str(base),
        architecture=architecture,
        output_format="pipeline" if pipeline else "transformer",
        component=component,
        max_shard_size=4096,
        trust_checkpoint=True,
    ), model


@pytest.fixture(scope="module")
def dtensor_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("architecture-dtensors")
    sources = {}
    for architecture in _TRANSFORMERS:
        if architecture == "BooguImagePipeline" and importlib.util.find_spec("boogu") is None:
            continue
        config, _ = _case(root / architecture, architecture)
        source = Path(config.local_dir)
        (source / "model_world_size_1_rank_0.pt").rename(source / "full.pt")
        write_json(source / "fsdp_config.json", {"world_size": 2, "FSDP_version": 2})
        sources[architecture] = source
    run = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("dtensor_checkpoint.py")), *map(str, sources.values())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return sources


@pytest.mark.parametrize("layout", ["single", "dtensor"])
@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_all_components_reload_and_forward(tmp_path, architecture, layout, request):
    config, model = _case(tmp_path, architecture)
    if layout == "dtensor":
        source = request.getfixturevalue("dtensor_sources")[architecture]
        model.load_state_dict(torch.load(source / "full.pt", weights_only=True))
        config = replace(config, local_dir=str(source))
    expected = run_forward(model, architecture)
    result = merge_model(config)
    manifest = validate_artifact(result.output_dir)
    assert manifest["artifact_type"] == "diffusers_transformer"
    assert manifest["tensor_directory"] == "."
    loaded = type(model).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run_forward(loaded, architecture), expected, rtol=1e-6, atol=1e-6)
    assert not (result.output_dir / "model_index.json").exists()


@pytest.mark.parametrize("architecture", sorted(_PIPELINES))
def test_all_pipelines_reload_and_forward(tmp_path, architecture):
    config, model = _case(tmp_path, architecture, pipeline=True)
    before = inventory(Path(config.base_model), tree_files(Path(config.base_model)))
    result = merge_model(config)
    loaded = getattr(diffusers, architecture).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.transformer.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(
        run_forward(loaded.transformer, architecture), run_forward(model, architecture), rtol=1e-6, atol=1e-6
    )
    after = inventory(result.output_dir, tree_files(result.output_dir))
    for name, digest in before.items():
        if not name.startswith("transformer/"):
            assert after[name] == digest, name
    assert validate_artifact(result.output_dir)["trained_components"] == ["transformer"]


def test_wan_second_transformer_and_options_are_not_confused(tmp_path):
    config, model = _case(tmp_path, "WanPipeline", pipeline=True, component="transformer_2")
    first = inventory(Path(config.base_model) / "transformer", tree_files(Path(config.base_model) / "transformer"))
    result = merge_model(config)
    loaded = diffusers.WanPipeline.from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.transformer_2.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run_forward(loaded.transformer_2, "WanPipeline"), run_forward(model, "WanPipeline"))
    assert inventory(result.output_dir / "transformer", tree_files(result.output_dir / "transformer")) == first
    index = read_json(result.output_dir / "model_index.json")
    assert index["boundary_ratio"] == 0.875 and index["expand_timesteps"] is True
    assert validate_artifact(result.output_dir)["tensor_directory"] == "transformer_2"


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_every_architecture_rejects_partial_checkpoint(tmp_path, architecture):
    config, model = _case(tmp_path, architecture)
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    torch.save(state, Path(config.local_dir) / "model_world_size_1_rank_0.pt")
    with pytest.raises(ValueError, match="Incomplete transformer state"):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_native_h3_is_not_a_diffusers_pipeline(tmp_path):
    config, _ = _case(tmp_path, "MiniMaxH3Pipeline")
    with pytest.raises(ValueError, match="complete base pipeline"):
        merge_model(replace(config, output_format="pipeline"))
    data = read_json(Path(config.base_model) / "config.json")
    data["_class_name"] = "MiniMaxH3DiTModel"
    write_json(Path(config.base_model) / "config.json", data)
    with pytest.raises(ValueError, match="native/fused"):
        merge_model(config)


def test_component_export_from_pipeline_does_not_copy_other_assets(tmp_path):
    config, model = _case(tmp_path, "FluxPipeline", pipeline=True)
    result = merge_model(replace(config, output_format="transformer"))
    loaded = type(model).from_pretrained(result.output_dir, local_files_only=True)
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert not (result.output_dir / "vae").exists()
    assert not (result.output_dir / "model_index.json").exists()


def test_only_wan_can_select_a_second_transformer(tmp_path):
    config, _ = _case(tmp_path, "FluxPipeline")
    with pytest.raises(ValueError, match="Only Wan"):
        merge_model(replace(config, component="transformer_2"))


@pytest.mark.parametrize("architecture", sorted(_TRANSFORMERS))
def test_every_architecture_dtype_policy_and_fp32_islands(tmp_path, architecture):
    from safetensors.torch import load_file

    config, model = _case(tmp_path, architecture)
    result = merge_model(replace(config, dtype="bfloat16"))
    values = {}
    for path in set(weight_files(result.output_dir).values()):
        values.update(load_file(path))
    islands = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    for key, value in model.state_dict().items():
        expected = value
        if value.is_floating_point():
            dtype = torch.float32 if any(part in key.split(".") for part in islands) else torch.bfloat16
            expected = value.to(dtype)
        torch.testing.assert_close(values[key], expected, rtol=0, atol=0)


def test_dual_wan_requires_an_explicit_component(tmp_path):
    config, _ = _case(tmp_path, "WanPipeline", pipeline=True, component="transformer_2")
    with pytest.raises(ValueError, match="explicitly select"):
        merge_model(replace(config, component=None))
    assert not Path(config.target_dir).exists()


@pytest.mark.parametrize("fault", ["option", "missing_model", "wrong_model", "wrong_tokenizer", "missing_processor"])
def test_pipeline_component_contract_is_not_just_an_architecture_allowlist(tmp_path, fault):
    arch = "QwenImageEditPlusPipeline" if fault == "missing_processor" else "WanPipeline"
    if fault == "wrong_tokenizer":
        arch = "FluxPipeline"
    config, _ = _case(tmp_path, arch, pipeline=True)
    root = Path(config.base_model)
    path = root / "model_index.json"
    index = read_json(path)
    if fault == "option":
        index["expand_timesteps"] = "false"
    elif fault == "missing_model":
        index["text_encoder"] = [None, None]
    elif fault == "wrong_model":
        index["text_encoder"] = ["transformers", "CLIPTextModel"]
    elif fault == "wrong_tokenizer":
        index["tokenizer"] = ["transformers", "T5Tokenizer"]
    else:
        index.pop("processor")
    write_json(path, index)
    with pytest.raises(ValueError):
        merge_model(config)
    assert not Path(config.target_dir).exists()


def test_component_directory_symlinks_cannot_escape_base(tmp_path):
    config, _ = _case(tmp_path, "FluxPipeline", pipeline=True)
    component = Path(config.base_model) / "transformer"
    external = tmp_path / "external-transformer"
    component.rename(external)
    component.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlinks"):
        merge_model(replace(config, output_format="transformer"))
    assert not Path(config.target_dir).exists()


def test_registry_covers_repository_diffusers_training_architectures():
    root = Path(__file__).resolve().parents[2] / "verl_omni/pipelines"
    found = set()
    for path in root.glob("*/diffusers_training_adapter.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "DiffusionModelBase":
                    found.add(ast.literal_eval(node.args[0]))
    # BAGEL builds NonDiffusersModelBase and requires native publishing, not ModelMixin.save_pretrained.
    assert found - {"OmniBagelForConditionalGeneration"} == set(_TRANSFORMERS)

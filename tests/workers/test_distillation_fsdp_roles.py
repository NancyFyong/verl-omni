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
"""Small GPU tests for role switching on FSDP1 and FSDP2 LoRA modules."""

import os
import shutil
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from tensordict import TensorDict
from torch import nn
from torch.distributed.tensor import DTensor
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.distillation.contracts import PhaseRequest, RoleBinding, RoleGroupSpec
from verl_omni.trainer.diffusion.distillation.recipes import build_plan
from verl_omni.workers.diffusion_distillation_worker import DistillationRoleRuntime
from verl_omni.workers.engine.fsdp.distillation_impl import DistillationRoleGroupEngine


class TinyCheckpointManager:
    def __init__(self, module, optimizer, scheduler):
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def checkpoint_path(local_path):
        rank = dist.get_rank() if dist.is_initialized() else 0
        return os.path.join(local_path, f"primary_rank_{rank}.pt")

    def save_checkpoint(self, local_path, **kwargs):
        os.makedirs(local_path, exist_ok=True)
        torch.save(
            {
                "model": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            self.checkpoint_path(local_path),
        )

    def load_checkpoint(self, local_path, **kwargs):
        state = torch.load(self.checkpoint_path(local_path), weights_only=False)
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, inputs):
        return self.proj(inputs)


def wrap_model(strategy):
    model = get_peft_model(
        TinyModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
        adapter_name="student",
    ).cuda()
    adapter_config = model.peft_config["student"]
    model.add_adapter("fake_score", adapter_config)
    model.add_adapter("student_ema", adapter_config)
    model.set_adapter("student")
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    fully_shard(model)
    return model


def wrap_independent_model(strategy, role):
    model = get_peft_model(
        TinyModel(),
        LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]),
        adapter_name="default",
    ).cuda()
    model.add_adapter(role, model.peft_config["default"])
    model.set_adapter(role)
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    fully_shard(model)
    return model


def engine_shell(module, adversarial=False):
    engine = object.__new__(DistillationRoleGroupEngine)
    engine.module = module
    engine.role_group = RoleGroupSpec(
        name="base", model_ref="/tiny", storage="shared_base_adapters", placement="colocated"
    )
    engine.role_bindings = {
        "student": RoleBinding("student", "base", "student", True, "student_optim"),
        "teacher_score": RoleBinding("teacher_score", "base", None, False, None),
        "fake_score": RoleBinding("fake_score", "base", "fake_score", True, "fake_score_optim"),
        "student_ema": RoleBinding("student_ema", "base", "student_ema", False, None),
    }
    if adversarial:
        from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import (
            QwenImageDistributionMatching,
        )

        engine.model_adapter = QwenImageDistributionMatching
        engine.role_bindings["discriminator"] = RoleBinding(
            "discriminator", "base", "discriminator", True, "discriminator_optim"
        )
    engine.optimizers = {}
    engine.lr_schedulers = {}
    engine.optimizer_configs = {}
    engine._active_role = "student"
    engine._primary_role = "student"
    role_parameters = {}
    trainable_roles = ("student", "fake_score") + (("discriminator",) if adversarial else ())
    for role in trainable_roles:
        with engine.use_role(role):
            role_parameters[role] = (
                engine.model_adapter.distillation_role_parameters(module, role)
                if adversarial
                else tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
            )
    engine._role_parameters = role_parameters
    engine.optimizers = {role: torch.optim.AdamW(parameters, lr=0.1) for role, parameters in role_parameters.items()}
    engine.lr_schedulers = {
        role: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for role, optimizer in engine.optimizers.items()
    }
    engine.optimizer_configs = {role: SimpleNamespace(clip_grad=1.0) for role in engine.optimizers}
    engine.optimizer = engine.optimizers["student"]
    engine.lr_scheduler = engine.lr_schedulers["student"]
    engine.optimizer_config = engine.optimizer_configs["student"]
    engine.rank = dist.get_rank()
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.checkpoint_manager = TinyCheckpointManager(engine.module, engine.optimizer, engine.lr_scheduler)
    return engine


def independent_engine_shell(module, role, trainable):
    engine = object.__new__(DistillationRoleGroupEngine)
    engine.module = module
    engine.role_group = SimpleNamespace(name=f"{role}_model", storage="independent_module")
    engine.role_bindings = {
        role: RoleBinding(role, f"{role}_model", role, trainable, f"{role}_optim" if trainable else None)
    }
    engine.optimizers = {}
    engine.lr_schedulers = {}
    engine.optimizer_configs = {}
    engine._active_role = role
    engine._primary_role = role if trainable else None
    return engine


def adapter_snapshot(engine, role):
    binding = engine.role_bindings[role]
    with engine._adapter_state_context(), torch.no_grad():
        parameters = engine._active_adapter_trainable_params(binding.adapter)
        return tuple(
            (parameter.full_tensor() if isinstance(parameter, DTensor) else parameter).detach().cpu().clone()
            for parameter in parameters
        )


def role_snapshot(engine, role):
    values = []
    binding = engine.role_bindings[role]
    with engine._adapter_state_context(), torch.no_grad():
        parameters = list(engine._active_adapter_trainable_params(binding.adapter))
        if role == "discriminator":
            parameters.extend(
                parameter for name, parameter in engine.module.named_parameters() if "proj_out.classifier." in name
            )
        for parameter in parameters:
            value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            values.append(value.detach().cpu().clone())
    return tuple(values)


def assert_tensors_equal(left, right):
    assert len(left) == len(right)
    for left_tensor, right_tensor in zip(left, right, strict=True):
        torch.testing.assert_close(left_tensor, right_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_distillation_role_switch_preserves_graph_ema_and_state(strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP role-isolation tests.")
    if dist.is_initialized():
        pytest.skip("This isolated one-rank FSDP test requires no existing process group.")

    with tempfile.TemporaryDirectory(prefix="distillation_fsdp_role_") as tmp_dir:
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{os.path.join(tmp_dir, 'rendezvous')}",
            rank=0,
            world_size=1,
        )
        try:
            torch.manual_seed(7)
            engine = engine_shell(wrap_model(strategy))
            engine.copy_adapter("student", "student_ema")
            student_before = adapter_snapshot(engine, "student")
            fake_before = adapter_snapshot(engine, "fake_score")
            ema_before = adapter_snapshot(engine, "student_ema")
            assert_tensors_equal(student_before, ema_before)

            inputs = torch.randn(2, 4, device="cuda")
            engine.optimizer_zero_grad("student")
            with engine.use_role("student") as module:
                student_loss = module(inputs).float().square().mean()
            with engine.use_role("teacher_score") as module:
                teacher_before = module(inputs).detach().clone()
                assert not torch.is_grad_enabled()
            with engine.use_role("fake_score", grad_enabled=False) as module:
                fake_output_before = module(inputs).detach().clone()
                assert not torch.is_grad_enabled()
            engine.backward_role("student", student_loss)
            engine.assert_gradient_isolation({"student"})
            engine.optimizers["student"].step()
            engine.update_role_ema("student", "student_ema", decay=0.5)

            student_after = adapter_snapshot(engine, "student")
            fake_after = adapter_snapshot(engine, "fake_score")
            ema_after = adapter_snapshot(engine, "student_ema")
            assert any(
                not torch.equal(before, after) for before, after in zip(student_before, student_after, strict=True)
            )
            assert_tensors_equal(fake_before, fake_after)
            for before, student, ema in zip(ema_before, student_after, ema_after, strict=True):
                torch.testing.assert_close(ema.float(), (before.float() + student.float()) * 0.5)
            with engine.use_role("teacher_score") as module:
                torch.testing.assert_close(module(inputs), teacher_before, rtol=0, atol=0)
            with engine.use_role("fake_score", grad_enabled=False) as module:
                torch.testing.assert_close(module(inputs), fake_output_before, rtol=0, atol=0)

            engine.optimizer_zero_grad("fake_score")
            with engine.use_role("fake_score") as module:
                fake_loss = module(inputs).float().square().mean()
            engine.backward_role("fake_score", fake_loss)
            engine.optimizers["fake_score"].step()
            engine.lr_schedulers["fake_score"].step()
            checkpoint_student = adapter_snapshot(engine, "student")
            checkpoint_fake = adapter_snapshot(engine, "fake_score")
            checkpoint_ema = adapter_snapshot(engine, "student_ema")
            checkpoint_path = os.path.join(tmp_dir, f"{strategy}_checkpoint")
            engine.save_role_group_checkpoint(checkpoint_path, global_step=1)

            with engine.use_role("student") as module:
                second_loss = module(inputs).float().square().mean()
            engine.optimizer_zero_grad("student")
            engine.backward_role("student", second_loss)
            engine.optimizers["student"].step()
            with engine._adapter_state_context(), torch.no_grad():
                for parameter in engine._active_adapter_trainable_params("fake_score"):
                    parameter.fill_(17.0)
            engine.load_role_group_checkpoint(checkpoint_path)
            assert_tensors_equal(adapter_snapshot(engine, "student"), checkpoint_student)
            assert_tensors_equal(adapter_snapshot(engine, "fake_score"), checkpoint_fake)
            assert_tensors_equal(adapter_snapshot(engine, "student_ema"), checkpoint_ema)

            torch.manual_seed(17)
            independent_student = independent_engine_shell(wrap_independent_model(strategy, "student"), "student", True)
            torch.manual_seed(19)
            independent_ema = independent_engine_shell(
                wrap_independent_model(strategy, "student_ema"), "student_ema", False
            )
            with independent_student._adapter_state_context(), torch.no_grad():
                for parameter in independent_student._active_adapter_trainable_params("student"):
                    parameter.fill_(4.0)
            with independent_ema._adapter_state_context(), torch.no_grad():
                for parameter in independent_ema._active_adapter_trainable_params("student_ema"):
                    parameter.fill_(0.0)
            independent_ema.update_module_ema_from(independent_student, decay=0.25)
            independent_values = adapter_snapshot(independent_ema, "student_ema")
            assert independent_values
            assert all(
                torch.allclose(value.float(), torch.full_like(value.float(), 3.0)) for value in independent_values
            )
        finally:
            dist.destroy_process_group()


def tiny_wan_ode_model():
    from diffusers import WanTransformer3DModel

    from verl_omni.pipelines.wan21_distillation.causal_attention import configure_causal_wan

    return configure_causal_wan(
        WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        ).cuda()
    )


def wrap_wan_ode_model(strategy):
    model = tiny_wan_ode_model()
    adapter_config = LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    model.add_adapter(adapter_config, adapter_name="student")
    model.add_adapter(adapter_config, adapter_name="student_ema")
    model.set_adapter("student")
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    for block in model.blocks:
        fully_shard(block)
    fully_shard(model)
    return model


def ode_engine_shell(module):
    from verl_omni.pipelines.wan21_distillation.diffusers_training_adapter import Wan21CausalODE

    engine = object.__new__(DistillationRoleGroupEngine)
    engine.module = module
    engine.model_adapter = Wan21CausalODE
    engine.role_group = RoleGroupSpec(
        name="causal_base", model_ref="/tiny", storage="shared_base_adapters", placement="colocated"
    )
    engine.role_bindings = {
        "student": RoleBinding("student", "causal_base", "student", True, "student_optim"),
        "student_ema": RoleBinding("student_ema", "causal_base", "student_ema", False, None),
    }
    engine.optimizers = {}
    engine.lr_schedulers = {}
    engine.optimizer_configs = {}
    engine._active_role = "student"
    engine._primary_role = "student"
    with engine.use_role("student"):
        parameters = engine.model_adapter.distillation_role_parameters(module, "student")
    engine._role_parameters = {"student": parameters}
    engine.optimizers = {"student": torch.optim.AdamW(parameters, lr=0.1)}
    engine.lr_schedulers = {"student": torch.optim.lr_scheduler.LambdaLR(engine.optimizers["student"], lambda _: 1.0)}
    engine.optimizer_configs = {"student": SimpleNamespace(clip_grad=1.0)}
    engine.optimizer = engine.optimizers["student"]
    engine.lr_scheduler = engine.lr_schedulers["student"]
    engine.optimizer_config = engine.optimizer_configs["student"]
    engine.rank = dist.get_rank()
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.checkpoint_manager = TinyCheckpointManager(engine.module, engine.optimizer, engine.lr_scheduler)
    return engine


def wrap_qwen_image_model(strategy, model_path, adversarial=False):
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    ).cuda()
    if adversarial:
        from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import (
            QwenImageDistributionMatching,
        )

        QwenImageDistributionMatching.build_discriminator_head(model)
    adapter_config = LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    model.add_adapter(adapter_config, adapter_name="student")
    model.add_adapter(adapter_config, adapter_name="fake_score")
    model.add_adapter(adapter_config, adapter_name="student_ema")
    if adversarial:
        model.add_adapter(adapter_config, adapter_name="discriminator")
    model.set_adapter("student")
    if adversarial:
        QwenImageDistributionMatching.configure_role_parameters(model, "discriminator")
        model.set_adapter("student")
    if strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, use_orig_params=True, device_id=torch.cuda.current_device())
    from torch.distributed.fsdp import fully_shard

    for block in model.transformer_blocks:
        fully_shard(block)
    fully_shard(model)
    return model


@pytest.fixture(scope="module")
def qwen_process_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen-Image FSDP distillation tests.")
    if dist.is_initialized():
        pytest.skip("This FSDP fixture creates its own process group.")
    with tempfile.TemporaryDirectory(prefix="qwen_image_dmd_fsdp_") as tmp_dir:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(
            backend="nccl",
            init_method="env://" if world_size > 1 else f"file://{os.path.join(tmp_dir, 'rendezvous')}",
            rank=int(os.environ.get("RANK", "0")),
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
@pytest.mark.parametrize("algorithm", ["dmd", "dmd2"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_qwen_image_dm_computer_on_fsdp(strategy, algorithm, batch_size, qwen_process_group):
    model_path = os.environ.get("QWEN_IMAGE_MODEL_PATH", os.path.expanduser("~/models/tiny-random/Qwen-Image"))
    if not os.path.isfile(os.path.join(model_path, "model_index.json")):
        pytest.skip(f"Tiny Qwen-Image checkpoint not found at {model_path}.")

    from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import QwenImageDMDComputer

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.manual_seed(7)
    engine = engine_shell(wrap_qwen_image_model(strategy, model_path))
    engine.scheduler = SimpleNamespace()
    engine.model_config = SimpleNamespace(fsdp_layer_prefixes=["transformer_blocks."])
    engine.ulysses_device_mesh = None
    engine.ulysses_sequence_parallel_size = 1
    plan = build_plan(
        algorithm,
        {
            "model_path": model_path,
            "conditioning_provider": "local_frozen_encoder",
            "fake_update_ratio": 2,
            "regression_type": "decoded_lpips",
            "rng_seed": 3,
        },
        frozenset({"distribution_matching"}),
    )
    runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.9, ema_start_step=0)
    model_config = SimpleNamespace(
        path=model_path,
        local_path=model_path,
        transformer_config={"in_channels": 64},
        pipeline=SimpleNamespace(
            height=64,
            width=64,
            num_inference_steps=4,
            max_sequence_length=64,
            guidance_scale=None,
        ),
    )
    computer = QwenImageDMDComputer(model_config, plan)
    batch = TensorDict({"dummy_tensor": torch.zeros(batch_size, 1, device="cuda")}, batch_size=[batch_size])
    tu.assign_non_tensor_stack(
        batch,
        "raw_prompt",
        [
            [{"role": "user", "content": "cat" if (rank + row) % 2 == 0 else "a red apple on a wooden table"}]
            for row in range(batch_size)
        ],
    )
    if algorithm == "dmd":
        pytest.importorskip("piq")
        batch["reference_noise"] = torch.randn(batch_size, 16, 64, device="cuda")
        batch["teacher_target_latents"] = torch.zeros(batch_size, 16, 64, device="cuda")
        tu.assign_non_tensor_stack(
            batch, "teacher_sampling_manifest", [{"scheduler": "tiny-qwen", "sample": row} for row in range(batch_size)]
        )
    for kind, role in (("student", "student"), ("fake_score", "fake_score"), ("fake_score", "fake_score")) * 3:
        request = PhaseRequest(
            kind=kind,
            global_step=0,
            repeat_index=0,
            batch_policy="fresh",
            trainable_roles=(role,),
            update_ema=kind == "student",
        )
        runtime.zero_grad(request.trainable_roles)
        computation = computer.compute_phase(request, batch, runtime)
        exits = torch.tensor([computation.metrics["rollout/exit_index"]], device="cuda")
        gathered = [torch.zeros_like(exits) for _ in range(world_size)]
        dist.all_gather(gathered, exits)
        assert all(value.item() == exits.item() for value in gathered)
        runtime.backward_micro_batch(request, computation, weight=1.0)
        optimizer_steps, _ = runtime.step_phase(request)
        assert optimizer_steps == {role: 1}

    tensors, peft_config = runtime.export_tensors(base_sync_done=True)
    exported = list(tensors)
    assert exported
    assert peft_config["r"] == 2
    assert all(name.startswith("transformer.") for name, _ in exported)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_qwen_image_dmd2_adversarial_roles_and_checkpoint(strategy, qwen_process_group):
    model_path = os.environ.get("QWEN_IMAGE_MODEL_PATH", os.path.expanduser("~/models/tiny-random/Qwen-Image"))
    if not os.path.isfile(os.path.join(model_path, "model_index.json")):
        pytest.skip(f"Tiny Qwen-Image checkpoint not found at {model_path}.")

    from verl_omni.pipelines.qwen_image_distillation.diffusers_training_adapter import QwenImageDMDComputer

    torch.manual_seed(11)
    engine = engine_shell(wrap_qwen_image_model(strategy, model_path, adversarial=True), adversarial=True)
    engine.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000})
    engine.model_config = SimpleNamespace(fsdp_layer_prefixes=["transformer_blocks."])
    engine.ulysses_device_mesh = None
    engine.ulysses_sequence_parallel_size = 1
    plan = build_plan(
        "dmd2",
        {
            "model_path": model_path,
            "profile": "paper",
            "conditioning_provider": "local_frozen_encoder",
            "fake_update_ratio": 1,
            "adversarial": {"mode": "cls_on_clean_image"},
            "rng_seed": 5,
        },
        frozenset({"distribution_matching", "adversarial"}),
    )
    runtime = DistillationRoleRuntime(plan, {"base": engine}, ema_decay=0.9, ema_start_step=0)
    model_config = SimpleNamespace(
        path=model_path,
        local_path=model_path,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        transformer_config={"in_channels": 64},
        pipeline=SimpleNamespace(
            height=64,
            width=64,
            num_inference_steps=4,
            max_sequence_length=64,
            guidance_scale=None,
        ),
    )
    computer = QwenImageDMDComputer(model_config, plan)
    computer.vae_fingerprint = "fixture"
    batch = TensorDict(
        {"dummy_tensor": torch.zeros(1, 1, device="cuda"), "real_latents": torch.zeros(1, 16, 64, device="cuda")},
        batch_size=[1],
    )
    tu.assign_non_tensor_stack(batch, "raw_prompt", [[{"role": "user", "content": "cat"}]])
    tu.assign_non_tensor_stack(
        batch,
        "real_latent_manifest",
        [{"normalization": "qwen_image", "vae_config_sha256": "fixture"}],
    )

    before = {role: role_snapshot(engine, role) for role in ("student", "fake_score", "discriminator")}
    student_request = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
    runtime.zero_grad(student_request.trainable_roles)
    student = computer.compute_phase(student_request, batch, runtime)
    runtime.backward_micro_batch(student_request, student, weight=1.0)
    steps, _ = runtime.step_phase(student_request)
    assert steps == {"student": 1}
    assert_tensors_equal(role_snapshot(engine, "fake_score"), before["fake_score"])
    assert_tensors_equal(role_snapshot(engine, "discriminator"), before["discriminator"])

    fake_request = PhaseRequest("fake_score", 1, 0, "fresh", ("fake_score", "discriminator"), False)
    runtime.zero_grad(fake_request.trainable_roles)
    observed_roles = []
    for role, computation in computer.iter_role_computations(fake_request, batch, runtime):
        observed_roles.append(role)
        role_request = PhaseRequest("fake_score", 1, 0, "fresh", (role,), False)
        runtime.backward_micro_batch(role_request, computation, weight=1.0)
    assert tuple(observed_roles) == fake_request.trainable_roles
    steps, _ = runtime.step_phase(fake_request)
    assert steps == {"fake_score": 1, "discriminator": 1}
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before["fake_score"], role_snapshot(engine, "fake_score"), strict=True)
    )
    discriminator_after = role_snapshot(engine, "discriminator")
    assert any(not torch.equal(old, new) for old, new in zip(before["discriminator"], discriminator_after, strict=True))
    exported, _ = runtime.export_tensors(base_sync_done=True)
    assert all("classifier" not in name for name, _ in exported)

    checkpoint_paths = [tempfile.mkdtemp(prefix="qwen_dmd2_adversarial_") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(checkpoint_paths, src=0)
    checkpoint_path = checkpoint_paths[0]
    try:
        saved = {role: role_snapshot(engine, role) for role in ("student", "fake_score", "discriminator")}
        engine.save_role_group_checkpoint(checkpoint_path, global_step=1)
        with torch.no_grad():
            for parameter in engine.parameters_for_role("discriminator"):
                parameter.add_(1)
        engine.load_role_group_checkpoint(checkpoint_path)
        for role, expected in saved.items():
            assert_tensors_equal(role_snapshot(engine, role), expected)
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(checkpoint_path)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_wan_ode_regression_role_ema_export_and_checkpoint(strategy, qwen_process_group):
    from verl_omni.pipelines.wan21_distillation.diffusers_training_adapter import WanODEComputer
    from verl_omni.utils.dataset.distillation import canonical_manifest_sha256

    torch.manual_seed(23)
    engine = ode_engine_shell(wrap_wan_ode_model(strategy))
    engine.scheduler = SimpleNamespace(config={"num_train_timesteps": 1000})
    engine.model_config = SimpleNamespace(fsdp_layer_prefixes=["blocks."])
    engine.ulysses_device_mesh = None
    engine.ulysses_sequence_parallel_size = 1
    manifest = {
        "teacher_model": "wan-teacher",
        "teacher_revision": "revision",
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "scheduler_config": {"shift": 8.0},
        "guidance_scale": 6.0,
        "negative_prompt": "",
        "timesteps": [1000.0, 500.0, 0.0],
        "vae": "wan-vae",
        "latent_layout": "SFCHW",
        "dtype": "float32",
        "height": 4,
        "width": 4,
        "num_frames": 6,
        "prompt_tokenizer": "umt5",
        "seed_policy": "fixture",
    }
    digest = canonical_manifest_sha256(manifest)
    plan = build_plan(
        "ode_regression",
        {
            "model_path": "/tiny",
            "trajectory_manifest_sha256": digest,
            "conditioning_provider": "precomputed",
            "frames_per_block": 2,
            "rng_seed": 9,
        },
        frozenset({"distribution_matching", "autoregressive", "ode_regression"}),
    )
    runtime = DistillationRoleRuntime(plan, {"causal_base": engine}, ema_decay=0.5, ema_start_step=0)
    model_config = SimpleNamespace(
        path="/tiny",
        local_path="/tiny",
        pipeline=SimpleNamespace(max_sequence_length=8),
    )
    computer = WanODEComputer(model_config, plan)
    trajectory = torch.randn(1, 3, 6, 4, 4, 4, device="cuda")
    batch = TensorDict(
        {
            "prompt_embeds": torch.randn(1, 3, 16, device="cuda"),
            "ode_latents": trajectory,
            "ode_timesteps": torch.tensor([[1000.0, 500.0, 0.0]], device="cuda"),
            "final_clean_latent": trajectory[:, -1].clone(),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor_stack(batch, "trajectory_manifest", [manifest])
    tu.assign_non_tensor_stack(batch, "trajectory_manifest_sha256", [digest])
    before = role_snapshot(engine, "student")
    ema_before = role_snapshot(engine, "student_ema")
    request = PhaseRequest("student", 0, 0, "fresh", ("student",), True)
    runtime.zero_grad(request.trainable_roles)
    computation = computer.compute_phase(request, batch, runtime)
    assert torch.isfinite(computation.losses["student"])
    runtime.backward_micro_batch(request, computation, weight=1.0)
    steps, _ = runtime.step_phase(request)
    assert steps == {"student": 1}
    assert any(not torch.equal(old, new) for old, new in zip(before, role_snapshot(engine, "student"), strict=True))
    runtime.update_ema()
    assert any(
        not torch.equal(old, new) for old, new in zip(ema_before, role_snapshot(engine, "student_ema"), strict=True)
    )

    exported, peft_config = runtime.export_tensors(base_sync_done=True)
    exported = {name.removeprefix("transformer."): tensor.detach().cpu() for name, tensor in exported}
    assert exported
    assert peft_config["r"] == 2
    reloaded = tiny_wan_ode_model()
    reloaded.add_adapter(
        LoraConfig(
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            target_modules=peft_config["target_modules"],
        ),
        adapter_name="student_ema",
    )
    incompatible = set_peft_model_state_dict(reloaded, exported, adapter_name="student_ema")
    assert not incompatible.unexpected_keys
    reloaded_state = get_peft_model_state_dict(reloaded, adapter_name="student_ema")
    assert set(reloaded_state) == set(exported)
    for name, tensor in reloaded_state.items():
        torch.testing.assert_close(tensor.cpu(), exported[name], rtol=0, atol=0)
    checkpoint_paths = [tempfile.mkdtemp(prefix="wan_ode_") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(checkpoint_paths, src=0)
    checkpoint_path = checkpoint_paths[0]
    try:
        saved_student = role_snapshot(engine, "student")
        saved_ema = role_snapshot(engine, "student_ema")
        engine.save_role_group_checkpoint(checkpoint_path, global_step=1)
        with torch.no_grad():
            for parameter in engine.parameters_for_role("student"):
                parameter.add_(1)
        engine.load_role_group_checkpoint(checkpoint_path)
        assert_tensors_equal(role_snapshot(engine, "student"), saved_student)
        assert_tensors_equal(role_snapshot(engine, "student_ema"), saved_ema)
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(checkpoint_path)

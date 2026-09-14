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

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.ray_diffusion_trainer import DistributionMatchingRayTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls

CONFIG_DIR = str(Path(verl_omni.__file__).parent / "trainer" / "config")


def make_config(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="diffusion_trainer",
            overrides=[
                "algorithm.trainer_type=distribution_matching",
                "algorithm.sample_source=offline",
                "actor_rollout_ref.model.algorithm=dmd2",
                "actor_rollout_ref.model.model_type=diffusion_dmd_model",
                "actor_rollout_ref.model.lora_rank=2",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "data.train_batch_size=8",
                "trainer.total_training_steps=3",
                "trainer.save_freq=-1",
                "trainer.test_freq=-1",
                "trainer.val_before_train=false",
                "trainer.resume_mode=disable",
                *overrides,
            ],
        )


class FakeTracking:
    records = []

    def __init__(self, **kwargs):
        pass

    def log(self, data, step):
        self.records.append((step, data))


class FakeDMDWorker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.stages = []

    def update_actor(self, data):
        self.stages.append(tu.get_non_tensor_data(data, "dmd_stage", default=None))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return tu.get_tensordict(
            {}, {"metrics": {"dmd/update_applied": outcome, "dmd/skip_nonfinite": 1 - outcome, "loss": 0.25}}
        )


def empty_batch():
    return TensorDict({}, batch_size=[8])


def make_trainer(outcomes):
    trainer = object.__new__(DistributionMatchingRayTrainer)
    trainer.config = make_config()
    trainer.dmd_config = omega_conf_to_dataclass(trainer.config.dmd)
    trainer.global_steps = 0
    trainer.failed = False
    trainer.optimizer_steps = {"student": 0, "fake_score": 0}
    trainer.data_epoch = 0
    trainer.total_training_steps = 3
    trainer.actor_rollout_wg = FakeDMDWorker(outcomes)
    trainer.next_batch = empty_batch
    trainer.export_student = MagicMock()
    return trainer


class TestDMDConfiguration:
    def test_route_and_production_preflight(self):
        config = make_config()
        DistributionMatchingRayTrainer.validate_config(config)
        assert _get_trainer_cls(config) is DistributionMatchingRayTrainer

    @pytest.mark.parametrize(
        "override",
        [
            "algorithm.sample_source=online",
            "actor_rollout_ref.model.algorithm=dmd",
            "actor_rollout_ref.actor.use_kl_loss=true",
            "actor_rollout_ref.actor.use_distill_loss=true",
            "distillation.enabled=true",
            "actor_rollout_ref.actor.ppo_epochs=2",
            "actor_rollout_ref.model.lora_rank=0",
            "actor_rollout_ref.actor.strategy=fsdp",
            "data.train_batch_size=7",
            "actor_rollout_ref.model.model_type=diffusion_model",
            "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
        ],
    )
    def test_unsupported_modes_fail_before_workers(self, override):
        with pytest.raises(ValueError):
            DistributionMatchingRayTrainer.validate_config(make_config([override]))

    def test_fingerprint_is_mapping_order_independent(self):
        trainer = make_trainer([1] * 9)
        first = trainer.configuration_fingerprint()
        from omegaconf import OmegaConf

        fields = OmegaConf.to_container(trainer.config.dmd)
        trainer.config.dmd = dict(reversed(list(fields.items())))
        assert trainer.configuration_fingerprint() == first


def run_dmd_trainer(trainer, entrypoint, monkeypatch):
    if entrypoint == "legacy":
        trainer.fit()
        return

    from verl_omni.trainer import main_diffusion_v1
    from verl_omni.trainer.diffusion.task_runner import TaskRunner

    trainer.config.trainer.use_v1 = True
    trainer.init_workers = MagicMock()
    monkeypatch.setattr(TaskRunner, "create_trainer", MagicMock(return_value=trainer))
    runner = main_diffusion_v1.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
    runner.init_agent_loop_manager = MagicMock(side_effect=AssertionError("DMD2 must not start an agent loop"))
    runner.run(trainer.config)
    trainer.init_workers.assert_called_once()
    assert runner.trainer is trainer


class TestDMDCycles:
    @pytest.mark.parametrize("entrypoint", ["legacy", "v1"])
    def test_normal_and_skipped_cycles_keep_separate_success_counts(self, monkeypatch, entrypoint):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        FakeTracking.records = []
        trainer = make_trainer([1, 1, 1, 0, 1, 0, 1, 0, 1])
        run_dmd_trainer(trainer, entrypoint, monkeypatch)
        assert trainer.global_steps == 3
        assert trainer.optimizer_steps == {"student": 2, "fake_score": 4}
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score", "fake_score"] * 3
        assert [step for step, _ in FakeTracking.records] == [1, 2, 3]
        assert "fake_score/0/loss" in FakeTracking.records[0][1]
        assert "fake_score/1/loss" in FakeTracking.records[0][1]
        trainer.export_student.assert_called_once()

    @pytest.mark.parametrize("entrypoint", ["legacy", "v1"])
    def test_all_skipped_budget_terminates_without_claiming_training_success(self, monkeypatch, entrypoint):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        trainer = make_trainer([0] * 9)
        with pytest.raises(RuntimeError, match="without successful updates"):
            run_dmd_trainer(trainer, entrypoint, monkeypatch)
        assert trainer.global_steps == 3
        assert trainer.optimizer_steps == {"student": 0, "fake_score": 0}
        trainer.export_student.assert_not_called()

    @pytest.mark.parametrize("entrypoint", ["legacy", "v1"])
    def test_partial_exception_is_not_a_numerical_retry(self, monkeypatch, entrypoint):
        monkeypatch.setattr("verl.utils.tracking.Tracking", FakeTracking)
        trainer = make_trainer([1, RuntimeError("injected rank failure"), 1])
        with pytest.raises(RuntimeError, match="injected rank failure"):
            run_dmd_trainer(trainer, entrypoint, monkeypatch)
        assert trainer.global_steps == 0
        assert trainer.optimizer_steps["student"] == 1  # This cannot roll back a real optimizer update.
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score"]
        trainer.export_student.assert_not_called()
        with pytest.raises(RuntimeError, match="must be reconstructed"):
            trainer.fit()
        assert trainer.actor_rollout_wg.stages == ["student", "fake_score"]

    def test_malformed_fractional_outcome_is_not_accepted(self):
        trainer = make_trainer([0.5])
        with pytest.raises(RuntimeError, match="Malformed"):
            trainer.update_stage("student", 0)


class TestDMDV1Routing:
    @pytest.mark.parametrize("use_v1", [False, True])
    def test_hydra_entrypoint_selects_requested_lifecycle(self, monkeypatch, use_v1):
        from verl_omni.trainer import main_diffusion, main_diffusion_v1

        config = make_config([f"trainer.use_v1={str(use_v1).lower()}"])
        v1_run, legacy_run = MagicMock(), MagicMock()
        monkeypatch.setattr(main_diffusion_v1, "run_diffusion_v1", v1_run)
        monkeypatch.setattr(main_diffusion, "run_diffusion", legacy_run)
        if use_v1:
            main_diffusion_v1.main.__wrapped__(config)
            v1_run.assert_called_once_with(config)
            legacy_run.assert_not_called()
        else:
            with pytest.warns(DeprecationWarning, match="legacy diffusion trainer"):
                main_diffusion_v1.main.__wrapped__(config)
            legacy_run.assert_called_once_with(config)
            v1_run.assert_not_called()

    def test_checkpoint_identity_does_not_depend_on_entrypoint(self):
        trainer = make_trainer([1] * 9)
        legacy_fingerprint = trainer.configuration_fingerprint()
        trainer.config.trainer.use_v1 = True
        trainer.config.trainer.v1.trainer_mode = "sync"
        assert trainer.configuration_fingerprint() == legacy_fingerprint

    @pytest.mark.parametrize(
        "override,match",
        [
            ("trainer.v1.trainer_mode=separate_async", "trainer_mode=sync"),
            ("transfer_queue.enable=true", "transfer_queue.enable=false"),
            ("algorithm.sample_source=online", "sample_source=offline"),
            ("actor_rollout_ref.model.algorithm=dmd", "model.algorithm=dmd2"),
            ("distillation.enabled=true", "OPD"),
        ],
    )
    def test_invalid_dmd_fails_before_ray_or_services(self, monkeypatch, override, match):
        from verl_omni.trainer import main_diffusion_v1

        config = make_config(["trainer.use_v1=true", override])
        ray_probe = MagicMock(side_effect=AssertionError("Must validate before Ray initialization"))
        monkeypatch.setattr(main_diffusion_v1.ray, "is_initialized", ray_probe)
        with pytest.raises(ValueError, match=match):
            main_diffusion_v1.run_diffusion_v1(config)
        ray_probe.assert_not_called()

    @pytest.mark.parametrize("failure", [None, "create", "init", "fit"])
    def test_dmd_bypasses_online_services_without_swallowing_errors(self, monkeypatch, failure):
        import transfer_queue as tq

        from verl_omni.trainer import main_diffusion_v1
        from verl_omni.trainer.diffusion import task_runner, v1

        config = make_config(["trainer.use_v1=true"])
        tq_init, tq_close = MagicMock(), MagicMock()
        online_selection = MagicMock(side_effect=AssertionError("Do not select a rollout-mode trainer"))
        monkeypatch.setattr(tq, "init", tq_init)
        monkeypatch.setattr(tq, "close", tq_close)
        monkeypatch.setattr(v1, "get_diffusion_trainer_cls", online_selection)
        trainer = MagicMock()
        create = MagicMock(return_value=trainer)
        monkeypatch.setattr(task_runner.TaskRunner, "create_trainer", create)
        if failure:
            target = {"create": create, "init": trainer.init_workers, "fit": trainer.fit}[failure]
            target.side_effect = RuntimeError(f"injected {failure} failure")
        runner = main_diffusion_v1.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
        runner.init_agent_loop_manager = MagicMock()
        if failure:
            with pytest.raises(RuntimeError, match=f"injected {failure}"):
                runner.run(config)
        else:
            runner.run(config)
            trainer.init_workers.assert_called_once_with()
            trainer.fit.assert_called_once_with()
        if failure in {"create", "init"}:
            trainer.fit.assert_not_called()
        create.assert_called_once_with(config)
        runner.init_agent_loop_manager.assert_not_called()
        online_selection.assert_not_called()
        tq_init.assert_not_called()
        tq_close.assert_not_called()
        assert config.transfer_queue.enable is False

    @pytest.mark.parametrize("trainer_type", ["policy_gradient", "direct_preference"])
    @pytest.mark.parametrize("failure", [None, "init", "fit"])
    def test_online_v1_lifecycle_is_unchanged(self, monkeypatch, trainer_type, failure):
        import transfer_queue as tq

        from verl_omni.trainer import main_diffusion_v1
        from verl_omni.trainer.diffusion import v1

        config = make_config([f"algorithm.trainer_type={trainer_type}", "algorithm.sample_source=online"])
        tq_init, tq_close = MagicMock(), MagicMock()
        monkeypatch.setattr(tq, "init", tq_init)
        monkeypatch.setattr(tq, "close", tq_close)
        trainer = MagicMock()
        trainer_class = MagicMock(return_value=trainer)
        select = MagicMock(return_value=trainer_class)
        monkeypatch.setattr(v1, "get_diffusion_trainer_cls", select)
        if failure:
            getattr(trainer, failure).side_effect = RuntimeError(f"injected {failure} failure")
        runner = main_diffusion_v1.DiffusionTaskRunnerV1.__ray_metadata__.modified_class()
        runner.agent_loop_manager = object()
        runner.init_agent_loop_manager = MagicMock()
        if failure:
            with pytest.raises(RuntimeError, match=f"injected {failure}"):
                runner.run(config)
        else:
            runner.run(config)
            runner.init_agent_loop_manager.assert_called_once_with()
            trainer.fit.assert_called_once_with(runner.agent_loop_manager)
        select.assert_called_once_with("sync")
        trainer_class.assert_called_once_with(config=config)
        tq_init.assert_called_once_with(config.transfer_queue)
        tq_close.assert_called_once_with()
        assert config.transfer_queue.enable is True

    def test_shared_construction_does_not_allocate_workers_or_start_training(self, monkeypatch, tmp_path):
        import verl.utils

        from verl_omni.pipelines.model_base import DiffusionModelBase
        from verl_omni.trainer import main_diffusion
        from verl_omni.trainer.diffusion import task_runner
        from verl_omni.utils import fs
        from verl_omni.utils.dataset import rl_dataset

        assert main_diffusion.TaskRunner is task_runner.TaskRunner
        config = make_config(["+actor_rollout_ref.model.architecture=QwenImagePipeline"])
        monkeypatch.setattr(fs, "resolve_model_local_dir", MagicMock(return_value=str(tmp_path)))
        monkeypatch.setattr(verl.utils, "hf_tokenizer", MagicMock(return_value="tokenizer"))
        monkeypatch.setattr(verl.utils, "hf_processor", MagicMock(return_value="processor"))
        adapter = MagicMock()
        adapter.prepare_processor_files.return_value = None
        monkeypatch.setattr(DiffusionModelBase, "get_class_by_name", MagicMock(return_value=adapter))
        monkeypatch.setattr(rl_dataset, "create_rl_dataset", MagicMock(side_effect=["train", "val"]))
        monkeypatch.setattr(rl_dataset, "create_rl_sampler", MagicMock(return_value="sampler"))
        monkeypatch.setattr(rl_dataset, "get_collate_fn", MagicMock(return_value="collate"))
        selected = MagicMock()
        monkeypatch.setattr(task_runner, "get_diffusion_trainer_cls", MagicMock(return_value=selected))
        runner = task_runner.TaskRunner()
        trainer = runner.create_trainer(config)
        assert trainer is selected.return_value
        assert selected.call_args.kwargs["train_dataset"] == "train"
        assert selected.call_args.kwargs["val_dataset"] == "val"
        assert selected.call_args.kwargs["train_sampler"] == "sampler"
        assert selected.call_args.kwargs["tokenizer"] == "tokenizer"
        trainer.init_workers.assert_not_called()
        trainer.fit.assert_not_called()


class TestDMDCheckpoint:
    def test_counter_mismatch_rejected_before_worker_load(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.n_gpus_per_node = 1
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        torch.save(
            {
                "version": 1,
                "configuration": trainer.configuration_fingerprint(),
                "global_step": 1,
                "optimizer_steps": {"student": 1, "fake_score": 2},
            },
            tmp_path / "trainer.pt",
        )
        (tmp_path / "data.pt").touch()
        actor = tmp_path / "actor"
        actor.mkdir()
        for kind in ("model", "optim", "extra_state"):
            (actor / f"{kind}_world_size_1_rank_0.pt").touch()
        torch.save(
            {"version": 1, "world_size": 1, "optimizer_steps": {"student": 0, "fake_score": 2}},
            actor / "dmd_state_rank_0.pt",
        )
        with pytest.raises(ValueError, match="counters do not match"):
            trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()

    def test_missing_shards_are_not_published(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.default_local_dir = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        with pytest.raises(ValueError, match="missing model_world"):
            trainer._save_checkpoint()
        assert not (tmp_path / "global_step_0").exists()
        assert not list(tmp_path.glob(".global_step_*"))

    def test_failed_save_preserves_latest_and_publishes_no_partial_cycle(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.default_local_dir = str(tmp_path)
        trainer.global_steps = 2
        trainer.train_dataloader = MagicMock()
        trainer.actor_rollout_wg = MagicMock()
        trainer.actor_rollout_wg.save_checkpoint.side_effect = RuntimeError("failed shard")
        tracker = tmp_path / "latest_checkpointed_iteration.txt"
        tracker.write_text("1")
        with pytest.raises(RuntimeError, match="failed shard"):
            trainer._save_checkpoint()
        assert tracker.read_text() == "1"
        assert not (tmp_path / "global_step_2").exists()
        assert not list(tmp_path.glob(".global_step_*"))

    def test_old_checkpoint_is_rejected_before_loading_workers(self, tmp_path):
        trainer = make_trainer([1] * 9)
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path)
        trainer.actor_rollout_wg = MagicMock()
        (tmp_path / "trainer.pt").touch()
        with pytest.raises(ValueError, match="Incomplete/old"):
            trainer._load_checkpoint()
        trainer.actor_rollout_wg.load_checkpoint.assert_not_called()

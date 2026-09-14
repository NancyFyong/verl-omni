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
"""Shared diffusion trainer construction for the legacy and V1 entrypoints."""

import json
import os
import socket

import ray
from omegaconf import OmegaConf
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.utils import need_reference_policy

from verl_omni.trainer.diffusion.ray_diffusion_trainer import (
    DirectPreferenceRayTrainer,
    PolicyGradientRayTrainer,
)


def get_diffusion_trainer_cls(config):
    """Return the trainer class selected by ``algorithm.trainer_type``."""
    trainer_type = config.algorithm.trainer_type
    if trainer_type == "policy_gradient":
        return PolicyGradientRayTrainer
    if trainer_type == "direct_preference":
        return DirectPreferenceRayTrainer
    if trainer_type == "distribution_matching":
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import DistributionMatchingRayTrainer

        return DistributionMatchingRayTrainer
    raise ValueError(
        f"Unsupported diffusion trainer_type {trainer_type!r}. "
        "Expected one of: 'policy_gradient', 'direct_preference', 'distribution_matching'."
    )


class TaskRunner:
    """Ray remote class for executing distributed diffusion training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor (and optional rollout/ref) workers using the unified model engine."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.trainer_type == "distribution_matching":
            from verl_omni.trainer.diffusion.ray_diffusion_trainer import DistributionMatchingRayTrainer
            from verl_omni.workers.dmd_worker import DMDTrainingWorker

            DistributionMatchingRayTrainer.validate_config(config)
            self.role_worker_mapping[Role.Actor] = ray.remote(DMDTrainingWorker)
            self.mapping[Role.Actor] = "global_pool"
            return DMDTrainingWorker, RayWorkerGroup

        from verl_omni.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        separate = config.actor_rollout_ref.get("separate", False)
        if separate:
            if not hasattr(Role, "Actor"):
                raise ValueError("Separate training without colocated rollout requires verl Role.Actor support.")
            role = Role.Actor
        elif config.algorithm.sample_source == "offline":
            if not hasattr(Role, "Actor"):
                raise ValueError("Offline training without rollout requires verl Role.Actor support.")
            role = Role.Actor
        elif need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config) and distillation_config.nnodes > 0:
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Register reward-model GPU pool for online sampling (used by RewardLoopManager)."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.sample_source == "online":
            if config.reward.reward_model.enable:
                # we do not use reward model workers, so we only register reward model in resource pool
                # without continue to register reward model worker in role mapping
                if config.reward.reward_model.enable_resource_pool:
                    self.mapping[Role.RewardModel] = "reward_pool"
                else:
                    self.mapping[Role.RewardModel] = "global_pool"
        elif config.algorithm.sample_source == "offline":
            return

    def add_teacher_model_worker(self, config, teacher_model_cls):
        """Add standalone teacher model workers when distillation runs on its own resource pool."""
        from verl.trainer.ppo.ray_trainer import Role

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config) and distillation_config.nnodes > 0:
            self.role_worker_mapping[Role.TeacherModel] = ray.remote(teacher_model_cls)
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        if not config.actor_rollout_ref.get("separate", False) or not need_reference_policy(config):
            return

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if ref_in_actor:
            return

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
        self.mapping[Role.RefPolicy] = "global_pool"

    def create_trainer(self, config):
        """Build the selected trainer with shared datasets, worker mappings and resource pools."""
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from verl_omni.utils.fs import resolve_model_local_dir

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        self.add_reward_model_resource_pool(config)

        self.add_teacher_model_worker(config, actor_rollout_cls)

        # Add a reference policy worker if KL loss is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Resolve the model path to an on-disk directory (downloads from HDFS or HF Hub
        # if necessary). `use_shm` enables shared-memory copy for faster reloads.
        local_path = resolve_model_local_dir(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        if config.actor_rollout_ref.model.tokenizer_path is None:
            tokenizer_path = os.path.join(local_path, "tokenizer")
            config.actor_rollout_ref.model.tokenizer_path = (
                tokenizer_path if os.path.exists(tokenizer_path) else local_path
            )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        model_config = config.actor_rollout_ref.model
        architecture = model_config.get("architecture")
        if architecture is None:
            model_index_path = os.path.join(local_path, "model_index.json")
            try:
                with open(model_index_path) as model_index_file:
                    architecture = json.load(model_index_file)["_class_name"]
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Unable to infer the diffusion architecture from {model_index_path}. "
                    "Set actor_rollout_ref.model.architecture explicitly."
                ) from exc

        from verl_omni.pipelines.model_base import DiffusionModelBase

        prepared_processor_path = DiffusionModelBase.get_class_by_name(
            architecture,
            model_config.algorithm,
            model_config.get("external_lib"),
        ).prepare_processor_files(local_path)
        processor_path = os.path.join(local_path, "processor")
        if prepared_processor_path is not None:
            processor_path = prepared_processor_path
        if not os.path.exists(processor_path):
            processor_path = local_path
        processor = hf_processor(processor_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler, get_collate_fn

        collate_fn = get_collate_fn(config.data)

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer_cls = get_diffusion_trainer_cls(config)
        return trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

    def run(self, config):
        """Construct, initialize and run the selected diffusion trainer."""
        trainer = self.create_trainer(config)
        trainer.init_workers()
        trainer.fit()

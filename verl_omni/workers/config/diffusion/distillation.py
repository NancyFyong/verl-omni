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

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config import FSDPOptimizerConfig


def default_fake_score_optimizer() -> FSDPOptimizerConfig:
    return FSDPOptimizerConfig(lr=2e-5, weight_decay=0.01, clip_grad=1.0, lr_scheduler_type="constant")


__all__ = [
    "DiffusionDistillationTeacherModelConfig",
    "DiffusionDistributionMatchingConfig",
    "DiffusionDistillationConfig",
]


@dataclass
class DiffusionDistillationTeacherModelConfig(BaseConfig):
    """Frozen diffusion distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Local path to the teacher checkpoint, a full pipeline checkpoint from the
        same pipeline family as the student.
    world_size (int):
        Number of GPUs this teacher occupies in the distillation resource pool.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"key", "world_size"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    world_size: int = 0

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")


@dataclass
class DiffusionDistributionMatchingConfig(BaseConfig):
    """Architecture-neutral DMD-family recipe selection.

    This config is active only when ``algorithm.trainer_type=distillation``.
    The existing parent ``enabled`` flag remains exclusively owned by on-policy
    distillation and must stay false for DMD-family training.
    """

    # Registered recipe name.
    recipe: str = "dmd2"
    # Optional recipe profile; null selects the recipe default.
    profile: Optional[str] = None
    # Optional fake-score phase count; null selects the recipe default.
    fake_update_ratio: Optional[int] = None
    # Number of fake/discriminator-only cycles before student updates begin.
    fake_warmup_cycles: int = 0
    # Optional registered rollout override; null selects the recipe default.
    rollout_strategy: Optional[str] = None
    # Optional data-mode override; null selects the recipe default.
    data_mode: Optional[str] = None
    # Semantic role exported to inference replicas.
    export_role: str = "student_ema"
    # Physical storage used by the initial colocated runtime.
    role_storage: str = "shared_base_adapters"
    # Per-device student phase micro-batch size.
    student_micro_batch_size_per_gpu: int = 1
    # Per-device fake-score phase micro-batch size.
    fake_score_micro_batch_size_per_gpu: int = 1
    # Independent fake-score optimizer and scheduler configuration.
    fake_score_optim: FSDPOptimizerConfig = field(default_factory=default_fake_score_optimizer)
    # EMA decay applied after successful student optimizer steps.
    ema_decay: float = 0.999
    # First completed student step that updates EMA.
    ema_start_step: int = 0
    # Conditioning source used by architecture phase runners.
    conditioning_provider: str = "local_frozen_encoder"
    # Negative prompt used by the guided frozen teacher.
    negative_prompt: str = " "
    # Standard CFG scale for the frozen teacher score.
    teacher_guidance_scale: float = 4.0
    # Teacher CFG normalization mode.
    teacher_cfg_norm: str = "layer_norm"
    # Linear time shift used by the few-step student rollout schedule.
    rollout_timestep_shift: float = 3.0
    # Lower and upper score-noising sigma bounds.
    score_sigma_min: float = 0.02
    score_sigma_max: float = 0.98
    # Rational time shift applied to score-noising samples.
    score_timestep_shift: float = 3.0
    # Number of discrete score-noising samples; zero selects continuous sampling.
    score_discrete_steps: int = 1000
    # Stabilizer for the DMD score-difference normalizer.
    normalization_epsilon: float = 1e-5
    # Distribution-matching loss weight.
    dmd_loss_weight: float = 1.0
    # DMD paired-regression distance.
    regression_type: str = "decoded_lpips"
    # DMD paired-regression loss weight.
    regression_loss_weight: float = 1.0
    # Base seed for worker-local rollout and score-noise generators.
    rng_seed: int = 0

    def __post_init__(self):
        valid_recipes = {"dmd", "dmd2", "causvid", "self_forcing"}
        if self.recipe not in valid_recipes:
            raise ValueError(f"Invalid recipe: {self.recipe}. Must be one of {sorted(valid_recipes)}")
        valid_profiles = {"distribution_only", "paper"}
        if self.profile is not None and self.profile not in valid_profiles:
            raise ValueError(f"Invalid profile: {self.profile}. Must be one of {sorted(valid_profiles)}")
        if self.fake_update_ratio is not None and self.fake_update_ratio <= 0:
            raise ValueError(f"fake_update_ratio must be greater than 0, got {self.fake_update_ratio}")
        if self.fake_warmup_cycles < 0:
            raise ValueError(f"fake_warmup_cycles must be non-negative, got {self.fake_warmup_cycles}")
        valid_rollout_strategies = {
            "backward_simulated",
            "consistency_renoise",
            "ode_euler",
            "one_step",
            "self_forced",
            "teacher_forced_causal",
        }
        if self.rollout_strategy is not None and self.rollout_strategy not in valid_rollout_strategies:
            raise ValueError(
                f"Invalid rollout_strategy: {self.rollout_strategy}. Must be one of {sorted(valid_rollout_strategies)}"
            )
        valid_data_modes = {"prompts", "prompt_and_real_latent", "regression_pairs"}
        if self.data_mode is not None and self.data_mode not in valid_data_modes:
            raise ValueError(f"Invalid data_mode: {self.data_mode}. Must be one of {sorted(valid_data_modes)}")
        valid_export_roles = {"student", "student_ema"}
        if self.export_role not in valid_export_roles:
            raise ValueError(f"Invalid export_role: {self.export_role}. Must be one of {sorted(valid_export_roles)}")
        valid_role_storage = {"shared_base_adapters", "colocated_independent"}
        if self.role_storage not in valid_role_storage:
            raise ValueError(f"Invalid role_storage: {self.role_storage}. Must be one of {sorted(valid_role_storage)}")
        if self.student_micro_batch_size_per_gpu <= 0:
            raise ValueError(
                f"student_micro_batch_size_per_gpu must be greater than 0, got {self.student_micro_batch_size_per_gpu}"
            )
        if self.fake_score_micro_batch_size_per_gpu <= 0:
            raise ValueError(
                "fake_score_micro_batch_size_per_gpu must be greater than 0, "
                f"got {self.fake_score_micro_batch_size_per_gpu}"
            )
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")
        if self.ema_start_step < 0:
            raise ValueError(f"ema_start_step must be non-negative, got {self.ema_start_step}")
        valid_conditioning_providers = {"local_frozen_encoder", "precomputed"}
        if self.conditioning_provider not in valid_conditioning_providers:
            raise ValueError(
                f"Invalid conditioning_provider: {self.conditioning_provider}. "
                f"Must be one of {sorted(valid_conditioning_providers)}"
            )
        if not isinstance(self.negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        if self.teacher_guidance_scale <= 1.0:
            raise ValueError(
                f"teacher_guidance_scale must be greater than 1 for guided teacher scoring, "
                f"got {self.teacher_guidance_scale}"
            )
        valid_cfg_norms = {"none", "layer_norm", "scalar"}
        if self.teacher_cfg_norm not in valid_cfg_norms:
            raise ValueError(
                f"Invalid teacher_cfg_norm: {self.teacher_cfg_norm}. Must be one of {sorted(valid_cfg_norms)}"
            )
        if self.rollout_timestep_shift < 1:
            raise ValueError(f"rollout_timestep_shift must be at least 1, got {self.rollout_timestep_shift}")
        if not 0.0 <= self.score_sigma_min < self.score_sigma_max <= 1.0:
            raise ValueError(
                "score sigma bounds must satisfy 0 <= score_sigma_min < score_sigma_max <= 1, "
                f"got [{self.score_sigma_min}, {self.score_sigma_max}]"
            )
        if self.score_timestep_shift < 1:
            raise ValueError(f"score_timestep_shift must be at least 1, got {self.score_timestep_shift}")
        if self.score_discrete_steps < 0:
            raise ValueError(f"score_discrete_steps must be non-negative, got {self.score_discrete_steps}")
        if self.normalization_epsilon <= 0:
            raise ValueError(f"normalization_epsilon must be positive, got {self.normalization_epsilon}")
        if self.dmd_loss_weight < 0:
            raise ValueError(f"dmd_loss_weight must be non-negative, got {self.dmd_loss_weight}")
        valid_regression_types = {"decoded_lpips", "latent_mse"}
        if self.regression_type not in valid_regression_types:
            raise ValueError(
                f"Invalid regression_type: {self.regression_type}. Must be one of {sorted(valid_regression_types)}"
            )
        if self.regression_loss_weight < 0:
            raise ValueError(f"regression_loss_weight must be non-negative, got {self.regression_loss_weight}")
        if self.recipe == "dmd2" and self.dmd_loss_weight == 0:
            raise ValueError("DMD2 requires dmd_loss_weight > 0")
        if self.recipe == "dmd" and self.dmd_loss_weight == 0 and self.regression_loss_weight == 0:
            raise ValueError("DMD requires at least one positive objective weight")
        if self.rng_seed < 0:
            raise ValueError(f"rng_seed must be non-negative, got {self.rng_seed}")


@dataclass
class DiffusionDistillationConfig(BaseConfig):
    """Diffusion distillation settings shared by OPD and DMD-family routing.

    ``enabled`` and the teacher-pool fields remain exclusive to OPD. DMD-family
    training is selected by ``algorithm.trainer_type=distillation`` and reads the
    nested ``distribution_matching`` config while keeping ``enabled=false``.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool. 0 colocates the teachers with the actor.
    teacher_models (dict[str, DiffusionDistillationTeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=ocr
    distillation.teacher_models.teacher_model.model_path=/ckpt/ocr_teacher
    +distillation.teacher_models.teacher_model2.key=aesthetic
    +distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=ocr
    +distillation.teacher_models.teacher_model1.model_path=/ckpt/ocr_teacher
    +distillation.teacher_models.teacher_model2.key=aesthetic
    +distillation.teacher_models.teacher_model2.model_path=/ckpt/aesthetic_teacher
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "distribution_matching"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DiffusionDistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    # DMD-family recipe settings; selected by algorithm.trainer_type rather than enabled.
    distribution_matching: DiffusionDistributionMatchingConfig = field(
        default_factory=DiffusionDistributionMatchingConfig
    )

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models = self._resolve_teacher_models()
        if self.nnodes > 0:
            teacher_world_size_sum = sum(teacher_model.world_size for teacher_model in self.teacher_models.values())
            total_pool_size = self.n_gpus_per_node * self.nnodes
            if teacher_world_size_sum != total_pool_size:
                raise ValueError(
                    f"Sum of teacher world_size ({teacher_world_size_sum}) must match "
                    f"the distillation resource pool size "
                    f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
                )

    def _resolve_teacher_models(self) -> dict[str, DiffusionDistillationTeacherModelConfig]:
        from verl.utils.config import omega_conf_to_dataclass

        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            teacher_model.world_size = self.n_gpus_per_node * self.nnodes
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(
                teacher_config, dataclass_type=DiffusionDistillationTeacherModelConfig
            )
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models

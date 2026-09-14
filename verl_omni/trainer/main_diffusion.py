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
"""Entrypoint for diffusion model RL training."""

import os

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available

from verl_omni.trainer.diffusion.ray_diffusion_trainer import validate_separate_config
from verl_omni.trainer.diffusion.task_runner import TaskRunner, get_diffusion_trainer_cls
from verl_omni.utils.config import validate_config
from verl_omni.utils.diffusion_attention import validate_attention_consistency
from verl_omni.utils.rl_insight import enable_rl_insight


def _count_controller_capture_ranges(profile_steps: list[int], profile_continuous_steps: bool) -> int:
    """Return the number of CUDA profiler capture ranges emitted by the controller."""
    steps = sorted(set(profile_steps))
    if not profile_continuous_steps:
        return len(steps)
    return sum(index == 0 or step != steps[index - 1] + 1 for index, step in enumerate(steps))


def _resolve_controller_nsight_options(config) -> dict:
    """Resolve controller Nsight options for the configured profiling steps."""
    nsight_options = OmegaConf.to_container(config.global_profiler.global_tool_config.nsys.controller_nsight_options)
    if nsight_options.get("capture-range") == "cudaProfilerApi" and nsight_options.get("capture-range-end") is None:
        capture_count = _count_controller_capture_ranges(
            OmegaConf.select(config, "global_profiler.steps"),
            OmegaConf.select(config, "global_profiler.profile_continuous_steps", default=False),
        )
        nsight_options["capture-range-end"] = f"repeat-shutdown:{capture_count}"
    return nsight_options


@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for diffusion model training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    OmegaConf.resolve(config)
    validate_config(config)
    validate_attention_consistency(config)
    run_diffusion(config)


def _determinism_requested(config) -> bool:
    """Whether reward inference determinism is requested."""
    rm_rollout = config.reward.reward_model.rollout
    return bool(config.reward.reward_model.get("enable", False) and rm_rollout.get("full_determinism", False))


def _export_full_determinism_env(config) -> None:
    """Set determinism switch env vars before ray.init() so actors inherit them."""
    os.environ["VERL_FULL_DETERMINISM"] = "1"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["PYTHONHASHSEED"] = str(config.reward.reward_model.rollout.get("seed", 42))


def _validate_grm_reward_function(config) -> None:
    """Require an explicit reward function when the RM is enabled."""
    rm_cfg = config.reward.reward_model
    if not rm_cfg.get("enable", False):
        return
    crf = config.reward.custom_reward_function
    if not crf.get("path"):
        raise ValueError(
            "reward.reward_model.enable=true requires reward.custom_reward_function.path. "
            "For GRM OCR scoring set it to 'verl_omni/utils/reward_score/genrm_ocr.py' with name 'compute_score_ocr'."
        )


def run_diffusion(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed diffusion training.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed diffusion training including Ray initialization
                settings, model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    OmegaConf.resolve(config)
    validate_separate_config(config)
    enable_rl_insight(config)
    _validate_grm_reward_function(config)
    # Before ray.init() so actors inherit these via runtime_env.
    if _determinism_requested(config):
        _export_full_determinism_env(config)

    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # When NVIDIA Nsight Systems is selected for the controller, launch the TaskRunner under nsys
    # using the Ray runtime env, mirroring verl/trainer/main_ppo.py.
    if (
        is_cuda_available
        and OmegaConf.select(config, "global_profiler.tool") == "nsys"
        and OmegaConf.select(config, "global_profiler.steps") is not None
        and len(OmegaConf.select(config, "global_profiler.steps")) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = _resolve_controller_nsight_options(config)
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _get_trainer_cls(config):
    """Return the shared algorithm trainer selection (legacy import compatibility)."""
    return get_diffusion_trainer_cls(config)


if __name__ == "__main__":
    main()

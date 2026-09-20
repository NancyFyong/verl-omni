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
"""Execute the six shell entrypoints with a fake Python, then compose real Hydra configs."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).parents[2]


def _launch(tmp_path, algorithm, task, mode, overrides=(), environment=None):
    model = tmp_path / "model"
    for part in ("FL2VA", "Ref2VA", "transformer", "transformer_ref"):
        (model / part).mkdir(parents=True, exist_ok=True)
    capture = tmp_path / "args"
    python = tmp_path / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE"\n')
    python.chmod(0o755)
    family = "flowgrpo_trainer" if algorithm == "flow_grpo" else "diffusionnft_trainer"
    script = ROOT / "examples" / family / "minimax_h3" / f"run_minimax_h3_{task}_lora_batch.sh"
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "MODEL_PATH": str(model),
        "DATA_DIR": str(tmp_path / "data"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "CAPTURE": str(capture),
        "ROLLOUT_MODE": mode,
    } | (environment or {})
    result = subprocess.run(["bash", str(script), *overrides], env=env, capture_output=True, text=True, timeout=30)
    args = capture.read_bytes().decode().rstrip("\0").split("\0") if capture.exists() else None
    return result, args, model


def _compose(args):
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion"]
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        return compose(config_name="diffusion_trainer", overrides=args[2:])


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("mode", ["request", "stepwise"])
def test_batch_recipe_composes_with_v1_hyperparameters(tmp_path, algorithm, task, mode):
    result, args, model = _launch(tmp_path, algorithm, task, mode)
    assert result.returncode == 0, result.stderr
    cfg = _compose(args)
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    partition = "Ref2VA" if task == "ref2va" else "FL2VA"
    actor_subdir = "transformer_ref" if task == "ref2va" else "transformer"
    assert cfg.actor_rollout_ref.model.path == str(model / partition)
    assert cfg.actor_rollout_ref.model.config_path == str(model / actor_subdir)
    assert cfg.actor_rollout_ref.model.algorithm == algorithm
    assert cfg.actor_rollout_ref.model.lora_rank == 64
    assert cfg.actor_rollout_ref.model.lora_alpha == 128
    assert cfg.data.train_batch_size == 32
    assert actor.optim.lr == pytest.approx(3e-4)
    assert actor.optim.weight_decay == pytest.approx(1e-4)
    assert actor.ppo_mini_batch_size == 16
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert not actor.fsdp_config.param_offload and not actor.fsdp_config.optimizer_offload
    assert not cfg.trainer.use_v1
    assert rollout.tensor_model_parallel_size == rollout.text_encoder_tp_size == 2
    assert cfg.trainer.n_gpus_per_node == 8
    assert rollout.n == 8
    assert rollout.max_num_seqs == 2
    assert rollout.step_execution == (mode == "stepwise")
    assert rollout.enforce_eager
    assert rollout.rollout_attn_backend == "FLASH_ATTN"
    assert cfg.actor_rollout_ref.model.attn_backend == "_flash_3_varlen_hub"
    assert rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms == 50
    assert not rollout.engine_kwargs.vllm_omni.enable_layerwise_offload
    assert not rollout.engine_kwargs.vllm_omni.enable_cpu_offload
    assert rollout.pipeline.task == rollout.val_kwargs.pipeline.task == task
    assert (rollout.pipeline.height, rollout.pipeline.width, rollout.pipeline.num_frames) == (256, 384, 121)
    assert (rollout.val_kwargs.pipeline.height, rollout.val_kwargs.pipeline.width) == (512, 768)
    assert rollout.pipeline.num_inference_steps == 10 and rollout.val_kwargs.pipeline.num_inference_steps == 40
    assert cfg.trainer.total_training_steps == 100
    assert cfg.trainer.save_freq == cfg.trainer.test_freq == 10
    assert cfg.reward.reward_functions.clap.required
    assert cfg.reward.reward_functions.imagebind.required
    if task == "fl2va":
        assert list(rollout.pipeline.frame_indices) == list(rollout.val_kwargs.pipeline.frame_indices) == [0]
    if task == "ref2va":
        assert rollout.max_prompt_embed_length == rollout.pipeline.max_sequence_length == 12288
        assert rollout.pipeline.reference_image_short_edge == 2048
    if algorithm == "diffusion_nft":
        assert cfg.algorithm.trainer_type == "direct_preference"
        assert actor.diffusion_loss.loss_mode == "diffusion_nft"
        assert actor.diffusion_loss.mix_beta == pytest.approx(0.1)
        assert cfg.algorithm.old_policy_update_interval == 2
        assert list(cfg.actor_rollout_ref.model.policy_state_adapters) == ["default", "old"]
        assert rollout.rollout_adapter == "old" and not rollout.calculate_log_probs
    else:
        assert cfg.algorithm.trainer_type == "policy_gradient"
        assert rollout.calculate_log_probs
        assert rollout.algo.sde_type == "cps" and rollout.algo.noise_level == pytest.approx(0.8)
        assert list(rollout.algo.sde_window_range) == [0, 8] and rollout.algo.sde_window_size == 3


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_batch_recipe_cli_overrides_environment(tmp_path, algorithm):
    result, args, _ = _launch(
        tmp_path,
        algorithm,
        "fl2va",
        "request",
        overrides=["actor_rollout_ref.rollout.max_num_seqs=4", "actor_rollout_ref.actor.optim.lr=1e-4"],
        environment={
            "MAX_NUM_SEQS": "3",
            "REQUEST_BATCH_MAX_WAIT_MS": "25",
            "NUM_GPUS": "4",
            "CLAP_MODEL_PATH": "/models/clap",
            "IMAGEBIND_MODEL_PATH": "/models/imagebind.pth",
        },
    )
    assert result.returncode == 0, result.stderr
    cfg = _compose(args)
    assert cfg.actor_rollout_ref.rollout.max_num_seqs == 4
    assert cfg.actor_rollout_ref.actor.optim.lr == pytest.approx(1e-4)
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms == 25
    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.reward.reward_functions.clap.model_name_or_path == "/models/clap"
    assert cfg.reward.reward_functions.imagebind.model_name_or_path == "/models/imagebind.pth"


@pytest.mark.parametrize("environment", [{"ROLLOUT_MODE": "invalid"}, {"NUM_GPUS": "3"}, {"MAX_NUM_SEQS": "0"}])
def test_invalid_batch_recipe_fails_before_launch(tmp_path, environment):
    result, args, _ = _launch(tmp_path, "flow_grpo", "fl2va", "request", environment=environment)
    assert result.returncode != 0
    assert args is None

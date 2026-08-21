#!/usr/bin/env bash
# MiniMax H3 FlowGRPO parity on the accelerator (1 GPU, no weights).
#
# Replays the CPU parity checks with the schedulers, sigmas and latents on CUDA:
# the capture loop vs vllm-omni's denoise loop at eta=0, and the training adapter
# vs the rollout log probs at eta>0. Catches device-placement regressions that the
# CPU job cannot see.

set -euo pipefail

export H3_PARITY_DEVICE=cuda

python3 -m pytest -s tests/pipelines/test_minimax_h3_flow_grpo_parity_on_cpu.py

#!/usr/bin/env bash
# MiniMax-H3 FL2VA diffusion_nft: ROLLOUT_MODE=request or stepwise.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/../../minimax_h3/run_batched_lora.sh" diffusion_nft fl2va "$@"

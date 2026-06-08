#!/usr/bin/env bash
# Launch one PickScore HTTP scorer server per visible GPU.
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
LOG_DIR=${LOG_DIR:-$REPO_ROOT/outputs/pickscore_http_server/logs}
HOST=${HOST:-0.0.0.0}
BASE_PORT=${BASE_PORT:-19084}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
WAIT_MS=${WAIT_MS:-20}
GPU_IDS=${GPU_IDS:-4,5,6,7}

mkdir -p "$LOG_DIR"
IFS=',' read -ra GPUS <<< "$GPU_IDS"

pids=()
for idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$idx]}
    port=$((BASE_PORT + idx))
    log_file="$LOG_DIR/pickscore_gpu${gpu}_port${port}.log"
    echo "Starting PickScore server on GPU $gpu, port $port -> $log_file"
    CUDA_VISIBLE_DEVICES=$gpu \
        python3 "$SCRIPT_DIR/pickscore_http_server.py" \
        --host "$HOST" \
        --port "$port" \
        --device cuda \
        --max-batch-size "$MAX_BATCH_SIZE" \
        --micro-batch-size "$MICRO_BATCH_SIZE" \
        --wait-ms "$WAIT_MS" \
        > "$log_file" 2>&1 &
    pids+=("$!")
done

echo "Started PickScore server PIDs: ${pids[*]}"
echo "Server URLs:"
for idx in "${!GPUS[@]}"; do
    echo "  http://127.0.0.1:$((BASE_PORT + idx))"
done

wait "${pids[@]}"

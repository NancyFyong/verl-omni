# MiniMax-H3 batched rollout recipes

Last updated: 09/20/2026

These opt-in examples cover T2VA, FL2VA and Ref2VA with FlowGRPO or DiffusionNFT.
All use `verl_omni.trainer.main_diffusion` (not V1/TransferQueue). They share
`run_batched_lora.sh`, which invokes the existing task/algorithm recipe with
common hyperparameters from the FL2VA FlowGRPO V1 example and explicit batching
settings. Existing non-batched launchers retain their single-request defaults.

## Entrypoints

| Task | FlowGRPO | DiffusionNFT |
| --- | --- | --- |
| T2VA | [run_minimax_h3_t2va_lora_batch.sh](../flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_batch.sh) | [run_minimax_h3_t2va_lora_batch.sh](../diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora_batch.sh) |
| FL2VA | [run_minimax_h3_fl2va_lora_batch.sh](../flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora_batch.sh) | [run_minimax_h3_fl2va_lora_batch.sh](../diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora_batch.sh) |
| Ref2VA | [run_minimax_h3_ref2va_lora_batch.sh](../flowgrpo_trainer/minimax_h3/run_minimax_h3_ref2va_lora_batch.sh) | [run_minimax_h3_ref2va_lora_batch.sh](../diffusionnft_trainer/minimax_h3/run_minimax_h3_ref2va_lora_batch.sh) |

Each entrypoint supports both execution modes. From the repository root, with
your training uv environment activated:

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/fl2va/parquets
export NUM_GPUS=4
export ROLLOUT_TP=2
export CLAP_MODEL_PATH=/path/to/clap
export IMAGEBIND_MODEL_PATH=/path/to/imagebind_huge.pth

# Whole-request batching, up to two requests per rollout replica.
ROLLOUT_MODE=request MAX_NUM_SEQS=2 REQUEST_BATCH_MAX_WAIT_MS=50 \
  bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_fl2va_lora_batch.sh

# Stepwise scheduling and packing, with the NFT old-policy adapter.
ROLLOUT_MODE=stepwise MAX_NUM_SEQS=2 \
  bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora_batch.sh
```

For **all six new entrypoints**, `MODEL_PATH` is the checkpoint **root**:
T2VA/FL2VA use `FL2VA/` and `transformer/`; Ref2VA uses `Ref2VA/` and
`transformer_ref/`. `DATA_DIR` must contain `train.parquet` and `test.parquet` for
the selected task. Use the data preparation instructions in the
[FlowGRPO](../flowgrpo_trainer/minimax_h3/README.md) or
[NFT](../diffusionnft_trainer/minimax_h3/README.md) README; these scripts do not
create data or download production model/reward weights.

`OUTPUT_DIR` defaults to `outputs/minimax_h3_<task>_<algorithm>_batch_<mode>`.
Set it explicitly for independent experiments. The base recipes retain their
logging, checkpoints, validation and reward implementations.

## Shared hyperparameters

| Setting | Default |
| --- | --- |
| GPUs / rollout TP / text-encoder TP | 8 / 2 / 2 |
| LoRA rank / alpha / learning rate | 64 / 128 / 3e-4 |
| Weight decay | 1e-4 |
| Prompt batch / PPO minibatch / per-GPU microbatch | 32 / 16 / 1 |
| Responses per prompt (`ROLLOUT_N`) | 8 |
| Train height × width / requested frames | 256 × 384 / 121 |
| Validation height × width / inference points | 512 × 768 / 40 |
| Training inference points (`INFER_STEPS`) | 10 |
| Training updates / maximum epochs | 100 / 15 |
| Checkpoint interval / retained checkpoints / validation interval | 10 / 1 / 10 |
| Actor gradient checkpointing / parameter and optimizer offload | on / off |

FlowGRPO retains CPS noise level 0.8, SDE window size 3, range `[0,8]`, contiguous
selection and window seed 42. `SDE_WINDOW_SIZE` and `SDE_WINDOW_END` can override
the window when changing `INFER_STEPS`. NFT retains its own loss, `default`/`old`
adapters, `rollout_adapter=old`, old-policy update interval 2 and decay schedule;
it emits clean latents and does **not** use FlowGRPO replay log probabilities.

Task-specific exceptions are intentional: FL2VA keeps `FRAME_INDICES=[0]`;
Ref2VA retains the reference recipe's 4096-token input limit, 12288 embedding
capacity (`REF_MAX_PROMPT_EMBEDS`) and reference-image short-edge settings.
`ROLLOUT_N` creates independent requests; each H3 request still produces one output.

## Batching controls

| Environment variable | Default | Hydra setting |
| --- | --- | --- |
| `ROLLOUT_MODE` | `request` | `rollout.step_execution=False`; `stepwise` selects `True` |
| `MAX_NUM_SEQS` | `2` | `actor_rollout_ref.rollout.max_num_seqs` |
| `REQUEST_BATCH_MAX_WAIT_MS` | `50` | `actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms` |
| `ACTOR_ATTN_BACKEND` | `_flash_3_varlen_hub` | `actor_rollout_ref.model.attn_backend` |
| `ROLLOUT_ATTN_BACKEND` | `FLASH_ATTN` | `actor_rollout_ref.rollout.rollout_attn_backend` |

The wait time collects a whole-request batch; it does not force a stepwise batch
to fill. `max_num_seqs` is a ceiling, not a guaranteed batch size.
On the validated H20 environment, `FLASH_ATTN` loads `fa3_fwd_interface` and
supports isolated multi-document attention in both the DiT and token refiner.
`FLASH_ATTN_3_HUB` is **not** interchangeable for this test: its current capability
gate makes H3 fall back to separate forwards. Install compatible Hopper FA3 and
Hub actor kernels; no dependency pin or kernel implementation is changed here.

The profile starts in eager mode and disables inherited rollout CPU/layerwise
offload. Distributed layerwise offload, cache acceleration and `quality=high`
are unsupported for interleaved requests. Do not replace LoRA weights while
requests are in flight; use the existing trainer's rollout-completion barrier.
Large models/batches may need a smaller `MAX_NUM_SEQS` or different GPU topology.
Do not assume the eight-GPU profile fits on two GPUs.

Additional Hydra arguments are forwarded last, overriding profile defaults:

```bash
ROLLOUT_MODE=stepwise bash examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_batch.sh \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.actor.optim.lr=1e-4
```

## Validation scope

All six entrypoints and both modes have shell-execution/Hydra-composition CPU
coverage, including override precedence and invalid settings. T2VA completed
two-GPU, two-update tiny-checkpoint runs for both algorithms and both modes under
SDPA and real packed FlashAttention. GPU captures verified two requests per DiT
forward, refiner/main-block kernel execution, finite payloads and document isolation.
NFT's matched request/step outputs were identical; FlowGRPO and Flash/SDPA
comparisons had small nonzero differences, not bitwise trajectory equivalence.

FL2VA and Ref2VA have CPU scheduler/payload/isolation coverage, **not equivalent
GPU qualification from this experiment**. The new production-profile scripts
were configuration-tested, not run with production weights. No convergence,
video quality or production throughput claim follows from the tiny tests.
See the task READMEs for the reproducible tiny-run commands (`--attention flash`).

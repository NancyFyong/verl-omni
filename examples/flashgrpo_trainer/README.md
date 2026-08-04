# Flash-GRPO Trainer

Training recipes using the **Flash-GRPO** algorithm — FlowGRPO with temporal
gradient rectification (per-timestep `coe` weighting that balances gradient
magnitude across denoising steps).

Reference: [Flash-GRPO paper](https://arxiv.org/abs/2605.15980) and
[code](https://github.com/Shredded-Pork/Flash-GRPO).

## Wan2.2 T2V with HPSv3 reward

```bash
# 8-GPU single-node
bash examples/flashgrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_flash.sh

# Override GPU count / TP
NUM_GPUS=16 ROLLOUT_TP=2 bash examples/flashgrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_flash.sh
```

### Key configuration differences from DanceGRPO

| Config | DanceGRPO | Flash-GRPO |
|--------|-----------|------------|
| `algorithm.adv_estimator` | `dance_grpo` | `flash_grpo` |
| `model.algorithm` | `dance_grpo` | `flash_grpo` |
| `diffusion_loss.loss_mode` | `dance_grpo` | `flash_grpo` |
| `rollout.algo.sde_type` | `dance_sde` | **`sde`** |

The `sde_type` must be `sde` (not `dance_sde`) because the `coe` temporal
weight is derived for the FlowGRPO SDE density form (`sigma_noise = std_dev_t *
sqrt(-dt)`). Using `dance_sde` with `flash_grpo` will trigger a warning and
silently degenerate to plain FlowGRPO (no temporal rectification).

### Current scope

This recipe implements Flash-GRPO's **temporal gradient rectification** (the
`coe` weighting). The paper's other two innovations — single-step training and
iso-temporal grouping — are planned for Phase 2 and are not yet active. Training
still iterates over the full `sde_window`, but with balanced per-timestep
gradients.

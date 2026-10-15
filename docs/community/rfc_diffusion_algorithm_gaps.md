# [RFC] Call for contributions: diffusion algorithms still missing from verl-omni

Last updated: 10/15/2026.

Related: [#717](https://github.com/verl-project/verl-omni/issues/717),
[#520](https://github.com/verl-project/verl-omni/issues/520),
[#700](https://github.com/verl-project/verl-omni/issues/700).

## Summary

This RFC asks for help implementing diffusion post-training algorithms that
[Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) already runs and that
verl-omni does not, plus **DiffusionOPSD**. DiffusionOPSD is not a Flow-Factory
trainer. It is on this list because we want it first.

**DiffusionOPSD is P0. AWM is P1.** Comment on this RFC before starting, so two
people do not take the same algorithm.

A contribution has the same shape as
[#717](https://github.com/verl-project/verl-omni/issues/717): one algorithm, one
PR, a registered loss, a recipe on a model we already support, a CPU test, and a
small GPU run with a reward curve.

## Motivation

Flow-Factory's trainer registry on `main` (read 2026-10-15) ships fourteen
`trainer_type` values. verl-omni `main` on the same date already has recipes for
the policy-gradient and preference methods we train today: Flow-GRPO,
Dance-GRPO, MixGRPO, GRPO-Guard, Flow-DPPO, online and offline Diffusion-DPO,
DiffusionNFT, DGPO, and DiffusionOPD.

These holes are still open:

| Priority | Algorithm | Flow-Factory `trainer_type` | Why it is on this list |
| --- | --- | --- | --- |
| P0 | DiffusionOPSD | — | On-policy self-distillation. Image-level reward gradients become explicit clean-prediction targets. |
| P1 | AWM | `awm` | Advantage-weighted flow matching. `DirectPreferenceRayTrainer` names it, and there is no loss or recipe. |
| open | CRD | `crd` | Centered reward distillation against a CFG teacher. |
| open | TDM | `tdm` | Trajectory distribution matching for few-step generators. |
| open | SFT | `sft` | Offline supervised flow matching on demonstration media. |

The open rows are welcome, not a required menu. Other published algorithms are
welcome too. DGPO, TDM-R1, DMDR, and the DMD2 loss already have an owner or a
merged PR. Leave those threads where they are.

## Goals / Non-goals

**Goals**

- Land DiffusionOPSD (P0) and AWM (P1) as runnable LoRA recipes, with a CPU test
  and a short GPU reward curve.
- Give each remaining Flow-Factory hole one owner, claimed in a comment here.
- Reuse the loss registry, an existing trainer, and an existing pipeline.
  SD3.5-Medium is the default first recipe.

**Non-goals**

- Reimplementing Flow-GRPO, Dance-GRPO, MixGRPO, GRPO-Guard, Flow-DPPO,
  Diffusion-DPO, DiffusionNFT, DGPO, or DiffusionOPD.
- Finishing the DMD2 trainer. [#520](https://github.com/verl-project/verl-omni/issues/520)
  owns the DMD2 loss and standalone config. A TDM contribution has to say how
  it shares that work.
- Taking TDM-R1 or DMDR. Volunteers already claimed both on
  [#717](https://github.com/verl-project/verl-omni/issues/717).
- Adding a new model family in the first PR. Z-Image-Turbo is a DiffusionOPSD
  paper backbone and stays a follow-up. SD3.5-Medium is the first recipe.
- Matching Flow-Factory or the papers' full tables. A small run whose training
  reward moves is enough to open review.

## Guide-level explanation

### How to claim an item

Comment here with the algorithm, the pipeline, and the GPUs you can run. Wait
for a maintainer to confirm, then open one implementation PR.

A claim can follow the DGPO note on #717: registry entry, CPU test, SD3.5-Medium
LoRA recipe, then a small GPU run and a reward curve.

### What a good contribution looks like

- **Algorithm + config** in
  [`verl_omni/trainer/diffusion/diffusion_algos.py`](../../verl_omni/trainer/diffusion/diffusion_algos.py)
  and `DiffusionLossConfig.valid_modes`, following FlowGRPO, DiffusionNFT, and
  DGPO.
- **A recipe** under `examples/` on a pipeline we already support. Prefer
  SD3.5-Medium LoRA. Qwen-Image is appropriate when the reference config is
  closer to that stack.
- **A CPU test** named `test_*_on_cpu.py`.
- **GPU validation** on a small run, with a reward curve in the PR
  (`reward_curve.png` or a tracker screenshot).
- **A short algorithm page** in `docs/algo/`, linked from `docs/index.md`.
- A comparison against the closest method we already ship is welcome and is
  optional. For DiffusionOPSD that neighbor is DiffusionNFT. For AWM it is
  online DPO or DiffusionNFT.

Integration guides:

- Forward-process objectives, including AWM:
  [integrating a direct-preference algorithm](../contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md).
- Reverse-process ratio objectives:
  [integrating a policy-gradient algorithm](../contributing/integrating_a_new_policy_gradient_algorithm_for_diffusion_model.md).
  DiffusionOPSD supervises clean predictions at a query state. The
  policy-gradient guide does not cover it.

### Already in verl-omni

Checked against the `register_diffusion_loss` registry and `docs/algo/` on
verl-omni `main`, 2026-10-15.

| Flow-Factory | verl-omni | Where |
| --- | --- | --- |
| GRPO (`Flow-SDE`, `Dance-SDE`, `CPS`, mixed SDE/ODE steps) | `flow_grpo`, `dance_grpo`, `mix_grpo`; scheduler `sde` / `dance_sde` / `cps` | `docs/algo/flowgrpo.md`, `docs/algo/mixgrpo.md` |
| GRPO-Guard | `grpo_guard` | `docs/algo/grpo_guard.md` |
| DPPO | `flow_dppo` | `docs/algo/flowdppo.md`. The V1 sync recipe is still open in [#690](https://github.com/verl-project/verl-omni/pull/690). |
| Online DPO and offline DPO | `dpo` | `docs/algo/diffusion_dpo.md` |
| DiffusionNFT | `diffusion_nft` | `docs/algo/diffusionnft.md` |
| DGPO | `dgpo` | `docs/algo/dgpo.md`, merged in [#733](https://github.com/verl-project/verl-omni/pull/733) |
| DiffusionOPD | `distill_kl`, `distill_fm_mse` | `docs/algo/diffusion_opd.md` |
| DMD2 | `dmd2` loss and `DiffusionDMDConfig` only | [#520](https://github.com/verl-project/verl-omni/issues/520). No end-to-end trainer or example. |

`BagelForSFT` is a BAGEL forward primitive. Flow-Factory `sft` is a separate
offline trainer and is still open below.

### Already claimed elsewhere

- **TDM-R1** and **DMDR**: volunteers on
  [#717](https://github.com/verl-project/verl-omni/issues/717). Continue that
  thread.
- **DMD2 loss and config**: [#520](https://github.com/verl-project/verl-omni/issues/520).
- **DGPO**: merged ([#733](https://github.com/verl-project/verl-omni/pull/733)).

## Reference-level explanation

### P0 — DiffusionOPSD

**Paper:** [On-Policy Self-Distillation in Diffusion Models](https://arxiv.org/abs/2608.24646)
· **Code:** [worldbench/DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD)
· **Site:** [diffusionopsd.github.io](https://diffusionopsd.github.io/)

A frozen behavior policy rolls out a trajectory and keeps a low-noise query
plus a clean-output anchor. Normalized reward ascent and reward descent build a
positive target and a negative target inside a bounded radius of that anchor.
The trainable policy fits the detached targets. An EMA copies the updated
weights back into the behavior policy, and the next round of queries is
collected again.

The public objective is a weighted sum of positive and negative
clean-prediction residuals, with detached mean-absolute normalizers. The
reference implementation scales that loss by `c_adv = 5`. The maintained
recipe (v2) uses one reward-ascent step, one trajectory per prompt, and a
rollout that stops at the query. The first verl-omni recipe should follow that
v2 path and say so in the algorithm page. The paper's two-step ascent is the
v1 setting.

DiffusionOPD, which verl-omni already trains, matches a student transition mean
to a frozen task teacher (`distill_kl`). DiffusionOPSD has no external teacher.
Its targets are reward gradients around the behavior policy's own anchor.
Pointing `distill_kl` at an EMA copy does not implement this algorithm.

Suggested split, matching the reference loop:

1. **Rollout.** The behavior policy stores the low-noise query and the clean
   anchor. Under v2 the loss does not need a decoded endpoint when the group
   weight is fixed.
2. **Target construction.** A differentiable reward produces bounded positive
   and negative clean targets. Drop the reward graph before the policy step.
3. **Fit.** One optimizer step on the detached-target loss. Register it as
   `diffusion_opsd`.
4. **Refresh.** EMA the trainable weights into the behavior policy.

The training reward has to be differentiable in the image or latent that the
ascent steps on. A VLM-as-judge OCR score is an evaluation metric here. Start
from a reward the reference repo already differentiates (PickScore, CLIP, or
HPSv2.1) and keep the existing black-box judges for eval.

First recipe: SD3.5-Medium LoRA and one of those differentiable rewards. Put
the loop on the V1 trainer when the batch contract fits. Z-Image-Turbo stays a
follow-up; verl-omni has no Z-Image pipeline.

CPU test: a fixed anchor, a fixed reward step, and fixed normalizers, checked
against the reference residual. That test should not load a reward model or a
GPU.

### P1 — AWM

**Paper:** [Advantage Weighted Matching](https://arxiv.org/abs/2509.25050)
· **Code:** Flow-Factory
[`awm`](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/trainers/rl/awm.py),
SD3.5 config
[`examples/awm/lora/sd3_5/default.yaml`](https://github.com/X-GenGroup/Flow-Factory/blob/main/examples/awm/lora/sd3_5/default.yaml)

AWM keeps the flow-matching velocity loss and weights it by the group
advantage. Sampling can stay on a deterministic ODE. Training noises the final
clean latent at freshly drawn timesteps. It sits in the same decoupled family
as DiffusionNFT and DGPO, with its own objective.

The Flow-Factory SD3.5 PickScore file is the reference point for the first
recipe. Read the values from that file rather than from the older guidance
snippet. On 2026-10-15 it sets:

- `awm_weighting: ghuber`, `ghuber_power: 0.25`
- ODE sampling, 14 train steps, 40 eval steps, resolution 512, guidance 1.0
- `ema_kl_beta: 1`, `ema_decay: 0.3`, `kl_beta: 0.0001`, v-based KL
- `clip_range: 1`, `adv_clip_range: 1.0`
- LoRA rank 32, PickScore

Flow-Factory's algorithm guide says AWM diverges when both the EMA KL and the
clip are turned off. Keep both in the first recipe, at the values in the
config above.

Extend `DirectPreferenceRayTrainer`. Its docstring already lists AWM. The
missing pieces are the weighted matching loss, the sampling-policy log-prob
used for the clip, and the EMA KL term. Register the loss as `awm` and add it
to `DiffusionLossConfig.valid_modes`.

CPU test: one batch of velocities, advantages, and sigmas, checked against the
ghuber-weighted clipped objective. Cover the EMA KL with a second case that
uses a frozen velocity.

### Open items

**CRD — Centered Reward Distillation.** Paper
[2603.14128](https://arxiv.org/abs/2603.14128). Flow-Factory `crd`. Implicit
rewards come from the velocity error of an old parameter snapshot, external
rewards are centered inside the prompt group, and a KL term anchors the policy
to a CFG teacher. The trainer keeps an old snapshot and a sampling snapshot
beside the trainable weights. A single call to the DPO or NFT loss does not
cover that.

**TDM — Trajectory Distribution Matching.** Paper
[2503.06674](https://arxiv.org/abs/2503.06674). Flow-Factory `tdm`. A few-step
ODE generator, a fake score, and a frozen real score, with a loss on every
trajectory boundary. Flow-Factory's `TDMTrainer` sits on the DMD2 role loop.
verl-omni has the DMD2 surrogate and the fake-score loss, and it has no trainer
that alternates those roles
([#520](https://github.com/verl-project/verl-omni/issues/520)). A TDM PR should
either wait for that loop or include the shared multi-role step and say which
choice it made. TDM-R1
([2603.07700](https://arxiv.org/abs/2603.07700)) adds a non-differentiable
reward on top of TDM and is already claimed on #717.

**SFT — supervised flow matching.** Flow-Factory `sft`. The trainer reads
demonstration media, encodes targets on demand, and applies a velocity loss at
sampled timesteps. There is no rollout and no training reward. The data path is
the offline DPO parquet path, with demonstration rows in place of win/lose
pairs. Leave `BagelForSFT` as the BAGEL architecture primitive it already is.

## Drawbacks

- DiffusionOPSD needs a differentiable reward. A recipe that only calls a
  black-box judge is a different algorithm and should use a different name.
- AWM's published SD3.5 config sets `ema_kl_beta: 1`. Dropping that term will
  not reproduce the reference run.
- `DirectPreferenceRayTrainer` fits AWM, because the batch is final latents and
  a forward objective. DiffusionOPSD also carries a query state, reward-ascent
  targets, and a behavior-policy EMA. If that batch does not match the NFT
  contract, give it a trainer hook of its own.

## Rationale and alternatives

Appending these rows to #717 was considered. #717 stays the list for TDM-R1 and
DMDR, and its DGPO item has landed. This RFC is the Flow-Factory gap list, with
DiffusionOPSD at P0 and AWM at P1, so new claims register in one place.

Putting every missing trainer in one patch was rejected. The review size that
has landed for this kind of work is one algorithm per PR, as with DGPO in
[#733](https://github.com/verl-project/verl-omni/pull/733).

## Implementation plan

- [ ] **P0 DiffusionOPSD.** `diffusion_opsd` loss, behavior-policy EMA, detached
      target construction, SD3.5-Medium LoRA recipe, CPU test, short GPU reward
      curve, `docs/algo/diffusion_opsd.md`.
- [ ] **P1 AWM.** `awm` loss, EMA KL and clip, SD3.5-Medium PickScore LoRA
      recipe aligned with Flow-Factory
      `examples/awm/lora/sd3_5/default.yaml`, CPU test, short GPU reward curve,
      `docs/algo/awm.md`.
- [ ] **CRD, TDM, SFT.** Unprioritized. Comment here with a one-paragraph plan
      before writing code.

## Unresolved questions

1. DiffusionOPSD v2 fixes the group weight and can stop the rollout at the
   query. Is v2 the only supported operating point, or should the first PR also
   keep the paper's two-step ascent as an option?
2. Which differentiable reward ships in the P0 recipe: PickScore, CLIP, or
   HPSv2.1? PickScore is already a verl-omni scorer. Today's scorer returns a
   scalar and does not return an image gradient.
3. Should AWM's first recipe stay on the SD3.5 PickScore config above, or move
   to the Qwen-Image OCR stack we use for FlowGRPO comparisons?
4. Does TDM wait for a runnable DMD2 trainer, or is a shared multi-role step
   inside the TDM PR acceptable?

Questions, corrections to the gap table, and algorithms beyond this list are
all welcome in the comments.

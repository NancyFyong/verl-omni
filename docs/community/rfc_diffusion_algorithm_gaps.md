# [RFC] Call for contributions: DiffusionOPSD & AWM for diffusion post-training

Last updated: 10/16/2026.

We'd love your help bringing new post-training algorithms into verl-omni. The shortlist below is published work with runnable reference code, and each fits the loss registry and recipe structure already in the repo — so a contribution is a focused PR, not a research project.

Treat this as suggestions, not a closed menu — other algorithms are welcome too, and a short RFC is a good way to start. If one of these catches your eye, please comment on the RFC issue first so we can coordinate.

---

## 1. DiffusionOPSD — On-Policy Self-Distillation in Diffusion Models

**Paper:** [arXiv 2608.24646](https://arxiv.org/abs/2608.24646) · **Code:** [worldbench/DiffusionOPSD](https://github.com/worldbench/DiffusionOPSD) · **Site:** [diffusionopsd.github.io](https://diffusionopsd.github.io/)

Turns an image-level reward into explicit clean-prediction targets at sampled low-noise queries. A frozen behavior policy collects the query state and clean-output anchor, differentiable reward ascent and descent build bounded positive and negative targets around that anchor, the trainable policy fits those detached targets, and an EMA refreshes the behavior policy for the next round.

The paper reports the best held-out score in 19 of 20 reward-matched settings across SD3.5-M and Z-Image-Turbo, with 40%–63% lower GPU-hours than DiffusionNFT.

## 2. AWM — Advantage Weighted Matching

**Paper:** [arXiv 2509.25050](https://arxiv.org/abs/2509.25050) · **Code:** [X-GenGroup/Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)

Weights the standard flow-matching velocity loss directly by per-sample group advantages, with PPO-style clipping and KL regularization toward an EMA copy for stability. Because training resamples forward-process timesteps from the final clean latent rather than replaying reverse SDE log-probs, sampling can stay on fast deterministic ODE solvers — the same decoupled family as DiffusionNFT and DGPO.

---

## What a good contribution looks like

Nothing here is meant to be intimidating — roughly in this shape:

- **Algorithm + config** in the loss/advantage registry (`verl_omni/trainer/diffusion/diffusion_algos.py` and friends), following the existing FlowGRPO / DiffusionNFT / DGPO patterns.
- **A recipe** under `examples/` on a pipeline we already support (SD3.5, Qwen-Image, FLUX, Wan2.2, LTX-2.3).
- **A CPU test** (`test_*_on_cpu.py`) in the style of the existing algorithm tests.
- **GPU validation** — a small-scale run with evidence, such as a `reward_curve.png`; a curve from even a modest run goes a long way.
- **A fair comparison** against the closest thing we already ship (e.g. DiffusionOPSD vs DiffusionNFT, AWM vs DiffusionNFT / DGPO) would be wonderful to see, but it's by no means required.

## Already in flight — to avoid duplicated effort

- **DGPO** — merged in [#733](https://github.com/verl-project/verl-omni/pull/733)
- **TDM-R1** and **DMDR** — claimed on [#717](https://github.com/verl-project/verl-omni/issues/717)
- **DMD2** — tracked in [#520](https://github.com/verl-project/verl-omni/issues/520)
- **Flow-DPPO (V1 sync recipe)** — [#690](https://github.com/verl-project/verl-omni/pull/690)
- **AlphaGRPO (BAGEL)** — [#728](https://github.com/verl-project/verl-omni/issues/728) / [#729](https://github.com/verl-project/verl-omni/pull/729)
- **OmniNFT** — [#664](https://github.com/verl-project/verl-omni/pull/664) / [#750](https://github.com/verl-project/verl-omni/pull/750)
- **DiffusionART** — [#744](https://github.com/verl-project/verl-omni/pull/744)

Other unshipped algorithms from the community (e.g. **CRD** — [arXiv 2603.14128](https://arxiv.org/abs/2603.14128), **TDM** — [arXiv 2503.06674](https://arxiv.org/abs/2503.06674)) are also welcome.

Questions or suggestions for the list? Please don't hesitate to comment on the RFC issue — we're glad to help you get started.

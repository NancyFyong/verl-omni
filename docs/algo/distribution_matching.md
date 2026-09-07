# Distribution-Matching Distillation

Last updated: 09/07/2026.

verl-omni supports Qwen-Image training with original DMD and the distribution-only
and adversarial profiles of DMD2. It also supports standalone ODE-regression
initialization of a causal Wan 2.1 student. The implementation follows the
multi-role design in [RFC #519](https://github.com/verl-project/verl-omni/issues/519):
trainable and frozen semantic roles are coordinated by the dedicated
`distillation` trainer.

This path is not policy-gradient RL and does not use a vLLM-Omni rollout to
build the differentiable student graph. Sampling for training runs inside the
FSDP/FSDP2 worker with `algorithm.sample_source=offline`.

## Objective

For clean student output $x_g$, independent noise $\epsilon$, and noise level
$\sigma$, Qwen-Image uses the rectified-flow convention

$$
x_\sigma = (1-\sigma)x_g + \sigma\epsilon,
\qquad v_{target} = \epsilon - x_g,
\qquad \hat{x}_0 = x_\sigma - \sigma v.
$$

The fake and teacher velocity predictions are converted to canonical clean
predictions. The student receives the detached normalized score difference

$$
g = \frac{\hat{x}_{0,fake}-\hat{x}_{0,real}}
         {\max(\operatorname{mean}(|x_g-\hat{x}_{0,real}|),\epsilon_{norm})}
$$

through the surrogate

$$
L_{DMD} = \frac{1}{2}\operatorname{MSE}
          \left(x_g,\operatorname{stopgrad}(x_g-g)\right).
$$

The normalizer covers every non-batch latent dimension and is not masked. All
score arithmetic and loss reductions run in fp32. The fake-score update uses
MSE against the detached target $\epsilon-x_g$.

The real teacher uses standard CFG,
`uncond + scale * (cond - uncond)`, with an explicit negative condition. The
fake score and distilled student use one conditional forward by default.

## Qwen-Image rollout

The Qwen adapter preserves the checkpoint's native conventions:

- normalized 5-D VAE latents are packed into Qwen image tokens;
- the transformer receives timesteps in `[0, 1]`;
- its velocity is `noise - x0`;
- the few-step rollout uses deterministic Euler transitions;
- all ranks sharing FSDP collectives use one broadcast rollout exit step;
- only that exit step retains a student autograd graph;
- all preceding rollout steps execute under `no_grad`;
- student, teacher, and fake-score forwards remain in evaluation mode.

The default four-step schedule applies the reference linear shift `3.0`, giving
sigmas `[1.0, 0.9, 0.75, 0.5, 0.0]`. Score timesteps are sampled discretely
from the model's 1,000 training timesteps, shifted once, and clamped to
`[0.02, 0.98]`. With `score_discrete_steps=0`, sigma is instead sampled
uniformly inside the configured bounds without timestep shifting. Sampling
from `[0, 1)` and clamping is not equivalent to that continuous distribution.

Raw prompts use one text-only user message. Conditioning applies the fixed
Qwen pipeline template before the encoder removes its 34-token prefix; a
generic chat template can leave short prompts with no encoded tokens.

A registered vLLM-Omni adapter uses the same sigma construction for
non-autograd inference. It requires deterministic sampling (`noise_level=0`)
and defaults to no inference CFG. A non-default training shift must also be set
as `actor_rollout_ref.rollout.algo.rollout_timestep_shift`. Request batching
reuses the vLLM-Omni scheduler and request collation with an explicit capability
flag. Use `step_execution=false`; unvalidated stepwise DMD execution is rejected.

## DMD and DMD2 profiles

`recipe=dmd2`, `profile=distribution_only` is the recommended first runnable
configuration. One student update is followed by `fake_update_ratio`
fake-score updates. It needs prompt conditioning only.

`recipe=dmd` adds paired trajectory regression. Each sample must provide:

- `reference_noise`;
- exactly one of `teacher_target_latents` or normalized `[0, 1]`
  `teacher_target_pixels`;
- a non-empty `teacher_sampling_manifest`;
- prompt conditioning.

`regression_type=decoded_lpips` decodes normalized Qwen latents through the
frozen checkpoint VAE and applies PIQ LPIPS. It is the paper-oriented mode and
requires the `distillation` dependency extra. `regression_type=latent_mse` is a
non-paper diagnostic variant.

`recipe=dmd2`, `profile=paper` adds the DMD2 non-saturating adversarial objective.
The student loss adds `softplus(-D(x_fake))`; each fake phase independently steps
the fake-score denoising optimizer and the discriminator optimizer with
`softplus(D(x_fake.detach())) + softplus(-D(x_real.detach()))`. The discriminator
uses its own LoRA adapter and classifier head while reusing only the frozen Qwen
base. Student adversarial evaluation freezes discriminator parameters but retains
gradients to `x_fake`.

The `diffusion_gan` mode re-noises real and generated latents at a uniformly
sampled integer timestep below `adversarial.max_timestep`; the
`cls_on_clean_image` mode classifies clean latents at timestep zero. Each row must
provide exactly one of `real_latents` plus a matching VAE-normalization manifest,
or finite RGB `real_pixels` in `[0, 1]`. This profile remains distinct from
original DMD: it does not add paired trajectory/LPIPS regression.

## Wan 2.1 causal ODE initialization

`recipe=ode_regression` trains only a causal `student` and its `student_ema`.
It is an initialization stage for later CausVid-style distribution matching;
it does not instantiate a teacher-score or fake-score role and must not be
reported as DMD training.

The first supported checkpoint family is `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
Model dimensions are loaded from the checkpoint rather than hardcoded. The
architecture adapter preserves Diffusers parameter keys and installs
parameter-free causal behavior:

- full-sequence training uses block-prefix attention, where a motion block can
  see itself and committed earlier blocks but not future blocks;
- temporal RoPE positions retain their absolute frame offsets;
- incremental execution processes exactly one temporal block per call and caches
  self-attention and cross-attention K/V per layer;
- a block updates all layer caches atomically only after a successful forward;
- cached prior context is detached, and context parallelism is rejected until a
  cache-ownership contract exists.

Every dataset row carries a deterministic ODE trajectory with layout
`[steps, latent_frames, channels, latent_height, latent_width]`, the matching
raw timestep vector, and a final clean latent equal to the last trajectory
state. A canonical manifest fingerprints the frozen teacher, revision,
scheduler, guidance, negative prompt, VAE, tokenizer, latent geometry, dtype,
and seed policy. Training selects one trajectory state per temporal block,
converts Wan's `noise - x0` velocity to canonical `x0`, and masks zero-timestep
positions from the MSE reduction. The loss is the global mean over active latent
elements, not the mean of per-sample means. The computer supplies its active-element
count as `loss_normalizer`; the runtime accumulates numerator gradients and divides
by the DP-averaged count before gradient clipping and the optimizer step. Reported
losses use the same denominator, independently of micro-batch partitioning.

The default exported causal schedule starts from unshifted timesteps
`[1000, 750, 500, 250]` and applies the rational shift `8.0` exactly once.
FSDP1/FSDP2 training, EMA, checkpoint/resume, and semantic student/EMA LoRA
export use the shared multi-role runtime. Ulysses sequence parallelism and
vLLM-Omni causal serving are not supported in this stage.

See `examples/distillation_trainer/wan21/README.md` for the trajectory contract
and launch command.

## Role storage and checkpoints

The recommended Qwen LoRA layout stores `student`, `fake_score`, `student_ema`, and,
for the adversarial profile, `discriminator` as named adapters over one frozen
base. `teacher_score` disables adapters. Standalone Wan ODE initialization uses
only `student` and `student_ema` adapters over one causal base. Student, fake-score, and optional
discriminator optimizers and schedulers are independent, and EMA is updated only
after a successful student optimizer step. FSDP1 requires
`use_orig_params=true`; FSDP2 is the recommended backend.

Composite checkpoints save the physical model once together with every role's
optimizer and scheduler, EMA state, distribution-matching RNG streams, controller
counters, dataloader state, and driver RNG. Only the semantic `student` or
`student_ema` role can be exported to inference.

## Configuration

The minimal routing fields are:

```bash
algorithm.trainer_type=distillation
algorithm.sample_source=offline
actor_rollout_ref.model.algorithm=dmd2
distillation.enabled=false
distillation.distribution_matching.recipe=dmd2
distillation.distribution_matching.profile=distribution_only
```

Same-resolution physical batches and sample-weighted gradient accumulation use
the existing worker loop, with independent student/fake micro-batch sizes. Each
original-DMD sample must retain its own regression provenance. One physical batch
shares a synchronized rollout exit, so batch-size comparisons must account for
changes in sampled forward counts.

Profiling uses the existing `DistProfiler` and `Tracking` components. Metrics keep
every repeated phase, reset at cycle boundaries, sum elapsed host intervals and
counts, and preserve peak memory across ranks. The driver reports cycle wall time
and separate student/fake sample throughput; nested host intervals must not be
summed into a GPU-time estimate. Logging steps include warmup cycles, while
`training/global_step` retains student-update semantics. See the example README
for metric definitions and controlled profiling commands.

Runnable LoRA recipes are:

- `examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh`;
- `examples/distillation_trainer/wan21/run_wan21_ode_lora.sh`.

See each adjacent README for data fields, installation, and the complete launch
command.

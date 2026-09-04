# Distribution-Matching Distillation

Last updated: 09/05/2026.

VeRL-Omni supports Qwen-Image training with DMD and the distribution-only
profile of DMD2. The implementation follows the multi-role design in
[RFC #519](https://github.com/verl-project/verl-omni/issues/519): a trainable
student, a frozen real-score teacher, a trainable fake-score model, and a
student EMA are coordinated by the dedicated `distillation` trainer.

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
- only the sampled exit step retains a student autograd graph;
- all preceding rollout steps execute under `no_grad`;
- student, teacher, and fake-score forwards remain in evaluation mode.

The default four-step schedule applies the reference linear shift `3.0`, giving
sigmas `[1.0, 0.9, 0.75, 0.5, 0.0]`. Score timesteps are sampled discretely
from the model's 1,000 training timesteps, shifted once, and clamped to
`[0.02, 0.98]`.

A registered vLLM-Omni adapter uses the same sigma construction for
non-autograd inference. It requires deterministic sampling (`noise_level=0`)
and defaults to no inference CFG. A non-default training shift must also be set
as `actor_rollout_ref.rollout.algo.rollout_timestep_shift`.

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

The DMD2 adversarial classifier profile is not part of this integration and
fails closed. It belongs to the later adversarial-runtime stage.

## Role storage and checkpoints

The recommended LoRA layout stores `student`, `fake_score`, and `student_ema`
as named adapters over one frozen Qwen base. `teacher_score` disables adapters.
Student and fake-score optimizers and schedulers are independent, and EMA is
updated only after a successful student optimizer step. FSDP1 requires
`use_orig_params=true`; FSDP2 is the recommended backend.

Composite checkpoints save the physical model once together with every role's
optimizer and scheduler, EMA state, phase-runner RNG streams, control-plane
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

The runnable LoRA recipe is
`examples/distillation_trainer/qwen_image/run_qwen_image_dmd2_lora.sh`.
See its adjacent README for data fields, installation, and the complete launch
command.

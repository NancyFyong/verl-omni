---
name: review-config-recipe
description: "Review checks for a verl-omni PR that adds or changes config fields, defaults, checkpoint-layout assumptions or example recipes: where a field must appear, key naming and reuse, path resolution, and whether the committed recipe is the one that was run. Use from pr-review or self-review."
---

# Review: config fields and recipes

The [config rule](../../rules/config.md) owns dataclass shape, validation, generated
YAML and deprecation; the [shell recipe rules](../../rules/code-style.md#example-shell-recipes)
own launcher structure. This skill covers what reviewers still catch after those.

## When it applies

```bash
git diff --name-only <base>...<head> | grep -E 'workers/config/|trainer/config/|examples/.*\.(sh|yaml)$|docs/examples/config\.md'
```

## Config fields

- **Every surface carries the field.** A new field lives in its dataclass, the
  hand-written group YAML, the regenerated `_generated_*.yaml` files, and
  `docs/examples/config.md` (#559 review asked for the last one). Check with
  `git grep -n "<field>" <head> -- verl_omni/trainer/config docs/examples/config.md`.
- **No near-duplicate names.** #559 first added `actor_rollout_ref.model.use_torch_compile`
  while the actor and FSDP engine configs already had a `use_torch_compile` with a
  different meaning; the reviewer called it confusing and it became
  `use_regional_compile`. Search the composed config for the new key's words.
- **Reuse an existing knob before adding one.** #341 review settled on reusing
  `sde_window_range` and `sde_window_size` plus one boolean switch instead of new
  window parameters.
- **Expose only what is tunable and validated.** #559 review: "if the configs are
  not tunable or not validated, I think we can keep them hard coded in scripts."
  The author may justify the option, as #559 did; ask either way.
- **A configured value is actually read.** #527's body described a configured
  synchronization dtype while the code hard-coded bf16; the reviewer asked to wire
  it or fix the wording. Trace each new field to its consumer.
- **Defaults and renames.** A changed default is a behavior change the body lists
  ([review-refactor](../review-refactor/SKILL.md)); a rename follows the
  [deprecation path](../../rules/config.md#backward-compatibility).

## Checkpoint and path resolution

`DiffusionModelConfig.__post_init__` resolves these (see
`verl_omni/workers/config/diffusion/model.py`):

- `architecture` comes from `model_index.json`'s `_class_name` unless set. A
  converted or non-diffusers checkpoint must carry the exact name the adapters
  registered ([review-new-architecture](../review-new-architecture/SKILL.md)).
- `tokenizer_path` defaults to `<model>/tokenizer/` and silently falls back to the
  model root. A model with several tokenizers declares them in `extra_tokenizers`,
  as FLUX does for CLIP and T5 (#532).
- A model diffusers cannot load overrides `build_module` (BAGEL, #66). Ask for a
  test that loads through that override.
- A recipe documents what each path points to and which layout it expects.

## Recipes

- **The committed recipe is the one that produced the evidence.** #332 review asked
  why the script's hyperparameters differed from the run shown in the body, and
  asked for a validation curve from them. Diff the script against the logged
  config of the reported run.
- **An unverified recipe says so, or waits.** #574 review preferred dropping a
  script whose combined reward did not rise, or a header comment stating what is
  not verified, as `examples/flowgrpo_trainer/bagel/run_bagel_ocr_lora_npu.sh` has.
- **Shared scripts stay general.** An opt-in flag is not named in a script that
  serves every setting (#614).
- A new recipe for an existing model also gets its row in `docs/start/models.md`.

## Evidence

- `scripts/generate_trainer_config.sh` output clean on the head with the pinned
  verl installed (see the config rule's traps).
- For a new field: a CPU test that the invalid values raise and the valid ones
  reach the consumer.
- For a recipe: `bash -n`, a stub-trainer argv capture, and the curve from exactly
  that recipe ([review-tests-ci](../review-tests-ci/SKILL.md)).

<!--
MAINTAINER GUIDE — The surface list and path-resolution facts are read from the
config directories, docs/examples/config.md and model.py. Keep dataclass and YAML
rules in rules/config.md, not here.
-->

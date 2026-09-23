---
name: review-media-contract
description: "Review checks for a verl-omni PR that produces or consumes generated media: declared DiffusionIOSpec and media_kind, uint8 pixels, video layouts, audio sample rates, reward-side media projection and export failure handling. Use from pr-review or self-review."
---

# Review: generated media contract

Adapters declare what they emit with a `DiffusionIOSpec` (see
`verl_omni/pipelines/rollout_media.py`). Loggers, exporters and reward managers must
read that declaration instead of guessing from tensor shapes. Pixel responses are
always uint8 (#373). The contributor side is in the
[pipelines rule](../../rules/pipelines.md) and the integration guides.

## When it applies

```bash
git diff <base>...<head> | grep -nE 'DiffusionIOSpec|MediaSpec|media_kind|resolve_is_video|audio_sample_rate|_to_uint8|ndim ?== ?5|shape\[1\] ?== ?3'
git diff --name-only <base>...<head> | grep -E 'rollout_media|reward_manager/media|utils/tracking|reward_score/'
```

## Checks

- **Declared kind first.** New code must call `resolve_is_video(ndim, media_kind)` or
  read the spec. A fresh `ndim == 5` or `shape[1] == 3` test is a new heuristic; the
  rank fallback exists only for legacy outputs with no declared kind.
- **Contract violations reject; observability failures degrade.** A declared kind
  that disagrees with the tensor rank, two kinds in one batch, or a wrong stream
  count must raise before scoring. A failed file export may warn and skip. #481
  review caught a declared `image` on a rank-5 tensor fail-opening per sample
  instead of rejecting.
- **Check the uint8 contract where the data is produced**, synchronously. Moving it
  into a background exporter turns a dtype bug into delayed warnings on every dump
  (#481).
- **Video layout.** Export and tracking normalize TCHW, CTHW and THWC before
  writing; #437 rewrote the export path and dropped CTHW normalization, fixed in
  #442. Channels-last output from a new model breaks shared helpers that assume
  channels-first, so check the layout conversion at the adapter boundary.
- **Audio sample rate comes from the spec or runtime metadata.** No model-specific
  default in shared code; #478 removed a MiniMax-only 32 kHz fallback that
  mislabeled other models' audio. A joint audio-video model declares an auxiliary
  audio stream with its rate, and the io-spec test asserts it.
- **Reward sees only generated media.** Reward managers project `audio`,
  `audio_sample_rate` and `media_kind` from the rollout batch and drop the dataset's
  copies, so conditioning audio is never scored as output (#481). A new reward
  path must go through the shared projection in `reward_loop/reward_manager/`.
- **Background exporters own copies.** Tensors handed to an export thread are
  cloned to CPU first ([review-async-state](../review-async-state/SKILL.md)).

## Evidence

- A CPU test per declared kind that feeds a mismatched rank and expects a raise.
- For a new video or audio model, a dumped sample actually opened: frames viewed,
  audio played at the declared rate. Non-degenerate pixel statistics do not show
  that content, orientation or colour order is right. The MiniMax H3 bring-up
  produced patch-grid videos with healthy statistics
  ([review-weight-sync](../review-weight-sync/SKILL.md) has the cause).

<!--
MAINTAINER GUIDE — The legacy rank fallback in rollout_media.py should shrink over
time; when it is removed, drop the "legacy outputs" caveat above.
-->

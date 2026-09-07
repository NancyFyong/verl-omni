# Named diffusion media artifacts

Last updated: 09/07/2026.

This is the first working slice of RFC #403 PR6, not the completed all-adapter
migration. MiniMax H3 **DiffusionNFT** emits named artifacts. Other adapters
(including H3 FlowGRPO) still use the primary/auxiliary contract and its legacy
consumer paths. GPU verification is required before completing that migration.

## Runtime contract

`verl_omni/pipelines/rollout_artifacts.py` defines:

- `ArtifactSpec(modality, representation, layout, sample_rate=None, fps=None)`:
  a per-request declaration. Representation is `latent` or `decoded`; latent is
  not a modality. There is no configurable dtype: the tensor owns its dtype.
- `MediaArtifact(spec, data)`: **one sample**, without an implicit batch axis.
- `validate_artifacts(items, specs, primary=..., context=..., requested=...)`:
  checks returned versus declared names, duplicate/missing/unexpected/requested
  artifacts, metadata, rank, channel axes and dtype, then normalizes decoded axes.

Decoded image/video must already be uint8 `[0,255]`; decoded audio is a floating
waveform with a positive sample rate. Decoded layouts normalize to `CHW`, `TCHW`
and `CT`. Image `HWC`, video `CTHW`/`THWC`, and mono audio `T` are accepted **only
when explicitly declared**. Model-native latent axes (including packed `LC` or
H3 audio `CLT`) remain unchanged, as do their floating dtype and storage.

An adapter constructs expected specs separately from its returned named items:

```python
specs = {
    "video_preview": ArtifactSpec("video", "decoded", "THWC", fps=24),
    "video_latent": ArtifactSpec("video", "latent", "CTHW"),
}
items = [
    ("video_preview", MediaArtifact(specs["video_preview"], decoded_video)),
    ("video_latent", MediaArtifact(specs["video_latent"], native_video_latent)),
]
output = with_media_artifacts(
    rollout_output_with_training_data,
    artifacts=items,
    specs=specs,
    primary="video_latent",
    preview="video_preview",
    context=f"pipeline=MyPipeline/my_algorithm, request_id={request_id}",
)
```

`with_media_artifacts` lives in `pipelines/diffusion_rollout_output.py`. Its
`primary` selects the temporary legacy `responses` projection; `preview` is the
explicit decoded dump/W&B selection; optional `audio` selects the temporary
legacy audio projection. All artifacts remain available regardless of primary
selection. No consumer may replace a missing named preview with latent data.

The existing sampling `extra_args` can carry `requested_outputs` for validation.
This initial slice checks requested names, but **does not skip decoding or
transferring unrequested outputs**: the pinned H3 forward always decodes. It does
not claim a selective-decoding optimization.

## Engine and training transport

The pinned vllm-omni formatter retains only selected payload keys in its public
multimodal output. The adapter therefore puts the complete named tensor mapping
under one modality-keyed `payload` entry, with `media_artifacts` declarations and
selectors in `metadata`. The engine's native formatter is exercised in CPU tests;
no installed upstream package is patched.

`DiffusionStrategy` validates and restores the mapping, exposes
`DiffusionOutput.artifacts`, `primary_artifact`, `preview_artifact`, and projects
only the explicitly selected primary to `diffusion_output`. Algorithm-owned
`rl`, prompt embeddings and trajectory fields remain separate and unchanged.

At the agent-loop boundary, tensors become ordinary fields named
`media_artifact__<name>`. Tensor-free spec dictionaries and selectors travel in
non-tensor metadata. Both ordinary batching and TransferQueue preserve them;
scorers reconstruct per-sample artifacts from those fields. This avoids hiding
large tensor payloads inside non-tensor object arrays.

Migrated consumers:

- V0/V1 rollout dumps and validation/W&B select the declared preview, even when
  `responses` contains floating latents. Named previews bypass rank/channel-count
  layout inference, carry their FPS to video export, and honor max-sample limits.
- CLAP selects the decoded `audio` artifact and its rate.
- ImageBind selects `audio` and `video_preview`, rather than interpreting the
  primary response as a video. The legacy branch remains for unmigrated adapters.
- The SD3 latent HTTP client accepts a declared `image_latent` in native `CHW`
  layout. It rejects other model layouts rather than coercing packed latents into
  the SD3 protocol. No SD3 producer migration is claimed in this slice.

Schema/representation/layout/selector errors fail synchronously. Filesystem,
FFmpeg and W&B failures retain the best-effort reporting behavior from PR4.

## MiniMax H3 NFT evidence and limits

The adapter declares four artifacts: `video_preview`, `audio`, `video_latent`,
and `audio_latent`. Its declarations follow the **pinned source**, not tensor
channel-size heuristics:

| Artifact | Pinned source at `ded893462` | Per-sample input layout |
| --- | --- | --- |
| video preview | `_prepare_minimax_h3_video_output` in `pipeline_minimax_h3.py` emits uint8 NTHWC | THWC |
| decoded audio | `MiniMaxH3AudioVAE.decode_latent` in `vae.py` returns NCT | CT |
| video latent | `minimax_h3_patchify_video_latent` in `packed_tokens.py` requires NCTHW | CTHW, unchanged |
| audio latent | `minimax_h3_pack_audio_latent` in `packed_tokens.py` requires CLT | CLT, unchanged |

Existing packed `latents_clean` and reference/replay tensors are not renamed or
repacked. CPU tests cover the real H3 postprocessor/formatter, Ref2VA replay
metadata and both transport paths. **These are not full-weight GPU layout or
quality verification.**

## Before PR6 is ready

- [ ] Verify H3 NFT T2VA/FL2VA/Ref2VA on GPU with decoded export and latent-primary scoring.
- [ ] Migrate H3 FlowGRPO, LTX, Wan, Qwen variants, SD3, Boogu and Bagel using their actual postprocessor outputs.
- [ ] GPU-verify every decoded layout, including request batching and step execution.
- [ ] Replace the legacy `DiffusionIOSpec.primary/auxiliary` declarations once every producer uses named artifacts.
- [ ] Remove remaining tuple/shape/representation fallback branches and legacy projections only after their consumers migrate.
- [ ] Add selective preview decoding/transfer if a model can actually honor requested outputs.
- [ ] Stress-test video dump queue memory/backpressure; CPU unit tests do not establish that property.

The live implementation and tests, not this checklist, are the source of truth
for what is complete. See `test_rollout_artifacts_on_cpu.py`,
`test_media_artifact_transport_on_cpu.py` and `test_artifact_preview_dump_on_cpu.py`.

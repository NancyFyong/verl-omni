# MiniCPM-o 4.5 thinker OPD

Last updated: 09/09/2026.

This recipe implements the first, thinker-stage part of [RFC #565](https://github.com/verl-project/verl-omni/issues/565), under [#345](https://github.com/verl-project/verl-omni/issues/345). The student generates a complete text response; a frozen teacher scores the same tokens and the existing verl reverse-KL policy-gradient loss updates the student.

## Supported scope

- MiniCPM-o **4.5**, text output from the thinker stage.
- Text, images, and at most one audio clip per prompt; images and audio can be combined.
- FSDP2, padded per-sample actor sequences, LLM LoRA, and merged-weight rollout synchronization.
- Selected-token reverse-KL OPD with a separate teacher resource pool.

Video, multiple audio clips per prompt, talker/codec-policy training, and streaming duplex are not supported by the **simplex** recipe. The separate experimental duplex recipe is described below. Unsupported inputs/stages raise errors rather than falling back to text-only training. There is no generated speech in the thinker-only topology. Speech-policy OPD is a separate RFC milestone; streaming duplex is tracked by [#566](https://github.com/verl-project/verl-omni/issues/566).

## Weights and dependencies

Use the repository's [installation instructions](../../../docs/start/install.md) and pinned vLLM-Omni revision. The actor uses the released MiniCPM remote model code with a targeted Transformers 5 initialization/Whisper compatibility adapter. Review the remote code before enabling `trust_remote_code`.

```bash
hf download openbmb/MiniCPM-o-4_5 \
  --revision 503e754207c94da6bb26850b4469f367c9ea3582 \
  --local-dir "$HOME/models/MiniCPM-o-4_5"
```

Provide a compatible frozen teacher checkpoint with the same tokenizer and special-token mapping. Vocabulary compatibility is checked before worker allocation. An identical teacher/student checkpoint is useful for a consistency smoke test, **not evidence of useful distillation or quality improvement**. A learning experiment needs a justified teacher advantage.

Do not downgrade the repository's Transformers version to match the model card's older reference environment. Use Python 3.12 with FlashInfer 0.6.16.post3: its communication module evaluates `array.array[int]`, which fails during worker initialization on Python 3.11.

## Data

Create `train.jsonl` and `test.jsonl` in an input directory. Text-only rows need only a `prompt` string:

```json
{"prompt":"Explain why the sky appears blue in two sentences."}
```

Optional image/audio paths are relative to the JSONL file:

```json
{"prompt":"Answer the spoken question about this scene.","images":["scene.png"],"audios":["question.wav"]}
```

A prompt may also be a message list with string content. In that case, include one `<image>` or `<audio>` placeholder for each media item, in the intended order. Audio files are decoded to mono and resampled to 16 kHz. Images are embedded in Parquet; audio paths are resolved to absolute paths, so the referenced audio must remain accessible on each worker node.

```bash
python examples/opd_trainer/minicpm_o/prepare_data.py \
  --input-dir "$HOME/data/minicpm_source" \
  --output-dir "$HOME/data/minicpm_simplex"
```

The converter writes `train.parquet` and `test.parquet`; it does not generate teacher answers or preference pairs. Keep held-out examples separate, and check source-media rights and speaker consent.

### MMK12 image-math data

The Qwen3-Omni OPD example uses [MMK12](https://huggingface.co/datasets/FanqingM/MMK12), an image-to-text K12 math dataset. MiniCPM reuses the same canonical converter and reward contract; it does not need a model-specific copy of either one. After downloading raw `train-*.parquet` and `test-*.parquet` shards, run:

```bash
bash examples/opd_trainer/minicpm_o/prepare_mmk12_data.sh \
  /path/to/raw/mmk12 \
  "$HOME/data/mmk12"
```

The output embeds one image per row and includes the ground-truth answer, parsed choice options, and an explicit `<think>...<answer>\boxed{...}</answer>` response instruction. Install the rule-based scorer dependency with `uv pip install math-verify`; no learned reward-model checkpoint is needed.

## Training

```bash
STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5" \
TEACHER_MODEL="$HOME/models/minicpm-o45-teacher" \
DATA_DIR="$HOME/data/minicpm_simplex" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora.sh
```

Defaults use four student GPUs and four teacher GPUs, TP=2 for each rollout/teacher replica, prompt/response budgets 1024/512, rank-32 LoRA, temperature 1, and two samples per prompt. Override `STUDENT_GPUS`, `TEACHER_GPUS`, `ROLLOUT_TP`, `TEACHER_TP`, `PROMPT_LENGTH`, or `RESPONSE_LENGTH` through the environment. Extra CLI arguments override the script's defaults.

For MMK12, use the dedicated wrapper:

```bash
STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5" \
TEACHER_MODEL="$HOME/models/minicpm-o45-teacher" \
DATA_DIR="$HOME/data/mmk12" \
bash examples/opd_trainer/minicpm_o/run_simplex_opd_lora_mmk12.sh
```

This wrapper changes the base objective from teacher-only OPD to **task reward plus OPD** by enabling `distillation.distillation_loss.use_task_rewards`. It selects `mmk12_reward.py`, whose normalized score combines answer correctness from `math_verify` with progressive `<answer>` and `\boxed{}` format credit. `REWARD_FUNCTION_PATH` can override the scorer path. Validation runs every ten steps by default, without validation before training.

Important settings:

| Setting | Requirement |
| --- | --- |
| `model.model_stage` | `thinker`; native serving stage 0 is named `llm` |
| `model.use_remove_padding` | `false`, preserving per-sample image/audio bounds |
| `model.lora.merge` | `true`; no native MiniCPM LoRA-manager patch is installed |
| `rollout.agent.default_agent_loop` | `minicpm_simplex_agent` |
| `rollout.agent.agent_loop_manager_class` | `verl_omni.pipelines.minicpm.agent_loop.MiniCPMAgentLoopManager` |
| `engine_kwargs.vllm_omni.async_chunk` | `false`; collect the complete bounded response |
| `distillation.distillation_loss` | `loss_mode=kl`, `use_policy_gradient=true`; the base recipe uses no task reward, while the MMK12 wrapper enables it |
| Teacher context | Full student prompt + response + one scoring token; configured by verl |

The recipe freezes vision/audio encoders and their projection modules and restricts LoRA to the LLM. `actor.freeze_vision_tower=false` avoids a Qwen-specific engine freeze path; the MiniCPM adapter owns encoder freezing.

In the base recipe, `use_task_rewards=false` skips training-time task reward while retaining teacher scoring. The MMK12 wrapper enables both signals and configures its rule-based scorer.

Both `data.train_batch_size × rollout.n` and `actor.ppo_mini_batch_size × rollout.n` must be divisible by the actor data-parallel size. Use microbatch 1 for a small smoke test; for four actor ranks and `rollout.n=2`, a PPO minibatch of 2 gives one sample per rank. The base recipe disables validation-before-training and periodic task-reward validation; the MMK12 wrapper enables periodic validation every ten steps.

## Replay and verification

The agent snapshots native processor outputs before rollout. Serving receives the source prompt token IDs; actor replay receives the matching media-expanded IDs. Teacher requests append the student's response IDs without decode/re-tokenize. The adapter checks the actual rollout prefix and the teacher's next-token IDs, including verl's final dummy scoring row.

Frozen encoder embeddings remain buffers under their original checkpoint names, avoiding conditional FSDP collectives and direct reads of sharded Whisper positional weights. Vision insertion is out of place so PEFT input-gradient hooks remain valid. Teacher fields are resized in synthetic zero-loss padding samples.

CPU contracts are in:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q --asyncio-mode=auto \
  tests/pipelines/test_minicpm_simplex_on_cpu.py \
  tests/trainer/omni/test_omni_distillation_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_omni_teacher_on_cpu.py
```

For real-weight validation, check complete rollout → teacher scoring → actor update → weight sync → fresh rollout over multiple steps, and repeat with image/audio prompts. Token equality alone does not establish probability parity or generation quality. Monitor distillation loss, gradients, teacher coverage, and post-sync behavior; missing/misaligned teacher or replay fields must fail the sample.

The shared omni teacher plumbing follows [#375](https://github.com/verl-project/verl-omni/pull/375), and the MiniCPM loading/Whisper compatibility overlaps the model foundation in [#550](https://github.com/verl-project/verl-omni/pull/550); neither requires a separate trainer or a forked distillation objective.

## Experimental bounded native duplex

`run_duplex_opd_lora.sh` implements the bounded Thinker trace/replay path of
[RFC #566](https://github.com/verl-project/verl-omni/issues/566). It is **not yet a
validated full-duplex training release**. CPU tests cover contracts, probabilities,
transport and gradients; real-weight multi-step actor/rollout/teacher agreement,
speech quality, and interruption/recovery remain acceptance gates. Full speech
rendering also requires the checkpoint's complete `assets/token2wav/` directory
and the native Code2Wav dependencies (including `onnx`, `onnxruntime` and
`s3tokenizer`); a Thinker-only checkpoint installation is insufficient.

The student uses the native vLLM-Omni session API with concurrent input arrival
and output collection. Its frozen Talker and Code2Wav remain active. The teacher
pool uses a Thinker-only pipeline and independently re-encodes each bounded causal
prefix. It never receives the student's encoder features or KV caches. Every
sampled action, including a listen-only decision, becomes one training window;
forced tokens have zero loss. After accepted native Thinker unit boundaries, the
bounded session is explicitly closed before weight/cache changes. Duplex requests
remain resumable (`finished=false`) between units; ordinary request completion
flags are not used as unit boundaries. Captured worker actions must match the
engine's accepted cumulative token sequence exactly. Async lookahead after a
unit terminator is recorded as discarded, cannot change native turn state, and
never contributes to the loss. Teacher results require a worker-side receipt
matching the prefix fingerprint; ordinary LM-sampler fallback is not accepted.

### Worktree-local runtime patch

The recipe opts into `VERL_OMNI_MINICPM_DUPLEX_OPD=1`. The existing
`vLLMOmniColocateWorkerExtension.__new__` installs
`verl_omni/pipelines/minicpm/duplex_patch.py` in each worker, following the repository's
`VLLMOmniHijack` pattern. No installed `site-packages` files are rewritten. The
patch is idempotent and checks the expected native APIs against the repository
pin (`ded8934626aaad1a3e816c3a1d9d742efc012d93`). Ordinary requests and speech-stage
sampling retain upstream behavior. The full deployment inherits the pinned native
MiniCPM YAML so its codec connectors and streaming TTS configuration are retained.

The native Thinker decoder has a two-pass distribution: first preserve its
`chunk_eos` boundary probability, then sample from the remaining constrained
vocabulary. The training patch computes the marginal **emitted-action** probability,
including native listen-to-speak aliasing. Actor and teacher use the same function.
Training uses temperature 1, no top-k/top-p truncation, no repetition penalty and
no character-count cutoff; these optional settings differ from the serving demo's
defaults. Token-budget-forced endings remain forced, not learned decisions. Do not
substitute raw vocabulary softmax probabilities for these behavior probabilities.

### Session manifests

Supply legally usable recordings with appropriate speaker consent. No original
MiniCPM duplex training corpus or benchmark is downloaded. A source row represents
input arrival, not a final assistant transcript. Paths are relative to the JSONL:

```json
{"session_id":"conversation-001","instructions":"Listen and respond naturally.","ref_audio":"speaker.wav","input_tracks":{"microphone":{"uri":"user.wav","sample_rate":16000}},"events":[{"seq":0,"type":"video_frame","uri":"frame.jpg","pts_ms":800,"available_at_ms":900},{"seq":1,"type":"audio","track":"microphone","start_ms":0,"end_ms":1000,"available_at_ms":1000,"is_speech":true},{"seq":2,"type":"audio","track":"microphone","start_ms":1000,"end_ms":2000,"available_at_ms":2000,"is_speech":true}],"max_duration_ms":8000,"max_context_tokens":2048,"max_actions":128,"max_tokens_per_unit":20}
```

The initial loader supports one contiguous mono 16 kHz input-audio channel and
optional timestamped image frames. Events must be ordered by availability, and
media cannot become model-visible before its declared time. Each audio event must
explicitly provide its native `is_speech` boolean (for example, from causal VAD);
optional `force_listen` is also a boolean, never inferred from the final transcript.
Native buffering and
first-chunk handling are reused, not replaced by independent one-second prompts.
The explicit speaker reference is for frozen speech rendering. Media must be
accessible on rollout nodes; allow its directory through vLLM-Omni's
`allowed_local_media_path` setting. The default bounded session is 8 seconds;
windows over 30 seconds, over-budget traces and unreplayable inputs are rejected.

```bash
python examples/opd_trainer/minicpm_o/prepare_duplex_data.py \
  --train "$HOME/data/duplex/train.jsonl" \
  --test "$HOME/data/duplex/test.jsonl" \
  --output-dir "$HOME/data/duplex/parquet"

STUDENT_MODEL="$HOME/models/MiniCPM-o-4_5" \
TEACHER_MODEL="$HOME/models/minicpm-o45-teacher" \
DATA_DIR="$HOME/data/duplex/parquet" \
VERL_OMNI_DUPLEX_ARTIFACT_DIR="$HOME/outputs/duplex" \
bash examples/opd_trainer/minicpm_o/run_duplex_opd_lora.sh \
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.allowed_local_media_path="$HOME/data/duplex"
```

The wrapper starts with three update cycles, rank-32 attention-only merged LoRA,
one PPO epoch and teacher-only OPD. It deliberately does not use MMK12's turn-based
answer reward. Profiling and real-model agreement are required before increasing
batching or session length. Full-prefix teacher replay is intentionally expensive;
incremental teacher caching is not enabled before parity is proven.

Each session saves a tensor-only `.pt` artifact containing the input manifest,
timed native text/audio outputs, native spans and action windows. Load trusted
artifacts with `torch.load(path, weights_only=True)`. This is a trace artifact,
not an automatically scored duplex benchmark. Context, loss and playback masks
are separate; playback masks remain false without an actual playback acknowledgement.
The bounded policy horizon can cut off a frozen speech tail; artifacts explicitly
record `speech_tail_may_be_truncated=true`. Complete speech drain/playback evaluation
is still required before claiming end-to-end conversational quality.

**Not implemented:** native cancellation's accepted-action/retained-prefix bridge,
partial rollback, automatic barge-in policy, long-lived cache resume, Talker/codec
training, and temporal task rewards. Cancellation events currently fail closed;
the CPU rollback/fence tests validate the contract, not live interruption support.
Do not interpret this boundary as a complete implementation of all RFC #566 milestones.

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 python -m pytest -q \
  tests/pipelines/test_minicpm_duplex_on_cpu.py \
  tests/utils/test_duplex_manifest_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_minicpm_duplex_runtime_on_cpu.py
```

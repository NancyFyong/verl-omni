# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded native session bridge; input arrival never waits for teacher scoring."""

import asyncio
import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import torch

from verl_omni.utils.dataset.duplex.manifest import validate_session_manifest

from .duplex_sampling import validate_duplex_sampling


def _check_control(result):
    if result.get("ok") is not True or result.get("error"):
        raise RuntimeError(f"Native duplex control failed: {result}")
    for stage in result.get("stage_results", []):
        value = stage.get("result", {})
        if value.get("supported") is False or value.get("error"):
            raise RuntimeError(f"Native duplex operation is unsupported or failed: {stage}")
    return result


def _audio_payload(manifest, event):
    import soundfile as sf

    track = manifest["input_tracks"][event["track"]]
    start, end = int(event["start_ms"] * 16), int(event["end_ms"] * 16)
    with sf.SoundFile(track["uri"]) as source:
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError("Duplex input must be mono 16 kHz; resample explicitly in data preparation.")
        source.seek(start)
        samples = source.read(end - start, dtype="float32")
    if len(samples) != end - start or not np.isfinite(samples).all():
        raise ValueError("Duplex audio source does not cover its declared finite sample range.")
    return {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": base64.b64encode(samples.astype("<f4").tobytes()).decode(),
        "is_speech": event["is_speech"],
        "force_listen": event.get("force_listen", False),
    }


def _video_frame(uri):
    from PIL import Image

    with Image.open(uri) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def _snapshot_output(output, timestamp_ns):
    inner = getattr(output, "request_output", None) or output
    completions = getattr(inner, "outputs", ()) or ()
    completion = completions[0] if completions else None
    mm = getattr(output, "multimodal_output", None) or getattr(completion, "multimodal_output", {}) or {}
    media = {}
    for key in ("audio", "sample_rate", "sr", "meta.duplex_epoch", "meta.duplex_turn_id"):
        value = mm.get(key)
        if isinstance(value, torch.Tensor):
            media[key] = value.detach().cpu()
        elif isinstance(value, str | int | float):
            media[key] = value
        elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
            media[key] = [item.detach().cpu() for item in value]
    return {
        "timestamp_ns": timestamp_ns,
        "text": getattr(completion, "text", ""),
        "token_ids": list(getattr(completion, "token_ids", ()) or ()),
        "media": media,
        "finished": bool(getattr(output, "finished", False)),
        "final_output_type": getattr(output, "final_output_type", None),
    }


def _rank_windows(results):
    ranks = []

    def visit(value):
        if isinstance(value, list) and (not value or isinstance(value[0], dict) and "action" in value[0]):
            ranks.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        else:
            raise RuntimeError(f"Duplex trace RPC returned an invalid/unsupported result: {value}")

    visit(results)
    if not ranks or not ranks[0]:
        raise RuntimeError("Native duplex produced no replayable actions.")
    expected = [(w["action"]["prefix_fingerprint"], w["action"]["token_id"], w["action"]["origin"]) for w in ranks[0]]
    for windows in ranks[1:]:
        actual = [(w["action"]["prefix_fingerprint"], w["action"]["token_id"], w["action"]["origin"]) for w in windows]
        if actual != expected:
            raise RuntimeError("TP ranks disagree on the native duplex trajectory.")
    return ranks[0]


def accepted_windows(windows, accepted_ids, stop_ids):
    """Remove only async lookahead after a unit terminator; verify engine acceptance."""
    ended, kept, discarded = set(), [], []
    for window in windows:
        action = window["action"]
        if action["seq"] in ended:
            action.update(context_mask=False, loss_mask=False, discarded=True)
            discarded.append(window)
            continue
        kept.append(window)
        if action["token_id"] in stop_ids:
            ended.add(action["seq"])
    if [window["action"]["token_id"] for window in kept] != accepted_ids:
        raise ValueError("Worker-sampled actions differ from the engine's accepted duplex output.")
    return kept, discarded


async def score_replay(server, prompt, params, request_id):
    """Require a worker receipt instead of silently accepting ordinary LM scores."""
    from verl.workers.rollout.replica import TokenOutput

    info = prompt["model_intermediate_buffer"]["duplex"]
    replay = info["opd_replay"]
    # vLLM can rename public request IDs before they reach model workers.
    info["opd_score_id"] = request_id
    final = None
    async for output in server.engine.generate(prompt=prompt, sampling_params_list=params, request_id=request_id):
        final = output
    inner = getattr(final, "request_output", None) or final
    target = replay["action"]["token_id"]
    if inner is None or not inner.outputs or list(inner.outputs[0].token_ids) != [target]:
        raise ValueError("Duplex teacher did not emit the requested scoring token.")
    pending = list(await server.engine.collective_rpc("take_minicpm_duplex_score", args=(request_id,), stage_ids=[0]))
    scores = []
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict) and "prefix_fingerprint" in value:
            scores.append(value)
        else:
            raise RuntimeError("Invalid or unsupported native teacher receipt RPC.")
    if not scores or any(
        score["prefix_fingerprint"] != replay["action"]["prefix_fingerprint"]
        or score["identity"] != replay["identity"]
        or score["token_id"] != target
        for score in scores
    ):
        raise ValueError("Duplex teacher receipt has a stale/mismatched prefix or action.")
    logprobs = torch.tensor([score["logprob"] for score in scores])
    if not torch.isfinite(logprobs).all() or not torch.allclose(
        logprobs, logprobs[:1].expand_as(logprobs), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Teacher TP ranks returned inconsistent or non-finite native scores.")
    return TokenOutput(
        token_ids=[target],
        log_probs=[scores[0]["logprob"]],
        stop_reason="completed",
        extra_fields={"duplex_prefix_fingerprint": replay["action"]["prefix_fingerprint"]},
    )


async def collect_session(server, manifest, params, request_id):
    """Drive concurrent native input/output, close the session, then drain its trace."""
    from verl.workers.rollout.replica import TokenOutput
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
    from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45NativeDuplexServingAdapter
    from vllm_omni.experimental.fullduplex.minicpmo45.input import MiniCPMO45PcmAppendBuffer
    from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities, DuplexSessionConfig
    from vllm_omni.experimental.fullduplex.request_client import DuplexRequestClient

    manifest = validate_session_manifest(manifest)
    validate_duplex_sampling(
        {
            name: getattr(params, name)
            for name in (
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "min_tokens",
            )
        }
    )
    if os.environ.get("VERL_OMNI_MINICPM_DUPLEX_OPD") != "1":
        raise RuntimeError("Enable the worktree's VERL_OMNI_MINICPM_DUPLEX_OPD=1 worker patch.")
    if server.lora_as_adapter:
        raise ValueError("Native duplex sessions currently require merged LoRA weight synchronization.")
    if manifest.get("max_context_tokens", 2048) > server.config.max_model_len:
        raise ValueError("Duplex context budget exceeds rollout.max_model_len.")
    config = DuplexSessionConfig(
        instructions=manifest["instructions"],
        modalities=["text", "audio"],
        ref_audio=manifest["ref_audio"],
        temperature=1.0,
        max_tokens=manifest.get("max_tokens_per_unit", 20),
        extra_body={"minicpmo45_native_duplex": True},
    )
    runtime = await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
        config, model_config=server.engine.model_config
    )
    runtime["duplex_stage_sampling_params"]["0"].update(
        temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=0
    )
    runtime["verl_opd"] = {
        "policy_version": server.global_steps,
        "max_context_tokens": manifest.get("max_context_tokens", 2048),
        "max_actions": manifest.get("max_actions", 128),
    }
    stop_ids = set(runtime["duplex_stage_sampling_params"]["0"].get("stop_token_ids", []))
    if not stop_ids:
        raise ValueError("Native duplex unit-end token IDs are unavailable.")
    fence = DuplexFence(session_id=request_id)
    active = getattr(server, "_active_native_sessions", None)
    if active is None:
        active = server._active_native_sessions = set()
    active.add(request_id)
    native_request = None
    source_samples = 0
    frame_provenance = []
    append_count = 0
    accepted_ids = []
    producer_done = asyncio.Event()
    outputs = []
    buffer = MiniCPMO45PcmAppendBuffer()
    start_ns = time.time_ns()
    start = asyncio.get_running_loop().time()

    async def submit(payload, operation):
        nonlocal native_request, append_count
        result = _check_control(
            await server.engine.append_duplex_input_async(
                request_id,
                mode="append_audio_chunk",
                payload=payload,
                operation_id=operation,
                fence=fence,
                expected_epoch=fence.epoch,
                collect_outputs=False,
            )
        )
        native_request, _ = DuplexRequestClient.request_info(result)
        if native_request is None:
            raise RuntimeError("Native duplex append did not create a data-plane request.")
        append_count += 1

    async def feed():
        nonlocal source_samples
        frames = []
        for event in manifest["events"]:
            await asyncio.sleep(max(0, start + event["available_at_ms"] / 1000 - asyncio.get_running_loop().time()))
            if event["type"] == "video_frame":
                frames.append(await asyncio.to_thread(_video_frame, event["uri"]))
                frame_provenance.append(dict(event))
                continue
            payload = await asyncio.to_thread(_audio_payload, manifest, event)
            payload["video_frames"], frames = frames, []
            while True:
                reservation = buffer.prepare_append(
                    payload, operation_id=f"{request_id}-{event['seq']}-{append_count}", chunk_period_ms=1000
                )
                if reservation is None:
                    break
                reservation.payload["source"] = {
                    "track": event["track"],
                    "start_sample": source_samples,
                    "end_sample": source_samples + reservation.byte_count // 4,
                    "available_at_ms": event["available_at_ms"],
                    "event_seq": event["seq"],
                    "video_frames": [frame_provenance.pop(0) for _ in reservation.payload.get("video_frames", [])],
                }
                source_samples += reservation.byte_count // 4
                try:
                    await submit(reservation.payload, reservation.operation_id)
                except BaseException:
                    reservation.rollback()
                    raise
                reservation.commit()
                payload = {**payload, "audio": "", "video_frames": []}
        if frames:
            raise ValueError("Camera frames remain without a model-visible audio unit.")
        if buffer.has_pending():
            reservation = buffer.prepare_append(
                {"audio": "", "format": "pcm_f32le", "sample_rate_hz": 16000},
                operation_id=f"{request_id}-final",
                chunk_period_ms=1000,
                flush=True,
            )
            if reservation is not None:
                reservation.payload["source"] = {
                    "track": event["track"],
                    "start_sample": source_samples,
                    "end_sample": source_samples + reservation.byte_count // 4,
                    "available_at_ms": event["available_at_ms"],
                    "event_seq": event["seq"],
                    "video_frames": [frame_provenance.pop(0) for _ in reservation.payload.get("video_frames", [])],
                }
                await submit(reservation.payload, reservation.operation_id)
                reservation.commit()
        if frame_provenance:
            raise ValueError("Camera frames remain without a model-visible audio unit.")
        producer_done.set()

    async def consume():
        nonlocal accepted_ids
        # Resumable requests remain unfinished between units. Count native policy
        # boundaries in the accepted cumulative sequence, not request.finished.
        while not producer_done.is_set() or sum(token in stop_ids for token in accepted_ids) < append_count:
            if native_request is None:
                await asyncio.sleep(0.01)
                continue
            batch = await server.engine.collect_duplex_data_plane_outputs_async(
                native_request, response_stage_id=0, timeout=0.25
            )
            for output in batch:
                snapshot = _snapshot_output(output, time.time_ns())
                outputs.append(snapshot)
                if snapshot["final_output_type"] == "text":
                    ids = snapshot["token_ids"]
                    if ids[: len(accepted_ids)] != accepted_ids:
                        raise ValueError("Native duplex rewrote accepted actions without a rollback boundary.")
                    accepted_ids = ids

    tasks = []
    drained = False
    try:
        _check_control(
            await server.engine.open_duplex_session_async(
                request_id,
                session_config=config.as_dict(),
                runtime_config=runtime,
                fence=fence,
                capabilities=DuplexCapabilities.minicpmo45_native().as_dict(),
            )
        )
        start_ns = time.time_ns()
        start = asyncio.get_running_loop().time()
        tasks = [asyncio.create_task(feed()), asyncio.create_task(consume())]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=manifest.get("max_duration_ms", 8000) / 1000 + 120)
        _check_control(
            await server.engine.close_duplex_session_async(request_id, fence=fence, reason="training_window_end")
        )
        windows = _rank_windows(
            await server.engine.collective_rpc("take_minicpm_duplex_trace", args=(request_id,), stage_ids=[0])
        )
        drained = True
        windows, discarded_windows = accepted_windows(windows, accepted_ids, stop_ids)
        if any(w["identity"]["policy_version"] != server.global_steps for w in windows):
            raise ValueError("Duplex policy changed while collecting a bounded session.")
        if not any(w["action"]["loss_mask"] for w in windows):
            raise ValueError("Duplex session contains no sampled actions; forced listening is not a training target.")
        root = Path(os.environ.get("VERL_OMNI_DUPLEX_ARTIFACT_DIR", "duplex_artifacts"))
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / f"{request_id}.pt"
        await asyncio.to_thread(
            torch.save,
            {
                "manifest": manifest,
                "start_ns": start_ns,
                "outputs": outputs,
                "windows": windows,
                "discarded_windows": discarded_windows,
                "termination_reason": "policy_window_end",
                "speech_tail_may_be_truncated": True,
            },
            artifact,
        )
        return TokenOutput(
            token_ids=[],
            log_probs=[],
            stop_reason="completed",
            extra_fields={
                "duplex_windows": windows,
                "duplex_artifact": str(artifact),
                "global_steps": server.global_steps,
            },
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if not drained:
            _check_control(
                await server.engine.close_duplex_session_async(request_id, fence=fence, reason="training_failure")
            )
            await server.engine.collective_rpc(
                "take_minicpm_duplex_trace", args=(request_id, 0, 0, True), stage_ids=[0]
            )
        active.remove(request_id)

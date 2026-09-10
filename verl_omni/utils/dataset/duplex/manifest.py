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
"""Validate bounded input timelines without deriving actions from transcripts."""

import json
import math


def validate_session_manifest(manifest):
    """Validate source ordering/availability before opening a native session."""
    # Parquet represents nested event lists as NumPy arrays; restore JSON-native values.
    manifest = json.loads(json.dumps(manifest, default=lambda value: value.tolist(), allow_nan=False))
    if not manifest.get("session_id") or not isinstance(manifest.get("instructions"), str):
        raise ValueError("Duplex manifests require session_id and instructions.")
    tracks = manifest["input_tracks"]
    events = manifest["events"]
    if not events:
        raise ValueError("A duplex session must contain timed input events.")
    previous_time = -1.0
    ends = {}
    audio_count = 0
    duration = float(manifest.get("max_duration_ms", 8000))
    if not math.isfinite(duration) or not 0 < duration <= 30000:
        raise ValueError("Bounded duplex sessions must be at most 30 seconds.")
    for seq, event in enumerate(events):
        available = float(event["available_at_ms"])
        if event["seq"] != seq or not math.isfinite(available) or not previous_time <= available <= duration:
            raise ValueError("Duplex events must have consecutive seq and ordered, bounded availability times.")
        previous_time = available
        kind = event["type"]
        if kind not in {"audio", "video_frame"}:
            raise NotImplementedError(
                "Live cancellation/rollback needs accepted-action acknowledgements; do not replay transcripts."
            )
        if kind == "video_frame":
            if not event.get("uri") or not 0 <= event["pts_ms"] <= available:
                raise ValueError("Video frames need a URI and causal presentation timestamp.")
            continue
        audio_count += 1
        if not isinstance(event.get("is_speech"), bool) or not isinstance(event.get("force_listen", False), bool):
            raise ValueError("Audio events require an explicit is_speech boolean and boolean force_listen, if set.")
        track = tracks[event["track"]]
        if track.get("sample_rate") != 16000 or not track.get("uri"):
            raise ValueError("MiniCPM duplex audio tracks must be explicit mono 16 kHz sources.")
        start, end = event["start_ms"], event["end_ms"]
        if not 0 <= start < end <= available or start != ends.get(event["track"], 0):
            raise ValueError("Audio spans must be contiguous per channel and available only after their end.")
        if any(float(value) * 16 != int(float(value) * 16) for value in (start, end)):
            raise ValueError("Audio boundaries must align to 16 kHz samples.")
        ends[event["track"]] = end
    if audio_count == 0 or len(ends) != 1:
        raise ValueError("The initial duplex recipe supports exactly one timed input-audio channel.")
    if not manifest.get("ref_audio"):
        raise ValueError("Frozen duplex speech rendering requires an explicit speaker-reference audio URI.")
    for name, default in (("max_context_tokens", 2048), ("max_actions", 128), ("max_tokens_per_unit", 20)):
        value = manifest.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    return manifest

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

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from verl_omni.utils.dataset.duplex.manifest import validate_session_manifest


def manifest():
    return {
        "session_id": "one",
        "instructions": "Listen and respond.",
        "ref_audio": "speaker.wav",
        "input_tracks": {"mic": {"uri": "user.wav", "sample_rate": 16000}},
        "events": [
            {
                "seq": 0,
                "type": "audio",
                "track": "mic",
                "start_ms": 0,
                "end_ms": 1000,
                "available_at_ms": 1000,
                "is_speech": True,
            },
            {
                "seq": 1,
                "type": "audio",
                "track": "mic",
                "start_ms": 1000,
                "end_ms": 2000,
                "available_at_ms": 2000,
                "is_speech": True,
            },
        ],
    }


def test_parquet_transport_preserves_the_timeline(tmp_path):
    source = tmp_path / "sessions.jsonl"
    source.write_text(json.dumps(manifest()) + "\n")
    path = Path(__file__).resolve().parents[2] / "examples/opd_trainer/minicpm_o/prepare_duplex_data.py"
    spec = importlib.util.spec_from_file_location("duplex_prep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.convert_sessions(source)
    parquet = tmp_path / "data.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    restored = pd.read_parquet(parquet).iloc[0]["extra_info"]["minicpm_duplex_session"]
    validate_session_manifest(restored)
    assert len(restored["events"]) == 2
    assert restored["events"][1]["start_ms"] == 1000
    assert restored["ref_audio"].startswith("file://")


@pytest.mark.parametrize("change", ["future", "overlap", "gap", "order", "duration", "rate", "cancel", "speaker"])
def test_invalid_or_unreplayable_input_fails(change):
    value = copy.deepcopy(manifest())
    if change == "future":
        value["events"][0]["available_at_ms"] = 100
    elif change == "overlap":
        value["events"][1]["start_ms"] = 500
    elif change == "gap":
        value["events"][1]["start_ms"] = 1500
    elif change == "order":
        value["events"][1]["seq"] = 0
    elif change == "duration":
        value["max_duration_ms"] = 31000
    elif change == "rate":
        value["input_tracks"]["mic"]["sample_rate"] = 24000
    elif change == "cancel":
        value["events"][1]["type"] = "barge_in"
    else:
        value.pop("ref_audio")
    with pytest.raises((ValueError, NotImplementedError)):
        validate_session_manifest(value)

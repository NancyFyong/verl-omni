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
"""Convert licensed, timed session manifests to the existing V1 parquet transport."""

import argparse
import json
from pathlib import Path

from verl_omni.utils.dataset.duplex.manifest import validate_session_manifest


def convert_sessions(source):
    """Preserve event timing and media references; do not add reference responses."""
    source = Path(source)
    rows = []
    seen = set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        manifest = json.loads(line)
        for track in manifest["input_tracks"].values():
            track["uri"] = str((source.parent / track["uri"]).resolve())
        for event in manifest["events"]:
            if "uri" in event:
                event["uri"] = str((source.parent / event["uri"]).resolve())
        # The native serving adapter accepts file:// references through MediaConnector.
        reference = manifest.get("ref_audio", "")
        if reference and "://" not in reference:
            manifest["ref_audio"] = (source.parent / reference).resolve().as_uri()
        manifest = validate_session_manifest(manifest)
        if manifest["session_id"] in seen:
            raise ValueError("Duplicate duplex session_id in the dataset.")
        seen.add(manifest["session_id"])
        rows.append(
            {
                "data_source": "minicpm_o45_duplex",
                "ability": "duplex_understanding",
                "prompt": [{"role": "user", "content": manifest["instructions"]}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"minicpm_duplex_session": manifest},
            }
        )
    if not rows:
        raise ValueError("Empty duplex session dataset.")
    return rows


def main():
    """Write train/test splits without touching the source session manifests."""
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in (("train", args.train), ("test", args.test)):
        pd.DataFrame(convert_sessions(path)).to_parquet(destination / f"{name}.parquet", index=False)


if __name__ == "__main__":
    main()

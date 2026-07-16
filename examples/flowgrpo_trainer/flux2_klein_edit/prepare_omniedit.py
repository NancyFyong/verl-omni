# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert real OmniEdit parquet shards to VeRL-Omni FLUX I2I data.

The source image is taken directly from ``src_img``. It is letterboxed to a
fixed square so condition latents have a batchable sequence length; no image is
generated or substituted. The long edit instruction (last entry in
``edited_prompt_list``) is used for both rollout text conditioning and
PickScore.
"""

import argparse
import hashlib
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps


def _image_bytes(payload: dict, image_size: int) -> tuple[bytes, str, tuple[int, int]]:
    raw = payload.get("bytes")
    if raw is None:
        raise ValueError(f"OmniEdit image payload has no bytes: {payload.get('path')}")
    source_hash = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        original_size = image.size
        resized = ImageOps.contain(image, (image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        offset = ((image_size - resized.width) // 2, (image_size - resized.height) // 2)
        canvas.paste(resized, offset)
        output = io.BytesIO()
        canvas.save(output, format="PNG")
    return output.getvalue(), source_hash, original_size


def _convert_row(row: dict, image_size: int) -> dict:
    prompts = row["edited_prompt_list"]
    if not prompts:
        raise ValueError(f"OmniEdit row {row['omni_edit_id']} has no edit instruction")
    instruction = prompts[-1].strip()
    source_bytes, source_hash, original_size = _image_bytes(row["src_img"], image_size)
    target_payload = row["edited_img"]
    target_raw = target_payload.get("bytes")
    target_hash = hashlib.sha256(target_raw).hexdigest() if target_raw is not None else None
    return {
        "data_source": "image_edit",
        "prompt": [{"role": "user", "content": instruction}],
        "negative_prompt": [{"role": "user", "content": " "}],
        "images": [{"bytes": source_bytes}],
        "reward_model": {"style": "model", "ground_truth": instruction},
        "extra_info": {
            "dataset": "sayakpaul/OmniEdit-mini",
            "omni_edit_id": row["omni_edit_id"],
            "task": row["task"],
            "source_path": row["src_img"].get("path"),
            "target_path": target_payload.get("path"),
            "source_sha256": source_hash,
            "target_sha256": target_hash,
            "source_width": original_size[0],
            "source_height": original_size[1],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()

    table = pq.read_table(args.input_parquet)
    required = {"omni_edit_id", "task", "src_img", "edited_img", "edited_prompt_list"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"OmniEdit parquet is missing columns: {sorted(missing)}")
    required_rows = args.train_size + args.val_size
    if table.num_rows < required_rows:
        raise ValueError(f"Need {required_rows} rows, but shard contains {table.num_rows}")

    rows = table.slice(0, required_rows).to_pylist()
    converted = [_convert_row(row, args.image_size) for row in rows]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted[: args.train_size]).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(converted[args.train_size :]).to_parquet(output_dir / "test.parquet")
    print(f"Wrote {args.train_size} real train and {args.val_size} real validation rows to {output_dir}")


if __name__ == "__main__":
    main()

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
"""Convert one-image MiniMax H3 Ref2VA JSONL splits to verl-omni parquet files."""

import argparse
from pathlib import Path

from prepare_fl2va_data import convert_image_conditioned_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = convert_image_conditioned_split(
        input_dir,
        "train",
        1,
        args.train_size,
        data_source="minimax_h3_ref2va",
        extra_info={"reference_types": ["image"]},
    )
    validation = convert_image_conditioned_split(
        input_dir,
        "test",
        1,
        args.val_size,
        data_source="minimax_h3_ref2va",
        extra_info={"reference_types": ["image"]},
    )
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()

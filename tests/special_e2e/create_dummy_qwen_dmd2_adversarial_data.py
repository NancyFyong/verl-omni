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
"""Create tiny prompt-plus-RGB parquet data for the Qwen DMD2 adversarial smoke test."""

import argparse
import os

import numpy as np
import pandas as pd

PROMPTS = ("A red circle", "A blue square", "A green triangle", "A yellow star")


def build_rows(size: int, image_size: int) -> list[dict]:
    """Build deterministic text/real-image pairs."""
    rows = []
    for index in range(size):
        image = np.zeros((3, image_size, image_size), dtype=np.float32)
        image[index % 3] = (index + 1) / (size + 1)
        rows.append(
            {
                "data_source": "dmd2_adversarial_smoke",
                "prompt": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
                "real_pixels": image.tolist(),
                "extra_info": {"index": index},
            }
        )
    return rows


def main() -> None:
    """Write train and validation parquet files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_size", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(build_rows(args.train_size, args.image_size)).to_parquet(
        os.path.join(args.output_dir, "train.parquet")
    )
    pd.DataFrame(build_rows(args.val_size, args.image_size)).to_parquet(os.path.join(args.output_dir, "test.parquet"))


if __name__ == "__main__":
    main()

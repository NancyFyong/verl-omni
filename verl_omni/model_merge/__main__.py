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
"""Command-line entrypoint for offline diffusion model publishing."""

import json

from .base_model_merger import generate_config_from_args, parse_args
from .output_validation import validate_artifact


def main() -> None:
    """Dispatch merge or portable validation from the shared CLI."""
    args = parse_args()
    if args.operation == "validate":
        manifest = validate_artifact(args.target_dir)
        print(json.dumps({"integrity": "passed", "architecture": manifest["architecture"]}))
        return

    config = generate_config_from_args(args)
    if config.backend == "fsdp":
        from .fsdp_model_merger import FSDPModelMerger

        merger = FSDPModelMerger(config)
    else:  # pragma: no cover - argparse and ModelMergerConfig both reject this.
        raise NotImplementedError(f"Unknown backend: {config.backend}")
    try:
        result = merger.merge_and_save()
    finally:
        merger.cleanup()
    print(json.dumps({"output_dir": str(result.output_dir), "manifest_path": str(result.manifest_path)}))


if __name__ == "__main__":
    main()

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
"""Command-line model merger; local_dir/target_dir follow the upstream verl CLI."""

import argparse
import json

from . import ModelMergerConfig, merge_model, validate_artifact


def main() -> None:
    """Merge a full Diffusers checkpoint or verify an existing artifact's integrity."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    merge = commands.add_parser("merge", help="Publish a Diffusers transformer or complete base pipeline")
    merge.add_argument("--backend", choices=["fsdp"], default="fsdp")
    merge.add_argument("--local_dir", required=True)
    merge.add_argument("--target_dir", required=True)
    merge.add_argument("--base-model", required=True)
    merge.add_argument("--architecture")
    merge.add_argument("--output-format", choices=["pipeline", "transformer"], default="pipeline")
    merge.add_argument("--component", choices=["transformer", "transformer_2"])
    merge.add_argument("--dtype", choices=["preserve", "float32", "float16", "bfloat16"], default="preserve")
    merge.add_argument("--max-shard-size", type=int, default=2 * 1024**3, help="Output shard budget in bytes")
    merge.add_argument("--trust-checkpoint", action="store_true", help="Acknowledge trusted pickle inputs")
    validate = commands.add_parser("validate", help="Verify portable output hashes, tensor metadata and index")
    validate.add_argument("--target_dir", required=True)
    args = vars(parser.parse_args())
    operation = args.pop("operation")
    if operation == "validate":
        manifest = validate_artifact(args["target_dir"])
        print(json.dumps({"integrity": "passed", "architecture": manifest["architecture"]}))
    else:
        result = merge_model(ModelMergerConfig(**args))
        print(json.dumps({"output_dir": str(result.output_dir), "manifest_path": str(result.manifest_path)}))


if __name__ == "__main__":
    main()

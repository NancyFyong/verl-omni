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
"""Native duplex collection and bounded teacher replay on the existing AR engine."""

from dataclasses import replace

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

from .omni_rollout_adapter import MiniCPMRolloutAdapter

SESSION_KEY = "minicpm_duplex_session"
REPLAY_KEY = "minicpm_duplex_replay"


@OmniRolloutPipelineBase.register("minicpmo_4_5_duplex")
class MiniCPMDuplexRolloutAdapter(MiniCPMRolloutAdapter):
    """Train the Thinker while retaining native frozen speech rendering."""

    supports_async_chunk = True

    @classmethod
    def _pipeline(cls, pipeline_mode):
        from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

        if pipeline_mode not in {"full", "thinker_only"}:
            raise ValueError("MiniCPM duplex supports full student or thinker_only teacher pipelines.")
        return replace(
            MINICPMO_4_5_PIPELINE,
            model_type=f"minicpmo_4_5_opd_{pipeline_mode}",
            stages=MINICPMO_4_5_PIPELINE.stages if pipeline_mode == "full" else MINICPMO_4_5_PIPELINE.stages[:1],
        )

    @classmethod
    def rollout_flags(cls, pipeline_mode="full"):
        """Keep the native Thinker-to-Talker hidden-state handoff."""
        return {0: {"return_hidden_states": True}} if pipeline_mode == "full" else {}

    @classmethod
    def get_deploy_config_base(cls, pipeline_mode="full"):
        """Preserve the pinned native codec connectors and streaming TTS settings."""
        if pipeline_mode != "full":
            return None
        from pathlib import Path

        import vllm_omni

        return str(Path(vllm_omni.__file__).parent / "deploy/minicpmo_4_5.yaml")

    @classmethod
    def get_stage_engine_extras(cls, stage_id, pipeline_mode="full"):
        """Keep per-stage capacities separate from the Thinker response budget."""
        if stage_id == 0:
            return {"enable_chunked_prefill": False, "enable_prefix_caching": False}
        return {
            "max_model_len": 4096 if stage_id == 1 else 65536,
            "max_num_batched_tokens": 8192 if stage_id == 1 else 65536,
        }

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        """Dispatch timed collection or exact token-native teacher replay."""
        options = dict(mm_processor_kwargs or {})
        if SESSION_KEY in options:
            return {"prompt_token_ids": prompt_ids, SESSION_KEY: options[SESSION_KEY]}
        if REPLAY_KEY in options:
            replay = options[REPLAY_KEY]
            if list(prompt_ids) != replay["action"]["prefix_ids"]:
                raise ValueError("Duplex teacher prompt differs from the recorded causal prefix.")
            identity = replay["identity"]
            return {
                "prompt_token_ids": prompt_ids,
                "model_intermediate_buffer": {
                    "duplex": {
                        "data_plane": True,
                        "session_id": identity["session_id"],
                        "incarnation": identity["incarnation"],
                        "epoch": identity["epoch"],
                        "seq": 0,
                        "payload": {},
                        "runtime_config": {"verl_opd": True},
                        "opd_replay": replay,
                    }
                },
            }
        raise ValueError("The MiniCPM duplex pipeline requires a session manifest or native replay window.")

    @classmethod
    async def generate_session(cls, server, prompt, params, request_id):
        """Collect native sessions or require an acknowledged teacher replay score."""
        from .duplex_runtime import collect_session, score_replay

        if SESSION_KEY in prompt:
            return await collect_session(server, prompt[SESSION_KEY], params, request_id)
        if "opd_replay" in prompt.get("model_intermediate_buffer", {}).get("duplex", {}):
            if params.max_tokens != 1:
                raise ValueError("Duplex teacher replay scores exactly one action per request.")
            return await score_replay(server, prompt, params, request_id)
        return None

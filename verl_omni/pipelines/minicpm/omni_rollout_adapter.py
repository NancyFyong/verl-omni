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
"""Bounded MiniCPM-o thinker topology and token-native multimodal prompts."""

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

MINICPM_PROMPT_KEY = "minicpm_prompt"


@OmniRolloutPipelineBase.register("minicpmo_4_5")
class MiniCPMRolloutAdapter(OmniRolloutPipelineBase):
    """Reuse native stage 0 without the streaming duplex or speech stages."""

    supports_async_chunk = False

    @classmethod
    def _pipeline(cls, pipeline_mode):
        from vllm_omni.config.stage_config import PipelineConfig
        from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

        if pipeline_mode != "thinker_only":
            raise ValueError("MiniCPM-o simplex training supports pipeline_mode=thinker_only only.")
        return PipelineConfig(
            model_type="minicpmo_4_5_thinker_only",
            model_arch=MINICPMO_4_5_PIPELINE.model_arch,
            hf_architectures=MINICPMO_4_5_PIPELINE.hf_architectures,
            hf_config_predicate=MINICPMO_4_5_PIPELINE.hf_config_predicate,
            stages=(MINICPMO_4_5_PIPELINE.stages[0],),
        )

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        return list(cls._pipeline(pipeline_mode).stages)

    @classmethod
    def get_pipeline_id(cls, pipeline_mode="thinker_only"):
        return cls._pipeline(pipeline_mode).model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode="thinker_only"):
        from vllm_omni.config.pipeline_registry import register_pipeline

        register_pipeline(cls._pipeline(pipeline_mode))

    @classmethod
    def weight_sync_stage_ids(cls, pipeline_mode="thinker_only"):
        cls.build_stage_configs(pipeline_mode)
        return [0]

    @classmethod
    def prepare_engine_prompt(cls, prompt_ids, model_config, multi_modal_data, mm_processor_kwargs=None):
        processor_kwargs = dict(mm_processor_kwargs or {})
        replay = processor_kwargs.pop(MINICPM_PROMPT_KEY, None)
        if replay is None:
            if multi_modal_data:
                raise ValueError("MiniCPM-o multimodal rollout requires the minicpm_simplex_agent prompt contract.")
            effective_ids = list(prompt_ids)
        else:
            prefix = replay["expanded_ids"]
            if list(prompt_ids[: len(prefix)]) != prefix:
                raise ValueError("MiniCPM-o rollout/teacher prefix differs from the actor's processed prompt.")
            # Only replace media-expanded prompt slots; never decode or re-tokenize the student response.
            effective_ids = [*replay["source_ids"], *prompt_ids[len(prefix) :]]
        return {"prompt_token_ids": effective_ids, "mm_processor_kwargs": processor_kwargs}

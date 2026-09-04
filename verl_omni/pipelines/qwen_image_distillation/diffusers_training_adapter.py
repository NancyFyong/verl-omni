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
"""Qwen-Image training adapter for DMD and DMD2."""

from verl_omni.pipelines.model_base import DiffusionModelBase, DistributionMatchingModelAdapter
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
from verl_omni.trainer.diffusion.distillation.contracts import DistillationPlan
from verl_omni.workers.config import DiffusionModelConfig

from .phase_runner import QwenImageDmdPhaseRunner, build_qwen_dmd_sigmas

__all__ = ["QwenImageDistributionMatching", "QwenImageDmdPhaseRunner", "build_qwen_dmd_sigmas"]


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dmd")
@DiffusionModelBase.register("QwenImagePipeline", algorithm="dmd2")
class QwenImageDistributionMatching(QwenImage, DistributionMatchingModelAdapter):
    """Qwen-Image architecture adapter for DMD and DMD2."""

    @classmethod
    def build_distillation_phase_runner(
        cls,
        model_config: DiffusionModelConfig,
        plan: DistillationPlan,
    ) -> QwenImageDmdPhaseRunner:
        """Build the architecture-owned Qwen DMD phase program."""
        return QwenImageDmdPhaseRunner(model_config, plan)

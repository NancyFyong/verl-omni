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
"""Native duplex windows on the existing V1 agent/teacher/OPD loss path."""

from uuid import uuid4

import ray
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ

from verl_omni.utils.dataset.duplex.manifest import validate_session_manifest

from .duplex_rollout_adapter import REPLAY_KEY, SESSION_KEY
from .duplex_sampling import validate_duplex_sampling


def window_to_output(window, artifact, prompt_length):
    """Use one sampled decision per window, including empty-text listen actions."""
    action = window["action"]
    if len(action["prefix_ids"]) > prompt_length:
        raise ValueError("Duplex prefix exceeds rollout.prompt_length; refusing causal truncation.")
    version = window["identity"]["policy_version"]
    return AgentLoopOutput(
        prompt_ids=action["prefix_ids"],
        response_ids=[action["token_id"]],
        response_mask=[int(action["loss_mask"])],
        response_logprobs=[action["logprob"]],
        reward_score=0.0,
        num_turns=1,
        metrics={},
        extra_fields={
            REPLAY_KEY: window,
            "duplex_artifact": artifact,
            "reward_extra_info": {},
            "min_global_steps": version,
            "max_global_steps": version,
        },
    )


@register("minicpm_duplex_agent")
class MiniCPMDuplexAgentLoop(AgentLoopBase):
    """Collect fresh, timed student actions rather than teacher conversations."""

    async def run(self, sampling_params, **kwargs):
        validate_duplex_sampling(sampling_params)
        manifest = validate_session_manifest(kwargs["extra_info"][SESSION_KEY])
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=[self.tokenizer.eos_token_id],
            sampling_params={**sampling_params, "max_tokens": 1, "logprobs": True},
            mm_processor_kwargs={SESSION_KEY: manifest},
        )
        windows = output.extra_fields["duplex_windows"]
        return [
            window_to_output(window, output.extra_fields["duplex_artifact"], self.rollout_config.prompt_length)
            for window in windows
        ]


@ray.remote
class MiniCPMDuplexAgentLoopWorker(AgentLoopWorkerTQ.__ray_metadata__.modified_class):
    """Score every causal window, including non-final outputs in a session."""

    def _compute_multi_modal_inputs(self, output, input_ids):
        # image_bound selects verl's existing per-sample (non-concatenating) MiniCPM transport.
        return {"image_bound": [], REPLAY_KEY: output.extra_fields[REPLAY_KEY]}

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs, mm_processor_kwargs=None):
        return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        if validate or not self.distillation_enabled or "teacher_ids" in output.extra_fields:
            return
        manager = self.teacher_server_manager
        if (
            manager.distillation_loss_config.loss_mode != "kl"
            or manager.distillation_loss_config.loss_settings.use_topk
        ):
            raise ValueError("Duplex OPD currently supports selected-token reverse KL only.")
        if manager.distillation_loss_config.use_task_rewards:
            raise ValueError("The initial duplex OPD recipe does not define a temporal task reward.")
        window = output.extra_fields[REPLAY_KEY]
        action = window["action"]
        sequence = [*prompt_ids, *response_ids]
        ids = torch.tensor([*sequence[1:], 0], dtype=torch.int32).unsqueeze(1)
        scores = torch.zeros((len(sequence), 1), dtype=torch.float32)
        if action["loss_mask"]:
            routing = (sample_kwargs or {}).get(self.teacher_key)
            key = manager._resolve_teacher_key(routing)
            # Only provenance/payloads travel to the teacher. It re-encodes media
            # with its own frozen encoders, never consumes student features or KV.
            replay = {
                **window,
                "spans": [{k: v for k, v in span.items() if k != "embeddings"} for span in window["spans"]],
            }
            result = await manager.teacher_client[key].generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params={
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "repetition_penalty": 1.0,
                    "logprobs": True,
                },
                mm_processor_kwargs={REPLAY_KEY: replay},
            )
            if (
                result.token_ids != response_ids
                or result.log_probs is None
                or len(result.log_probs) != 1
                or result.extra_fields.get("duplex_prefix_fingerprint") != action["prefix_fingerprint"]
            ):
                raise ValueError("Duplex teacher did not acknowledge the exact native prefix and sampled action.")
            scores[len(prompt_ids) - 1, 0] = result.log_probs[0]
            if not torch.isfinite(scores).all():
                raise ValueError("Duplex teacher returned a non-finite selected-token score.")
        output.extra_fields["teacher_ids"] = ids
        output.extra_fields["teacher_logprobs"] = scores

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        outputs = output if isinstance(output, list) else [output]
        for item in outputs:
            await self._compute_teacher_logprobs(item, item.prompt_ids, item.response_ids, validate, kwargs)
        return await super()._agent_loop_postprocess(outputs, validate, **kwargs)


class MiniCPMDuplexAgentLoopManager(AgentLoopManagerTQ):
    """Reuse the V1 teacher resource pool and TransferQueue orchestration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = MiniCPMDuplexAgentLoopWorker

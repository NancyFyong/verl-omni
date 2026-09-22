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

import os

import ray
from verl.workers.rollout.vllm_rollout import ServerAdapter

from verl_omni.workers.rollout.base import get_rollout_sequence_parallel_size, get_rollout_world_size


class vLLMOmniServerAdapter(ServerAdapter):
    """Reuse verl's transport with SP-aware replica and IPC rank mapping."""

    def __init__(self, config, model_config, device_mesh, replica_rank: int = -1):
        super().__init__(config, model_config, device_mesh, replica_rank=replica_rank)
        if get_rollout_sequence_parallel_size(self.config) == 1:
            return

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        world_size = get_rollout_world_size(self.config)
        self.replica_rank = rank // world_size if replica_rank == -1 else replica_rank
        self.rollout_rank = rank % world_size
        self.node_rank = self.rollout_rank // local_world_size
        self._has_server = self.rollout_rank == 0
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"

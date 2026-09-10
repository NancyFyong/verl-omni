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

import asyncio
import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from verl_omni.pipelines.minicpm.duplex_runtime import _rank_windows, collect_session
from verl_omni.utils.dataset.duplex.contracts import DuplexTrace


def windows():
    trace = DuplexTrace("session", 0, 0, 3, 32, 8)
    trace.input_span(0, [1, 2], torch.ones(2, 4), {})
    trace.action(3, -1.0, policy={}, origin="sampled", seq=1, timestamp_ns=1)
    trace.action(3, -1.0, policy={}, origin="sampled", seq=2, timestamp_ns=2)
    return trace.windows()


def test_rank_trace_disagreement_fails_closed():
    expected = windows()
    other = copy.deepcopy(expected)
    other[0]["action"]["token_id"] = 7
    with pytest.raises(RuntimeError, match="TP ranks"):
        _rank_windows([[expected, other]])


@pytest.mark.parametrize("fail_append", [False, True])
def test_input_arrival_is_not_blocked_by_output_and_cleanup_is_awaited(tmp_path, monkeypatch, fail_append):
    from vllm import SamplingParams
    from vllm_omni.experimental.fullduplex.minicpmo45.adapter import MiniCPMO45NativeDuplexServingAdapter

    audio = tmp_path / "input.wav"
    sf.write(audio, np.zeros(32000, dtype=np.float32), 16000)
    manifest = {
        "session_id": "source",
        "instructions": "Respond naturally.",
        "ref_audio": "speaker.wav",
        "input_tracks": {"mic": {"uri": str(audio), "sample_rate": 16000}},
        "events": [
            {
                "seq": 0,
                "type": "audio",
                "track": "mic",
                "start_ms": 0,
                "end_ms": 1000,
                "available_at_ms": 1000,
                "is_speech": True,
            },
            {
                "seq": 1,
                "type": "audio",
                "track": "mic",
                "start_ms": 1000,
                "end_ms": 2000,
                "available_at_ms": 2000,
                "is_speech": True,
            },
        ],
        "max_context_tokens": 32,
        "max_duration_ms": 2100,
    }
    monkeypatch.setenv("VERL_OMNI_MINICPM_DUPLEX_OPD", "1")
    monkeypatch.setenv("VERL_OMNI_DUPLEX_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    async def prepare(cls, config, model_config):
        return {"duplex_stage_sampling_params": {"0": {"stop_token_ids": [3]}}}

    monkeypatch.setattr(MiniCPMO45NativeDuplexServingAdapter, "prepare_runtime_config", classmethod(prepare))

    class Engine:
        model_config = None

        def __init__(self):
            self.queue = asyncio.Queue()
            self.appended = []
            self.closed = False
            self.discarded = False
            self.received_outputs = 0

        async def open_duplex_session_async(self, *args, **kwargs):
            assert kwargs["runtime_config"]["verl_opd"]["policy_version"] == 3
            assert kwargs["capabilities"]["input_modes"] == ["append_audio_chunk"]
            assert kwargs["capabilities"]["supports_input_append"] is True
            return {"ok": True}

        async def append_duplex_input_async(self, session_id, **kwargs):
            assert kwargs["collect_outputs"] is False
            self.appended.append(kwargs["payload"])
            if len(self.appended) == 2:
                assert self.received_outputs == 0
                if fail_append:
                    raise RuntimeError("append failed")
                for count in (1, 2):
                    await self.queue.put(
                        SimpleNamespace(
                            finished=False,
                            final_output_type="text",
                            multimodal_output={},
                            request_output=SimpleNamespace(
                                outputs=[SimpleNamespace(token_ids=[3] * count, text="hello")]
                            ),
                        )
                    )
            return {
                "ok": True,
                "stage_results": [
                    {
                        "result": {
                            "data_plane_append": True,
                            "supported": True,
                            "request_id": "native",
                            "response_stage_id": 2,
                        }
                    }
                ],
            }

        async def collect_duplex_data_plane_outputs_async(self, request, response_stage_id, timeout):
            assert response_stage_id == 0
            try:
                output = await asyncio.wait_for(self.queue.get(), timeout)
                self.received_outputs += 1
                return [output]
            except TimeoutError:
                return []

        async def close_duplex_session_async(self, *args, **kwargs):
            self.closed = True
            return {"ok": True}

        async def collective_rpc(self, method, args, stage_ids):
            assert self.closed and stage_ids == [0]
            assert method == "take_minicpm_duplex_trace"
            self.discarded = len(args) == 4 and args[-1]
            return [[windows()]]

    async def run():
        engine = Engine()
        server = SimpleNamespace(
            engine=engine, lora_as_adapter=False, global_steps=3, config=SimpleNamespace(max_model_len=32)
        )
        params = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1)
        if fail_append:
            with pytest.raises(RuntimeError, match="append failed"):
                await collect_session(server, manifest, params, "session")
            assert engine.discarded
        else:
            output = await collect_session(server, manifest, params, "session")
            artifact = torch.load(output.extra_fields["duplex_artifact"], weights_only=True)
            assert len(artifact["outputs"]) == 2
            assert artifact["speech_tail_may_be_truncated"] is True
            assert engine.appended[0]["source"]["end_sample"] == 16000
            assert engine.appended[1]["source"]["start_sample"] == 16000
        assert engine.closed
        assert not server._active_native_sessions

    asyncio.run(run())


def test_synthetic_padding_clears_duplex_replay_and_keeps_teacher_shapes():
    from verl.trainer.ppo import padding_utils

    from verl_omni.trainer.omni.distillation import install_teacher_padding

    install_teacher_padding()
    source = {
        "input_ids": torch.tensor([1, 2, 3]),
        "position_ids": torch.arange(3),
        "multi_modal_inputs": {"image_bound": [], "minicpm_duplex_replay": windows()[0]},
        "teacher_ids": torch.ones(3, 1, dtype=torch.long),
        "teacher_logprobs": torch.zeros(3, 1),
    }
    sample, _ = padding_utils.construct_minimal_padding_template(source, {}, 0)
    assert sample["multi_modal_inputs"]["minicpm_duplex_replay"] == {}
    assert sample["teacher_ids"].shape == (2, 1)


def test_native_sessions_fence_weight_and_cache_changes():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

    server = SimpleNamespace(_active_native_sessions={"session"})
    with pytest.raises(RuntimeError, match="drain"):
        vLLMOmniHttpServer._require_drained_native_sessions(server)


def test_worktree_worker_extension_installs_patch_in_fresh_process():
    pytest.importorskip("vllm_omni")
    root = Path(__file__).resolve().parents[4]
    env = {**os.environ, "VERL_OMNI_MINICPM_DUPLEX_OPD": "1", "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
vLLMOmniColocateWorkerExtension()
assert MiniCPMO45OmniForConditionalGeneration._verl_duplex_opd_patched
print('WORKER_PATCH_OK')
""",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WORKER_PATCH_OK" in result.stdout

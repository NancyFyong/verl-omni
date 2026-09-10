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
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minicpm.duplex_agent_loop import MiniCPMDuplexAgentLoopWorker, window_to_output
from verl_omni.pipelines.minicpm.duplex_patch import install_minicpm_duplex_patch
from verl_omni.pipelines.minicpm.duplex_rollout_adapter import MiniCPMDuplexRolloutAdapter
from verl_omni.pipelines.minicpm.duplex_sampling import duplex_log_distribution, validate_duplex_sampling
from verl_omni.pipelines.minicpm.duplex_training import forward_duplex
from verl_omni.utils.dataset.duplex.contracts import DuplexTrace

POLICY = dict(chunk_eos=4, listen=2, speak=3, forbidden=[0, 4], alias_listen=False, forced_token=None)


@pytest.mark.parametrize("alias", [False, True])
def test_two_pass_marginal_distribution_and_gradient(alias):
    logits = torch.tensor([0.2, -0.5, 1.2, 0.8, -0.2, 0.4, 0.6], requires_grad=True)
    logq = duplex_log_distribution(logits, **{**POLICY, "alias_listen": alias})
    raw = logits.softmax(-1)
    second = logits.detach().clone()
    second[[0, 4]] = -torch.inf
    expected = (1 - raw.detach()[4]) * second.softmax(-1)
    expected[4] = raw.detach()[4]
    if alias:
        expected[3] += expected[2]
        expected[2] = 0
    torch.testing.assert_close(logq.exp(), expected)
    torch.testing.assert_close(logq.exp().sum(), torch.tensor(1.0))
    (-logq[3]).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("forced", [2, 4])
def test_forced_tokens_are_deterministic(forced):
    q = duplex_log_distribution(torch.randn(7), **{**POLICY, "forced_token": forced}).exp()
    assert q[forced] == 1
    assert q.sum() == 1


@pytest.mark.parametrize(
    "params", [{"temperature": 0.7}, {"top_k": 20}, {"top_p": 0.8}, {"repetition_penalty": 1.05}, {"min_tokens": 1}]
)
def test_unsupported_probability_processing_fails(params):
    with pytest.raises(ValueError):
        validate_duplex_sampling(params)


def make_window():
    trace = DuplexTrace("session", 0, 0, 3, 32, 10)
    trace.input_span(0, [5, 6], torch.arange(8).reshape(2, 4).float(), {"seq": 1})
    trace.action(2, -1.0, policy=POLICY, origin="sampled", seq=1, timestamp_ns=10)
    return trace, trace.windows()[0]


def test_future_input_cannot_change_an_earlier_window():
    trace, first = make_window()
    trace.input_span(3, [5], torch.ones(1, 4) * 100, {"seq": 2})
    trace.action(3, -0.8, policy=POLICY, origin="sampled", seq=2, timestamp_ns=20)
    earlier = trace.windows()[0]
    assert earlier["action"] == first["action"]
    assert len(earlier["spans"]) == 1
    assert earlier["action"]["prefix_ids"] == [5, 6]


def test_cancellation_distinguishes_context_loss_and_playback():
    trace, _ = make_window()
    trace.action(3, -0.8, policy=POLICY, origin="sampled", seq=1, timestamp_ns=20)
    trace.cancel(retained_tokens=3, played_actions=())
    kept, rolled_back = trace.actions
    assert kept["loss_mask"] and kept["context_mask"] and not kept["playback_mask"]
    assert not rolled_back["loss_mask"] and not rolled_back["context_mask"]
    with pytest.raises(ValueError, match="fence"):
        trace.check_fence(incarnation=0, epoch=1, policy_version=3)


def test_context_policy_and_memory_limits_fail_closed():
    trace, _ = make_window()
    with pytest.raises(ValueError, match="fence"):
        trace.check_fence(incarnation=0, epoch=0, policy_version=4)
    with pytest.raises(ValueError, match="prefix"):
        trace.input_span(10, [1], torch.zeros(1, 4), {})
    limited = DuplexTrace("s", 0, 0, 0, 4, 1, max_trace_bytes=2)
    with pytest.raises(ValueError, match="byte budget"):
        limited.input_span(0, [1], torch.zeros(1, 4), {})


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 4)
        self.embed.requires_grad_(False)
        self.head = torch.nn.Linear(4, 7, bias=False)
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        if attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=self.head(inputs_embeds.cumsum(1)))


def test_actor_replay_matches_policy_and_backpropagates_with_padding():
    torch.manual_seed(4)
    _, window = make_window()
    lm = TinyLM()
    module = SimpleNamespace(llm=lm)
    ids = torch.tensor([[0, 5, 6, 2], [1, 1, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    out = forward_duplex(module, ids, mask, None, [window, {}])
    raw = lm.head(window["spans"][0]["embeddings"].sum(0))
    expected = duplex_log_distribution(raw, **POLICY)
    torch.testing.assert_close(out.logits[0, 2], expected)
    (-out.logits[0, 2, 2]).backward()
    assert lm.embed.weight.grad is None
    assert torch.isfinite(lm.head.weight.grad).all()
    assert lm.head.weight.grad.abs().sum() > 0


def test_actor_rejects_prefix_mismatch():
    _, window = make_window()
    with pytest.raises(ValueError, match="IDs"):
        forward_duplex(SimpleNamespace(llm=TinyLM()), torch.tensor([[5, 1, 2]]), torch.ones(1, 3), None, [window])


def test_listen_only_actions_survive_agent_and_multimodal_transport():
    from verl.utils.model import extract_multi_modal_inputs

    _, window = make_window()
    output = window_to_output(window, "artifact.pt", 32)
    assert output.response_ids == [2]
    assert output.response_mask == [1]
    worker_cls = MiniCPMDuplexAgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    item = worker._compute_multi_modal_inputs(output, None)
    gathered = extract_multi_modal_inputs([item, {"image_bound": [], "minicpm_duplex_replay": {}}])
    assert len(gathered["minicpm_duplex_replay"]) == 2
    assert gathered["minicpm_duplex_replay"][0]["action"]["token_id"] == 2


@pytest.fixture
def native():
    class Model:
        model_stage = "llm"
        max_new_speak_tokens_per_chunk = 20

        def __init__(self, encoder_offset=0):
            self.encoder_offset = encoder_offset
            self.helper = SimpleNamespace(sessions={})
            self.recorded = []

        def get_input_embeddings(self, ids):
            return ids.float().unsqueeze(-1).expand(-1, 4)

        def _duplex_data_plane_helper(self):
            return self.helper

        def preprocess(self, input_ids, input_embeds=None, **kwargs):
            info = kwargs["duplex"]
            key = (info["session_id"], info["incarnation"])
            self.helper.sessions.setdefault(
                key, SimpleNamespace(current_turn_ended=True, pending_terminator_token=None)
            )
            ids = torch.tensor(info["payload"]["native_ids"])
            return ids, self.get_input_embeddings(ids) + self.encoder_offset, {"duplex": {"success": True}}

        def prepare_duplex_sampling(self, logits, sampling_metadata, rows):
            self.rows = rows
            for row in rows:
                if row.payload.get("force_listen"):
                    logits[row.row_idx] = -torch.inf
                    logits[row.row_idx, 2] = 0

        def _sample_minicpmo45_native_duplex_stage0(self, logits, sampling_metadata, *, duplex_rows=None):
            return "upstream"

        def on_requests_finished(self, finished_req_ids):
            return None

        def _minicpmo45_native_duplex_token_ids(self):
            return {"listen_token_id": 2, "tts_bos_token_id": 3, "chunk_eos_token_id": 4}

        def _minicpmo45_native_forbidden_token_ids(self, token_ids):
            return [0, 4]

        def _minicpmo45_duplex_state_for_row(self, index):
            row = self.rows[index]
            return self.helper.sessions[(row.session_id, row.incarnation)]

        def _record_minicpmo45_duplex_generation_token(self, row_idx, sampled):
            self.recorded.append(sampled)

        def _record_minicpmo45_duplex_terminator(self, row_idx, sampled, token_ids):
            pass

    install_minicpm_duplex_patch(Model)
    return Model


def run_native(model, *, force_listen=False, replay=None):
    payload = {"native_ids": [5, 6], "is_speech": True, "force_listen": force_listen}
    info = {
        "session_id": "session",
        "incarnation": 0,
        "epoch": 0,
        "seq": 1,
        "payload": payload,
        "runtime_config": {"verl_opd": {"policy_version": 3, "max_context_tokens": 32, "max_actions": 10}},
    }
    if replay is not None:
        info["opd_replay"] = replay
        info["opd_score_id"] = "stable-score"
    ids, embeddings, _ = model.preprocess(
        torch.zeros(2, dtype=torch.long), duplex=info, request_id="request", duplex_prompt_len=2, duplex_token_offset=0
    )
    row = SimpleNamespace(
        row_idx=0, request_id="request", session_id="session", incarnation=0, seq=1, payload=payload, max_tokens=20
    )
    meta = SimpleNamespace(output_token_ids=[[]], generators={0: torch.Generator().manual_seed(8)})
    logits = torch.tensor([[0.2, 0.0, 0.6, -0.1, 0.3, 0.9, 1.1]])
    model.prepare_duplex_sampling(logits, meta, [row])
    sampled = model._sample_minicpmo45_native_duplex_stage0(logits, meta, duplex_rows=[0])
    return ids, embeddings, sampled


def test_worker_patch_exports_real_ids_exact_logprobs_and_is_idempotent(native):
    model = native()
    patched = native.preprocess
    install_minicpm_duplex_patch(native)
    assert native.preprocess is patched
    ids, _, sampled = run_native(model)
    assert ids.tolist() == [5, 6]
    action = model._verl_duplex_traces[("session", 0, 0)].actions[0]
    assert action["prefix_ids"] == [5, 6]
    assert action["origin"] == "sampled"
    assert action["token_id"] == sampled.sampled_token_ids[0, 0]
    assert action["logprob"] == pytest.approx(sampled.logprobs_tensors.logprobs[0, 0].item())
    model.prepare_duplex_sampling(torch.zeros(1, 7), SimpleNamespace(), [])
    assert model._sample_minicpmo45_native_duplex_stage0(torch.zeros(1, 7), None) == "upstream"


def test_worker_patch_masks_forced_listening(native):
    model = native()
    _, _, sampled = run_native(model, force_listen=True)
    action = model._verl_duplex_traces[("session", 0, 0)].actions[0]
    assert sampled.sampled_token_ids[0, 0] == 2
    assert action["origin"] == "forced" and not action["loss_mask"]


def test_teacher_rebuilds_native_inputs_with_its_own_encoder(native):
    student = native(encoder_offset=100)
    run_native(student)
    window = student._verl_duplex_traces[("session", 0, 0)].windows()[0]
    stripped = {**window, "spans": [{k: v for k, v in span.items() if k != "embeddings"} for span in window["spans"]]}
    teacher = native(encoder_offset=1)
    _, embeddings, sampled = run_native(teacher, replay=stripped)
    torch.testing.assert_close(embeddings[:, 0], torch.tensor([6.0, 7.0]))
    assert sampled.sampled_token_ids[0, 0] == window["action"]["token_id"]
    assert not teacher.helper.sessions
    assert not teacher._verl_duplex_traces
    assert teacher._verl_duplex_scores["stable-score"]["prefix_fingerprint"] == window["action"]["prefix_fingerprint"]


def test_teacher_checks_prefix_before_model_execution(native):
    student = native()
    run_native(student)
    window = copy.deepcopy(student._verl_duplex_traces[("session", 0, 0)].windows()[0])
    window["action"]["prefix_ids"][0] = 1
    with pytest.raises(ValueError, match="fingerprint"):
        run_native(native(), replay=window)


def test_teacher_window_coverage_not_only_last_action():
    _, window = make_window()
    output = window_to_output(window, "artifact.pt", 32)
    calls = []

    class Client:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            assert all("embeddings" not in s for s in kwargs["mm_processor_kwargs"]["minicpm_duplex_replay"]["spans"])
            return SimpleNamespace(
                token_ids=[2],
                log_probs=[-0.5],
                extra_fields={"duplex_prefix_fingerprint": window["action"]["prefix_fingerprint"]},
            )

    cls = MiniCPMDuplexAgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.distillation_enabled = True
    worker.teacher_key = "data_source"
    worker.teacher_server_manager = SimpleNamespace(
        distillation_loss_config=SimpleNamespace(
            loss_mode="kl", use_task_rewards=False, loss_settings=SimpleNamespace(use_topk=False)
        ),
        _resolve_teacher_key=lambda _: "teacher",
        teacher_client={"teacher": Client()},
    )
    asyncio.run(worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False))
    asyncio.run(worker._compute_teacher_logprobs(output, output.prompt_ids, output.response_ids, False))
    assert len(calls) == 1
    assert output.extra_fields["teacher_ids"].shape == (3, 1)
    torch.testing.assert_close(output.extra_fields["teacher_logprobs"][:, 0], torch.tensor([0.0, -0.5, 0.0]))


def test_duplex_teacher_prompt_is_not_decoded_or_retokenized():
    _, window = make_window()
    prompt = MiniCPMDuplexRolloutAdapter.prepare_engine_prompt([5, 6], None, {}, {"minicpm_duplex_replay": window})
    assert prompt["prompt_token_ids"] == [5, 6]
    assert prompt["model_intermediate_buffer"]["duplex"]["opd_replay"]["action"]["token_id"] == 2


def test_async_post_terminator_decode_is_discarded_without_changing_context(native):
    from verl_omni.pipelines.minicpm.duplex_runtime import accepted_windows

    model = native()
    run_native(model, force_listen=True)
    state = model.helper.sessions[("session", 0)]
    state.pending_terminator_token = 2
    model._sample_minicpmo45_native_duplex_stage0(
        torch.zeros(1, 7), SimpleNamespace(output_token_ids=None, generators={}), duplex_rows=[0]
    )
    trace = model._verl_duplex_traces[("session", 0, 0)]
    assert trace.actions[-1]["discarded"]
    assert not trace.actions[-1]["loss_mask"]
    assert state.pending_terminator_token == 2
    kept, discarded = accepted_windows(trace.windows(), [2], {2, 4})
    assert len(kept) == len(discarded) == 1
    with pytest.raises(ValueError, match="accepted"):
        accepted_windows(trace.windows(), [6], {2, 4})


def test_sampling_refresh_picks_up_replay_metadata_installed_during_prefill():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRunnerMixin

    from verl_omni.pipelines.minicpm.duplex_patch import _install_sampling_refresh

    _install_sampling_refresh()
    model = SimpleNamespace(_verl_duplex_requests={"request": {}})

    def prepare(logits, metadata, rows):
        model._verl_duplex_sampling_rows = rows

    model.prepare_duplex_sampling = prepare
    runner = DuplexSamplingRunnerMixin()
    runner.model = model
    runner.input_batch = SimpleNamespace(req_ids=["request"])
    runner.requests = {"request": SimpleNamespace(sampling_params=SimpleNamespace(max_tokens=1))}
    runner.model_intermediate_buffer = {
        "request": {"duplex": {"data_plane": True, "session_id": "session", "incarnation": 0, "seq": 1, "payload": {}}}
    }
    runner._init_duplex_sampling_state()
    runner._resolve_duplex_sampling_hook()
    assert not runner._duplex_sampling_helper.active_request_ids
    runner._apply_duplex_sampling(torch.zeros(1, 7), SimpleNamespace())
    assert model._verl_duplex_sampling_rows[0].request_id == "request"


@pytest.mark.parametrize("receipt_kind", ["valid", "missing", "stale"])
def test_teacher_requires_native_worker_score_receipt(receipt_kind):
    from verl_omni.pipelines.minicpm.duplex_runtime import score_replay

    _, window = make_window()
    prompt = MiniCPMDuplexRolloutAdapter.prepare_engine_prompt([5, 6], None, {}, {"minicpm_duplex_replay": window})

    class Engine:
        async def generate(self, **kwargs):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2])])

        async def collective_rpc(self, method, args, stage_ids):
            assert method == "take_minicpm_duplex_score" and stage_ids == [0]
            if receipt_kind == "missing":
                return [None]
            return [
                [
                    {
                        "prefix_fingerprint": window["action"]["prefix_fingerprint"]
                        if receipt_kind == "valid"
                        else "stale",
                        "identity": window["identity"],
                        "token_id": 2,
                        "logprob": -0.5,
                    }
                ]
            ]

    async def run():
        if receipt_kind == "valid":
            result = await score_replay(SimpleNamespace(engine=Engine()), prompt, None, "score")
            assert result.log_probs == [-0.5]
        else:
            with pytest.raises((ValueError, RuntimeError)):
                await score_replay(SimpleNamespace(engine=Engine()), prompt, None, "score")

    asyncio.run(run())

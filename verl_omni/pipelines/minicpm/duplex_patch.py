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
"""Opt-in worker patch for vLLM-Omni ded8934's native MiniCPM duplex runtime.

Like VLLMOmniHijack, this patches classes in the worker process, not installed
source files. Ordinary requests and frozen speech stages keep upstream behavior.
"""

import copy
import inspect
import time
from functools import wraps

import torch

from verl_omni.utils.dataset.duplex.contracts import DuplexTrace, prefix_fingerprint

from .duplex_sampling import duplex_log_distribution


def _request_state(model):
    if not hasattr(model, "_verl_duplex_requests"):
        model._verl_duplex_requests = {}
        model._verl_duplex_traces = {}
        model._verl_duplex_replay_embeddings = {}
        model._verl_duplex_scores = {}
    return model._verl_duplex_requests


def _trace_key(info):
    return (info["session_id"], info["incarnation"], info["epoch"])


def _replay_embeddings(model, window, request_id, original_preprocess, device):
    action, spans = window["action"], window["spans"]
    ids = action["prefix_ids"]
    if prefix_fingerprint(window["identity"], ids, spans, action["policy"]) != action["prefix_fingerprint"]:
        raise ValueError("Teacher received a mismatched duplex prefix fingerprint.")
    embeddings = model.get_input_embeddings(torch.tensor(ids, device=device)).clone()
    helper = model._duplex_data_plane_helper()
    session_id = f"opd-teacher-{request_id}"
    try:
        for span in spans:
            meta = span["metadata"]
            info = copy.deepcopy(meta["duplex"])
            info["session_id"] = session_id
            info["incarnation"] = 0
            key = (session_id, 0)
            state = helper.sessions.get(key)
            if state is not None:
                state.pending_terminator_token = meta["pending_terminator"]
            expected = torch.tensor(span["ids"], device=device)
            actual, features, _ = original_preprocess(
                model,
                input_ids=expected,
                duplex=info,
                duplex_token_offset=span["offset"],
                duplex_prompt_len=meta["prompt_len"],
            )
            if not torch.equal(actual, expected):
                raise ValueError("Teacher's native media expansion differs from the student's prefix.")
            start, end = span["offset"], span["offset"] + len(span["ids"])
            if end > len(ids):
                raise ValueError("Teacher replay includes input beyond the action's causal prefix.")
            embeddings[start:end] = features.to(embeddings)
    finally:
        helper.sessions.pop((session_id, 0), None)
    return embeddings


def _install_sampling_refresh():
    from vllm_omni.experimental.fullduplex.model_executor import DuplexSamplingRunnerMixin

    original = DuplexSamplingRunnerMixin._apply_duplex_sampling
    if getattr(original, "_verl_duplex_refresh", False):
        return

    @wraps(original)
    def apply(runner, logits, metadata):
        model = runner.model
        requests = getattr(model, "_verl_duplex_requests", {})
        active = set(runner.input_batch.req_ids).intersection(requests)
        if active:
            runner._resolve_duplex_sampling_hook()
            for request_id in active:
                runner._duplex_sampling_helper.refresh_active_request(runner, request_id)
        result = original(runner, logits, metadata)
        if active and not getattr(model, "_verl_duplex_sampling_rows", ()):
            raise RuntimeError("Duplex replay reached prefill but not the native probability sampler.")
        return result

    apply._verl_duplex_refresh = True
    DuplexSamplingRunnerMixin._apply_duplex_sampling = apply


def install_minicpm_duplex_patch(model_cls=None):
    """Install idempotently in each worker before model construction."""
    if model_cls is None:
        import json
        from importlib.metadata import distribution

        pin = "ded8934626aaad1a3e816c3a1d9d742efc012d93"
        source = json.loads(distribution("vllm-omni").read_text("direct_url.json") or "{}")
        if source.get("vcs_info", {}).get("commit_id") != pin:
            raise RuntimeError(f"MiniCPM duplex patch requires the validated vLLM-Omni git revision {pin}.")
        _install_sampling_refresh()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
            MiniCPMO45OmniForConditionalGeneration,
        )

        model_cls = MiniCPMO45OmniForConditionalGeneration
    if getattr(model_cls, "_verl_duplex_opd_patched", False):
        return
    for method, arguments in {
        "preprocess": {"input_ids", "input_embeds"},
        "prepare_duplex_sampling": {"logits", "sampling_metadata", "rows"},
        "_sample_minicpmo45_native_duplex_stage0": {"logits", "sampling_metadata", "duplex_rows"},
        "on_requests_finished": {"finished_req_ids"},
    }.items():
        if not arguments <= inspect.signature(getattr(model_cls, method)).parameters.keys():
            raise RuntimeError(f"Unsupported vLLM-Omni duplex API: {method}; use the repository's pinned revision.")
    original_preprocess = model_cls.preprocess
    original_prepare = model_cls.prepare_duplex_sampling
    original_sample = model_cls._sample_minicpmo45_native_duplex_stage0
    original_finished = model_cls.on_requests_finished

    @wraps(original_preprocess)
    def preprocess(self, input_ids, input_embeds=None, **kwargs):
        info = kwargs.get("duplex", {})
        enabled = info.get("runtime_config", {}).get("verl_opd")
        if self.model_stage != "llm" or not enabled:
            return original_preprocess(self, input_ids, input_embeds, **kwargs)
        requests = _request_state(self)
        request_id = kwargs["request_id"]
        requests[request_id] = info
        offset, prompt_len = kwargs["duplex_token_offset"], kwargs["duplex_prompt_len"]
        replay = info.get("opd_replay")
        if replay is not None:
            if offset >= len(replay["action"]["prefix_ids"]):
                # Async scheduling may run one discarded decode after the scoring token.
                return input_ids, self.get_input_embeddings(input_ids), {}
            if request_id not in self._verl_duplex_replay_embeddings:
                self._verl_duplex_replay_embeddings[request_id] = _replay_embeddings(
                    self, replay, request_id, original_preprocess, input_ids.device
                )
            features = self._verl_duplex_replay_embeddings[request_id]
            ids = replay["action"]["prefix_ids"][offset : offset + len(input_ids)]
            if len(ids) != len(input_ids):
                raise ValueError("Teacher duplex replay scheduled an invalid prefix span.")
            return (
                torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype),
                features[offset : offset + len(ids)],
                {},
            )

        config = info["runtime_config"]["verl_opd"]
        key = _trace_key(info)
        trace = self._verl_duplex_traces.get(key)
        if trace is None:
            trace = DuplexTrace(*key, config["policy_version"], config["max_context_tokens"], config["max_actions"])
            self._verl_duplex_traces[key] = trace
        trace.check_fence(incarnation=info["incarnation"], epoch=info["epoch"], policy_version=config["policy_version"])
        helper = self._duplex_data_plane_helper()
        state = helper.sessions.get((info["session_id"], info["incarnation"]))
        pending = getattr(state, "pending_terminator_token", None)
        result = original_preprocess(self, input_ids, input_embeds, **kwargs)
        actual, features, update = result
        if offset < prompt_len:
            if update.get("duplex", {}).get("success") is not True:
                raise ValueError(f"Native duplex prefill failed: {update}")
            metadata = {
                "duplex": {
                    k: info[k]
                    for k in (
                        "session_id",
                        "incarnation",
                        "epoch",
                        "seq",
                        "payload",
                        "runtime_config",
                        "session_config",
                        "data_plane",
                        "final",
                        "turn_id",
                        "turn_seq",
                        "mode",
                        "scheduler_token_budget",
                        "scheduler_token_id",
                    )
                    if k in info
                },
                "prompt_len": prompt_len,
                "pending_terminator": pending,
            }
            trace.input_span(offset, actual.tolist(), features, metadata)
        elif actual.tolist() != trace.prefix_ids[offset : offset + len(actual)]:
            raise ValueError("Duplex decode tokens differ from the captured native history.")
        return result

    @wraps(original_prepare)
    def prepare(self, logits, sampling_metadata, rows):
        requests = _request_state(self)
        training = [row for row in rows if row.request_id in requests]
        self._verl_duplex_sampling_rows = training
        if not training:
            return original_prepare(self, logits, sampling_metadata, rows)
        if len(training) != len(rows) or len(rows) != logits.shape[0]:
            raise ValueError("Do not mix duplex OPD and ordinary requests in a sampling batch.")
        self._verl_duplex_raw_logits = logits.clone()
        original_prepare(self, logits, sampling_metadata, rows)
        self._verl_duplex_forced = {}
        for row in rows:
            if torch.isfinite(logits[row.row_idx]).sum().item() == 1:
                self._verl_duplex_forced[row.row_idx] = int(logits[row.row_idx].argmax().item())

    @wraps(original_sample)
    def sample(self, logits, sampling_metadata, *, duplex_rows=None):
        rows = getattr(self, "_verl_duplex_sampling_rows", ())
        if not rows:
            return original_sample(self, logits, sampling_metadata, duplex_rows=duplex_rows)
        from vllm.v1.outputs import LogprobsTensors, SamplerOutput

        raw_logits = self._verl_duplex_raw_logits
        token_ids = self._minicpmo45_native_duplex_token_ids()
        selected, logprobs, ranks = [], [], []
        for row in rows:
            info = self._verl_duplex_requests[row.request_id]
            replay = info.get("opd_replay")
            if row.seq != info.get("seq"):
                raise RuntimeError("Duplex sampling sequence differs from its native prefill.")
            discarded = False
            score_id = info.get("opd_score_id")
            receipt = self._verl_duplex_scores.get(score_id)
            if replay is not None:
                if not isinstance(score_id, str) or not score_id:
                    raise ValueError("Duplex teacher replay requires a stable scoring request ID.")
                policy = replay["action"]["policy"]
            else:
                state = self._minicpmo45_duplex_state_for_row(row.row_idx)
                if state is None:
                    raise ValueError("Native duplex session state is missing at sampling.")
                trace = self._verl_duplex_traces[_trace_key(info)]
                recent = [action for action in trace.actions if action["seq"] == row.seq]
                terminators = {
                    token_ids.get(name, -1)
                    for name in ("listen_token_id", "chunk_eos_token_id", "chunk_tts_eos_token_id", "turn_eos_token_id")
                }
                ended = next((action["token_id"] for action in recent if action["token_id"] in terminators), None)
                forced = self._verl_duplex_forced.get(row.row_idx)
                limit = min(row.max_tokens or 20, getattr(self, "max_new_speak_tokens_per_chunk", 20) or 20)
                if ended is not None:
                    forced, discarded = ended, True
                    state.last_terminator_token = ended
                elif forced is None and len(recent) >= max(1, limit - 1):
                    forced = token_ids["chunk_eos_token_id"]
                policy = {
                    "chunk_eos": token_ids["chunk_eos_token_id"],
                    "listen": token_ids["listen_token_id"],
                    "speak": token_ids["tts_bos_token_id"],
                    "forbidden": self._minicpmo45_native_forbidden_token_ids(token_ids),
                    "alias_listen": not state.current_turn_ended and not info["payload"].get("force_listen", False),
                    "forced_token": forced,
                }
            distribution = duplex_log_distribution(raw_logits[row.row_idx], **policy)
            if replay is not None:
                token = replay["action"]["token_id"]
                if receipt is None:
                    self._verl_duplex_scores[score_id] = {
                        "prefix_fingerprint": replay["action"]["prefix_fingerprint"],
                        "identity": replay["identity"],
                        "token_id": token,
                        "logprob": float(distribution[token].item()),
                    }
                else:
                    distribution = distribution.clone()
                    distribution[token] = receipt["logprob"]
            else:
                generator = sampling_metadata.generators.get(row.row_idx)
                token = int(torch.multinomial(distribution.exp(), 1, generator=generator).item())
                if not discarded:
                    if policy["forced_token"] is None and token != policy["chunk_eos"]:
                        self._record_minicpmo45_duplex_generation_token(row.row_idx, token)
                    self._record_minicpmo45_duplex_terminator(row.row_idx, token, token_ids)
                self._verl_duplex_traces[_trace_key(info)].action(
                    token,
                    float(distribution[token].item()),
                    policy=policy,
                    origin="forced" if policy["forced_token"] is not None else "sampled",
                    seq=row.seq,
                    timestamp_ns=time.time_ns(),
                )
                if discarded:
                    trace.actions[-1].update(context_mask=False, loss_mask=False, discarded=True)
            selected.append(token)
            logprobs.append(distribution[token])
            ranks.append((distribution > distribution[token]).sum() + 1)
        ids = torch.tensor(selected, dtype=torch.int32, device=logits.device).unsqueeze(1)
        return SamplerOutput(
            sampled_token_ids=ids,
            logprobs_tensors=LogprobsTensors(
                ids, torch.stack(logprobs).unsqueeze(1), torch.stack(ranks).to(torch.int32)
            ),
        )

    @wraps(original_finished)
    def finished(self, finished_req_ids):
        requests = _request_state(self)
        for request_id in finished_req_ids:
            requests.pop(request_id, None)
            self._verl_duplex_replay_embeddings.pop(request_id, None)
        return original_finished(self, finished_req_ids)

    model_cls.preprocess = preprocess
    model_cls.prepare_duplex_sampling = prepare
    model_cls._sample_minicpmo45_native_duplex_stage0 = sample
    model_cls.on_requests_finished = finished
    model_cls._verl_duplex_opd_patched = True

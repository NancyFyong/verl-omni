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

"""vLLM-Omni rollout adapter for MiniMax H3 T2VA, FL2VA, and Ref2VA FlowGRPO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _STEP_AUDIO_ANCHOR,
    _STEP_AUDIO_NOISE_PRED,
    _STEP_AUDIO_ROWS,
    _STEP_BRANCH,
    _STEP_COND_ANCHOR,
    _STEP_SHAPE,
    _STEP_SIGMAS_AUDIO,
    _STEP_SIGMAS_VIDEO,
    _STEP_TRANSFORMER,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    ref2va_reference_image_short_edge,
    serialize_ref_blocks,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    H3_AUDIO_SHIFT,
    H3_VIDEO_SHIFT,
    combine_log_probs,
    configure_flow_scheduler,
    flatten_joint_latents,
    h3_sigma_schedules,
    sample_h3_transition,
)
from .weight_sync import MiniMaxH3WeightSyncMixin

__all__ = ["MiniMaxH3PipelineWithLogProb"]


@dataclass
class _FlowGRPOState:
    """Algorithm-private state; never shared between in-flight requests."""

    task: str
    noise_level: float
    sde_type: str
    selected: set[int]
    generator: torch.Generator
    video_scheduler: FlowMatchSDEDiscreteScheduler
    audio_scheduler: FlowMatchSDEDiscreteScheduler
    replay: dict[str, torch.Tensor]
    current_latents: list[torch.Tensor] = field(default_factory=list)
    next_latents: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)


def _flow_grpo_options(sampling) -> dict[str, Any]:
    """Resolve the same sampling options for serial and request-local execution."""
    if int(sampling.num_outputs_per_prompt or 1) != 1:
        raise NotImplementedError("MiniMax H3 FlowGRPO supports one output per request.")
    extra = sampling.extra_args or {}
    return {
        "noise_level": float(extra.get("noise_level", 0.8)),
        "sde_type": str(extra.get("sde_type", "cps")),
        "window_size": extra.get("sde_window_size"),
        "window_range": extra.get("sde_window_range"),
        "sde_contiguous": bool(extra.get("sde_contiguous", True)),
        "seed": int(extra.get("sde_window_seed", 42)) + max(int(extra.get("global_steps", 1)) - 1, 0),
        "max_text_len": int(sampling.max_sequence_length or 1024),
    }


def _pad_first_dim(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[0] > target:
        raise ValueError(f"MiniMax H3 metadata length {value.shape[0]} exceeds configured cap {target}.")
    return F.pad(value, (0, 0) * (value.ndim - 1) + (0, target - value.shape[0]))


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3PipelineWithLogProb(MiniMaxH3WeightSyncMixin, MiniMaxH3Pipeline):
    """Adapt H3 FlowGRPO for request and step batching with isolated replay state.

    Overrides:
        - ``diffuse`` retains the serial/offload loop with FlowGRPO transitions.
        - Step hooks keep dual schedulers, RNG and replay data in each request's state.
        - ``forward`` runs either a serial request or a finite wave using those same hooks.
          Agent Loop token IDs and the training-output contract are preserved in both modes.

    The weight-sync mixin extends upstream prompt encoding while retaining its text-encoder TP collectives.
    """

    supports_request_batch = True

    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=32000),),
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        self._reference_image_short_edge = validate_ref2va_reference_image_short_edge()
        super().__init__(od_config=od_config, prefix=prefix)
        self.install_h3_lora_layout()
        self._flow_grpo_noise_level = 0.8
        self._flow_grpo_sde_type = "cps"
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_seed = 42
        self._flow_grpo_trajectory: dict[str, torch.Tensor] = {}
        self._h3_max_text_len = 1024

    def _configure_flow_grpo(self, request: OmniDiffusionRequest) -> None:
        options = _flow_grpo_options(request.sampling_params)
        self._h3_max_text_len = options.pop("max_text_len")
        for name, value in options.items():
            setattr(self, f"_flow_grpo_{name}", value)
        self._flow_grpo_trajectory = {}

    def _layout_outputs(
        self,
        branch: MiniMaxH3DenoiseBranch,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        *,
        max_text_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        max_text_len = self._h3_max_text_len if max_text_len is None else max_text_len
        used_seq_len = int(packed["cu_seqlens"][1].item())
        video_rows = int(branch.img_pos.shape[0])
        audio_rows = int(branch.audio_pos.shape[0])
        layout_cap = video_rows + audio_rows + max_text_len
        text_len = int(text_embeddings.shape[0])
        if text_len > max_text_len:
            raise ValueError(f"MiniMax H3 encoded text length {text_len} exceeds max_sequence_length={max_text_len}.")

        prompt = F.pad(text_embeddings, (0, 0, 0, max_text_len - text_len)).unsqueeze(0)
        prompt_mask = F.pad(
            torch.ones(text_len, dtype=torch.long, device=text_embeddings.device),
            (0, max_text_len - text_len),
        ).unsqueeze(0)
        position_ids = _pad_first_dim(packed["img_position_ids"][:used_seq_len], layout_cap).unsqueeze(0)
        token_tags = _pad_first_dim(branch.static_kwargs["token_tags"][:used_seq_len], layout_cap).unsqueeze(0)
        text_indices = _pad_first_dim(packed["text_pos"].view(-1), max_text_len).unsqueeze(0)
        return {
            "prompt_embeds": prompt,
            "prompt_embeds_mask": prompt_mask,
            "h3_seq_len": torch.tensor([used_seq_len], device=text_embeddings.device),
            "h3_video_rows": torch.tensor([video_rows], device=text_embeddings.device),
            "h3_audio_rows": torch.tensor([audio_rows], device=text_embeddings.device),
            "h3_position_ids": position_ids,
            "h3_token_tags": token_tags,
            "h3_video_indices": branch.img_pos.unsqueeze(0),
            "h3_audio_indices": branch.audio_pos.unsqueeze(0),
            "h3_text_indices": text_indices,
            "h3_video_update_mask": branch.update_mask_dev.unsqueeze(0),
        }

    def _ref2va_replay_outputs(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        target_video_rows: torch.Tensor,
        target_audio_rows: torch.Tensor,
        visual_anchor: torch.Tensor | None,
        audio_anchor: torch.Tensor | None,
        ref_blocks: list[dict[str, Any]],
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        max_text_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        max_text_len = self._h3_max_text_len if max_text_len is None else max_text_len
        text_len = int(text_embeddings.shape[0])
        if text_len > max_text_len:
            raise ValueError(f"MiniMax H3 encoded text length {text_len} exceeds max_sequence_length={max_text_len}.")
        prompt = F.pad(text_embeddings, (0, 0, 0, max_text_len - text_len)).unsqueeze(0)
        prompt_mask = F.pad(
            torch.ones(text_len, dtype=torch.long, device=text_embeddings.device),
            (0, max_text_len - text_len),
        ).unsqueeze(0)
        prompt_tags = F.pad(text_tags, (0, max_text_len - text_len)).unsqueeze(0)
        ref_block_meta, ref_block_count = serialize_ref_blocks(ref_blocks)
        condition_video = (
            visual_anchor
            if visual_anchor is not None
            else target_video_rows.new_zeros((0, target_video_rows.shape[-1]))
        )
        condition_audio = (
            audio_anchor if audio_anchor is not None else target_audio_rows.new_zeros((0, target_audio_rows.shape[-1]))
        )
        return {
            "prompt_embeds": prompt,
            "prompt_embeds_mask": prompt_mask,
            "prompt_token_tags": prompt_tags,
            "latent_meta": torch.tensor(
                [[target_video_rows.shape[0], target_audio_rows.shape[0], latent_t, latent_h, latent_w, audio_t]],
                dtype=torch.long,
                device=text_embeddings.device,
            ),
            "condition_video_rows": condition_video.unsqueeze(0),
            "condition_audio_rows": condition_audio.unsqueeze(0),
            "condition_video_row_count": torch.tensor([[condition_video.shape[0]]], dtype=torch.long),
            "condition_audio_row_count": torch.tensor([[condition_audio.shape[0]]], dtype=torch.long),
            "ref_block_meta": ref_block_meta.to(text_embeddings.device).unsqueeze(0),
            "ref_block_count": torch.tensor([[ref_block_count]], dtype=torch.long),
        }

    def _prepare_flow_state(
        self,
        state: StepRequestState,
        options: dict[str, Any],
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
        base_schedule: Sequence[float] | None = None,
    ) -> StepRequestState:
        target_video_rows, target_audio_rows = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        target_video_rows = target_video_rows.to(self.device)
        target_audio_rows = target_audio_rows.to(self.device)

        if task == "ref2va":
            if not ref_blocks:
                raise ValueError("MiniMax H3 Ref2VA requires reference block metadata.")
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        elif task in {"t2va", "fl2va"}:
            keyframe_indices = list(keyframe_frame_indices or [])
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=keyframe_indices if task == "fl2va" else None,
                frame_count=num_frames if task == "fl2va" else None,
            )
        else:
            raise NotImplementedError(f"MiniMax H3 FlowGRPO supports t2va, fl2va, and ref2va, got {task!r}.")
        if base_schedule is not None:
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support distilled checkpoint sigma schedules.")
        if not math.isclose(video_shift, H3_VIDEO_SHIFT, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
            audio_shift, H3_AUDIO_SHIFT, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "MiniMax H3 FlowGRPO requires video_shift=12.0 and audio_shift=3.0 "
                "to keep rollout and Actor sigma schedules aligned."
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if task == "fl2va" and (visual_anchor is None or not keyframe_indices):
            raise ValueError("MiniMax H3 FL2VA rollout did not provide complete visual condition metadata.")
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("MiniMax H3 visual condition shape is missing.")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            ).to(self.device)

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("MiniMax H3 reference audio length is missing.")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            ).to(self.device)

        video_rows = target_video_rows.new_zeros((branch.img_pos.shape[0], target_video_rows.shape[-1]))
        video_rows[branch.update_mask_dev] = target_video_rows
        num_condition_video = int((~branch.update_mask_dev).sum().item())
        if visual_anchor is None:
            if num_condition_video:
                raise ValueError("MiniMax H3 Ref2VA visual condition rows are missing.")
        elif visual_anchor.shape[0] != num_condition_video:
            raise ValueError(
                f"MiniMax H3 visual condition rows {visual_anchor.shape[0]} do not match layout rows "
                f"{num_condition_video}."
            )
        else:
            video_rows[~branch.update_mask_dev] = visual_anchor

        audio_rows = target_audio_rows.new_zeros((branch.audio_pos.shape[0], target_audio_rows.shape[-1]))
        audio_rows[branch.audio_update_mask_dev] = target_audio_rows
        num_condition_audio = int((~branch.audio_update_mask_dev).sum().item())
        if audio_anchor is None:
            if num_condition_audio:
                raise ValueError("MiniMax H3 Ref2VA audio condition rows are missing.")
        elif audio_anchor.shape[0] != num_condition_audio:
            raise ValueError(
                f"MiniMax H3 audio condition rows {audio_anchor.shape[0]} do not match layout rows "
                f"{num_condition_audio}."
            )
        else:
            audio_rows[~branch.audio_update_mask_dev] = audio_anchor

        video_sigmas, audio_sigmas = h3_sigma_schedules(num_steps, video_shift, audio_shift)
        video_scheduler = FlowMatchSDEDiscreteScheduler()
        audio_scheduler = FlowMatchSDEDiscreteScheduler()
        configure_flow_scheduler(video_scheduler, video_sigmas, self.device)
        configure_flow_scheduler(audio_scheduler, audio_sigmas, self.device)
        num_transitions = num_steps - 1
        if options["window_size"] is None:
            selected = set(range(num_transitions))
        else:
            window_size = int(options["window_size"])
            low, high = options["window_range"] or [0, num_transitions]
            high = min(high, num_transitions)
            if low < 0 or window_size <= 0 or high - low < window_size:
                raise ValueError(
                    f"Invalid MiniMax H3 SDE window: size={window_size}, "
                    f"range={[low, high]}, transitions={num_transitions}."
                )
            step_generator = torch.Generator().manual_seed(options["seed"])
            if options["sde_contiguous"]:
                start = int(torch.randint(low, high - window_size + 1, (1,), generator=step_generator).item())
                selected = set(range(start, start + window_size))
            else:
                order = torch.randperm(high - low, generator=step_generator)[:window_size].tolist()
                selected = {low + index for index in order}
        if task == "ref2va":
            replay_outputs = self._ref2va_replay_outputs(
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                target_video_rows=target_video_rows,
                target_audio_rows=target_audio_rows,
                visual_anchor=visual_anchor,
                audio_anchor=audio_anchor,
                ref_blocks=ref_blocks,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                max_text_len=options["max_text_len"],
            )
        else:
            replay_outputs = self._layout_outputs(branch, packed, text_embeddings, max_text_len=options["max_text_len"])
        state.latents = video_rows.float()
        state.timesteps = 1.0 - torch.tensor(video_sigmas[:-1], dtype=torch.float32, device=self.device)
        state.step_index = 0
        state.do_true_cfg = False
        state.extra.update(
            {
                _STEP_BRANCH: branch,
                _STEP_TRANSFORMER: self._transformer_for_task(task),
                _STEP_AUDIO_ROWS: audio_rows.float(),
                _STEP_COND_ANCHOR: visual_anchor,
                _STEP_AUDIO_ANCHOR: audio_anchor,
                _STEP_SIGMAS_VIDEO: video_sigmas,
                _STEP_SIGMAS_AUDIO: audio_sigmas,
                _STEP_SHAPE: {"latent_t": latent_t, "latent_h": latent_h, "latent_w": latent_w, "audio_t": audio_t},
                "flow_grpo": _FlowGRPOState(
                    task=task,
                    noise_level=options["noise_level"],
                    sde_type=options["sde_type"],
                    selected=selected,
                    generator=torch.Generator(device=self.device).manual_seed(seed + 1),
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    replay=replay_outputs,
                ),
            }
        )
        return state

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs) -> None:
        """Shared FlowGRPO transition for serial, request-batch and step execution."""
        flow = state.extra["flow_grpo"]
        branch = state.extra[_STEP_BRANCH]
        video_rows, audio_rows = state.latents, state.extra[_STEP_AUDIO_ROWS]
        audio_velocity = state.extra.pop(_STEP_AUDIO_NOISE_PRED)
        step = state.step_index
        selected = step in flow.selected
        video_transition = sample_h3_transition(
            flow.video_scheduler,
            video_rows[branch.update_mask_dev].unsqueeze(0),
            noise_pred[branch.update_mask_dev].unsqueeze(0),
            step,
            noise_level=flow.noise_level if selected else 0.0,
            sde_type=flow.sde_type,
            generator=flow.generator,
            return_log_prob=selected,
        )
        audio_transition = sample_h3_transition(
            flow.audio_scheduler,
            audio_rows[branch.audio_update_mask_dev].unsqueeze(0),
            audio_velocity[branch.audio_update_mask_dev].unsqueeze(0),
            step,
            noise_level=flow.noise_level if selected else 0.0,
            sde_type=flow.sde_type,
            generator=flow.generator,
            return_log_prob=selected,
        )
        next_video_rows, next_audio_rows = video_rows.clone(), audio_rows.clone()
        next_video_rows[branch.update_mask_dev] = video_transition[0][0]
        next_audio_rows[branch.audio_update_mask_dev] = audio_transition[0][0]
        if state.extra[_STEP_COND_ANCHOR] is not None:
            next_video_rows[~branch.update_mask_dev] = state.extra[_STEP_COND_ANCHOR]
        if state.extra[_STEP_AUDIO_ANCHOR] is not None:
            next_audio_rows[~branch.audio_update_mask_dev] = state.extra[_STEP_AUDIO_ANCHOR]
        if selected:
            video_log_prob, audio_log_prob = video_transition[1], audio_transition[1]
            if video_log_prob is None or audio_log_prob is None:
                raise RuntimeError("MiniMax H3 rollout did not compute log probabilities.")
            if flow.task == "ref2va":
                current_video, current_audio = (
                    video_rows[branch.update_mask_dev],
                    audio_rows[branch.audio_update_mask_dev],
                )
                next_video, next_audio = (
                    next_video_rows[branch.update_mask_dev],
                    next_audio_rows[branch.audio_update_mask_dev],
                )
            else:
                current_video, current_audio = video_rows, audio_rows
                next_video, next_audio = next_video_rows, next_audio_rows
            flow.current_latents.append(flatten_joint_latents(current_video.unsqueeze(0), current_audio.unsqueeze(0)))
            flow.next_latents.append(flatten_joint_latents(next_video.unsqueeze(0), next_audio.unsqueeze(0)))
            flow.log_probs.append(combine_log_probs(video_log_prob, audio_log_prob))
            flow.step_indices.append(step)
        state.latents = next_video_rows.float()
        state.extra[_STEP_AUDIO_ROWS] = next_audio_rows.float()
        state.step_index += 1

    def _trajectory(self, state: StepRequestState) -> dict[str, torch.Tensor]:
        """Materialize one request's selected transitions and Actor replay metadata."""
        flow = state.extra["flow_grpo"]
        if not flow.current_latents:
            raise RuntimeError("MiniMax H3 rollout selected no stochastic transitions.")
        video_sigmas = [state.extra[_STEP_SIGMAS_VIDEO][i] for i in flow.step_indices]
        audio_sigmas = [state.extra[_STEP_SIGMAS_AUDIO][i] for i in flow.step_indices]
        return {
            "all_latents": torch.stack(flow.current_latents, dim=1),
            "all_next_latents": torch.stack(flow.next_latents, dim=1),
            "all_timesteps": (1.0 - torch.tensor(video_sigmas, device=self.device)).unsqueeze(0),
            "all_log_probs": torch.stack(flow.log_probs, dim=1),
            "h3_step_indices": torch.tensor(flow.step_indices, device=self.device).unsqueeze(0),
            "h3_audio_timesteps": (1.0 - torch.tensor(audio_sigmas, device=self.device)).unsqueeze(0),
            **flow.replay,
        }

    def diffuse(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the serial/offload path while sharing transitions with step mode."""
        options = {
            name: getattr(self, f"_flow_grpo_{name}")
            for name in (
                "noise_level",
                "sde_type",
                "window_size",
                "window_range",
                "sde_contiguous",
                "seed",
            )
        }
        options["max_text_len"] = self._h3_max_text_len
        state = self._prepare_flow_state(
            StepRequestState(request_id="serial", sampling=OmniDiffusionSamplingParams()), options, **kwargs
        )
        branch, transformer = state.extra[_STEP_BRANCH], state.extra[_STEP_TRANSFORMER]
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            with self.progress_bar(total=state.total_steps) as progress:
                while not state.denoise_completed:
                    step = state.step_index
                    sigma = float(state.extra[_STEP_SIGMAS_VIDEO][step])
                    video_t, audio_t = 1.0 - sigma, 1.0 - float(state.extra[_STEP_SIGMAS_AUDIO][step])
                    self.record_denoise_step(step, normalized_timestep=sigma)
                    video_velocity, audio_velocity = transformer(
                        **branch.forward_kwargs(
                            video_rows=state.latents,
                            audio_rows=state.extra[_STEP_AUDIO_ROWS],
                            t_video=video_t,
                            t_audio=audio_t,
                            imgvid_cond_timestep=max(video_t, MINIMAX_H3_IMGVID_COND_TIMESTEP),
                            audio_ref_cond_timestep=max(audio_t, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                        )
                    )
                    state.extra[_STEP_AUDIO_NOISE_PRED] = audio_velocity
                    self.step_scheduler(state, video_velocity)
                    progress.update()
        self.record_denoise_step(None)
        self._flow_grpo_trajectory = self._trajectory(state)
        video_latent = minimax_h3_unpatchify_video_tokens(
            state.latents[branch.update_mask_dev],
            latent_shape=(kwargs["latent_t"], kwargs["latent_h"] // 2, kwargs["latent_w"] // 2, 24),
            patch_size=(1, 2, 2),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            state.extra[_STEP_AUDIO_ROWS][branch.audio_update_mask_dev],
            audio_t=kwargs["audio_t"] * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    def prepare_encode(self, state: StepRequestState, **kwargs) -> StepRequestState:
        """Encode one request; shared by continuous and whole-request batching."""
        options = _flow_grpo_options(state.sampling)
        if getattr(self, "_dlo_residency_controller", None) is not None:
            raise ValueError(
                "MiniMax H3 FlowGRPO batching/step execution does not support distributed layerwise offload."
            )
        if getattr(state.sampling, "quality", None) == "high":
            raise ValueError("MiniMax H3 FlowGRPO batching/step execution does not support quality=high Cache-DiT.")
        cache_backend = getattr(getattr(self, "od_config", None), "cache_backend", None)
        if cache_backend not in (None, "none"):
            raise ValueError("MiniMax H3 FlowGRPO batching/step execution does not support cache acceleration.")
        extra = state.sampling.extra_args or {}
        short_edge = extra.get(
            "reference_image_short_edge", getattr(state.sampling, "reference_image_short_edge", None)
        )
        if short_edge is None:
            short_edge = getattr(self, "_reference_image_short_edge", None)
        with ref2va_reference_image_short_edge(short_edge):
            try:
                self._ensure_prompt_text(SimpleNamespace(prompt=state.prompt, sampling_params=state.sampling))
                prompt, media = self._extract_prompt(state.prompt)
                context = self._prepare_request_inputs(
                    prompt=prompt,
                    multi_modal_data=media,
                    sampling=state.sampling,
                    text_conditioning=self._extract_text_conditioning(state.prompt),
                    prepared_reference_videos=self._extract_prepared_reference_videos(state.prompt),
                )
            finally:
                self._h3_prompt_ids = None
        self._prepare_flow_state(state, options, **self._denoise_kwargs(context))
        state.extra[_STEP_SHAPE].update(height=context["height"], width=context["width"])
        state.extra["flow_grpo_policy_version"] = extra.get("global_steps")
        return state

    def denoise_step(self, input_batch, *, states=None, **kwargs):
        """Reuse upstream packed/fallback forwards without mixing policy labels."""
        batch_states = list(input_batch.states if states is None else states)
        versions = {state.extra["flow_grpo_policy_version"] for state in batch_states}
        if len(versions) != 1:
            raise ValueError("MiniMax H3 cannot batch requests from different rollout policy versions.")
        return super().denoise_step(input_batch, states=batch_states, **kwargs)

    def post_decode(self, state: StepRequestState, **kwargs) -> DiffusionOutput:
        """Decode both modalities and attach this request's CPU training payload."""
        return self._with_trajectory(super().post_decode(state, **kwargs), self._trajectory(state))

    def _forward_batch(self, request: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Run a finite request wave through the same per-request step lifecycle."""
        versions = {(req.sampling_params.extra_args or {}).get("global_steps") for req in request.requests}
        if len(versions) != 1:
            raise ValueError("MiniMax H3 cannot batch requests from different rollout policy versions.")
        states = []
        for req in request.requests:
            states.append(
                self.prepare_encode(
                    StepRequestState(request_id=req.request_id, prompt=req.prompt, sampling=req.sampling_params)
                )
            )
        # The upstream H3 denoise_step packs compatible requests and falls back
        # to per-request forwards for mixed DiTs / unsupported attention backends.
        # It stores audio predictions per state; video predictions are row-concatenated.
        while active := [state for state in states if not state.denoise_completed]:
            prediction = self.denoise_step(InputBatch.make_batch(active), states=active)
            pieces = prediction.split([state.latents.shape[0] for state in active])
            for state, noise_pred in zip(active, pieces, strict=True):
                self.step_scheduler(state, noise_pred)
        return [self.post_decode(state) for state in states]

    @torch.no_grad()
    def forward(self, request: OmniDiffusionRequest | DiffusionRequestBatch) -> DiffusionOutput | list[DiffusionOutput]:
        if isinstance(request, OmniDiffusionRequest):
            request = DiffusionRequestBatch(requests=[request])
        if len(request.requests) > 1:
            return self._forward_batch(request)
        req = request.requests[0]
        self._configure_flow_grpo(req)
        extra_args = req.sampling_params.extra_args or {}
        short_edge = extra_args.get(
            "reference_image_short_edge",
            getattr(req.sampling_params, "reference_image_short_edge", None),
        )
        if short_edge is None:
            short_edge = getattr(self, "_reference_image_short_edge", None)
        with ref2va_reference_image_short_edge(short_edge):
            self._ensure_prompt_text(request)
            try:
                output = super().forward(request)
            finally:
                self._h3_prompt_ids = None
        if not self._flow_grpo_trajectory:
            raise RuntimeError("MiniMax H3 FlowGRPO rollout produced no trajectory.")
        return self._with_trajectory(output, self._flow_grpo_trajectory)

    @staticmethod
    def _with_trajectory(output: DiffusionOutput, trajectory: dict[str, torch.Tensor]) -> DiffusionOutput:
        """Use one output schema for serial, request-batch and stepwise rollouts."""
        replay_fields = (
            "all_next_latents",
            "h3_step_indices",
            "h3_audio_timesteps",
            "h3_video_rows",
            "h3_audio_rows",
            "h3_seq_len",
            "h3_position_ids",
            "h3_token_tags",
            "h3_video_indices",
            "h3_audio_indices",
            "h3_text_indices",
            "h3_video_update_mask",
            "prompt_token_tags",
            "latent_meta",
            "condition_video_rows",
            "condition_audio_rows",
            "condition_video_row_count",
            "condition_audio_row_count",
            "ref_block_meta",
            "ref_block_count",
        )
        replay_fields = tuple(key for key in replay_fields if key in trajectory)
        return with_rollout_data(
            output,
            trajectory_latents=trajectory["all_latents"],
            trajectory_log_probs=trajectory["all_log_probs"],
            trajectory_timesteps=trajectory["all_timesteps"],
            prompt_embeddings={
                "prompt_embeds": trajectory["prompt_embeds"],
                "prompt_embeds_mask": trajectory["prompt_embeds_mask"],
            },
            rl={key: trajectory[key] for key in replay_fields},
            to_cpu=True,
        )

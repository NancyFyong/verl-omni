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
"""Automodel (``nemo_automodel``) engine for omni models.

Registered as ``model_type="omni_model", backend="automodel"``. It subclasses
verl's :class:`~verl.workers.engine.automodel.transformer_impl.AutomodelEngineWithLMHead`
and swaps only the model-build step to ``NeMoAutoModelForMultimodalLM`` plus the omni
adapter's ``configure_model`` (mirroring ``OmniFSDPEngine._build_module``), reusing
verl's optimizer / LR schedule / checkpoint / offload / forward machinery unchanged.

Scope (v1): image+text inputs, inheriting the LM-head forward path
(``prepare_model_inputs`` already merges ``multi_modal_inputs`` and handles 3D mrope
``position_ids``). LoRA / rollout weight-sync parity, audio/video, and Megatron
sharding are deferred.

The build-vs-parallelize ordering is the integration's one GPU-only unknown: nemo's
``from_pretrained`` applies parallelization internally, whereas the omni adapter's
``configure_model`` does module surgery (re-register with ``AutoModelForCausalLM``,
redirect ``forward``, force ``tie_word_embeddings=False``, unfuse MoE). This must be
validated against the installed nemo/verl versions on GPU; the helper is deliberately
kept as the single seam where that ordering can be adjusted.
"""

import logging

from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.workers.engine.automodel.transformer_impl import AutomodelEngineWithLMHead
from verl.workers.engine.automodel.utils import maybe_fully_shard_optimizer
from verl.workers.engine.base import EngineRegistry

logger = logging.getLogger(__name__)


def build_omni_distributed_setup(engine_config, distributed_config, device_mesh, moe_mesh):
    """Wrap verl's separate distributed arguments into nemo's ``DistributedSetup``.

    ``NeMoAutoModel.from_pretrained`` accepts distributed settings only as a single
    ``distributed_setup`` object; ``moe_mesh``, ``distributed_config``,
    ``activation_checkpointing``, ``moe_config`` and ``tp_plan`` are listed in nemo's
    ``_DISTRIBUTED_SETUP_ONLY_KWARGS`` and raise ``TypeError`` if passed separately.
    verl still threads them individually, so the adaptation happens here.

    ``MeshContext.from_meshes`` is used rather than ``DistributedSetup.build`` because
    verl has already created the meshes; ``build`` would create its own from
    parallelism sizes and ignore them.

    Args:
        engine_config: ``AutomodelEngineConfig``; supplies ``activation_checkpointing``,
            ``ep_size`` and ``moe_config``.
        distributed_config: ``FSDP2Config`` / ``MegatronFSDPConfig`` / ``DDPConfig``.
        device_mesh: Pre-created device mesh (or None for DDP).
        moe_mesh: Pre-created MoE mesh (or None).

    Returns:
        A ``DistributedSetup`` carrying the meshes, strategy and MoE policy.
    """
    from nemo_automodel.components.distributed.config import DistributedSetup
    from nemo_automodel.components.distributed.mesh import MeshContext

    moe_parallel_config = None
    if engine_config.ep_size > 1:
        from nemo_automodel.components.distributed.config import MoEParallelizerConfig

        moe_kwargs = dict(engine_config.moe_config) if engine_config.moe_config else {}
        if hasattr(distributed_config, "mp_policy"):
            moe_kwargs.setdefault("mp_policy", distributed_config.mp_policy)
        moe_parallel_config = MoEParallelizerConfig(**moe_kwargs)

    return DistributedSetup(
        mesh_context=MeshContext.from_meshes(device_mesh=device_mesh, moe_mesh=moe_mesh),
        strategy_config=distributed_config,
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=engine_config.activation_checkpointing,
    )


def build_automodel_omni_model(model_config, engine_config, distributed_config, device_mesh, moe_mesh):
    """Build an omni multimodal model via ``NeMoAutoModelForMultimodalLM`` + omni adapter.

    Swaps verl's CausalLM model class for the multimodal one and applies the omni
    adapter's ``configure_model`` after the build, as ``OmniFSDPEngine._build_module``
    does for the FSDP backend. Distributed settings go through
    :func:`build_omni_distributed_setup`.

    Args:
        model_config: ``OmniModelConfig`` with ``local_path``, ``hf_config``,
            ``architecture``, ``model_stage``, ``trust_remote_code``.
        engine_config: ``AutomodelEngineConfig`` with distributed / dtype settings.
        distributed_config: ``FSDP2Config`` / ``MegatronFSDPConfig`` / ``DDPConfig``.
        device_mesh: Pre-created device mesh (or None for DDP).
        moe_mesh: Pre-created MoE mesh (or None).

    Returns:
        A HuggingFace multimodal model with nemo's distributed infrastructure applied
        and the omni adapter's ``configure_model`` surgery performed.
    """
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForMultimodalLM
    from verl.utils.torch_dtypes import PrecisionType

    from verl_omni.pipelines.model_base import OmniModelBase

    kwargs = {}

    if engine_config.enable_fp8:
        from nemo_automodel.components.quantization.fp8 import FP8Config

        kwargs["fp8_config"] = FP8Config()

    if engine_config.enable_compile:
        from nemo_automodel.components.utils.compile_utils import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    # Qwen/Llama with ep_size<=1: use the HF implementation, matching build_automodel_model.
    architecture = (model_config.architecture or "").lower()
    if engine_config.ep_size <= 1 and ("qwen" in architecture or "llama" in architecture):
        kwargs["force_hf"] = True

    if engine_config.backend_config and not kwargs.get("force_hf", False):
        from nemo_automodel.components.models.common.utils import BackendConfig

        kwargs["backend"] = BackendConfig(**dict(engine_config.backend_config))

    kwargs["attn_implementation"] = engine_config.attn_implementation
    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    module = NeMoAutoModelForMultimodalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.local_path,
        config=model_config.hf_config,
        distributed_setup=build_omni_distributed_setup(engine_config, distributed_config, device_mesh, moe_mesh),
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )

    adapter_cls = OmniModelBase.get_class_by_name(
        model_config.architecture,
        model_config.model_stage,
        model_config.get("external_lib"),
    )
    return adapter_cls.configure_model(module, model_config)


@EngineRegistry.register(model_type="omni_model", backend=["automodel"], device=["cuda"])
class OmniAutomodelEngine(AutomodelEngineWithLMHead):
    """Automodel engine for omni models (image+text)."""

    def initialize(self):
        """Build the multimodal model, then reuse verl's optimizer/LR/checkpoint setup."""
        self.module = build_automodel_omni_model(
            self.model_config, self.engine_config, self.distributed_config, self.device_mesh, self.moe_mesh
        )
        log_gpu_memory_usage("After Automodel omni model build", logger=logger)

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(self.module)
            maybe_fully_shard_optimizer(self.module, self.optimizer, self.distributed_config)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None
        self._build_checkpointer()

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)
        get_torch_device().empty_cache()

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


def build_omni_distributed_config(engine_config, world_size):
    """Build ``(distributed_config, device_mesh, moe_mesh)`` against nemo's current mesh API.

    Replaces verl's ``build_distributed_config_from_engine_config``, which imports
    ``create_device_mesh`` from ``nemo_automodel.components.distributed.mesh_utils``.
    That helper was made private (``_create_device_meshes``) in nemo PR #2266, so the
    import raises ``ImportError`` on every released nemo 0.5.0 — the PyPI wheel and the
    ``v0.5.0`` tag both post-date the rename. The public entry point is now
    ``MeshContext.build``, used here; the strategy-config half is unchanged from verl's
    version, and the meshes are returned bare so the inherited engine code (grad-norm
    scaling, dp group, checkpoint ranks) keeps working against them.

    Args:
        engine_config: ``AutomodelEngineConfig``; supplies ``distributed_strategy``,
            the ``mp_*`` dtypes and the parallelism sizes.
        world_size: Total number of processes in the job.

    Returns:
        Tuple of ``(distributed_config, device_mesh, moe_mesh)``, matching the shape
        verl's helper returned.
    """
    from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    strategy = engine_config.distributed_strategy
    if strategy == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        distributed_config = FSDP2Config(
            sequence_parallel=engine_config.sequence_parallel,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=PrecisionType.to_dtype(engine_config.mp_param_dtype),
                reduce_dtype=PrecisionType.to_dtype(engine_config.mp_reduce_dtype),
                output_dtype=PrecisionType.to_dtype(engine_config.mp_output_dtype),
                cast_forward_inputs=True,
            ),
            activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=engine_config.defer_fsdp_grad_sync,
        )
    elif strategy == "megatron_fsdp":
        distributed_config = MegatronFSDPConfig(activation_checkpointing=engine_config.activation_checkpointing)
    elif strategy == "ddp":
        distributed_config = DDPConfig(activation_checkpointing=engine_config.activation_checkpointing)
    else:
        raise ValueError(f"Unsupported distributed_strategy: {strategy}")

    mesh_context = MeshContext.build(
        distributed_config,
        ParallelismSizes(
            dp_replicate_size=engine_config.dp_replicate_size,
            tp_size=engine_config.tp_size,
            pp_size=engine_config.pp_size,
            cp_size=engine_config.cp_size,
            ep_size=engine_config.ep_size,
        ),
        world_size=world_size,
    )
    return distributed_config, mesh_context.device_mesh, mesh_context.moe_mesh


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

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
        """Initialize like the parent, but build the meshes via nemo's current API.

        The parent's ``__init__`` calls verl's ``build_distributed_config_from_engine_config``,
        which raises ``ImportError`` against every released nemo 0.5.0 (see
        :func:`build_omni_distributed_config`). That helper is patched out for the
        duration of the ``super().__init__`` call rather than after it, because the
        parent would otherwise fail before returning.
        """
        import verl.workers.engine.automodel.transformer_impl as _verl_automodel

        original = _verl_automodel.build_distributed_config_from_engine_config
        _verl_automodel.build_distributed_config_from_engine_config = build_omni_distributed_config
        try:
            super().__init__(model_config, engine_config, optimizer_config, checkpoint_config, **kwargs)
        finally:
            _verl_automodel.build_distributed_config_from_engine_config = original

    def _build_optimizer(self, module):
        """Build the optimizer against nemo's current ``build_optimizer`` location and signature.

        The parent imports ``build_optimizer`` from ``nemo_automodel.recipes.llm.train_ft``
        and calls it as ``(module, cfg, distributed_config, device_mesh)``. nemo PR #2190
        moved the function to ``components.optim.optimizer`` and changed the signature to
        ``(model, config, *, device_mesh=None)`` — ``distributed_config`` is gone and
        ``device_mesh`` is keyword-only — so the inherited version raises ``ImportError``
        on every released nemo 0.5.0.

        The same PR also dropped ``ConfigNode`` support: the config now has to be an
        ``OptimizerConfig`` or a ``(import_path, kwargs)`` tuple. The tuple form is nemo's
        documented escape hatch for external integrations, so ``_target_`` becomes the
        path element and the remaining keys the kwargs. Which keys are collected is
        unchanged from the parent.
        """
        from nemo_automodel.components.optim.optimizer import build_optimizer

        config = self.optimizer_config
        opt_dict = {
            "_target_": f"{config.optimizer_impl}.{config.optimizer}",
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "eps": config.eps,
            "betas": list(config.betas),
        }

        if config.master_weights:
            opt_dict["master_weights"] = config.master_weights
        if config.store_param_remainders:
            opt_dict["store_param_remainders"] = config.store_param_remainders

        _short_to_torch = {"bf16": "torch.bfloat16", "fp32": "torch.float32", "fp16": "torch.float16"}
        for attr in ("exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"):
            val = getattr(config, attr, None)
            if val is not None:
                opt_dict[attr] = _short_to_torch.get(val, val)

        if config.override_optimizer_config:
            opt_dict.update(config.override_optimizer_config)

        optimizer_path = opt_dict.pop("_target_")
        optimizers = build_optimizer(module, (optimizer_path, opt_dict), device_mesh=self.device_mesh)
        assert len(optimizers) == 1, f"Expected 1 optimizer, got {len(optimizers)}"
        return optimizers[0]

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

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
"""Tiny GPU numerical regression: torchrun --standalone --nproc-per-node=2 this_file.py.

Checks packed varlen forward/backward with LoRA and checkpointing, then repeats
under FSDP2. Uses no checkpoint, rollout engine, model download, or Ray cluster.
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from verl.utils.fsdp_utils import apply_fsdp2

from tests.pipelines.test_minimax_h3_flow_grpo_packed_on_cpu import _flow_batch, _run, _schedulers, _serial
from tests.pipelines.test_minimax_h3_packed_forward_on_cpu import _forward, _inputs, _loss, _models
from verl_omni.trainer.diffusion.diffusion_algos import FlowGRPOLoss
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig


def _setup_models(device, mesh, sharded, backend):
    serial, packed = _models(lora=True, checkpointing=True)
    for model in (serial, packed):
        for name, parameter in model.named_parameters():
            if not any(part in name for part in model._keep_in_fp32_modules):
                parameter.data = parameter.data.to(torch.bfloat16)
        model.to(device)
        model.set_adapter("default")
    if backend == "_flash_3_varlen_hub":
        serial.set_attention_backend(backend)
    packed.set_attention_backend(backend)
    if sharded:
        for model in (serial, packed):
            base = model.base_model.model
            kwargs = dict(mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32))
            apply_fsdp2(
                model,
                kwargs,
                {
                    "wrap_policy": {
                        "transformer_layer_cls_to_wrap": [
                            "MiniMaxH3TransformerBlock",
                            "MiniMaxH3TokenRefinerBlock",
                        ]
                    }
                },
            )
            assert all(
                isinstance(block, FSDPModule)
                for block in [
                    *base.token_refiner.refiner_blocks,
                    *base.transformer_blocks,
                ]
            )
    return serial, packed


def _check(device, mesh, sharded, backend):
    serial, packed = _setup_models(device, mesh, sharded, backend)
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in _inputs(serial).items()
    }
    optimizers = [
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01) for model in (serial, packed)
    ]
    for step in range(2):
        previous, reference = [], []
        for model, enabled in ((serial, False), (packed, True)):
            with torch.no_grad():
                model.set_adapter("old")
                previous.append(_forward(model, inputs, enabled))
                with model.disable_adapter():
                    reference.append(_forward(model, inputs, enabled))
            model.set_adapter("default")
        outputs = [_forward(serial, inputs, False), _forward(packed, inputs, True)]
        for pair in (previous, reference, outputs):
            torch.testing.assert_close(pair[0], pair[1], rtol=2e-2, atol=1e-2)
        with torch.no_grad():
            changed = {
                key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
            }
            changed["video_rows"][1] += 9
            isolated = _forward(packed, changed, True)
            torch.testing.assert_close(isolated[[0, 2]], outputs[1][[0, 2]], rtol=2e-2, atol=1e-2)
            assert not torch.allclose(isolated[1], outputs[1][1])
        losses = [_loss(output, old, ref) for output, old, ref in zip(outputs, previous, reference, strict=True)]
        torch.testing.assert_close(losses[0], losses[1], rtol=2e-2, atol=1e-2)
        for loss in losses:
            loss.backward()
        serial_params = dict(serial.named_parameters())
        compared = 0
        for name, parameter in packed.named_parameters():
            if not parameter.requires_grad:
                continue
            expected, actual = serial_params[name].grad, parameter.grad
            assert expected is not None and actual is not None, name
            if sharded:
                expected, actual = expected.full_tensor(), actual.full_tensor()
            torch.testing.assert_close(expected.float(), actual.float(), rtol=5e-2, atol=1e-3, msg=name)
            compared += 1
        assert compared > 0
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(
            f"rank={dist.get_rank()} backend={backend} fsdp2={sharded} step={step + 1} lora_gradients={compared}: PASS",
            flush=True,
        )


def _check_flow_grpo(device, mesh, sharded, backend, task):
    serial, packed = _setup_models(device, mesh, sharded, backend)
    data = _flow_batch(serial, task).to(device)
    schedulers = _schedulers(device)
    optimizers = [
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01) for model in (serial, packed)
    ]
    config = SimpleNamespace(diffusion_loss=DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.2))
    for step in range(2):
        with torch.no_grad():
            serial.set_adapter("old")
            old = _serial(serial, data, schedulers)[0]
            serial.set_adapter("default")
        outputs = [_serial(serial, data, schedulers), _run(packed, data, True, schedulers)]
        for a, b in zip(*outputs, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-2, atol=1e-3)
        with torch.no_grad():
            changed = data.clone()
            changed["prompt_embeds"][1] += 10
            isolated = _run(packed, changed, True, schedulers)
            for a, b in zip(isolated, outputs[1], strict=True):
                torch.testing.assert_close(a[[0, 2]], b[[0, 2]], rtol=2e-2, atol=1e-3)
        losses = [
            FlowGRPOLoss.compute_loss(
                old_log_prob=old,
                log_prob=output[0],
                advantages=torch.tensor([0.6, -0.8, 0.3], device=device),
                config=config,
            )[0]
            for output in outputs
        ]
        torch.testing.assert_close(*losses, rtol=2e-2, atol=1e-3)
        for loss in losses:
            loss.backward()
        expected_params = dict(serial.named_parameters())
        expected_grads, actual_grads = [], []
        for name, parameter in packed.named_parameters():
            if not parameter.requires_grad:
                continue
            a, b = expected_params[name].grad, parameter.grad
            assert a is not None and b is not None, name
            if sharded:
                a, b = a.full_tensor(), b.full_tensor()
            expected_grads.append(a.float().flatten())
            actual_grads.append(b.float().flatten())
        a, b = torch.cat(expected_grads), torch.cat(actual_grads)
        assert a.norm() > 0 and torch.isfinite(b).all()
        relative_error = ((a - b).norm() / a.norm()).item()
        assert relative_error < 0.05, relative_error
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(
            f"rank={dist.get_rank()} FlowGRPO task={task} fsdp2={sharded} step={step + 1} "
            f"LoRA relative gradient error={relative_error:.6f}: PASS",
            flush=True,
        )


def main():
    """Exercise real varlen kernels and distributed parameter hooks on tiny H3 weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attn-backend", choices=["_flash_3_varlen_hub", "torch_varlen", "native"], default="_flash_3_varlen_hub"
    )
    parser.add_argument("--algorithm", choices=["diffusion_nft", "flow_grpo"], default="diffusion_nft")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        for sharded in (False, True):
            if args.algorithm == "flow_grpo":
                for task in ("t2va", "fl2va", "ref2va"):
                    _check_flow_grpo(device, mesh, sharded, args.attn_backend, task)
            else:
                _check(device, mesh, sharded=sharded, backend=args.attn_backend)
        dist.barrier()
        if dist.get_rank() == 0:
            print("MiniMax H3 packed forward + varlen backward + FSDP2: PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

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

"""PickScore HTTP scorer service.

Protocol compatible with ``verl_omni.utils.reward_score.http_scorer_client``:

Request body: ``pickle.dumps({"images": List[bytes], "prompts": List[str], "metadata": dict})``
Response body: ``pickle.dumps({"scores": List[float]})``

Run one server per GPU, for example:

    CUDA_VISIBLE_DEVICES=4 python examples/reward_servers/pickscore_http_server.py --port 19084
    CUDA_VISIBLE_DEVICES=5 python examples/reward_servers/pickscore_http_server.py --port 19085
    CUDA_VISIBLE_DEVICES=6 python examples/reward_servers/pickscore_http_server.py --port 19086
    CUDA_VISIBLE_DEVICES=7 python examples/reward_servers/pickscore_http_server.py --port 19087
"""

import argparse
import asyncio
import io
import logging
import pickle
from dataclasses import dataclass
from typing import Any

import torch
from aiohttp import web
from PIL import Image

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "yuvalkirstain/PickScore_v1"
DEFAULT_SCORE_SCALE = 26.0


@dataclass
class ScoreRequest:
    images: list[bytes]
    prompts: list[str]
    future: asyncio.Future[list[float]]


class PickScoreScorer:
    def __init__(self, device: str, score_scale: float):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.score_scale = score_scale
        logger.info("Loading PickScore processor: %s", PROCESSOR_NAME)
        self.processor = CLIPProcessor.from_pretrained(PROCESSOR_NAME)
        logger.info("Loading PickScore model: %s on %s", MODEL_NAME, device)
        self.model = CLIPModel.from_pretrained(MODEL_NAME).eval().to(device)

    @staticmethod
    def _extract_feature_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        raise TypeError(f"Unsupported PickScore feature output type: {type(output)}")

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @torch.no_grad()
    def score(self, image_bytes: list[bytes], prompts: list[str], micro_batch_size: int) -> list[float]:
        if len(image_bytes) != len(prompts):
            raise ValueError(f"images/prompts length mismatch: {len(image_bytes)} != {len(prompts)}")

        all_scores: list[float] = []
        for start in range(0, len(image_bytes), micro_batch_size):
            batch_images = [self._decode_image(item) for item in image_bytes[start : start + micro_batch_size]]
            batch_prompts = prompts[start : start + micro_batch_size]

            image_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {key: value.to(device=self.device) for key, value in image_inputs.items()}

            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {key: value.to(device=self.device) for key, value in text_inputs.items()}

            image_embs = self._extract_feature_tensor(self.model.get_image_features(**image_inputs))
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

            text_embs = self._extract_feature_tensor(self.model.get_text_features(**text_inputs))
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

            raw_scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
            scores = raw_scores / self.score_scale
            all_scores.extend(float(score.cpu()) for score in scores)

        return all_scores


class BatchServer:
    def __init__(self, scorer: PickScoreScorer, max_batch_size: int, wait_ms: float, micro_batch_size: int):
        self.scorer = scorer
        self.max_batch_size = max_batch_size
        self.wait_s = wait_ms / 1000.0
        self.micro_batch_size = micro_batch_size
        self.queue: asyncio.Queue[ScoreRequest | None] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None

    async def start(self, app: web.Application):
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self, app: web.Application):
        await self.queue.put(None)
        if self.worker_task is not None:
            await self.worker_task

    async def score(self, image_bytes: list[bytes], prompts: list[str]) -> list[float]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        await self.queue.put(ScoreRequest(images=image_bytes, prompts=prompts, future=future))
        return await future

    async def _batch_worker(self):
        while True:
            first = await self.queue.get()
            if first is None:
                break

            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.wait_s
            while sum(len(item.images) for item in batch) < self.max_batch_size:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self.queue.put(None)
                    break
                batch.append(item)

            try:
                flat_images = [image for req in batch for image in req.images]
                flat_prompts = [prompt for req in batch for prompt in req.prompts]
                scores = await asyncio.to_thread(self.scorer.score, flat_images, flat_prompts, self.micro_batch_size)

                offset = 0
                for req in batch:
                    count = len(req.images)
                    req.future.set_result(scores[offset : offset + count])
                    offset += count
            except Exception as exc:
                logger.exception("PickScore batch failed")
                for req in batch:
                    req.future.set_exception(exc)


async def handle_score(request: web.Request) -> web.Response:
    server: BatchServer = request.app["batch_server"]
    try:
        payload = pickle.loads(await request.read())
        images = payload["images"]
        prompts = payload["prompts"]
        scores = await server.score(images, prompts)
        return web.Response(body=pickle.dumps({"scores": scores}), content_type="application/octet-stream")
    except Exception as exc:
        logger.exception("PickScore request failed")
        return web.Response(body=pickle.dumps({"error": str(exc)}), status=500, content_type="application/octet-stream")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


def create_app(args: argparse.Namespace) -> web.Application:
    scorer = PickScoreScorer(device=args.device, score_scale=args.score_scale)
    batch_server = BatchServer(
        scorer=scorer,
        max_batch_size=args.max_batch_size,
        wait_ms=args.wait_ms,
        micro_batch_size=args.micro_batch_size,
    )
    app = web.Application(client_max_size=args.client_max_size_mb * 1024**2)
    app["batch_server"] = batch_server
    app.router.add_post("/", handle_score)
    app.router.add_post("/score", handle_score)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(batch_server.start)
    app.on_cleanup.append(batch_server.stop)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PickScore HTTP scorer server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19084)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-scale", type=float, default=DEFAULT_SCORE_SCALE)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--wait-ms", type=float, default=20.0)
    parser.add_argument("--client-max-size-mb", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    web.run_app(create_app(args), host=args.host, port=args.port)

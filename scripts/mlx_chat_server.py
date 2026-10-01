#!/usr/bin/env python3
"""Serve an MLX chat model with fast ``n`` sampling for the rap candidate planner.

vllm-metal accepts the OpenAI ``n`` parameter but, on Apple Silicon, decodes the
choices with almost no batching gain (n=16 costs ~4x n=1 per token). This
server implements the subset of ``/v1/chat/completions`` the render server uses
and makes ``n`` cheap in two ways:

- shared prefill: the prompt is prefilled once and its KV cache is copied to
  all ``n`` sequences;
- cross-request batching: requests that arrive within ``--batch-window-ms`` of
  each other (for example both bars of one chunk, with
  ``--concurrent-bar-generation`` on the render server) decode in one batch.

Endpoints: ``GET /health``, ``GET /v1/models``, ``GET /metrics`` (the vLLM
counter names client_simulation.py reads), ``POST /v1/chat/completions``.

Run it in its own uv environment:

    uv run --project envs/rap-mlx-chat python scripts/mlx_chat_server.py \
        --model-path /Volumes/ZBW-SSD1/models/qwen2.5-7b-instruct-a09a3545-mlx-q4g64 \
        --served-model-name qwen-rap

mlx and mlx-lm are imported only when the model is loaded, so request
validation is importable (and unit-tested) from the main environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

SERVER_REVISION = "streammuse.mlx_chat_server.v1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_MAX_N = 64
_MAX_TOKENS = 512
_MAX_MESSAGES = 64
_MAX_MESSAGE_CHARS = 32_000
_ALLOWED_FIELDS = frozenset(
    {"model", "messages", "n", "max_tokens", "temperature", "top_p", "stream", "seed"}
)
_ROLES = frozenset({"system", "user", "assistant"})


class ChatRequestError(ValueError):
    """The request is malformed or outside this server's contract (HTTP 400)."""


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Mapping[str, str], ...]
    n: int
    max_tokens: int
    temperature: float
    top_p: float


def parse_chat_request(payload: object, *, model_id: str) -> ChatRequest:
    if not isinstance(payload, Mapping):
        raise ChatRequestError("request body must be a JSON object")
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise ChatRequestError(f"unsupported request fields: {', '.join(unknown)}")
    if payload.get("model") != model_id:
        raise ChatRequestError("requested model is not served here")
    if payload.get("stream", False) is not False:
        raise ChatRequestError("streaming responses are not supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= _MAX_MESSAGES:
        raise ChatRequestError("messages must be a non-empty list")
    parsed: list[Mapping[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or set(message) != {"role", "content"}
            or message["role"] not in _ROLES
            or not isinstance(message["content"], str)
            or len(message["content"]) > _MAX_MESSAGE_CHARS
        ):
            raise ChatRequestError("each message needs a known role and string content")
        parsed.append({"role": message["role"], "content": message["content"]})
    return ChatRequest(
        messages=tuple(parsed),
        n=_bounded_int(payload.get("n", 1), "n", 1, _MAX_N),
        max_tokens=_bounded_int(payload.get("max_tokens", 16), "max_tokens", 1, _MAX_TOKENS),
        temperature=_bounded_float(payload.get("temperature", 1.0), "temperature", 0.0, 5.0),
        top_p=_bounded_float(payload.get("top_p", 1.0), "top_p", 0.0, 1.0),
    )


def completion_payload(
    *,
    model_id: str,
    texts: Sequence[str],
    finish_reasons: Sequence[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, object]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": text},
                "finish_reason": reason,
            }
            for index, (text, reason) in enumerate(zip(texts, finish_reasons, strict=True))
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@dataclass
class _Job:
    request: ChatRequest
    prompt: list[int]
    future: Future = field(default_factory=Future)


@dataclass
class _Counters:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    requests: int = 0
    batches: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MlxChatRuntime:
    """One resident MLX model; a single scheduler thread owns the GPU."""

    def __init__(
        self,
        model_path: Path,
        *,
        batch_window_seconds: float,
        max_batch_sequences: int,
        cache_limit_bytes: int,
    ) -> None:
        import mlx.core as mx
        from mlx_lm import load

        # Every batch allocates fresh KV caches; without a bound MLX keeps the
        # freed buffers in its pool until it has claimed most of unified memory.
        mx.set_cache_limit(cache_limit_bytes)
        self._model, self._tokenizer = load(str(model_path))
        self._batch_window = batch_window_seconds
        self._max_batch_sequences = max_batch_sequences
        self._queue: queue.Queue[_Job] = queue.Queue()
        self.counters = _Counters()
        self._worker = threading.Thread(target=self._serve_forever, daemon=True)
        self._worker.start()

    def submit(self, request: ChatRequest) -> Future:
        prompt = self._tokenizer.apply_chat_template(
            [dict(message) for message in request.messages], add_generation_prompt=True
        )
        if len(prompt) < 2:
            raise ChatRequestError("prompt is too short")
        job = _Job(request=request, prompt=list(prompt))
        self._queue.put(job)
        return job.future

    def _serve_forever(self) -> None:
        while True:
            jobs = [self._queue.get()]
            sequences = jobs[0].request.n
            deadline = time.monotonic() + self._batch_window
            while sequences < self._max_batch_sequences:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                jobs.append(job)
                sequences += job.request.n
            try:
                results = self._run_batch(jobs)
            except BaseException as exc:  # surface to every waiting request
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue
            for job, result in zip(jobs, results, strict=True):
                job.future.set_result(result)

    def _run_batch(self, jobs: list[_Job]) -> list[dict[str, object]]:
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        generator = BatchGenerator(
            self._model,
            stop_tokens=[[token] for token in self._tokenizer.eos_token_ids],
            completion_batch_size=max(32, sum(job.request.n for job in jobs)),
            prefill_batch_size=8,
        )
        owners: dict[int, tuple[int, int]] = {}
        try:
            for job_index, job in enumerate(jobs):
                # Prefill everything but the last prompt token once; each of the
                # n sequences resumes from a private copy of that cache.
                cache = make_prompt_cache(self._model)
                self._model(mx.array(job.prompt[:-1])[None], cache=cache)
                mx.eval([layer.state for layer in cache])
                sampler = make_sampler(temp=job.request.temperature, top_p=job.request.top_p)
                uids = generator.insert(
                    [job.prompt[-1:]] * job.request.n,
                    [job.request.max_tokens] * job.request.n,
                    caches=[copy.deepcopy(cache) for _ in range(job.request.n)],
                    samplers=[sampler] * job.request.n,
                )
                for choice_index, uid in enumerate(uids):
                    owners[uid] = (job_index, choice_index)
            tokens: dict[int, list[int]] = {uid: [] for uid in owners}
            reasons: dict[int, str] = {}
            while responses := generator.next_generated():
                for response in responses:
                    if response.finish_reason != "stop":
                        tokens[response.uid].append(response.token)
                    if response.finish_reason is not None:
                        reasons[response.uid] = response.finish_reason
        finally:
            generator.close()
            del generator
            mx.clear_cache()

        results: list[dict[str, object]] = []
        for job_index, job in enumerate(jobs):
            uids = sorted(
                (uid for uid, (owner, _) in owners.items() if owner == job_index),
                key=lambda uid: owners[uid][1],
            )
            texts = [self._tokenizer.decode(tokens[uid]) for uid in uids]
            completion_tokens = sum(len(tokens[uid]) for uid in uids)
            results.append(
                {
                    "texts": texts,
                    "finish_reasons": [
                        "stop" if reasons.get(uid) == "stop" else "length" for uid in uids
                    ],
                    "prompt_tokens": len(job.prompt),
                    "completion_tokens": completion_tokens,
                }
            )
        with self.counters.lock:
            self.counters.batches += 1
            self.counters.requests += len(jobs)
            self.counters.prompt_tokens += sum(len(job.prompt) for job in jobs)
            self.counters.generation_tokens += sum(int(r["completion_tokens"]) for r in results)
        return results


def create_app(runtime: MlxChatRuntime, *, model_id: str, model_root: str) -> FastAPI:
    app = FastAPI(title="StreamMUSE MLX chat")

    @app.get("/health")
    def health() -> Mapping[str, object]:
        return {"status": "ok", "server": SERVER_REVISION}

    @app.get("/v1/models")
    def models() -> Mapping[str, object]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "mlx-lm", "root": model_root}],
        }

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        counters = runtime.counters
        with counters.lock:
            lines = [
                f"vllm:prompt_tokens_total {counters.prompt_tokens}",
                f"vllm:generation_tokens_total {counters.generation_tokens}",
                f"vllm:request_success_total {counters.requests}",
                f"streammuse:mlx_chat_batches_total {counters.batches}",
            ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/v1/chat/completions")
    async def chat_completions(http_request: Request) -> JSONResponse:
        try:
            request = parse_chat_request(json.loads(await http_request.body()), model_id=model_id)
            future = runtime.submit(request)
        except (ValueError, UnicodeDecodeError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        result = await run_in_threadpool(future.result)
        return JSONResponse(
            completion_payload(
                model_id=model_id,
                texts=result["texts"],
                finish_reasons=result["finish_reasons"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
            )
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", default="qwen-rap")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--max-batch-sequences", type=int, default=64)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        print("refusing non-loopback bind", file=sys.stderr)
        return 2
    runtime = MlxChatRuntime(
        args.model_path,
        batch_window_seconds=args.batch_window_ms / 1000.0,
        max_batch_sequences=args.max_batch_sequences,
        cache_limit_bytes=int(args.cache_limit_gb * 1024**3),
    )
    app = create_app(
        runtime, model_id=args.served_model_name, model_root=str(args.model_path.resolve())
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ChatRequestError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ChatRequestError(f"{name} must be a number in [{minimum}, {maximum}]")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

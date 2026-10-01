#!/usr/bin/env python3
"""Serve MOSS-TTS-v1.5 on Apple Silicon through mlx-audio, on loopback only.

The HTTP surface is the subset of the SGLang-Omni speech API that
``streammuse.infrastructure.rap.sglang_moss_tts`` uses, so the render server's
``mlx`` backend reuses the same bounded client:

- ``GET /health``
- ``GET /v1/models``: the configured model plus the pinned runtime identity
- ``POST /v1/audio/speech``: JSON in, 24 kHz mono PCM16 WAV out

Run it in its own uv environment (it needs mlx and mlx-audio, which conflict
with the main project's vendored transformers):

    uv run --project envs/rap-mlx-moss python scripts/mlx_moss_server.py \
        --model-path /Volumes/ZBW-SSD1/models/moss-tts-v1.5-cdd3b911-mlx-8bit \
        --model-revision cdd3b911b1585e3f2dbc7775ef10f9926f58850a \
        --quantization affine-q8-g64 \
        --allowed-media-root /path/to/reference/voices

mlx and mlx-audio are imported only when the model is loaded, so the request
validation here is importable (and unit-tested) from the main environment.
FastAPI is imported at module level so route annotations resolve.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import threading
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

SERVER_IDENTITY = "streammuse-mlx-moss"
SERVER_REVISION = "streammuse.mlx_moss_server.v1"
OUTPUT_SAMPLE_RATE_HZ = 24_000
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Upstream MossTTSDelayModel.generate defaults for the text channel; the render
# server only sends the audio sampling settings, exactly like the H200 backends.
UPSTREAM_TEXT_SAMPLING = {"text_temperature": 1.5, "text_top_p": 1.0, "text_top_k": 50}

_MAX_INPUT_CHARS = 2_000
_MAX_INSTRUCTION_CHARS = 1_000
_MAX_NEW_TOKENS = 4_096
_MAX_TOKEN_COUNT = 1_024
_MAX_REFERENCE_BYTES = 64 * 1024 * 1024
_ALLOWED_FIELDS = frozenset(
    {
        "model",
        "input",
        "voice",
        "response_format",
        "stream",
        "ref_audio",
        "ref_text",
        "language",
        "instructions",
        "token_count",
        "max_new_tokens",
        "audio_temperature",
        "audio_top_p",
        "audio_top_k",
        "audio_repetition_penalty",
        "seed",
    }
)


class SpeechRequestError(ValueError):
    """The request is malformed or outside this server's contract (HTTP 400)."""


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    reference_path: Path
    language: str | None
    instruction: str | None
    token_count: int
    max_new_tokens: int
    audio_temperature: float
    audio_top_p: float
    audio_top_k: int
    audio_repetition_penalty: float
    seed: int


def parse_speech_request(
    payload: object, *, model_id: str, allowed_media_root: Path
) -> SpeechRequest:
    """Validate one ``/v1/audio/speech`` body against the bounded contract."""
    if not isinstance(payload, Mapping):
        raise SpeechRequestError("request body must be a JSON object")
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise SpeechRequestError(f"unsupported request fields: {', '.join(unknown)}")
    if payload.get("model") != model_id:
        raise SpeechRequestError("requested model is not served here")
    if payload.get("response_format", "wav") != "wav":
        raise SpeechRequestError("only response_format=wav is supported")
    if payload.get("stream", False) is not False:
        raise SpeechRequestError("streaming responses are not supported")
    text = _bounded_text(payload.get("input"), "input", _MAX_INPUT_CHARS)
    language = _optional_text(payload.get("language"), "language", 64)
    instruction = _optional_text(
        payload.get("instructions"), "instructions", _MAX_INSTRUCTION_CHARS
    )
    reference_path = _reference_path(payload.get("ref_audio"), allowed_media_root)
    return SpeechRequest(
        text=text,
        reference_path=reference_path,
        language=language,
        instruction=instruction,
        token_count=_bounded_int(payload.get("token_count"), "token_count", 1, _MAX_TOKEN_COUNT),
        max_new_tokens=_bounded_int(
            payload.get("max_new_tokens"), "max_new_tokens", 1, _MAX_NEW_TOKENS
        ),
        audio_temperature=_bounded_float(
            payload.get("audio_temperature"), "audio_temperature", 0.0, 10.0
        ),
        audio_top_p=_bounded_float(payload.get("audio_top_p"), "audio_top_p", 0.0, 1.0),
        audio_top_k=_bounded_int(payload.get("audio_top_k"), "audio_top_k", 0, 4_096),
        audio_repetition_penalty=_bounded_float(
            payload.get("audio_repetition_penalty"), "audio_repetition_penalty", 0.0, 10.0
        ),
        seed=_bounded_int(payload.get("seed"), "seed", 0, 2**32 - 1),
    )


def encode_pcm16_wav(samples: np.ndarray, *, sample_rate_hz: int) -> bytes:
    """Encode mono float audio in [-1, 1] as a PCM16 WAV."""
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError("generated audio is empty or non-finite")
    pcm = np.clip(np.round(mono * 32767.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def weights_sha256(model_path: Path) -> str:
    """Hash every safetensors shard (name + bytes) under the model directory."""
    digest = hashlib.sha256()
    shards = sorted(model_path.rglob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors weights under {model_path}")
    for shard in shards:
        digest.update(shard.relative_to(model_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with shard.open("rb") as source:
            for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def runtime_identity(
    *,
    model_path: Path,
    model_revision: str,
    quantization: str,
    audio_tokenizer_revision: str,
) -> dict[str, str]:
    """Describe the pinned MLX runtime; the render server checks it against its pins."""
    from importlib.metadata import distribution, version

    commit = "unknown"
    try:
        direct_url = distribution("mlx-audio").read_text("direct_url.json")
        if direct_url:
            commit = str(json.loads(direct_url).get("vcs_info", {}).get("commit_id") or "unknown")
    except Exception:
        commit = "unknown"
    return {
        "server": SERVER_REVISION,
        "mlx_version": version("mlx"),
        "mlx_audio_version": version("mlx-audio"),
        "mlx_audio_commit": commit,
        "model_revision": model_revision,
        "quantization": quantization,
        "weights_sha256": weights_sha256(model_path),
        "audio_tokenizer_revision": audio_tokenizer_revision,
    }


class MossMlxRuntime:
    """One resident MLX MOSS model; generation is serialized."""

    def __init__(
        self,
        model_path: Path,
        *,
        wired_limit_bytes: int | None = None,
        cache_limit_bytes: int | None = None,
    ) -> None:
        import mlx.core as mx
        from mlx_audio.tts import load

        self._mx = mx
        if wired_limit_bytes:
            mx.set_wired_limit(wired_limit_bytes)
        if cache_limit_bytes is not None:
            # Bound MLX's freed-buffer pool so it cannot grow into swap.
            mx.set_cache_limit(cache_limit_bytes)
        self._model = load(str(model_path))
        if self._model.sample_rate != OUTPUT_SAMPLE_RATE_HZ:
            raise ValueError(
                f"model sample rate {self._model.sample_rate} is not {OUTPUT_SAMPLE_RATE_HZ}"
            )
        self._lock = threading.Lock()
        self._reference_cache: dict[str, Any] = {}

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, dict[str, float]]:
        with self._lock:
            started = time.perf_counter()
            codes = self._reference_codes(request.reference_path)
            encoded = time.perf_counter()
            self._mx.random.seed(request.seed)
            result = next(
                self._model.generate(
                    text=request.text,
                    prompt_audio_codes=codes,
                    max_tokens=request.max_new_tokens,
                    tokens=request.token_count,
                    instruction=request.instruction,
                    language=request.language,
                    audio_temperature=request.audio_temperature,
                    audio_top_p=request.audio_top_p,
                    audio_top_k=request.audio_top_k,
                    audio_repetition_penalty=request.audio_repetition_penalty,
                    **UPSTREAM_TEXT_SAMPLING,
                )
            )
            audio = np.asarray(result.audio, dtype=np.float32)
            generated = time.perf_counter()
            body = encode_pcm16_wav(audio, sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ)
            finished = time.perf_counter()
        return body, {
            "reference": (encoded - started) * 1000.0,
            "generate": (generated - encoded) * 1000.0,
            "encode": (finished - generated) * 1000.0,
            "codec_tokens": float(result.token_count),
        }

    def _reference_codes(self, path: Path) -> Any:
        content = path.read_bytes()
        if not content or len(content) > _MAX_REFERENCE_BYTES:
            raise SpeechRequestError("reference audio size is invalid")
        key = hashlib.sha256(content).hexdigest()
        cached = self._reference_cache.get(key)
        if cached is None:
            cached = self._model.encode_reference_audio(str(path))
            self._mx.eval(cached)
            self._reference_cache[key] = cached
        return cached


def create_app(
    runtime: MossMlxRuntime,
    *,
    model_id: str,
    identity: Mapping[str, str],
    allowed_media_root: Path,
) -> FastAPI:
    app = FastAPI(title="StreamMUSE MLX MOSS")
    model_entry = {
        "id": model_id,
        "object": "model",
        "owned_by": SERVER_IDENTITY,
        "revision": identity["model_revision"],
        "streammuse_runtime": dict(identity),
    }

    @app.get("/health")
    def health() -> Mapping[str, object]:
        return {"status": "ok", "identity": SERVER_IDENTITY}

    @app.get("/v1/models")
    def models() -> Mapping[str, object]:
        return {"object": "list", "data": [model_entry]}

    @app.post("/v1/audio/speech")
    async def speech(http_request: Request) -> Response:
        try:
            payload = json.loads(await http_request.body())
            request = parse_speech_request(
                payload, model_id=model_id, allowed_media_root=allowed_media_root
            )
        except (ValueError, UnicodeDecodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        # Generation is not interruptible; a client that disconnects mid-render
        # just never reads this response.
        body, timings = await run_in_threadpool(runtime.synthesize, request)
        server_timing = ", ".join(
            f"{name};dur={value:.1f}" for name, value in timings.items() if name != "codec_tokens"
        )
        return Response(
            content=body,
            media_type="audio/wav",
            headers={
                "Server-Timing": server_timing,
                "X-StreamMUSE-Codec-Tokens": str(int(timings["codec_tokens"])),
            },
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument(
        "--audio-tokenizer-revision",
        default="3cd226ba2947efa357ef453bcad111b6eafba782",
    )
    parser.add_argument("--allowed-media-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--wired-limit-gb", type=float, default=None)
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        print("refusing non-loopback bind", file=sys.stderr)
        return 2
    model_path = args.model_path.resolve()
    if not (model_path / "audio_tokenizer").is_dir():
        print(
            f"{model_path}/audio_tokenizer is missing; place the pinned MOSS audio tokenizer "
            "there so mlx-audio does not fetch an unpinned one",
            file=sys.stderr,
        )
        return 2
    identity = runtime_identity(
        model_path=model_path,
        model_revision=args.model_revision,
        quantization=args.quantization,
        audio_tokenizer_revision=args.audio_tokenizer_revision,
    )
    print(json.dumps({"runtime_identity": identity}, sort_keys=True), flush=True)
    wired = int(args.wired_limit_gb * 1024**3) if args.wired_limit_gb else None
    runtime = MossMlxRuntime(
        model_path,
        wired_limit_bytes=wired,
        cache_limit_bytes=int(args.cache_limit_gb * 1024**3),
    )
    app = create_app(
        runtime,
        model_id=args.model_id,
        identity=identity,
        allowed_media_root=args.allowed_media_root.resolve(),
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _bounded_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SpeechRequestError(f"{name} must be a non-empty string of at most {maximum} chars")
    return value


def _optional_text(value: object, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, name, maximum)


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SpeechRequestError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise SpeechRequestError(f"{name} must be a number in [{minimum}, {maximum}]")
    return float(value)


def _reference_path(value: object, allowed_media_root: Path) -> Path:
    if not isinstance(value, str) or len(value) > 2048:
        raise SpeechRequestError("ref_audio must be a file:// URI")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise SpeechRequestError("ref_audio must be an absolute local file:// URI")
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_relative_to(allowed_media_root) or not path.is_file():
        raise SpeechRequestError("ref_audio is outside the allowed media root")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

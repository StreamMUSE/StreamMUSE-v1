from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from scipy.io import wavfile

from streammuse.application.rap.execution import (
    ExecutionCancelled,
    ExecutionDeadlineExceeded,
    SynthesisExecutionContext,
)
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.infrastructure.rap.moss_generation import DEFAULT_BASE_SEED
from streammuse.infrastructure.rap.moss_tts import (
    MossBackendUnavailable,
    MossInvalidOutput,
    MossRequestRejected,
)
from streammuse.infrastructure.rap.sglang_moss_tts import (
    SglangMossConfig,
    SglangMossSynthesizer,
)


def _request() -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id="song",
        chunk_index=2,
        start_bar=4,
        end_bar=6,
        text="steady motion",
        syllables=(
            SyllableTarget("steady", 0, ("S", "T", "EH1"), 1, 1.0, 0, 1, 1, 0.2),
            SyllableTarget("motion", 0, ("M", "OW1"), 1, 1.0, 2, 4, 4, 0.8),
        ),
    )


def _wav_bytes(
    *, rate: int = 24_000, channels: int = 1, silent: bool = False
) -> bytes:
    samples = np.zeros(2_400, dtype=np.float32) if silent else np.linspace(-0.4, 0.4, 2_400, dtype=np.float32)
    if channels == 2:
        samples = np.column_stack((samples, samples))
    output = io.BytesIO()
    wavfile.write(output, rate, samples)
    return output.getvalue()


def _config(tmp_path: Path, **options: object) -> SglangMossConfig:
    audio = tmp_path / "reference.wav"
    text = tmp_path / "reference.txt"
    audio.write_bytes(_wav_bytes())
    text.write_text("reference words", encoding="utf-8")
    model_revision = options.pop("model_revision", "model-revision-1")
    return SglangMossConfig.from_files(
        base_url="http://127.0.0.1:30000",
        model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        model_revision=str(model_revision),
        reference_audio_uri="file:///srv/moss/reference.wav",
        reference_audio_file=audio,
        reference_text_file=text,
        runtime_config_sha256="3" * 64,
        **options,
    )


def _execution() -> SynthesisExecutionContext:
    return SynthesisExecutionContext.from_timeout(
        10.0, correlation_id="request-correlation"
    )


def test_exact_request_mapping_and_valid_atomic_wav(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "audio/wav; charset=binary", "server": "omni/1"},
            content=_wav_bytes(),
        )

    client = httpx.Client(
        base_url="http://127.0.0.1:30000",
        transport=httpx.MockTransport(handler),
    )
    synthesizer = SglangMossSynthesizer(_config(tmp_path), client=client)
    output = tmp_path / "artifact" / "source.wav"

    result = synthesizer.synthesize(_request(), output, execution=_execution())

    payload = json.loads(requests[0].content)
    assert payload == {
        "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "input": "steady motion",
        "voice": "default",
        "response_format": "wav",
        "stream": False,
        "ref_audio": "file:///srv/moss/reference.wav",
        "ref_text": "reference words",
        "language": "English",
        "instructions": "clear, rhythmically spoken rap with restrained pitch",
        "token_count": round(_request().duration_seconds * 12.5),
        "max_new_tokens": 256,
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
        "seed": DEFAULT_BASE_SEED + 2_000,
    }
    assert requests[0].headers["X-StreamMUSE-Correlation-ID"] == "request-correlation"
    assert output.read_bytes() == _wav_bytes()
    assert not list(output.parent.glob("*.partial.wav"))
    assert result.sample_rate_hz == 24_000
    assert result.frame_count == 2_400
    assert result.serving_metadata.backend == "sglang-omni"
    assert result.serving_metadata.response_bytes == len(_wav_bytes())


def test_probe_accepts_empty_health_and_requires_exact_model(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, headers={"server": "sglang-omni/0.1.4"})
        return httpx.Response(
            200,
            json={"data": [{"id": "OpenMOSS-Team/MOSS-TTS-v1.5"}]},
        )

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )
    health = synthesizer.probe()
    assert calls == ["/health", "/v1/models"]
    assert health["ready"] is True
    assert health["version"] == "sglang-omni/0.1.4"


@pytest.mark.parametrize(
    ("response", "message"),
    (
        (httpx.Response(503), "returned an error"),
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (
            httpx.Response(
                200,
                headers={"content-length": str(256 * 1024 + 1)},
            ),
            "exceeds the configured byte limit",
        ),
    ),
)
def test_probe_rejects_unhealthy_or_invalid_health_responses(
    tmp_path: Path, response: httpx.Response, message: str
) -> None:
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(lambda _: response),
        ),
    )

    with pytest.raises(MossBackendUnavailable, match=message):
        synthesizer.probe()


def test_probe_converts_transport_timeout_to_backend_unavailable(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow health endpoint", request=request)

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(MossBackendUnavailable, match="probe timed out"):
        synthesizer.probe()


def test_probe_bounds_streamed_health_response_without_content_length(
    tmp_path: Path,
) -> None:
    class OversizedProbeStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (256 * 1024)
            yield b"x"

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=OversizedProbeStream())
            ),
        ),
    )

    with pytest.raises(MossBackendUnavailable, match="response is too large"):
        synthesizer.probe()


def test_hermetic_fastapi_round_trip_recovers_after_service_error(
    tmp_path: Path,
) -> None:
    app = FastAPI()
    service_available = False
    speech_payloads: list[dict[str, object]] = []

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200, headers={"server": "sglang-omni/0.1.4"})

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "data": [
                {
                    "id": "OpenMOSS-Team/MOSS-TTS-v1.5",
                    "revision": "model-revision-1",
                }
            ]
        }

    @app.post("/v1/audio/speech")
    async def speech(request: Request) -> Response:
        nonlocal service_available
        speech_payloads.append(await request.json())
        if not service_available:
            return Response(status_code=503)
        return Response(
            content=_wav_bytes(),
            media_type="audio/wav",
            headers={"server": "sglang-omni/0.1.4"},
        )

    client = TestClient(app, base_url="http://127.0.0.1:30000")
    synthesizer = SglangMossSynthesizer(_config(tmp_path), client=client)
    assert synthesizer.probe()["model_revision"] == "model-revision-1"

    output = tmp_path / "fastapi" / "source.wav"
    with pytest.raises(MossBackendUnavailable):
        synthesizer.synthesize(_request(), output, execution=_execution())
    assert not output.exists()

    service_available = True
    result = synthesizer.synthesize(_request(), output, execution=_execution())

    assert output.read_bytes() == _wav_bytes()
    assert result.serving_metadata.server_version == "sglang-omni/0.1.4"
    assert len(speech_payloads) == 2
    assert all(payload["stream"] is False for payload in speech_payloads)
    synthesizer.close()


def test_probe_reports_matching_revision_and_rejects_mismatch(tmp_path: Path) -> None:
    revision = "model-revision-1"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "OpenMOSS-Team/MOSS-TTS-v1.5",
                        "revision": revision,
                    }
                ]
            },
        )

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )
    assert synthesizer.probe()["model_revision"] == revision

    mismatched = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200)
                if request.url.path == "/health"
                else httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "OpenMOSS-Team/MOSS-TTS-v1.5",
                                "model_revision": "different-revision",
                            }
                        ]
                    },
                )
            ),
        ),
    )
    with pytest.raises(MossBackendUnavailable, match="revision"):
        mismatched.probe()


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"data": {}},
        {
            "data": [
                {"id": "OpenMOSS-Team/MOSS-TTS-v1.5"},
                {"id": "OpenMOSS-Team/MOSS-TTS-v1.5"},
            ]
        },
    ),
)
def test_probe_rejects_invalid_or_ambiguous_model_lists(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    responses = iter((httpx.Response(200), httpx.Response(200, json=payload)))
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(lambda _: next(responses)),
        ),
    )
    with pytest.raises(MossBackendUnavailable):
        synthesizer.probe()


def test_probe_rejects_missing_model_and_invalid_json(tmp_path: Path) -> None:
    responses = iter(
        (
            httpx.Response(200, json={}),
            httpx.Response(200, content=b"not-json"),
        )
    )
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(lambda _: next(responses)),
        ),
    )
    with pytest.raises(MossBackendUnavailable, match="invalid JSON"):
        synthesizer.probe()


@pytest.mark.parametrize("status", (400, 404))
def test_request_rejection_is_not_retried(tmp_path: Path, status: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "/private/secret"}})

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(MossRequestRejected) as caught:
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )
    assert calls == 1
    assert "/private/secret" not in str(caught.value)


@pytest.mark.parametrize("status", (429, 500, 503))
def test_transient_status_is_unavailable_and_not_retried(tmp_path: Path, status: int) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status)

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(MossBackendUnavailable):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )
    assert calls == 1


@pytest.mark.parametrize(
    "response",
    (
        httpx.Response(200, headers={"content-type": "text/plain"}, content=b"secret"),
        httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"truncated"),
        httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes(rate=16_000)),
        httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes(channels=2)),
        httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes(silent=True)),
    ),
)
def test_invalid_outputs_do_not_replace_existing_success(
    tmp_path: Path, response: httpx.Response
) -> None:
    output = tmp_path / "source.wav"
    output.write_bytes(b"existing-success")
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(lambda _: response),
        ),
    )
    with pytest.raises(MossInvalidOutput):
        synthesizer.synthesize(_request(), output, execution=_execution())
    assert output.read_bytes() == b"existing-success"
    assert not list(tmp_path.glob("*.partial.wav"))


def test_content_length_and_streamed_body_are_bounded(tmp_path: Path) -> None:
    config = _config(tmp_path, response_byte_limit=1024)
    synthesizer = SglangMossSynthesizer(
        config,
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "audio/wav", "content-length": "1025"},
                    content=b"",
                )
            ),
        ),
    )
    with pytest.raises(MossInvalidOutput, match="byte limit"):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )


@pytest.mark.parametrize("content_length", (" 100", "+100", "1.0", "-1"))
def test_content_length_must_be_strict_decimal(
    tmp_path: Path, content_length: str
) -> None:
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={
                        "content-type": "audio/wav",
                        "content-length": content_length,
                    },
                    content=_wav_bytes(),
                )
            ),
        ),
    )
    with pytest.raises(MossInvalidOutput, match="Content-Length"):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )


def test_streamed_body_without_content_length_is_still_bounded(
    tmp_path: Path,
) -> None:
    class OversizedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"a" * 700
            yield b"b" * 700

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path, response_byte_limit=1024),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "audio/wav"},
                    stream=OversizedStream(),
                )
            ),
        ),
    )
    output = tmp_path / "render" / "source.wav"

    with pytest.raises(MossInvalidOutput, match="byte limit"):
        synthesizer.synthesize(_request(), output, execution=_execution())

    assert not output.exists()
    assert not list(output.parent.glob("*.partial.wav"))


@pytest.mark.parametrize("error_type", (httpx.ConnectError, httpx.ReadTimeout))
def test_transport_failures_are_bounded_and_never_retried(
    tmp_path: Path, error_type: type[httpx.RequestError]
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_type("service unavailable", request=request)

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(MossBackendUnavailable):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )

    assert calls == 1


def test_expired_execution_does_not_open_an_http_request(tmp_path: Path) -> None:
    now = [10.0]
    execution = SynthesisExecutionContext(
        deadline_monotonic=11.0,
        correlation_id="expired-before-http",
        clock=lambda: now[0],
    )
    now[0] = 11.0
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=_wav_bytes())

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(handler),
        ),
    )

    with pytest.raises(ExecutionDeadlineExceeded):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=execution
        )

    assert calls == 0


def test_server_header_that_looks_like_a_private_path_is_sanitized(
    tmp_path: Path,
) -> None:
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={
                        "content-type": "audio/wav",
                        "server": "/home/private/sglang-build",
                    },
                    content=_wav_bytes(),
                )
            ),
        ),
    )

    result = synthesizer.synthesize(
        _request(), tmp_path / "source.wav", execution=_execution()
    )

    assert result.serving_metadata.server_version == "unknown"


def test_cancelled_download_leaves_no_output_or_partial(tmp_path: Path) -> None:
    execution = _execution()
    response_closed = False

    class CancellingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield _wav_bytes()[:100]
            execution.cancel("client_disconnected")
            yield _wav_bytes()[100:]

        def close(self) -> None:
            nonlocal response_closed
            response_closed = True

    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "audio/wav"},
                    stream=CancellingStream(),
                )
            ),
        ),
    )
    output = tmp_path / "render" / "source.wav"
    with pytest.raises(ExecutionCancelled):
        synthesizer.synthesize(_request(), output, execution=execution)
    assert not output.exists()
    assert not list(output.parent.glob("*.partial.wav"))
    assert response_closed is True
    assert execution.cancellation_outcome == "transport_closed_abort_unconfirmed"


def test_config_rejects_credentials_bad_reference_text_and_mismatched_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="without credentials"):
        SglangMossConfig(
            base_url="http://user:secret@127.0.0.1:30000",
            model_id="model",
            model_revision="revision",
            reference_audio_uri="file:///reference.wav",
            reference_audio_sha256="1" * 64,
            reference_text="words",
            reference_text_sha256="2" * 64,
            runtime_config_sha256="3" * 64,
        )
    text = tmp_path / "bad.txt"
    text.write_bytes(b"\xff")
    audio = tmp_path / "ref.wav"
    audio.write_bytes(_wav_bytes())
    with pytest.raises(ValueError, match="UTF-8"):
        SglangMossConfig.from_files(
            base_url="http://127.0.0.1:30000",
            model_id="model",
            model_revision="revision",
            reference_audio_uri="file:///reference.wav",
            reference_audio_file=audio,
            reference_text_file=text,
            runtime_config_sha256="3" * 64,
        )


@pytest.mark.parametrize(
    "revision", ("", "main", "master", "latest", "unknown", "unavailable")
)
def test_config_requires_immutable_model_revision(
    tmp_path: Path, revision: str
) -> None:
    with pytest.raises(ValueError):
        _config(tmp_path, model_revision=revision)


def test_close_is_idempotent_and_prevents_reuse(tmp_path: Path) -> None:
    synthesizer = SglangMossSynthesizer(
        _config(tmp_path),
        client=httpx.Client(
            base_url="http://127.0.0.1:30000",
            transport=httpx.MockTransport(lambda _: httpx.Response(200)),
        ),
    )
    synthesizer.close()
    synthesizer.close()
    with pytest.raises(MossBackendUnavailable, match="closed"):
        synthesizer.synthesize(
            _request(), tmp_path / "source.wav", execution=_execution()
        )

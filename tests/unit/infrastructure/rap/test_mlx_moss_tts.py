from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
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
from streammuse.infrastructure.rap.mlx_moss_tts import (
    MLX_RUNTIME_IDENTITY_FIELDS,
    MlxMossConfig,
    MlxMossSynthesizer,
    runtime_identity_sha256,
)
from streammuse.infrastructure.rap.moss_generation import DEFAULT_BASE_SEED
from streammuse.infrastructure.rap.moss_tts import MossBackendUnavailable

_IDENTITY = {
    "mlx_version": "0.32.3",
    "mlx_audio_version": "0.5.7",
    "mlx_audio_commit": "94c7716212b2228f178d2f9c7619a591fd1b0b78",
    "quantization": "affine-q8-g64",
    "weights_sha256": "a" * 64,
    "audio_tokenizer_revision": "3cd226ba2947efa357ef453bcad111b6eafba782",
}


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


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    wavfile.write(output, 24_000, np.linspace(-0.4, 0.4, 2_400, dtype=np.float32))
    return output.getvalue()


def _config(**options: object) -> MlxMossConfig:
    return MlxMossConfig(
        base_url="http://127.0.0.1:8030",
        model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
        model_revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        reference_audio_uri="file:///voices/reference.wav",
        reference_audio_sha256="b" * 64,
        runtime_identity=dict(options.pop("runtime_identity", _IDENTITY)),
        **options,
    )


def _models_payload(**runtime_overrides: str) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": "OpenMOSS-Team/MOSS-TTS-v1.5",
                "revision": "cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
                "streammuse_runtime": {**_IDENTITY, **runtime_overrides},
            }
        ],
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://127.0.0.1:8030", transport=httpx.MockTransport(handler))


def _execution(timeout: float = 10.0) -> SynthesisExecutionContext:
    return SynthesisExecutionContext.from_timeout(timeout, correlation_id="request-correlation")


def test_request_omits_reference_text_and_carries_seed_and_token_target(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes())

    synthesizer = MlxMossSynthesizer(_config(), client=_client(handler))
    result = synthesizer.synthesize(_request(), tmp_path / "source.wav", execution=_execution())

    payload = json.loads(requests[0].content)
    assert payload == {
        "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "input": "steady motion",
        "voice": "default",
        "response_format": "wav",
        "stream": False,
        "ref_audio": "file:///voices/reference.wav",
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
    metadata = result.serving_metadata
    assert metadata.backend == "mlx"
    assert metadata.server_identity == "mlx"
    assert metadata.reference_text_sha256 == "unavailable"
    assert metadata.config_sha256 == runtime_identity_sha256(_IDENTITY)
    assert result.model_revision == "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"


def test_probe_accepts_exact_runtime_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=_models_payload())

    probe = MlxMossSynthesizer(_config(), client=_client(handler)).probe()

    assert probe["identity"] == "mlx"
    assert probe["model_revision"] == "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"


@pytest.mark.parametrize("field", MLX_RUNTIME_IDENTITY_FIELDS)
def test_probe_rejects_any_runtime_identity_drift(field: str) -> None:
    drifted = "c" * 64 if field == "weights_sha256" else "drifted"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=_models_payload(**{field: drifted}))

    with pytest.raises(MossBackendUnavailable, match=f"MLX MOSS runtime {field}"):
        MlxMossSynthesizer(_config(), client=_client(handler)).probe()


def test_probe_rejects_service_without_runtime_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        payload = _models_payload()
        del payload["data"][0]["streammuse_runtime"]
        return httpx.Response(200, json=payload)

    with pytest.raises(MossBackendUnavailable, match="does not report its runtime identity"):
        MlxMossSynthesizer(_config(), client=_client(handler)).probe()


def test_config_rejects_unpinned_or_malformed_identity() -> None:
    with pytest.raises(ValueError, match="missing"):
        _config(runtime_identity={k: v for k, v in _IDENTITY.items() if k != "quantization"})
    with pytest.raises(ValueError, match="immutable pin"):
        _config(runtime_identity={**_IDENTITY, "mlx_audio_commit": "main"})
    with pytest.raises(ValueError, match="SHA-256"):
        _config(runtime_identity={**_IDENTITY, "weights_sha256": "not-a-hash"})
    with pytest.raises(ValueError):
        MlxMossConfig(
            base_url="http://10.0.0.5:8030",
            model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
            model_revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
            reference_audio_uri="file:///voices/reference.wav",
            reference_audio_sha256="b" * 64,
            runtime_identity=_IDENTITY,
        )


def test_expired_deadline_stops_before_any_request(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes())

    execution = SynthesisExecutionContext.from_timeout(0.001, correlation_id="late")
    time.sleep(0.01)
    with pytest.raises(ExecutionDeadlineExceeded):
        MlxMossSynthesizer(_config(), client=_client(handler)).synthesize(
            _request(), tmp_path / "source.wav", execution=execution
        )
    assert calls == []


def test_cancellation_before_request_leaves_no_output(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=_wav_bytes())

    execution = _execution()
    execution.cancel()
    output = tmp_path / "source.wav"
    with pytest.raises(ExecutionCancelled):
        MlxMossSynthesizer(_config(), client=_client(handler)).synthesize(
            _request(), output, execution=execution
        )
    assert not output.exists()

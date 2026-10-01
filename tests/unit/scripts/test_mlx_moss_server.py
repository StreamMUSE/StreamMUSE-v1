from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.mlx_moss_server import (
    SpeechRequestError,
    encode_pcm16_wav,
    parse_speech_request,
    weights_sha256,
)

MODEL = "OpenMOSS-Team/MOSS-TTS-v1.5"


def _payload(reference: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": MODEL,
        "input": "steady motion",
        "voice": "default",
        "response_format": "wav",
        "stream": False,
        "ref_audio": reference.as_uri(),
        "language": "English",
        "instructions": "clear, rhythmically spoken rap with restrained pitch",
        "token_count": 67,
        "max_new_tokens": 256,
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
        "seed": 20262816,
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def media(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "voices"
    root.mkdir()
    reference = root / "reference.wav"
    reference.write_bytes(encode_pcm16_wav(np.zeros(240, dtype=np.float32) + 0.1, sample_rate_hz=16_000))
    return root, reference


def test_parses_the_render_server_request_shape(media: tuple[Path, Path]) -> None:
    root, reference = media

    request = parse_speech_request(_payload(reference), model_id=MODEL, allowed_media_root=root)

    assert request.text == "steady motion"
    assert request.reference_path == reference.resolve()
    assert request.token_count == 67
    assert request.max_new_tokens == 256
    assert request.seed == 20262816
    assert (request.audio_temperature, request.audio_top_p, request.audio_top_k) == (1.7, 0.8, 25)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"model": "other"}, "not served"),
        ({"stream": True}, "streaming"),
        ({"response_format": "mp3"}, "wav"),
        ({"token_count": 0}, "token_count"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"audio_top_p": 1.5}, "audio_top_p"),
        ({"input": "   "}, "input"),
        ({"unexpected": 1}, "unsupported request fields"),
        ({"ref_audio": "https://example.test/voice.wav"}, "file://"),
    ),
)
def test_rejects_out_of_contract_requests(
    media: tuple[Path, Path], overrides: dict[str, object], message: str
) -> None:
    root, reference = media
    with pytest.raises(SpeechRequestError, match=message):
        parse_speech_request(_payload(reference, **overrides), model_id=MODEL, allowed_media_root=root)


def test_rejects_reference_outside_allowed_media_root(tmp_path: Path, media: tuple[Path, Path]) -> None:
    root, _ = media
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    escape = root / ".." / "outside.wav"

    for path in (outside, escape):
        with pytest.raises(SpeechRequestError, match="allowed media root"):
            parse_speech_request(
                _payload(root, ref_audio=f"file://{path}"), model_id=MODEL, allowed_media_root=root
            )


def test_encodes_mono_pcm16_and_rejects_non_finite_audio() -> None:
    body = encode_pcm16_wav(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32), sample_rate_hz=24_000)

    with wave.open(io.BytesIO(body)) as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24_000)
        frames = np.frombuffer(handle.readframes(4), dtype="<i2")
    assert frames.tolist() == [0, 16384, -32767, 32767]

    with pytest.raises(ValueError):
        encode_pcm16_wav(np.array([np.nan], dtype=np.float32), sample_rate_hz=24_000)


def test_weights_hash_covers_every_shard_name_and_byte(tmp_path: Path) -> None:
    model = tmp_path / "model"
    (model / "audio_tokenizer").mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"backbone")
    (model / "audio_tokenizer" / "model.safetensors").write_bytes(b"codec")
    first = weights_sha256(model)

    (model / "audio_tokenizer" / "model.safetensors").write_bytes(b"codec2")
    assert weights_sha256(model) != first

    with pytest.raises(ValueError):
        weights_sha256(tmp_path / "empty")

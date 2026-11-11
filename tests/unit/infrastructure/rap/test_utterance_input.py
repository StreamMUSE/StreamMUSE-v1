"""Bounded browser utterances and an optional, persistent CPU recognizer."""

from __future__ import annotations

import io
import struct
import wave
from types import SimpleNamespace

import pytest

from streammuse.infrastructure.rap.utterance_input import (
    MAX_WAV_BYTES,
    FasterWhisperUtteranceTranscriber,
    InvalidUtterance,
    TranscriberUnavailable,
    validate_wav,
)


def wav_bytes(*, seconds=0.1, rate=16000, channels=1, width=2, value=2000):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        sample = struct.pack("<h", value) if width == 2 else bytes([127]) * width
        writer.writeframes(sample * int(seconds * rate) * channels)
    return output.getvalue()


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_validates_browser_pcm_wav_without_creating_files(rate):
    result = validate_wav(wav_bytes(seconds=0.5, rate=rate))
    assert result.duration_seconds == 0.5
    assert result.sample_rate == rate
    assert not result.silence


@pytest.mark.parametrize(
    "payload",
    [
        b"", b"/tmp/speech.wav", b"RIFFbad", wav_bytes()[:-2],
        wav_bytes(channels=2), wav_bytes(width=1), wav_bytes(rate=96000),
        wav_bytes(rate=4000), wav_bytes(seconds=0),
        wav_bytes() + b"unexpected trailing data",
    ],
)
def test_rejects_malformed_non_mono_and_unsupported_audio(payload):
    with pytest.raises(InvalidUtterance):
        validate_wav(payload)


def test_rejects_declared_frame_count_larger_than_available_data():
    payload = bytearray(wav_bytes())
    struct.pack_into("<I", payload, 40, 320000)
    with pytest.raises(InvalidUtterance):
        validate_wav(bytes(payload))


def test_duration_limit_is_inclusive_and_not_only_a_byte_limit():
    assert validate_wav(wav_bytes(seconds=15, rate=48000)).duration_seconds == 15
    with pytest.raises(InvalidUtterance) as error:
        validate_wav(wav_bytes(seconds=15.01, rate=8000))
    assert error.value.status_code == 413
    with pytest.raises(InvalidUtterance) as error:
        validate_wav(b"x" * (MAX_WAV_BYTES + 1))
    assert error.value.status_code == 413


def test_silence_does_not_load_model_or_hallucinate_a_transcript():
    def forbidden_factory(**kwargs):
        pytest.fail("Silence must not initialize Whisper")

    transcriber = FasterWhisperUtteranceTranscriber(whisper_model_factory=forbidden_factory)
    result = transcriber.transcribe(wav_bytes(value=0))
    assert result.text == ""
    assert result.silence
    assert result.duration_seconds == 0.1
    assert transcriber.status()["state"] == "unloaded"


def test_model_is_lazy_persistent_cpu_int8_english_and_reads_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    instances = []
    streams = []

    class Model:
        def transcribe(self, audio, **kwargs):
            assert isinstance(audio, io.BytesIO)
            assert audio.read(4) == b"RIFF"
            assert kwargs["language"] == "en"
            assert kwargs["vad_filter"] is True
            streams.append(audio)
            return iter([SimpleNamespace(text=" hello "), SimpleNamespace(text=" world ")]), None

    def factory(**kwargs):
        assert kwargs["model_size_or_path"] == "base.en"
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"
        assert kwargs["local_files_only"] is True
        assert kwargs["num_workers"] == 1
        instances.append(Model())
        return instances[-1]

    transcriber = FasterWhisperUtteranceTranscriber(model="base", whisper_model_factory=factory)
    assert not instances
    assert transcriber.status()["available"]
    assert transcriber.transcribe(wav_bytes()).text == "hello world"
    assert transcriber.transcribe(wav_bytes()).text == "hello world"
    assert len(instances) == 1
    assert all(stream.closed for stream in streams)
    assert list(tmp_path.iterdir()) == []
    assert transcriber.status()["state"] == "ready"


def test_vad_empty_result_is_silence():
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([]), None))
    transcriber = FasterWhisperUtteranceTranscriber(whisper_model_factory=lambda **kwargs: model)
    result = transcriber.transcribe(wav_bytes())
    assert result.text == ""
    assert result.silence


def test_disabled_and_missing_dependency_are_graceful(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    transcriber = FasterWhisperUtteranceTranscriber()
    assert transcriber.status()["state"] == "unavailable"
    assert not transcriber.status()["available"]
    with pytest.raises(TranscriberUnavailable):
        transcriber.transcribe(wav_bytes())
    disabled = FasterWhisperUtteranceTranscriber(enabled=False)
    assert disabled.status()["state"] == "disabled"
    with pytest.raises(TranscriberUnavailable):
        disabled.transcribe(wav_bytes())


def test_model_load_failure_is_cached_and_does_not_leak_local_path():
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        raise OSError("/private/model/cache/secret is missing")

    transcriber = FasterWhisperUtteranceTranscriber(whisper_model_factory=factory)
    for _ in range(2):
        with pytest.raises(TranscriberUnavailable):
            transcriber.transcribe(wav_bytes())
    assert len(calls) == 1
    assert transcriber.status()["state"] == "unavailable"
    assert "/private/" not in str(transcriber.status())


def test_decoder_failure_closes_stream_and_leaves_no_tempfiles(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    streams = []

    def fail(audio, **kwargs):
        streams.append(audio)
        raise RuntimeError("decode failed")

    transcriber = FasterWhisperUtteranceTranscriber(
        whisper_model_factory=lambda **kwargs: SimpleNamespace(transcribe=fail)
    )
    with pytest.raises(RuntimeError):
        transcriber.transcribe(wav_bytes())
    assert streams[0].closed
    assert list(tmp_path.iterdir()) == []


def test_only_small_or_base_english_models_are_allowed():
    with pytest.raises(ValueError):
        FasterWhisperUtteranceTranscriber(model="/user/provided/path")


def test_environment_can_select_small_or_disable_without_loading(monkeypatch):
    monkeypatch.setenv("STREAMMUSE_ASR_MODEL", "small.en")
    monkeypatch.setenv("STREAMMUSE_MICROPHONE_ENABLED", "false")
    transcriber = FasterWhisperUtteranceTranscriber()
    assert transcriber.status()["model"] == "small.en"
    assert transcriber.status()["state"] == "disabled"


def test_malformed_chunk_alignment_and_duplicate_data_are_rejected():
    payload = bytearray(wav_bytes())
    duplicate = b"data" + struct.pack("<I", 2) + b"\x00\x00"
    payload.extend(duplicate)
    struct.pack_into("<I", payload, 4, len(payload) - 8)
    with pytest.raises(InvalidUtterance):
        validate_wav(bytes(payload))

    payload = bytearray(wav_bytes())
    struct.pack_into("<H", payload, 32, 4)
    with pytest.raises(InvalidUtterance):
        validate_wav(bytes(payload))

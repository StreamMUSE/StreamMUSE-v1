from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from streammuse.application.tasks import SpeechOutputConfig
from streammuse.domain.tasks import SpeechRequest
from streammuse.infrastructure.voice import SpeechSynthesisError
from streammuse.infrastructure.voice.speaker import SpeakerPlayback
from streammuse.infrastructure.voice.speech_sink import AudioSpeechOutput
from streammuse.infrastructure.voice.synthesizer import SynthesizedAudio


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    @property
    def provenance(self):
        return {"backend": "fake", "rate": 1.0}

    def synthesize(self, text: str) -> SynthesizedAudio:
        self.calls.append(text)
        return SynthesizedAudio(
            np.ones(max(1, len(text)), dtype=np.float32),
            16_000,
        )

    def close(self) -> None:
        self.closed += 1


class FailingSynthesizer(FakeSynthesizer):
    def synthesize(self, text: str) -> SynthesizedAudio:
        raise SpeechSynthesisError(f"cannot synthesize {text}")


class FakeSpeaker:
    def __init__(self) -> None:
        self.calls = 0
        self.drains = 0
        self.closed = 0

    def start(self) -> None:
        return None

    @property
    def provenance(self):
        return {"device": {"name": "fake"}}

    def play(self, audio: SynthesizedAudio) -> SpeakerPlayback:
        self.calls += 1
        return SpeakerPlayback(
            completed_normally=True,
            playback_start_offset_ms=1.0,
            first_dac_sample_offset_ms=2.0,
            playback_drained_offset_ms=3.0,
            stream_inactive_offset_ms=3.0,
            sample_rate_hz=audio.sample_rate_hz,
            device="fake",
        )

    def drain(self) -> None:
        self.drains += 1

    def abort_active(self) -> None:
        return None

    def close(self) -> None:
        self.closed += 1


def _request(text: str, turn_id: int = 0) -> SpeechRequest:
    return SpeechRequest(
        turn_id=turn_id,
        actor="llm",
        text=text,
        source_text=text,
    )


def test_prepare_respects_entry_and_byte_limits() -> None:
    synth = FakeSynthesizer()
    sink = AudioSpeechOutput(
        SpeechOutputConfig(
            mode="audio",
            backend="null",
            cache_max_entries=2,
            cache_max_bytes=12,
        ),
        synthesizer=synth,
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
    )
    sink.start()

    sink.prepare(("a", "bb", "ccc"))

    assert sink.provenance["prewarm_entry_count"] == 2
    assert sink.provenance["prewarm_truncated"] is True
    assert synth.calls == ["a", "bb"]


def test_cache_hit_has_zero_synthesis_latency() -> None:
    synth = FakeSynthesizer()
    speaker = FakeSpeaker()
    sink = AudioSpeechOutput(
        SpeechOutputConfig(mode="audio", backend="null"),
        synthesizer=synth,
        speaker=speaker,  # type: ignore[arg-type]
    )
    sink.start()
    sink.prepare(("Zip",))

    playback = sink.speak(_request("Zip"))

    assert synth.calls == ["Zip"]
    assert playback.status == "ok"
    assert playback.cached is True
    assert playback.synthesis_ms == 0.0
    assert playback.completed_normally is True


def test_cache_miss_skip_does_not_touch_synthesizer_or_speaker() -> None:
    synth = FakeSynthesizer()
    speaker = FakeSpeaker()
    sink = AudioSpeechOutput(
        SpeechOutputConfig(
            mode="audio",
            backend="null",
            prewarm=False,
            cache_miss="skip",
        ),
        synthesizer=synth,
        speaker=speaker,  # type: ignore[arg-type]
    )
    sink.start()

    playback = sink.speak(_request("Zip"))

    assert playback.status == "cache_miss_skipped"
    assert synth.calls == []
    assert speaker.calls == 0


def test_save_audio_writes_atomic_turn_artifact(tmp_path: Path) -> None:
    sink = AudioSpeechOutput(
        SpeechOutputConfig(
            mode="audio",
            backend="null",
            prewarm=False,
            save_audio=True,
        ),
        synthesizer=FakeSynthesizer(),
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
        artifact_root=tmp_path,
    )
    sink.start()

    playback = sink.speak(_request("Zip", turn_id=2))
    path = tmp_path / "0003_turn_0002_llm.wav"

    assert playback.audio_artifact == (
        "artifacts/turn/0003_turn_0002_llm.wav"
    )
    assert playback.artifact_persistence_ms >= 0.0
    assert not list(tmp_path.glob("*.tmp"))
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1


def test_expected_synthesis_failure_returns_structured_status() -> None:
    sink = AudioSpeechOutput(
        SpeechOutputConfig(
            mode="audio",
            backend="null",
            prewarm=False,
        ),
        synthesizer=FailingSynthesizer(),
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
    )
    sink.start()

    playback = sink.speak(_request("Zip"))

    assert playback.status == "synthesis_failed"
    assert playback.completed_normally is False
    assert playback.error == {
        "type": "SpeechSynthesisError",
        "message": "cannot synthesize Zip",
    }


def test_artifact_failure_preserves_completed_playback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sink = AudioSpeechOutput(
        SpeechOutputConfig(
            mode="audio",
            backend="null",
            prewarm=False,
            save_audio=True,
        ),
        synthesizer=FakeSynthesizer(),
        speaker=FakeSpeaker(),  # type: ignore[arg-type]
        artifact_root=tmp_path,
    )
    sink.start()

    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "streammuse.infrastructure.voice.speech_sink.wave.open",
        fail_open,
    )
    playback = sink.speak(_request("Zip"))

    assert playback.status == "artifact_failed"
    assert playback.completed_normally is True
    assert playback.playback_drained_offset_ms is not None
    assert playback.audio_artifact is None
    assert playback.error is not None
    assert not list(tmp_path.glob("*.tmp"))

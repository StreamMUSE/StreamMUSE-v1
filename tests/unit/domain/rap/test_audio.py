"""Tests for immutable realtime rap audio contracts."""

from dataclasses import FrozenInstanceError

import pytest

from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNoticeKind,
    AudioWarningCode,
    AudioWarningSeverity,
    PlaybackState,
    PcmAudio,
    RapEventType,
)


def test_pcm_audio_requires_exact_float32_frame_bytes() -> None:
    audio_format = AudioFormat(sample_rate_hz=48_000, channels=2)

    with pytest.raises(ValueError, match="frame byte length"):
        PcmAudio(format=audio_format, frame_count=2, data=b"short")


def test_pcm_audio_accepts_exact_stereo_float32_data() -> None:
    audio_format = AudioFormat(sample_rate_hz=48_000, channels=2)
    audio = PcmAudio(format=audio_format, frame_count=2, data=bytes(2 * 2 * 4))

    assert audio.duration_seconds == pytest.approx(2 / 48_000)


def test_rap_audio_event_names_are_canonical() -> None:
    assert [event.value for event in RapEventType][-19:] == [
        "audio_render_started",
        "audio_render_completed",
        "pronunciation_fallback",
        "timing_pressure",
        "forced_bar_fit",
        "synthesis_failed",
        "bar_audio_ready",
        "bar_audio_committed",
        "chunk_request_submitted",
        "chunk_remote_completed",
        "chunk_remote_rejected",
        "chunk_committed",
        "chunk_fallback_activated",
        "bar_playback_started",
        "bar_playback_completed",
        "stop_requested",
        "session_reset",
        "audio_underrun",
        "audio_device_failed",
    ]


def test_audio_contract_enums_are_complete() -> None:
    assert {item.value for item in PlaybackState} == {
        "stopped",
        "priming",
        "running",
        "stop_requested",
        "closed",
    }
    assert {item.value for item in AudioWarningSeverity} == {"info", "warning", "error"}
    assert {item.value for item in AudioWarningCode} == {
        "pronunciation_fallback",
        "timing_pressure",
        "forced_bar_fit",
        "synthesis_failed",
        "audio_deadline_miss",
        "audio_underrun",
        "audio_device_failed",
    }
    assert {item.value for item in AudioPlaybackNoticeKind} == {
        "bar_started",
        "bar_completed",
        "stopped",
        "underrun",
        "device_failed",
    }


def test_prepared_rap_bar_is_immutable() -> None:
    from streammuse.domain.rap import PreparedRapBar

    prepared = PreparedRapBar(
        bar=0,
        text="one",
        source="generated",
        fallback_reason=None,
        scheduled=(),
        audio=PcmAudio(AudioFormat(), 0, b""),
        diagnostics=(),
        warnings=(),
        render_latency_ms=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        prepared.text = "two"  # type: ignore[misc]

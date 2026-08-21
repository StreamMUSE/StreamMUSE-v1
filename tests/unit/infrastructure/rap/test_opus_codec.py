"""Tests for the bounded ffmpeg Opus adapter."""

from __future__ import annotations

import subprocess

import pytest

from streammuse.infrastructure.rap.opus_codec import FFmpegOpusCodec, FFmpegOpusCodecError


def test_encode_uses_24khz_mono_48kbps_libopus_audio_vbr(monkeypatch) -> None:
    calls: list[tuple[list[str], bytes]] = []

    def run(command, *, input, capture_output, check, timeout):
        calls.append((command, input))
        return subprocess.CompletedProcess(command, 0, stdout=b"opus-bytes", stderr=b"")

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.subprocess.run", run)
    codec = FFmpegOpusCodec(timeout_seconds=2.0)

    encoded = codec.encode_pcm16_mono_24khz(b"\x01\x00" * 12, expected_frame_count=12)

    assert encoded == b"opus-bytes"
    assert calls == [
        (
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                "-i", "pipe:0", "-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-application", "audio",
                "-compression_level", "10", "-f", "opus", "pipe:1",
            ],
            b"\x01\x00" * 12,
        )
    ]


def test_decode_rejects_non_exact_manifest_frame_count(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command, *, input, capture_output, check, timeout):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"\x01\x00" * 11, stderr=b"")

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.subprocess.run", run)
    codec = FFmpegOpusCodec(timeout_seconds=2.0)

    with pytest.raises(FFmpegOpusCodecError, match="frame count"):
        codec.decode_to_pcm16_mono_24khz(b"opus", expected_frame_count=12)

    assert commands[0][commands[0].index("-f") + 1] == "ogg"
    assert "-t" in commands[0]


def test_real_ffmpeg_round_trip_preserves_exact_128000_frame_count() -> None:
    frame_count = 128_000
    pcm = b"\x01\x00" * frame_count
    codec = FFmpegOpusCodec(executable="/opt/homebrew/bin/ffmpeg", timeout_seconds=15.0)

    encoded = codec.encode_pcm16_mono_24khz(pcm, expected_frame_count=frame_count)
    decoded = codec.decode_to_pcm16_mono_24khz(encoded, expected_frame_count=frame_count)

    assert encoded
    assert len(decoded) == frame_count * 2


def test_missing_ffmpeg_has_a_clear_error(monkeypatch) -> None:
    def run(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.subprocess.run", run)

    with pytest.raises(FFmpegOpusCodecError, match="ffmpeg is required"):
        FFmpegOpusCodec().encode_pcm16_mono_24khz(b"\x01\x00", expected_frame_count=1)


def test_decode_rejects_manifest_frame_count_above_bounded_output_limit() -> None:
    codec = FFmpegOpusCodec(max_decoded_bytes=4)

    with pytest.raises(FFmpegOpusCodecError, match="decoded output exceeds"):
        codec.decode_to_pcm16_mono_24khz(b"opus", expected_frame_count=3)

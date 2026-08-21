"""Tests for the bounded ffmpeg Opus adapter."""

from __future__ import annotations

import shutil
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
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is unavailable")
    frame_count = 128_000
    pcm = b"\x01\x00" * frame_count
    codec = FFmpegOpusCodec(executable=executable, timeout_seconds=15.0)

    encoded = codec.encode_pcm16_mono_24khz(pcm, expected_frame_count=frame_count)
    decoded = codec.decode_to_pcm16_mono_24khz(encoded, expected_frame_count=frame_count)

    assert encoded
    assert len(decoded) == frame_count * 2


def test_real_ffmpeg_decode_rejects_non_opus_ogg() -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is unavailable")
    frame_count = 2_400
    encoded = subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "flac",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=b"\x01\x00" * frame_count,
        capture_output=True,
        check=True,
    ).stdout

    with pytest.raises(FFmpegOpusCodecError, match="failed"):
        FFmpegOpusCodec(executable=executable).decode_to_pcm16_mono_24khz(
            encoded, expected_frame_count=frame_count
        )


def test_decode_terminates_ffmpeg_when_cancelled(monkeypatch) -> None:
    cancelled = False

    class Process:
        returncode = None

        def __init__(self) -> None:
            self.calls = 0
            self.terminated = False
            self.killed = False

        def communicate(self, input=None, timeout=None):
            nonlocal cancelled
            self.calls += 1
            if self.calls == 1:
                cancelled = True
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = Process()
    monkeypatch.setattr(
        "streammuse.infrastructure.rap.opus_codec.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(FFmpegOpusCodecError, match="cancelled"):
        FFmpegOpusCodec().decode_to_pcm16_mono_24khz(
            b"opus",
            expected_frame_count=1,
            timeout_seconds=1.0,
            cancelled=lambda: cancelled,
        )

    assert process.terminated
    assert not process.killed


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

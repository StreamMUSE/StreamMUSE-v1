"""Bounded stdin/stdout FFmpeg adapter for 24 kHz mono Opus transport."""

from __future__ import annotations

import subprocess
from time import monotonic
from typing import Callable


_SAMPLE_WIDTH_BYTES = 2
_MAX_ENCODED_BYTES = 4 * 1024 * 1024


class FFmpegOpusCodecError(RuntimeError):
    """FFmpeg cannot safely encode or decode the requested Opus payload."""


class FFmpegOpusCodec:
    """Use FFmpeg pipes only; callers retain all package and duration checks."""

    def __init__(
        self,
        *,
        executable: str = "ffmpeg",
        timeout_seconds: float = 10.0,
        max_encoded_bytes: int = _MAX_ENCODED_BYTES,
        max_decoded_bytes: int = _MAX_ENCODED_BYTES,
    ) -> None:
        if not executable:
            raise ValueError("ffmpeg executable must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("ffmpeg timeout_seconds must be positive")
        if max_encoded_bytes <= 0:
            raise ValueError("max_encoded_bytes must be positive")
        if max_decoded_bytes <= 0:
            raise ValueError("max_decoded_bytes must be positive")
        self._executable = executable
        self._timeout_seconds = float(timeout_seconds)
        self._max_encoded_bytes = max_encoded_bytes
        self._max_decoded_bytes = max_decoded_bytes
        self.encoder_identity = "ffmpeg/libopus"

    def probe(self) -> None:
        """Fail at server startup when FFmpeg or its libopus encoder is unavailable."""
        completed = self._run([self._executable, "-hide_banner", "-encoders"], b"")
        if b"libopus" not in completed.stdout:
            raise FFmpegOpusCodecError("ffmpeg does not provide the required libopus encoder")
        version = self._run([self._executable, "-version"], b"").stdout.decode("utf-8", "replace").splitlines()
        self.encoder_identity = ((version[0] if version else "ffmpeg") + " / libopus")[:128]

    def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
        self._validate_pcm(pcm, expected_frame_count)
        result = self._run(
            [
                self._executable, "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1",
                "-i", "pipe:0", "-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-application", "audio",
                "-compression_level", "10", "-f", "opus", "pipe:1",
            ],
            pcm,
        ).stdout
        if not result:
            raise FFmpegOpusCodecError("ffmpeg produced an empty Opus stream")
        if len(result) > self._max_encoded_bytes:
            raise FFmpegOpusCodecError("ffmpeg Opus output exceeds configured limit")
        return result

    def decode_to_pcm16_mono_24khz(
        self,
        encoded: bytes,
        *,
        expected_frame_count: int,
        timeout_seconds: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bytes:
        if not encoded or len(encoded) > self._max_encoded_bytes:
            raise FFmpegOpusCodecError("Opus input exceeds configured limit or is empty")
        if type(expected_frame_count) is not int or expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be a positive integer")
        expected_bytes = expected_frame_count * _SAMPLE_WIDTH_BYTES
        if expected_bytes > self._max_decoded_bytes:
            raise FFmpegOpusCodecError("manifest decoded output exceeds configured limit")
        command = [
            self._executable, "-hide_banner", "-loglevel", "error", "-c:a", "libopus", "-f", "ogg",
            "-i", "pipe:0", "-ar", "24000", "-ac", "1", "-t", f"{expected_frame_count / 24_000:.9f}",
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ]
        if cancelled is None:
            output = self._run(command, encoded, timeout_seconds=timeout_seconds).stdout
        else:
            output = self._run_cancellable(
                command,
                encoded,
                timeout_seconds=timeout_seconds,
                cancelled=cancelled,
            ).stdout
        if len(output) != expected_bytes:
            raise FFmpegOpusCodecError("ffmpeg decoded Opus frame count does not match the manifest")
        return output

    @staticmethod
    def _validate_pcm(pcm: bytes, expected_frame_count: int) -> None:
        if not isinstance(pcm, bytes):
            raise ValueError("PCM input must be bytes")
        if type(expected_frame_count) is not int or expected_frame_count <= 0:
            raise ValueError("expected_frame_count must be a positive integer")
        if len(pcm) != expected_frame_count * _SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM input frame count does not match the manifest")

    def _run(
        self,
        command: list[str],
        data: bytes,
        *,
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        timeout = self._effective_timeout(timeout_seconds)
        try:
            completed = subprocess.run(
                command,
                input=data,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as error:
            raise FFmpegOpusCodecError("ffmpeg is required for Opus transport but was not found on PATH") from error
        except subprocess.TimeoutExpired as error:
            raise FFmpegOpusCodecError("ffmpeg Opus operation exceeded its timeout") from error
        except OSError as error:
            raise FFmpegOpusCodecError("ffmpeg Opus operation could not be started") from error
        if completed.returncode != 0:
            self._raise_process_failure(completed.stderr)
        return completed

    def _run_cancellable(
        self,
        command: list[str],
        data: bytes,
        *,
        timeout_seconds: float | None,
        cancelled: Callable[[], bool],
    ) -> subprocess.CompletedProcess[bytes]:
        timeout = self._effective_timeout(timeout_seconds)
        deadline = monotonic() + timeout
        if cancelled():
            raise FFmpegOpusCodecError("ffmpeg Opus operation was cancelled")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise FFmpegOpusCodecError(
                "ffmpeg is required for Opus transport but was not found on PATH"
            ) from error
        except OSError as error:
            raise FFmpegOpusCodecError("ffmpeg Opus operation could not be started") from error

        first_communicate = True
        while True:
            if cancelled():
                self._stop_process(process)
                raise FFmpegOpusCodecError("ffmpeg Opus operation was cancelled")
            remaining = deadline - monotonic()
            if remaining <= 0:
                self._stop_process(process)
                raise FFmpegOpusCodecError("ffmpeg Opus operation exceeded its timeout")
            try:
                stdout, stderr = process.communicate(
                    input=data if first_communicate else None,
                    timeout=min(remaining, 0.05),
                )
                break
            except subprocess.TimeoutExpired:
                first_communicate = False

        if process.returncode != 0:
            self._raise_process_failure(stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def _effective_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self._timeout_seconds
        if timeout_seconds <= 0:
            raise FFmpegOpusCodecError("ffmpeg Opus operation exceeded its timeout")
        return min(self._timeout_seconds, float(timeout_seconds))

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    @staticmethod
    def _raise_process_failure(stderr: bytes) -> None:
        detail = stderr.decode("utf-8", "replace").strip().splitlines()
        suffix = f": {detail[-1][:200]}" if detail else ""
        raise FFmpegOpusCodecError(f"ffmpeg Opus operation failed{suffix}")

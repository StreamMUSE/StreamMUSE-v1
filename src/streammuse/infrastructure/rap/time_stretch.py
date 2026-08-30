"""Pitch-preserving duration fitting for locally rendered rap vocals."""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from math import isfinite
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from streammuse.domain.rap import PcmAudio


_MAX_ENDPOINT_CORRECTION_FRAMES = 2
_MAX_ATTEMPTS = 2


class RubberBandTimeStretcher:
    """Apply an R3 time stretch while leaving pitch scale unchanged."""

    def __init__(
        self,
        *,
        binary: str = "rubberband",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not binary:
            raise ValueError("rubberband binary must not be empty")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("rubberband timeout must be positive and finite")
        self._binary = binary
        self._timeout_seconds = timeout_seconds

    def stretch(self, audio: PcmAudio, target_frames: int) -> PcmAudio:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        if audio.frame_count <= 0:
            raise ValueError("source audio must not be empty")
        if audio.format.sample_width_bytes != 4:
            raise ValueError("rubberband input must be float32 PCM")

        source = np.frombuffer(audio.data, dtype=np.float32).reshape(
            audio.frame_count,
            audio.format.channels,
        )
        requested_seconds = target_frames / audio.format.sample_rate_hz
        with tempfile.TemporaryDirectory(prefix="streammuse-rap-stretch-") as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            wavfile.write(input_path, audio.format.sample_rate_hz, source)

            for attempt in range(_MAX_ATTEMPTS):
                output_path.unlink(missing_ok=True)
                self._run_rubberband(
                    input_path=input_path,
                    output_path=output_path,
                    requested_seconds=requested_seconds,
                )
                stretched = _load_output(
                    output_path,
                    sample_rate_hz=audio.format.sample_rate_hz,
                    channels=audio.format.channels,
                )
                frame_delta = target_frames - stretched.shape[0]
                if abs(frame_delta) <= _MAX_ENDPOINT_CORRECTION_FRAMES:
                    corrected = _correct_tail(stretched, target_frames)
                    return PcmAudio(audio.format, target_frames, corrected.tobytes())
                if attempt + 1 < _MAX_ATTEMPTS:
                    if stretched.shape[0] > 0:
                        requested_seconds *= target_frames / stretched.shape[0]
                    continue
                raise RuntimeError(
                    "rubberband did not produce the requested duration: "
                    f"expected {target_frames} frames, got {stretched.shape[0]}"
                )

        raise RuntimeError("rubberband duration fitting exhausted unexpectedly")

    def _run_rubberband(
        self,
        *,
        input_path: Path,
        output_path: Path,
        requested_seconds: float,
    ) -> None:
        try:
            subprocess.run(
                [
                    self._binary,
                    "--quiet",
                    "--fine",
                    "--duration",
                    f"{requested_seconds:.12f}",
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("rubberband executable is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("rubberband time stretch timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else "unknown error"
            raise RuntimeError(f"rubberband time stretch failed: {detail}") from exc


class RubberBandTimeMapStretcher:
    """Apply one pitch-preserving R3 warp through a sparse sample time map."""

    def __init__(
        self,
        *,
        binary: str = "rubberband",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not binary:
            raise ValueError("rubberband binary must not be empty")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("rubberband timeout must be positive and finite")
        self._binary = binary
        self._timeout_seconds = timeout_seconds

    def stretch(
        self,
        audio: PcmAudio,
        target_frames: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> PcmAudio:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        if audio.frame_count <= 0:
            raise ValueError("source audio must not be empty")
        if audio.format.sample_width_bytes != 4:
            raise ValueError("rubberband input must be float32 PCM")
        _validate_time_map(
            time_map,
            source_frames=audio.frame_count,
            target_frames=target_frames,
        )
        source = np.frombuffer(audio.data, dtype=np.float32).reshape(
            audio.frame_count,
            audio.format.channels,
        )
        wav_source = source[:, 0] if audio.format.channels == 1 else source
        active_map = time_map
        with tempfile.TemporaryDirectory(prefix="streammuse-rap-timemap-") as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            map_path = Path(temp_dir) / "time_map.txt"
            wavfile.write(input_path, audio.format.sample_rate_hz, wav_source)

            for attempt in range(_MAX_ATTEMPTS):
                map_path.write_text(
                    "".join(f"{source_frame} {target_frame}\n" for source_frame, target_frame in active_map),
                    encoding="utf-8",
                )
                output_path.unlink(missing_ok=True)
                self._run_rubberband(
                    input_path=input_path,
                    output_path=output_path,
                    time_map_path=map_path,
                    target_frames=target_frames,
                    sample_rate_hz=audio.format.sample_rate_hz,
                )
                stretched = _load_output(
                    output_path,
                    sample_rate_hz=audio.format.sample_rate_hz,
                    channels=audio.format.channels,
                )
                frame_delta = target_frames - stretched.shape[0]
                if abs(frame_delta) <= _MAX_ENDPOINT_CORRECTION_FRAMES:
                    corrected = _correct_tail(stretched, target_frames)
                    return PcmAudio(audio.format, target_frames, corrected.tobytes())
                if attempt + 1 < _MAX_ATTEMPTS and stretched.shape[0] > 0:
                    source_endpoint, target_endpoint = active_map[-1]
                    corrected_endpoint = target_endpoint + frame_delta
                    if corrected_endpoint > active_map[-2][1]:
                        active_map = (*active_map[:-1], (source_endpoint, corrected_endpoint))
                        continue
                raise RuntimeError(
                    "rubberband time-map warp did not produce the requested duration: "
                    f"expected {target_frames} frames, got {stretched.shape[0]}"
                )

        raise RuntimeError("rubberband time-map duration fitting exhausted unexpectedly")

    def _run_rubberband(
        self,
        *,
        input_path: Path,
        output_path: Path,
        time_map_path: Path,
        target_frames: int,
        sample_rate_hz: int,
    ) -> None:
        try:
            subprocess.run(
                [
                    self._binary,
                    "--quiet",
                    "--duration",
                    f"{target_frames / sample_rate_hz:.12f}",
                    "--timemap",
                    str(time_map_path),
                    "--fine",
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("rubberband executable is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("rubberband time-map warp timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else "unknown error"
            raise RuntimeError(f"rubberband time-map warp failed: {detail}") from exc

def _load_output(path: Path, *, sample_rate_hz: int, channels: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Chunk \(non-data\) not understood, skipping it\.",
            category=WavFileWarning,
        )
        rendered_rate, rendered = wavfile.read(path)
    if rendered_rate != sample_rate_hz:
        raise RuntimeError(
            f"rubberband changed the sample rate from {sample_rate_hz} to {rendered_rate} Hz"
        )
    samples = _to_float32(rendered)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[1] != channels:
        raise RuntimeError(
            f"rubberband changed the channel count from {channels} to "
            f"{samples.shape[1] if samples.ndim == 2 else 'invalid'}"
        )
    return samples.astype(np.float32, copy=False)


def _to_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.dtype.kind in {"i", "u"}:
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        return array.astype(np.float32) / np.float32(scale)
    return array.astype(np.float32, copy=False)


def _correct_tail(samples: np.ndarray, target_frames: int) -> np.ndarray:
    if samples.shape[0] >= target_frames:
        return np.array(samples[:target_frames], dtype=np.float32, copy=True)
    padding = np.zeros(
        (target_frames - samples.shape[0], samples.shape[1]),
        dtype=np.float32,
    )
    return np.concatenate((samples, padding), axis=0)


def _validate_time_map(
    time_map: tuple[tuple[int, int], ...],
    *,
    source_frames: int,
    target_frames: int,
) -> None:
    if len(time_map) < 2 or time_map[0] != (0, 0):
        raise ValueError("time map must begin at source and target frame zero")
    if time_map[-1] != (source_frames - 1, target_frames - 1):
        raise ValueError("time map must end at the source and target endpoints")
    if any(
        right_source <= left_source or right_target <= left_target
        for (left_source, left_target), (right_source, right_target) in zip(
            time_map,
            time_map[1:],
        )
    ):
        raise ValueError("time map frames must be strictly increasing")

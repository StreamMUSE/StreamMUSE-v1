"""Offline audio assembly helpers for rap protocol comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from streammuse.domain.rap import AudioFormat, PcmAudio
from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.contracts import TwoBarRenderRequest
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


TARGET_SAMPLE_RATE_HZ = 48_000
CHUNK_FRAME_COUNT = 256_000
SONG_FRAME_COUNT = 6_400_000
TARGET_VOCAL_FORMAT = AudioFormat(sample_rate_hz=TARGET_SAMPLE_RATE_HZ, channels=1, sample_width_bytes=4)
TARGET_STEREO_FORMAT = AudioFormat(sample_rate_hz=TARGET_SAMPLE_RATE_HZ, channels=2, sample_width_bytes=4)
_TEMPO = Tempo(90.0, 4, 4)
_COMMON_DRUM_TEMPLATE = BUILTIN_TEMPLATES.get("baseline_straight_9")


@dataclass(frozen=True)
class WavMetadata:
    sample_rate_hz: int
    channels: int
    frame_count: int
    dtype: str


@dataclass(frozen=True)
class ChunkAssemblyDiagnostic:
    chunk_index: int
    source_frames: int
    output_frames: int
    action: str
    message: str
    adjusted_frames: int = 0


@dataclass(frozen=True)
class VocalStemAssembly:
    audio: PcmAudio
    diagnostics: tuple[ChunkAssemblyDiagnostic, ...]


@dataclass(frozen=True)
class MixResult:
    audio: PcmAudio
    peak_before_limiter: float
    applied_gain: float


def validate_wav_metadata(path: Path | str) -> WavMetadata:
    sample_rate_hz, samples = wavfile.read(Path(path))
    array = np.asarray(samples)
    if sample_rate_hz <= 0:
        raise ValueError("WAV sample rate must be positive")
    channels = 1 if array.ndim == 1 else int(array.shape[1])
    frame_count = int(array.shape[0])
    if frame_count <= 0:
        raise ValueError("WAV must contain at least one frame")
    return WavMetadata(sample_rate_hz=sample_rate_hz, channels=channels, frame_count=frame_count, dtype=str(array.dtype))


def load_wav_mono_float32(path: Path | str) -> PcmAudio:
    sample_rate_hz, samples = wavfile.read(Path(path))
    if sample_rate_hz <= 0:
        raise ValueError("WAV sample rate must be positive")
    mono = _to_mono_float32(samples)
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        factor = gcd(sample_rate_hz, TARGET_SAMPLE_RATE_HZ)
        mono = resample_poly(mono, TARGET_SAMPLE_RATE_HZ // factor, sample_rate_hz // factor).astype(np.float32, copy=False)
    return _pcm_audio(TARGET_VOCAL_FORMAT, mono.reshape(-1, 1))


def assemble_vocal_stem(
    requests: Sequence[TwoBarRenderRequest],
    *,
    chunk_paths_by_index: Mapping[int, Path | str],
    listening_wav_path: Path | str | None = None,
    float32_wav_path: Path | str | None = None,
) -> VocalStemAssembly:
    if not requests:
        raise ValueError("requests must not be empty")

    chunks: list[np.ndarray] = []
    diagnostics: list[ChunkAssemblyDiagnostic] = []
    for request in requests:
        try:
            path = chunk_paths_by_index[request.chunk_index]
        except KeyError as exc:
            raise ValueError(f"missing chunk WAV for chunk {request.chunk_index}") from exc
        audio = load_wav_mono_float32(path)
        samples = _samples(audio)[:, 0]
        fitted, diagnostic = _fit_chunk(request.chunk_index, samples)
        chunks.append(fitted.reshape(-1, 1))
        diagnostics.append(diagnostic)

    stem = _pcm_audio(TARGET_VOCAL_FORMAT, np.concatenate(chunks, axis=0))
    if listening_wav_path is not None:
        write_listening_wav(listening_wav_path, stem)
    if float32_wav_path is not None:
        write_float32_wav(float32_wav_path, stem)
    return VocalStemAssembly(audio=stem, diagnostics=tuple(diagnostics))


def render_common_drums(
    requests: Sequence[TwoBarRenderRequest],
    *,
    song_index: int,
    listening_wav_path: Path | str | None = None,
    float32_wav_path: Path | str | None = None,
) -> PcmAudio:
    if song_index < 0:
        raise ValueError("song_index must be nonnegative")
    if not requests:
        raise ValueError("requests must not be empty")

    total_bars = max(request.end_bar for request in requests)
    renderer = ProceduralBoomBapRenderer(seed=20260816 + song_index * 10_000)
    bar_samples = [
        _samples(renderer.render(_COMMON_DRUM_TEMPLATE, _TEMPO, TARGET_STEREO_FORMAT, bar))
        for bar in range(total_bars)
    ]
    drums = _pcm_audio(TARGET_STEREO_FORMAT, np.concatenate(bar_samples, axis=0))
    if listening_wav_path is not None:
        write_listening_wav(listening_wav_path, drums)
    if float32_wav_path is not None:
        write_float32_wav(float32_wav_path, drums)
    return drums


def mix_stems(
    vocals: PcmAudio,
    drums: PcmAudio,
    *,
    vocal_gain: float = 0.80,
    drum_gain: float = 0.45,
    peak_limit: float = 0.98,
    listening_wav_path: Path | str | None = None,
    float32_wav_path: Path | str | None = None,
) -> MixResult:
    if vocals.frame_count != drums.frame_count:
        raise ValueError("vocals and drums must have the same frame count")
    if drums.format != TARGET_STEREO_FORMAT:
        raise ValueError("drums must be 48 kHz stereo float32 audio")
    if peak_limit <= 0:
        raise ValueError("peak_limit must be positive")

    vocal_samples = _samples(vocals)
    if vocals.format != TARGET_VOCAL_FORMAT:
        raise ValueError("vocals must be 48 kHz mono float32 audio")
    stereo_vocals = np.repeat(vocal_samples, TARGET_STEREO_FORMAT.channels, axis=1)
    mixed = stereo_vocals * np.float32(vocal_gain) + _samples(drums) * np.float32(drum_gain)
    peak_before = float(np.max(np.abs(mixed), initial=0.0))
    applied_gain = 1.0
    if peak_before > peak_limit:
        applied_gain = peak_limit / peak_before
        mixed = mixed * np.float32(applied_gain)
    audio = _pcm_audio(TARGET_STEREO_FORMAT, mixed)
    if listening_wav_path is not None:
        write_listening_wav(listening_wav_path, audio)
    if float32_wav_path is not None:
        write_float32_wav(float32_wav_path, audio)
    return MixResult(audio=audio, peak_before_limiter=peak_before, applied_gain=applied_gain)


def write_listening_wav(path: Path | str, audio: PcmAudio) -> None:
    samples = _samples(audio)
    pcm16 = np.asarray(np.clip(samples, -1.0, 1.0) * 32767.0, dtype=np.int16)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, audio.format.sample_rate_hz, pcm16)


def write_float32_wav(path: Path | str, audio: PcmAudio) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, audio.format.sample_rate_hz, _samples(audio).astype(np.float32, copy=False))


def _fit_chunk(chunk_index: int, samples: np.ndarray) -> tuple[np.ndarray, ChunkAssemblyDiagnostic]:
    source_frames = int(samples.shape[0])
    if source_frames == CHUNK_FRAME_COUNT:
        return (
            samples.astype(np.float32, copy=False),
            ChunkAssemblyDiagnostic(
                chunk_index=chunk_index,
                source_frames=source_frames,
                output_frames=CHUNK_FRAME_COUNT,
                action="identity",
                message="chunk already matched the two-bar target",
            ),
        )
    if source_frames < CHUNK_FRAME_COUNT:
        adjusted_frames = CHUNK_FRAME_COUNT - source_frames
        padded = np.pad(samples.astype(np.float32, copy=False), (0, adjusted_frames))
        return (
            padded,
            ChunkAssemblyDiagnostic(
                chunk_index=chunk_index,
                source_frames=source_frames,
                output_frames=CHUNK_FRAME_COUNT,
                action="pad_silence",
                message="chunk shorter than two bars; padded trailing silence",
                adjusted_frames=adjusted_frames,
            ),
        )
    adjusted_frames = source_frames - CHUNK_FRAME_COUNT
    return (
        samples[:CHUNK_FRAME_COUNT].astype(np.float32, copy=False),
        ChunkAssemblyDiagnostic(
            chunk_index=chunk_index,
            source_frames=source_frames,
            output_frames=CHUNK_FRAME_COUNT,
            action="truncate",
            message="chunk exceeded the two-bar target and was truncated",
            adjusted_frames=adjusted_frames,
        ),
    )


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 2:
        array = array.mean(axis=1, dtype=np.float32)
    elif array.ndim != 1:
        raise ValueError("WAV samples must be one- or two-dimensional")
    return _pcm_to_float32(array)


def _pcm_to_float32(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.floating):
        return np.asarray(samples, dtype=np.float32)
    if samples.dtype == np.uint8:
        return (samples.astype(np.float32) - 128.0) / 128.0
    if samples.dtype == np.int16:
        return samples.astype(np.float32) / 32768.0
    if samples.dtype == np.int32:
        return samples.astype(np.float32) / float(1 << 31)
    raise ValueError(f"unsupported WAV dtype: {samples.dtype}")


def _pcm_audio(audio_format: AudioFormat, samples: np.ndarray) -> PcmAudio:
    normalised = np.asarray(samples, dtype=np.float32)
    if normalised.ndim == 1:
        normalised = normalised.reshape(-1, 1)
    return PcmAudio(audio_format, int(normalised.shape[0]), normalised.tobytes())


def _samples(audio: PcmAudio) -> np.ndarray:
    return np.frombuffer(audio.data, dtype=np.float32).reshape(audio.frame_count, audio.format.channels)

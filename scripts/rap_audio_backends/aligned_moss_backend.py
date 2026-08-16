"""Protocol 4 backend: MFA-aligned, piecewise-warped MOSS chunk rendering."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, TwoBarRenderRequest
from streammuse.experiments.rap_audio_protocols.warp import (
    StretchRegionFn,
    VowelAnchor,
    load_textgrid_phone_intervals,
    match_vowel_anchors,
    piecewise_pitch_preserving_warp,
)


MfaCommand = Callable[..., None]


@dataclass(frozen=True)
class PendingAlignedChunk:
    request: TwoBarRenderRequest
    source_wav_path: Path
    expected_source_sha256: str


@dataclass(frozen=True)
class StagedAlignedChunk:
    request: TwoBarRenderRequest
    source_wav_path: Path
    source_sha256: str
    staged_wav_path: Path
    staged_lab_path: Path
    expected_textgrid_path: Path


@dataclass(frozen=True)
class AlignedChunkRenderResult:
    record: ChunkRenderRecord
    anchor_map: tuple[VowelAnchor, ...]
    stretch_ratios: tuple[float, ...]
    output_wav_path: Path | None


class MontrealForcedAlignerCommand:
    """Lazy CLI boundary for Montreal Forced Aligner."""

    def __init__(self, *, binary: str = "mfa") -> None:
        self._binary = binary

    def __call__(
        self,
        *,
        corpus_dir: Path,
        output_dir: Path,
        dictionary_name: str,
        acoustic_model_name: str,
    ) -> None:
        try:
            subprocess.run(
                [
                    self._binary,
                    "align",
                    str(corpus_dir),
                    dictionary_name,
                    acoustic_model_name,
                    str(output_dir),
                    "--clean",
                    "--quiet",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("mfa binary is not installed") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "unknown MFA failure"
            raise RuntimeError(f"MFA alignment failed: {stderr}") from exc


def stage_alignment_inputs(
    pending_chunks: Sequence[PendingAlignedChunk],
    staging_dir: Path | str,
    *,
    output_dir: Path | str,
) -> tuple[StagedAlignedChunk, ...]:
    staging_root = Path(staging_dir)
    output_root = Path(output_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    staged = []
    for pending in pending_chunks:
        source_sha256 = verify_source_wav_sha(pending.source_wav_path, pending.expected_source_sha256)
        stem = _chunk_stem(pending.request)
        staged_wav_path = staging_root / f"{stem}.wav"
        staged_lab_path = staging_root / f"{stem}.lab"
        shutil.copyfile(pending.source_wav_path, staged_wav_path)
        staged_lab_path.write_text(pending.request.text, encoding="utf-8")
        staged.append(
            StagedAlignedChunk(
                request=pending.request,
                source_wav_path=Path(pending.source_wav_path),
                source_sha256=source_sha256,
                staged_wav_path=staged_wav_path,
                staged_lab_path=staged_lab_path,
                expected_textgrid_path=output_root / f"{stem}.TextGrid",
            )
        )
    return tuple(staged)


def run_forced_alignment(
    corpus_dir: Path | str,
    output_dir: Path | str,
    *,
    dictionary_name: str = "english_us_arpa",
    acoustic_model_name: str = "english_us_arpa",
    mfa_command: MfaCommand | None = None,
) -> None:
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    corpus_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    (mfa_command or MontrealForcedAlignerCommand())(
        corpus_dir=corpus_path,
        output_dir=output_path,
        dictionary_name=dictionary_name,
        acoustic_model_name=acoustic_model_name,
    )


def render_aligned_chunk(
    *,
    request: TwoBarRenderRequest,
    source_wav_path: Path | str,
    expected_source_sha256: str,
    textgrid_path: Path | str,
    output_wav_path: Path | str,
    attempts: int = 1,
    stretch_region: StretchRegionFn | None = None,
    crossfade_seconds: float = 0.005,
) -> AlignedChunkRenderResult:
    output_path = Path(output_wav_path)
    source_sha256: str | None = None
    sample_rate_hz: int | None = None
    try:
        source_path = Path(source_wav_path)
        source_sha256 = verify_source_wav_sha(source_path, expected_source_sha256)
        sample_rate_hz, samples = _load_native_mono_float32(source_path)
        anchors = match_vowel_anchors(
            load_textgrid_phone_intervals(textgrid_path),
            request.syllables,
            sample_rate_hz=sample_rate_hz,
        )
        warped = piecewise_pitch_preserving_warp(
            samples,
            sample_rate_hz=sample_rate_hz,
            anchors=anchors,
            target_frame_count=round(request.duration_seconds * sample_rate_hz),
            stretch_region=stretch_region,
            crossfade_seconds=crossfade_seconds,
            source_sha256=source_sha256,
        )
        _write_native_float32_wav(output_path, sample_rate_hz, warped.samples)
        record = ChunkRenderRecord(
            protocol_id=ProtocolId.MOSS_ALIGNED,
            song_id=request.song_id,
            chunk_index=request.chunk_index,
            request_sha256=request.sha256,
            success=True,
            output_path=str(output_path),
            output_sha256=file_sha256(output_path),
            source_chunk_sha256=source_sha256,
            sample_rate_hz=sample_rate_hz,
            attempts=attempts,
        )
        return AlignedChunkRenderResult(
            record=record,
            anchor_map=warped.anchor_map,
            stretch_ratios=tuple(region.stretch_ratio for region in warped.stretch_regions),
            output_wav_path=output_path,
        )
    except Exception as exc:
        output_sha256: str | None = None
        failed_output_path: Path | None = None
        error = f"aligned_moss_backend failed: {exc}"
        if sample_rate_hz is not None:
            try:
                frame_count = round(request.duration_seconds * sample_rate_hz)
                _write_native_float32_wav(
                    output_path,
                    sample_rate_hz,
                    np.zeros(frame_count, dtype=np.float32),
                )
                output_sha256 = file_sha256(output_path)
                failed_output_path = output_path
            except Exception as silence_exc:
                error = f"{error}; silence emission failed: {silence_exc}"
        record = ChunkRenderRecord(
            protocol_id=ProtocolId.MOSS_ALIGNED,
            song_id=request.song_id,
            chunk_index=request.chunk_index,
            request_sha256=request.sha256,
            success=False,
            output_path=str(failed_output_path) if failed_output_path is not None else None,
            output_sha256=output_sha256,
            source_chunk_sha256=source_sha256,
            sample_rate_hz=sample_rate_hz,
            attempts=attempts,
            error=error,
        )
        return AlignedChunkRenderResult(
            record=record,
            anchor_map=(),
            stretch_ratios=(),
            output_wav_path=failed_output_path,
        )


def verify_source_wav_sha(path: Path | str, expected_source_sha256: str) -> str:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_source_sha256:
        raise ValueError(
            "MOSS source WAV SHA-256 mismatch: "
            f"expected {expected_source_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def _chunk_stem(request: TwoBarRenderRequest) -> str:
    return f"{request.song_id}__chunk_{request.chunk_index:02d}"


def _load_native_mono_float32(path: Path) -> tuple[int, np.ndarray]:
    sample_rate_hz, samples = wavfile.read(path)
    if sample_rate_hz <= 0:
        raise ValueError("source WAV sample rate must be positive")
    return sample_rate_hz, _to_mono_float32(samples)


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.dtype.kind in {"i", "u"}:
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        float_samples = array.astype(np.float32) / np.float32(scale)
    else:
        float_samples = array.astype(np.float32, copy=False)
    if float_samples.ndim == 1:
        return float_samples
    return np.mean(float_samples, axis=1, dtype=np.float32)


def _write_native_float32_wav(path: Path, sample_rate_hz: int, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate_hz, np.asarray(samples, dtype=np.float32))

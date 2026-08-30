"""Protocol 4 backend: MFA-aligned, piecewise-warped MOSS chunk rendering."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    TwoBarRenderRequest,
    canonical_json_dumps,
)
from streammuse.experiments.rap_audio_protocols.stress import apply_stress_envelope
from streammuse.experiments.rap_audio_protocols.warp import (
    FullChunkWarpFn,
    PhoneVowelMismatchError,
    RubberBandTimeMapStretcher,
    StretchRegionFn,
    VowelAnchor,
    WORD_TIER_FALLBACK_PREFIX,
    continuous_pitch_preserving_warp,
    load_textgrid_phone_intervals,
    load_textgrid_word_intervals,
    match_vowel_anchors,
    match_vowel_anchors_with_word_fallback,
    piecewise_pitch_preserving_warp,
    promote_vowel_anchors_to_syllable_onsets,
    regularize_anchor_targets,
    regularize_gentle_sparse_anchors,
)


MfaCommand = Callable[..., None]


class AlignedWarpMode(str, Enum):
    PIECEWISE_VOWEL_R2 = "piecewise_vowel_r2"
    CONTINUOUS_VOWEL_R3 = "continuous_vowel_r3"
    CONTINUOUS_ONSET_R3 = "continuous_onset_r3"
    CONTINUOUS_ONSET_CONSTRAINED_R3_STRESS = "continuous_onset_constrained_r3_stress"
    CONTINUOUS_ONSET_GENTLE_SPARSE_R3 = "continuous_onset_gentle_sparse_r3"
    CONTINUOUS_ONSET_R2_SMOOTH = "continuous_onset_r2_smooth"


_ONSET_MODES = frozenset(
    {
        AlignedWarpMode.CONTINUOUS_ONSET_R3,
        AlignedWarpMode.CONTINUOUS_ONSET_CONSTRAINED_R3_STRESS,
        AlignedWarpMode.CONTINUOUS_ONSET_GENTLE_SPARSE_R3,
        AlignedWarpMode.CONTINUOUS_ONSET_R2_SMOOTH,
    }
)


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
    fallback_count: int
    boundary_adjustment_count: int
    source_boundary_adjustment_count: int
    output_wav_path: Path | None
    diagnostics_path: Path | None
    mode: str = AlignedWarpMode.PIECEWISE_VOWEL_R2.value
    stress_applied: bool = False
    peak_limited: bool = False


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
    stretch_full_chunk: FullChunkWarpFn | None = None,
    crossfade_seconds: float = 0.005,
    mode: AlignedWarpMode | str = AlignedWarpMode.PIECEWISE_VOWEL_R2,
) -> AlignedChunkRenderResult:
    output_path = Path(output_wav_path)
    diagnostics_path = output_path.with_suffix(output_path.suffix + ".alignment.json")
    source_sha256: str | None = None
    sample_rate_hz: int | None = None
    requested_mode = mode.value if isinstance(mode, AlignedWarpMode) else str(mode)
    stress_payload: dict[str, Any] = {"applied": False}
    regularization_payload: dict[str, Any] = {"applied": False}
    try:
        selected_mode = AlignedWarpMode(requested_mode)
        diagnostics_path.unlink(missing_ok=True)
        source_path = Path(source_wav_path)
        source_sha256 = verify_source_wav_sha(source_path, expected_source_sha256)
        sample_rate_hz, samples = _load_native_mono_float32(source_path)
        phone_intervals = load_textgrid_phone_intervals(textgrid_path)
        try:
            anchors = match_vowel_anchors(
                phone_intervals,
                request.syllables,
                sample_rate_hz=sample_rate_hz,
                target_duration_seconds=request.duration_seconds,
            )
        except PhoneVowelMismatchError as strict_error:
            try:
                anchors = match_vowel_anchors_with_word_fallback(
                    phone_intervals,
                    load_textgrid_word_intervals(textgrid_path),
                    request.syllables,
                    sample_rate_hz=sample_rate_hz,
                    request_words=_request_words(request.text),
                    target_duration_seconds=request.duration_seconds,
                )
                if not any(
                    anchor.aligned_phone.startswith(WORD_TIER_FALLBACK_PREFIX)
                    for anchor in anchors
                ):
                    raise ValueError("word-tier fallback produced no synthesized anchors")
            except ValueError as fallback_error:
                raise ValueError(
                    f"{strict_error}; word-tier fallback failed: {fallback_error}"
                ) from fallback_error
        target_frame_count = round(request.duration_seconds * sample_rate_hz)
        if selected_mode in _ONSET_MODES:
            anchors = promote_vowel_anchors_to_syllable_onsets(
                phone_intervals,
                request.syllables,
                anchors,
                sample_rate_hz=sample_rate_hz,
            )
        if selected_mode is AlignedWarpMode.CONTINUOUS_ONSET_CONSTRAINED_R3_STRESS:
            anchors = regularize_anchor_targets(
                anchors,
                request.syllables,
                sample_rate_hz=sample_rate_hz,
                source_frame_count=len(samples),
                target_frame_count=target_frame_count,
            )
            target_drift_seconds = tuple(
                anchor.target_seconds - anchor.requested_target_seconds for anchor in anchors
            )
            regularization_payload = {
                "applied": True,
                "min_stretch_ratio": 0.5,
                "max_stretch_ratio": 2.0,
                "stress_priority": 4.0,
                "target_drift_seconds": list(target_drift_seconds),
                "max_absolute_target_drift_seconds": max(
                    abs(value) for value in target_drift_seconds
                ),
            }
        elif selected_mode is AlignedWarpMode.CONTINUOUS_ONSET_GENTLE_SPARSE_R3:
            selection = regularize_gentle_sparse_anchors(
                anchors,
                request.syllables,
                sample_rate_hz=sample_rate_hz,
                source_frame_count=len(samples),
                target_frame_count=target_frame_count,
            )
            target_drift_seconds = tuple(
                anchor.target_seconds - anchor.requested_target_seconds
                for anchor in selection.regularized_anchors
            )
            anchors = selection.anchors
            regularization_payload = {
                "applied": True,
                "policy": "gentle_sparse_r3",
                "min_stretch_ratio": 0.75,
                "max_stretch_ratio": 1.35,
                "stress_priority": 4.0,
                "minimum_target_stress": 0.8,
                "minimum_boundary_strength": 2,
                "input_anchor_count": len(selection.regularized_anchors),
                "effective_anchor_count": len(selection.anchors),
                "selected_anchor_indices": list(selection.selected_indices),
                "omitted_anchor_indices": list(selection.omitted_indices),
                "target_drift_seconds": list(target_drift_seconds),
                "max_absolute_target_drift_seconds": max(
                    abs(value) for value in target_drift_seconds
                ),
            }

        if selected_mode is AlignedWarpMode.PIECEWISE_VOWEL_R2:
            warped = piecewise_pitch_preserving_warp(
                samples,
                sample_rate_hz=sample_rate_hz,
                anchors=anchors,
                target_frame_count=target_frame_count,
                stretch_region=stretch_region,
                crossfade_seconds=crossfade_seconds,
                source_sha256=source_sha256,
            )
        else:
            full_chunk_stretcher = stretch_full_chunk
            if full_chunk_stretcher is None:
                if selected_mode is AlignedWarpMode.CONTINUOUS_ONSET_R2_SMOOTH:
                    full_chunk_stretcher = RubberBandTimeMapStretcher(
                        engine="r2",
                        smoothing=True,
                    )
                else:
                    full_chunk_stretcher = RubberBandTimeMapStretcher(engine="r3")
            warped = continuous_pitch_preserving_warp(
                samples,
                sample_rate_hz=sample_rate_hz,
                anchors=anchors,
                target_frame_count=target_frame_count,
                stretch_full_chunk=full_chunk_stretcher,
                source_sha256=source_sha256,
            )

        rendered_samples = warped.samples
        peak_limited = False
        stress_applied = (
            selected_mode is AlignedWarpMode.CONTINUOUS_ONSET_CONSTRAINED_R3_STRESS
        )
        if stress_applied:
            stress_syllables = tuple(
                replace(syllable, target_seconds=anchor.target_seconds)
                for syllable, anchor in zip(request.syllables, warped.anchor_map)
            )
            stress_result = apply_stress_envelope(
                rendered_samples,
                sample_rate_hz=sample_rate_hz,
                syllables=stress_syllables,
            )
            rendered_samples = stress_result.samples
            peak_limited = stress_result.peak_limited
            stress_payload = {
                "applied": True,
                "weak_db": -1.0,
                "strong_db": 2.5,
                "ramp_seconds": 0.025,
                "input_rms": stress_result.input_rms,
                "output_rms": stress_result.output_rms,
                "peak_limited": stress_result.peak_limited,
                "syllable_gain_db": list(stress_result.syllable_gain_db),
                "target_seconds": [item.target_seconds for item in stress_syllables],
                "requested_target_seconds": [
                    item.target_seconds for item in request.syllables
                ],
            }
        _write_native_float32_wav(output_path, sample_rate_hz, rendered_samples)
        stretch_ratios = tuple(region.stretch_ratio for region in warped.stretch_regions)
        fallback_count = sum(
            anchor.aligned_phone.startswith(WORD_TIER_FALLBACK_PREFIX)
            for anchor in warped.anchor_map
        )
        boundary_adjustment_count = sum(anchor.boundary_adjusted for anchor in warped.anchor_map)
        source_boundary_adjustment_count = sum(
            anchor.source_boundary_adjusted for anchor in warped.anchor_map
        )
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
        _write_atomic_alignment_diagnostics(
            diagnostics_path,
            {
                "schema_version": "streammuse.rap_audio_protocols.alignment_diagnostics.v2",
                "success": True,
                "mode": selected_mode.value,
                "request_sha256": request.sha256,
                "source_sha256": source_sha256,
                "output_sha256": record.output_sha256,
                "anchor_map": [_anchor_diagnostic_payload(anchor) for anchor in warped.anchor_map],
                "stretch_ratios": list(stretch_ratios),
                "fallback_count": fallback_count,
                "boundary_adjustment_count": boundary_adjustment_count,
                "source_boundary_adjustment_count": source_boundary_adjustment_count,
                "timing_regularization": regularization_payload,
                "stress": stress_payload,
                "error": None,
            },
        )
        return AlignedChunkRenderResult(
            record=record,
            anchor_map=warped.anchor_map,
            stretch_ratios=stretch_ratios,
            fallback_count=fallback_count,
            boundary_adjustment_count=boundary_adjustment_count,
            source_boundary_adjustment_count=source_boundary_adjustment_count,
            output_wav_path=output_path,
            diagnostics_path=diagnostics_path,
            mode=selected_mode.value,
            stress_applied=stress_applied,
            peak_limited=peak_limited,
        )
    except Exception as exc:
        output_sha256: str | None = None
        failed_output_path: Path | None = None
        failed_diagnostics_path: Path | None = None
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
            if failed_output_path is not None:
                try:
                    _write_atomic_alignment_diagnostics(
                        diagnostics_path,
                        {
                            "schema_version": "streammuse.rap_audio_protocols.alignment_diagnostics.v2",
                            "success": False,
                            "mode": requested_mode,
                            "request_sha256": request.sha256,
                            "source_sha256": source_sha256,
                            "output_sha256": output_sha256,
                            "anchor_map": [],
                            "stretch_ratios": [],
                            "fallback_count": 0,
                            "boundary_adjustment_count": 0,
                            "source_boundary_adjustment_count": 0,
                            "timing_regularization": regularization_payload,
                            "stress": stress_payload,
                            "error": error,
                        },
                    )
                    failed_diagnostics_path = diagnostics_path
                except Exception as diagnostics_exc:
                    error = f"{error}; alignment diagnostics emission failed: {diagnostics_exc}"
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
            fallback_count=0,
            boundary_adjustment_count=0,
            source_boundary_adjustment_count=0,
            output_wav_path=failed_output_path,
            diagnostics_path=failed_diagnostics_path,
            mode=requested_mode,
        )


def verify_source_wav_sha(path: Path | str, expected_source_sha256: str) -> str:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_source_sha256:
        raise ValueError(
            "MOSS source WAV SHA-256 mismatch: "
            f"expected {expected_source_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def _request_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*", text))


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


def _write_atomic_alignment_diagnostics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_json_dumps(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _anchor_diagnostic_payload(anchor: VowelAnchor) -> dict[str, Any]:
    payload = asdict(anchor)
    payload["effective_source_seconds"] = anchor.source_seconds
    payload["effective_source_sample"] = anchor.source_sample
    payload["effective_target_seconds"] = anchor.target_seconds
    return payload

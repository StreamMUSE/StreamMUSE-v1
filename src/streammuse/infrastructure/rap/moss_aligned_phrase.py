"""Connected MOSS phrase rendering with MMS onsets and one R3 time map."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.io import wavfile

from streammuse.application.rap.chunk_orchestration import (
    PhraseRenderFailed,
    PhraseRenderResult,
    PhraseVocalRenderer,
)
from streammuse.experiments.rap_audio_protocols.contracts import TwoBarRenderRequest
from streammuse.experiments.rap_audio_protocols.warp import (
    RubberBandTimeMapStretcher,
    VowelAnchor,
    continuous_pitch_preserving_warp,
)
from streammuse.infrastructure.rap.mms_forced_alignment import (
    MmsAlignmentResult,
    PhraseForcedAligner,
    SyllableOnsetMap,
    map_syllable_onsets,
    normalize_mms_transcript,
)
from streammuse.infrastructure.rap.moss_tts import MossPhraseResult


_OUTPUT_SAMPLE_RATE_HZ = 24_000
_MAX_DIAGNOSTIC_ANCHORS = 128
_MAX_WARNINGS = 32
_MAX_WARNING_LENGTH = 512
_WIDE_STRETCH_MIN = 0.5
_WIDE_STRETCH_MAX = 2.0


class _Synthesizer(Protocol):
    def synthesize(
        self,
        request: TwoBarRenderRequest,
        output_wav: Path,
    ) -> MossPhraseResult: ...


class _FullChunkStretcher(Protocol):
    def __call__(
        self,
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray: ...


OnsetMapper = Callable[..., SyllableOnsetMap]
StretcherFactory = Callable[..., _FullChunkStretcher]


class MossAlignedPhraseRenderer(PhraseVocalRenderer):
    """Render one exact-duration phrase through persistent H200 components."""

    def __init__(
        self,
        *,
        synthesizer: _Synthesizer,
        aligner: PhraseForcedAligner,
        onset_mapper: OnsetMapper = map_syllable_onsets,
        stretcher_factory: StretcherFactory = RubberBandTimeMapStretcher,
        rubberband_version: str = "Rubber Band R3",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._synthesizer = synthesizer
        self._aligner = aligner
        self._onset_mapper = onset_mapper
        self._stretcher = stretcher_factory(engine="r3", smoothing=False)
        self._rubberband_version = rubberband_version
        self._clock = clock
        self._lock = threading.Lock()

    def render(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
    ) -> PhraseRenderResult:
        with self._lock:
            return self._render_locked(request, Path(workspace))

    def _render_locked(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
    ) -> PhraseRenderResult:
        _validate_request_transcript(request)
        workspace.mkdir(parents=True, exist_ok=True)
        source_path = workspace / "moss-source.wav"
        vocal_path = workspace / "vocal.wav"
        vocal_partial_path = workspace / ".vocal.wav.partial"
        alignment_path = workspace / "alignment.json"
        vocal_path.unlink(missing_ok=True)
        vocal_partial_path.unlink(missing_ok=True)

        stage = "moss"
        stage_timings: dict[str, float] = {"moss": 0.0, "aligner": 0.0, "warp": 0.0}
        alignment: MmsAlignmentResult | None = None
        mapped: SyllableOnsetMap | None = None
        moss_result: MossPhraseResult | None = None
        try:
            started = self._clock()
            moss_result = self._synthesizer.synthesize(request, source_path)
            stage_timings["moss"] = _elapsed_ms(self._clock, started)
            source_sample_rate_hz, source_samples = _load_source_audio(
                source_path,
                moss_result,
            )
            if source_sample_rate_hz != _OUTPUT_SAMPLE_RATE_HZ:
                raise PhraseRenderFailed(
                    f"MOSS source rate must be {_OUTPUT_SAMPLE_RATE_HZ} Hz, got {source_sample_rate_hz}"
                )

            stage = "aligner"
            started = self._clock()
            alignment = self._aligner.align(source_path, request.text)
            mapped = self._onset_mapper(
                alignment,
                request.syllables,
                source_sample_rate_hz=source_sample_rate_hz,
                source_frame_count=len(source_samples),
            )
            stage_timings["aligner"] = _elapsed_ms(self._clock, started)

            stage = "warp"
            started = self._clock()
            target_frame_count = round(
                request.duration_seconds * _OUTPUT_SAMPLE_RATE_HZ
            )
            warped = continuous_pitch_preserving_warp(
                source_samples,
                sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
                anchors=mapped.anchors,
                target_frame_count=target_frame_count,
                stretch_full_chunk=self._stretcher,
                source_sha256=moss_result.source_wav_sha256,
            )
            output_samples = _validate_output_samples(
                warped.samples,
                expected_frame_count=target_frame_count,
            )
            vocal_wav, output_metrics = _encode_pcm16_wav(
                output_samples,
                sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
            )
            vocal_partial_path.write_bytes(vocal_wav)
            _validate_pcm16_wav(
                vocal_partial_path,
                expected_frame_count=target_frame_count,
                expected_sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
            )
            os.replace(vocal_partial_path, vocal_path)
            stage_timings["warp"] = _elapsed_ms(self._clock, started)

            stretch_ratios = tuple(
                region.stretch_ratio for region in warped.stretch_regions
            )
            warnings = _bounded_warnings(
                (
                    *moss_result.warnings,
                    *mapped.warnings,
                    *(
                        f"wide local stretch ratio retained: {ratio:.3f}"
                        for ratio in stretch_ratios
                        if ratio < _WIDE_STRETCH_MIN or ratio > _WIDE_STRETCH_MAX
                    ),
                )
            )
            retained_artifacts = {
                "source_wav": str(source_path.resolve()),
                "alignment_json": str(alignment_path.resolve()),
                "vocal_wav": str(vocal_path.resolve()),
            }
            alignment_diagnostics = _alignment_diagnostics(
                alignment=alignment,
                mapped=mapped,
                effective_anchors=warped.anchor_map,
                stretch_ratios=stretch_ratios,
                retained_artifacts=retained_artifacts,
            )
            audio_diagnostics = {
                "source_sha256": moss_result.source_wav_sha256,
                "source_sample_rate_hz": moss_result.sample_rate_hz,
                "source_frame_count": moss_result.frame_count,
                "source_duration_seconds": moss_result.duration_seconds,
                "reference_voice_sha256": moss_result.reference_voice_sha256,
                "sha256": hashlib.sha256(vocal_wav).hexdigest(),
                "sample_rate_hz": _OUTPUT_SAMPLE_RATE_HZ,
                "frame_count": target_frame_count,
                "duration_seconds": target_frame_count / _OUTPUT_SAMPLE_RATE_HZ,
                "channels": 1,
                "sample_width_bytes": 2,
                "encoding": "PCM16",
                "peak": output_metrics["peak"],
                "rms": output_metrics["rms"],
                "retained_artifacts": retained_artifacts,
                "moss_generation_settings": dict(
                    moss_result.resolved_generation_settings
                ),
            }
            diagnostic_payload = {
                "schema_version": "streammuse.moss_mms_phrase_alignment.v1",
                "success": True,
                "request_sha256": request.sha256,
                "stress_applied": False,
                "timing_regularization_applied": False,
                "alignment": _json_safe(alignment_diagnostics),
                "audio": _json_safe(audio_diagnostics),
                "warnings": list(warnings),
                "error": None,
            }
            _write_bounded_json(alignment_path, diagnostic_payload)
            return PhraseRenderResult(
                vocal_wav=vocal_wav,
                alignment_diagnostics=alignment_diagnostics,
                audio_diagnostics=audio_diagnostics,
                model_tool_versions={
                    "moss": f"{moss_result.model_id}@{moss_result.model_revision}",
                    "aligner": (
                        f"{alignment.aligner_identity}; {alignment.aligner_version}"
                    ),
                    "rubberband": self._rubberband_version,
                },
                warnings=warnings,
                stage_timings_ms=stage_timings,
            )
        except Exception as exc:
            vocal_partial_path.unlink(missing_ok=True)
            vocal_path.unlink(missing_ok=True)
            failure = (
                exc
                if isinstance(exc, PhraseRenderFailed)
                else PhraseRenderFailed(f"{stage} phrase render failed: {exc}")
            )
            _write_failure_diagnostics(
                alignment_path,
                request=request,
                stage=stage,
                error=failure,
                source_path=source_path,
                alignment=alignment,
                mapped=mapped,
                stage_timings=stage_timings,
            )
            if failure is exc:
                raise
            raise failure from exc


def _validate_request_transcript(request: TwoBarRenderRequest) -> None:
    if not isinstance(request, TwoBarRenderRequest):
        raise PhraseRenderFailed("renderer requires a two-bar render request")
    transcript_words = normalize_mms_transcript(request.text)
    planned_words: list[str] = []
    active_word: str | None = None
    expected_index = 0
    for syllable in request.syllables:
        normalized = normalize_mms_transcript(syllable.word)
        if len(normalized) != 1:
            raise PhraseRenderFailed("planned syllable word must normalize to one word")
        word = normalized[0]
        if syllable.index_in_word == 0:
            planned_words.append(word)
            active_word = word
            expected_index = 1
        elif active_word != word or syllable.index_in_word != expected_index:
            raise PhraseRenderFailed(
                "request syllable words and indices do not form the transcript"
            )
        else:
            expected_index += 1
    if tuple(planned_words) != transcript_words:
        raise PhraseRenderFailed(
            "request transcript and ordered syllable target words disagree"
        )
    previous_target = -1.0
    for syllable in request.syllables:
        if syllable.target_seconds <= previous_target:
            raise PhraseRenderFailed(
                "request syllable target times must be strictly increasing"
            )
        previous_target = syllable.target_seconds


def _load_source_audio(
    path: Path,
    result: MossPhraseResult,
) -> tuple[int, np.ndarray]:
    if result.output_wav.resolve() != path.resolve():
        raise PhraseRenderFailed("MOSS result path escaped the caller workspace")
    try:
        sample_rate_hz, samples = wavfile.read(path)
    except Exception as exc:
        raise PhraseRenderFailed("raw MOSS source WAV is unreadable") from exc
    array = np.asarray(samples)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise PhraseRenderFailed("raw MOSS source WAV must be nonempty mono audio")
    if sample_rate_hz <= 0 or not np.isfinite(array).all():
        raise PhraseRenderFailed("raw MOSS source WAV has invalid rate or samples")
    mono = _to_float32(array)
    if float(np.max(np.abs(mono.astype(np.float64)))) == 0.0:
        raise PhraseRenderFailed("raw MOSS source WAV must not be silent")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        result.source_wav_sha256 != actual_sha256
        or result.sample_rate_hz != sample_rate_hz
        or result.frame_count != len(mono)
    ):
        raise PhraseRenderFailed("MOSS source metadata does not match retained WAV")
    return int(sample_rate_hz), mono


def _to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype.kind in {"i", "u"}:
        limits = np.iinfo(samples.dtype)
        scale = max(abs(limits.min), limits.max)
        return samples.astype(np.float32) / np.float32(scale)
    return samples.astype(np.float32, copy=False)


def _validate_output_samples(
    samples: np.ndarray,
    *,
    expected_frame_count: int,
) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size != expected_frame_count:
        raise PhraseRenderFailed(
            f"warp produced {mono.size} frames, expected {expected_frame_count}"
        )
    if not np.isfinite(mono).all():
        raise PhraseRenderFailed("warp output contains non-finite samples")
    if float(np.max(np.abs(mono.astype(np.float64)))) == 0.0:
        raise PhraseRenderFailed("warp output must not be silent")
    return mono


def _encode_pcm16_wav(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
) -> tuple[bytes, Mapping[str, float]]:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.rint(clipped * np.float32(32767.0)).astype("<i2")
    if not np.any(pcm16):
        raise PhraseRenderFailed("encoded PCM16 output must not be silent")
    normalized = pcm16.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(normalized)))
    rms = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm16.tobytes())
    return buffer.getvalue(), {"peak": peak, "rms": rms}


def _validate_pcm16_wav(
    path: Path,
    *,
    expected_frame_count: int,
    expected_sample_rate_hz: int,
) -> None:
    try:
        with wave.open(str(path), "rb") as rendered:
            properties = (
                rendered.getframerate(),
                rendered.getnchannels(),
                rendered.getsampwidth(),
                rendered.getnframes(),
                rendered.getcomptype(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise PhraseRenderFailed("encoded vocal WAV is unreadable") from exc
    expected = (expected_sample_rate_hz, 1, 2, expected_frame_count, "NONE")
    if properties != expected:
        raise PhraseRenderFailed(
            f"encoded vocal WAV format mismatch: expected {expected}, got {properties}"
        )


def _alignment_diagnostics(
    *,
    alignment: MmsAlignmentResult,
    mapped: SyllableOnsetMap,
    effective_anchors: Sequence[VowelAnchor],
    stretch_ratios: tuple[float, ...],
    retained_artifacts: Mapping[str, str],
) -> Mapping[str, object]:
    limited_diagnostics = mapped.anchor_diagnostics[:_MAX_DIAGNOSTIC_ANCHORS]
    limited_anchors = tuple(effective_anchors[:_MAX_DIAGNOSTIC_ANCHORS])
    return {
        "normalized_transcript": alignment.normalized_transcript,
        "coverage": mapped.coverage,
        "confidence": alignment.confidence,
        "method_counts": dict(mapped.method_counts),
        "fallback_count": sum(
            count
            for method, count in mapped.method_counts.items()
            if method != "orthographic_vowel_groups"
        ),
        "anchor_count": len(mapped.anchors),
        "anchors_truncated": len(mapped.anchors) > _MAX_DIAGNOSTIC_ANCHORS,
        "anchor_map": tuple(dict(item) for item in limited_diagnostics),
        "source_anchors_seconds": tuple(
            anchor.source_seconds for anchor in limited_anchors
        ),
        "source_anchors_samples": tuple(
            anchor.source_sample for anchor in limited_anchors
        ),
        "target_anchors_seconds": tuple(
            anchor.target_seconds for anchor in limited_anchors
        ),
        "target_anchors_samples": tuple(
            anchor.target_sample for anchor in limited_anchors
        ),
        "stretch_ratios": stretch_ratios[: _MAX_DIAGNOSTIC_ANCHORS + 1],
        "retained_artifacts": dict(retained_artifacts),
    }


def _bounded_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    bounded = tuple(
        str(warning)[:_MAX_WARNING_LENGTH] for warning in warnings if str(warning)
    )
    deduplicated = tuple(dict.fromkeys(bounded))
    if len(deduplicated) <= _MAX_WARNINGS:
        return deduplicated
    return (
        *deduplicated[: _MAX_WARNINGS - 1],
        f"warnings truncated: {len(deduplicated) - (_MAX_WARNINGS - 1)} omitted",
    )


def _write_failure_diagnostics(
    path: Path,
    *,
    request: TwoBarRenderRequest,
    stage: str,
    error: Exception,
    source_path: Path,
    alignment: MmsAlignmentResult | None,
    mapped: SyllableOnsetMap | None,
    stage_timings: Mapping[str, float],
) -> None:
    payload = {
        "schema_version": "streammuse.moss_mms_phrase_alignment.v1",
        "success": False,
        "request_sha256": request.sha256,
        "failed_stage": stage,
        "error": str(error)[:_MAX_WARNING_LENGTH],
        "source_wav": str(source_path.resolve()) if source_path.exists() else None,
        "normalized_transcript": (
            alignment.normalized_transcript if alignment is not None else None
        ),
        "mapped_anchor_count": len(mapped.anchors) if mapped is not None else 0,
        "stage_timings_ms": dict(stage_timings),
    }
    try:
        _write_bounded_json(path, payload)
    except OSError:
        path.with_name(f".{path.name}.partial").unlink(missing_ok=True)


def _write_bounded_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.partial")
    temporary_path.unlink(missing_ok=True)
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    temporary_path.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return max(0.0, (clock() - started) * 1000.0)

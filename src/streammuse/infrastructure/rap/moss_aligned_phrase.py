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
    PhraseSourceResult,
    PhraseVocalRenderer,
    RenderBudgetExpired,
)
from streammuse.application.rap.execution import (
    ExecutionCancelled,
    ExecutionDeadlineExceeded,
    SynthesisExecutionContext,
)
from streammuse.domain.rap.remote_chunk import (
    REMOTE_CHUNK_MONITORING_SCHEMA_VERSION,
    REMOTE_CHUNK_SAMPLE_RATE_HZ,
    RemoteRapChunkRequest,
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
from streammuse.infrastructure.rap.phrase_warp import (
    MossWarpPolicy,
    R3_STRETCHER_OPTIONS,
    FullChunkStretcher as _FullChunkStretcher,
    WarpPlan as _WarpPlan,
    prepare_warp_input as _prepare_warp_input,
    resolve_warp_plan as _resolve_warp_plan,
    effective_diagnostic_anchors as _effective_diagnostic_anchors,
    to_float32 as _to_float32,
    validate_output_samples as _validate_output_samples,
    encode_pcm16_wav as _encode_pcm16_wav,
    validate_pcm16_wav as _validate_pcm16_wav,
)


_OUTPUT_SAMPLE_RATE_HZ = REMOTE_CHUNK_SAMPLE_RATE_HZ
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
        *,
        execution: SynthesisExecutionContext,
    ) -> MossPhraseResult: ...




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
        warp_policy: MossWarpPolicy | str = MossWarpPolicy.GENTLE_SPARSE_R3,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._synthesizer = synthesizer
        self._aligner = aligner
        self._onset_mapper = onset_mapper
        self._stretcher = stretcher_factory(**R3_STRETCHER_OPTIONS)
        self._rubberband_version = rubberband_version
        self._warp_policy = MossWarpPolicy(warp_policy)
        self._clock = clock
        self._lock = threading.Lock()

    def render(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> PhraseRenderResult:
        active_execution = execution or SynthesisExecutionContext.from_timeout(
            3600.0, correlation_id=f"moss-render-{getattr(request, 'chunk_index', 0)}"
        )
        while True:
            _renderer_checkpoint(active_execution)
            if self._lock.acquire(
                timeout=max(0.001, min(0.05, active_execution.remaining_seconds()))
            ):
                break
        try:
            return self._render_locked(
                request, Path(workspace), execution=active_execution
            )
        finally:
            self._lock.release()

    def render_source(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> PhraseSourceResult:
        """Synthesize and align one phrase, leaving the R3 warp to the v2 client."""
        active_execution = execution or SynthesisExecutionContext.from_timeout(
            3600.0, correlation_id=f"moss-source-{getattr(request, 'chunk_index', 0)}"
        )
        while True:
            _renderer_checkpoint(active_execution)
            if self._lock.acquire(
                timeout=max(0.001, min(0.05, active_execution.remaining_seconds()))
            ):
                break
        try:
            return self._render_locked(
                request, Path(workspace), execution=active_execution, source_only=True
            )
        finally:
            self._lock.release()

    def _render_locked(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
        *,
        execution: SynthesisExecutionContext,
        source_only: bool = False,
    ) -> PhraseRenderResult | PhraseSourceResult:
        _renderer_checkpoint(execution)
        try:
            workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PhraseRenderFailed(
                f"renderer workspace preparation failed: {exc}"
            ) from exc
        source_path = workspace / "source.wav"
        source_partial_path = workspace / ".source.partial.wav"
        vocal_path = workspace / "vocal.wav"
        vocal_partial_path = workspace / ".vocal.wav.partial"
        alignment_path = workspace / "mms_alignment.json"
        alignment_partial_path = workspace / ".mms_alignment.json.partial"
        failure_path = workspace / "render_failure.json"
        failure_partial_path = workspace / ".render_failure.json.partial"
        owned_paths = (
            source_path,
            source_partial_path,
            vocal_path,
            vocal_partial_path,
            alignment_path,
            alignment_partial_path,
            failure_path,
            failure_partial_path,
        )
        try:
            for path in owned_paths:
                path.unlink(missing_ok=True)
        except OSError as exc:
            raise PhraseRenderFailed(
                f"renderer stale-artifact cleanup failed: {exc}"
            ) from exc

        stage = "preflight"
        stage_timings: dict[str, float] = {"moss": 0.0, "aligner": 0.0, "warp": 0.0}
        alignment: MmsAlignmentResult | None = None
        mapped: SyllableOnsetMap | None = None
        moss_result: MossPhraseResult | None = None
        try:
            _validate_request_transcript(request)
            _renderer_checkpoint(execution)

            stage = "moss"
            started = self._clock()
            moss_result = self._synthesizer.synthesize(
                request, source_path, execution=execution
            )
            stage_timings["moss"] = _elapsed_ms(self._clock, started)
            _renderer_checkpoint(execution)
            source_sample_rate_hz, source_samples = _load_source_audio(
                source_path,
                moss_result,
            )
            if source_sample_rate_hz != _OUTPUT_SAMPLE_RATE_HZ:
                raise PhraseRenderFailed(
                    f"MOSS source rate must be {_OUTPUT_SAMPLE_RATE_HZ} Hz, got {source_sample_rate_hz}"
                )

            stage = "aligner"
            _renderer_checkpoint(execution)
            started = self._clock()
            with execution.uninterruptible():
                alignment = self._aligner.align(source_path, request.text)
            _renderer_checkpoint(execution)
            mapped = self._onset_mapper(
                alignment,
                request.syllables,
                source_sample_rate_hz=source_sample_rate_hz,
                source_frame_count=len(source_samples),
            )
            stage_timings["aligner"] = _elapsed_ms(self._clock, started)

            if source_only:
                _renderer_checkpoint(execution)
                result = _source_phrase_result(
                    request=request,
                    source_path=source_path,
                    moss_result=moss_result,
                    alignment=alignment,
                    mapped=mapped,
                    stage_timings={
                        "moss": stage_timings["moss"],
                        "aligner": stage_timings["aligner"],
                    },
                )
                _write_json_atomic(
                    alignment_path,
                    _source_alignment_artifact(
                        request=request,
                        moss_result=moss_result,
                        alignment=alignment,
                        mapped=mapped,
                        warnings=result.warnings,
                    ),
                )
                _renderer_checkpoint(execution)
                return result

            stage = "warp"
            _renderer_checkpoint(execution)
            started = self._clock()
            target_frame_count = RemoteRapChunkRequest.frame_count_for(
                request.tempo_bpm,
                _OUTPUT_SAMPLE_RATE_HZ,
            )
            warp_input = _prepare_warp_input(
                source_samples,
                mapped.anchors,
                sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
                source_sha256=moss_result.source_wav_sha256,
            )
            warp_plan = _resolve_warp_plan(
                request,
                warp_input,
                target_frame_count=target_frame_count,
                sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
                requested_policy=self._warp_policy,
            )
            research_warnings = tuple(
                dict.fromkeys(
                    (*moss_result.warnings, *mapped.warnings, *warp_plan.warnings)
                )
            )
            _write_json_atomic(
                alignment_path,
                _complete_alignment_artifact(
                    request=request,
                    moss_result=moss_result,
                    alignment=alignment,
                    mapped=mapped,
                    diagnostic_anchors=warp_input.diagnostic_anchors,
                    effective_anchors=(),
                    warp_plan=warp_plan,
                    endpoint_policy=warp_input.endpoint_policy,
                    stretch_ratios=(),
                    output_wav=None,
                    output_metrics=None,
                    target_frame_count=target_frame_count,
                    warnings=research_warnings,
                    warp_status="pending",
                ),
            )
            with execution.uninterruptible():
                warped = continuous_pitch_preserving_warp(
                    warp_input.samples,
                    sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
                    anchors=warp_plan.anchors,
                    target_frame_count=target_frame_count,
                    stretch_full_chunk=self._stretcher,
                    source_sha256=warp_input.source_sha256,
                )
            _renderer_checkpoint(execution)
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
            _renderer_checkpoint(execution)
            stage_timings["warp"] = _elapsed_ms(self._clock, started)

            stretch_ratios = tuple(
                region.stretch_ratio for region in warped.stretch_regions
            )
            research_warnings = tuple(
                dict.fromkeys(
                    (
                        *moss_result.warnings,
                        *mapped.warnings,
                        *warp_plan.warnings,
                        *(
                            f"wide local stretch ratio retained: {ratio:.3f}"
                            for ratio in stretch_ratios
                            if ratio < _WIDE_STRETCH_MIN or ratio > _WIDE_STRETCH_MAX
                        ),
                    )
                )
            )
            warnings = _bounded_warnings(research_warnings)
            full_alignment_artifact = _complete_alignment_artifact(
                request=request,
                moss_result=moss_result,
                alignment=alignment,
                mapped=mapped,
                diagnostic_anchors=warp_input.diagnostic_anchors,
                effective_anchors=warped.anchor_map,
                warp_plan=warp_plan,
                endpoint_policy=warp_input.endpoint_policy,
                stretch_ratios=stretch_ratios,
                output_wav=vocal_wav,
                output_metrics=output_metrics,
                target_frame_count=target_frame_count,
                warnings=research_warnings,
                warp_status="complete",
            )
            alignment_diagnostics = _alignment_diagnostics(
                mapped=mapped,
                diagnostic_anchors=_effective_diagnostic_anchors(
                    warp_input,
                    warp_plan,
                    warped.anchor_map,
                    source_frame_count=len(warp_input.samples),
                    target_frame_count=target_frame_count,
                    sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ,
                ),
                stretch_ratios=stretch_ratios,
            )
            audio_diagnostics = {
                "sample_rate_hz": _OUTPUT_SAMPLE_RATE_HZ,
                "frame_count": target_frame_count,
                "duration_seconds": target_frame_count / _OUTPUT_SAMPLE_RATE_HZ,
                "peak": output_metrics["peak"],
            }
            result = PhraseRenderResult(
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
                monitoring_summary={
                    "schema_version": REMOTE_CHUNK_MONITORING_SCHEMA_VERSION,
                    "alignment_method": alignment.aligner_identity,
                    "alignment_confidence": alignment.confidence,
                    "source_wav_sha256": moss_result.source_wav_sha256,
                },
                moss_serving_metadata=moss_result.serving_metadata.to_payload(),
            )
            _renderer_checkpoint(execution)
            _write_json_atomic(alignment_path, full_alignment_artifact)
            _renderer_checkpoint(execution)
            os.replace(vocal_partial_path, vocal_path)
            return result
        except BaseException as exc:
            _best_effort_unlink(vocal_partial_path, vocal_path)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ExecutionDeadlineExceeded):
                failure = RenderBudgetExpired("accepted render request budget expired")
                _write_failure_diagnostics(
                    failure_path,
                    request=request,
                    stage=stage,
                    error=failure,
                    source_path=source_path,
                    alignment=alignment,
                    mapped=mapped,
                    stage_timings=stage_timings,
                )
                raise failure from exc
            if isinstance(exc, ExecutionCancelled):
                failure = PhraseRenderFailed("render execution was cancelled")
                _write_failure_diagnostics(
                    failure_path,
                    request=request,
                    stage=stage,
                    error=failure,
                    source_path=source_path,
                    alignment=alignment,
                    mapped=mapped,
                    stage_timings=stage_timings,
                )
                raise failure from exc
            failure = (
                exc
                if isinstance(exc, PhraseRenderFailed)
                else PhraseRenderFailed(f"{stage} phrase render failed: {exc}")
            )
            _write_failure_diagnostics(
                failure_path,
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


def _renderer_checkpoint(execution: SynthesisExecutionContext) -> None:
    execution.checkpoint()


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


















def _alignment_diagnostics(
    *,
    mapped: SyllableOnsetMap,
    diagnostic_anchors: Sequence[VowelAnchor],
    stretch_ratios: tuple[float, ...],
) -> Mapping[str, object]:
    limited_anchors = tuple(diagnostic_anchors[:_MAX_DIAGNOSTIC_ANCHORS])
    return {
        "fallback_counts": {
            "phoneme_weighted_character": int(
                mapped.method_counts.get("phoneme_weighted_character", 0)
            ),
            "word_duration_proportional": int(
                mapped.method_counts.get("word_duration_proportional", 0)
            ),
        },
        "source_anchors": tuple(anchor.source_seconds for anchor in limited_anchors),
        "target_anchors": tuple(anchor.target_seconds for anchor in limited_anchors),
        "local_warp_ratios": stretch_ratios[: _MAX_DIAGNOSTIC_ANCHORS + 1],
    }


def _alignment_span_payload(
    alignment: MmsAlignmentResult,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    character_spans = tuple(
        {
            "word": span.word,
            "word_index": span.word_index,
            "character": span.character,
            "character_index": span.character_index,
            "start_seconds": span.start_seconds,
            "end_seconds": span.end_seconds,
            "score": span.score,
        }
        for span in alignment.character_spans
    )
    word_spans = tuple(
        {
            "word": span.word,
            "word_index": span.word_index,
            "start_seconds": span.start_seconds,
            "end_seconds": span.end_seconds,
            "score": span.score,
            "characters": tuple(
                {
                    "word": character.word,
                    "word_index": character.word_index,
                    "character": character.character,
                    "character_index": character.character_index,
                    "start_seconds": character.start_seconds,
                    "end_seconds": character.end_seconds,
                    "score": character.score,
                }
                for character in span.characters
            ),
        }
        for span in alignment.word_spans
    )
    return character_spans, word_spans


def _aligner_payload(alignment: MmsAlignmentResult) -> Mapping[str, object]:
    return {
        "identity": alignment.aligner_identity,
        "version": alignment.aligner_version,
        "alignment_time_ms": alignment.alignment_time_ms,
        "duration_seconds": alignment.duration_seconds,
        "confidence": alignment.confidence,
        "warnings": alignment.warnings,
        "source_timebase": {
            "sample_rate_hz": alignment.source_sample_rate_hz,
            "frame_count": alignment.source_frame_count,
            "duration_seconds": alignment.source_duration_seconds,
        },
        "inference_timebase": {
            "sample_rate_hz": alignment.inference_sample_rate_hz,
            "frame_count": alignment.inference_frame_count,
            "duration_seconds": alignment.duration_seconds,
        },
        "emission_frame_count": alignment.emission_frame_count,
    }


def _source_payload(moss_result: MossPhraseResult) -> Mapping[str, object]:
    return {
        "artifact": "source.wav",
        "sha256": moss_result.source_wav_sha256,
        "sample_rate_hz": moss_result.sample_rate_hz,
        "frame_count": moss_result.frame_count,
        "duration_seconds": moss_result.duration_seconds,
        "reference_voice_sha256": moss_result.reference_voice_sha256,
        "model_id": moss_result.model_id,
        "model_revision": moss_result.model_revision,
        "generation_time_ms": moss_result.generation_time_ms,
        "generation_settings": dict(moss_result.resolved_generation_settings),
        "warnings": moss_result.warnings,
    }


def _source_alignment_artifact(
    *,
    request: TwoBarRenderRequest,
    moss_result: MossPhraseResult,
    alignment: MmsAlignmentResult,
    mapped: SyllableOnsetMap,
    warnings: Sequence[str],
) -> Mapping[str, object]:
    """Private v2 evidence: everything up to MMS onsets; the client owns the warp."""
    character_spans, word_spans = _alignment_span_payload(alignment)
    anchors = tuple(
        {
            **dict(diagnostic),
            "planned_phone": anchor.planned_phone,
            "aligned_evidence": anchor.aligned_phone,
            "anchor_kind": anchor.anchor_kind,
        }
        for anchor, diagnostic in zip(mapped.anchors, mapped.anchor_diagnostics, strict=True)
    )
    return {
        "schema_version": "streammuse.mms_alignment.v1",
        "request_sha256": request.sha256,
        "normalized_transcript": alignment.normalized_transcript,
        "aligner": _aligner_payload(alignment),
        "source": _source_payload(moss_result),
        "character_spans": character_spans,
        "word_spans": word_spans,
        "mapping": {
            "coverage": mapped.coverage,
            "method_counts": dict(mapped.method_counts),
            "warnings": mapped.warnings,
            "anchors": anchors,
        },
        "warp": {"status": "deferred_to_client"},
        "output": None,
        "warnings": tuple(warnings),
    }


def _source_phrase_result(
    *,
    request: TwoBarRenderRequest,
    source_path: Path,
    moss_result: MossPhraseResult,
    alignment: MmsAlignmentResult,
    mapped: SyllableOnsetMap,
    stage_timings: Mapping[str, float],
) -> PhraseSourceResult:
    source_wav, peak, frame_count = _source_transport_wav(source_path)
    warnings = _bounded_warnings(
        tuple(dict.fromkeys((*moss_result.warnings, *mapped.warnings)))
    )
    return PhraseSourceResult(
        source_wav=source_wav,
        alignment_diagnostics={
            "fallback_counts": {
                "phoneme_weighted_character": int(
                    mapped.method_counts.get("phoneme_weighted_character", 0)
                ),
                "word_duration_proportional": int(
                    mapped.method_counts.get("word_duration_proportional", 0)
                ),
            },
            "source_onsets": tuple(anchor.source_seconds for anchor in mapped.anchors),
            "onset_confidence": tuple(
                min(1.0, max(0.0, float(diagnostic["confidence"])))
                for diagnostic in mapped.anchor_diagnostics
            ),
        },
        audio_diagnostics={
            "sample_rate_hz": _OUTPUT_SAMPLE_RATE_HZ,
            "frame_count": frame_count,
            "duration_seconds": frame_count / _OUTPUT_SAMPLE_RATE_HZ,
            "peak": peak,
        },
        model_tool_versions={
            "moss": f"{moss_result.model_id}@{moss_result.model_revision}",
            "aligner": f"{alignment.aligner_identity}; {alignment.aligner_version}",
        },
        warnings=warnings,
        stage_timings_ms=dict(stage_timings),
        monitoring_summary={
            "schema_version": REMOTE_CHUNK_MONITORING_SCHEMA_VERSION,
            "alignment_method": alignment.aligner_identity,
            "alignment_confidence": alignment.confidence,
            "source_wav_sha256": moss_result.source_wav_sha256,
        },
        moss_serving_metadata=moss_result.serving_metadata.to_payload(),
    )


def _source_transport_wav(path: Path) -> tuple[bytes, float, int]:
    """PCM16 transport copy of the raw phrase.

    A PCM16 source keeps its exact samples, so the client's float32 view (int16 /
    32768) equals the server's and an identical warp yields identical bytes.
    Float sources are quantized once with the same encoder as the final vocal.
    """
    try:
        sample_rate_hz, samples = wavfile.read(path)
    except Exception as exc:
        raise PhraseRenderFailed("raw MOSS source WAV is unreadable") from exc
    array = np.asarray(samples)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if sample_rate_hz != _OUTPUT_SAMPLE_RATE_HZ or array.ndim != 1 or array.size == 0:
        raise PhraseRenderFailed("raw MOSS source WAV must be 24 kHz mono audio")
    if array.dtype == np.int16:
        pcm16 = array.astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(_OUTPUT_SAMPLE_RATE_HZ)
            output.writeframes(pcm16.tobytes())
        source_wav = buffer.getvalue()
    else:
        source_wav, _ = _encode_pcm16_wav(_to_float32(array), sample_rate_hz=_OUTPUT_SAMPLE_RATE_HZ)
        with wave.open(io.BytesIO(source_wav), "rb") as encoded:
            pcm16 = np.frombuffer(encoded.readframes(encoded.getnframes()), dtype="<i2")
    if not np.any(pcm16):
        raise PhraseRenderFailed("raw MOSS source WAV must not be silent")
    peak = float(np.max(np.abs(pcm16.astype(np.float64)))) / 32768.0
    return source_wav, min(1.0, peak), int(pcm16.size)


def _complete_alignment_artifact(
    *,
    request: TwoBarRenderRequest,
    moss_result: MossPhraseResult,
    alignment: MmsAlignmentResult,
    mapped: SyllableOnsetMap,
    diagnostic_anchors: Sequence[VowelAnchor],
    effective_anchors: Sequence[VowelAnchor],
    warp_plan: _WarpPlan,
    endpoint_policy: Mapping[str, object],
    stretch_ratios: tuple[float, ...],
    output_wav: bytes | None,
    output_metrics: Mapping[str, float] | None,
    target_frame_count: int,
    warnings: Sequence[str],
    warp_status: str,
) -> Mapping[str, object]:
    character_spans, word_spans = _alignment_span_payload(alignment)
    anchors = tuple(
        {
            **dict(diagnostic),
            "planned_phone": anchor.planned_phone,
            "aligned_evidence": anchor.aligned_phone,
            "requested_source_seconds": anchor.requested_source_seconds,
            "requested_source_sample": anchor.requested_source_sample,
            "requested_target_seconds": anchor.requested_target_seconds,
            "anchor_kind": anchor.anchor_kind,
            "source_boundary_adjusted": anchor.source_boundary_adjusted,
            "target_boundary_adjusted": anchor.boundary_adjusted,
            "warp_source_seconds": diagnostic_anchor.source_seconds,
            "warp_source_sample": diagnostic_anchor.source_sample,
            "endpoint_role": (
                "boundary" if endpoint_policy["applied"] and index == 0 else "interior"
            ),
        }
        for index, (anchor, diagnostic, diagnostic_anchor) in enumerate(
            zip(
                mapped.anchors,
                mapped.anchor_diagnostics,
                diagnostic_anchors,
                strict=True,
            )
        )
    )
    effective = tuple(
        {
            "word": anchor.word,
            "index_in_word": anchor.index_in_word,
            "source_seconds": anchor.source_seconds,
            "source_sample": anchor.source_sample,
            "target_seconds": anchor.target_seconds,
            "target_sample": anchor.target_sample,
            "source_boundary_adjusted": anchor.source_boundary_adjusted,
            "target_boundary_adjusted": anchor.boundary_adjusted,
            "anchor_kind": anchor.anchor_kind,
        }
        for anchor in effective_anchors
    )
    output = None
    if output_wav is not None and output_metrics is not None:
        output = {
            "artifact": "vocal.wav",
            "sha256": hashlib.sha256(output_wav).hexdigest(),
            "sample_rate_hz": _OUTPUT_SAMPLE_RATE_HZ,
            "frame_count": target_frame_count,
            "duration_seconds": target_frame_count / _OUTPUT_SAMPLE_RATE_HZ,
            "channels": 1,
            "sample_width_bytes": 2,
            "encoding": "PCM16",
            "peak": output_metrics["peak"],
            "rms": output_metrics["rms"],
        }
    return {
        "schema_version": "streammuse.mms_alignment.v1",
        "request_sha256": request.sha256,
        "normalized_transcript": alignment.normalized_transcript,
        "aligner": _aligner_payload(alignment),
        "source": _source_payload(moss_result),
        "character_spans": character_spans,
        "word_spans": word_spans,
        "mapping": {
            "coverage": mapped.coverage,
            "method_counts": dict(mapped.method_counts),
            "warnings": mapped.warnings,
            "anchors": anchors,
            "effective_warp_anchors": effective,
            "endpoint_policy": dict(endpoint_policy),
        },
        "warp": {
            "status": warp_status,
            "engine": "r3",
            "smoothing": False,
            "stress_applied": False,
            "policy_requested": warp_plan.requested_policy.value,
            "policy_resolved": warp_plan.resolved_policy.value,
            "timing_regularization_applied": (
                warp_plan.timing_regularization_applied
            ),
            "ratio_bounds": (
                list(warp_plan.ratio_bounds)
                if warp_plan.ratio_bounds is not None
                else None
            ),
            "selected_anchor_indices": warp_plan.selected_request_indices,
            "omitted_anchor_indices": warp_plan.omitted_request_indices,
            "target_drift_seconds": warp_plan.target_drift_seconds,
            "local_warp_ratios": stretch_ratios,
        },
        "output": output,
        "warnings": tuple(warnings),
    }


def _best_effort_unlink(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


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
    request: object,
    stage: str,
    error: Exception,
    source_path: Path,
    alignment: MmsAlignmentResult | None,
    mapped: SyllableOnsetMap | None,
    stage_timings: Mapping[str, float],
) -> None:
    request_sha256 = getattr(request, "sha256", None)
    if not isinstance(request_sha256, str):
        request_sha256 = None
    payload = {
        "schema_version": "streammuse.moss_mms_phrase_alignment.v1",
        "success": False,
        "request_sha256": request_sha256,
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
        _write_json_atomic(path, payload)
    except Exception:
        _best_effort_unlink(path.with_name(f".{path.name}.partial"))


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.partial")
    temporary_path.unlink(missing_ok=True)
    try:
        encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
        temporary_path.write_text(encoded + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        _best_effort_unlink(temporary_path)


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

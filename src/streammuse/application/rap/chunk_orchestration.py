"""Server-side planning and orchestration for remote two-bar rap chunks."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from streammuse.application.rap.scoring import (
    lexical_continuity,
    rank_candidates,
    rhyme_quality,
)
from streammuse.application.rap.execution import (
    ExecutionCancelled,
    ExecutionDeadlineExceeded,
    SynthesisExecutionContext,
)
from streammuse.application.rap.service import CandidateGenerator, ProsodyAnalyzer
from streammuse.domain.rap import (
    CandidateBatch,
    CandidateEvaluation,
    CandidateRequest,
    ProsodyAnalysis,
    RemoteCandidateStats,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
    RemoteSelectedBar,
    REMOTE_CHUNK_ARTIFACT_IDS,
    REMOTE_CHUNK_SCHEMA_VERSION_V2,
    ScoreWeights,
    normalize_text,
)
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.infrastructure.rap.chunk_package import encode_chunk_package


class RenderBudgetExpired(RuntimeError):
    """Raised when the accepted request budget leaves no generation time."""


class NoValidCandidates(RuntimeError):
    """Raised when either requested bar has no selectable lyric."""

    def __init__(
        self,
        message: str,
        *,
        candidate_ledger: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        self.candidate_ledger = candidate_ledger


class PhraseRenderFailed(RuntimeError):
    """Raised when phrase rendering cannot produce a valid exact-duration WAV."""


_SENSITIVE_DIAGNOSTIC_ASSIGNMENT = re.compile(
    r"(?i)\b(?P<name>(?:[a-z][a-z0-9_]*_)?(?:api[-_]?key|access[-_]?key(?:[-_]?id)?|key|token|password|secret|authorization))"
    r"(?P<separator>\s*(?::|=)\s*)(?:(?:bearer\s+)?\[redacted\]|bearer\s+[^\s,;}\]]+|[^\s,;}\]]+)"
)
_BEARER_DIAGNOSTIC = re.compile(r"(?i)\bbearer\s+(?:\[redacted\]|[^\s,;}\]]+)")
_ASCII_DIGIT_RUN = re.compile(r"[0-9]+")
_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_MAX_DIAGNOSTIC_CHARS = 256


def verbalize_ascii_digits(text: str) -> str:
    """Replace each ASCII digit with its individual English spoken form."""

    def replace(match: re.Match[str]) -> str:
        words = " ".join(_DIGIT_WORDS[int(character)] for character in match.group())
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        prefix = " " if before.isascii() and before.isalpha() else ""
        suffix = " " if after.isascii() and after.isalpha() else ""
        return f"{prefix}{words}{suffix}"

    return _ASCII_DIGIT_RUN.sub(replace, text)


def _bounded_diagnostic_text(value: object) -> str:
    sanitized = _SENSITIVE_DIAGNOSTIC_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}[REDACTED]",
        str(value),
    )
    sanitized = _BEARER_DIAGNOSTIC.sub("Bearer [REDACTED]", sanitized)
    return " ".join(sanitized.split())[:_MAX_DIAGNOSTIC_CHARS]


def _bounded_exception_text(error: BaseException) -> str:
    detail = _bounded_diagnostic_text(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return _freeze(value)  # type: ignore[return-value]


@dataclass(frozen=True)
class ChunkLyricPlan:
    """A selected two-bar lyric pair and the complete planning evidence."""

    request: RemoteRapChunkRequest
    selected_bars: tuple[RemoteSelectedBar, RemoteSelectedBar]
    render_request: TwoBarRenderRequest
    candidate_stats: RemoteCandidateStats
    stage_timings_ms: Mapping[str, float]
    candidate_ledger: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, RemoteRapChunkRequest):
            raise ValueError("chunk lyric plan requires a remote chunk request")
        if not isinstance(self.selected_bars, tuple) or len(self.selected_bars) != 2:
            raise ValueError("chunk lyric plan requires two selected bars")
        if not isinstance(self.render_request, TwoBarRenderRequest):
            raise ValueError("chunk lyric plan requires a two-bar render request")
        if not isinstance(self.candidate_stats, RemoteCandidateStats):
            raise ValueError("chunk lyric plan requires candidate stats")
        if not isinstance(self.stage_timings_ms, Mapping) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in self.stage_timings_ms.values()
        ):
            raise ValueError(
                "chunk lyric stage timings must be finite non-negative values"
            )
        object.__setattr__(
            self, "stage_timings_ms", _frozen_mapping(self.stage_timings_ms)
        )
        object.__setattr__(
            self,
            "candidate_ledger",
            tuple(_frozen_mapping(item) for item in self.candidate_ledger),
        )

    @property
    def stats(self) -> RemoteCandidateStats:
        """Compatibility alias for callers that use the shorter plan name."""
        return self.candidate_stats


@dataclass(frozen=True)
class PhraseRenderResult:
    """Renderer output before the transport manifest is finalized."""

    vocal_wav: bytes
    alignment_diagnostics: Mapping[str, object]
    audio_diagnostics: Mapping[str, object]
    model_tool_versions: Mapping[str, str]
    warnings: tuple[str, ...]
    stage_timings_ms: Mapping[str, float]
    monitoring_summary: Mapping[str, object]
    moss_serving_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.vocal_wav, bytes):
            raise ValueError("phrase vocal_wav must be bytes")
        for name, value in (
            ("alignment_diagnostics", self.alignment_diagnostics),
            ("audio_diagnostics", self.audio_diagnostics),
            ("model_tool_versions", self.model_tool_versions),
            ("stage_timings_ms", self.stage_timings_ms),
            ("monitoring_summary", self.monitoring_summary),
            ("moss_serving_metadata", self.moss_serving_metadata),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"phrase {name} must be a mapping")
        required_stages = {"moss", "aligner", "warp"}
        if not required_stages.issubset(self.stage_timings_ms) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in self.stage_timings_ms.values()
        ):
            raise ValueError(
                "phrase stage timings must include finite moss, aligner, and warp values"
            )
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) for item in self.warnings
        ):
            raise ValueError("phrase warnings must be a tuple of strings")
        object.__setattr__(
            self, "alignment_diagnostics", _frozen_mapping(self.alignment_diagnostics)
        )
        object.__setattr__(
            self, "audio_diagnostics", _frozen_mapping(self.audio_diagnostics)
        )
        object.__setattr__(
            self, "model_tool_versions", _frozen_mapping(self.model_tool_versions)
        )
        object.__setattr__(
            self, "stage_timings_ms", _frozen_mapping(self.stage_timings_ms)
        )
        object.__setattr__(
            self, "monitoring_summary", _frozen_mapping(self.monitoring_summary)
        )
        object.__setattr__(
            self,
            "moss_serving_metadata",
            _frozen_mapping(self.moss_serving_metadata),
        )


@dataclass(frozen=True)
class PhraseSourceResult:
    """Raw MOSS phrase and measured syllable onsets for a v2 chunk (no warp)."""

    source_wav: bytes
    alignment_diagnostics: Mapping[str, object]
    audio_diagnostics: Mapping[str, object]
    model_tool_versions: Mapping[str, str]
    warnings: tuple[str, ...]
    stage_timings_ms: Mapping[str, float]
    monitoring_summary: Mapping[str, object]
    moss_serving_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_wav, bytes):
            raise ValueError("phrase source_wav must be bytes")
        for name, value in (
            ("alignment_diagnostics", self.alignment_diagnostics),
            ("audio_diagnostics", self.audio_diagnostics),
            ("model_tool_versions", self.model_tool_versions),
            ("stage_timings_ms", self.stage_timings_ms),
            ("monitoring_summary", self.monitoring_summary),
            ("moss_serving_metadata", self.moss_serving_metadata),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"phrase {name} must be a mapping")
        if not {"moss", "aligner"}.issubset(self.stage_timings_ms) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in self.stage_timings_ms.values()
        ):
            raise ValueError("phrase stage timings must include finite moss and aligner values")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(item, str) for item in self.warnings
        ):
            raise ValueError("phrase warnings must be a tuple of strings")
        for name in (
            "alignment_diagnostics",
            "audio_diagnostics",
            "model_tool_versions",
            "stage_timings_ms",
            "monitoring_summary",
            "moss_serving_metadata",
        ):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))


class PhraseVocalRenderer(Protocol):
    """Replaceable phrase renderer implemented by the persistent H200 worker."""

    def render(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
        *,
        execution: SynthesisExecutionContext,
    ) -> PhraseRenderResult:
        """Render one connected two-bar vocal phrase (v1: synthesized and warped)."""

    def render_source(
        self,
        request: TwoBarRenderRequest,
        workspace: Path,
        *,
        execution: SynthesisExecutionContext,
    ) -> PhraseSourceResult:
        """Synthesize and align one phrase without warping it (v2)."""


@dataclass(frozen=True)
class RemoteChunkRenderArtifact:
    """Validated server artifact ready for transport packaging."""

    manifest: RemoteRapChunkManifest
    vocal_wav: bytes
    candidate_ledger: tuple[Mapping[str, object], ...]
    workspace: Path
    moss_serving_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RemoteRapChunkManifest):
            raise ValueError("remote render artifact requires a manifest")
        if not isinstance(self.vocal_wav, bytes):
            raise ValueError("remote render artifact vocal_wav must be bytes")
        if not isinstance(self.workspace, Path):
            raise ValueError("remote render artifact workspace must be a Path")
        object.__setattr__(
            self,
            "candidate_ledger",
            tuple(_frozen_mapping(item) for item in self.candidate_ledger),
        )
        object.__setattr__(
            self,
            "moss_serving_metadata",
            _frozen_mapping(self.moss_serving_metadata),
        )


@dataclass
class _CandidateRecord:
    candidate_id: str
    text: str
    normalized_text: str
    analysis: ProsodyAnalysis
    source: str
    source_order: int
    provider_choice_index: int | None
    wave: int
    ledger_index: int
    evaluation: CandidateEvaluation | None = None


@dataclass
class _BarState:
    position: int
    attempted_count: int = 0
    received_count: int = 0
    wave_index: int = 0
    records: list[_CandidateRecord] = field(default_factory=list)
    normalized_seen: set[str] = field(default_factory=set)

    @property
    def valid_count(self) -> int:
        return sum(
            item.evaluation is not None and item.evaluation.valid
            for item in self.records
        )

    def selectable(self, minimum_score: float) -> tuple[_CandidateRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.evaluation is not None
            and item.evaluation.valid
            and item.evaluation.total_score is not None
            and item.evaluation.total_score >= minimum_score
        )


class ChunkCandidatePlanner:
    """Generate, hard-gate, rank, and pair two bars under one time budget."""

    def __init__(
        self,
        generator: CandidateGenerator,
        analyzer: ProsodyAnalyzer,
        weights: ScoreWeights,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        bar_generators: tuple[CandidateGenerator, CandidateGenerator] | None = None,
    ) -> None:
        """``bar_generators`` (one per bar, each with its own client) lets the
        two initial waves generate concurrently; evaluation stays in bar order."""
        if bar_generators is not None and len(bar_generators) != 2:
            raise ValueError("bar_generators must provide exactly one generator per bar")
        self._generator = generator
        self._bar_generators = bar_generators
        self._analyzer = analyzer
        self._weights = weights
        self._monotonic = monotonic

    def plan(
        self,
        request: RemoteRapChunkRequest,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> ChunkLyricPlan:
        if not isinstance(request, RemoteRapChunkRequest):
            raise ValueError("chunk planning requires a RemoteRapChunkRequest")
        started_at = self._monotonic()
        active_execution = execution or SynthesisExecutionContext(
            deadline_monotonic=started_at + request.remaining_budget_ms / 1000.0,
            correlation_id=request.request_id,
            clock=self._monotonic,
            validation_time_monotonic=started_at,
        )
        render_reserve_seconds = request.policy.render_reserve_ms / 1000.0
        if _execution_remaining_seconds(active_execution) <= render_reserve_seconds:
            raise RenderBudgetExpired(
                "render reserve consumes the complete accepted request budget"
            )

        states = [_BarState(0), _BarState(1)]
        ledger: list[dict[str, object]] = []
        generation_ms = 0.0
        evaluation_ms = 0.0
        waves_started = 0

        initial_wave_size = min(
            request.policy.initial_candidates, request.policy.maximum_candidates
        )
        if self._bar_generators is not None and initial_wave_size > 0:
            if _execution_remaining_seconds(active_execution) <= render_reserve_seconds:
                raise RenderBudgetExpired(
                    "generation cutoff expired before the first candidate wave"
                )
            waves_started += len(states)
            generated, evaluated = self._run_concurrent_initial_waves(
                request, states, initial_wave_size, ledger, active_execution
            )
            generation_ms += generated
            evaluation_ms += evaluated
        else:
            for state in states:
                wave_size = initial_wave_size
                if wave_size <= 0:
                    continue
                if (
                    _execution_remaining_seconds(active_execution)
                    <= render_reserve_seconds
                ):
                    if waves_started == 0:
                        raise RenderBudgetExpired(
                            "generation cutoff expired before the first candidate wave"
                        )
                    break
                waves_started += 1
                generated, evaluated = self._run_wave(
                    request, state, wave_size, ledger, active_execution
                )
                generation_ms += generated
                evaluation_ms += evaluated

        while True:
            progressed = False
            cutoff_reached = False
            for state in states:
                if state.valid_count >= request.policy.minimum_valid_candidates:
                    continue
                remaining = request.policy.maximum_candidates - state.attempted_count
                wave_size = min(request.policy.rescue_candidates, remaining)
                if wave_size <= 0:
                    continue
                if (
                    _execution_remaining_seconds(active_execution)
                    <= render_reserve_seconds
                ):
                    if waves_started == 0:
                        raise RenderBudgetExpired(
                            "generation cutoff expired before the first candidate wave"
                        )
                    cutoff_reached = True
                    break
                waves_started += 1
                generated, evaluated = self._run_wave(
                    request, state, wave_size, ledger, active_execution
                )
                generation_ms += generated
                evaluation_ms += evaluated
                progressed = True
            if cutoff_reached or not progressed:
                break

        selectable_by_bar = [
            state.selectable(request.policy.minimum_score) for state in states
        ]
        if any(not candidates for candidates in selectable_by_bar):
            has_valid_unselectable = any(
                state.valid_count and not selectable
                for state, selectable in zip(states, selectable_by_bar, strict=True)
            )
            message = (
                "minimum score left at least one bar without a selectable candidate"
                if has_valid_unselectable
                else "no selectable candidate for at least one requested bar"
            )
            raise NoValidCandidates(
                message,
                candidate_ledger=tuple(_frozen_mapping(item) for item in ledger),
            )

        left, right, pair_diagnostics = self._select_pair(
            selectable_by_bar[0],
            selectable_by_bar[1],
            second_topic=request.bars[1].topic,
        )
        summaries = [self._bar_summaries(request, state, ledger) for state in states]
        selected_bars = tuple(
            self._selected_bar(
                request,
                state,
                selected,
                pair_diagnostics,
                summaries[state.position],
            )
            for state, selected in zip(states, (left, right), strict=True)
        )
        render_request = self._render_request(request, selected_bars)
        stats = self._candidate_stats(request, states, ledger)
        elapsed_ms = max(0.0, (self._monotonic() - started_at) * 1000.0)
        accounted_ms = generation_ms + evaluation_ms
        total_ms = max(elapsed_ms, accounted_ms)
        return ChunkLyricPlan(
            request=request,
            selected_bars=selected_bars,  # type: ignore[arg-type]
            render_request=render_request,
            candidate_stats=stats,
            stage_timings_ms={
                "generation": generation_ms,
                "evaluation": evaluation_ms,
                "overhead": total_ms - accounted_ms,
                "total": total_ms,
            },
            candidate_ledger=tuple(_frozen_mapping(item) for item in ledger),
        )

    def _run_concurrent_initial_waves(
        self,
        request: RemoteRapChunkRequest,
        states: list[_BarState],
        wave_size: int,
        ledger: list[dict[str, object]],
        execution: SynthesisExecutionContext,
    ) -> tuple[float, float]:
        _execution_checkpoint(execution)
        prepared = [
            (state, state.wave_index, self._next_candidate_request(request, state, wave_size))
            for state in states
        ]
        started = self._monotonic()
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [
                pool.submit(
                    self._generate_batch,
                    self._generator_for(state),
                    candidate_request,
                    execution,
                )
                for state, _, candidate_request in prepared
            ]
            results = [future.result() for future in futures]
        # Generation overlapped, so its stage time is wall time, not a sum.
        generation_ms = max(0.0, (self._monotonic() - started) * 1000.0)
        evaluation_ms = 0.0
        for (state, wave, candidate_request), (batch, error, _) in zip(
            prepared, results, strict=True
        ):
            _, evaluated = self._record_wave(
                request,
                state,
                wave,
                wave_size,
                candidate_request,
                batch,
                error,
                0.0,
                ledger,
                execution,
            )
            evaluation_ms += evaluated
        return generation_ms, evaluation_ms

    def _generator_for(self, state: _BarState) -> CandidateGenerator:
        if self._bar_generators is None:
            return self._generator
        return self._bar_generators[state.position]

    def _run_wave(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        wave_size: int,
        ledger: list[dict[str, object]],
        execution: SynthesisExecutionContext,
    ) -> tuple[float, float]:
        _execution_checkpoint(execution)
        wave = state.wave_index
        candidate_request = self._next_candidate_request(request, state, wave_size)
        batch, error, generation_ms = self._generate_batch(
            self._generator_for(state), candidate_request, execution
        )
        return self._record_wave(
            request,
            state,
            wave,
            wave_size,
            candidate_request,
            batch,
            error,
            generation_ms,
            ledger,
            execution,
        )

    def _next_candidate_request(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        wave_size: int,
    ) -> CandidateRequest:
        bar = request.bars[state.position]
        wave = state.wave_index
        candidate_request = CandidateRequest(
            request_id=f"{request.request_id}:bar:{bar.bar}:wave:{wave}",
            target_bar=bar.bar,
            topic=bar.topic,
            flow_template=bar.flow_template,
            count=wave_size,
            context_lines=request.context_lines,
            seed=request.seed + state.position * 10_000 + wave,
        )
        state.wave_index += 1
        state.attempted_count += wave_size
        return candidate_request

    def _generate_batch(
        self,
        generator: CandidateGenerator,
        candidate_request: CandidateRequest,
        execution: SynthesisExecutionContext,
    ) -> tuple[CandidateBatch | None, Exception | None, float]:
        generation_started = self._monotonic()
        try:
            with execution.uninterruptible():
                batch = generator.generate(candidate_request)
            if not isinstance(batch, CandidateBatch):
                raise ValueError("candidate generator returned a malformed batch")
            if batch.request_id != candidate_request.request_id:
                raise ValueError("candidate generator returned a mismatched request_id")
        except (RenderBudgetExpired, PhraseRenderFailed):
            raise
        except Exception as exc:
            generation_ms = max(0.0, (self._monotonic() - generation_started) * 1000.0)
            return None, exc, generation_ms
        generation_ms = max(0.0, (self._monotonic() - generation_started) * 1000.0)
        return batch, None, generation_ms

    def _record_wave(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        wave: int,
        wave_size: int,
        candidate_request: CandidateRequest,
        batch: CandidateBatch | None,
        exc: Exception | None,
        generation_ms: float,
        ledger: list[dict[str, object]],
        execution: SynthesisExecutionContext,
    ) -> tuple[float, float]:
        bar = request.bars[state.position]
        if exc is not None or batch is None:
            evaluation_started = self._monotonic()
            ledger.append(
                {
                    "bar": bar.bar,
                    "wave": wave,
                    "status": "generation_error",
                    "reasons": (type(exc).__name__, str(exc)),
                    "requested_count": wave_size,
                }
            )
            self._rerank(request, state, ledger)
            evaluation_ms = max(0.0, (self._monotonic() - evaluation_started) * 1000.0)
            return generation_ms, evaluation_ms

        _execution_checkpoint(execution)

        evaluation_started = self._monotonic()
        if isinstance(batch.warning, str) and batch.warning.strip():
            ledger.append(
                {
                    "bar": bar.bar,
                    "wave": wave,
                    "source": batch.source,
                    "status": "generation_warning",
                    "reasons": (_bounded_diagnostic_text(batch.warning),),
                    "requested_count": wave_size,
                }
            )
        if batch.error_type is not None:
            ledger.append(
                {
                    "bar": bar.bar,
                    "wave": wave,
                    "source": batch.source,
                    "status": "generation_error",
                    "reasons": tuple(
                        item
                        for item in (batch.error_type, batch.error_message)
                        if isinstance(item, str) and item
                    ),
                    "requested_count": wave_size,
                }
            )
        for response_index, raw_text in enumerate(batch.candidates):
            source_order = state.received_count
            state.received_count += 1
            provider_choice_index = (
                batch.provider_choice_indices[response_index]
                if batch.provider_choice_indices
                else None
            )
            candidate_id = (
                f"{request.request_id}:bar:{bar.bar}:candidate:{source_order}"
            )
            text = verbalize_ascii_digits(raw_text.strip())
            base_row: dict[str, object] = {
                "bar": bar.bar,
                "wave": wave,
                "candidate_id": candidate_id,
                "source": batch.source,
                "source_order": source_order,
                "provider_choice_index": provider_choice_index,
                "text": text,
            }
            if response_index >= wave_size:
                ledger.append(
                    {
                        **base_row,
                        "status": "over_returned",
                        "reasons": ("over_returned",),
                    }
                )
                continue
            normalized = normalize_text(text)
            if not normalized:
                ledger.append(
                    {
                        **base_row,
                        "status": "malformed",
                        "reasons": ("empty_normalized_text",),
                    }
                )
                continue
            if normalized in state.normalized_seen:
                ledger.append(
                    {
                        **base_row,
                        "status": "duplicate",
                        "reasons": ("duplicate_normalized_text",),
                    }
                )
                continue
            state.normalized_seen.add(normalized)
            try:
                analysis = self._analyzer.analyze(text)
            except Exception as exc:
                ledger.append(
                    {
                        **base_row,
                        "normalized_text": normalized,
                        "status": "analysis_error",
                        "reasons": (type(exc).__name__, str(exc)),
                    }
                )
                continue
            ledger_index = len(ledger)
            ledger.append(
                {
                    **base_row,
                    "normalized_text": normalized,
                    "status": "pending_evaluation",
                    "reasons": (),
                }
            )
            state.records.append(
                _CandidateRecord(
                    candidate_id=candidate_id,
                    text=text,
                    normalized_text=normalized,
                    analysis=analysis,
                    source=batch.source,
                    source_order=source_order,
                    provider_choice_index=provider_choice_index,
                    wave=wave,
                    ledger_index=ledger_index,
                )
            )
        self._rerank(request, state, ledger)
        evaluation_ms = max(0.0, (self._monotonic() - evaluation_started) * 1000.0)
        _execution_checkpoint(execution)
        return generation_ms, evaluation_ms

    def _rerank(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        ledger: list[dict[str, object]],
    ) -> None:
        bar = request.bars[state.position]
        result = rank_candidates(
            tuple(
                (item.candidate_id, item.text, item.analysis) for item in state.records
            ),
            template=bar.flow_template,
            topic=bar.topic,
            history=request.context_lines,
            rhyme_anchors={},
            weights=self._weights,
            minimum_score=request.policy.minimum_score,
            segment_start_bar=request.bars[0].bar,
            target_bar=bar.bar,
        )
        for record, evaluation in zip(state.records, result.evaluations, strict=True):
            record.evaluation = evaluation
            component_scores = {item.name: item.value for item in evaluation.components}
            component_contributions = {
                item.name: item.contribution for item in evaluation.components
            }
            if not evaluation.valid:
                status = "rejected"
                reasons = evaluation.rejection_reasons
            elif (
                evaluation.total_score is None
                or evaluation.total_score < request.policy.minimum_score
            ):
                status = "below_minimum_score"
                reasons = ("minimum_score_not_met",)
            else:
                status = "selectable"
                reasons = ()
            ledger[record.ledger_index].update(
                {
                    "status": status,
                    "reasons": reasons,
                    "score": evaluation.total_score,
                    "component_scores": component_scores,
                    "component_contributions": component_contributions,
                    "syllable_count": len(evaluation.analysis.syllables),
                }
            )

    @staticmethod
    def _select_pair(
        left: Sequence[_CandidateRecord],
        right: Sequence[_CandidateRecord],
        *,
        second_topic: str,
    ) -> tuple[_CandidateRecord, _CandidateRecord, Mapping[str, object]]:
        ranked_pairs: list[
            tuple[
                tuple[float, float, float, int, int],
                _CandidateRecord,
                _CandidateRecord,
                dict[str, object],
            ]
        ] = []
        for left_item, right_item in product(left, right):
            assert (
                left_item.evaluation is not None
                and left_item.evaluation.total_score is not None
            )
            assert (
                right_item.evaluation is not None
                and right_item.evaluation.total_score is not None
            )
            mean_score = (
                left_item.evaluation.total_score + right_item.evaluation.total_score
            ) / 2.0
            continuity = lexical_continuity(
                right_item.analysis,
                (left_item.analysis,),
                second_topic,
            )
            rhyme = rhyme_quality(
                right_item.analysis.end_rhyme_tail, left_item.analysis.end_rhyme_tail
            )
            key = (
                mean_score,
                continuity,
                rhyme,
                -left_item.source_order,
                -right_item.source_order,
            )
            diagnostics = {
                "mean_total_score": mean_score,
                "lexical_continuity": continuity,
                "rhyme_quality": rhyme,
                "source_order": (left_item.source_order, right_item.source_order),
                "ordering": "mean_total_score, lexical_continuity, rhyme_quality, source_order",
            }
            ranked_pairs.append((key, left_item, right_item, diagnostics))
        _key, selected_left, selected_right, diagnostics = max(
            ranked_pairs, key=lambda item: item[0]
        )
        return selected_left, selected_right, _frozen_mapping(diagnostics)

    def _bar_summaries(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        ledger: Sequence[Mapping[str, object]],
    ) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
        selectable = sorted(
            state.selectable(request.policy.minimum_score),
            key=lambda item: (-(item.evaluation.total_score or 0.0), item.source_order),
        )
        top = tuple(self._top_summary(request, state, item) for item in selectable[:8])
        rejected = tuple(
            self._rejection_summary(row)
            for row in ledger
            if row.get("bar") == request.bars[state.position].bar
            and row.get("candidate_id")
            and row.get("status") != "selectable"
            and row.get("status") != "over_returned"
        )[:8]
        return top, rejected

    @staticmethod
    def _top_summary(
        request: RemoteRapChunkRequest,
        state: _BarState,
        record: _CandidateRecord,
    ) -> Mapping[str, object]:
        assert (
            record.evaluation is not None and record.evaluation.total_score is not None
        )
        return _frozen_mapping(
            {
                "bar": request.bars[state.position].bar,
                "candidate_id": record.candidate_id,
                "text": record.text,
                "score": record.evaluation.total_score,
                "component_scores": {
                    item.name: item.value for item in record.evaluation.components
                },
                "source_order": record.source_order,
            }
        )

    @staticmethod
    def _rejection_summary(row: Mapping[str, object]) -> Mapping[str, object]:
        return _frozen_mapping(
            {
                "bar": row["bar"],
                "candidate_id": row["candidate_id"],
                "text": row.get("text", ""),
                "reasons": tuple(
                    str(item) for item in row.get("reasons", ("rejected",))
                ),
                "source_order": row["source_order"],
            }
        )

    def _candidate_stats(
        self,
        request: RemoteRapChunkRequest,
        states: Sequence[_BarState],
        ledger: Sequence[Mapping[str, object]],
    ) -> RemoteCandidateStats:
        selectable = [
            (state, item)
            for state in states
            for item in state.selectable(request.policy.minimum_score)
        ]
        selectable.sort(
            key=lambda pair: (
                -(pair[1].evaluation.total_score or 0.0),
                pair[0].position,
                pair[1].source_order,
            )
        )
        top = tuple(
            self._top_summary(request, state, item) for state, item in selectable[:8]
        )
        rejected_rows = [
            row
            for row in ledger
            if row.get("candidate_id")
            and row.get("status") != "selectable"
            and row.get("status") != "over_returned"
        ]
        rejections = tuple(self._rejection_summary(row) for row in rejected_rows[:8])
        return RemoteCandidateStats(
            requested_count=sum(state.attempted_count for state in states),
            parseable_count=sum(len(state.records) for state in states),
            valid_count=sum(state.valid_count for state in states),
            selectable_count=len(selectable),
            top_candidates=top,
            rejections=rejections,
        )

    def _selected_bar(
        self,
        request: RemoteRapChunkRequest,
        state: _BarState,
        selected: _CandidateRecord,
        pair_diagnostics: Mapping[str, object],
        summaries: tuple[
            tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]
        ],
    ) -> RemoteSelectedBar:
        evaluation = selected.evaluation
        assert evaluation is not None and evaluation.total_score is not None
        component_scores = {item.name: item.value for item in evaluation.components}
        component_contributions = {
            item.name: item.contribution for item in evaluation.components
        }
        top_candidates, rejections = summaries
        return RemoteSelectedBar.create(
            request.bars[state.position],
            text=selected.text,
            scheduled=evaluation.scheduled,
            score=evaluation.total_score,
            diagnostics={
                "candidate_id": selected.candidate_id,
                "source": selected.source,
                "source_order": selected.source_order,
                "provider_choice_index": selected.provider_choice_index,
                "component_scores": component_scores,
                "component_contributions": component_contributions,
                "pair_tiebreak": pair_diagnostics,
                "top_candidates": top_candidates,
                "representative_rejections": rejections,
            },
        )

    @staticmethod
    def _render_request(
        request: RemoteRapChunkRequest,
        selected_bars: tuple[RemoteSelectedBar, RemoteSelectedBar],
    ) -> TwoBarRenderRequest:
        return chunk_render_request(request, selected_bars)


def chunk_render_request(
    request: RemoteRapChunkRequest,
    selected_bars: tuple[RemoteSelectedBar, RemoteSelectedBar],
) -> TwoBarRenderRequest:
    """The phrase render request for two selected bars.

    Shared by the server (which renders it) and the Mac v2 client (which warps
    the returned source phrase onto the same targets), so both sides compute
    bit-identical target times.
    """
    chunk_start_tick = selected_bars[0].bar * 16
    seconds_per_tick = 60.0 / request.tempo_bpm / 4.0
    syllables = tuple(
        SyllableTarget(
            word=scheduled.syllable.word,
            index_in_word=scheduled.syllable.index_in_word,
            phonemes=scheduled.syllable.phonemes,
            lexical_stress=scheduled.syllable.stress,
            target_stress=scheduled.slot.accent,
            boundary_strength=scheduled.slot.boundary_strength,
            absolute_tick=scheduled.slot.tick,
            tick_in_chunk=scheduled.slot.tick - chunk_start_tick,
            target_seconds=(scheduled.slot.tick - chunk_start_tick)
            * seconds_per_tick,
        )
        for selected in selected_bars
        for scheduled in selected.scheduled
    )
    return TwoBarRenderRequest(
        song_id=request.session_id,
        chunk_index=request.chunk_index,
        start_bar=selected_bars[0].bar,
        end_bar=selected_bars[1].bar + 1,
        text="\n".join(item.text for item in selected_bars),
        syllables=syllables,
        tempo_bpm=request.tempo_bpm,
    )


class RapChunkOrchestrator:
    """Plan lyrics, render one phrase, and validate the resulting artifact."""

    def __init__(
        self,
        planner: ChunkCandidatePlanner,
        renderer: PhraseVocalRenderer,
        *,
        workspace_root: Path,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._planner = planner
        self._renderer = renderer
        self._workspace_root = Path(workspace_root)
        self._monotonic = monotonic

    def render(
        self,
        request: RemoteRapChunkRequest,
        *,
        execution: SynthesisExecutionContext | None = None,
    ) -> RemoteChunkRenderArtifact:
        started_at = self._monotonic()
        active_execution = execution or SynthesisExecutionContext(
            deadline_monotonic=started_at + request.remaining_budget_ms / 1000.0,
            correlation_id=request.request_id,
            clock=self._monotonic,
            validation_time_monotonic=started_at,
        )
        _execution_checkpoint(active_execution)
        lyric_plan = self._planner.plan(request, execution=active_execution)
        _execution_checkpoint(active_execution)
        if lyric_plan.request.canonical_json_bytes() != request.canonical_json_bytes():
            raise PhraseRenderFailed("planner returned a mismatched request identity")
        workspace = self._workspace_root / request.request_id
        try:
            workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PhraseRenderFailed(
                f"workspace preparation failed: {_bounded_exception_text(exc)}"
            ) from exc
        # v2 returns the raw phrase and its onsets; the Mac plans and applies R3.
        source_only = request.schema_version == REMOTE_CHUNK_SCHEMA_VERSION_V2
        try:
            if source_only:
                render_source = getattr(self._renderer, "render_source", None)
                if not callable(render_source):
                    raise PhraseRenderFailed(
                        "phrase renderer does not support v2 source-phrase chunks"
                    )
                phrase = render_source(
                    lyric_plan.render_request,
                    workspace,
                    execution=active_execution,
                )
            else:
                phrase = self._renderer.render(
                    lyric_plan.render_request,
                    workspace,
                    execution=active_execution,
                )
        except PhraseRenderFailed:
            raise
        except Exception as exc:
            raise PhraseRenderFailed(
                f"phrase renderer failed: {_bounded_exception_text(exc)}"
            ) from exc
        expected_type = PhraseSourceResult if source_only else PhraseRenderResult
        if not isinstance(phrase, expected_type):
            raise PhraseRenderFailed("phrase renderer returned a malformed result")
        _execution_checkpoint(active_execution)
        packaged_audio = phrase.source_wav if source_only else phrase.vocal_wav

        stage_timings = {
            "generation": float(lyric_plan.stage_timings_ms.get("generation", 0.0)),
            "evaluation": float(lyric_plan.stage_timings_ms.get("evaluation", 0.0)),
            "moss": float(phrase.stage_timings_ms["moss"]),
            "aligner": float(phrase.stage_timings_ms["aligner"]),
            "packaging": 0.0,
        }
        if not source_only:
            stage_timings["warp"] = float(phrase.stage_timings_ms["warp"])
        planning_overhead = float(lyric_plan.stage_timings_ms.get("overhead", 0.0))
        measured_total = max(0.0, (self._monotonic() - started_at) * 1000.0)
        stage_timings["total"] = max(
            measured_total, sum(stage_timings.values()) + planning_overhead
        )
        planning_warnings = tuple(
            str(reason)
            for row in lyric_plan.candidate_ledger
            if row.get("status") == "generation_warning"
            for reason in row.get("reasons", ())
        )
        warnings = tuple(
            (
                *planning_warnings,
                *phrase.warnings,
                "packaging timing is provisional",
            )
        )
        try:
            diagnostics = RemoteRapChunkDiagnostics(
                accepted_request_budget_ms=request.remaining_budget_ms,
                resolved_policy=request.policy,
                candidate_stats=lyric_plan.candidate_stats,
                stage_timings_ms=stage_timings,
                alignment_diagnostics=phrase.alignment_diagnostics,
                audio_diagnostics=phrase.audio_diagnostics,
                model_tool_versions=phrase.model_tool_versions,
                warnings=warnings,
                monitoring_summary={
                    **phrase.monitoring_summary,
                    "artifact_ids": dict(REMOTE_CHUNK_ARTIFACT_IDS),
                },
                schema_version=request.schema_version,
            )
            manifest = RemoteRapChunkManifest(
                request_id=request.request_id,
                chunk_index=request.chunk_index,
                tempo_bpm=request.tempo_bpm,
                output_sample_rate_hz=request.output_sample_rate_hz,
                expected_frame_count=request.expected_frame_count,
                selected_bars=lyric_plan.selected_bars,
                diagnostics=diagnostics,
                vocal_sha256=hashlib.sha256(packaged_audio).hexdigest(),
                schema_version=request.schema_version,
            )
            # Task 4 owns measured packaging. Encoding here is solely the shared
            # contract validator for format, duration, silence, and hash.
            _execution_checkpoint(active_execution)
            encode_chunk_package(manifest, packaged_audio)
            _execution_checkpoint(active_execution)
        except (TypeError, ValueError) as exc:
            raise PhraseRenderFailed(
                f"rendered phrase failed package validation: {exc}"
            ) from exc
        return RemoteChunkRenderArtifact(
            manifest=manifest,
            vocal_wav=packaged_audio,
            candidate_ledger=lyric_plan.candidate_ledger,
            workspace=workspace,
            moss_serving_metadata=phrase.moss_serving_metadata,
        )


def _execution_checkpoint(execution: SynthesisExecutionContext) -> None:
    try:
        execution.checkpoint()
    except ExecutionDeadlineExceeded as exc:
        raise RenderBudgetExpired("accepted render request budget expired") from exc
    except ExecutionCancelled as exc:
        raise PhraseRenderFailed("render execution was cancelled") from exc


def _execution_remaining_seconds(execution: SynthesisExecutionContext) -> float:
    try:
        execution.raise_if_cancelled()
        remaining = execution.remaining_seconds()
    except ExecutionCancelled as exc:
        raise PhraseRenderFailed("render execution was cancelled") from exc
    if remaining <= 0:
        raise RenderBudgetExpired("accepted render request budget expired")
    return remaining

"""Scenario-aware rolling rap planning with frozen no-gap fallbacks."""

from __future__ import annotations

import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import TYPE_CHECKING, Callable

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.monitoring import RapEventPublisher
from streammuse.application.rap.monitoring_payloads import flow_template_payload, scheduled_syllables_payload
from streammuse.application.rap.scoring import rank_candidates
from streammuse.application.rap.service import CandidateGenerator, ProsodyAnalyzer
from streammuse.domain.rap import (
    AlignedLine,
    CandidateBatch,
    CandidateRequest,
    FlowTemplate,
    ProsodyAnalysis,
    RapEventType,
    RapScenario,
    ScenarioSegment,
    ScheduledSyllable,
    ScoreWeights,
    SelectionResult,
    materialize_flow,
)
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.templates import TemplateCatalog

if TYPE_CHECKING:
    from streammuse.application.rap.audio_coordination import BarAudioCoordinator
    from streammuse.domain.rap import PreparedRapBar


@dataclass
class PlannedRapBar:
    bar: int
    segment: ScenarioSegment
    template: FlowTemplate
    analysis: ProsodyAnalysis
    scheduled: tuple[ScheduledSyllable, ...]
    text: str
    source: str
    fallback_reason: str | None
    request_id: str | None = None
    frozen: bool = False


@dataclass(frozen=True)
class _PlanningResult:
    request: CandidateRequest
    batch: CandidateBatch
    selection: SelectionResult
    response_completed_monotonic: float
    decision_completed_monotonic: float


class RollingRapController:
    """Reserve fallback bars synchronously and replace only before freeze."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        scenario: RapScenario,
        templates: TemplateCatalog,
        fallback_catalog: PrevalidatedFallbackCatalog,
        analyzer: ProsodyAnalyzer,
        weights: ScoreWeights,
        publisher: RapEventPublisher | None,
        primary_generator: CandidateGenerator | None,
        candidate_count: int,
        lookahead_bars: int,
        minimum_score: float,
        seed: int,
        planning_bar_limit: int | None = None,
        emit: Callable[[ScheduledSyllable], None] | None = None,
        stop_primary: Callable[[], None] | None = None,
        close_primary: Callable[[], None] | None = None,
        audio_coordinator: BarAudioCoordinator | None = None,
        on_audio_committed: Callable[[PreparedRapBar], None] | None = None,
        executor: Executor | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if candidate_count <= 0 or lookahead_bars <= 0:
            raise ValueError("candidate_count and lookahead_bars must be positive")
        if planning_bar_limit is not None and planning_bar_limit <= 0:
            raise ValueError("planning_bar_limit must be positive or None")
        if tempo.ticks_per_beat != 4 or tempo.beats_per_bar != 4:
            raise ValueError("rap showcase requires four ticks per beat and four beats per bar")
        self._tempo = tempo
        self._scenario = scenario
        self._templates = templates
        self._fallback_catalog = fallback_catalog
        self._analyzer = analyzer
        self._weights = weights
        self._publisher = publisher
        self._primary_generator = primary_generator
        self._candidate_count = candidate_count
        self._lookahead_bars = lookahead_bars
        self._minimum_score = minimum_score
        self._seed = seed
        self._planning_bar_limit = planning_bar_limit
        self._emit = emit
        self._stop_primary = stop_primary
        self._close_primary = close_primary
        self._audio_coordinator = audio_coordinator
        self._on_audio_committed = on_audio_committed
        self._executor = executor or (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammuse-rap-planner")
            if primary_generator is not None
            else None
        )
        self._clock = monotonic
        self._lock = RLock()
        # Serialize lifecycle changes without holding the state lock while
        # resetting or closing the coordinator's independently locked state.
        self._lifecycle_lock = RLock()
        self._bars: dict[int, PlannedRapBar] = {}
        self._frozen_history: list[ProsodyAnalysis] = []
        self._rhyme_anchors: dict[tuple[int, str], tuple[str, ...]] = {}
        self._future: Future[_PlanningResult] | None = None
        self._future_bar: int | None = None
        self._next_primary_bar = 1
        self._audio_primary_plans: dict[int, PlannedRapBar] = {}
        self._audio_committed_bars: set[int] = set()
        self._last_tick = -1
        self._clock_origin: float | None = None
        self._lifecycle_epoch = 0
        self._started = False
        self._closed = False

    @property
    def scenario(self) -> RapScenario:
        return self._scenario

    def start(self) -> None:
        committed = None
        delivery_epoch = None
        with self._lifecycle_lock:
            with self._lock:
                if self._started or self._closed:
                    return
                self._started = True
                startup_last_bar = self._bounded_last_bar(self._lookahead_bars - 1)
                self._reserve_through(startup_last_bar)
                if self._audio_coordinator is not None:
                    committed = self._commit_audio_bar(0, tick=None)
                    delivery_epoch = self._lifecycle_epoch
                self._submit_next_primary(current_bar=0)
        self._notify_audio_committed(committed, delivery_epoch)

    def on_tick(self, tick: int) -> None:
        committed = None
        delivery_epoch = None
        with self._lock:
            if not self._started or self._closed:
                return
            self._last_tick = tick
            now = self._clock()
            if self._clock_origin is None:
                self._clock_origin = now - self._tempo.tick_to_seconds(tick)
            current_bar = tick // self._tempo.ticks_per_bar
            self._reserve_through(self._planning_ceiling(current_bar))
            self._drain_primary_result()
            if self._audio_coordinator is not None:
                self._drain_audio_primary_result()
                if tick % self._tempo.ticks_per_bar == self._tempo.ticks_per_bar - 1:
                    committed = self._commit_audio_bar(current_bar + 1, tick=tick)
                    delivery_epoch = self._lifecycle_epoch
            if tick % self._tempo.ticks_per_bar == 0:
                self._freeze(current_bar, tick)
            self._submit_next_primary(current_bar=current_bar)
            beat = (tick % self._tempo.ticks_per_bar) // self._tempo.ticks_per_beat
            tick_in_beat = tick % self._tempo.ticks_per_beat
            self._event(RapEventType.TICK, bar=current_bar, tick=tick, payload={"beat": beat, "tick_in_beat": tick_in_beat})
            scheduled = () if self._audio_coordinator is not None else tuple(
                item for item in self._bars[current_bar].scheduled if item.slot.tick == tick
            )
            assert not scheduled or self._bars[current_bar].frozen

        self._notify_audio_committed(committed, delivery_epoch)
        for item in scheduled:
            actual = self._clock()
            planned = (self._clock_origin or actual) + self._tempo.tick_to_seconds(tick)
            payload = {
                "word": item.syllable.word,
                "label": item.syllable.label,
                "stressed": item.syllable.stressed,
                "beat": item.slot.beat,
                "tick_in_beat": item.slot.tick_in_beat,
                "planned_monotonic": planned,
                "actual_monotonic": actual,
                "jitter_ms": (actual - planned) * 1000.0,
            }
            self._event(RapEventType.SYLLABLE_EMITTED, bar=item.slot.bar, tick=tick, payload=payload)
            if self._emit is not None:
                try:
                    self._emit(item)
                except Exception:
                    continue

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._lifecycle_epoch += 1
                future = self._future
                executor = self._executor
                stop_primary = self._stop_primary
                close_primary = self._close_primary
                audio_coordinator = self._audio_coordinator
            if future is not None:
                future.cancel()
            if stop_primary is not None:
                try:
                    stop_primary()
                except Exception as exc:
                    self._event(
                        RapEventType.GENERATION_FAILED,
                        payload={"error_type": "abort_error", "error_message": str(exc)},
                    )
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if close_primary is not None:
                try:
                    close_primary()
                except Exception as exc:
                    self._event(
                        RapEventType.GENERATION_FAILED,
                        payload={"error_type": "close_error", "error_message": str(exc)},
                    )
            if audio_coordinator is not None:
                audio_coordinator.close()

    def reset(self) -> None:
        """Clear an audio-controlled planning session after playback has stopped."""

        with self._lifecycle_lock:
            with self._lock:
                if self._audio_coordinator is None:
                    raise RuntimeError("reset is available only for audio-controlled sessions")
                if self._closed:
                    raise RuntimeError("cannot reset a closed controller")
                self._lifecycle_epoch += 1
                future = self._future
                self._future = None
                self._future_bar = None
                self._bars.clear()
                self._frozen_history.clear()
                self._rhyme_anchors.clear()
                self._audio_primary_plans.clear()
                self._audio_committed_bars.clear()
                self._next_primary_bar = 1
                self._last_tick = -1
                self._clock_origin = None
                self._started = False
                audio_coordinator = self._audio_coordinator
            if future is not None:
                future.cancel()
            audio_coordinator.reset()

    def bar_for(self, index: int) -> PlannedRapBar:
        with self._lock:
            return self._bars[index]

    def bar_state(self, index: int) -> str:
        with self._lock:
            bar = self._bars.get(index)
            return "unreserved" if bar is None else "frozen" if bar.frozen else "reserved"

    def line_for_bar(self, bar: int) -> AlignedLine:
        item = self.bar_for(bar)
        return AlignedLine(item.text, item.analysis.syllables, item.scheduled, score=0.0)

    def line_source_for_bar(self, bar: int) -> str:
        return self.bar_for(bar).source

    def _reserve_through(self, last_bar: int) -> None:
        for bar in range(last_bar + 1):
            if bar in self._bars:
                continue
            segment = self._scenario.segment_for_bar(bar)
            template = self._templates.get(segment.template_id)
            request = self._request_for_bar(bar, template)
            fallback = self._fallback_catalog.line_for(request)
            reason = "initial_bar" if bar == 0 else "no_primary_generator" if self._primary_generator is None else "generation_pending"
            self._bars[bar] = PlannedRapBar(
                bar=bar,
                segment=segment,
                template=template,
                analysis=fallback.analysis,
                scheduled=align_exact(fallback.analysis, materialize_flow(template, bar)),
                text=fallback.text,
                source=fallback.source,
                fallback_reason=reason,
            )
            if self._audio_coordinator is not None:
                self._audio_coordinator.reserve_fallback(replace(self._bars[bar]))
            self._event(
                RapEventType.BAR_RESERVED,
                bar=bar,
                tick=self._last_tick if self._last_tick >= 0 else None,
                payload={
                    "source": fallback.source,
                    "text": fallback.text,
                    "topic": segment.topic,
                    "template_id": template.template_id,
                    "flow": flow_template_payload(template),
                    "fallback": True,
                    "fallback_reason": reason,
                },
            )

    def _submit_next_primary(self, *, current_bar: int) -> None:
        if self._primary_generator is None or self._executor is None or self._future is not None:
            return
        target_bar = max(self._next_primary_bar, current_bar + 1)
        if target_bar > self._planning_ceiling(current_bar):
            return
        self._reserve_through(target_bar)
        target = self._bars[target_bar]
        if target.frozen:
            self._next_primary_bar = target_bar + 1
            return
        request = self._request_for_bar(target_bar, target.template)
        target.request_id = request.request_id
        history = tuple(self._frozen_history)
        anchors = dict(self._rhyme_anchors)
        self._event(
            RapEventType.BAR_PLANNING_STARTED,
            bar=target_bar,
            tick=self._last_tick if self._last_tick >= 0 else None,
            request_id=request.request_id,
            payload={
                "topic": target.segment.topic,
                "template_id": target.template.template_id,
                "required_syllables": request.required_syllables,
                "candidate_count": request.count,
                "context_lines": list(request.context_lines),
                "seed": request.seed,
                "flow": flow_template_payload(request.flow_template),
            },
        )
        self._future = self._executor.submit(self._generate_and_rank, request, target.segment, history, anchors)
        self._future_bar = target_bar
        self._next_primary_bar = target_bar + 1

    def _generate_and_rank(
        self,
        request: CandidateRequest,
        segment: ScenarioSegment,
        history: tuple[ProsodyAnalysis, ...],
        anchors: dict[tuple[int, str], tuple[str, ...]],
    ) -> _PlanningResult:
        assert self._primary_generator is not None
        batch = self._primary_generator.generate(request)
        response_completed_monotonic = self._clock()
        candidates = tuple(
            (f"{request.request_id}-candidate-{index + 1}", text, self._analyzer.analyze(text))
            for index, text in enumerate(batch.candidates)
        )
        selection = rank_candidates(
            candidates,
            template=request.flow_template,
            topic=segment.topic,
            history=history,
            rhyme_anchors=anchors,
            weights=self._weights,
            minimum_score=self._minimum_score,
            segment_start_bar=segment.start_bar,
            target_bar=request.target_bar,
        )
        return _PlanningResult(
            request,
            batch,
            selection,
            response_completed_monotonic=response_completed_monotonic,
            decision_completed_monotonic=self._clock(),
        )

    def _drain_primary_result(self) -> None:
        future = self._future
        target_bar = self._future_bar
        if future is None or target_bar is None or not future.done():
            return
        self._future = None
        self._future_bar = None
        try:
            result = future.result()
        except Exception as exc:
            target = self._bars[target_bar]
            late = target.frozen or target_bar * self._tempo.ticks_per_bar <= self._last_tick
            if not late:
                target.fallback_reason = "generation_error"
            self._event(
                RapEventType.GENERATION_FAILED,
                bar=target_bar,
                tick=self._last_tick,
                request_id=target.request_id,
                payload={"error_type": "generation_error", "error_message": str(exc), "late": late},
            )
            return

        target = self._bars[target_bar]
        batch = result.batch
        deadline_slack_ms = self._deadline_slack_ms(target_bar, result.response_completed_monotonic)
        decision_deadline_slack_ms = self._deadline_slack_ms(target_bar, result.decision_completed_monotonic)
        late = target.frozen or (deadline_slack_ms is not None and deadline_slack_ms <= 0.0)
        decision_late = target.frozen or (
            decision_deadline_slack_ms is not None and decision_deadline_slack_ms <= 0.0
        )
        self._event(
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            bar=target_bar,
            tick=self._last_tick,
            request_id=result.request.request_id,
            payload={
                "source": batch.source,
                "candidate_count": len(batch.candidates),
                "prompt": list(batch.prompt_json),
                "raw_response": batch.raw_response,
                "latency_ms": batch.latency_ms,
                "prompt_tokens": batch.prompt_tokens,
                "completion_tokens": batch.completion_tokens,
                "warning": batch.warning,
                "error_type": batch.error_type,
                "error_message": batch.error_message,
                "late": late,
                "deadline_slack_ms": deadline_slack_ms,
                "response_completed_monotonic": result.response_completed_monotonic,
                "decision_completed_monotonic": result.decision_completed_monotonic,
                "decision_deadline_slack_ms": decision_deadline_slack_ms,
                "decision_late": decision_late,
            },
        )
        selected_id = result.selection.selected.candidate_id if result.selection.selected else None
        for evaluation in result.selection.evaluations:
            word_sources = []
            seen_words = set()
            for syllable in evaluation.analysis.syllables:
                if syllable.word in seen_words:
                    continue
                seen_words.add(syllable.word)
                word_sources.append({"word": syllable.word, "source": syllable.analysis_source})
            self._event(
                RapEventType.CANDIDATE_EVALUATED,
                bar=target_bar,
                tick=self._last_tick,
                request_id=result.request.request_id,
                payload={
                    "candidate_id": evaluation.candidate_id,
                    "text": evaluation.text,
                    "normalized_text": evaluation.analysis.normalized_text,
                    "syllables": [asdict(item) for item in evaluation.analysis.syllables],
                    "word_analysis_sources": word_sources,
                    "oov_words": list(evaluation.analysis.oov_words),
                    "valid": evaluation.valid,
                    "rejection_reasons": list(evaluation.rejection_reasons),
                    "components": [asdict(item) for item in evaluation.components],
                    "total_score": evaluation.total_score,
                    "selected": evaluation.candidate_id == selected_id,
                },
            )
        if batch.error_type:
            if not late:
                target.fallback_reason = batch.error_type
            self._event(
                RapEventType.GENERATION_FAILED,
                bar=target_bar,
                tick=self._last_tick,
                request_id=result.request.request_id,
                payload={
                    "error_type": batch.error_type,
                    "error_message": batch.error_message,
                    "late": late,
                },
            )
            return
        if decision_late:
            return
        selected = result.selection.selected
        if selected is None:
            reason = result.selection.fallback_reason or "no_valid_candidate"
            target.fallback_reason = "no_valid_candidate" if reason == "no_valid_candidates" else reason
            return
        primary_plan = replace(
            target,
            analysis=selected.analysis,
            scheduled=selected.scheduled,
            text=selected.text,
            source=batch.source,
            fallback_reason=None,
        )
        if self._audio_coordinator is not None:
            self._audio_primary_plans[target_bar] = primary_plan
            self._audio_coordinator.submit_primary(replace(primary_plan))
            return
        self._apply_primary_plan(target, primary_plan, selected.candidate_id, selected.total_score)

    def _drain_audio_primary_result(self) -> None:
        assert self._audio_coordinator is not None
        for bar, primary_plan in tuple(self._audio_primary_plans.items()):
            prepared = self._audio_coordinator.poll_primary(bar)
            if prepared is None:
                continue
            target = self._bars.get(bar)
            if target is not None and not target.frozen and prepared.source == primary_plan.source:
                self._apply_primary_plan(target, primary_plan, candidate_id=None, total_score=None)
            self._audio_primary_plans.pop(bar, None)

    def _apply_primary_plan(
        self,
        target: PlannedRapBar,
        primary_plan: PlannedRapBar,
        candidate_id: str | None,
        total_score: float | None,
    ) -> None:
        previous_source = target.source
        target.analysis = primary_plan.analysis
        target.scheduled = primary_plan.scheduled
        target.text = primary_plan.text
        target.source = primary_plan.source
        target.fallback_reason = primary_plan.fallback_reason
        self._event(
            RapEventType.BAR_REPLACED,
            bar=target.bar,
            tick=self._last_tick,
            request_id=target.request_id,
            payload={
                "previous_source": previous_source,
                "source": target.source,
                "text": target.text,
                "candidate_id": candidate_id,
                "total_score": total_score,
                "flow": flow_template_payload(target.template),
                "scheduled_syllables": scheduled_syllables_payload(target.scheduled, bar=target.bar),
                "fallback": False,
                "fallback_reason": None,
            },
        )

    def _commit_audio_bar(self, bar_index: int, *, tick: int | None) -> PreparedRapBar | None:
        assert self._audio_coordinator is not None
        if bar_index in self._audio_committed_bars or bar_index not in self._bars:
            return None
        prepared = self._audio_coordinator.commit(bar_index)
        primary_plan = self._audio_primary_plans.pop(bar_index, None)
        target = self._bars[bar_index]
        if primary_plan is not None and prepared.source == primary_plan.source and not target.frozen:
            self._apply_primary_plan(target, primary_plan, candidate_id=None, total_score=None)
        self._audio_committed_bars.add(bar_index)
        deadline_slack_ms = self._tempo.tick_to_seconds(1) * 1000.0
        self._event(
            RapEventType.BAR_AUDIO_COMMITTED,
            bar=bar_index,
            tick=tick,
            request_id=target.request_id,
            payload={
                "source": prepared.source,
                "warnings": [warning.code.value for warning in prepared.warnings],
                "render_latency_ms": prepared.render_latency_ms,
                "frame_count": prepared.audio.frame_count,
                "deadline_slack_ms": deadline_slack_ms,
            },
        )
        return prepared

    def _notify_audio_committed(self, prepared: PreparedRapBar | None, epoch: int | None) -> None:
        if prepared is None or epoch is None:
            return
        # Keep delivery ordered against reset/close, but do not retain the
        # controller state lock while a playback callback may reenter us.
        with self._lifecycle_lock:
            with self._lock:
                if epoch != self._lifecycle_epoch or not self._started or self._closed:
                    return
                callback = self._on_audio_committed
            if callback is not None:
                callback(prepared)

    def _freeze(self, bar_index: int, tick: int) -> None:
        bar = self._bars[bar_index]
        if bar.frozen:
            return
        if self._audio_coordinator is not None:
            assert bar_index in self._audio_committed_bars
        if bar.source == "prevalidated_fallback" and bar.fallback_reason == "generation_pending":
            bar.fallback_reason = "deadline_miss"
        bar.frozen = True
        self._frozen_history.append(bar.analysis)
        rhyme_group = next((slot.rhyme_group for slot in reversed(bar.template.slots) if slot.rhyme_group), None)
        if rhyme_group and bar.analysis.end_rhyme_tail:
            self._rhyme_anchors.setdefault((bar.segment.start_bar, rhyme_group), bar.analysis.end_rhyme_tail)
        fallback = bar.source == "prevalidated_fallback"
        payload = {
            "text": bar.text,
            "source": bar.source,
            "fallback": fallback,
            "fallback_reason": bar.fallback_reason,
            "topic": bar.segment.topic,
            "template_id": bar.template.template_id,
            "flow": flow_template_payload(bar.template),
            "scheduled_syllables": scheduled_syllables_payload(bar.scheduled, bar=bar_index),
        }
        self._event(RapEventType.BAR_FROZEN, bar=bar_index, tick=tick, request_id=bar.request_id, payload=payload)
        if fallback:
            self._event(RapEventType.FALLBACK_ACTIVATED, bar=bar_index, tick=tick, request_id=bar.request_id, payload=payload)

    def _request_for_bar(self, bar: int, template: FlowTemplate) -> CandidateRequest:
        segment = self._scenario.segment_for_bar(bar)
        context = tuple(item.text for item in self._bars.values() if item.frozen)[-4:]
        return CandidateRequest(
            request_id=f"{self._scenario.scenario_id}-bar-{bar}-seed-{self._seed + bar}",
            target_bar=bar,
            topic=segment.topic,
            flow_template=template,
            count=self._candidate_count,
            context_lines=context,
            seed=self._seed + bar,
        )

    def _planning_ceiling(self, current_bar: int) -> int:
        return self._bounded_last_bar(current_bar + self._lookahead_bars)

    def _bounded_last_bar(self, requested: int) -> int:
        if self._planning_bar_limit is None:
            return requested
        return min(requested, self._planning_bar_limit - 1)

    def _deadline_slack_ms(self, bar: int, completed_monotonic: float) -> float | None:
        if self._clock_origin is None:
            return None
        deadline = self._clock_origin + self._tempo.tick_to_seconds(bar * self._tempo.ticks_per_bar)
        return (deadline - completed_monotonic) * 1000.0

    def _event(self, event_type: RapEventType, **kwargs: object) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, **kwargs)

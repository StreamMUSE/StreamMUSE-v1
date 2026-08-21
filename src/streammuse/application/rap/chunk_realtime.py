"""Rolling two-bar realtime preparation with immutable Mac deadlines."""

from __future__ import annotations

import time
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from math import isfinite
from threading import RLock
from typing import Callable, Mapping

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.audio_service import RapBarRenderer, RapChunkPreparationStrategy
from streammuse.application.rap.monitoring import RapEventPublisher
from streammuse.application.rap.monitoring_payloads import flow_template_payload, scheduled_syllables_payload
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.application.rap.service import ProsodyAnalyzer
from streammuse.domain.rap import (
    CandidateRequest,
    PreparedRapBar,
    PreparedRapChunk,
    RapEventType,
    RapScenario,
    RemoteCandidatePolicy,
    RemoteRapBarRequest,
    RemoteRapChunkRequest,
    materialize_flow,
)
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.templates import TemplateCatalog


@dataclass(frozen=True)
class _RemoteCompletion:
    chunk: PreparedRapChunk
    completed_monotonic: float


@dataclass
class _ChunkWork:
    epoch: int
    start_bar: int
    request: RemoteRapChunkRequest
    deadline_monotonic: float
    fallbacks: tuple[PreparedRapBar, PreparedRapBar]
    plans: tuple[PlannedRapBar, PlannedRapBar]
    future: Future[_RemoteCompletion]
    accepted: PreparedRapChunk | None = None
    rejection_state: str | None = None
    rejection_message: str | None = None
    deadline_slack_ms: float | None = None


class RollingRapChunkController:
    """Select and queue remote or ready local audio in exact two-bar units."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        scenario: RapScenario,
        templates: TemplateCatalog,
        fallback_catalog: PrevalidatedFallbackCatalog,
        analyzer: ProsodyAnalyzer,
        fallback_renderer: RapBarRenderer,
        preparation_strategy: RapChunkPreparationStrategy,
        publisher: RapEventPublisher | None,
        enqueue: Callable[[PreparedRapBar], None],
        session_id: str,
        policy: RemoteCandidatePolicy,
        seed: int,
        planning_bar_limit: int | None,
        startup_timeout_seconds: float,
        rolling_timeout_seconds: float,
        executor: Executor | None = None,
        preparation_executor: Executor | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if tempo.ticks_per_beat != 4 or tempo.beats_per_bar != 4:
            raise ValueError("remote rap chunks require four ticks per beat and four beats per bar")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(policy, RemoteCandidatePolicy):
            raise ValueError("policy must be a RemoteCandidatePolicy")
        for name, value in (
            ("startup_timeout_seconds", startup_timeout_seconds),
            ("rolling_timeout_seconds", rolling_timeout_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if planning_bar_limit is not None:
            if not isinstance(planning_bar_limit, int) or isinstance(planning_bar_limit, bool) or planning_bar_limit <= 0:
                raise ValueError("planning_bar_limit must be positive or None")
            if planning_bar_limit % 2:
                raise ValueError("finite remote planning_bar_limit must be zero or even")
        if not scenario.loop:
            planning_bar_limit = scenario.total_bars if planning_bar_limit is None else planning_bar_limit
            if planning_bar_limit % 2:
                raise ValueError("finite remote planning_bar_limit must be zero or even")
            if planning_bar_limit > scenario.total_bars:
                raise ValueError("planning_bar_limit exceeds the non-looping scenario")

        self._tempo = tempo
        self._scenario = scenario
        self._templates = templates
        self._fallback_catalog = fallback_catalog
        self._analyzer = analyzer
        self._fallback_renderer = fallback_renderer
        self._strategy = preparation_strategy
        self._publisher = publisher
        self._enqueue = enqueue
        self._session_id = session_id
        self._policy = policy
        self._seed = seed
        self._planning_bar_limit = planning_bar_limit
        self._startup_timeout = float(startup_timeout_seconds)
        self._rolling_timeout = float(rolling_timeout_seconds)
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammuse-rap-chunk")
        self._preparation_executor = preparation_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="streammuse-rap-fallback",
        )
        self._clock = monotonic
        self._lock = RLock()
        self._lifecycle_lock = RLock()
        self._epoch = 0
        self._started = False
        self._stopping = False
        self._stop_successor_bar: int | None = None
        self._closed = False
        self._clock_origin: float | None = None
        self._last_tick = -1
        self._pending: _ChunkWork | None = None
        self._submission: Future[_ChunkWork] | None = None
        self._committed: dict[int, PreparedRapBar] = {}
        self._plans: dict[int, PlannedRapBar] = {}
        self._context_lines: list[str] = []

    @property
    def scenario(self) -> RapScenario:
        return self._scenario

    def start(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or self._started:
                    return
                self._started = True
                self._stopping = False
                self._stop_successor_bar = None
                started = self._clock()
                work = self._build_work(
                    0,
                    started + self._startup_timeout,
                    self._epoch,
                    tuple(self._context_lines[-4:]),
                )
                self._pending = work
            selected = self._resolve(work, wait_seconds=max(0.0, work.deadline_monotonic - self._clock()), final=True)
            bars = self._commit(work, selected, tick=None, deliver=True)
            with self._lock:
                if self._clock_origin is None:
                    self._clock_origin = self._clock()
                self._submit_next_locked()
        self._deliver(bars)

    def on_tick(self, tick: int) -> None:
        bars: tuple[PreparedRapBar, PreparedRapBar] | None = None
        with self._lifecycle_lock:
            with self._lock:
                if not self._started or self._closed:
                    return
                self._last_tick = tick
                current_bar = tick // self._tempo.ticks_per_bar
                if self._stopping:
                    self._emit_tick(current_bar, tick)
                    return
                self._poll_submission_locked()
                work = self._pending
                if work is not None:
                    self._resolve(work, wait_seconds=None, final=False)
                    if tick == work.start_bar * self._tempo.ticks_per_bar - 1:
                        selected = self._resolve(work, wait_seconds=None, final=True)
                        bars = self._commit(work, selected, tick=tick, deliver=True)
                        self._schedule_next_locked()
                self._emit_tick(current_bar, tick)
        self._deliver(bars)

    def request_stop(self, *, successor_bar: int | None) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or not self._started or self._stopping:
                    return
                if successor_bar is not None and successor_bar < 0:
                    raise ValueError("stop requires a non-negative successor bar")
                self._stopping = True
                self._stop_successor_bar = successor_bar
                submission = self._submission
                self._submission = None
                if submission is not None:
                    submission.cancel()
                work = self._pending
                self._pending = None
                if work is not None:
                    work.future.cancel()
            if work is not None or submission is not None:
                self._strategy.abort()
            if work is not None:
                with self._lock:
                    self._pending = work
                    self._reject(work, "cancelled", "remote wait cancelled by stop")
                    if successor_bar is not None and successor_bar not in self._committed:
                        self._commit(work, None, tick=self._last_tick, deliver=False)
                    self._pending = None

    def resume_audio(self, bar: int) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or not self._started:
                    raise RuntimeError("cannot resume an inactive audio controller")
                if not self._stopping or self._stop_successor_bar is None:
                    raise RuntimeError("audio controller is not stopped for resume")
                if bar != self._stop_successor_bar or bar not in self._committed:
                    raise ValueError("audio resume requires the retained successor bar")
                prepared = self._committed[bar]
            self._enqueue(prepared)

    def resume_after_stop(self) -> None:
        trailing: list[PreparedRapBar] = []
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or not self._started or not self._stopping:
                    raise RuntimeError("audio controller is not stopped for resume")
                successor = self._stop_successor_bar
                if successor is None:
                    raise RuntimeError("terminal audio stop requires reset before resume")
                pair_end = successor if successor % 2 else successor + 1
                trailing = [self._committed[bar] for bar in range(successor + 1, pair_end + 1) if bar in self._committed]
                self._stopping = False
                self._stop_successor_bar = None
                self._submit_next_locked()
            for prepared in trailing:
                self._enqueue(prepared)

    def reset(self) -> int:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("cannot reset a closed controller")
                self._epoch += 1
                epoch = self._epoch
                submission = self._submission
                self._submission = None
                if submission is not None:
                    submission.cancel()
                work = self._pending
                self._pending = None
                self._started = False
                self._stopping = False
                self._stop_successor_bar = None
                self._clock_origin = None
                self._last_tick = -1
                self._committed.clear()
                self._plans.clear()
                self._context_lines.clear()
                if work is not None:
                    work.future.cancel()
            self._strategy.abort()
            return epoch

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._epoch += 1
                submission = self._submission
                self._submission = None
                if submission is not None:
                    submission.cancel()
                work = self._pending
                self._pending = None
                if work is not None:
                    work.future.cancel()
            self._strategy.abort()
            self._preparation_executor.shutdown(wait=True, cancel_futures=True)
            self._strategy.close()
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _submit_next_locked(self) -> None:
        if self._stopping or self._closed or self._pending is not None:
            return
        start_bar = max(self._committed, default=-1) + 1
        if start_bar % 2:
            start_bar += 1
        if not self._has_pair(start_bar):
            return
        now = self._clock()
        deadline = self._rolling_deadline(start_bar, now)
        self._pending = self._build_work(
            start_bar,
            deadline,
            self._epoch,
            tuple(self._context_lines[-4:]),
        )

    def _schedule_next_locked(self) -> None:
        if self._stopping or self._closed or self._pending is not None or self._submission is not None:
            return
        start_bar = max(self._committed, default=-1) + 1
        if start_bar % 2:
            start_bar += 1
        if not self._has_pair(start_bar):
            return
        now = self._clock()
        deadline = self._rolling_deadline(start_bar, now)
        self._submission = self._preparation_executor.submit(
            self._build_work,
            start_bar,
            deadline,
            self._epoch,
            tuple(self._context_lines[-4:]),
        )

    def _poll_submission_locked(self) -> None:
        submission = self._submission
        if submission is None or not submission.done():
            return
        self._submission = None
        try:
            work = submission.result()
        except CancelledError:
            return
        except BaseException as error:
            self._event(
                RapEventType.GENERATION_FAILED,
                tick=self._last_tick,
                payload={
                    "error_type": "fallback_preparation_error",
                    "error_message": f"{type(error).__name__}: {error}",
                },
            )
            return
        if work.epoch != self._epoch or self._stopping or self._closed:
            work.future.cancel()
            self._strategy.abort()
            return
        self._pending = work

    def _build_work(
        self,
        start_bar: int,
        deadline: float,
        epoch: int,
        context_lines: tuple[str, ...],
    ) -> _ChunkWork:
        self._require_active_epoch(epoch)
        if not self._has_pair(start_bar):
            raise ValueError("remote chunk requires two available bars")
        plans_list = []
        fallbacks_list = []
        for bar in (start_bar, start_bar + 1):
            self._require_active_epoch(epoch)
            plan = self._fallback_plan(bar, context_lines, epoch)
            plans_list.append(plan)
            fallbacks_list.append(self._render_fallback(plan, epoch))
        plans = (plans_list[0], plans_list[1])
        fallbacks = (fallbacks_list[0], fallbacks_list[1])
        self._require_active_epoch(epoch)
        request = RemoteRapChunkRequest.create(
            session_id=self._session_id,
            chunk_index=start_bar // 2,
            bars=tuple(RemoteRapBarRequest(plan.bar, plan.segment.topic, plan.template) for plan in plans),
            tempo_bpm=self._tempo.bpm,
            remaining_budget_ms=max(1, int((deadline - self._clock()) * 1000.0)),
            policy=self._policy,
            context_lines=context_lines,
            seed=self._seed + start_bar // 2,
        )
        future = self._executor.submit(self._prepare_remote, request, deadline, epoch)
        work = _ChunkWork(epoch, start_bar, request, deadline, fallbacks, plans, future)
        self._event(
            RapEventType.CHUNK_REQUEST_SUBMITTED,
            bar=start_bar,
            request_id=request.request_id,
            payload=self._chunk_payload(work, state="requested", renderer_decision="pending"),
        )
        return work

    def _prepare_remote(self, request: RemoteRapChunkRequest, deadline: float, epoch: int) -> _RemoteCompletion:
        chunk = self._strategy.prepare(request, deadline_monotonic=deadline)
        return _RemoteCompletion(chunk, self._clock())

    def _resolve(self, work: _ChunkWork, *, wait_seconds: float | None, final: bool) -> PreparedRapChunk | None:
        if work.epoch != self._epoch or work is not self._pending:
            return None
        if work.accepted is not None:
            return work.accepted
        if work.rejection_state is not None:
            return None
        if wait_seconds is None and not work.future.done():
            if final:
                work.future.cancel()
                self._strategy.abort()
                slack_ms = (work.deadline_monotonic - self._clock()) * 1000.0
                self._reject(work, "deadline_miss", "remote chunk was not ready at commitment", slack_ms)
            return None
        try:
            completion = work.future.result(timeout=wait_seconds)
        except TimeoutError:
            work.future.cancel()
            self._strategy.abort()
            self._reject(work, "startup_timeout", "remote chunk missed startup timeout")
            return None
        except BaseException as error:
            self._reject(work, "failed", f"{type(error).__name__}: {error}")
            return None

        slack_ms = (work.deadline_monotonic - completion.completed_monotonic) * 1000.0
        work.deadline_slack_ms = slack_ms
        self._event(
            RapEventType.CHUNK_REMOTE_COMPLETED,
            bar=work.start_bar,
            request_id=work.request.request_id,
            payload=self._chunk_payload(
                work,
                state="returned",
                renderer_decision=completion.chunk.renderer,
                chunk=completion.chunk,
                deadline_slack_ms=slack_ms,
            ),
        )
        if completion.completed_monotonic >= work.deadline_monotonic:
            self._reject(work, "late", "remote chunk completed at or after its immutable deadline", slack_ms)
            return None
        invalid = self._validation_error(work, completion.chunk)
        if invalid is not None:
            self._reject(work, "invalid", invalid, slack_ms)
            return None
        work.accepted = completion.chunk
        for prepared in completion.chunk.bars:
            self._event(
                RapEventType.BAR_AUDIO_READY,
                bar=prepared.bar,
                request_id=work.request.request_id,
                payload=self._audio_payload(prepared),
            )
        return completion.chunk

    def _validation_error(self, work: _ChunkWork, chunk: object) -> str | None:
        if not isinstance(chunk, PreparedRapChunk):
            return "remote strategy returned an invalid chunk value"
        if chunk.request_id != work.request.request_id or chunk.chunk_index != work.request.chunk_index:
            return "remote chunk identity does not match its request"
        if chunk.renderer != "moss_aligned_remote":
            return "remote chunk renderer is incompatible"
        if tuple(item.bar for item in chunk.bars) != (work.start_bar, work.start_bar + 1):
            return "remote chunk bars do not match its request"
        if any(item.source != "moss_aligned_remote" for item in chunk.bars):
            return "remote prepared bars have incompatible provenance"
        return None

    def _reject(self, work: _ChunkWork, state: str, message: str, slack_ms: float | None = None) -> None:
        if work.rejection_state is not None:
            return
        work.rejection_state = state
        work.rejection_message = message
        work.deadline_slack_ms = slack_ms
        self._event(
            RapEventType.CHUNK_REMOTE_REJECTED,
            bar=work.start_bar,
            request_id=work.request.request_id,
            payload=self._chunk_payload(
                work,
                state=state,
                renderer_decision="prevalidated_fallback",
                deadline_slack_ms=slack_ms,
                warnings=(message,),
            ),
        )

    def _commit(
        self,
        work: _ChunkWork,
        selected: PreparedRapChunk | None,
        *,
        tick: int | None,
        deliver: bool,
    ) -> tuple[PreparedRapBar, PreparedRapBar] | None:
        if work.epoch != self._epoch:
            return None
        fallback = selected is None
        reason = work.rejection_state or "remote_unavailable"
        bars = selected.bars if selected is not None else tuple(replace(item, fallback_reason=reason) for item in work.fallbacks)
        renderer = selected.renderer if selected is not None else "prevalidated_fallback"
        for prepared, plan in zip(bars, work.plans, strict=True):
            self._committed[prepared.bar] = prepared
            self._plans[prepared.bar] = replace(
                plan,
                text=prepared.text,
                source=prepared.source,
                fallback_reason=prepared.fallback_reason,
                scheduled=prepared.scheduled,
                request_id=work.request.request_id,
                frozen=True,
            )
            payload = {
                "text": prepared.text,
                "source": prepared.source,
                "fallback": fallback,
                "fallback_reason": prepared.fallback_reason,
                "topic": plan.segment.topic,
                "template_id": plan.template.template_id,
                "flow": flow_template_payload(plan.template),
                "scheduled_syllables": scheduled_syllables_payload(prepared.scheduled, bar=prepared.bar),
            }
            self._event(RapEventType.BAR_FROZEN, bar=prepared.bar, tick=tick, request_id=work.request.request_id, payload=payload)
            if fallback:
                self._event(RapEventType.FALLBACK_ACTIVATED, bar=prepared.bar, tick=tick, request_id=work.request.request_id, payload=payload)
            self._event(
                RapEventType.BAR_AUDIO_COMMITTED,
                bar=prepared.bar,
                tick=tick,
                request_id=work.request.request_id,
                payload={**self._audio_payload(prepared), "coordinator_epoch": self._epoch},
            )
        self._context_lines.extend(item.text for item in bars)
        self._pending = None
        payload = self._chunk_payload(
            work,
            state="fallback" if fallback else "committed",
            renderer_decision=renderer,
            chunk=selected,
            selected_lines=tuple(item.text for item in bars),
            deadline_slack_ms=work.deadline_slack_ms,
            warnings=(work.rejection_message,) if work.rejection_message else (),
        )
        if fallback:
            self._event(RapEventType.CHUNK_FALLBACK_ACTIVATED, bar=work.start_bar, tick=tick, request_id=work.request.request_id, payload=payload)
        self._event(RapEventType.CHUNK_COMMITTED, bar=work.start_bar, tick=tick, request_id=work.request.request_id, payload=payload)
        return bars if deliver else None

    def _fallback_plan(self, bar: int, context_lines: tuple[str, ...], epoch: int) -> PlannedRapBar:
        segment = self._scenario.segment_for_bar(bar)
        template = self._templates.get(segment.template_id)
        request = CandidateRequest(
            request_id=f"{self._scenario.scenario_id}-bar-{bar}-fallback",
            target_bar=bar,
            topic=segment.topic,
            flow_template=template,
            count=1,
            context_lines=context_lines,
            seed=self._seed + bar,
        )
        fallback = self._fallback_catalog.line_for(request)
        plan = PlannedRapBar(
            bar=bar,
            segment=segment,
            template=template,
            analysis=fallback.analysis,
            scheduled=align_exact(fallback.analysis, materialize_flow(template, bar)),
            text=fallback.text,
            source=fallback.source,
            fallback_reason="remote_pending",
        )
        with self._lock:
            self._require_active_epoch(epoch)
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
                    "fallback_reason": "remote_pending",
                },
            )
        return plan

    def _render_fallback(self, plan: PlannedRapBar, epoch: int) -> PreparedRapBar:
        with self._lock:
            self._require_active_epoch(epoch)
            self._event(RapEventType.AUDIO_RENDER_STARTED, bar=plan.bar, payload={"source": plan.source})
        prepared = self._fallback_renderer.render(plan)
        if prepared.bar != plan.bar:
            raise ValueError("fallback renderer returned the wrong bar")
        with self._lock:
            self._require_active_epoch(epoch)
            self._event(RapEventType.AUDIO_RENDER_COMPLETED, bar=plan.bar, payload=self._audio_payload(prepared))
            self._event(RapEventType.BAR_AUDIO_READY, bar=plan.bar, payload=self._audio_payload(prepared))
        return prepared

    def _require_active_epoch(self, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch or self._closed or self._stopping:
                raise CancelledError

    def _has_pair(self, start_bar: int) -> bool:
        if start_bar < 0 or start_bar % 2:
            return False
        if self._planning_bar_limit is not None and start_bar + 1 >= self._planning_bar_limit:
            return False
        if not self._scenario.loop and start_bar + 1 >= self._scenario.total_bars:
            return False
        return True

    def _rolling_deadline(self, start_bar: int, now: float) -> float:
        if self._clock_origin is None:
            playback_deadline = now + self._rolling_timeout
        else:
            commit_tick = start_bar * self._tempo.ticks_per_bar - 1
            playback_deadline = self._clock_origin + self._tempo.tick_to_seconds(commit_tick)
        return min(now + self._rolling_timeout, playback_deadline)

    def _deliver(self, bars: tuple[PreparedRapBar, PreparedRapBar] | None) -> None:
        if bars is not None:
            for prepared in bars:
                self._enqueue(prepared)

    def _emit_tick(self, current_bar: int, tick: int) -> None:
        beat = (tick % self._tempo.ticks_per_bar) // self._tempo.ticks_per_beat
        tick_in_beat = tick % self._tempo.ticks_per_beat
        self._event(RapEventType.TICK, bar=current_bar, tick=tick, payload={"beat": beat, "tick_in_beat": tick_in_beat})

    @staticmethod
    def _audio_payload(prepared: PreparedRapBar) -> dict[str, object]:
        return {
            "source": prepared.source,
            "warnings": [warning.code.value for warning in prepared.warnings],
            "render_latency_ms": prepared.render_latency_ms,
            "frame_count": prepared.audio.frame_count,
        }

    def _chunk_payload(
        self,
        work: _ChunkWork,
        *,
        state: str,
        renderer_decision: str,
        chunk: PreparedRapChunk | None = None,
        selected_lines: tuple[str, str] | tuple[()] | None = None,
        deadline_slack_ms: float | None = None,
        warnings: tuple[str, ...] = (),
    ) -> dict[str, object]:
        selected_lines = selected_lines if selected_lines is not None else (() if chunk is None else tuple(item.text for item in chunk.bars))
        diagnostics = chunk.diagnostics if chunk is not None else {}
        diagnostic_warnings = diagnostics.get("warnings", ()) if isinstance(diagnostics, Mapping) else ()
        all_warnings = tuple(str(item) for item in (*warnings, *tuple(diagnostic_warnings)[:8]))
        return {
            "state": state,
            "renderer_decision": renderer_decision,
            "chunk_index": work.request.chunk_index,
            "bars": [work.start_bar, work.start_bar + 1],
            "selected_lines": list(selected_lines),
            "flows": [flow_template_payload(item.template) for item in work.plans],
            "stage_timings_ms": self._stage_timings(diagnostics),
            "deadline_slack_ms": deadline_slack_ms,
            "warnings": list(all_warnings[:8]),
        }

    @staticmethod
    def _stage_timings(diagnostics: Mapping[str, object]) -> dict[str, float]:
        value = diagnostics.get("stage_timings_ms")
        if not isinstance(value, Mapping):
            manifest = diagnostics.get("manifest")
            if isinstance(manifest, Mapping):
                remote_diagnostics = manifest.get("diagnostics")
                if isinstance(remote_diagnostics, Mapping):
                    value = remote_diagnostics.get("stage_timings_ms")
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key): float(item)
            for key, item in tuple(value.items())[:8]
            if isinstance(item, (int, float)) and not isinstance(item, bool) and isfinite(item)
        }

    def _event(self, event_type: RapEventType, **kwargs: object) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, **kwargs)

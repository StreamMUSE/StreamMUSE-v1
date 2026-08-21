"""Rolling two-bar realtime preparation with immutable Mac deadlines."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, replace
from math import isfinite
from threading import RLock
from typing import Callable, Mapping

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.audio_service import RapBarRenderer, RapChunkPreparationStrategy
from streammuse.application.rap.chunk_audio import RemoteChunkResponseRejected
from streammuse.application.rap.monitoring import RapEventPublisher
from streammuse.application.rap.monitoring_payloads import (
    bounded_chunk_event_payload,
    flow_template_payload,
    remote_generation_input_summary,
    scheduled_syllables_payload,
)
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


_MAX_EVENT_WARNING_BYTES = 256
_MAX_EVENT_NAME_BYTES = 64


@dataclass(frozen=True)
class _RemoteCompletion:
    chunk: object | None
    completed_monotonic: float
    rejection: RemoteChunkResponseRejected | None = None


@dataclass(frozen=True)
class _StagedChunk:
    epoch: int
    start_bar: int
    deadline_monotonic: float
    context_lines: tuple[str, ...]
    fallbacks: tuple[PreparedRapBar, PreparedRapBar]
    plans: tuple[PlannedRapBar, PlannedRapBar]


@dataclass
class _StagingTask:
    epoch: int
    start_bar: int
    deadline_monotonic: float
    context_lines: tuple[str, ...]
    future: Future[_StagedChunk] | None = None
    staged: _StagedChunk | None = None
    cancelled: bool = False
    retain_for_stop: bool = False
    failure_reported: bool = False


@dataclass
class _ChunkWork:
    epoch: int
    start_bar: int
    request: RemoteRapChunkRequest
    deadline_monotonic: float
    fallbacks: tuple[PreparedRapBar, PreparedRapBar]
    plans: tuple[PlannedRapBar, PlannedRapBar]
    future: Future[_RemoteCompletion] | None = None
    remote_authorized: bool = False
    remote_started: bool = False
    accepted: PreparedRapChunk | None = None
    observed: PreparedRapChunk | None = None
    returned_diagnostics: Mapping[str, object] | None = None
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
        cancellation_executor: Executor | None = None,
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
        self._cancellation_executor = cancellation_executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="streammuse-rap-cancel",
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
        self._submission: _StagingTask | None = None
        self._abort_barrier: Future[object] | None = None
        self._committed: dict[int, PreparedRapBar] = {}
        self._plans: dict[int, PlannedRapBar] = {}
        self._context_lines: list[str] = []

    @property
    def scenario(self) -> RapScenario:
        return self._scenario

    @property
    def terminal_bar_limit(self) -> int | None:
        """Return the finite remote planning boundary, if this run has one."""

        return self._planning_bar_limit

    def start(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or self._started:
                    return
                self._started = True
                self._stopping = False
                self._stop_successor_bar = None
                epoch = self._epoch
                deadline = self._clock() + self._startup_timeout
                task = _StagingTask(epoch, 0, deadline, tuple(self._context_lines[-4:]))

            staged = self._stage_pair(task)
            with self._lock:
                self._require_staging_active_locked(task)
                work = self._activate_staged_locked(staged)

            selected, cancel_remote = self._wait_for_startup(work)
            with self._lock:
                bars = self._commit(work, selected, tick=None, deliver=True)

            self._deliver(bars)
            with self._lock:
                if work.epoch != self._epoch or self._closed:
                    return
                self._clock_origin = self._clock()

            abort_future = self._dispatch_abort(work) if cancel_remote else None
            with self._lock:
                self._install_abort_barrier_locked(work.epoch, abort_future)
                self._schedule_next_locked()

    def on_tick(self, tick: int) -> None:
        bars: tuple[PreparedRapBar, PreparedRapBar] | None = None
        cancel_work: _ChunkWork | None = None
        with self._lifecycle_lock:
            with self._lock:
                if not self._started or self._closed:
                    return
                self._last_tick = tick
                current_bar = tick // self._tempo.ticks_per_bar
                if self._stopping:
                    self._emit_tick(current_bar, tick)
                    return

                bars = self._poll_staging_locked(tick)
                work = self._pending
                if bars is None and work is not None:
                    self._resolve_ready_locked(work)
                    guard_tick = work.start_bar * self._tempo.ticks_per_bar - 1
                    if tick >= guard_tick:
                        selected, cancel_remote = self._final_selection_locked(work)
                        bars = self._commit(work, selected, tick=tick, deliver=True)
                        if cancel_remote:
                            cancel_work = work
                self._emit_tick(current_bar, tick)

            self._deliver(bars)
            abort_future = self._dispatch_abort(cancel_work) if cancel_work is not None else None
            with self._lock:
                if cancel_work is not None:
                    self._install_abort_barrier_locked(cancel_work.epoch, abort_future)
                if bars is not None:
                    self._schedule_next_locked()

    def request_stop(self, *, successor_bar: int | None) -> None:
        abort_remote = False
        barrier: Future[object] | None = None
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or not self._started or self._stopping:
                    return
                if successor_bar is not None and successor_bar < 0:
                    raise ValueError("stop requires a non-negative successor bar")
                self._stopping = True
                self._stop_successor_bar = successor_bar

                task = self._submission
                if task is not None:
                    retain = successor_bar is not None and task.start_bar <= successor_bar <= task.start_bar + 1
                    task.retain_for_stop = retain
                    task.cancelled = not retain
                    if not retain:
                        self._submission = None
                        if task.future is not None:
                            task.future.cancel()
                    elif task.future is not None and task.future.done():
                        self._finish_staging_locked(task)

                work = self._pending
                self._pending = None
                if work is not None:
                    abort_remote = self._revoke_remote_locked(work)
                    self._reject(work, "cancelled", "remote wait cancelled by stop")
                    if successor_bar is not None and successor_bar not in self._committed:
                        if work.start_bar <= successor_bar <= work.start_bar + 1:
                            self._commit(work, None, tick=self._last_tick, deliver=False)
                barrier = self._abort_barrier
                self._abort_barrier = None

            self._wait_for_abort_barrier(barrier)
            if abort_remote and work is not None:
                self._abort_strategy(work)

    def resume_audio(self, bar: int) -> None:
        task: _StagingTask | None = None
        with self._lock:
            if self._stopping and bar == self._stop_successor_bar and bar not in self._committed:
                candidate = self._submission
                if candidate is not None and candidate.retain_for_stop and candidate.start_bar <= bar <= candidate.start_bar + 1:
                    task = candidate
        if task is not None and task.future is not None:
            try:
                task.future.result()
            except BaseException:
                pass
            with self._lock:
                self._finish_staging_locked(task)

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
        with self._lifecycle_lock:
            with self._lock:
                if self._closed or not self._started or not self._stopping:
                    raise RuntimeError("audio controller is not stopped for resume")
                successor = self._stop_successor_bar
                if successor is None:
                    raise RuntimeError("terminal audio stop requires reset before resume")
                pair_end = successor if successor % 2 else successor + 1
                trailing = tuple(
                    self._committed[bar]
                    for bar in range(successor + 1, pair_end + 1)
                    if bar in self._committed
                )
                self._stopping = False
                self._stop_successor_bar = None

            for prepared in trailing:
                self._enqueue(prepared)

            with self._lock:
                successor_tick = successor * self._tempo.ticks_per_bar
                self._clock_origin = self._clock() - self._tempo.tick_to_seconds(successor_tick)
                self._schedule_next_locked()

    def reset(self) -> int:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("cannot reset a closed controller")
                self._epoch += 1
                epoch = self._epoch
                task = self._submission
                self._submission = None
                if task is not None:
                    task.cancelled = True
                    if task.future is not None:
                        task.future.cancel()
                work = self._pending
                self._pending = None
                if work is not None:
                    self._revoke_remote_locked(work)
                barrier = self._abort_barrier
                self._abort_barrier = None
                self._started = False
                self._stopping = False
                self._stop_successor_bar = None
                self._clock_origin = None
                self._last_tick = -1
                self._committed.clear()
                self._plans.clear()
                self._context_lines.clear()

            self._wait_for_abort_barrier(barrier)
            if work is None:
                self._strategy.abort()
            else:
                self._abort_strategy(work)
            return epoch

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._epoch += 1
                task = self._submission
                self._submission = None
                if task is not None:
                    task.cancelled = True
                    if task.future is not None:
                        task.future.cancel()
                work = self._pending
                self._pending = None
                if work is not None:
                    self._revoke_remote_locked(work)
                self._abort_barrier = None

            self._cancellation_executor.shutdown(wait=True, cancel_futures=False)
            self._strategy.abort()
            self._preparation_executor.shutdown(wait=True, cancel_futures=True)
            self._strategy.close()
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _stage_pair(self, task: _StagingTask) -> _StagedChunk:
        with self._lock:
            self._require_staging_active_locked(task)
        if not self._has_pair(task.start_bar):
            raise ValueError("remote chunk requires two available bars")
        plans_list: list[PlannedRapBar] = []
        fallbacks_list: list[PreparedRapBar] = []
        for bar in (task.start_bar, task.start_bar + 1):
            with self._lock:
                self._require_staging_active_locked(task)
            plan = self._fallback_plan(bar, task.context_lines, task)
            plans_list.append(plan)
            fallbacks_list.append(self._render_fallback(plan, task))
        with self._lock:
            self._require_staging_active_locked(task)
        return _StagedChunk(
            task.epoch,
            task.start_bar,
            task.deadline_monotonic,
            task.context_lines,
            (fallbacks_list[0], fallbacks_list[1]),
            (plans_list[0], plans_list[1]),
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
        task = _StagingTask(
            self._epoch,
            start_bar,
            self._rolling_deadline(start_bar, now),
            tuple(self._context_lines[-4:]),
        )
        self._submission = task
        try:
            future = self._preparation_executor.submit(self._stage_pair, task)
        except BaseException as error:
            self._submission = None
            self._fallback_preparation_failed_locked(task, error)
            return
        task.future = future
        future.add_done_callback(lambda _future, task=task: self._staging_completed(task))

    def _staging_completed(self, task: _StagingTask) -> None:
        with self._lock:
            self._finish_staging_locked(task)

    def _finish_staging_locked(self, task: _StagingTask) -> None:
        if task is not self._submission or task.future is None or not task.future.done():
            return
        if task.epoch != self._epoch or self._closed or task.cancelled:
            self._submission = None
            return
        if task.staged is None:
            try:
                task.staged = task.future.result()
            except CancelledError:
                self._submission = None
                return
            except BaseException as error:
                self._submission = None
                self._fallback_preparation_failed_locked(task, error)
                return

        if self._stopping:
            successor = self._stop_successor_bar
            if task.retain_for_stop and successor is not None and task.start_bar <= successor <= task.start_bar + 1:
                work = self._work_from_staged_locked(task.staged)
                self._reject(work, "cancelled", "remote submission cancelled by stop")
                self._submission = None
                self._commit(work, None, tick=self._last_tick, deliver=False)
            else:
                self._submission = None
            return

        barrier = self._abort_barrier
        if barrier is not None and not barrier.done():
            return
        if barrier is not None:
            self._abort_barrier = None
        if self._clock() >= task.deadline_monotonic:
            return
        staged = task.staged
        self._submission = None
        self._activate_staged_locked(staged)

    def _poll_staging_locked(self, tick: int) -> tuple[PreparedRapBar, PreparedRapBar] | None:
        task = self._submission
        if task is None:
            return None
        self._finish_staging_locked(task)
        if task is not self._submission or task.staged is None:
            return None
        guard_tick = task.start_bar * self._tempo.ticks_per_bar - 1
        if tick < guard_tick:
            return None
        staged = task.staged
        self._submission = None
        work = self._work_from_staged_locked(staged)
        slack_ms = (work.deadline_monotonic - self._clock()) * 1000.0
        self._reject(work, "deadline_miss", "local fallback became ready at or after commitment", slack_ms)
        return self._commit(work, None, tick=tick, deliver=True)

    def _activate_staged_locked(self, staged: _StagedChunk) -> _ChunkWork:
        if staged.epoch != self._epoch or self._closed or self._stopping:
            raise CancelledError
        if self._pending is not None:
            raise RuntimeError("remote chunk activation requires an empty pending slot")
        work = self._work_from_staged_locked(staged)
        work.remote_authorized = True
        self._pending = work
        try:
            future = self._executor.submit(self._prepare_remote, work)
        except BaseException as error:
            future = Future()
            future.set_exception(error)
        work.future = future
        self._event(
            RapEventType.CHUNK_REQUEST_SUBMITTED,
            bar=work.start_bar,
            request_id=work.request.request_id,
            payload=self._chunk_payload(work, state="requested", renderer_decision="pending"),
        )
        return work

    def _work_from_staged_locked(self, staged: _StagedChunk) -> _ChunkWork:
        request = RemoteRapChunkRequest.create(
            session_id=self._session_id,
            chunk_index=staged.start_bar // 2,
            bars=tuple(RemoteRapBarRequest(plan.bar, plan.segment.topic, plan.template) for plan in staged.plans),
            tempo_bpm=self._tempo.bpm,
            remaining_budget_ms=max(1, int((staged.deadline_monotonic - self._clock()) * 1000.0)),
            policy=self._policy,
            context_lines=staged.context_lines,
            seed=self._seed + staged.start_bar // 2,
        )
        return _ChunkWork(
            staged.epoch,
            staged.start_bar,
            request,
            staged.deadline_monotonic,
            staged.fallbacks,
            staged.plans,
        )

    def _prepare_remote(self, work: _ChunkWork) -> _RemoteCompletion:
        with self._lock:
            if (
                work.epoch != self._epoch
                or work is not self._pending
                or not work.remote_authorized
                or self._stopping
                or self._closed
            ):
                raise CancelledError
            work.remote_started = True
        with self._lock:
            if (
                work.epoch != self._epoch
                or work is not self._pending
                or not work.remote_authorized
                or self._stopping
                or self._closed
            ):
                raise CancelledError
        try:
            chunk = self._strategy.prepare(
                work.request, deadline_monotonic=work.deadline_monotonic
            )
        except RemoteChunkResponseRejected as rejection:
            return _RemoteCompletion(None, self._clock(), rejection)
        return _RemoteCompletion(chunk, self._clock())

    def _wait_for_startup(self, work: _ChunkWork) -> tuple[PreparedRapChunk | None, bool]:
        future = work.future
        if future is None:
            return None, False
        wait_seconds = max(0.0, work.deadline_monotonic - self._clock())
        try:
            completion = future.result(timeout=wait_seconds)
        except TimeoutError:
            with self._lock:
                if work is not self._pending or work.epoch != self._epoch:
                    return None, False
                cancel_remote = self._revoke_remote_locked(work)
                self._reject(work, "startup_timeout", "remote chunk missed startup timeout")
                return None, cancel_remote
        except BaseException as error:
            with self._lock:
                if work is not self._pending or work.epoch != self._epoch:
                    return None, False
                self._reject(work, "failed", self._error_message(error))
                return None, False
        with self._lock:
            return self._consume_completion_locked(work, completion), False

    def _resolve_ready_locked(self, work: _ChunkWork) -> PreparedRapChunk | None:
        if work.epoch != self._epoch or work is not self._pending:
            return None
        if work.accepted is not None:
            return work.accepted
        if work.rejection_state is not None:
            return None
        future = work.future
        if future is None or not future.done():
            return None
        try:
            completion = future.result(timeout=None)
        except BaseException as error:
            self._reject(work, "failed", self._error_message(error))
            return None
        return self._consume_completion_locked(work, completion)

    def _consume_completion_locked(
        self,
        work: _ChunkWork,
        completion: _RemoteCompletion,
    ) -> PreparedRapChunk | None:
        if work.epoch != self._epoch or work is not self._pending:
            return None
        if work.accepted is not None:
            return work.accepted
        if work.rejection_state is not None:
            return None
        if completion.rejection is not None:
            rejection = completion.rejection
            evidence = rejection.evidence
            if (
                evidence.request_id == work.request.request_id
                and evidence.chunk_index == work.request.chunk_index
            ):
                work.returned_diagnostics = evidence.diagnostics
            slack_ms = (
                work.deadline_monotonic - completion.completed_monotonic
            ) * 1000.0
            work.deadline_slack_ms = slack_ms
            self._event(
                RapEventType.CHUNK_REMOTE_COMPLETED,
                bar=work.start_bar,
                request_id=work.request.request_id,
                payload=self._chunk_payload(
                    work,
                    state="returned_rejected",
                    renderer_decision="moss_aligned_remote",
                    deadline_slack_ms=slack_ms,
                ),
            )
            self._reject(
                work,
                "late"
                if completion.completed_monotonic >= work.deadline_monotonic
                else "invalid",
                self._error_message(rejection),
                slack_ms,
            )
            return None
        chunk = completion.chunk
        invalid = self._validation_error(work, chunk)
        valid_chunk = chunk if isinstance(chunk, PreparedRapChunk) else None
        work.observed = valid_chunk
        renderer = valid_chunk.renderer if valid_chunk is not None else "invalid"
        slack_ms = (work.deadline_monotonic - completion.completed_monotonic) * 1000.0
        work.deadline_slack_ms = slack_ms
        self._event(
            RapEventType.CHUNK_REMOTE_COMPLETED,
            bar=work.start_bar,
            request_id=work.request.request_id,
            payload=self._chunk_payload(
                work,
                state="returned",
                renderer_decision=renderer,
                chunk=valid_chunk,
                deadline_slack_ms=slack_ms,
            ),
        )
        if completion.completed_monotonic >= work.deadline_monotonic:
            self._reject(work, "late", "remote chunk completed at or after its immutable deadline", slack_ms)
            return None
        if invalid is not None:
            self._reject(work, "invalid", invalid, slack_ms)
            return None
        assert valid_chunk is not None
        work.accepted = valid_chunk
        work.remote_authorized = False
        for prepared in valid_chunk.bars:
            self._event(
                RapEventType.BAR_AUDIO_READY,
                bar=prepared.bar,
                request_id=work.request.request_id,
                payload=self._audio_payload(prepared),
            )
        return valid_chunk

    def _final_selection_locked(self, work: _ChunkWork) -> tuple[PreparedRapChunk | None, bool]:
        selected = self._resolve_ready_locked(work)
        if selected is not None or work.rejection_state is not None:
            return selected, False
        cancel_remote = self._revoke_remote_locked(work)
        slack_ms = (work.deadline_monotonic - self._clock()) * 1000.0
        self._reject(work, "deadline_miss", "remote chunk was not ready at commitment", slack_ms)
        return None, cancel_remote

    @staticmethod
    def _validation_error(work: _ChunkWork, chunk: object) -> str | None:
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

    def _revoke_remote_locked(self, work: _ChunkWork) -> bool:
        work.remote_authorized = False
        future = work.future
        if future is None or future.done():
            return False
        future.cancel()
        return True

    def _dispatch_abort(self, work: _ChunkWork) -> Future[object] | None:
        try:
            return self._cancellation_executor.submit(self._abort_strategy, work)
        except RuntimeError:
            return None

    def _abort_strategy(self, work: _ChunkWork) -> None:
        while True:
            try:
                self._strategy.abort()
            except BaseException as error:
                with self._lock:
                    if work.epoch == self._epoch and not self._closed:
                        self._event(
                            RapEventType.GENERATION_FAILED,
                            tick=self._last_tick if self._last_tick >= 0 else None,
                            payload={
                                "error_type": "remote_abort_error",
                                "error_message": self._error_message(error),
                            },
                        )
                return
            future = work.future
            if not work.remote_started or future is None or future.done():
                return
            try:
                future.result(timeout=0.01)
            except TimeoutError:
                time.sleep(0.001)
                continue
            except BaseException:
                return
            return

    def _install_abort_barrier_locked(self, epoch: int, future: Future[object] | None) -> None:
        if future is None:
            return
        self._abort_barrier = future
        future.add_done_callback(
            lambda completed, epoch=epoch: self._abort_barrier_completed(epoch, completed)
        )

    def _abort_barrier_completed(self, epoch: int, future: Future[object]) -> None:
        with self._lock:
            if self._abort_barrier is future:
                self._abort_barrier = None
            if epoch != self._epoch or self._closed:
                return
            task = self._submission
            if task is not None:
                self._finish_staging_locked(task)

    @staticmethod
    def _wait_for_abort_barrier(future: Future[object] | None) -> None:
        if future is None:
            return
        try:
            future.result()
        except BaseException:
            pass

    def _reject(self, work: _ChunkWork, state: str, message: str, slack_ms: float | None = None) -> None:
        if work.rejection_state is not None:
            return
        bounded_message = self._bounded_text(message, _MAX_EVENT_WARNING_BYTES)
        work.rejection_state = state
        work.rejection_message = bounded_message
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
                warnings=(bounded_message,),
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
        bars = selected.bars if selected is not None else tuple(
            replace(item, fallback_reason=reason) for item in work.fallbacks
        )
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
            self._event(
                RapEventType.BAR_FROZEN,
                bar=prepared.bar,
                tick=tick,
                request_id=work.request.request_id,
                payload=payload,
            )
            if fallback:
                self._event(
                    RapEventType.FALLBACK_ACTIVATED,
                    bar=prepared.bar,
                    tick=tick,
                    request_id=work.request.request_id,
                    payload=payload,
                )
            self._event(
                RapEventType.BAR_AUDIO_COMMITTED,
                bar=prepared.bar,
                tick=tick,
                request_id=work.request.request_id,
                payload={**self._audio_payload(prepared), "coordinator_epoch": self._epoch},
            )
        self._context_lines.extend(item.text for item in bars)
        if self._pending is work:
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
            self._event(
                RapEventType.CHUNK_FALLBACK_ACTIVATED,
                bar=work.start_bar,
                tick=tick,
                request_id=work.request.request_id,
                payload=payload,
            )
        self._event(
            RapEventType.CHUNK_COMMITTED,
            bar=work.start_bar,
            tick=tick,
            request_id=work.request.request_id,
            payload=payload,
        )
        return bars if deliver else None

    def _fallback_plan(self, bar: int, context_lines: tuple[str, ...], task: _StagingTask) -> PlannedRapBar:
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
            self._require_staging_active_locked(task)
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

    def _render_fallback(self, plan: PlannedRapBar, task: _StagingTask) -> PreparedRapBar:
        with self._lock:
            self._require_staging_active_locked(task)
            self._event(RapEventType.AUDIO_RENDER_STARTED, bar=plan.bar, payload={"source": plan.source})
        prepared = self._fallback_renderer.render(plan)
        if prepared.bar != plan.bar:
            raise ValueError("fallback renderer returned the wrong bar")
        with self._lock:
            self._require_staging_active_locked(task)
            self._event(RapEventType.AUDIO_RENDER_COMPLETED, bar=plan.bar, payload=self._audio_payload(prepared))
            self._event(RapEventType.BAR_AUDIO_READY, bar=plan.bar, payload=self._audio_payload(prepared))
        return prepared

    def _require_staging_active_locked(self, task: _StagingTask) -> None:
        if task.epoch != self._epoch or self._closed or task.cancelled:
            raise CancelledError
        if self._stopping and not task.retain_for_stop:
            raise CancelledError

    def _fallback_preparation_failed_locked(self, task: _StagingTask, error: BaseException) -> None:
        if task.failure_reported or task.epoch != self._epoch or self._closed or task.cancelled:
            return
        task.failure_reported = True
        self._event(
            RapEventType.GENERATION_FAILED,
            tick=self._last_tick if self._last_tick >= 0 else None,
            payload={
                "error_type": "fallback_preparation_error",
                "error_message": self._error_message(error),
            },
        )

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
        self._event(
            RapEventType.TICK,
            bar=current_bar,
            tick=tick,
            payload={"beat": beat, "tick_in_beat": tick_in_beat},
        )

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
        chunk = chunk if chunk is not None else work.observed
        diagnostics = (
            chunk.diagnostics
            if chunk is not None
            else work.returned_diagnostics or {}
        )
        if selected_lines is None:
            if chunk is not None:
                selected_lines = tuple(item.text for item in chunk.bars)
            elif isinstance(diagnostics, Mapping):
                returned_lines = diagnostics.get("selected_lines")
                selected_lines = (
                    tuple(returned_lines[:2])
                    if isinstance(returned_lines, (list, tuple))
                    else ()
                )
            else:
                selected_lines = ()
        raw: dict[str, object] = {
            "state": state,
            "renderer_decision": renderer_decision,
            "coordinator_epoch": work.epoch,
            "chunk_index": work.request.chunk_index,
            "bars": [work.start_bar, work.start_bar + 1],
            "selected_lines": selected_lines,
            "flows": [flow_template_payload(item.template) for item in work.plans],
            "prompt_summary": remote_generation_input_summary(work.request),
            "context_lines": work.request.context_lines,
            "request_budget_ms": work.request.remaining_budget_ms,
            "deadline_slack_ms": deadline_slack_ms,
            "failure_reason": work.rejection_message,
            "warnings": warnings,
            "hashes": {
                "request_sha256": hashlib.sha256(
                    work.request.canonical_json_bytes()
                ).hexdigest()
            },
        }
        if isinstance(diagnostics, Mapping):
            for key in (
                "candidate_counts",
                "selected_scores",
                "selected_schedules",
                "prompt_summary",
                "context_lines",
                "stage_timings_ms",
                "request_budget_ms",
                "elapsed_ms",
                "alignment",
                "stretch_warnings",
                "hashes",
                "artifact_refs",
                "transfer",
                "transfer_bytes",
            ):
                try:
                    value = diagnostics[key]
                except (KeyError, LookupError, TypeError):
                    continue
                raw[key] = value
            try:
                diagnostic_warnings = diagnostics["warnings"]
            except (KeyError, LookupError, TypeError):
                diagnostic_warnings = ()
            if isinstance(diagnostic_warnings, (list, tuple)):
                raw["warnings"] = (*warnings, *diagnostic_warnings)
        return bounded_chunk_event_payload(raw)

    @staticmethod
    def _bounded_text(value: object, max_bytes: int) -> str:
        try:
            text = str(value)
        except BaseException:
            text = f"<{type(value).__name__}>"
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")

    @classmethod
    def _error_message(cls, error: BaseException) -> str:
        try:
            message = f"{type(error).__name__}: {error}"
        except BaseException:
            message = type(error).__name__
        return cls._bounded_text(message, _MAX_EVENT_WARNING_BYTES)

    def _event(self, event_type: RapEventType, **kwargs: object) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, **kwargs)

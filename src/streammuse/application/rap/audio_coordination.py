"""Fallback-first rendering and immutable prepared-bar commitment."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
import time
from typing import Callable, Literal

from streammuse.application.rap.audio_service import RapBarRenderer
from streammuse.application.rap.monitoring import RapEventPublisher
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import AudioWarning, AudioWarningCode, PreparedRapBar, RapEventType


@dataclass(frozen=True)
class _RenderWork:
    plan: PlannedRapBar
    role: Literal["fallback", "primary"]
    epoch: int


class BarAudioCoordinator:
    """Render fallback audio first and commit one immutable bar per index."""

    def __init__(
        self,
        renderer: RapBarRenderer,
        *,
        publisher: RapEventPublisher | None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._renderer = renderer
        self._publisher = publisher
        self._monotonic = monotonic
        # Reserve one of the coordinator's two workers for fallback audio.
        # Primary renders may accumulate across lookahead bars, but they must
        # never consume the capacity needed to keep the safe path gap-free.
        self._fallback_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammuse-rap-fallback")
        self._primary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammuse-rap-primary")
        self._lock = RLock()
        self._epoch = 0
        self._closed = False
        self._fallback_work: dict[int, _RenderWork] = {}
        self._fallback_futures: dict[int, Future[PreparedRapBar]] = {}
        self._fallback_results: dict[int, PreparedRapBar] = {}
        self._primary_work: dict[int, _RenderWork] = {}
        self._primary_futures: dict[int, Future[PreparedRapBar]] = {}
        self._primary_results: dict[int, PreparedRapBar] = {}
        self._primary_completed_monotonic: dict[int, float] = {}
        self._primary_polled: set[int] = set()
        self._committed: dict[int, PreparedRapBar] = {}

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def reserve_fallback(self, plan: PlannedRapBar) -> None:
        """Start one fallback render for a bar; duplicate reservations are no-ops."""

        with self._lock:
            self._require_open()
            if plan.bar in self._committed or plan.bar in self._fallback_work:
                return
            work = _RenderWork(plan=plan, role="fallback", epoch=self._epoch)
            self._fallback_work[plan.bar] = work
            self._fallback_futures[plan.bar] = self._fallback_executor.submit(self._render, work)

    def submit_primary(self, plan: PlannedRapBar) -> None:
        """Render the latest selected primary while the fallback remains active."""

        with self._lock:
            self._require_open()
            if plan.bar in self._committed:
                return
            if plan.bar not in self._fallback_work:
                raise ValueError("primary audio requires a reserved fallback")
            previous = self._primary_futures.get(plan.bar)
            if previous is not None:
                previous.cancel()
            work = _RenderWork(plan=plan, role="primary", epoch=self._epoch)
            self._primary_work[plan.bar] = work
            self._primary_results.pop(plan.bar, None)
            self._primary_completed_monotonic.pop(plan.bar, None)
            self._primary_polled.discard(plan.bar)
            self._primary_futures[plan.bar] = self._primary_executor.submit(self._render, work)

    def poll_primary(self, bar: int, *, deadline_monotonic: float | None = None) -> PreparedRapBar | None:
        """Return a completed, uncommitted primary result at most once."""

        with self._lock:
            if bar in self._committed or bar in self._primary_polled:
                return None
            result = self._primary_results.get(bar)
            if result is None or not self._primary_met_deadline_locked(bar, deadline_monotonic):
                return None
            self._primary_polled.add(bar)
            return result

    def commit(self, bar: int, *, deadline_monotonic: float | None = None) -> PreparedRapBar:
        """Atomically prefer ready primary audio, otherwise wait only for fallback."""

        with self._lock:
            self._require_open()
            committed = self._committed.get(bar)
            if committed is not None:
                return committed
            fallback_work = self._fallback_work.get(bar)
            fallback_future = self._fallback_futures.get(bar)
            if fallback_work is None or fallback_future is None:
                raise ValueError("commit requires a reserved fallback")
            primary = self._primary_results.get(bar)
            if primary is not None and self._primary_met_deadline_locked(bar, deadline_monotonic):
                return self._commit_locked(bar, primary)

        # Deliberately wait only for the safe fallback. A primary completion is
        # checked again below before immutable commitment.
        fallback_future.result()
        with self._lock:
            self._require_open()
            committed = self._committed.get(bar)
            if committed is not None:
                return committed
            if self._fallback_work.get(bar) is not fallback_work:
                raise RuntimeError("fallback audio was reset before commitment")
            primary = self._primary_results.get(bar)
            if primary is not None and self._primary_met_deadline_locked(bar, deadline_monotonic):
                return self._commit_locked(bar, primary)
            fallback = self._fallback_results.get(bar)
            if fallback is None:
                raise RuntimeError("fallback rendering completed without an audio result")
            return self._commit_locked(bar, fallback)

    def reset(self) -> None:
        """Cancel uncommitted work and allow a fresh audio-controlled session."""

        with self._lock:
            self._require_open()
            self._epoch += 1
            futures = tuple(self._fallback_futures.values()) + tuple(self._primary_futures.values())
            self._clear_locked()
        for future in futures:
            future.cancel()

    def pause(self, successor_bar: int | None) -> None:
        """Cancel future planning work while retaining one restart-safe fallback."""

        with self._lock:
            self._require_open()
            if successor_bar is None:
                futures = tuple(self._fallback_futures.values()) + tuple(self._primary_futures.values())
                self._clear_locked()
                for future in futures:
                    future.cancel()
                return
            successor = self._committed.get(successor_bar)
            fallback_work = self._fallback_work.get(successor_bar)
            fallback_future = self._fallback_futures.get(successor_bar)
            fallback_result = self._fallback_results.get(successor_bar)
            if successor is None and (fallback_work is None or fallback_future is None):
                raise ValueError("pause requires a reserved successor bar")
            futures = tuple(
                future
                for bar, future in self._fallback_futures.items()
                if successor is not None or bar != successor_bar
            ) + tuple(self._primary_futures.values())
            self._clear_locked()
            if successor is not None:
                self._committed[successor_bar] = successor
            else:
                assert fallback_work is not None and fallback_future is not None
                self._fallback_work[successor_bar] = fallback_work
                self._fallback_futures[successor_bar] = fallback_future
                if fallback_result is not None:
                    self._fallback_results[successor_bar] = fallback_result
        for future in futures:
            future.cancel()

    def close(self) -> None:
        """Permanently prevent new work and release the coordinator executor."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            futures = tuple(self._fallback_futures.values()) + tuple(self._primary_futures.values())
            self._clear_locked()
        for future in futures:
            future.cancel()
        self._fallback_executor.shutdown(wait=True, cancel_futures=True)
        self._primary_executor.shutdown(wait=True, cancel_futures=True)

    def _render(self, work: _RenderWork) -> PreparedRapBar:
        self._event(
            RapEventType.AUDIO_RENDER_STARTED,
            bar=work.plan.bar,
            payload={
                "source": work.plan.source,
                "role": work.role,
                "text": work.plan.text,
                "coordinator_epoch": work.epoch,
            },
        )
        try:
            prepared = self._renderer.render(work.plan)
        except Exception as exc:
            self._event(
                RapEventType.AUDIO_RENDER_COMPLETED,
                bar=work.plan.bar,
                payload={
                    "source": work.plan.source,
                    "role": work.role,
                    "ready": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "coordinator_epoch": work.epoch,
                },
            )
            raise

        completed_monotonic = self._monotonic()
        with self._lock:
            accepted = self._accept_result_locked(work, prepared, completed_monotonic)
        payload = self._audio_payload(prepared, role=work.role, accepted=accepted, coordinator_epoch=work.epoch)
        self._event(RapEventType.AUDIO_RENDER_COMPLETED, bar=prepared.bar, payload=payload)
        if accepted:
            self._event(RapEventType.BAR_AUDIO_READY, bar=prepared.bar, payload=payload)
            for warning in prepared.warnings:
                self._publish_warning(prepared, warning, coordinator_epoch=work.epoch)
        return prepared

    def _accept_result_locked(
        self,
        work: _RenderWork,
        prepared: PreparedRapBar,
        completed_monotonic: float,
    ) -> bool:
        if self._closed or work.epoch != self._epoch or prepared.bar in self._committed:
            return False
        if work.role == "fallback":
            if self._fallback_work.get(prepared.bar) is not work:
                return False
            self._fallback_results[prepared.bar] = prepared
            return True
        if self._primary_work.get(prepared.bar) is not work:
            return False
        self._primary_results[prepared.bar] = prepared
        self._primary_completed_monotonic[prepared.bar] = completed_monotonic
        return True

    def _primary_met_deadline_locked(self, bar: int, deadline_monotonic: float | None) -> bool:
        if deadline_monotonic is None:
            return True
        completed = self._primary_completed_monotonic.get(bar)
        return completed is not None and completed <= deadline_monotonic

    def _commit_locked(self, bar: int, prepared: PreparedRapBar) -> PreparedRapBar:
        self._committed[bar] = prepared
        self._fallback_results.pop(bar, None)
        self._primary_results.pop(bar, None)
        self._primary_completed_monotonic.pop(bar, None)
        self._fallback_work.pop(bar, None)
        self._primary_work.pop(bar, None)
        self._fallback_futures.pop(bar, None)
        self._primary_futures.pop(bar, None)
        self._primary_polled.add(bar)
        return prepared

    def _clear_locked(self) -> None:
        self._fallback_work.clear()
        self._fallback_futures.clear()
        self._fallback_results.clear()
        self._primary_work.clear()
        self._primary_futures.clear()
        self._primary_results.clear()
        self._primary_completed_monotonic.clear()
        self._primary_polled.clear()
        self._committed.clear()

    def _publish_warning(self, prepared: PreparedRapBar, warning: AudioWarning, *, coordinator_epoch: int) -> None:
        event_type = {
            AudioWarningCode.PRONUNCIATION_FALLBACK: RapEventType.PRONUNCIATION_FALLBACK,
            AudioWarningCode.TIMING_PRESSURE: RapEventType.TIMING_PRESSURE,
            AudioWarningCode.FORCED_BAR_FIT: RapEventType.FORCED_BAR_FIT,
            AudioWarningCode.SYNTHESIS_FAILED: RapEventType.SYNTHESIS_FAILED,
        }.get(warning.code)
        if event_type is None:
            return
        diagnostic = next(
            (item for item in prepared.diagnostics if item.slot_index == warning.slot_index),
            None,
        )
        self._event(
            event_type,
            bar=prepared.bar,
            payload={
                "code": warning.code.value,
                "severity": warning.severity.value,
                "message": warning.message,
                "slot_index": warning.slot_index,
                "word": warning.word,
                "available_ms": warning.available_ms,
                "rendered_ms": warning.rendered_ms,
                "compression_ratio": warning.compression_ratio,
                "overlap_ms": warning.overlap_ms,
                "action": warning.action,
                "source": diagnostic.pronunciation_source if diagnostic is not None else None,
                "target_sample": diagnostic.target_sample if diagnostic is not None else None,
                "renderer_phonemes": list(diagnostic.renderer_phonemes) if diagnostic is not None else [],
                "coordinator_epoch": coordinator_epoch,
            },
        )

    @staticmethod
    def _audio_payload(
        prepared: PreparedRapBar,
        *,
        role: str,
        accepted: bool,
        coordinator_epoch: int,
    ) -> dict[str, object]:
        pronunciation_sources: dict[str, int] = {}
        for diagnostic in prepared.diagnostics:
            pronunciation_sources[diagnostic.pronunciation_source] = (
                pronunciation_sources.get(diagnostic.pronunciation_source, 0) + 1
            )
        return {
            "source": prepared.source,
            "role": role,
            "text": prepared.text,
            "render_latency_ms": prepared.render_latency_ms,
            "synthesis_latency_ms": sum(item.synthesis_latency_ms for item in prepared.diagnostics),
            "vocal_syllable_count": len(prepared.diagnostics),
            "vocal_source_frames": sum(item.source_frames for item in prepared.diagnostics),
            "vocal_fitted_frames": sum(item.fitted_frames for item in prepared.diagnostics),
            "vocal_rendered_frames": sum(
                item.fitted_frames if item.rendered_frames is None else item.rendered_frames
                for item in prepared.diagnostics
            ),
            "vocal_cropped_frames": sum(item.cropped_frames for item in prepared.diagnostics),
            "pronunciation_sources": pronunciation_sources,
            "frame_count": prepared.audio.frame_count,
            "warnings": [warning.code.value for warning in prepared.warnings],
            "accepted": accepted,
            "coordinator_epoch": coordinator_epoch,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("audio coordinator is closed")

    def _event(self, event_type: RapEventType, **kwargs: object) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, **kwargs)

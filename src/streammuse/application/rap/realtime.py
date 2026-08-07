"""Rolling, tick-driven delivery of beat-aligned rap syllables."""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import RLock
from typing import Callable

from streammuse.application.rap.alignment import choose_best_line
from streammuse.application.rap.rhythm import build_bar_slots
from streammuse.application.rap.service import CandidateGenerator
from streammuse.domain.rap import AlignedLine, CandidateBatch, CandidateRequest, ScheduledSyllable
from streammuse.domain.timing import Tempo


class RollingRapController:
    """Keep aligned lyric bars ready and emit syllables on music-service ticks."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        topic: str,
        pattern: str,
        fallback_generator: CandidateGenerator,
        primary_generator: CandidateGenerator | None = None,
        candidate_count: int = 12,
        lookahead_bars: int = 2,
        emit: Callable[[ScheduledSyllable], None],
        close_primary: Callable[[], None] | None = None,
        executor: Executor | None = None,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if lookahead_bars <= 0:
            raise ValueError("lookahead_bars must be positive")

        self._tempo = tempo
        self._topic = topic
        self._pattern = pattern
        self._fallback_generator = fallback_generator
        self._primary_generator = primary_generator
        self._candidate_count = candidate_count
        self._lookahead_bars = lookahead_bars
        self._emit = emit
        self._close_primary = close_primary
        self._executor = executor or (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="streammuse-rap")
            if primary_generator is not None
            else None
        )

        self._lock = RLock()
        self._started = False
        self._closed = False
        self._last_tick = -1
        self._lines: dict[int, AlignedLine] = {}
        self._line_sources: dict[int, str] = {}
        self._used_texts: set[str] = set()
        self._future: Future[CandidateBatch] | None = None
        self._future_bar: int | None = None
        self._next_primary_bar = 1

    def start(self) -> None:
        """Synchronously preload fallback bars before the first music tick."""
        with self._lock:
            if self._started or self._closed:
                return
            self._started = True
            self._ensure_fallback_through(self._lookahead_bars - 1)
            self._submit_next_primary(current_bar=0)

    def on_tick(self, tick: int) -> None:
        """Emit events exactly on ``tick`` without waiting for model generation."""
        with self._lock:
            if not self._started or self._closed:
                return
            self._last_tick = tick
            self._drain_primary_result()
            current_bar = tick // self._tempo.ticks_per_bar
            self._ensure_fallback_through(current_bar + self._lookahead_bars - 1)
            self._submit_next_primary(current_bar=current_bar)
            events = tuple(
                event
                for line in self._lines.values()
                for event in line.events
                if event.slot.tick == tick
            )

        for event in events:
            try:
                self._emit(event)
            except Exception:
                # Presentation failures cannot interrupt the shared music clock.
                continue

    def close(self) -> None:
        """Cancel optional background work without delaying service shutdown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._future
            executor = self._executor
            close_primary = self._close_primary

        if future is not None:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if close_primary is not None:
            try:
                close_primary()
            except Exception:
                pass

    def line_for_bar(self, bar: int) -> AlignedLine:
        """Return the current planned line for a bar for diagnostics and tests."""
        with self._lock:
            return self._lines[bar]

    def line_source_for_bar(self, bar: int) -> str:
        """Return the candidate source that currently owns a bar."""
        with self._lock:
            return self._line_sources[bar]

    def _ensure_fallback_through(self, last_bar: int) -> None:
        for bar in range(last_bar + 1):
            if bar in self._lines:
                continue
            batch = self._fallback_generator.generate(self._request_for_bar(bar, source="fallback"))
            line = self._select_line(batch, bar)
            if line is None:
                raise ValueError("fallback rap generator returned no one-bar line")
            self._store_line(bar, line, batch.source)

    def _submit_next_primary(self, *, current_bar: int) -> None:
        if self._primary_generator is None or self._executor is None or self._future is not None:
            return
        target_bar = max(self._next_primary_bar, current_bar + 1)
        self._next_primary_bar = target_bar + 1
        self._future = self._executor.submit(self._primary_generator.generate, self._request_for_bar(target_bar, source="primary"))
        self._future_bar = target_bar

    def _drain_primary_result(self) -> None:
        future = self._future
        target_bar = self._future_bar
        if future is None or target_bar is None or not future.done():
            return

        self._future = None
        self._future_bar = None
        try:
            batch = future.result()
        except Exception:
            return

        target_tick = target_bar * self._tempo.ticks_per_bar
        if batch.source != "local_chat" or target_tick <= self._last_tick:
            return
        line = self._select_line(batch, target_bar)
        if line is not None:
            self._store_line(target_bar, line, batch.source)

    def _select_line(self, batch: CandidateBatch, bar: int) -> AlignedLine | None:
        candidates = tuple(candidate.strip() for candidate in batch.candidates if candidate.strip())
        if not candidates:
            return None
        available = tuple(candidate for candidate in candidates if candidate not in self._used_texts) or candidates
        line = choose_best_line(available, build_bar_slots(self._tempo, self._pattern, bar))
        if line.overflow_count or not line.events:
            return None
        return line

    def _store_line(self, bar: int, line: AlignedLine, source: str) -> None:
        self._lines[bar] = line
        self._line_sources[bar] = source
        self._used_texts.add(line.text)

    def _request_for_bar(self, bar: int, *, source: str) -> CandidateRequest:
        return CandidateRequest(
            request_id=f"rolling-{source}-bar-{bar}",
            target_bar=bar,
            topic=self._topic,
            template_id=self._pattern,
            required_syllables=len(build_bar_slots(self._tempo, self._pattern, bar)),
            count=self._candidate_count,
            context_lines=(),
            seed=bar,
        )

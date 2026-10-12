"""Server side of protocol v3: one session keeps lyrics and audio ahead of playback.

Two background loops per session:

- the **writer** keeps the latest lyric for every bar in ``P+1 .. P+lyric_lead``
  (``P`` = the last bar the client committed), writing bars in order because
  each line follows the one before;
- the **renderer** keeps the latest audio for every bar in ``P+1 .. P+audio_lead``,
  lowest bar first, so without edits it always renders the bar needed next.

The client fetches audio by ``lyric_id``. Ready audio returns at once; if only
an older revision of the bar is ready, that is returned and the requested one
jumps the queue; otherwise the fetch waits for it (up to the client's budget).

A listener edit supersedes every not-yet-committed bar from ``max(from_bar, P+1)``:
their lyrics are rewritten as new revisions. Audio already rendered for old
revisions is kept, because the client plays the old version when the new one
is not ready in time. Rendering is not cancelled mid-way; queued work for
superseded lines is simply never started.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import threading
import time
from typing import Protocol

from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.lyric_writer import (
    LyricWriteFailed,
    LyricWriteRequest,
    SequentialLyricWriter,
)
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord, PlanEdit

FETCH_READY = "ready"
FETCH_OLDER_VERSION = "older_version"
FETCH_PENDING = "pending"
FETCH_FAILED = "failed"
FETCH_SUPERSEDED = "superseded"
FETCH_UNKNOWN = "unknown_lyric"

_HISTORY_LINES = 4
_KEEP_BEHIND_BARS = 8
_MAX_WRITE_ATTEMPTS = 3


@dataclass(frozen=True)
class RenderedBarAudio:
    lyric_id: str
    bar: int
    revision: int
    package: bytes
    media_type: str
    manifest: BarAudioManifest
    render_ms: float


class BarAudioRenderer(Protocol):
    def render(
        self, record: BarLyricRecord, *, tempo_bpm: float, previous: BarLyricRecord | None = None
    ) -> RenderedBarAudio:
        """Render ``record``; ``previous`` is the latest line of the bar before, to continue from."""
        ...


class LyricSessionListener(Protocol):
    """Events pushed to the client over the session channel."""

    def lyric(self, record: BarLyricRecord) -> None: ...

    def superseded(self, bars: tuple[int, ...], edit_seq: int) -> None: ...

    def audio_ready(self, bar: int, lyric_id: str) -> None: ...

    def error(self, message: str, *, bar: int | None = None) -> None: ...


@dataclass(frozen=True)
class AudioFetchResult:
    status: str
    requested_lyric_id: str
    audio: RenderedBarAudio | None = None


@dataclass
class _AudioEntry:
    status: str  # "queued" | "rendering" | "ready" | "failed"
    audio: RenderedBarAudio | None = None
    error: str | None = None
    attempts: int = 0


@dataclass
class _BarState:
    revisions: list[BarLyricRecord] = field(default_factory=list)
    latest: BarLyricRecord | None = None
    next_revision: int = 0
    write_failures: int = 0
    committed_text: str | None = None
    committed_template: object | None = None


class LyricStreamSession:
    def __init__(
        self,
        *,
        session_id: str,
        tempo_bpm: float,
        plan: LivePlan,
        writer: SequentialLyricWriter,
        renderer: BarAudioRenderer,
        listener: LyricSessionListener,
        start_bar: int = 0,
        seed: int = 0,
        lyric_lead: int = 3,
        audio_lead: int = 2,
        history: tuple[tuple[int, str], ...] = (),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if lyric_lead < 1 or audio_lead < 1 or lyric_lead < audio_lead:
            raise ValueError("leads must be positive and lyric_lead must cover audio_lead")
        self.session_id = session_id
        self._tempo_bpm = float(tempo_bpm)
        self._plan = plan
        self._writer = writer
        self._renderer = renderer
        self._listener = listener
        self._seed = int(seed)
        self._lyric_lead = lyric_lead
        self._audio_lead = audio_lead
        self._clock = monotonic
        self._condition = threading.Condition()
        self._bars: dict[int, _BarState] = {}
        self._by_id: dict[str, BarLyricRecord] = {}
        self._audio: dict[str, _AudioEntry] = {}
        self._bumped: list[str] = []
        self._playhead = start_bar - 1
        self._closed = False
        self._threads: list[threading.Thread] = []
        self._stats = {"lyrics_written": 0, "audio_rendered": 0, "superseded": 0, "write_failures": 0}
        for bar, text in history:
            state = self._state(bar)
            state.committed_text = text
            state.committed_template = None

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        for name, target in (("writer", self._writer_loop), ("renderer", self._render_loop)):
            thread = threading.Thread(target=target, name=f"lyric-session-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self, timeout: float = 30.0) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout)

    # -- client messages -----------------------------------------------------------

    def apply_edit(self, edit: PlanEdit) -> tuple[int, ...]:
        with self._condition:
            self._plan = self._plan.with_edit(edit)
            effective = max(edit.from_bar, self._playhead + 1)
            superseded = []
            for bar, state in sorted(self._bars.items()):
                if bar >= effective and state.latest is not None:
                    state.latest = None
                    state.next_revision = max(state.next_revision, state.revisions[-1].revision + 1)
                    state.write_failures = 0
                    superseded.append(bar)
            # Bars not written yet also start a new revision so their ids differ.
            for bar in range(effective, self._playhead + self._lyric_lead + 1):
                state = self._state(bar)
                if state.revisions and bar not in superseded:
                    state.next_revision = max(state.next_revision, state.revisions[-1].revision + 1)
            self._stats["superseded"] += len(superseded)
            self._condition.notify_all()
        if superseded:
            self._listener.superseded(tuple(superseded), edit.edit_seq)
        return tuple(superseded)

    def committed(self, bar: int, lyric_id: str | None, text: str) -> None:
        with self._condition:
            state = self._state(bar)
            record = self._by_id.get(lyric_id) if lyric_id else None
            state.committed_text = text
            state.committed_template = record.flow_template if record is not None else self._plan.for_bar(bar).flow_template
            if bar > self._playhead:
                self._playhead = bar
            self._prune_locked()
            self._condition.notify_all()

    def fetch_audio(
        self, bar: int, lyric_id: str, wait_seconds: float, *, accept_older: bool = True
    ) -> AudioFetchResult:
        """Audio for ``lyric_id``; with ``accept_older`` an older ready revision returns at once."""
        deadline = self._clock() + max(0.0, wait_seconds)
        with self._condition:
            record = self._by_id.get(lyric_id)
            if record is None or record.bar != bar:
                return AudioFetchResult(FETCH_UNKNOWN, lyric_id)
            while True:
                entry = self._audio.get(lyric_id)
                if entry is not None and entry.status == "ready":
                    return AudioFetchResult(FETCH_READY, lyric_id, entry.audio)
                older = self._best_older_locked(record)
                state = self._bars[bar]
                if state.latest is None or state.latest.lyric_id != lyric_id:
                    # A newer revision exists (or is being written): do not render this one.
                    return AudioFetchResult(FETCH_SUPERSEDED, lyric_id, older)
                if entry is not None and entry.status == "failed" and entry.attempts >= 2:
                    return AudioFetchResult(FETCH_FAILED, lyric_id, older)
                if entry is None or entry.status == "failed":
                    if lyric_id not in self._bumped:
                        self._bumped.append(lyric_id)
                        self._condition.notify_all()
                if older is not None and accept_older:
                    return AudioFetchResult(FETCH_OLDER_VERSION, lyric_id, older)
                remaining = deadline - self._clock()
                if remaining <= 0 or self._closed:
                    return AudioFetchResult(FETCH_PENDING, lyric_id)
                self._condition.wait(remaining)

    def latest_records(self) -> tuple[BarLyricRecord, ...]:
        """The latest line of every bar after the playhead (resent on reconnect)."""
        with self._condition:
            return tuple(
                state.latest
                for bar, state in sorted(self._bars.items())
                if bar > self._playhead and state.latest is not None
            )

    def lyric(self, lyric_id: str) -> BarLyricRecord | None:
        with self._condition:
            return self._by_id.get(lyric_id)

    def snapshot(self) -> Mapping[str, object]:
        with self._condition:
            return {
                "playhead": self._playhead,
                "edit_seq": self._plan.edit_seq,
                "latest": {bar: state.latest.revision for bar, state in self._bars.items() if state.latest},
                "audio": {
                    lyric_id: entry.status
                    for lyric_id, entry in self._audio.items()
                },
                **self._stats,
            }

    # -- writer --------------------------------------------------------------------

    def _writer_loop(self) -> None:
        while True:
            with self._condition:
                job = None
                while not self._closed:
                    job = self._next_write_locked()
                    if job is not None:
                        break
                    self._condition.wait(1.0)
                if self._closed:
                    return
            request, plan_seq = job
            try:
                written = self._writer.write(request)
            except LyricWriteFailed as error:
                with self._condition:
                    state = self._state(request.bar)
                    state.write_failures += 1
                    self._stats["write_failures"] += 1
                self._listener.error(f"lyric writer failed: {error}", bar=request.bar)
                continue
            except Exception as error:  # generator transport failure; retry after a pause
                with self._condition:
                    self._state(request.bar).write_failures += 1
                    self._stats["write_failures"] += 1
                    self._condition.wait(0.5)
                self._listener.error(f"lyric writer error: {type(error).__name__}: {error}", bar=request.bar)
                continue
            published = None
            with self._condition:
                state = self._state(request.bar)
                previous = self._bars.get(request.bar - 1)
                still_valid = (
                    not self._closed
                    and state.latest is None
                    and state.next_revision == request.revision
                    and self._plan.for_bar(request.bar).edit_seq == plan_seq
                    and (
                        request.previous_line is None
                        or self._line_for_locked(request.bar - 1, previous) == request.previous_line
                    )
                )
                if still_valid:
                    record = written.record
                    state.revisions.append(record)
                    state.latest = record
                    state.next_revision = record.revision + 1
                    self._by_id[record.lyric_id] = record
                    self._stats["lyrics_written"] += 1
                    published = record
                    self._condition.notify_all()
            if published is not None:
                self._listener.lyric(published)

    def _next_write_locked(self) -> tuple[LyricWriteRequest, int] | None:
        for bar in range(self._playhead + 1, self._playhead + self._lyric_lead + 1):
            state = self._state(bar)
            if state.latest is not None:
                continue
            if state.write_failures >= _MAX_WRITE_ATTEMPTS:
                continue  # this bar falls back on the client; keep writing later bars
            previous = self._bars.get(bar - 1)
            previous_line = self._line_for_locked(bar - 1, previous)
            if bar - 1 > self._playhead and (previous is None or previous.latest is None):
                if previous is None or previous.write_failures < _MAX_WRITE_ATTEMPTS:
                    return None  # write in order: the previous bar first
            plan = self._plan.for_bar(bar)
            previous_template = None
            if previous is not None:
                previous_template = (
                    previous.latest.flow_template if previous.latest is not None else previous.committed_template
                )
            request = LyricWriteRequest(
                session_id=self.session_id,
                bar=bar,
                revision=state.next_revision,
                topic=plan.topic,
                flow_template=plan.flow_template,
                history=self._history_locked(bar),
                seed=self._seed + bar * 1_000 + state.next_revision * 10 + state.write_failures,
                previous_line=previous_line,
                previous_template=previous_template,  # type: ignore[arg-type]
            )
            return request, plan.edit_seq
        return None

    def _line_for_locked(self, bar: int, state: _BarState | None) -> str | None:
        if state is None:
            return None
        if bar <= self._playhead and state.committed_text is not None:
            return state.committed_text
        if state.latest is not None:
            return state.latest.text
        return state.committed_text

    def _history_locked(self, bar: int) -> tuple[str, ...]:
        lines = []
        for previous in range(bar - _HISTORY_LINES, bar):
            line = self._line_for_locked(previous, self._bars.get(previous))
            if line:
                lines.append(line)
        return tuple(lines)

    # -- renderer ------------------------------------------------------------------

    def _render_loop(self) -> None:
        while True:
            with self._condition:
                record = None
                while not self._closed:
                    record = self._next_render_locked()
                    if record is not None:
                        break
                    self._condition.wait(1.0)
                if self._closed:
                    return
                entry = self._audio.setdefault(record.lyric_id, _AudioEntry("queued"))
                entry.status = "rendering"
                entry.attempts += 1
                before = self._bars.get(record.bar - 1)
                previous = None if before is None else before.latest
            try:
                audio = self._renderer.render(record, tempo_bpm=self._tempo_bpm, previous=previous)
                error = None
            except Exception as failure:  # reported to the fetching client
                audio, error = None, f"{type(failure).__name__}: {failure}"
            with self._condition:
                entry = self._audio[record.lyric_id]
                if audio is not None:
                    entry.status, entry.audio = "ready", audio
                    self._stats["audio_rendered"] += 1
                else:
                    entry.status, entry.error = "failed", error
                self._condition.notify_all()
            if audio is not None:
                self._listener.audio_ready(record.bar, record.lyric_id)
            else:
                self._listener.error(f"bar audio render failed: {error}", bar=record.bar)

    def _next_render_locked(self) -> BarLyricRecord | None:
        while self._bumped:
            lyric_id = self._bumped.pop(0)
            record = self._by_id.get(lyric_id)
            if record is None:
                continue
            state = self._bars.get(record.bar)
            entry = self._audio.get(lyric_id)
            if state is None or state.latest is None or state.latest.lyric_id != lyric_id:
                continue
            if entry is not None and entry.status in ("rendering", "ready"):
                continue
            if entry is not None and entry.attempts >= 2:
                continue
            return record
        for bar in range(self._playhead + 1, self._playhead + self._audio_lead + 1):
            state = self._bars.get(bar)
            if state is None or state.latest is None:
                continue
            entry = self._audio.get(state.latest.lyric_id)
            if entry is None:
                return state.latest
        return None

    def _best_older_locked(self, record: BarLyricRecord) -> RenderedBarAudio | None:
        state = self._bars.get(record.bar)
        if state is None:
            return None
        for older in reversed(state.revisions):
            if older.revision >= record.revision:
                continue
            entry = self._audio.get(older.lyric_id)
            if entry is not None and entry.status == "ready":
                return entry.audio
        return None

    # -- bookkeeping ---------------------------------------------------------------

    def _state(self, bar: int) -> _BarState:
        state = self._bars.get(bar)
        if state is None:
            state = _BarState()
            self._bars[bar] = state
        return state

    def _prune_locked(self) -> None:
        cutoff = self._playhead - _KEEP_BEHIND_BARS
        for bar in [bar for bar in self._bars if bar < cutoff]:
            state = self._bars.pop(bar)
            for record in state.revisions:
                self._by_id.pop(record.lyric_id, None)
                self._audio.pop(record.lyric_id, None)

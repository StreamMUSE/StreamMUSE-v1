"""Continuous steering over one long v3 session per playback.

The bounded adapter opens a new server session for every request, so a steering
edit pays for a connection and a cold session, and the first bar of each request
cannot continue the bar before it. Here one session lasts from Start to Stop:

- the first request of a playback opens it;
- a later request with the same input version only extends coverage: it tells
  the server how far playback has got, and the server writes and renders on;
- a request with a new input version is a steering edit: one ``update`` makes
  the server rewrite every bar from the request's first bar;
- every bar the server produces is fetched and prepared once, then delivered to
  the live request of the current version that covers it. A bar that is ready
  before any request covers it waits here and is delivered the moment one does.

**Reporting the playhead.** The server works on bars ``P+1 .. P+lead``, where
``P`` is the last bar the client reported as committed, and an edit takes effect
from ``max(from_bar, P+1)``. The adapter never sees playback itself, only
requests, so it derives ``P`` from when the service sends them:

- a first request or an edit starts at the bar after the one playing: ``P = start - 1``;
- the service extends coverage once no more than half a request's length remains,
  so a request for bars ``[s, s+m)`` arrives while bar ``s - m//2 - 1`` plays.

Both are at or behind the true playhead (never ahead, which would delay an
edit). Because the second lags, the leads are larger than v1.0's 3 and 2:
``audio_lead = max(2, m + m//2)`` and ``lyric_lead = audio_lead + 1``, which is
exactly enough for the server's window to reach the end of every request.

All session state belongs to one worker thread. ``submit`` and ``cancel`` only
queue, so they are safe on the audio callback.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import logging
import queue
from threading import Event, Lock, Thread
import uuid

from streammuse.domain.rap import FlowTemplate
from streammuse.domain.rap.lyric_stream import (
    BarLyricRecord, CONTINUATION_HANDOFF_CAPABILITY, LYRIC_STREAM_SCHEMA_VERSION, PlanEdit,
)
from streammuse.domain.rap.remote_chunk import _flow_payload
from streammuse.infrastructure.rap.inference_backend import build_span, fetch_and_prepare

_LOGGER = logging.getLogger(__name__)
_KEEP_BEHIND_BARS = 8


def session_leads(request_bars):
    """(lyric_lead, audio_lead) that keep the server's window over every request; see the module text."""
    audio = max(2, request_bars + request_bars // 2)
    return audio + 1, audio


@dataclass
class _Work:
    request: object
    on_span: object
    on_event: object
    deadline: float
    start_bar: int
    end_bar: int
    delivered: set = field(default_factory=set)
    done: bool = False  # complete, failed or canceled: nothing more is delivered or reported


@dataclass
class _Session:
    key: tuple
    remote_id: str
    stream: object
    holder: object
    preparer: object
    bpm: float
    version: int
    topic: str
    template: FlowTemplate
    start_bar: int
    playhead: int
    lyric_lead: int
    audio_lead: int
    token: object = field(default_factory=object)
    stopped: Event = field(default_factory=Event)
    edit_seq: int = 0
    latest: dict = field(default_factory=dict)        # bar -> newest usable BarLyricRecord
    seen_revision: dict = field(default_factory=dict)  # bar -> highest revision received
    min_revision: dict = field(default_factory=dict)   # bar -> lowest revision of the current plan
    fetching: dict = field(default_factory=dict)       # lyric id -> Event that abandons the fetch
    ready: dict = field(default_factory=dict)          # bar -> (record, prepared, timings, continuation)
    accepted: dict = field(default_factory=dict)       # bar -> record the service last accepted
    failed: dict = field(default_factory=dict)         # bar -> why the current plan's line never came
    announced: set = field(default_factory=set)
    idle_since: float | None = None
    last_work: _Work | None = None


class SteeringSessionBackend:
    """See the module text. ``submit``/``cancel``/``close`` mirror RealInferenceBackend."""

    def __init__(self, *, stream_factory, capabilities, acquire_source, release_source,
                 forget_transports, fetchers, preparer_factory, templates, continuations, clock,
                 request_timeout_seconds, idle_close_seconds=0.5, max_requests=16):
        self._stream_factory, self._capabilities = stream_factory, capabilities
        self._acquire_source, self._release_source = acquire_source, release_source
        self._forget_transports, self._fetchers = forget_transports, fetchers
        self._preparer_factory, self._templates = preparer_factory, templates
        self._continuations, self._clock = continuations, clock
        self._timeout, self._idle_close, self._limit = request_timeout_seconds, idle_close_seconds, max_requests
        self._lock, self._closed = Lock(), False
        self._seen, self._known = OrderedDict(), {}
        self._events = queue.Queue()
        self._live = {}  # request id -> _Work; worker thread only
        self._session = None
        self._closers = []
        self._thread = None

    # -- called by the service (possibly on the audio callback): queue only ---------

    def submit(self, request, on_span, on_event):
        with self._lock:
            if self._closed:
                raise RuntimeError("inference backend is closed")
            if request.request_id in self._seen:
                return False
            if len(self._known) >= self._limit:
                raise RuntimeError("inference backend request capacity exhausted")
            work = _Work(copy.deepcopy(request), on_span, on_event, self._clock() + self._timeout,
                         request.start_tick // 16, (request.start_tick + request.duration_ticks) // 16)
            self._known[request.request_id] = work
            self._seen[request.request_id] = None
            if len(self._seen) > 1024:
                self._seen.popitem(last=False)
            if self._thread is None:
                self._thread = Thread(target=self._run, name="inference-v3-steering", daemon=True)
                self._thread.start()
            self._events.put(("submit", work))
        return True

    def cancel(self, request_id):
        with self._lock:
            known = request_id in self._known
            if known:
                self._events.put(("cancel", request_id))
        return known

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            self._events.put(("close",))
        if thread is not None:
            thread.join(30)
        for closer in self._closers:
            closer.join(10)

    # -- worker thread ---------------------------------------------------------

    def _run(self):
        while True:
            try:
                event = self._events.get(timeout=0.05)
            except queue.Empty:
                event = ("tick",)
            try:
                if event[0] == "close":
                    for work in list(self._live.values()):
                        self._finish(work, "canceled", client_fenced=True,
                                     server_cancel_confirmed=False, gpu_cancel_confirmed=False)
                    self._end_session("backend closed")
                    return
                self._handle(event)
                self._housekeeping()
            except Exception as error:  # a bug here must not end the thread silently
                _LOGGER.exception("steering session worker failed")
                self._fail_session(error)

    def _handle(self, event):
        kind = event[0]
        if kind == "submit":
            self._on_submit(event[1])
        elif kind == "cancel":
            work = self._live.get(event[1])
            if work is not None:
                self._finish(work, "canceled", client_fenced=True,
                             server_cancel_confirmed=False, gpu_cancel_confirmed=False)
            else:
                self._forget(event[1])
        elif kind in ("message", "prepared", "failure", "abandoned"):
            session = self._session
            if session is None or event[1] is not session.token:
                return  # from a session that has ended
            if kind == "message":
                self._on_message(session, event[2])
            elif kind == "prepared":
                self._on_prepared(session, *event[2:])
            elif kind == "failure":
                self._on_fetch_failure(session, event[2], event[3])
            else:
                session.fetching.pop(event[2].lyric_id, None)

    def _housekeeping(self):
        now = self._clock()
        overdue = [work for work in self._live.values() if now >= work.deadline]
        if any(not work.done for work in overdue):
            # Something is stuck (for example the server lost the session). Start over:
            # the next request opens a new session from its own first bar.
            self._fail_session(TimeoutError("steering request timed out"))
        for work in overdue:
            self._finish(work, "canceled")  # complete long ago and never released: just forget it
        session = self._session
        if session is None:
            return
        if self._live:
            session.idle_since = None
        elif session.idle_since is None:
            session.idle_since = now
        elif now - session.idle_since >= self._idle_close:
            # Stop: every request was canceled and none followed. An edit cancels and
            # resubmits in one step, so it never looks idle for this long.
            self._end_session("no live request")

    # -- requests --------------------------------------------------------------

    def _event(self, work, kind, **values):
        request = work.request
        try:
            work.on_event({"type": kind, "request_id": request.request_id, "session_id": request.session_id,
                           "epoch": request.epoch, "input_version": request.input_version, **values})
        except Exception:
            _LOGGER.exception("inference event callback failed")

    def _forget(self, request_id):
        with self._lock:
            self._known.pop(request_id, None)

    def _finish(self, work, kind, **values):
        """End a request: report ``kind`` unless it already completed, and stop delivering to it."""
        self._live.pop(work.request.request_id, None)
        self._forget(work.request.request_id)
        if not work.done:
            work.done = True
            self._event(work, kind, **values)

    def _on_submit(self, work):
        request = work.request
        key = (request.session_id, request.epoch, request.bpm)
        session = self._session
        if session is not None and session.key != key:
            self._end_session("new playback")
            session = None
        self._live[request.request_id] = work
        template = self._templates.get(request.template_id)
        if not isinstance(template, FlowTemplate):
            self._finish(work, "error", message="ValueError: unknown flow template")
            return
        if session is None:
            try:
                session = self._open(work, template)
            except Exception as error:
                self._forget_transports()
                self._finish(work, "error", message=f"{type(error).__name__}: {error}"[:512])
                return
            self._session = session
        elif request.input_version != session.version:
            self._steer(session, work, template)
        else:
            self._report_playhead(session, work.start_bar - (work.end_bar - work.start_bar) // 2 - 1, work)
        session.last_work = work
        session.idle_since = None
        failures = [reason for bar, reason in session.failed.items() if work.start_bar <= bar < work.end_bar]
        if failures:
            self._finish(work, "error", message=failures[0])
            return
        self._start_fetches(session)
        self._deliver_ready(session)

    def _open(self, work, template):
        request = work.request
        stream = self._stream_factory()
        holder = None
        try:
            capabilities = self._capabilities(stream)
            lyric_lead, audio_lead = session_leads(work.end_bar - work.start_bar)
            remote_id = "inference-steer-" + uuid.uuid4().hex
            history = request.history[-4:]
            payload = {
                "type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION, "capabilities": [],
                "tempo_bpm": request.bpm, "start_bar": work.start_bar,
                "seed": request.config.get("seed", 0), "loop": True,
                # One looping one-bar segment: the same topic and flow for every bar until an edit.
                "segments": [{"start_bar": 0, "bars": 1, "topic": request.topic,
                              "flow_template": _flow_payload(template)}],
                "edits": [], "history": [{"bar": work.start_bar - len(history) + index, "text": line}
                                         for index, line in enumerate(history)],
                "lyric_lead": lyric_lead, "audio_lead": audio_lead,
            }
            previous = self._continuations.before(request, request.start_tick)
            if previous is not None and CONTINUATION_HANDOFF_CAPABILITY in capabilities:
                payload["capabilities"] = [CONTINUATION_HANDOFF_CAPABILITY]
                payload["continue_from"] = {"lyric_id": previous}
            holder = self._acquire_source()
            session = _Session(
                key=(request.session_id, request.epoch, request.bpm), remote_id=remote_id, stream=stream,
                holder=holder, preparer=self._preparer_factory(request), bpm=request.bpm,
                version=request.input_version, topic=request.topic, template=template,
                start_bar=work.start_bar, playhead=work.start_bar - 1,
                lyric_lead=lyric_lead, audio_lead=audio_lead)
            token = session.token
            stream.open(remote_id, payload, lambda message: self._events.put(("message", token, message)))
        except Exception:
            if holder is not None:
                self._release_source(holder, failed=True)
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._event(work, "steering_session", remote_session_id=remote_id, start_bar=work.start_bar,
                    lyric_lead=lyric_lead, audio_lead=audio_lead,
                    continue_from=payload.get("continue_from", {}).get("lyric_id"))
        return session

    def _steer(self, session, work, template):
        request = work.request
        session.version, session.topic, session.template = request.input_version, request.topic, template
        session.edit_seq += 1
        edit = PlanEdit(session.edit_seq, work.start_bar, request.topic, template)
        # Everything from the edit onward is rewritten by the server. Lines of the old
        # plan may still be in flight: a bar's revision must now exceed any seen before.
        for bar in [bar for bar in session.latest if bar >= work.start_bar]:
            self._drop_bar(session, bar)
        for bar in [bar for bar in session.failed if bar >= work.start_bar]:
            del session.failed[bar]
        for bar, revision in session.seen_revision.items():
            if bar >= work.start_bar:
                session.min_revision[bar] = revision + 1
        # The edit goes first. The reported playhead is behind the edit's first bar either
        # way, so the edit takes effect from that bar; but moving the playhead first would
        # widen the server's window over old-plan lines, and it would start rendering one
        # (which cannot be interrupted) just before the edit makes it worthless.
        self._send(session, {"type": "update", "edit": edit.to_payload()})
        self._report_playhead(session, work.start_bar - 1, work)
        self._event(work, "plan_update", edit_seq=session.edit_seq, from_bar=work.start_bar,
                    reported_playhead=session.playhead)

    def _report_playhead(self, session, target, work):
        """Tell the server every bar up to ``target`` is committed; it never moves back."""
        if target <= session.playhead:
            return
        for bar in range(session.playhead + 1, target + 1):
            record = session.accepted.get(bar)
            self._send(session, {"type": "committed", "bar": bar,
                                 "lyric_id": None if record is None else record.lyric_id,
                                 "text": "" if record is None else record.text})
        session.playhead = target
        self._event(work, "playhead_reported", bar=target)
        cutoff = target - _KEEP_BEHIND_BARS
        for table in (session.latest, session.seen_revision, session.min_revision, session.ready,
                      session.accepted, session.failed):
            for bar in [bar for bar in table if bar < cutoff]:
                del table[bar]

    def _send(self, session, message):
        try:
            session.stream.send(message)
        except Exception as error:
            raise ConnectionError(f"steering session send failed: {type(error).__name__}: {error}") from error

    # -- server messages ---------------------------------------------------------

    def _covering(self, session, bar):
        for work in self._live.values():
            if (not work.done and work.request.input_version == session.version
                    and work.start_bar <= bar < work.end_bar):
                return work
        return None

    def _on_message(self, session, message):
        kind = message.get("type")
        if kind == "error":
            self._on_server_error(session, message)
            return
        if kind != "lyric":
            return
        record = BarLyricRecord.from_payload(message["record"])
        if record.session_id != session.remote_id:
            raise ValueError("lyric belongs to another session")
        bar = record.bar
        if bar <= session.playhead - _KEEP_BEHIND_BARS or record.revision <= session.seen_revision.get(bar, -1):
            return
        session.seen_revision[bar] = record.revision
        if (record.revision < session.min_revision.get(bar, 0)
                or record.topic != session.topic or record.flow_template != session.template):
            return  # written for the plan before the latest edit
        self._drop_bar(session, bar)
        session.latest[bar] = record
        work = self._covering(session, bar)
        if work is not None:
            self._announce(session, work, record)
        self._start_fetches(session)

    def _on_server_error(self, session, message):
        bar, attempt, limit = message.get("bar"), message.get("attempt"), message.get("max_attempts")
        text = str(message.get("message", "lyric stream error"))[:512]
        if type(bar) is not int:
            raise RuntimeError(text)  # not about one bar: the session is unusable
        work = self._covering(session, bar)
        if message.get("code") == "lyric_write_failed":
            if (message.get("retryable") is True and type(attempt) is int and type(limit) is int
                    and 0 < attempt < limit):
                if work is not None:
                    self._event(work, "retry_warning", code="lyric_write_failed", bar=bar - work.start_bar,
                                retryable=True, attempt=attempt, max_attempts=limit, message=text)
            else:
                self._bar_failed(session, bar, text)  # the server gave up on this bar
        # Other per-bar errors (a failed render) are settled by the fetch: the server retries on demand.

    def _bar_failed(self, session, bar, reason):
        """No audio will come for ``bar`` under the current plan: fail the request that needs it."""
        self._drop_bar(session, bar)
        session.failed[bar] = reason
        work = self._covering(session, bar)
        if work is not None:
            self._finish(work, "error", message=reason)

    def _announce(self, session, work, record):
        if record.lyric_id in session.announced:
            return
        session.announced.add(record.lyric_id)
        if len(session.announced) > 512:
            session.announced = {item.lyric_id for item in session.latest.values()}
        counts = {}
        for key in ("candidates_requested", "candidates_valid", "candidates_parsed"):
            value = record.provenance.get(key)
            counts[key] = value if type(value) is int and value >= 0 else None
        self._event(work, "candidate_counts", bar=record.bar - work.start_bar, decision_id=record.lyric_id,
                    revision=record.revision, source="server_selected_lyric_provenance", **counts)

    # -- audio -------------------------------------------------------------------

    def _drop_bar(self, session, bar):
        record = session.latest.pop(bar, None)
        session.ready.pop(bar, None)
        if record is not None:
            abandon = session.fetching.pop(record.lyric_id, None)
            if abandon is not None:
                abandon.set()

    def _start_fetches(self, session):
        # Only bars inside the server's audio window: asking for a later one would make
        # the server render it ahead of bars that play sooner.
        for bar in sorted(session.latest):
            record = session.latest[bar]
            delivered = session.accepted.get(bar)
            if (session.playhead < bar <= session.playhead + session.audio_lead
                    and bar not in session.ready and record.lyric_id not in session.fetching
                    and (delivered is None or delivered.lyric_id != record.lyric_id)):
                abandon = Event()
                session.fetching[record.lyric_id] = abandon
                self._fetchers.submit(self._fetch, session, record, abandon)

    def _fetch(self, session, record, abandon):
        """Runs on a fetch worker; reports back through the event queue."""
        deadline = self._clock() + self._timeout

        def alive():
            if self._clock() >= deadline:
                raise TimeoutError("steering bar audio timed out")
            return not session.stopped.is_set() and not abandon.is_set()

        try:
            result = fetch_and_prepare(session.holder.source, session.preparer, record, bpm=session.bpm,
                                       physical_bar=record.bar, clock=self._clock, alive=alive,
                                       pause=session.stopped.wait)
        except Exception as error:
            self._events.put(("failure", session.token, record, error))
            return
        self._events.put(("abandoned", session.token, record) if result is None
                         else ("prepared", session.token, record, *result))

    def _on_prepared(self, session, record, prepared, timings, continuation):
        session.fetching.pop(record.lyric_id, None)
        current = session.latest.get(record.bar)
        if current is None or current.lyric_id != record.lyric_id:
            return
        session.ready[record.bar] = (record, prepared, timings, continuation)
        self._deliver_ready(session)

    def _on_fetch_failure(self, session, record, error):
        session.fetching.pop(record.lyric_id, None)
        current = session.latest.get(record.bar)
        if current is None or current.lyric_id != record.lyric_id or session.stopped.is_set():
            return  # a superseded line is expected to fail
        self._bar_failed(session, record.bar, f"{type(error).__name__}: {error}"[:512])

    def _deliver_ready(self, session):
        for bar in sorted(session.ready):
            work = self._covering(session, bar)
            if work is None:
                continue  # rendered ahead of any request; delivered when one covers it
            record, prepared, timings, continuation = session.ready.pop(bar)
            self._announce(session, work, record)
            relative = bar - work.start_bar
            try:
                span = build_span(work.request, relative, prepared.audio, record=record, timings=timings,
                                  final=len(work.delivered | {relative}) == work.end_bar - work.start_bar)
            except Exception as error:
                self._finish(work, "error", message=f"{type(error).__name__}: {error}"[:512])
                continue
            try:
                accepted = work.on_span(span) is not False
            except Exception as error:
                self._finish(work, "error", message=f"{type(error).__name__}: {error}"[:512])
                continue
            if not accepted:
                # The service has moved on from this request (played, canceled or superseded).
                self._finish(work, "canceled", client_fenced=True,
                             server_cancel_confirmed=False, gpu_cancel_confirmed=False)
                continue
            work.delivered.add(relative)
            session.accepted[bar] = record
            self._continuations.remember(work.request, (bar + 1) * 16, record.text, record.lyric_id)
            if timings:
                extra = {} if continuation is None else {"continuation": continuation}
                self._event(work, "stage_timings", bar=relative, output_id=span.output_id, timings_ms=timings,
                            clock_domains={"client": "client_monotonic_duration",
                                           "server": "server_reported_duration"}, **extra)
            if len(work.delivered) == work.end_bar - work.start_bar and not work.done:
                # Complete, but still live: the service cancels it once it has played.
                work.done = True
                self._event(work, "complete", bars_delivered=len(work.delivered))

    # -- session end ---------------------------------------------------------------

    def _fail_session(self, error):
        message = f"{type(error).__name__}: {error}"[:512]
        for work in list(self._live.values()):
            self._finish(work, "error", message=message)
        self._forget_transports()
        self._end_session(message, failed=True)

    def _end_session(self, reason, *, failed=False):
        session, self._session = self._session, None
        if session is None:
            return
        session.stopped.set()
        for abandon in session.fetching.values():
            abandon.set()

        def cleanup():
            # Closing a WebSocket can take seconds; the next playback must not wait for it.
            sent, errors = False, []
            try:
                session.stream.send({"type": "stop"})
                sent = True
            except Exception as error:
                errors.append(f"server_stop: {type(error).__name__}: {error}"[:512])
            try:
                session.stream.close()
            except Exception as error:
                errors.append(f"lyric_stream: {type(error).__name__}: {error}"[:512])
            self._release_source(session.holder, failed=failed)
            if session.last_work is not None:
                self._event(session.last_work, "session_cleanup", stop_sent=sent, reason=reason,
                            server_cancel_confirmed=False, gpu_cancel_confirmed=False,
                            remote_session_id=session.remote_id, cleanup_errors=errors)

        closer = Thread(target=cleanup, name="inference-v3-steering-close", daemon=True)
        self._closers = [item for item in self._closers if item.is_alive()] + [closer]
        closer.start()

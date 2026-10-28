"""Protocol v3 routes on the render server: the lyric stream session and bar audio.

``WS /v1/rap/sessions/{session_id}`` carries JSON messages:

client → server
  ``start``      tempo, start bar, seed, the scheduled segments, any edits, recent history
  ``update``     one :class:`PlanEdit` (theme and/or flow from a bar onward)
  ``committed``  the bar the client just committed: lyric id (or null), source, text
  ``stop``       end the session

server → client
  ``started``, ``lyric`` (one :class:`BarLyricRecord`), ``superseded``,
  ``audio_ready``, ``error``

``GET /v1/rap/sessions/{session_id}/bars/{bar}/audio?lyric_id=…&wait_ms=…&accept_older=1``
returns the bar package for that lyric (200), an older revision of the bar
(200, ``X-StreamMUSE-Fetch-Status: older_version``), or 202 when nothing is
ready within ``wait_ms``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any

# Module-level imports: FastAPI resolves the route annotations from module globals.
from fastapi import Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.lyric_session import (
    FETCH_FAILED,
    FETCH_OLDER_VERSION,
    FETCH_PENDING,
    FETCH_READY,
    FETCH_SUPERSEDED,
    FETCH_UNKNOWN,
    BarAudioRenderer,
    LyricStreamSession,
)
from streammuse.application.rap.lyric_writer import SequentialLyricWriter
from streammuse.domain.rap import FlowTemplate
from streammuse.domain.rap.lyric_stream import (
    LYRIC_STREAM_SCHEMA_VERSION, LYRIC_STREAM_CAPABILITIES, BOUNDED_TURNS_CAPABILITY,
    TEMPO_UPDATES_CAPABILITY, BarLyricRecord, PlanEdit, TempoUpdate,
)
from streammuse.domain.rap.remote_chunk import _flow_from_payload

MAX_WAIT_MS = 10_000
_MAX_SESSIONS = 8
_LOGGER = logging.getLogger("uvicorn.error")


@dataclass
class _Segment:
    start_bar: int
    bars: int
    topic: str
    flow_template: FlowTemplate


def plan_from_start(payload: Mapping[str, Any]) -> LivePlan:
    """Build the live plan from a ``start`` message's segments and edits."""
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("start requires a non-empty segments array")
    segments = []
    for item in raw_segments:
        if not isinstance(item, Mapping):
            raise ValueError("segments must be objects")
        segment = _Segment(
            int(item["start_bar"]), int(item["bars"]), str(item["topic"]), _flow_from_payload(item["flow_template"])
        )
        if segment.bars <= 0 or segment.start_bar < 0 or not segment.topic.strip():
            raise ValueError("segments need a positive length and a topic")
        segments.append(segment)
    segments.sort(key=lambda item: item.start_bar)
    total = max(item.start_bar + item.bars for item in segments)
    loop = bool(payload.get("loop", True))

    def base(bar: int) -> tuple[str, FlowTemplate]:
        effective = bar % total if loop else min(bar, total - 1)
        for segment in segments:
            if segment.start_bar <= effective < segment.start_bar + segment.bars:
                return segment.topic, segment.flow_template
        return segments[-1].topic, segments[-1].flow_template

    plan = LivePlan(base)
    for edit in payload.get("edits") or ():
        plan = plan.with_edit(PlanEdit.from_payload(edit))
    return plan


class _QueueListener:
    """Bridges session events (from worker threads) onto the WebSocket's event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target: tuple[asyncio.AbstractEventLoop, asyncio.Queue] | None = None

    def attach(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        with self._lock:
            self._target = (loop, queue)

    def detach(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if self._target is not None and self._target[1] is queue:
                self._target = None

    def _send(self, message: dict[str, object]) -> None:
        with self._lock:
            target = self._target
        if target is None:
            return
        loop, queue = target
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except RuntimeError:  # the connection's loop is gone
            pass

    def lyric(self, record: BarLyricRecord) -> None:
        self._send({"type": "lyric", "record": record.to_payload()})

    def superseded(self, bars: tuple[int, ...], edit_seq: int) -> None:
        self._send({"type": "superseded", "bars": list(bars), "edit_seq": edit_seq})

    def audio_ready(self, bar: int, lyric_id: str) -> None:
        self._send({"type": "audio_ready", "bar": bar, "lyric_id": lyric_id})

    def error(self, message: str, *, bar: int | None = None) -> None:
        self._send({"type": "error", "message": message[:512], "bar": bar})


@dataclass
class _Entry:
    session: LyricStreamSession
    listener: _QueueListener
    close_writer: Callable[[], None]


class LyricStreamService:
    """Session registry. ``writer_factory`` returns a writer and a close hook per session."""

    def __init__(
        self,
        *,
        writer_factory: Callable[[], tuple[SequentialLyricWriter, Callable[[], None]]],
        renderer: BarAudioRenderer,
        default_lyric_lead: int = 3,
        default_audio_lead: int = 2,
    ) -> None:
        self._writer_factory = writer_factory
        self._renderer = renderer
        self._lyric_lead = default_lyric_lead
        self._audio_lead = default_audio_lead
        self._lock = threading.Lock()
        self._sessions: dict[str, _Entry] = {}

    def session(self, session_id: str) -> LyricStreamSession | None:
        with self._lock:
            entry = self._sessions.get(session_id)
        return entry.session if entry is not None else None

    def start(self, session_id: str, payload: Mapping[str, Any]) -> tuple[_Entry, bool]:
        """Create the session, or reattach to a live one (a reconnect). Returns (entry, created)."""
        if payload.get("schema_version") != LYRIC_STREAM_SCHEMA_VERSION:
            raise ValueError("unsupported lyric stream schema version")
        capabilities = payload.get("capabilities", [])
        if not isinstance(capabilities, list) or any(not isinstance(value, str) for value in capabilities):
            raise ValueError("capabilities must be an array of strings")
        if (payload.get("end_bar_exclusive") is not None or payload.get("turn_id") is not None
                ) and BOUNDED_TURNS_CAPABILITY not in capabilities:
            raise ValueError("bounded turn capability must be negotiated")
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                snapshot = existing.session.snapshot()
                if (snapshot["turn_id"] != payload.get("turn_id")
                        or snapshot["end_bar_exclusive"] != payload.get("end_bar_exclusive")):
                    raise ValueError("reconnect cannot change the bounded turn identity")
                return existing, False
            if len(self._sessions) >= _MAX_SESSIONS:
                raise ValueError("too many live lyric sessions")
        writer, close_writer = self._writer_factory()
        try:
            history = tuple(
                (int(item["bar"]), str(item["text"])) for item in payload.get("history") or () if str(item.get("text", ""))
            )
            listener = _QueueListener()
            session = LyricStreamSession(
                session_id=session_id,
                tempo_bpm=float(payload["tempo_bpm"]),
                plan=plan_from_start(payload),
                writer=writer,
                renderer=self._renderer,
                listener=listener,
                start_bar=int(payload.get("start_bar", 0)),
                seed=int(payload.get("seed", 0)),
                lyric_lead=int(payload.get("lyric_lead", self._lyric_lead)),
                audio_lead=int(payload.get("audio_lead", self._audio_lead)),
                history=history,
                end_bar_exclusive=payload.get("end_bar_exclusive"),
                turn_id=payload.get("turn_id"),
            )
        except Exception:
            close_writer()
            raise
        entry = _Entry(session, listener, close_writer)
        with self._lock:
            existing = self._sessions.get(session_id)
            full = len(self._sessions) >= _MAX_SESSIONS
            if existing is None and not full:
                self._sessions[session_id] = entry
        if existing is not None or full:
            close_writer()
            if existing is None:
                raise ValueError("too many live lyric sessions")
            snapshot = existing.session.snapshot()
            if (snapshot["turn_id"] != payload.get("turn_id")
                    or snapshot["end_bar_exclusive"] != payload.get("end_bar_exclusive")):
                raise ValueError("reconnect cannot change the bounded turn identity")
            return existing, False
        # Not started here: the caller attaches its listener first, then starts it,
        # so the first lines are not emitted before anyone is listening.
        return entry, True

    def stop(self, session_id: str) -> None:
        with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is not None:
            entry.session.close()
            entry.close_writer()

    def close(self) -> None:
        with self._lock:
            ids = list(self._sessions)
        for session_id in ids:
            self.stop(session_id)


def mount_lyric_stream_routes(app, service: LyricStreamService) -> None:
    @app.get("/v1/rap/capabilities")
    async def capabilities():
        return {"schema_version": LYRIC_STREAM_SCHEMA_VERSION, "capabilities": list(LYRIC_STREAM_CAPABILITIES)}

    @app.websocket("/v1/rap/sessions/{session_id}")
    async def lyric_session(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        entry = None
        negotiated: set[str] = set()

        async def pump() -> None:
            while True:
                message = await queue.get()
                await websocket.send_text(json.dumps(message, separators=(",", ":")))
                if message.get("type") == "tempo_updated" or message.get("operation") == "tempo_update":
                    _LOGGER.info("tempo_control_sent session=%s edit_seq=%s type=%s unix_ns=%s",
                                 session_id, message.get("edit_seq"), message.get("type"), time.time_ns())

        sender = asyncio.create_task(pump())
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                kind = message.get("type")
                try:
                    if kind == "start":
                        entry, created = await run_in_threadpool(service.start, session_id, message)
                        entry.listener.attach(loop, queue)
                        negotiated = set(message.get("capabilities", [])) & set(LYRIC_STREAM_CAPABILITIES)
                        started = {"type": "started", "session_id": session_id, "created": created}
                        if "capabilities" in message:
                            started["capabilities"] = sorted(negotiated)
                        await queue.put(started)
                        if created:
                            entry.session.start()
                        else:
                            # Reconnect: resend the latest line of every live bar.
                            for record in entry.session.latest_records():
                                entry.listener.lyric(record)
                    elif entry is None:
                        await queue.put({"type": "error", "message": "send start first", "bar": None})
                    elif kind == "update":
                        edit = PlanEdit.from_payload(message["edit"])
                        await run_in_threadpool(entry.session.apply_edit, edit)
                    elif kind == "tempo_update":
                        received_ns, processing_started = time.time_ns(), time.monotonic()
                        _LOGGER.info("tempo_control_received session=%s edit_seq=%s unix_ns=%s",
                                     session_id, (message.get("update") or {}).get("edit_seq"), received_ns)
                        if TEMPO_UPDATES_CAPABILITY not in negotiated:
                            raise ValueError("tempo update capability was not negotiated")
                        update = TempoUpdate.from_payload(message["update"])
                        ack = await run_in_threadpool(entry.session.update_tempo, update)
                        ack = {**ack, "server_received_unix_ns": received_ns,
                               "server_processing_ms": (time.monotonic() - processing_started) * 1000}
                        _LOGGER.info("tempo_control_queued session=%s edit_seq=%s revision=%s processing_ms=%.1f",
                                     session_id, update.edit_seq, ack["revision"], ack["server_processing_ms"])
                        await queue.put(ack)
                    elif kind == "committed":
                        await run_in_threadpool(
                            entry.session.committed,
                            int(message["bar"]),
                            message.get("lyric_id"),
                            str(message.get("text", "")),
                        )
                    elif kind == "stop":
                        await run_in_threadpool(service.stop, session_id)
                        entry = None
                        break
                    else:
                        await queue.put({"type": "error", "message": f"unknown message {kind!r}", "bar": None})
                except (KeyError, TypeError, ValueError) as error:
                    failure = {"type": "error", "message": f"{type(error).__name__}: {error}"[:512], "bar": None}
                    if kind == "tempo_update":
                        failure["operation"] = "tempo_update"
                        failure["edit_seq"] = (message.get("update") or {}).get("edit_seq")
                    await queue.put(failure)
        except WebSocketDisconnect:
            pass
        finally:
            if entry is not None:
                entry.listener.detach(queue)
            # Let queued messages flush briefly before the socket goes away.
            await asyncio.sleep(0)
            sender.cancel()

    @app.get("/v1/rap/sessions/{session_id}/bars/{bar}/audio")
    async def bar_audio(
        session_id: str,
        bar: int,
        lyric_id: str = Query(..., min_length=64, max_length=64),
        wait_ms: int = Query(0, ge=0, le=MAX_WAIT_MS),
        accept_older: bool = Query(True),
    ):
        session = service.session(session_id)
        if session is None:
            return JSONResponse({"error": {"code": "unknown_session"}}, status_code=404)
        result = await run_in_threadpool(
            session.fetch_audio, bar, lyric_id, wait_ms / 1000.0, accept_older=accept_older
        )
        headers = {"X-StreamMUSE-Fetch-Status": result.status, "X-StreamMUSE-Requested-Lyric-Id": lyric_id}
        if result.audio is not None and result.status in (FETCH_READY, FETCH_OLDER_VERSION, FETCH_SUPERSEDED, FETCH_FAILED):
            headers["X-StreamMUSE-Lyric-Id"] = result.audio.lyric_id
            headers["X-StreamMUSE-Render-Ms"] = f"{result.audio.render_ms:.3f}"
            return Response(result.audio.package, media_type=result.audio.media_type, headers=headers)
        if result.status == FETCH_UNKNOWN:
            return JSONResponse({"error": {"code": "unknown_lyric"}}, status_code=404, headers=headers)
        if result.status == FETCH_PENDING:
            return Response(status_code=202, headers=headers)
        code = 409 if result.status == FETCH_SUPERSEDED else 503
        return JSONResponse({"error": {"code": result.status}}, status_code=code, headers=headers)

"""Bounded inference requests over the existing v3 lyric/MOSS/local-warp pipeline.

Factories create transports and a BarAudioPreparer. Each bounded session owns its
lyric stream; the bar audio source and the server's capability list are shared by
all sessions of one backend, so a request does not pay for them again. submit,
cancel and prepare never perform network or preparation work on the caller's
thread. silence_factory receives the request and ABSOLUTE destination bar and
returns mixed drums-only PcmAudio. No phoneme splits or sub-bar generation are
advertised by this legacy adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
import copy
from dataclasses import dataclass, field, replace
import hashlib
import logging
from math import isfinite
import queue
from threading import Event, Lock
import time
import uuid

import numpy as np

from streammuse.application.rap.monitoring_payloads import scheduled_syllables_payload
from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate, materialize_flow
from streammuse.domain.rap.inference import AudioSpan, InferenceRequest
from streammuse.domain.rap.lyric_stream import (
    BarLyricRecord, CONTINUATION_HANDOFF_CAPABILITY, LYRIC_STREAM_SCHEMA_VERSION,
)
from streammuse.domain.rap.remote_chunk import _flow_payload, _thaw

_LOGGER = logging.getLogger(__name__)


class ContinuationIndex:
    """Which server lyric each delivered line was, so a later session can continue from it.

    The server keeps the source audio of its recent lines by lyric id. A new
    session names the line just before its first bar; this index finds that id
    either by position (the bar ending where the session starts, for a timeline
    that only moves forward) or by the text of the last performed line.
    """

    def __init__(self, limit=64):
        self._lock, self._limit = Lock(), limit
        self._by_text, self._by_end = OrderedDict(), OrderedDict()

    def remember(self, request, end_tick, text, lyric_id):
        with self._lock:
            for table, key in ((self._by_text, text), (self._by_end, (request.session_id, request.epoch, end_tick))):
                table.pop(key, None)
                table[key] = lyric_id
                while len(table) > self._limit:
                    table.popitem(last=False)

    def before(self, request, start_tick):
        with self._lock:
            if request.config.get("mode") == "steering":
                # Ticks only grow within an epoch here, so position is exact; it also
                # finds a bar that was delivered but has not been performed yet.
                found = self._by_end.get((request.session_id, request.epoch, start_tick))
                if found is not None:
                    return found
            return self._by_text.get(request.history[-1]) if request.history else None


def fetch_and_prepare(source, preparer, record, *, bpm, physical_bar, clock, alive, pause):
    """Fetch one line's audio and prepare it for the bar it will play in.

    ``record`` is the server's line (its own session id and bar numbering);
    ``physical_bar`` is where it plays locally. Returns ``(prepared, timings,
    continuation)``, or ``None`` once ``alive()`` is false. A missing, failed
    or mismatched result raises.
    """
    started = clock()
    while alive():
        fetched = source.fetch(record.session_id, record.bar, record.lyric_id,
                               wait_ms=1000, accept_older=False)
        if not alive():
            return None
        if fetched.status == "pending":
            pause(0.05)
            continue
        if fetched.status != "ready":
            raise RuntimeError(f"remote bar audio: {fetched.status}")
        if (fetched.lyric_id != record.lyric_id or fetched.manifest is None
                or fetched.source_wav is None or fetched.manifest.lyric_id != record.lyric_id
                or fetched.manifest.bar != record.bar
                or fetched.manifest.tempo_bpm != bpm):
            raise ValueError("remote audio identity or tempo mismatch")
        fetched_at = clock()
        slots = materialize_flow(record.flow_template, physical_bar)
        local_record = replace(record, bar=physical_bar, scheduled=tuple(
            replace(item, slot=slot) for item, slot in zip(record.scheduled, slots)))
        local_manifest = replace(fetched.manifest, bar=physical_bar, lyric_id=local_record.lyric_id)
        prepared = preparer.prepare(local_record, local_manifest, fetched.source_wav)
        if prepared.text != record.text:
            raise ValueError("prepared audio text does not match selected lyric")
        if not alive():
            return None
        timings = {"client_fetch_decode_ms": max(0.0, (fetched_at - started) * 1000),
                   "client_prepare_ms": max(0.0, (clock() - fetched_at) * 1000)}
        for key, value in fetched.manifest.stage_timings_ms.items():
            if type(value) in (int, float) and isfinite(value) and value >= 0:
                timings[f"server_{key}_ms"] = float(value)
        warp_ms = getattr(prepared, "render_latency_ms", None)
        if type(warp_ms) in (int, float) and isfinite(warp_ms) and warp_ms >= 0:
            timings["client_warp_ms"] = float(warp_ms)
        continuation = _thaw(fetched.manifest.diagnostics).get("continuation")
        return prepared, timings, continuation if isinstance(continuation, dict) else None
    return None


def build_span(request, bar, audio, *, record=None, timings=None, declared_silence=False, final=False):
    """The AudioSpan for request-relative ``bar``; ``record`` is the server's line, in its own bar numbering."""
    if audio.format.sample_width_bytes != 4:
        raise ValueError("prepared audio must be float32 PCM")
    start = request.start_tick + bar * 16
    frame_at = lambda tick: round(tick * audio.format.sample_rate_hz * 60 / request.bpm / 4)
    frames = frame_at(start + 16) - frame_at(start)
    if abs(audio.frame_count - frames) > 1:
        raise ValueError("prepared audio duration differs from destination coverage")
    width = audio.format.channels * 4
    # Match the existing relocation policy for a one-frame rounding difference.
    data = audio.data[:frames * width].ljust(frames * width, b"\0")
    if not np.all(np.isfinite(np.frombuffer(data, dtype=np.float32))):
        raise ValueError("prepared PCM contains non-finite samples")
    syllables = () if record is None else tuple(
        {**payload, "start_tick": start + payload["tick_in_bar"],
         "duration_ticks": item.slot.duration_ticks, "text": payload["display_label"]}
        for item, payload in zip(
            record.scheduled, scheduled_syllables_payload(record.scheduled, bar=record.bar)))
    return AudioSpan(
        request_id=request.request_id, output_id=f"{request.request_id}:bar:{bar}",
        epoch=request.epoch, input_version=request.input_version, start_tick=start,
        duration_ticks=16, sample_rate=audio.format.sample_rate_hz, channels=audio.format.channels,
        pcm=data, revision=0 if record is None else record.revision, safe_split_ticks=(),
        declared_silence=declared_silence, text="" if record is None else record.text,
        syllables=syllables, stage_timings=timings or {},
        first_vocal_tick=None if record is None else record.scheduled[0].slot.tick - record.bar * 16,
        final=final)


@dataclass
class _SharedSource:
    """The bar audio source every session of a backend fetches through."""

    source: object
    users: int = 0
    retired: bool = False
    closed: bool = False


@dataclass
class _RequestWork:
    request: InferenceRequest
    on_span: object
    on_event: object
    deadline: float
    canceled: Event = field(default_factory=Event)
    delivered: set = field(default_factory=set)
    lyric_ids: dict = field(default_factory=dict)  # request-relative bar -> server lyric id


class RealInferenceBackend:
    """One bounded server session per contiguous non-silent run of bars.

Full bounded lookahead permits incremental delivery without falsely reporting
playback commitment to advance the server cursor. The caller still owns final
session/epoch fencing at playback: a callback already executing when cancel is
called cannot be recalled. close waits for owned workers/transport cleanup.

Between requests the backend keeps what does not belong to one session: the
server's capability list, the bar audio source (and its pooled connections) and,
after ``prepare``, one lyric stream whose WebSocket handshake is already done.
The next session sends its ``start`` on that stream. A failed session drops all
three, so the request after it connects from scratch as before.
    """

    capabilities = {
        "renderer": "moss_lyric_stream", "generation_unit": "bar", "delivery_unit": "bar",
        "commit_quantum_ticks": 16, "min_prefix_ticks": 16, "guard_ms": 0,
        "max_immutable_lookahead_ticks": 16, "ticks_per_beat": 4, "beats_per_bar": 4,
        "max_request_bars": 64, "llm_ttft_available": False,
        "fallback_policy": "none",
        "ttft_unavailable_reason": "v3 publishes complete selected lyrics, not first-token events",
        "server_cancellation": "best_effort_stop; GPU cancellation unconfirmed",
    }

    def __init__(self, *, stream_factory, source_factory, preparer_factory, templates,
                 clock=time.monotonic, silence_factory=None, request_timeout_seconds=90,
                 max_requests=4, fetch_workers=3, spare_max_age_seconds=30.0,
                 warm_connections=2, continuation_index=None, steering_adapter="bounded"):
        if (type(request_timeout_seconds) not in (int, float)
                or not isfinite(request_timeout_seconds) or request_timeout_seconds <= 0):
            raise ValueError("request timeout must be finite and positive")
        if any(type(n) is not int or n <= 0 for n in (max_requests, fetch_workers)):
            raise ValueError("worker limits must be positive integers")
        if (type(spare_max_age_seconds) not in (int, float)
                or not isfinite(spare_max_age_seconds) or spare_max_age_seconds <= 0):
            raise ValueError("spare connection age limit must be finite and positive")
        if type(warm_connections) is not int or warm_connections < 0:
            raise ValueError("warm connection count must be a non-negative integer")
        if steering_adapter not in ("bounded", "session"):
            raise ValueError("steering adapter must be 'bounded' or 'session'")
        self._stream_factory, self._source_factory = stream_factory, source_factory
        self._preparer_factory, self._templates = preparer_factory, templates
        self._clock, self._silence_factory = clock, silence_factory
        self._timeout, self._limit = request_timeout_seconds, max_requests
        self._lock, self._closed = Lock(), False
        self._requests, self._seen = {}, OrderedDict()
        self._workers = ThreadPoolExecutor(max_workers=max_requests, thread_name_prefix="inference-v3")
        self._fetchers = ThreadPoolExecutor(max_workers=fetch_workers, thread_name_prefix="inference-v3-audio")
        self._spare_max_age, self._warm_connections = spare_max_age_seconds, warm_connections
        self._continuations = continuation_index or ContinuationIndex()
        self._server_capabilities = None
        self._source = None
        self._spare = None  # (stream, remote session id, handshake time)
        self._preparing = {}
        self._preparer = ThreadPoolExecutor(max_workers=3, thread_name_prefix="inference-v3-prepare")
        # Which adapter serves steering requests is reported with the other capabilities.
        self.capabilities = {**type(self).capabilities, "steering_adapter": steering_adapter}
        self._steering = None
        if steering_adapter == "session":
            from streammuse.infrastructure.rap.inference_steering import SteeringSessionBackend

            self._steering = SteeringSessionBackend(
                stream_factory=stream_factory, capabilities=self._capabilities_from,
                acquire_source=self._acquire_source, release_source=self._release_source,
                forget_transports=self._forget_transports, fetchers=self._fetchers,
                preparer_factory=preparer_factory, templates=templates,
                continuations=self._continuations, clock=clock,
                request_timeout_seconds=request_timeout_seconds)

    def submit(self, request, on_span, on_event):
        self._validate_request(request)
        if (self._steering is not None and request.config.get("mode") == "steering"
                and not request.flow_ticks):
            # One long session per playback instead of one bounded session per request.
            return self._steering.submit(request, on_span, on_event)
        with self._lock:
            if self._closed:
                raise RuntimeError("inference backend is closed")
            if request.request_id in self._seen:
                return False
            if len(self._requests) >= self._limit:
                raise RuntimeError("inference backend request capacity exhausted")
            work = _RequestWork(copy.deepcopy(request), on_span, on_event, self._clock() + self._timeout)
            self._requests[request.request_id] = work
            self._seen[request.request_id] = None
            if len(self._seen) > 1024:
                self._seen.popitem(last=False)
            self._workers.submit(self._run, work)
        return True

    def cancel(self, request_id):
        with self._lock:
            work = self._requests.get(request_id)
            if work is not None:
                work.canceled.set()
        if work is None and self._steering is not None:
            return self._steering.cancel(request_id)
        return work is not None

    def prepare(self):
        """Get transports ready for the next request; returns at once.

        In the background: learn the server's capabilities, open (or refresh)
        the pooled bar audio connections, and finish a lyric stream handshake.
        Every step is optional. Whatever is missing or stale when the request
        arrives is done then, exactly as without ``prepare``.
        """
        with self._lock:
            if self._closed:
                return False
            for name, job in (("capabilities", self._prepare_capabilities),
                              ("source", self._prepare_source), ("spare", self._prepare_spare)):
                running = self._preparing.get(name)
                if running is None or running.done():
                    self._preparing[name] = self._preparer.submit(job)
        return True

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for work in self._requests.values():
                work.canceled.set()
        if self._steering is not None:
            self._steering.close()  # before the fetch workers and the shared source go away
        self._preparer.shutdown(wait=True, cancel_futures=True)
        self._workers.shutdown(wait=True)
        self._fetchers.shutdown(wait=True, cancel_futures=True)
        self._drop_spare()
        with self._lock:
            holder, self._source = self._source, None
            if holder is not None:
                holder.retired = True
        self._close_if_unused(holder)

    # -- transports kept between requests ---------------------------------------

    def _capabilities_from(self, stream):
        with self._lock:
            known = self._server_capabilities
        if known is None:
            known = frozenset(stream.get_capabilities())
            with self._lock:
                self._server_capabilities = known
        return known

    def _acquire_source(self):
        with self._lock:
            holder = self._source
            if holder is not None:
                holder.users += 1
                return holder
        created = _SharedSource(self._source_factory(), users=1)
        with self._lock:
            if self._source is None:
                self._source = created
                return created
            holder = self._source
            holder.users += 1
        # Another session installed one first; this one was never used.
        created.users, created.retired = 0, True
        self._close_if_unused(created)
        return holder

    def _release_source(self, holder, *, failed=False):
        with self._lock:
            holder.users -= 1
            if failed and self._source is holder:
                # Another session may still be fetching through it; it is closed
                # when the last one lets go.
                self._source = None
                holder.retired = True
        self._close_if_unused(holder)

    def _close_if_unused(self, holder):
        if holder is None:
            return
        with self._lock:
            if not holder.retired or holder.users or holder.closed:
                return
            holder.closed = True
        try:
            holder.source.close()
        except Exception as error:
            _LOGGER.debug("inference audio source close failed: %s", error)

    def _take_spare(self):
        with self._lock:
            spare, self._spare = self._spare, None
        if spare is None:
            return None
        stream, session_id, connected_at = spare
        if self._clock() - connected_at <= self._spare_max_age:
            return stream, session_id
        self._close_quietly(stream)
        return None

    def _drop_spare(self):
        with self._lock:
            spare, self._spare = self._spare, None
        if spare is not None:
            self._close_quietly(spare[0])

    @staticmethod
    def _close_quietly(stream):
        try:
            stream.close()
        except Exception:
            pass

    def _forget_transports(self):
        """After a failed session nothing kept from before it is trusted."""
        with self._lock:
            self._server_capabilities = None
        self._drop_spare()

    def _await_prepared(self, work, name):
        with self._lock:
            running = self._preparing.get(name)
        while running is not None and not running.done() and self._check(work):
            wait_futures([running], timeout=0.05)

    def _prepare_capabilities(self):
        stream = None
        try:
            with self._lock:
                if self._server_capabilities is not None or self._closed:
                    return
            stream = self._stream_factory()
            self._capabilities_from(stream)
        except Exception as error:
            _LOGGER.debug("inference prepare: capabilities unavailable: %s", error)
        finally:
            if stream is not None:
                self._close_quietly(stream)

    def _prepare_source(self):
        holder = None
        try:
            holder = self._acquire_source()
            warm = getattr(holder.source, "warm", None)
            if callable(warm) and self._warm_connections:
                warm(self._warm_connections)
        except Exception as error:
            _LOGGER.debug("inference prepare: audio connections not warmed: %s", error)
        finally:
            if holder is not None:
                self._release_source(holder)

    def _prepare_spare(self):
        stream = None
        try:
            with self._lock:
                spare = self._spare
                # Replace a handshake that would be too old by the time it is used.
                if self._closed or (spare is not None
                                    and self._clock() - spare[2] <= self._spare_max_age / 2):
                    return
            stream = self._stream_factory()
            connect = getattr(stream, "connect", None)
            if not callable(connect):
                return
            session_id = "inference-" + uuid.uuid4().hex
            connect(session_id)
            with self._lock:
                if self._closed:
                    return
                replaced, self._spare = self._spare, (stream, session_id, self._clock())
                stream = None
            if replaced is not None:
                self._close_quietly(replaced[0])
        except Exception as error:
            _LOGGER.debug("inference prepare: early handshake failed: %s", error)
        finally:
            if stream is not None:
                self._close_quietly(stream)

    @staticmethod
    def _validate_request(request):
        if not isinstance(request, InferenceRequest):
            raise ValueError("request must be an InferenceRequest")
        if request.ticks_per_beat != 4 or request.beats_per_bar != 4:
            raise ValueError("v3 supports only 4/4 with four ticks per beat")
        if request.start_tick % 16 or request.duration_ticks % 16:
            raise ValueError("v3 requires whole-bar aligned request coverage")
        if request.duration_ticks > 64 * 16:
            raise ValueError("v3 request exceeds 64-bar bounded lookahead")
        if not all(isinstance(v, str) and v.strip() for v in
                   (request.session_id, request.request_id, request.topic, request.template_id)):
            raise ValueError("request identities, topic and template must be nonempty")

    def _event(self, work, kind, **values):
        work.on_event({"type": kind, "request_id": work.request.request_id,
                       "session_id": work.request.session_id, "epoch": work.request.epoch,
                       "input_version": work.request.input_version, **values})

    def _check(self, work):
        if work.canceled.is_set():
            return False
        if self._clock() >= work.deadline:
            raise TimeoutError("bounded inference request timed out")
        return True

    def _flows(self, request):
        count = request.duration_ticks // 16
        if not request.flow_ticks and request.config.get("mode") != "cycle":
            template = self._templates.get(request.template_id)
            if not isinstance(template, FlowTemplate):
                raise ValueError("unknown flow template")
            return [template] * count
        result = []
        for bar in range(count):
            ticks = sorted({t - bar * 16 for t in request.flow_ticks if bar * 16 <= t < (bar + 1) * 16})
            if not ticks:
                result.append(None)
                continue
            digest = hashlib.sha256(repr(ticks).encode()).hexdigest()[:16]
            result.append(FlowTemplate(
                f"inference-capture-{digest}", "Captured inference flow", 4, 4,
                tuple(FlowSlot(t, end - t, 0.5) for t, end in zip(ticks, ticks[1:] + [16])),
                FlowProvenance("captured", "inference-request")))
        return result

    def _run(self, work):
        try:
            if not self._check(work):
                return
            flows = self._flows(work.request)
            preparer = self._preparer_factory(work.request) if any(flows) else None
            for bar, flow in enumerate(flows):
                if not self._check(work):
                    return
                if flow is None:
                    if self._silence_factory is None:
                        raise ValueError("captured silent bars require a local silence_factory")
                    audio = self._silence_factory(work.request, work.request.start_tick // 16 + bar)
                    self._deliver(work, bar, audio, declared_silence=True)
            bar = 0
            while bar < len(flows) and self._check(work):
                if flows[bar] is None:
                    bar += 1
                    continue
                end = bar + 1
                while end < len(flows) and flows[end] is not None:
                    end += 1
                self._run_session(work, flows, bar, end, preparer)
                bar = end
            if not work.canceled.is_set():
                self._event(work, "complete", bars_delivered=len(work.delivered))
        except Exception as error:
            if not work.canceled.is_set():
                self._event(work, "error", message=f"{type(error).__name__}: {error}"[:512])
        finally:
            if work.canceled.is_set():
                self._event(work, "canceled", client_fenced=True,
                            server_cancel_confirmed=False, gpu_cancel_confirmed=False)
            with self._lock:
                self._requests.pop(work.request.request_id, None)

    def _run_session(self, work, flows, start, end, preparer):
        request = work.request
        session_id = None
        messages = queue.Queue(maxsize=256)
        overflow, stopped = Event(), Event()
        latest, futures, delivered = {}, [], set()
        stream = holder = None
        opened = failed = False
        handoff, handoff_capability = {}, ()

        def enqueue(kind, value):
            if work.canceled.is_set() or stopped.is_set():
                return
            try:
                messages.put_nowait((kind, value))
            except queue.Full:
                overflow.set()

        def start_payload():
            return {
                "type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
                "capabilities": ["bounded_turns_v1", *handoff_capability], "tempo_bpm": request.bpm,
                **handoff,
                "start_bar": start, "end_bar_exclusive": end, "turn_id": session_id,
                "seed": request.config.get("seed", 0), "loop": False,
                "segments": [{"start_bar": bar, "bars": 1, "topic": request.topic,
                              "flow_template": _flow_payload(flows[bar])} for bar in range(start, end)],
                "edits": [], "history": [
                    {"bar": start - len(request.history[-4:]) + i, "text": line}
                    for i, line in enumerate(request.history[-4:])],
                "lyric_lead": end - start, "audio_lead": end - start,
            }

        try:
            # A prepare still in flight finishes sooner than starting over.
            self._await_prepared(work, "capabilities")
            self._await_prepared(work, "spare")
            spare = self._take_spare()
            if spare is not None:
                stream, session_id = spare
            else:
                stream, session_id = self._stream_factory(), "inference-" + uuid.uuid4().hex
            capabilities = self._capabilities_from(stream)
            if "bounded_turns_v1" not in capabilities:
                raise ValueError("renderer must support bounded_turns_v1")
            if not self._check(work):
                return
            # A new session has no previous bar to continue from. Name the line that
            # was delivered just before this one so the voice carries over; a server
            # without the capability gets the same start message as before.
            previous_lyric = self._previous_lyric(work, start)
            if previous_lyric is not None and CONTINUATION_HANDOFF_CAPABILITY in capabilities:
                handoff = {"continue_from": {"lyric_id": previous_lyric}}
                handoff_capability = (CONTINUATION_HANDOFF_CAPABILITY,)
            holder = self._acquire_source()
            source = holder.source
            on_message = lambda message: enqueue("message", message)
            # A partially successful open may have created a server session.
            opened = True
            try:
                stream.open(session_id, start_payload(), on_message)
            except Exception:
                if spare is None:
                    raise
                # The early handshake did not survive; connect the usual way once.
                self._close_quietly(stream)
                stream, session_id = self._stream_factory(), "inference-" + uuid.uuid4().hex
                stream.open(session_id, start_payload(), on_message)
            while self._check(work) and len(delivered) < end - start:
                if overflow.is_set():
                    raise RuntimeError("bounded lyric message queue overflow")
                try:
                    kind, value = messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                if kind == "prepared":
                    record, prepared, timings, continuation = value
                    if latest.get(record.bar) != record.lyric_id:
                        continue
                    self._deliver(work, record.bar, prepared.audio, record=record,
                                  timings=timings, continuation=continuation)
                    delivered.add(record.bar)
                    continue
                if kind == "failure":
                    record, error = value
                    if latest.get(record.bar) == record.lyric_id:
                        raise error
                    continue
                if value.get("type") == "error":
                    attempt, limit, bar = value.get("attempt"), value.get("max_attempts"), value.get("bar")
                    if (value.get("code") == "lyric_write_failed" and value.get("retryable") is True
                            and type(attempt) is int and type(limit) is int and 0 < attempt < limit
                            and type(bar) is int and start <= bar < end):
                        self._event(work, "retry_warning", code="lyric_write_failed", bar=bar,
                                    retryable=True, attempt=attempt, max_attempts=limit,
                                    message=str(value.get("message", "lyric writer retry"))[:512])
                        continue
                    raise RuntimeError(str(value.get("message", "lyric stream error")))
                if value.get("type") == "started":
                    if "bounded_turns_v1" not in value.get("capabilities", ()):
                        raise ValueError("renderer did not negotiate bounded_turns_v1")
                    continue
                if value.get("type") != "lyric":
                    continue
                record = BarLyricRecord.from_payload(value["record"])
                if record.session_id != session_id or not start <= record.bar < end:
                    raise ValueError("lyric belongs to another session or coverage")
                if record.flow_template != flows[record.bar] or record.topic != request.topic:
                    raise ValueError("lyric does not match requested flow/topic")
                prior = latest.get(record.bar)
                if prior == record.lyric_id:
                    continue
                # Revision monotonicity is checked independently of transport order.
                revision_key = (record.bar, "revision")
                if record.revision <= latest.get(revision_key, -1):
                    continue
                latest[record.bar], latest[revision_key] = record.lyric_id, record.revision
                self._candidate_event(work, record)
                futures.append(self._fetchers.submit(
                    self._fetch_prepare, work, stopped, source, preparer, record, enqueue))
        except Exception:
            failed = not work.canceled.is_set()
            raise
        finally:
            stopped.set()
            for future in futures:
                future.cancel()
            sent = False
            cleanup_errors = []
            if stream is not None and opened:
                try:
                    stream.send({"type": "stop"})
                    sent = True
                except Exception as error:
                    cleanup_errors.append(f"server_stop: {type(error).__name__}: {error}"[:512])
            if stream is not None:
                try:
                    stream.close()
                except Exception as error:
                    cleanup_errors.append(f"lyric_stream: {type(error).__name__}: {error}"[:512])
            if failed:
                self._forget_transports()
            if holder is not None:
                self._release_source(holder, failed=failed)
            if opened:
                self._event(work, "session_cleanup", stop_sent=sent,
                            server_cancel_confirmed=False, gpu_cancel_confirmed=False,
                            remote_session_id=session_id, cleanup_errors=cleanup_errors)

    def _candidate_event(self, work, record):
        counts = {}
        for key in ("candidates_requested", "candidates_valid", "candidates_parsed"):
            value = record.provenance.get(key)
            counts[key] = value if type(value) is int and value >= 0 else None
        self._event(work, "candidate_counts", bar=record.bar, decision_id=record.lyric_id,
                    revision=record.revision, source="server_selected_lyric_provenance", **counts)

    def _fetch_prepare(self, work, stopped, source, preparer, record, enqueue):
        try:
            result = fetch_and_prepare(
                source, preparer, record, bpm=work.request.bpm,
                physical_bar=work.request.start_tick // 16 + record.bar, clock=self._clock,
                alive=lambda: not stopped.is_set() and self._check(work), pause=work.canceled.wait)
            if result is not None:
                enqueue("prepared", (record, *result))
        except Exception as error:
            enqueue("failure", (record, error))

    def _previous_lyric(self, work, start):
        """The server lyric id of the line just before request-relative bar ``start``, if known."""
        earlier = [bar for bar in work.lyric_ids if bar < start]
        if earlier:
            return work.lyric_ids[max(earlier)]
        return self._continuations.before(work.request, work.request.start_tick + start * 16)

    def _deliver(self, work, bar, audio, *, record=None, timings=None, declared_silence=False,
                 continuation=None):
        if not self._check(work):
            return
        request = work.request
        start = request.start_tick + bar * 16
        span = build_span(request, bar, audio, record=record, timings=timings,
                          declared_silence=declared_silence,
                          final=len(work.delivered | {bar}) == request.duration_ticks // 16)
        if work.canceled.is_set():
            return
        if work.on_span(span) is False:
            work.canceled.set()
            return
        work.delivered.add(bar)
        if record is not None:
            # ``record`` is the server's line; its id is what the server's source cache is keyed by.
            work.lyric_ids[bar] = record.lyric_id
            self._continuations.remember(request, start + 16, record.text, record.lyric_id)
        if timings:
            # ``continuation`` is the server's own account of whether this bar continued a line.
            extra = {} if continuation is None else {"continuation": continuation}
            self._event(work, "stage_timings", bar=bar, output_id=span.output_id,
                        timings_ms=timings, clock_domains={"client": "client_monotonic_duration",
                                                          "server": "server_reported_duration"}, **extra)

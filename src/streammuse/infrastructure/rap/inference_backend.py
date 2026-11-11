"""Bounded inference requests over the existing v3 lyric/MOSS/local-warp pipeline.

Factories create request-owned transports and a BarAudioPreparer. submit/cancel
never perform network or preparation work. silence_factory receives the request
and ABSOLUTE destination bar and returns mixed drums-only PcmAudio. No phoneme
splits or sub-bar generation are advertised by this legacy adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field, replace
import hashlib
from math import isfinite
import queue
from threading import Event, Lock
import time
import uuid

import numpy as np

from streammuse.application.rap.monitoring_payloads import scheduled_syllables_payload
from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate, materialize_flow
from streammuse.domain.rap.inference import AudioSpan, InferenceRequest
from streammuse.domain.rap.lyric_stream import BarLyricRecord, LYRIC_STREAM_SCHEMA_VERSION
from streammuse.domain.rap.remote_chunk import _flow_payload


@dataclass
class _RequestWork:
    request: InferenceRequest
    on_span: object
    on_event: object
    deadline: float
    canceled: Event = field(default_factory=Event)
    delivered: set = field(default_factory=set)


class RealInferenceBackend:
    """One bounded server session per contiguous non-silent run of bars.

Full bounded lookahead permits incremental delivery without falsely reporting
playback commitment to advance the server cursor. The caller still owns final
session/epoch fencing at playback: a callback already executing when cancel is
called cannot be recalled. close waits for owned workers/transport cleanup.
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
                 max_requests=4, fetch_workers=3):
        if (type(request_timeout_seconds) not in (int, float)
                or not isfinite(request_timeout_seconds) or request_timeout_seconds <= 0):
            raise ValueError("request timeout must be finite and positive")
        if any(type(n) is not int or n <= 0 for n in (max_requests, fetch_workers)):
            raise ValueError("worker limits must be positive integers")
        self._stream_factory, self._source_factory = stream_factory, source_factory
        self._preparer_factory, self._templates = preparer_factory, templates
        self._clock, self._silence_factory = clock, silence_factory
        self._timeout, self._limit = request_timeout_seconds, max_requests
        self._lock, self._closed = Lock(), False
        self._requests, self._seen = {}, OrderedDict()
        self._workers = ThreadPoolExecutor(max_workers=max_requests, thread_name_prefix="inference-v3")
        self._fetchers = ThreadPoolExecutor(max_workers=fetch_workers, thread_name_prefix="inference-v3-audio")

    def submit(self, request, on_span, on_event):
        self._validate_request(request)
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
        return work is not None

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for work in self._requests.values():
                work.canceled.set()
        self._workers.shutdown(wait=True)
        self._fetchers.shutdown(wait=True, cancel_futures=True)

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
        session_id = "inference-" + uuid.uuid4().hex
        messages = queue.Queue(maxsize=256)
        overflow, stopped = Event(), Event()
        latest, futures, delivered = {}, [], set()
        stream = source = None
        opened = False

        def enqueue(kind, value):
            if work.canceled.is_set() or stopped.is_set():
                return
            try:
                messages.put_nowait((kind, value))
            except queue.Full:
                overflow.set()

        try:
            stream = self._stream_factory()
            if "bounded_turns_v1" not in stream.get_capabilities():
                raise ValueError("renderer must support bounded_turns_v1")
            if not self._check(work):
                return
            source = self._source_factory()
            payload = {
                "type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
                "capabilities": ["bounded_turns_v1"], "tempo_bpm": request.bpm,
                "start_bar": start, "end_bar_exclusive": end, "turn_id": session_id,
                "seed": request.config.get("seed", 0), "loop": False,
                "segments": [{"start_bar": bar, "bars": 1, "topic": request.topic,
                              "flow_template": _flow_payload(flows[bar])} for bar in range(start, end)],
                "edits": [], "history": [
                    {"bar": start - len(request.history[-4:]) + i, "text": line}
                    for i, line in enumerate(request.history[-4:])],
                "lyric_lead": end - start, "audio_lead": end - start,
            }
            # A partially successful open may have created a server session.
            opened = True
            stream.open(session_id, payload, lambda message: enqueue("message", message))
            while self._check(work) and len(delivered) < end - start:
                if overflow.is_set():
                    raise RuntimeError("bounded lyric message queue overflow")
                try:
                    kind, value = messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                if kind == "prepared":
                    record, prepared, timings = value
                    if latest.get(record.bar) != record.lyric_id:
                        continue
                    self._deliver(work, record.bar, prepared.audio, record=record,
                                  timings=timings)
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
            for name, resource in (("audio_source", source), ("lyric_stream", stream)):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        cleanup_errors.append(f"{name}: {type(error).__name__}: {error}"[:512])
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
            started = self._clock()
            while not stopped.is_set() and self._check(work):
                fetched = source.fetch(record.session_id, record.bar, record.lyric_id,
                                       wait_ms=1000, accept_older=False)
                if stopped.is_set() or not self._check(work):
                    return
                if fetched.status == "pending":
                    work.canceled.wait(0.05)
                    continue
                if fetched.status != "ready":
                    raise RuntimeError(f"remote bar audio: {fetched.status}")
                if (fetched.lyric_id != record.lyric_id or fetched.manifest is None
                        or fetched.source_wav is None or fetched.manifest.lyric_id != record.lyric_id
                        or fetched.manifest.bar != record.bar
                        or fetched.manifest.tempo_bpm != work.request.bpm):
                    raise ValueError("remote audio identity or tempo mismatch")
                fetched_at = self._clock()
                physical = work.request.start_tick // 16 + record.bar
                slots = materialize_flow(record.flow_template, physical)
                local_record = replace(record, bar=physical, scheduled=tuple(
                    replace(item, slot=slot) for item, slot in zip(record.scheduled, slots)))
                local_manifest = replace(fetched.manifest, bar=physical, lyric_id=local_record.lyric_id)
                prepared = preparer.prepare(local_record, local_manifest, fetched.source_wav)
                if prepared.text != record.text:
                    raise ValueError("prepared audio text does not match selected lyric")
                if stopped.is_set() or not self._check(work):
                    return
                timings = {"client_fetch_decode_ms": max(0.0, (fetched_at - started) * 1000),
                           "client_prepare_ms": max(0.0, (self._clock() - fetched_at) * 1000)}
                for key, value in fetched.manifest.stage_timings_ms.items():
                    if type(value) in (int, float) and isfinite(value) and value >= 0:
                        timings[f"server_{key}_ms"] = float(value)
                warp_ms = getattr(prepared, "render_latency_ms", None)
                if type(warp_ms) in (int, float) and isfinite(warp_ms) and warp_ms >= 0:
                    timings["client_warp_ms"] = float(warp_ms)
                enqueue("prepared", (record, prepared, timings))
                return
        except Exception as error:
            enqueue("failure", (record, error))

    def _deliver(self, work, bar, audio, *, record=None, timings=None, declared_silence=False):
        if not self._check(work):
            return
        request = work.request
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
                record.scheduled, scheduled_syllables_payload(record.scheduled, bar=bar)))
        span = AudioSpan(
            request_id=request.request_id, output_id=f"{request.request_id}:bar:{bar}",
            epoch=request.epoch, input_version=request.input_version, start_tick=start,
            duration_ticks=16, sample_rate=audio.format.sample_rate_hz, channels=audio.format.channels,
            pcm=data, revision=0 if record is None else record.revision, safe_split_ticks=(),
            declared_silence=declared_silence, text="" if record is None else record.text,
            syllables=syllables, stage_timings=timings or {},
            first_vocal_tick=None if record is None else record.scheduled[0].slot.tick - bar * 16,
            final=len(work.delivered | {bar}) == request.duration_ticks // 16)
        if work.canceled.is_set():
            return
        if work.on_span(span) is False:
            work.canceled.set()
            return
        work.delivered.add(bar)
        if timings:
            self._event(work, "stage_timings", bar=bar, output_id=span.output_id,
                        timings_ms=timings, clock_domains={"client": "client_monotonic_duration",
                                                          "server": "server_reported_duration"})

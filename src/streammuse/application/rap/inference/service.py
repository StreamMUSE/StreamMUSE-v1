"""One inference session: musical callback authority, intent and lifecycle control."""

from collections import deque
from dataclasses import asdict
import csv
import copy
import io
import math
from threading import RLock
import time

import numpy as np

from streammuse.application.rap.inference.delivery import SpanBuffer
from streammuse.application.rap.inference.metrics import coverage_summary, latency_summary
from streammuse.domain.rap.inference import InferenceRequest


class InferenceService:
    def __init__(self, *, session_id, backend, clock=time.monotonic, sample_rate=48000,
                 channels=2, drums=None, output=None, ownership_lock=None,
                 claim=None, legacy_state=None, prepare=None):
        self.session_id, self.backend, self.clock = session_id, backend, clock
        self.sample_rate, self.channels = sample_rate, channels
        self.output, self._prepare = output, prepare
        self._lock = ownership_lock or RLock()
        self._claim, self._legacy_state = claim, legacy_state
        self.config = {"mode": "cycle", "bpm": 60.0, "beats_per_bar": 4,
                       "ticks_per_beat": 4, "user_beats": 4, "model_beats": 8,
                       "topic": "creative flow", "template_id": "baseline_straight_9"}
        self._drums = drums
        self.epoch = 0
        self.cycle_id = 0
        self.owner_id = None
        self.phase = "STOPPED"
        self.state = "stopped"
        self.error = None
        self.warning = None
        self._closed = False
        self._phase_frame = 0
        self._phase_started_ms = None
        self._model_started_ms = None
        self._ready_ms = None
        self._released_ms = None
        self._owner_seen = 0.0
        self._owner_lost = False
        self._stop_frame = None
        self._version = 0
        self._version_accepted = {}
        self._playing_version = None
        self._playing = None
        self._taps = set()
        self._events_seen = set()
        self._sequence = 0
        self._history = deque(maxlen=16)
        self._performed = set()
        self._missing = deque(maxlen=128)
        self._span_metrics = deque(maxlen=512)
        self._requests = {}
        self._trace = deque(maxlen=4096)
        self._buffer = self._new_buffer()
        self._metrics = self._new_metrics()
        self._metrics_started_ms = self._ms()
        self._active_wall_ms = 0.0
        self._active_started_ms = None

    def _new_metrics(self):
        return {"llm_ttft_ms": None, "ttft_unavailable_reason": "backend does not observe content tokens",
                "response_start_latency_ms": None, "steering_playback_latency_ms": None,
                "playable_prefix_ready_ms": None, "ready_to_start_overhead_ms": None,
                "first_vocal_onset_ms": None, "missing_duration_ms": 0.0,
                "stale_duration_ms": 0.0, "eligible_duration_ms": 0.0,
                "canceled_count": 0, "error_count": 0, "responses": 0}

    def _new_buffer(self):
        return SpanBuffer(sample_rate=self.sample_rate, channels=self.channels, frame_at=self._frame_at)

    def _frame_at(self, tick):
        return round(tick * self.sample_rate * 60 / self.config["bpm"] / self.config["ticks_per_beat"])

    def _tick_at(self, frame):
        return frame * self.config["bpm"] * self.config["ticks_per_beat"] / (60 * self.sample_rate)

    def _ms(self):
        return self.clock() * 1000

    def _event(self, kind, **payload):
        self._trace.append({"type": kind, "at_ms": self._ms(), "clock_domain": "client_monotonic",
                            "session_id": self.session_id, "epoch": self.epoch, **payload})

    @property
    def active(self):
        return self.state in ("running", "stopping")

    def configure(self, payload, *, _snapshot=True):
        if not isinstance(payload, dict):
            raise ValueError("configuration must be an object")
        with self._lock:
            if self.active:
                raise RuntimeError("configuration changes require stopped inference")
            payload = dict(payload)
            for alias, name in (("user_bars", "user_beats"), ("model_bars", "model_beats")):
                if alias in payload:
                    if name in payload:
                        raise ValueError(f"use either {alias} or {name}")
                    payload[name] = payload.pop(alias) * self.config["beats_per_bar"]
            if set(payload) - set(self.config):
                raise ValueError("unsupported inference setting: " + ", ".join(sorted(set(payload) - set(self.config))))
            candidate = {**self.config, **payload}
            if candidate["mode"] not in ("cycle", "steering"):
                raise ValueError("mode must be cycle or steering")
            if type(candidate["bpm"]) not in (int, float) or not math.isfinite(candidate["bpm"]) or not 30 <= candidate["bpm"] <= 240:
                raise ValueError("bpm must be finite and between 30 and 240")
            for name in ("user_beats", "model_beats", "beats_per_bar", "ticks_per_beat"):
                if type(candidate[name]) is not int or not 1 <= candidate[name] <= 64:
                    raise ValueError(f"{name} must be an integer between 1 and 64")
            if self.backend.capabilities.get("generation_unit") == "bar":
                if candidate["beats_per_bar"] != 4 or candidate["ticks_per_beat"] != 4:
                    raise ValueError("the real adapter supports 4/4 and four ticks per beat")
                if candidate["user_beats"] % 4 or candidate["model_beats"] % 4:
                    raise ValueError("the real adapter requires whole-bar turns")
            self._validate_intent(candidate)
            if self._prepare is not None:
                self._drums = self._prepare(candidate)
            self.config = candidate
            self._buffer = self._new_buffer()
            self._event("configured", config=dict(candidate))
        return self.snapshot() if _snapshot else None

    @staticmethod
    def _validate_intent(payload):
        for name in ("topic", "template_id"):
            if name in payload and (not isinstance(payload[name], str) or not payload[name].strip()
                                    or len(payload[name]) > 2000):
                raise ValueError(f"{name} must be nonempty text of at most 2000 characters")

    def start(self, owner_id=None, *, _snapshot=True):
        with self._lock:
            if self._closed:
                raise RuntimeError("inference service is closed")
            if self.active:
                if self.config["mode"] == "cycle" and owner_id != self.owner_id:
                    raise RuntimeError("capture is owned by another client")
                started = False
            else:
                self._start_locked(owner_id)
                started = True
        try:
            if started and self.output is not None:
                self.output.start(self.audio_callback)
                with self._lock:
                    canceled = not self.active or self._closed
                if canceled:
                    self.output.suspend()
        except Exception as error:
            with self._lock:
                self._finish_stop(str(error), failed=True)
            raise
        return self.snapshot() if _snapshot else None

    def _start_locked(self, owner_id):
        with self._lock:
            if self.config["mode"] == "cycle" and (not isinstance(owner_id, str) or not owner_id):
                raise ValueError("cycle start requires owner_id")
            if self._claim is not None:
                self._claim()
            if self._prepare is not None:
                self._drums = self._prepare(self.config)
            self.epoch += 1
            self.error = None
            self.warning = None
            self.state = "running"
            self._active_started_ms = self._ms()
            self.owner_id = owner_id if self.config["mode"] == "cycle" else None
            self._owner_seen = self.clock()
            self._owner_lost = False
            self._stop_frame = None
            self._playing = self._playing_version = None
            self._performed.clear()
            self._missing.clear()
            self._buffer = self._new_buffer()
            self._ready_ms = self._released_ms = self._model_started_ms = None
            self._phase_frame = 0
            self._phase_started_ms = None
            self._version += 1
            self._version_accepted.clear()
            if self.config["mode"] == "cycle":
                self._begin_user()
            else:
                self.phase = "PREPARING"
                self._phase_started_ms = self._ms()
                self._request(0, self._model_ticks())
            self._event("started", mode=self.config["mode"], owner_id=self.owner_id)

    def _begin_user(self):
        self.phase = "USER"
        self.cycle_id += 1
        self._phase_frame = 0
        self._phase_started_ms = None
        self._taps.clear()
        self._events_seen.clear()
        self._buffer = self._new_buffer()
        self._ready_ms = None
        self._model_started_ms = None
        self._playing = self._playing_version = None
        self._missing.clear()
        for name in ("first_vocal_onset_ms", "response_start_latency_ms",
                     "playable_prefix_ready_ms", "ready_to_start_overhead_ms"):
            self._metrics[name] = None
        self._event("phase", phase="USER", cycle_id=self.cycle_id)

    def _model_ticks(self):
        return self.config["model_beats"] * self.config["ticks_per_beat"]

    def _request(self, start, duration, flow=()):
        self._sequence += 1
        request = InferenceRequest(session_id=self.session_id, epoch=self.epoch,
            request_id=f"{self.session_id}:{self.epoch}:{self._sequence}", input_version=self._version,
            start_tick=start, duration_ticks=duration, bpm=self.config["bpm"],
            ticks_per_beat=self.config["ticks_per_beat"], beats_per_bar=self.config["beats_per_bar"],
            topic=self.config["topic"], template_id=self.config["template_id"], flow_ticks=tuple(flow),
            history=tuple(self._history), config=dict(self.config))
        now = self._ms()
        row = {"request_id": request.request_id, "state": "pending", "epoch": self.epoch,
               "input_version": self._version, "released_ms": now, "ttft_ms": None,
               "ready_ms": None, "latency_ms": None, "context": asdict(request),
               "metrics": {}, "trace": [], "candidates": []}
        row.update(response_start_latency_ms=None, steering_playback_latency_ms=None,
                   complete_latency_ms=None, playable_prefix_ready_ms=None, startup_latency_ms=None,
                   first_valid_candidate_ms=None, first_vocal_onset_ms=None, played_frames=0,
                   missing_frames=0, valid_candidates=None, evaluated_candidates=None,
                   requested_candidates=None, parsed_candidates=None, decisions={})
        row["first_vocal_origin_ms"] = self._version_accepted.get(self._version, now)
        row["first_vocal_origin"] = "steering_edit" if self._version in self._version_accepted else "request_released"
        self._requests[request.request_id] = row
        if len(self._requests) > 128:
            oldest = next(iter(self._requests))
            if self._requests[oldest]["state"] == "pending":
                self.backend.cancel(oldest)
            del self._requests[oldest]
        if self.phase in ("WAIT", "PREPARING"):
            self._released_ms = now
        self._event("request_released", request_id=request.request_id, context=asdict(request))
        try:
            self.backend.submit(request, self.accept_span, self.backend_event)
        except Exception as error:
            self.backend_event({"type": "error", "request_id": request.request_id, "message": str(error)})

    def accept_span(self, span):
        with self._lock:
            row = self._requests.get(span.request_id)
            if (not self.active or self.phase == "USER" or span.epoch != self.epoch or row is None
                    or row["state"] in ("canceled", "played", "completed_missing", "error")):
                self._event("discarded", output_id=span.output_id, reason="canceled_or_old_epoch")
                return False
            if span.input_version != self._version or row["input_version"] != span.input_version:
                self._event("discarded", output_id=span.output_id, reason="superseded_input")
                return False
            context = row["context"]
            if not context["start_tick"] <= span.start_tick < span.end_tick <= context["start_tick"] + context["duration_ticks"]:
                raise ValueError("audio span is outside requested coverage")
            now = self._ms()
            accepted = self._buffer.accept(span, now)
            if not accepted:
                return False
            row["state"] = "ready" if span.final else "preparing"
            row["ready_ms"] = row["ready_ms"] if row["ready_ms"] is not None else now
            row["latency_ms"] = row["ready_ms"] - row["released_ms"]
            row["metrics"].update(span.stage_timings)
            contiguous_end = context["start_tick"]
            for entry in sorted(self._buffer.entries, key=lambda item: item.span.start_tick):
                if entry.span.request_id == span.request_id and entry.span.start_tick <= contiguous_end:
                    contiguous_end = max(contiguous_end, entry.span.end_tick)
            if (row["playable_prefix_ready_ms"] is None and contiguous_end - context["start_tick"]
                    >= self.backend.capabilities.get("min_prefix_ticks", 1)):
                row["playable_prefix_ready_ms"] = now - row["released_ms"]
            if contiguous_end >= context["start_tick"] + context["duration_ticks"]:
                if row["complete_latency_ms"] is None:
                    row["complete_latency_ms"] = now - row["released_ms"]
                row["state"] = "ready"
            self._event("decoded_ready", request_id=span.request_id, output_id=span.output_id,
                        start_tick=span.start_tick, duration_ticks=span.duration_ticks,
                        stage_timings=span.stage_timings)
            prefix = self._buffer.prefix_ticks(self._version)
            if self._ready_ms is None and prefix >= self.backend.capabilities.get("min_prefix_ticks", 1):
                self._ready_ms = now
                if self._released_ms is not None:
                    self._metrics["playable_prefix_ready_ms"] = now - self._released_ms
            return True

    def backend_event(self, payload):
        with self._lock:
            row = self._requests.get(payload.get("request_id"))
            if row is None:
                return
            row["trace"].append(dict(payload))
            row["trace"] = row["trace"][-64:]
            if (row["epoch"] != self.epoch or row["input_version"] != self._version
                    or row["state"] in ("canceled", "played", "completed_missing", "error")):
                self._event("backend_late", payload=dict(payload))
                return
            self._event("backend", payload=dict(payload))
            if payload.get("type") == "candidate_counts":
                decision_id = str(payload.get("decision_id", payload.get("bar")))
                row["decisions"][decision_id] = dict(payload)
                for source, destination in (("candidates_requested", "requested_candidates"),
                        ("candidates_valid", "valid_candidates"), ("candidates_parsed", "parsed_candidates"),
                        ("candidates_evaluated", "evaluated_candidates")):
                    counts = [item.get(source) for item in row["decisions"].values()]
                    row[destination] = sum(counts) if all(type(value) is int and value >= 0 for value in counts) else None
                if payload.get("candidates_valid", 0) and row["first_valid_candidate_ms"] is None:
                    row["first_valid_candidate_ms"] = self._ms() - row["released_ms"]
                    row["first_valid_candidate_scope"] = "client receipt of server-validated decision"
            if payload.get("type") == "stage_timings":
                row["metrics"].setdefault("stages", []).append(dict(payload))
            if payload.get("type") == "error":
                row["state"] = "error"
                self.error = str(payload.get("message", "backend failed"))[:512]
                self._metrics["error_count"] += 1
                if self.phase in ("WAIT", "PREPARING"):
                    self._finish_stop(self.error, failed=True)

    def _cancel_pending(self):
        for request_id, row in self._requests.items():
            if row["state"] not in ("canceled", "error", "played", "completed_missing"):
                row["state"] = "canceled"
                row["cancellation"] = "client_fenced; server cancellation unconfirmed"
                self._metrics["canceled_count"] += 1
                self.backend.cancel(request_id)

    def _finish_stop(self, reason=None, failed=False):
        if self._active_started_ms is not None:
            self._active_wall_ms += max(0, self._ms() - self._active_started_ms)
            self._active_started_ms = None
        self._cancel_pending()
        self.state = "error" if failed else "stopped"
        self.phase = "STOPPED"
        self._phase_frame = 0
        self._stop_frame = None
        if reason:
            self.error = reason
        self._event("stopped", reason=reason)

    def fail(self, message):
        with self._lock:
            if self.active:
                self._metrics["error_count"] += 1
                self._finish_stop(str(message)[:512], failed=True)

    def warn(self, message):
        with self._lock:
            self.warning = str(message)[:512]
            self._event("warning", message=self.warning)

    def stop(self, *, _snapshot=True):
        with self._lock:
            if self.phase in ("WAIT", "PREPARING"):
                self._finish_stop()
            elif self.active:
                self.state = "stopping"
                bar_ticks = self.config["beats_per_bar"] * self.config["ticks_per_beat"]
                bar = math.floor(self._tick_at(self._phase_frame) / bar_ticks)
                self._stop_frame = self._frame_at((bar + 1) * bar_ticks)
        return self.snapshot() if _snapshot else None

    def reset(self, *, _snapshot=True):
        with self._lock:
            self._finish_stop()
            self.epoch += 1
            self.error = None
            self.owner_id = None
            self._history.clear()
            self._performed.clear()
            self._missing.clear()
            self._span_metrics.clear()
            self._requests.clear()
            self._trace.clear()
            self._buffer = self._new_buffer()
            self._metrics = self._new_metrics()
            self._metrics_started_ms = self._ms()
            self._active_wall_ms = 0.0
            self._active_started_ms = None
            self._playing = self._playing_version = None
            self._ready_ms = self._released_ms = self._model_started_ms = None
        return self.snapshot() if _snapshot else None

    def steer(self, payload, *, _snapshot=True):
        if not isinstance(payload, dict) or not payload or set(payload) - {"topic", "template_id"}:
            raise ValueError("steer accepts topic and template_id")
        self._validate_intent(payload)
        with self._lock:
            if self.config["mode"] != "steering":
                raise RuntimeError("live steering requires steering mode")
            self.config.update(payload)
            self._version += 1
            self._version_accepted[self._version] = self._ms()
            self._cancel_pending()
            if self.active:
                quantum = self.backend.capabilities.get("commit_quantum_ticks", 1)
                tick = self._tick_at(max(self._phase_frame, self._buffer.frozen_until))
                start = math.ceil(tick / quantum) * quantum
                if self.phase == "PREPARING":
                    start = 0
                    self._buffer = self._new_buffer()
                    self._ready_ms = None
                self._request(start, self._model_ticks())
            self._event("steered", input_version=self._version, **payload)
        return self.snapshot() if _snapshot else None

    def tap(self, payload, *, _snapshot=True):
        self._tap_locked(payload)
        return self.snapshot() if _snapshot else None

    def _tap_locked(self, payload):
        with self._lock:
            self._lease_check()
            if not isinstance(payload, dict):
                raise ValueError("tap must be an object")
            if self.config["mode"] != "cycle" or self.phase != "USER" or self.state != "running":
                raise RuntimeError("capture is only open during USER")
            if payload.get("owner_id") != self.owner_id or self._owner_lost:
                raise RuntimeError("capture belongs to another client or the lease expired")
            for key, value in (("session_id", self.session_id), ("epoch", self.epoch), ("cycle_id", self.cycle_id)):
                if payload.get(key) != value:
                    raise ValueError(f"tap {key} is stale")
            identity = payload.get("event_id")
            if not isinstance(identity, str) or not identity or len(identity) > 200:
                raise ValueError("tap requires a bounded event_id")
            if identity in self._events_seen:
                return
            elapsed = payload.get("phase_elapsed_ms")
            duration = self.config["user_beats"] * 60000 / self.config["bpm"]
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed < duration:
                raise ValueError("calibrated tap time lies outside USER")
            uncertainty = payload.get("clock_uncertainty_ms", 0)
            if type(uncertainty) not in (int, float) or not math.isfinite(uncertainty) or not 0 <= uncertainty <= 250:
                raise ValueError("capture clock uncertainty must be between 0 and 250 ms")
            if elapsed > self._phase_frame * 1000 / self.sample_rate + max(50, uncertainty):
                raise ValueError("calibrated tap lies in the future")
            if len(self._events_seen) >= 4096:
                raise ValueError("capture event limit reached")
            tick = round(elapsed * self.config["bpm"] * self.config["ticks_per_beat"] / 60000)
            tick = min(tick, self.config["user_beats"] * self.config["ticks_per_beat"] - 1)
            self._events_seen.add(identity)
            self._taps.add(tick)
            self._owner_seen = self.clock()
            self._event("tap", event_id=identity, cycle_id=self.cycle_id, tick=tick,
                        phase_elapsed_ms=elapsed, clock_uncertainty_ms=uncertainty)

    def heartbeat(self, payload, *, _snapshot=True):
        with self._lock:
            self._lease_check()
            if not isinstance(payload, dict) or payload.get("owner_id") != self.owner_id or self._owner_lost:
                raise RuntimeError("heartbeat does not own capture")
            self._owner_seen = self.clock()
        return self.snapshot() if _snapshot else None

    def _lease_check(self):
        if self.active and self.config["mode"] == "cycle" and self.clock() - self._owner_seen > 6:
            self._owner_lost = True
            self.error = "capture owner heartbeat expired"
            if self.phase == "WAIT":
                self._finish_stop(self.error)

    def _drum_slice(self, frame, count):
        if self._drums is None or len(self._drums) == 0:
            return np.zeros((count, self.channels), dtype=np.float32)
        return self._drums[(np.arange(count) + frame) % len(self._drums)]

    def audio_callback(self, frames):
        """Called only by the shared runtime's active device callback.

        The frame cursor, not browser time or a polling timer, owns musical time.
        No network, decoding or generation is performed on this path.
        """
        output = np.zeros((frames, self.channels), dtype=np.float32)
        with self._lock:
            self._lease_check()
            cursor = 0
            while cursor < frames and self.active:
                if self.phase in ("WAIT", "PREPARING"):
                    if self._ready_ms is None:
                        break
                    self.phase = "MODEL"
                    self._phase_frame = 0
                    self._phase_started_ms = self._model_started_ms = self._ms() + cursor * 1000 / self.sample_rate
                    self._metrics["ready_to_start_overhead_ms"] = self._model_started_ms - self._ready_ms
                    if self.config["mode"] == "cycle":
                        self._metrics["response_start_latency_ms"] = self._model_started_ms - self._released_ms
                    for row in self._requests.values():
                        if row["epoch"] == self.epoch and row["input_version"] == self._version:
                            field = "response_start_latency_ms" if self.config["mode"] == "cycle" else "startup_latency_ms"
                            row[field] = self._model_started_ms - row["released_ms"]
                            if field == "startup_latency_ms":
                                self._metrics[field] = row[field]
                    self._metrics["responses"] += 1
                    self._event("model_started", observed_callback_ms=self._model_started_ms)
                if self._phase_started_ms is None:
                    self._phase_started_ms = self._ms() + cursor * 1000 / self.sample_rate
                if self.phase == "USER":
                    end = self._frame_at(self.config["user_beats"] * self.config["ticks_per_beat"])
                    if self._stop_frame is not None:
                        end = min(end, self._stop_frame)
                    count = min(frames - cursor, end - self._phase_frame)
                    output[cursor:cursor + count] = self._drum_slice(self._phase_frame, count)
                    self._phase_frame += count
                    cursor += count
                    if self._phase_frame == end:
                        if self.state == "stopping" or self._owner_lost:
                            self._finish_stop(self.error)
                        elif not self._taps:
                            self.error = "No rhythm captured"
                            self._begin_user()
                        else:
                            user_ticks = self.config["user_beats"] * self.config["ticks_per_beat"]
                            flow = tuple(tick + offset for offset in range(0, self._model_ticks(), user_ticks)
                                         for tick in sorted(self._taps) if tick + offset < self._model_ticks())
                            self.phase = "WAIT"
                            self.error = None
                            self._phase_frame = 0
                            self._phase_started_ms = self._ms() + cursor * 1000 / self.sample_rate
                            self._version += 1
                            self._request(0, self._model_ticks(), flow)
                    continue
                turn_end = (self._frame_at(self._model_ticks()) if self.config["mode"] == "cycle"
                            else self._phase_frame + frames - cursor)
                if self._stop_frame is not None:
                    turn_end = min(turn_end, self._stop_frame)
                entry, until = self._buffer.select(self._phase_frame, self._version,
                    self.backend.capabilities.get("commit_quantum_ticks", 1))
                count = min(frames - cursor, turn_end - self._phase_frame, until - self._phase_frame)
                if count <= 0:
                    break
                duration_ms = count * 1000 / self.sample_rate
                self._metrics["eligible_duration_ms"] += duration_ms
                if entry is None:
                    self._playing = self._playing_version = None
                    output[cursor:cursor + count] = self._drum_slice(self._phase_frame, count)
                    self._metrics["missing_duration_ms"] += duration_ms
                    if self._missing and self._missing[-1]["end_frame"] == self._phase_frame:
                        self._missing[-1]["end_frame"] += count
                    else:
                        self._missing.append({"start_frame": self._phase_frame,
                                              "end_frame": self._phase_frame + count})
                else:
                    span = entry.span
                    offset = self._phase_frame - self._frame_at(span.start_tick)
                    output[cursor:cursor + count] = entry.audio[offset:offset + count]
                    if span.input_version != self._version:
                        self._metrics["stale_duration_ms"] += duration_ms
                    self._playing_version = span.input_version
                    self._playing = span
                    if entry.committed_frame == self._phase_frame:
                        deadline = (self._model_started_ms + self._phase_frame * 1000 / self.sample_rate
                                    if self._phase_frame > 0 else None)
                        commitment = {"request_id": span.request_id, "output_id": span.output_id,
                            "epoch": self.epoch, "input_version": span.input_version,
                            "start_tick": self._tick_at(self._phase_frame),
                            "duration_ticks": self._tick_at(until - self._phase_frame),
                            "start_frame": self._phase_frame, "end_frame": until,
                            "state": "stale" if span.input_version != self._version else "committed",
                            "ready_ms": entry.ready_ms, "committed_ms": self._ms(),
                            "deadline_ms": deadline,
                            "deadline_margin_ms": deadline - entry.ready_ms if deadline is not None else None}
                        self._span_metrics.append(commitment)
                        self._event("committed", reason="audio_callback", **commitment)
                    if span.output_id not in self._performed:
                        self._performed.add(span.output_id)
                        if span.text:
                            self._history.append(span.text)
                        accepted = self._version_accepted.get(span.input_version)
                        if accepted is not None and span.input_version == self._version:
                            self._metrics["steering_playback_latency_ms"] = self._ms() + cursor * 1000 / self.sample_rate - accepted
                            row = self._requests.get(span.request_id)
                            if row is not None:
                                row["steering_playback_latency_ms"] = self._metrics["steering_playback_latency_ms"]
                            self._version_accepted.pop(span.input_version, None)
                    if span.first_vocal_tick is not None:
                        onset = self._frame_at(span.start_tick + span.first_vocal_tick)
                        if self._phase_frame <= onset < self._phase_frame + count:
                            row = self._requests.get(span.request_id)
                            if row is not None and row["first_vocal_onset_ms"] is None:
                                row["first_vocal_onset_ms"] = (self._model_started_ms + onset * 1000 / self.sample_rate
                                                                - row["first_vocal_origin_ms"])
                                if span.input_version == self._version:
                                    self._metrics["first_vocal_onset_ms"] = row["first_vocal_onset_ms"]
                self._account_playback(self._phase_frame, count, entry)
                self._phase_frame += count
                cursor += count
                self._buffer.discard_before(self._phase_frame)
                if self._stop_frame is not None and self._phase_frame >= self._stop_frame:
                    self._finish_stop()
                elif self.config["mode"] == "cycle" and self._phase_frame >= turn_end:
                    self._cancel_pending()
                    if self._owner_lost:
                        self._finish_stop(self.error)
                    else:
                        self._begin_user()
                elif self.config["mode"] == "steering":
                    self._extend_steering()
        return output

    def _account_playback(self, start, count, entry):
        for row in self._requests.values():
            if row["epoch"] != self.epoch or row["state"] in ("canceled", "error", "played", "completed_missing"):
                continue
            context = row["context"]
            first = self._frame_at(context["start_tick"])
            end = self._frame_at(context["start_tick"] + context["duration_ticks"])
            covered = max(0, min(end, start + count) - max(first, start))
            if covered:
                field = "played_frames" if entry is not None and entry.span.request_id == row["request_id"] else "missing_frames"
                row[field] += covered
            if start + count >= end:
                row["state"] = "played" if row["played_frames"] == end - first else "completed_missing"
                self.backend.cancel(row["request_id"])

    def _extend_steering(self):
        rows = [row for row in self._requests.values() if row["epoch"] == self.epoch
                and row["input_version"] == self._version and row["state"] != "canceled"]
        end = max((row["context"]["start_tick"] + row["context"]["duration_ticks"] for row in rows), default=0)
        quantum = self.backend.capabilities.get("commit_quantum_ticks", 1)
        current = math.ceil(self._tick_at(self._phase_frame) / quantum) * quantum
        if end - current <= self._model_ticks() // 2:
            self._request(max(end, current), self._model_ticks())

    def snapshot(self):
        return copy.deepcopy(self._snapshot_data())

    @staticmethod
    def _request_data(row):
        # Contexts and event dictionaries are append-only. Freeze only their
        # mutable containers under the callback lock; deep-copy after release.
        metrics = dict(row["metrics"])
        if "stages" in metrics:
            metrics["stages"] = list(metrics["stages"])
        return {**row, "metrics": metrics, "trace": list(row["trace"]),
                "decisions": dict(row["decisions"]), "candidates": list(row["candidates"])}

    def _snapshot_data(self):
        with self._lock:
            self._lease_check()
            now = self._ms()
            musical_elapsed = self._phase_frame * 1000 / self.sample_rate
            elapsed = (max(0.0, now - self._phase_started_ms) if self.phase in ("WAIT", "PREPARING")
                       and self._phase_started_ms is not None else musical_elapsed)
            total_ticks = (self.config["user_beats"] * self.config["ticks_per_beat"]
                           if self.phase == "USER" else self._model_ticks())
            current = {"tick": self._tick_at(self._phase_frame), "elapsed_ms": elapsed,
                       "clock_domain": "client_monotonic",
                       "start_tick": self._playing.start_tick if self._playing else None,
                       "duration_ticks": self._playing.duration_ticks if self._playing else None,
                       "phase_elapsed_ms": elapsed,
                       "remaining_ticks": max(0, total_ticks - self._tick_at(self._phase_frame))
                           if self.phase in ("USER", "MODEL") and self.config["mode"] == "cycle" else None,
                       "text": self._playing.text if self._playing and self.phase == "MODEL" else "",
                       "syllables": list(self._playing.syllables) if self._playing else [],
                       "flow": [{"tick": tick, "velocity": 1} for tick in sorted(self._taps)],
                       "requested_version": self._version, "playing_version": self._playing_version,
                       "scheduled_start_ms": self._model_started_ms, "observed_start_ms": self._model_started_ms,
                       "status": self.error or self.warning}
            capabilities = {**self.backend.capabilities,
                "startup_buffer_policy": "minimum contiguous decoded prefix",
                "configurable": list(self.config), "audio_controls_supported": False,
                "audio_controls_reason": "inference uses stopped-only shared output-device selection"}
            legacy = self._legacy_state() if self._legacy_state is not None else None
            metrics = dict(self._metrics)
            metrics["response_latency_summary"] = latency_summary(
                row["response_start_latency_ms"] for row in self._requests.values())
            metrics["steering_latency_summary"] = latency_summary(
                row["steering_playback_latency_ms"] for row in self._requests.values())
            known_valid = [decision["candidates_valid"] for row in self._requests.values()
                           for decision in row["decisions"].values()
                           if type(decision.get("candidates_valid")) is int]
            metrics["mean_valid_candidates"] = sum(known_valid) / len(known_valid) if known_valid else None
            metrics["candidate_decision_count"] = len(known_valid)
            metrics["coverage"] = coverage_summary(eligible_ms=metrics["eligible_duration_ms"],
                missing_ms=metrics["missing_duration_ms"], stale_ms=metrics["stale_duration_ms"],
                fallback_ms=0, wall_ms=self._active_wall_ms + (
                    max(0, now - self._active_started_ms) if self._active_started_ms is not None else 0))
            metrics["coverage"]["wall_time_basis"] = "active runs only: USER, WAIT, MODEL, PREPARING; stopped idle excluded"
            if self.backend.capabilities.get("fallback_policy") != "none":
                metrics["coverage"].update(fallback_ms=None, fallback_rate=None,
                    fallback_unavailable_reason="adapter has not declared measured fallback coverage")
            spans = self._buffer.snapshot() + [
                {"span_id": f"missing:{self.epoch}:{self.cycle_id}:{item['start_frame']}",
                 "state": "missing", "start_tick": self._tick_at(item["start_frame"]),
                 "duration_ticks": self._tick_at(item["end_frame"] - item["start_frame"]),
                 "input_version": None, **item} for item in self._missing]
            return {"supported": True, "session_id": self.session_id, "epoch": self.epoch,
                    "active_mode": self.config["mode"], "legacy_mode": legacy,
                    "state": self.state, "phase": self.phase, "cycle_id": self.cycle_id,
                    "owner_id": self.owner_id, "phase_elapsed_ms": elapsed,
                    "config": dict(self.config), "capabilities": capabilities, "current": current,
                    "spans": sorted(spans, key=lambda item: item["start_tick"]),
                    "audio": self.output.snapshot() if self.output is not None and hasattr(self.output, "snapshot") else None,
                    "requests": [self._request_data(row) for row in list(self._requests.values())[-64:]],
                    "metrics": metrics, "error": self.error, "warning": self.warning}

    def export_trace(self):
        with self._lock:
            result = {"schema_version": 1, "clock_domain": "client_monotonic", "units": "milliseconds",
                      "snapshot": self._snapshot_data(), "events": list(self._trace)}
        return copy.deepcopy(result)

    def export_metrics_csv(self):
        with self._lock:
            requests = [self._request_data(row) for row in self._requests.values()]
            spans = list(self._span_metrics)
        buffer = io.StringIO()
        names = ("record_type", "request_id", "epoch", "input_version", "state", "released_ms", "ready_ms",
                 "latency_ms", "complete_latency_ms", "playable_prefix_ready_ms", "ttft_ms",
                 "response_start_latency_ms", "startup_latency_ms", "steering_playback_latency_ms",
                 "first_valid_candidate_ms", "first_vocal_onset_ms", "first_vocal_origin_ms", "first_vocal_origin",
                 "requested_candidates", "parsed_candidates", "valid_candidates", "evaluated_candidates",
                 "output_id", "start_tick", "duration_ticks", "start_frame", "end_frame", "committed_ms",
                 "deadline_ms", "deadline_margin_ms")
        writer = csv.DictWriter(buffer, names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({"record_type": "response", **row} for row in requests)
        writer.writerows({"record_type": "span", **row} for row in spans)
        return buffer.getvalue()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._finish_stop()
            self._closed = True
        if self.output is not None:
            self.output.close()
        self.backend.close()

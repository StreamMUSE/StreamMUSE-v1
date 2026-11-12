"""V3 adapter contracts with deterministic transports and the real bar preparer."""

from dataclasses import replace
import hashlib
import io
from threading import Condition, Event, Thread
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from streammuse.application.rap.bar_audio import BarAudioPreparer, PreparedBarVocal
from streammuse.application.rap.bar_stream import BarAudioFetch
from streammuse.application.rap.monitoring_payloads import scheduled_syllables_payload
from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate, ScheduledSyllable, Syllable, materialize_flow
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, RapMixConfig
from streammuse.domain.rap.inference import InferenceRequest
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.remote_chunk import _flow_from_payload
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.inference_backend import RealInferenceBackend


FORMAT = AudioFormat(sample_rate_hz=100, channels=1)
TEMPLATE = FlowTemplate("test", "Test", 4, 4,
                        tuple(FlowSlot(t, 1, 0.5) for t in (0, 4, 8, 12)),
                        FlowProvenance("test", "test"))


def request(**changes):
    values = dict(session_id="shared", epoch=2, request_id="request-1", input_version=3,
                  start_tick=0, duration_ticks=32, bpm=120, ticks_per_beat=4,
                  beats_per_bar=4, topic="space", template_id="test",
                  flow_ticks=(2, 6, 10, 14, 18, 22, 26, 30), history=("previous line",),
                  config={"mode": "cycle", "seed": 12})
    return InferenceRequest(**{**values, **changes})


def pcm(req, physical_bar, *, audio_format=FORMAT, value=0.25):
    frame_at = lambda tick: round(tick * audio_format.sample_rate_hz * 60 / req.bpm / 4)
    frames = frame_at((physical_bar + 1) * 16) - frame_at(physical_bar * 16)
    return PcmAudio(audio_format, frames,
                    np.full((frames, audio_format.channels), value, dtype=np.float32).tobytes())


def wav(frames=24000):
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(np.full(frames, 1000, dtype="<i2").tobytes())
    return output.getvalue()


class Harness:
    def __init__(self, **options):
        self.changed = Condition()
        self.streams, self.sources, self.prepared, self.spans, self.events = [], [], [], [], []
        self.records, self.fetch_gates = {}, {}
        self.fetch_started = Event()
        self.open_gate = self.stop_gate = None
        self.fetch_status = "ready"
        self.wrong_audio = False
        self.capabilities = {"bounded_turns_v1"}
        self.source_close_error = self.stream_close_error = False
        self.preparer_override = None
        self.capability_calls = self.open_count = 0
        self.supports_connect = False
        self.connect_gate = None
        self.stale_handshakes = set()
        self.warmed = []
        self.backend = RealInferenceBackend(
            stream_factory=self.stream_factory, source_factory=self.source_factory,
            preparer_factory=self.preparer_factory, templates={"test": TEMPLATE},
            silence_factory=options.pop("silence_factory", lambda req, bar: pcm(req, bar, value=0.1)),
            **options)

    def notify(self):
        with self.changed:
            self.changed.notify_all()

    def wait(self, predicate):
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=3), (self.spans, self.events)

    def stream_factory(self):
        harness = self

        class Stream:
            opened = closed = False

            def __init__(self):
                self.sent = []

            connected = None

            def get_capabilities(self):
                harness.capability_calls += 1
                if harness.open_gate is not None:
                    assert harness.open_gate.wait(3)
                return frozenset(harness.capabilities)

            def handshake(self, session_id):
                if harness.connect_gate is not None:
                    assert harness.connect_gate.wait(3)
                self.connected = session_id
                harness.notify()

            def open(self, session_id, start, on_message):
                if self.connected in harness.stale_handshakes:
                    raise ConnectionError("handshake went stale")
                assert self.connected in (None, session_id)
                self.session_id, self.start, self.handler = session_id, start, on_message
                harness.open_count += 1
                self.order = harness.open_count
                self.opened = True
                on_message({"type": "started", "capabilities": ["bounded_turns_v1"]})
                harness.notify()

            def send(self, message):
                if harness.stop_gate is not None:
                    assert harness.stop_gate.wait(3)
                self.sent.append(message)

            def close(self):
                self.closed = True
                harness.notify()
                if harness.stream_close_error:
                    raise RuntimeError("stream close failed")

        stream = Stream()
        if self.supports_connect:
            stream.connect = stream.handshake
        self.streams.append(stream)
        self.notify()
        return stream

    def source_factory(self):
        harness = self

        class Source:
            closed = False

            def fetch(self, session_id, bar, lyric_id, *, wait_ms, accept_older):
                assert not accept_older
                harness.fetch_started.set()
                gate = harness.fetch_gates.get(bar)
                if gate is not None:
                    assert gate.wait(3)
                record = harness.records[lyric_id]
                source = wav()
                manifest = BarAudioManifest(
                    lyric_id, bar, harness.current.bpm, 24000, hashlib.sha256(source).hexdigest(),
                    tuple(0.1 * (i + 1) for i in range(len(record.scheduled))),
                    (0.9,) * len(record.scheduled), {"moss": 11.0, "aligner": 3.0})
                return BarAudioFetch(harness.fetch_status,
                                     lyric_id="wrong" if harness.wrong_audio else lyric_id,
                                     manifest=manifest, source_wav=source, request_ms=7.0)

            def warm(self, connections):
                harness.warmed.append(connections)
                harness.notify()

            def close(self):
                self.closed = True
                harness.notify()
                if harness.source_close_error:
                    raise RuntimeError("source close failed")

        source = Source()
        self.sources.append(source)
        return source

    def preparer_factory(self, req):
        if self.preparer_override is not None:
            return self.preparer_override(req)
        harness = self

        class Preparer:
            def prepare(self, record, manifest, source_wav):
                assert manifest.lyric_id == record.lyric_id and manifest.bar == record.bar
                harness.prepared.append((record, manifest))
                return SimpleNamespace(audio=pcm(req, record.bar), text=record.text,
                                       render_latency_ms=2.0, scheduled=record.scheduled)

        return Preparer()

    def submit(self, req=None):
        self.current = req or request()
        self.backend.submit(self.current, self.on_span, self.on_event)

    def on_span(self, span):
        self.spans.append(span)
        self.notify()
        return True

    def on_event(self, event):
        self.events.append(event)
        self.notify()

    def opened(self, index=0):
        """The index-th stream that carried a session, in the order sessions opened."""
        sessions = lambda: sorted((s for s in self.streams if s.opened), key=lambda s: s.order)
        self.wait(lambda: len(sessions()) > index)
        return sessions()[index]

    def run_prepare(self):
        """Run prepare and wait for its background work."""
        assert self.backend.prepare() is True
        for future in list(self.backend._preparing.values()):
            future.result(timeout=3)

    def record(self, bar, *, revision=0, stream=None):
        stream = stream or self.opened()
        segment = next(s for s in stream.start["segments"]
                       if s["start_bar"] <= bar < s["start_bar"] + s["bars"])
        flow = _flow_from_payload(segment["flow_template"])
        return BarLyricRecord(
            stream.session_id, bar, revision, self.current.topic, flow, "test line",
            tuple(ScheduledSyllable(slot, Syllable("test", i, len(flow.slots), 1, ("AH1",), "test"))
                  for i, slot in enumerate(materialize_flow(flow, bar))), 0.5,
            {"candidates_requested": 32, "candidates_valid": 7})

    def emit(self, bar, *, record=None, stream=None):
        stream = stream or self.opened()
        record = record or self.record(bar, stream=stream)
        self.records[record.lyric_id] = record
        stream.handler({"type": "lyric", "record": record.to_payload()})
        return record

    def close(self):
        for gate in (self.open_gate, self.stop_gate, *self.fetch_gates.values()):
            if gate is not None:
                gate.set()
        self.backend.close()


@pytest.fixture
def harness():
    value = Harness()
    yield value
    value.close()


def test_submit_is_nonblocking_and_preserves_bounded_context(harness):
    harness.open_gate = Event()
    returned = Event()
    thread = Thread(target=lambda: (harness.submit(), returned.set()))
    thread.start()
    try:
        assert returned.wait(0.5), "submit blocked on network preflight"
    finally:
        harness.open_gate.set()
        thread.join(3)
    stream = harness.opened()
    assert stream.start["tempo_bpm"] == 120 and stream.start["loop"] is False
    assert stream.start["end_bar_exclusive"] == 2
    assert stream.start["audio_lead"] == stream.start["lyric_lead"] == 2
    assert stream.start["history"][-1]["text"] == "previous line"
    assert stream.start["seed"] == 12
    assert all(s["topic"] == "space" for s in stream.start["segments"])
    assert harness.backend.capabilities["commit_quantum_ticks"] == 16
    assert harness.backend.capabilities["generation_unit"] == "bar"
    assert harness.backend.capabilities["fallback_policy"] == "none"


def test_first_bar_emits_before_remaining_turn_and_closes_session_on_completion(harness):
    harness.submit()
    harness.emit(0)
    harness.wait(lambda: len(harness.spans) == 1)
    first = harness.spans[0]
    assert (first.request_id, first.epoch, first.input_version) == ("request-1", 2, 3)
    assert (first.start_tick, first.duration_ticks, first.frame_count) == (0, 16, 200)
    assert first.first_vocal_tick == 2 and first.safe_split_ticks == ()
    assert not first.final
    harness.emit(1)
    harness.wait(lambda: len(harness.spans) == 2 and all(s.closed for s in harness.streams))
    assert harness.spans[-1].start_tick == 16 and harness.spans[-1].final
    assert harness.streams[0].sent == [{"type": "stop"}]
    # The audio source outlives the session: the next turn reuses its connections.
    assert len(harness.sources) == 1 and not harness.sources[0].closed
    harness.backend.close()
    assert harness.sources[0].closed


def test_out_of_order_ready_bars_do_not_block_behind_slow_earlier_fetch(harness):
    harness.fetch_gates[0] = Event()
    harness.submit()
    harness.emit(0)
    harness.emit(1)
    harness.wait(lambda: len(harness.spans) == 1)
    assert harness.spans[0].start_tick == 16
    harness.fetch_gates[0].set()
    harness.wait(lambda: len(harness.spans) == 2)
    assert harness.spans[1].start_tick == 0


@pytest.mark.parametrize("start_tick", [0, 64])
def test_syllables_include_absolute_geometry_and_preserve_scheduled_payload(harness, start_tick):
    harness.submit(request(start_tick=start_tick))
    records = {bar: harness.emit(bar) for bar in (1, 0)}
    harness.wait(lambda: len(harness.spans) == 2)
    for span in harness.spans:
        bar = (span.start_tick - start_tick) // 16
        record = records[bar]
        legacy = scheduled_syllables_payload(record.scheduled, bar=bar)
        assert len(span.syllables) == len(record.scheduled)
        for payload, original, scheduled in zip(span.syllables, legacy, record.scheduled):
            assert {key: payload[key] for key in original} == original
            assert payload["start_tick"] == span.start_tick + original["tick_in_bar"]
            assert payload["duration_ticks"] == scheduled.slot.duration_ticks
            assert payload["text"] == original["display_label"]
            assert span.start_tick <= payload["start_tick"]
            assert payload["start_tick"] + payload["duration_ticks"] <= span.start_tick + 16


def test_duplicate_record_has_one_decision_and_one_delivery(harness):
    harness.submit()
    record = harness.emit(0)
    harness.emit(0, record=record)
    harness.emit(1)
    harness.wait(lambda: all(s.closed for s in harness.streams))
    assert len(harness.spans) == 2
    counts = [e for e in harness.events if e["type"] == "candidate_counts"]
    assert len(counts) == 2
    assert counts[0]["candidates_requested"] == 32 and counts[0]["candidates_valid"] == 7
    assert counts[0]["candidates_parsed"] is None


def test_measured_client_and_server_durations_are_distinct(harness):
    harness.submit()
    harness.emit(0)
    harness.wait(lambda: bool(harness.spans))
    timings = harness.spans[0].stage_timings
    assert timings["server_moss_ms"] == 11 and timings["server_aligner_ms"] == 3
    assert timings["client_fetch_decode_ms"] >= 0 and timings["client_prepare_ms"] >= 0
    assert timings["client_warp_ms"] == 2
    assert "llm_ttft_ms" not in timings and "network_ms" not in timings


def test_cancel_never_blocks_on_network_and_fences_late_fetch(harness):
    harness.fetch_gates[0] = Event()
    harness.stop_gate = Event()
    harness.submit()
    harness.emit(0)
    assert harness.fetch_started.wait(1)
    returned = Event()
    thread = Thread(target=lambda: (harness.backend.cancel("request-1"), returned.set()))
    thread.start()
    try:
        assert returned.wait(0.5), "cancel performed blocking network work"
    finally:
        harness.stop_gate.set()
        harness.fetch_gates[0].set()
        thread.join(3)
    harness.wait(lambda: any(e["type"] == "canceled" for e in harness.events))
    cancellation = next(e for e in harness.events if e["type"] == "canceled")
    assert cancellation["client_fenced"] is True
    assert cancellation["server_cancel_confirmed"] is False
    assert cancellation["gpu_cancel_confirmed"] is False
    harness.wait(lambda: all(s.closed for s in harness.streams))
    assert harness.spans == []


@pytest.mark.parametrize("bad", ["wrong_audio", "failed", "wrong_session", "wrong_flow"])
def test_invalid_remote_result_is_an_error_not_playable_audio(harness, bad):
    harness.wrong_audio = bad == "wrong_audio"
    if bad == "failed":
        harness.fetch_status = "failed"
    harness.submit()
    record = harness.record(0)
    if bad == "wrong_session":
        record = replace(record, session_id="another-session")
    if bad == "wrong_flow":
        record = replace(record, flow_template=TEMPLATE, scheduled=tuple(
            replace(item, slot=slot) for item, slot in zip(record.scheduled, materialize_flow(TEMPLATE, 0))))
    harness.emit(0, record=record)
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    assert harness.spans == []


def test_disconnect_is_reported_without_inventing_fallback(harness):
    harness.submit()
    harness.opened().handler({"type": "error", "message": "lyric stream disconnected"})
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    assert harness.spans == []


def test_retryable_writer_failure_preserves_inflight_prefix_and_recovers(harness):
    harness.fetch_gates[0] = Event()
    harness.submit()
    harness.emit(0)
    assert harness.fetch_started.wait(1)
    stream = harness.opened()
    stream.handler({"type": "error", "message": "no_valid_candidates", "bar": 1,
                    "code": "lyric_write_failed", "retryable": True,
                    "attempt": 1, "max_attempts": 3})
    harness.wait(lambda: any(e["type"] == "retry_warning" for e in harness.events))
    notification = next(e for e in harness.events if e["type"] == "retry_warning")
    assert notification["bar"] == 1 and notification["attempt"] == 1
    assert not stream.closed and not any(e["type"] == "error" for e in harness.events)
    harness.fetch_gates[0].set()
    harness.wait(lambda: len(harness.spans) == 1)
    assert harness.spans[0].start_tick == 0
    harness.emit(1)
    harness.wait(lambda: any(e["type"] == "complete" for e in harness.events))
    assert [span.start_tick for span in harness.spans] == [0, 16]


@pytest.mark.parametrize("fields", [
    {"code": "lyric_write_failed", "retryable": False, "attempt": 3, "max_attempts": 3},
    {"code": "protocol_error", "retryable": True, "attempt": 1, "max_attempts": 3},
    {"code": "lyric_write_failed", "retryable": "true", "attempt": 1, "max_attempts": 3},
    {"code": "lyric_write_failed", "retryable": True, "attempt": 3, "max_attempts": 3},
])
def test_exhausted_or_untrusted_errors_remain_fatal(harness, fields):
    harness.submit()
    harness.opened().handler({"type": "error", "message": "failed", "bar": 1, **fields})
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    assert not any(e["type"] == "retry_warning" for e in harness.events)


def test_retryable_writer_notification_does_not_extend_request_timeout():
    harness = Harness(request_timeout_seconds=0.15)
    try:
        harness.submit()
        harness.opened().handler({"type": "error", "message": "no_valid_candidates", "bar": 0,
                                  "code": "lyric_write_failed", "retryable": True,
                                  "attempt": 1, "max_attempts": 3})
        harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
        assert any(e["type"] == "retry_warning" for e in harness.events)
        assert "timed out" in next(e["message"] for e in harness.events if e["type"] == "error")
    finally:
        harness.close()


def test_absolute_relocation_preserves_fractional_frame_geometry(harness):
    harness.submit(request(start_tick=48, bpm=137, duration_ticks=16, flow_ticks=(2, 6, 10, 14)))
    harness.emit(0)
    harness.wait(lambda: bool(harness.spans))
    result = harness.spans[0]
    assert result.start_tick == 48
    assert result.frame_count == round(64 * 100 * 60 / 137 / 4) - round(48 * 100 * 60 / 137 / 4)
    assert harness.prepared[0][0].bar == harness.prepared[0][1].bar == 3
    assert harness.prepared[0][0].scheduled[0].slot.tick == 50


def test_empty_capture_emits_declared_silence_without_remote_generation(harness):
    harness.submit(request(flow_ticks=()))
    harness.wait(lambda: len(harness.spans) == 2)
    assert harness.streams == []
    assert all(s.declared_silence and s.first_vocal_tick is None for s in harness.spans)
    assert all(np.allclose(np.frombuffer(s.pcm, dtype=np.float32), 0.1) for s in harness.spans)


def test_silent_leading_bar_does_not_shift_later_capture(harness):
    harness.submit(request(flow_ticks=(18, 22, 26, 30)))
    harness.wait(lambda: bool(harness.spans))
    assert harness.spans[0].start_tick == 0 and harness.spans[0].declared_silence
    stream = harness.opened()
    assert stream.start["start_bar"] == 1 and stream.start["end_bar_exclusive"] == 2
    harness.emit(1)
    harness.wait(lambda: len(harness.spans) == 2)
    assert harness.spans[-1].start_tick == 16 and harness.spans[-1].first_vocal_tick == 2


def test_steering_without_explicit_flow_uses_selected_template(harness):
    harness.submit(request(flow_ticks=(), config={"mode": "steering"}))
    stream = harness.opened()
    assert _flow_from_payload(stream.start["segments"][0]["flow_template"]) == TEMPLATE


@pytest.mark.parametrize("changes", [{"ticks_per_beat": 7}, {"beats_per_bar": 3},
                                     {"start_tick": 1}, {"duration_ticks": 17}])
def test_unsupported_fine_grained_geometry_is_rejected(harness, changes):
    with pytest.raises(ValueError):
        harness.backend.submit(request(flow_ticks=(), **changes), harness.on_span, harness.on_event)
    assert harness.streams == []


def test_close_is_idempotent_and_rejects_new_work(harness):
    harness.submit()
    harness.opened()
    harness.backend.close()
    harness.backend.close()
    assert all(s.closed for s in harness.streams)
    with pytest.raises(RuntimeError):
        harness.submit(request(request_id="after-close"))


def test_timeout_is_visible_and_releases_session():
    harness = Harness(request_timeout_seconds=0.1)
    try:
        harness.submit()
        harness.opened()
        harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
        assert "tim" in next(e["message"] for e in harness.events if e["type"] == "error").lower()
        harness.wait(lambda: all(s.closed for s in harness.streams))
    finally:
        harness.close()


def test_existing_bar_preparer_keeps_captured_leading_rest(harness):
    class Warper:
        def warp(self, request, source_wav, source_onsets, *, target_frame_count):
            return PreparedBarVocal(wav(target_frame_count), 4.0)

    class Drums:
        def render(self, template, tempo, audio_format, bar):
            frames = round((bar + 1) * 4 * 60 / tempo.bpm * 48000) - round(bar * 4 * 60 / tempo.bpm * 48000)
            return PcmAudio(audio_format, frames, bytes(frames * 8))

    harness.preparer_override = lambda req: BarAudioPreparer(
        warper=Warper(), drums=Drums(), tempo=Tempo(req.bpm, 4, 4), audio_format=AudioFormat(),
        mix_config=RapMixConfig(vocal_gain=1, drum_gain=0))
    harness.submit(request(duration_ticks=16, flow_ticks=(2, 6, 10, 14)))
    harness.emit(0)
    harness.wait(lambda: bool(harness.spans))
    samples = np.frombuffer(harness.spans[0].pcm, dtype=np.float32).reshape(-1, 2)
    assert harness.spans[0].frame_count == 96000
    assert not samples[:12000].any() and samples[12100:].any()


def test_superseded_revision_fetch_cannot_emit_after_newer_record(harness):
    harness.fetch_gates[0] = Event()
    harness.submit(request(duration_ticks=16, flow_ticks=(2, 6, 10, 14)))
    harness.emit(0)
    harness.emit(0, record=harness.record(0, revision=1))
    harness.wait(lambda: len([e for e in harness.events if e["type"] == "candidate_counts"]) == 2)
    harness.fetch_gates[0].set()
    harness.wait(lambda: all(s.closed for s in harness.streams))
    assert len(harness.spans) == 1 and harness.spans[0].revision == 1


def test_duplicate_request_submission_does_not_create_another_remote_session(harness):
    harness.submit()
    harness.opened()
    assert harness.backend.submit(harness.current, harness.on_span, harness.on_event) is False
    assert len(harness.streams) == 1


def test_capacity_is_bounded_without_a_hidden_unbounded_request_queue():
    harness = Harness(max_requests=1)
    try:
        harness.submit()
        harness.opened()
        with pytest.raises(RuntimeError, match="capacity"):
            harness.backend.submit(request(request_id="another"), harness.on_span, harness.on_event)
        assert len(harness.streams) == 1
    finally:
        harness.close()


def test_old_server_without_bounded_turns_is_rejected_before_open(harness):
    harness.capabilities.clear()
    harness.submit()
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    assert not harness.streams[0].opened
    harness.wait(lambda: harness.streams[0].closed)


def test_cancellation_during_preflight_never_starts_a_session(harness):
    harness.open_gate = Event()
    harness.submit()
    harness.wait(lambda: bool(harness.streams))
    harness.backend.cancel("request-1")
    harness.open_gate.set()
    harness.wait(lambda: any(e["type"] == "canceled" for e in harness.events))
    assert not harness.streams[0].opened and harness.streams[0].closed


def test_silent_bars_require_explicit_local_drum_composition():
    harness = Harness(silence_factory=None)
    try:
        harness.submit(request(flow_ticks=()))
        harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
        assert harness.spans == [] and harness.streams == []
    finally:
        harness.close()


def test_cleanup_reports_a_failing_socket_close_and_survives_a_failing_source_close(harness):
    harness.source_close_error = harness.stream_close_error = True
    harness.submit(request(duration_ticks=16, flow_ticks=(2, 6, 10, 14)))
    harness.emit(0)
    harness.wait(lambda: any(e["type"] in ("error", "session_cleanup") for e in harness.events))
    assert harness.streams[0].closed
    cleanup = next(e for e in harness.events if e["type"] == "session_cleanup")
    assert cleanup["cleanup_errors"] and cleanup["stop_sent"] is True
    harness.wait(lambda: any(e["type"] == "complete" for e in harness.events))
    harness.backend.close()  # the shared source's close error stays inside the backend
    assert harness.sources[0].closed


def test_middle_silent_bar_uses_separate_bounded_runs_not_invented_vocals(harness):
    harness.submit(request(duration_ticks=48, flow_ticks=(2, 6, 10, 14, 34, 38, 42, 46)))
    stream = harness.opened()
    assert (stream.start["start_bar"], stream.start["end_bar_exclusive"]) == (0, 1)
    harness.emit(0)
    next_stream = harness.opened(1)
    assert stream.closed and stream.session_id != next_stream.session_id
    assert (next_stream.start["start_bar"], next_stream.start["end_bar_exclusive"]) == (2, 3)
    harness.emit(2, stream=next_stream)
    harness.wait(lambda: len(harness.spans) == 3)
    assert {s.start_tick for s in harness.spans} == {0, 16, 32}
    assert next(s for s in harness.spans if s.start_tick == 16).declared_silence


@pytest.mark.parametrize("fault", ["nan", "length", "text"])
def test_corrupt_prepared_audio_never_becomes_playable(harness, fault):
    def factory(req):
        def prepare(record, manifest, source):
            audio = pcm(req, record.bar, value=float("nan") if fault == "nan" else 0.25)
            if fault == "length":
                audio = PcmAudio(audio.format, audio.frame_count - 2, audio.data[:-8])
            return SimpleNamespace(audio=audio, text="wrong" if fault == "text" else record.text)
        return SimpleNamespace(prepare=prepare)
    harness.preparer_override = factory
    harness.submit(request(duration_ticks=16, flow_ticks=(2, 6, 10, 14)))
    harness.emit(0)
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    assert harness.spans == []


def turn(harness, request_id, **changes):
    """One complete two-bar turn; returns the stream that carried it."""
    before = harness.open_count
    harness.submit(request(request_id=request_id, **changes))
    stream = harness.opened(before)
    for bar in (0, 1):
        harness.emit(bar, stream=stream)
    harness.wait(lambda: any(e["type"] == "complete" and e["request_id"] == request_id
                             for e in harness.events))
    return stream


def test_turns_share_one_capability_lookup_and_one_audio_source(harness):
    first = turn(harness, "turn-1")
    second = turn(harness, "turn-2")
    assert first is not second and first.session_id != second.session_id
    assert harness.capability_calls == 1
    assert len(harness.sources) == 1 and not harness.sources[0].closed
    assert len(harness.spans) == 4


def test_prepare_returns_at_once_and_the_next_session_uses_the_early_handshake(harness):
    harness.supports_connect = True
    harness.connect_gate = harness.open_gate = Event()
    returned = Event()
    thread = Thread(target=lambda: (harness.backend.prepare(), returned.set()))
    thread.start()
    try:
        assert returned.wait(0.5), "prepare blocked on network work"
    finally:
        harness.connect_gate.set()
        thread.join(3)
    harness.run_prepare()
    spare = next(s for s in harness.streams if s.connected)
    assert not spare.opened and harness.warmed == [2]
    stream = turn(harness, "turn-1")
    assert stream is spare and stream.session_id == spare.connected
    assert stream.start["turn_id"] == spare.connected
    assert harness.capability_calls == 1
    # No second stream was needed for the session itself.
    assert [s for s in harness.streams if s.opened] == [spare]


def test_request_arriving_during_prepare_waits_for_the_handshake_in_flight(harness):
    harness.supports_connect = True
    harness.connect_gate = Event()
    assert harness.backend.prepare() is True
    harness.submit()
    harness.wait(lambda: any(s.connected is None and not s.closed for s in harness.streams))
    assert not any(s.opened for s in harness.streams)
    harness.connect_gate.set()
    stream = harness.opened()
    assert stream.connected == stream.session_id


def test_repeated_prepare_keeps_a_recent_handshake_and_refreshes_connections(harness):
    harness.supports_connect = True
    harness.run_prepare()
    harness.run_prepare()
    assert len([s for s in harness.streams if s.connected]) == 1
    assert harness.warmed == [2, 2] and len(harness.sources) == 1


def test_stale_early_handshake_falls_back_to_a_fresh_connection(harness):
    harness.supports_connect = True
    harness.run_prepare()
    spare = next(s for s in harness.streams if s.connected)
    harness.stale_handshakes.add(spare.connected)
    stream = turn(harness, "turn-1")
    assert stream is not spare and spare.closed and not spare.opened
    assert stream.start["turn_id"] == stream.session_id != spare.connected
    assert not any(e["type"] == "error" for e in harness.events)


def test_early_handshake_older_than_the_limit_is_closed_not_used():
    now = [100.0]
    harness = Harness(clock=lambda: now[0], spare_max_age_seconds=30)
    try:
        harness.supports_connect = True
        harness.run_prepare()
        spare = next(s for s in harness.streams if s.connected)
        now[0] += 31
        stream = turn(harness, "turn-1")
        assert stream is not spare and spare.closed and not spare.opened
    finally:
        harness.close()


def test_failed_session_forgets_capabilities_source_and_handshake(harness):
    harness.supports_connect = True
    harness.run_prepare()
    harness.submit()
    harness.opened().handler({"type": "error", "message": "lyric stream disconnected"})
    harness.wait(lambda: any(e["type"] == "error" for e in harness.events))
    harness.wait(lambda: harness.sources[0].closed)
    harness.run_prepare()
    assert harness.capability_calls == 2 and len(harness.sources) == 2
    stream = turn(harness, "turn-2")
    assert stream.connected == stream.session_id
    assert not harness.sources[1].closed


def test_canceled_session_keeps_shared_transports(harness):
    harness.submit()
    harness.opened()
    harness.backend.cancel("request-1")
    harness.wait(lambda: any(e["type"] == "canceled" for e in harness.events))
    turn(harness, "turn-2")
    assert harness.capability_calls == 1 and len(harness.sources) == 1
    assert not harness.sources[0].closed


def test_close_closes_an_unused_early_handshake_and_rejects_prepare(harness):
    harness.supports_connect = True
    harness.run_prepare()
    spare = next(s for s in harness.streams if s.connected)
    harness.backend.close()
    assert spare.closed and all(s.closed for s in harness.sources)
    assert harness.backend.prepare() is False


def test_prepare_without_early_handshake_support_keeps_the_request_path(harness):
    harness.run_prepare()
    assert not any(s.connected for s in harness.streams)
    stream = turn(harness, "turn-1")
    assert stream.connected is None and harness.capability_calls == 1


def test_prepare_failures_are_silent_and_the_request_still_runs(harness):
    harness.supports_connect = True

    def broken():
        raise OSError("renderer unreachable")

    original = harness.backend._stream_factory
    harness.backend._stream_factory = broken
    harness.run_prepare()
    harness.backend._stream_factory = original
    turn(harness, "turn-1")
    assert not any(e["type"] == "error" for e in harness.events)

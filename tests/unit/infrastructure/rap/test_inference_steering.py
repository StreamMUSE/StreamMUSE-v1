"""The long-session steering adapter, with deterministic transports."""

from threading import Event, Thread
import time

import pytest

from streammuse.domain.rap import ScheduledSyllable, Syllable, materialize_flow
from streammuse.domain.rap.lyric_stream import BarLyricRecord, PlanEdit
from streammuse.domain.rap.remote_chunk import _flow_from_payload, _flow_payload
from streammuse.infrastructure.rap.inference_steering import session_leads

from tests.unit.infrastructure.rap.test_inference_backend import Harness, TEMPLATE, request

OTHER = TEMPLATE.__class__("other", "Other", 4, 4, TEMPLATE.slots[:3], TEMPLATE.provenance)


def steering(request_id, start_bar=0, version=3, **changes):
    values = dict(request_id=request_id, start_tick=start_bar * 16, input_version=version,
                  flow_ticks=(), config={"mode": "steering", "seed": 12})
    return request(**{**values, **changes})


class Steering(Harness):
    def __init__(self, **options):
        super().__init__(steering_adapter="session", **options)
        self.backend._templates["other"] = OTHER
        self.backend._steering._templates = self.backend._templates
        self.backend._steering._idle_close = 0.05

    def line(self, bar, *, revision=0, topic="space", template=TEMPLATE, text=None, stream=None):
        """Emit the server's line for an absolute bar of the long session."""
        stream = stream or self.opened()
        record = BarLyricRecord(
            stream.session_id, bar, revision, topic, template, text or f"{topic} bar {bar} r{revision}",
            tuple(ScheduledSyllable(slot, Syllable("test", i, len(template.slots), 1, ("AH1",), "test"))
                  for i, slot in enumerate(materialize_flow(template, bar))), 0.5,
            {"candidates_requested": 32, "candidates_valid": 7})
        self.records[record.lyric_id] = record
        stream.handler({"type": "lyric", "record": record.to_payload()})
        return record

    def spans_for(self, request_id):
        return [span for span in self.spans if span.request_id == request_id]

    def events_of(self, kind, request_id=None):
        return [e for e in self.events
                if e["type"] == kind and (request_id is None or e["request_id"] == request_id)]

    def sent(self, kind, stream=None):
        return [m for m in (stream or self.opened()).sent if m["type"] == kind]


def poll(predicate, timeout=3.0):
    """For state that changes without notifying the harness."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


@pytest.fixture
def harness():
    value = Steering()
    yield value
    value.close()


def test_leads_cover_a_request_from_the_lagging_playhead():
    assert session_leads(2) == (4, 3)  # the default two-bar request
    assert session_leads(1) == (3, 2)
    assert session_leads(4) == (7, 6)
    for bars in range(1, 17):
        lyric, audio = session_leads(bars)
        # A request for [s, s+bars) arrives with P = s - bars//2 - 1, and the server renders
        # up to P + audio. That must reach the request's last bar, s + bars - 1.
        reported = -(bars // 2) - 1
        assert reported + audio >= bars - 1 and lyric == audio + 1 and audio >= 2


def test_first_request_opens_one_unbounded_session(harness):
    harness.submit(steering("r1", history=("line a", "line b")))
    stream = harness.opened()
    start = stream.start
    assert "turn_id" not in start and "end_bar_exclusive" not in start
    assert (start["start_bar"], start["lyric_lead"], start["audio_lead"]) == (0, 4, 3)
    assert start["loop"] is True and start["seed"] == 12 and start["tempo_bpm"] == 120
    assert start["segments"] == [{"start_bar": 0, "bars": 1, "topic": "space",
                                  "flow_template": _flow_payload(TEMPLATE)}]
    assert start["history"] == [{"bar": -2, "text": "line a"}, {"bar": -1, "text": "line b"}]
    assert start["capabilities"] == [] and "continue_from" not in start
    assert harness.backend.capabilities["steering_adapter"] == "session"
    harness.wait(lambda: bool(harness.events_of("steering_session")))
    opened = harness.events_of("steering_session")[0]
    assert opened["remote_session_id"] == stream.session_id and opened["request_id"] == "r1"
    assert (opened["lyric_lead"], opened["audio_lead"], opened["continue_from"]) == (4, 3, None)


def test_bars_are_delivered_to_the_request_that_covers_them(harness):
    harness.submit(steering("r1"))
    records = [harness.line(bar) for bar in (0, 1)]
    harness.wait(lambda: len(harness.spans_for("r1")) == 2)
    spans = sorted(harness.spans_for("r1"), key=lambda s: s.start_tick)
    assert [(s.start_tick, s.output_id, s.input_version, s.epoch) for s in spans] == [
        (0, "r1:bar:0", 3, 2), (16, "r1:bar:1", 3, 2)]
    assert [s.text for s in spans] == [r.text for r in records]
    assert spans[0].first_vocal_tick == 0 and spans[1].first_vocal_tick == 0
    # Bars are prepared in parallel: ``final`` marks whichever span completed the request.
    assert [s.final for s in harness.spans_for("r1")] == [False, True]
    assert spans[1].syllables[0]["start_tick"] == 16 and spans[1].syllables[0]["tick_in_bar"] == 0
    harness.wait(lambda: bool(harness.events_of("complete", "r1")))
    assert len(harness.events_of("candidate_counts", "r1")) == 2
    assert sorted(e["bar"] for e in harness.events_of("stage_timings", "r1")) == [0, 1]
    assert len(harness.streams) == 1


def test_same_version_request_reports_the_playhead_and_takes_bars_rendered_ahead(harness):
    harness.submit(steering("r1"))
    first = [harness.line(bar) for bar in (0, 1)]
    harness.wait(lambda: len(harness.spans_for("r1")) == 2)
    ahead = harness.line(2)  # inside the audio window, but no request covers bar 2 yet
    poll(lambda: 2 in harness.backend._steering._session.ready)  # fetched and prepared, held
    assert len(harness.spans) == 2
    stream = harness.opened()
    assert harness.sent("committed") == []

    harness.submit(steering("r2", start_bar=2))
    harness.wait(lambda: len(harness.spans_for("r2")) == 1)
    assert len(harness.streams) == 1, "an extension must not open another session"
    # The service extends once half the request is left: bars [2, 4) arrive while bar 0 plays.
    assert harness.sent("committed") == [
        {"type": "committed", "bar": 0, "lyric_id": first[0].lyric_id, "text": first[0].text}]
    assert harness.spans_for("r2")[0].start_tick == 32 and harness.spans_for("r2")[0].text == ahead.text
    assert harness.sent("update") == []
    harness.line(3)
    harness.wait(lambda: bool(harness.events_of("complete", "r2")))

    harness.submit(steering("r3", start_bar=4))
    harness.wait(lambda: len(harness.sent("committed")) == 3)
    assert [m["bar"] for m in harness.sent("committed")] == [0, 1, 2]
    assert stream.closed is False


def test_bars_beyond_the_audio_window_are_not_fetched_until_the_playhead_moves(harness):
    harness.submit(steering("r1"))
    for bar in (0, 1, 2):
        harness.line(bar)
    harness.wait(lambda: len(harness.spans_for("r1")) == 2)
    late = harness.line(3)  # the lyric lead is one bar longer than the audio lead
    fetched = lambda: {record.bar for record, _ in harness.prepared}
    poll(lambda: fetched() == {0, 1, 2})
    assert late.lyric_id not in harness.backend._steering._session.fetching
    harness.submit(steering("r2", start_bar=2))
    harness.wait(lambda: len(harness.spans_for("r2")) == 2)
    assert fetched() == {0, 1, 2, 3}


def test_edit_sends_one_update_and_delivers_rewritten_bars_to_the_new_request(harness):
    harness.submit(steering("r1"))
    old = [harness.line(bar) for bar in (0, 1, 2)]
    harness.wait(lambda: len(harness.spans_for("r1")) == 2)
    harness.submit(steering("r2", start_bar=2))
    harness.wait(lambda: len(harness.spans_for("r2")) == 1)
    stream = harness.opened()
    assert [m["type"] for m in stream.sent] == ["committed"]  # the extension reported bar 0

    # The service's steer while bar 2 plays: cancel everything pending, request from bar 3.
    for request_id in ("r1", "r2"):
        harness.backend.cancel(request_id)
    harness.submit(steering("r3", start_bar=3, version=4, topic="ocean", template_id="other"))
    harness.wait(lambda: len(harness.sent("committed")) == 3)
    assert len(harness.streams) == 1
    edit = PlanEdit.from_payload(harness.sent("update")[0]["edit"])
    assert (edit.edit_seq, edit.from_bar, edit.topic, edit.flow_template) == (1, 3, "ocean", OTHER)
    # The edit is sent before the playhead moves: moving it first would let the server start
    # rendering an old-plan bar it is about to discard. Bars 1 and 2 are then reported, with
    # the lines the service actually accepted for them.
    assert [m["type"] for m in stream.sent] == ["committed", "update", "committed", "committed"]
    assert [(m["bar"], m["lyric_id"]) for m in harness.sent("committed")] == [
        (0, old[0].lyric_id), (1, old[1].lyric_id), (2, old[2].lyric_id)]

    # A line of the old plan that was still in flight is not delivered as the new version.
    harness.line(4, topic="space")
    new = [harness.line(bar, revision=1, topic="ocean", template=OTHER) for bar in (3, 4)]
    harness.wait(lambda: len(harness.spans_for("r3")) == 2)
    spans = sorted(harness.spans_for("r3"), key=lambda s: s.start_tick)
    assert [(s.start_tick, s.input_version, s.revision, s.text) for s in spans] == [
        (48, 4, 1, new[0].text), (64, 4, 1, new[1].text)]
    assert len(spans[0].syllables) == 3
    # r1 had completed; r2 was cut short by the edit.
    assert not harness.events_of("canceled", "r1") and len(harness.events_of("canceled", "r2")) == 1
    update = harness.events_of("plan_update", "r3")[0]
    assert (update["from_bar"], update["reported_playhead"]) == (3, 2)


def test_edit_that_keeps_topic_and_flow_still_requires_a_newer_revision(harness):
    harness.submit(steering("r1"))
    for bar in (0, 1):
        harness.line(bar)
    harness.wait(lambda: len(harness.spans_for("r1")) == 2)
    harness.backend.cancel("r1")
    harness.submit(steering("r2", start_bar=1, version=4))
    harness.wait(lambda: bool(harness.sent("update")))
    harness.line(1)  # the same revision again: the old plan's line
    rewritten = harness.line(1, revision=1)
    harness.wait(lambda: len(harness.spans_for("r2")) == 1)
    assert harness.spans_for("r2")[0].revision == 1 and harness.spans_for("r2")[0].text == rewritten.text


def test_second_edit_increases_the_edit_sequence(harness):
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    for index, version in enumerate((4, 5), start=1):
        harness.submit(steering(f"edit-{version}", start_bar=1, version=version, topic=f"topic {version}"))
        harness.wait(lambda: len(harness.sent("update")) == index)
    assert [m["edit"]["edit_seq"] for m in harness.sent("update")] == [1, 2]
    assert [m["edit"]["topic"] for m in harness.sent("update")] == ["topic 4", "topic 5"]


def test_session_closes_after_every_request_is_canceled_and_a_later_one_reopens(harness):
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    first = harness.opened()
    harness.backend.cancel("r1")
    harness.wait(lambda: first.closed)
    assert first.sent[-1] == {"type": "stop"}
    assert len(harness.events_of("canceled", "r1")) == 1
    harness.wait(lambda: bool(harness.events_of("session_cleanup")))
    cleanup = harness.events_of("session_cleanup")[0]
    assert cleanup["remote_session_id"] == first.session_id and cleanup["stop_sent"] is True
    assert cleanup["reason"] == "no live request"

    harness.submit(steering("r2", start_bar=3, history=("space bar 0 r0",)))
    second = harness.opened(1)
    assert second is not first and second.start["start_bar"] == 3
    assert second.start["history"] == [{"bar": 2, "text": "space bar 0 r0"}]


def test_edit_cancel_and_resubmit_does_not_close_the_session(harness):
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    harness.backend.cancel("r1")
    harness.submit(steering("r2", start_bar=1, version=4, topic="ocean"))
    harness.wait(lambda: bool(harness.sent("update")))
    time.sleep(0.2)  # well past the idle grace
    assert len(harness.streams) == 1 and not harness.opened().closed


def test_new_playback_closes_the_old_session_and_continues_its_last_line(harness):
    harness.capabilities.add("continuation_handoff_v1")
    harness.submit(steering("r1"))
    last = harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    first = harness.opened()
    harness.submit(steering("r2", epoch=3, history=(last.text,)))
    second = harness.opened(1)
    harness.wait(lambda: first.closed)
    assert second.start["continue_from"] == {"lyric_id": last.lyric_id}
    assert second.start["capabilities"] == ["continuation_handoff_v1"]
    assert second.start["start_bar"] == 0


def test_session_level_error_fails_live_requests_and_the_next_request_recovers(harness):
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    first = harness.opened()
    first.handler({"type": "error", "message": "lyric stream disconnected", "bar": None})
    harness.wait(lambda: bool(harness.events_of("error", "r1")))
    harness.wait(lambda: first.closed)
    harness.submit(steering("r2", start_bar=2))
    second = harness.opened(1)
    assert second.start["start_bar"] == 2
    harness.line(2, stream=second)
    harness.wait(lambda: len(harness.spans_for("r2")) == 1)
    assert harness.capability_calls == 2  # nothing kept from before the failure is trusted


def test_writer_retry_is_a_warning_and_an_exhausted_bar_fails_only_its_request(harness):
    harness.submit(steering("r1"))
    stream = harness.opened()
    failure = {"type": "error", "message": "no_valid_candidates", "bar": 1, "code": "lyric_write_failed"}
    stream.handler({**failure, "retryable": True, "attempt": 1, "max_attempts": 3})
    harness.wait(lambda: bool(harness.events_of("retry_warning", "r1")))
    assert harness.events_of("retry_warning", "r1")[0]["bar"] == 1
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    stream.handler({**failure, "retryable": False, "attempt": 3, "max_attempts": 3})
    harness.wait(lambda: bool(harness.events_of("error", "r1")))
    assert not stream.closed
    # The session goes on: the next request is served by the same session.
    harness.submit(steering("r2", start_bar=2))
    harness.line(2)
    harness.wait(lambda: len(harness.spans_for("r2")) == 1)
    assert len(harness.streams) == 1


def test_bar_that_failed_before_any_request_covered_it_fails_that_request_on_arrival(harness):
    harness.submit(steering("r1"))
    stream = harness.opened()
    stream.handler({"type": "error", "message": "no_valid_candidates", "bar": 2, "code": "lyric_write_failed",
                    "retryable": False, "attempt": 3, "max_attempts": 3})
    harness.line(0)
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    harness.submit(steering("r2", start_bar=2))
    harness.wait(lambda: bool(harness.events_of("error", "r2")))
    assert "no_valid_candidates" in harness.events_of("error", "r2")[0]["message"]


def test_render_error_message_alone_does_not_fail_a_request(harness):
    harness.submit(steering("r1"))
    harness.opened().handler({"type": "error", "message": "bar audio render failed: boom", "bar": 0,
                              "code": "session_error", "retryable": False})
    harness.line(0)  # the server rendered it on the retry
    harness.wait(lambda: len(harness.spans_for("r1")) == 1)
    assert not harness.events_of("error")


def test_failed_audio_fetch_fails_the_covering_request(harness):
    harness.fetch_status = "failed"
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: bool(harness.events_of("error", "r1")))
    assert harness.spans == [] and not harness.opened().closed


def test_submit_and_cancel_only_queue(harness):
    harness.open_gate = Event()  # the capability lookup blocks
    returned = Event()

    def calls():
        harness.submit(steering("r1"))
        harness.backend.cancel("r1")
        returned.set()

    thread = Thread(target=calls)
    thread.start()
    try:
        assert returned.wait(0.5), "submit or cancel did network work"
    finally:
        harness.open_gate.set()
        thread.join(3)
    harness.wait(lambda: bool(harness.events_of("canceled", "r1")))
    assert harness.spans == []


def test_cyclic_requests_keep_using_bounded_sessions(harness):
    harness.submit(request())
    stream = harness.opened()
    assert stream.start["turn_id"] == stream.session_id and stream.start["end_bar_exclusive"] == 2


def test_duplicate_request_is_ignored_and_capacity_is_bounded(harness):
    harness.submit(steering("r1"))
    assert harness.backend.submit(harness.current, harness.on_span, harness.on_event) is False
    for index in range(15):
        harness.backend.submit(steering(f"fill-{index}", start_bar=2 + index), harness.on_span, harness.on_event)
    with pytest.raises(RuntimeError, match="capacity"):
        harness.backend.submit(steering("too-many", start_bar=40), harness.on_span, harness.on_event)


def test_close_cancels_live_requests_and_stops_the_session(harness):
    harness.submit(steering("r1"))
    stream = harness.opened()
    harness.backend.close()
    assert stream.closed and stream.sent[-1] == {"type": "stop"}
    assert len(harness.events_of("canceled", "r1")) == 1
    with pytest.raises(RuntimeError):
        harness.submit(steering("after-close"))


def test_stuck_request_times_out_and_resets_the_session():
    harness = Steering(request_timeout_seconds=0.2)
    try:
        harness.submit(steering("r1"))
        stream = harness.opened()
        harness.wait(lambda: bool(harness.events_of("error", "r1")))
        assert "timed out" in harness.events_of("error", "r1")[0]["message"]
        harness.wait(lambda: stream.closed)
    finally:
        harness.close()


def test_unknown_template_fails_the_request_without_a_session(harness):
    harness.submit(steering("r1", template_id="missing"))
    harness.wait(lambda: bool(harness.events_of("error", "r1")))
    assert harness.streams == []


def test_server_continuation_diagnostics_are_reported_per_bar(harness):
    harness.continuation = {"used": True, "prefix_lyric_id": "a" * 64}
    harness.submit(steering("r1"))
    harness.line(0)
    harness.wait(lambda: bool(harness.events_of("stage_timings", "r1")))
    assert harness.events_of("stage_timings", "r1")[0]["continuation"] == harness.continuation


def test_default_adapter_still_opens_a_bounded_session_per_steering_request():
    harness = Harness()
    try:
        harness.submit(steering("r1"))
        stream = harness.opened()
        assert stream.start["turn_id"] == stream.session_id
        assert harness.backend.capabilities["steering_adapter"] == "bounded"
        assert harness.backend._steering is None
    finally:
        harness.close()

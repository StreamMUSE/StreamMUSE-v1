import threading
import pytest

from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.lyric_writer import LyricWriterPolicy, SequentialLyricWriter
from streammuse.domain.rap import ScoreWeights
from streammuse.domain.rap.lyric_stream import PlanEdit
from streammuse.domain.rap.story import StoryContext, StoryEvent
from streammuse.infrastructure.rap.generators import _build_messages, _build_independent_messages
from tests.unit.application.rap.test_lyric_writer import _Generator, _Analyzer, _WEIGHTS, _request
from tests.unit.application.rap.test_lyric_session import _session, _Listener, _wait, _Generator as SessionGenerator
from tests.unit.application.rap.test_bar_stream import _controller, _Stream, _Source


def story():
    return StoryContext().with_event(StoryEvent("map", 1, "a courier loses the map", 1.0, "typed"))


def test_story_survives_rescue_but_recent_drafts_are_dropped():
    generator = _Generator([RuntimeError("retry"), ("four words fit here",)])
    writer = SequentialLyricWriter(generator, _Analyzer(), _WEIGHTS,
                                   LyricWriterPolicy(candidates=1, rescue_candidates=1, minimum_valid=1))
    written = writer.write(_request(story_context=story()))
    assert [r.story_context for r in generator.requests] == [story(), story()]
    assert generator.requests[1].context_lines == ()
    assert written.record.provenance["story_event_ids"] == ("map",)
    for build in (_build_messages, _build_independent_messages):
        prompt = build(generator.requests[0])[1]["content"]
        assert "a courier loses the map" in prompt
        assert "not confirmed playback" in prompt
        assert "Recent frozen lines" not in prompt


def test_topic_edits_do_not_replace_story_and_disabled_story_is_unchanged():
    request = _request()
    plan = LivePlan(lambda bar: ("sea", request.flow_template))
    assert plan.for_bar(0).story_context is None
    plan = plan.with_edit(PlanEdit(1, 2, story_context=story()))
    plan = plan.with_edit(PlanEdit(2, 3, topic="city"))
    assert plan.for_bar(1).story_context is None
    assert plan.for_bar(3).topic == "city"
    assert plan.for_bar(3).story_context == story()


def test_edit_during_generation_discards_old_result_and_retries_with_story():
    entered, release = threading.Event(), threading.Event()
    class Blocked(SessionGenerator):
        def generate(self, request):
            if not self.requests:
                entered.set()
                assert release.wait(5)
            return super().generate(request)
    generator, listener = Blocked(), _Listener()
    writer = SequentialLyricWriter(generator, _Analyzer(), ScoreWeights(),
                                   LyricWriterPolicy(candidates=2, rescue_candidates=0, minimum_valid=1))
    session = _session(writer=writer, listener=listener)
    try:
        assert entered.wait(2)
        session.apply_edit(PlanEdit(1, 0, story_context=story()))
        release.set()
        _wait(lambda: len(listener.lyrics) == 3)
        assert generator.requests[0].story_context is None
        assert all(r.story_context == story() for r in generator.requests[1:])
        assert all(r.provenance["story_event_ids"] == ("map",) for r in listener.lyrics)
    finally:
        release.set()
        session.close()


def test_controller_deduplicates_and_reset_clears_session_story():
    stream = _Stream()
    controller = _controller(stream, _Source(), [])
    try:
        first = controller.submit_story_event("a courier loses the map", event_id="map")
        duplicate = controller.submit_story_event("a courier loses the map", event_id="map")
        assert first["event"] == duplicate["event"]
        assert duplicate["duplicate"] is True
        assert len(controller._plan.edits) == 1
        assert first["effective_from_bar"] == controller.next_editable_bar
        assert controller._plan.for_bar(0).topic == controller._base_plan().for_bar(0).topic
        assert controller._start_message(start_bar=0)["edits"][0]["story_context"]["version"] == 1
        assert stream.sent == []  # queued in start payload before the connection opens
        controller.reset()
        assert controller.story_snapshot()["events"] == []
    finally:
        controller.close()


def test_failed_send_can_retry_same_event_without_another_edit():
    stream = _Stream()
    stream.capabilities = {"story_context_v1"}
    controller = _controller(stream, _Source(), [])
    controller._started = True
    original_send = stream.send
    def fail(message):
        raise ConnectionError("offline")
    stream.send = fail
    try:
        with pytest.raises(ConnectionError):
            controller.submit_story_event("a courier loses the map", event_id="map")
        stream.send = original_send
        result = controller.submit_story_event("a courier loses the map", event_id="map")
        assert result["duplicate"] is True
        assert len(controller._plan.edits) == 1
        assert len(stream.sent) == 1
        controller.submit_story_event("a courier loses the map", event_id="map")
        assert len(stream.sent) == 1
    finally:
        controller.close()


def test_story_rejected_on_live_peer_without_protocol_capability():
    stream = _Stream()
    stream.capabilities = ()
    controller = _controller(stream, _Source(), [])
    controller._started = True
    try:
        with pytest.raises(RuntimeError, match="story_context_v1"):
            controller.submit_story_event("map lost")
        assert controller.story_snapshot()["events"] == []
    finally:
        controller.close()


def test_session_story_edit_retry_is_idempotent_and_does_not_change_committed_bar():
    listener = _Listener()
    session = _session(listener=listener)
    try:
        _wait(lambda: len(listener.lyrics) == 3)
        committed = listener.lyrics[0]
        session.committed(0, committed.lyric_id, committed.text)
        edit = PlanEdit(1, 1, story_context=story())
        session.apply_edit(edit)
        assert session.apply_edit(edit) == ()
        _wait(lambda: any(r.bar == 1 and r.provenance.get("story_version") == 1 for r in listener.lyrics))
        assert session.lyric(committed.lyric_id) == committed
        assert not any(r.bar == 0 and r.provenance.get("story_version") for r in listener.lyrics)
        assert session._plan.for_bar(2).story_context == story()
        assert session._plan.for_bar(2).topic == "sea"
    finally:
        session.close()


def test_story_edit_reopens_failed_bar_and_old_failure_does_not_consume_new_retry():
    entered, release = threading.Event(), threading.Event()
    class FailingWriter:
        def write(self, request):
            entered.set()
            assert release.wait(5)
            raise RuntimeError("old request failed")
    session = _session(writer=FailingWriter())
    try:
        assert entered.wait(2)
        with session._condition:
            session._state(0).write_failures = 3
        session.apply_edit(PlanEdit(1, 0, story_context=story()))
        with session._condition:
            assert session._state(0).write_failures == 0
            session._closed = True
        release.set()
        session.close()
        assert session._state(0).write_failures == 0
    finally:
        release.set()
        session.close()


def test_three_incremental_directions_survive_serialized_start_and_topic_changes():
    from streammuse.presentation.rap_lyric_stream_server import plan_from_start
    stream = _Stream()
    controller = _controller(stream, _Source(), [])
    directions = ("a courier loses the map", "a stranger offers help", "the stranger wants the package")
    try:
        for index, text in enumerate(directions):
            controller.submit_story_event(text, event_id=f"event-{index}")
        controller.update_plan(from_bar=0, topic="city")
        restored = plan_from_start(controller._start_message(start_bar=0))
        assert restored.for_bar(0).topic == "city"
        assert tuple(e.text for e in restored.for_bar(0).story_context.events) == directions
    finally:
        controller.close()


def test_story_provenance_is_exposed_for_exact_reserved_and_chosen_lyric():
    from dataclasses import replace
    from tests.unit.application.rap.test_bar_stream import _record, _fetch, _start_with, _Publisher
    from streammuse.domain.rap.events import RapEventType
    stream, source, publisher = _Stream(), _Source(), _Publisher()
    record = replace(_record(0), provenance={"story_version": 2, "story_event_ids": ["a", "b"]})
    source.ready[record.lyric_id] = _fetch(record)
    controller = _start_with(stream, source, [], record, publisher=publisher)
    try:
        for kind in (RapEventType.BAR_RESERVED, RapEventType.BAR_FROZEN):
            payload = next(data["payload"] for event, data in publisher.events if event == kind)
            assert payload["story_version"] == 2
            assert tuple(payload["story_event_ids"]) == ("a", "b")
            assert payload["lyric_id"] == record.lyric_id
    finally:
        controller.close()

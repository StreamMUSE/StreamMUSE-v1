import threading

import pytest

from streammuse.domain.rap.lyric_stream import PlanEdit
from streammuse.domain.rap.story import StoryConflict
from tests.unit.application.rap.test_bar_stream import _controller, _Stream, _Source


def started():
    stream = _Stream()
    stream.capabilities = {"story_context_v1"}
    controller = _controller(stream, _Source(), [])
    controller._started = True
    return controller, stream


@pytest.mark.parametrize("retry", [False, True])
def test_blocked_story_send_allows_ticks_stop_reset_and_freezes_ack_session(retry):
    controller, stream = started()
    entered, release, responsive = threading.Event(), threading.Event(), threading.Event()
    replies, errors = [], []
    original_session = controller.session_id
    if retry:
        def fail(message):
            raise ConnectionError("offline")
        stream.send = fail
        with pytest.raises(ConnectionError):
            controller.submit_story_event("map lost", "map")
    def blocked(message):
        if message["type"] == "update":
            entered.set()
            assert release.wait(5)
    stream.send = blocked
    def submit():
        try:
            replies.append(controller.submit_story_event("map lost", "map"))
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=submit)
    def controls():
        controller.story_snapshot()
        controller.on_tick(0)
        controller.request_stop(successor_bar=None)
        controller.reset()
        responsive.set()
    reader = threading.Thread(target=controls)
    try:
        worker.start()
        assert entered.wait(2)
        reader.start()
        assert responsive.wait(2), "playback/control lock remained held during socket send"
        release.set()
        worker.join(2)
        assert not errors
        assert replies[0]["session_id"] == original_session
        assert controller.session_id != original_session
        assert controller.story_snapshot()["events"] == []
        assert controller._pending_story_edits == set()
    finally:
        release.set()
        worker.join(5)
        if reader.ident is not None:
            reader.join(5)
        controller.close()


@pytest.mark.parametrize("intervening", ["topic", "flow"])
@pytest.mark.parametrize("count", [1, 2])
def test_retry_after_intervening_edit_rebases_full_context_without_new_events(intervening, count):
    controller, stream = started()
    remote = controller._base_plan()
    rejected, sent = [], []
    def fail(message):
        raise ConnectionError("lost before delivery")
    stream.send = fail
    try:
        for index in range(count):
            with pytest.raises(ConnectionError):
                controller.submit_story_event(f"direction {index}", f"id-{index}")
        originals = controller.story_snapshot()["events"]
        def deliver(message):
            nonlocal remote
            if message["type"] != "update":
                return
            edit = PlanEdit.from_payload(message["edit"])
            sent.append(edit)
            try:
                remote = remote.with_edit(edit)
            except ValueError as error:
                rejected.append(str(error))
        stream.send = deliver
        change = {"topic": "city"} if intervening == "topic" else {
            "template_id": remote.for_bar(0).flow_template.template_id}
        controller.update_plan(from_bar=0, **change)
        result = controller.submit_story_event("direction 0", "id-0")
        assert result["duplicate"] is True
        assert not rejected
        assert remote.for_bar(0).story_context.to_payload()["events"] == originals
        assert controller.story_snapshot()["events"] == originals
        assert controller._pending_story_edits == set()
        sent_before = len(sent)
        for index in range(count):
            controller.submit_story_event(f"direction {index}", f"id-{index}")
        assert len(sent) == sent_before
    finally:
        controller.close()


def test_expected_session_is_checked_before_mutation_and_normal_retry_is_idempotent():
    controller = _controller(_Stream(), _Source(), [])
    old = controller.session_id
    try:
        first = controller.submit_story_event("map lost", "map", expected_session_id=old)
        assert controller.submit_story_event("map lost", "map", expected_session_id=old)["duplicate"]
        controller.reset()
        with pytest.raises(StoryConflict, match="session"):
            controller.submit_story_event("map lost", "map", source="microphone", expected_session_id=old)
        assert controller.story_snapshot()["events"] == []
        assert controller._plan.edits == ()
        assert first["session_id"] == old
    finally:
        controller.close()


def test_concurrent_topic_cannot_overtake_blocked_story_delivery():
    controller, stream = started()
    entered, release = threading.Event(), threading.Event()
    delivered = []
    def send(message):
        if message["type"] != "update":
            return
        if message["edit"]["edit_seq"] == 1:
            entered.set()
            assert release.wait(5)
        delivered.append(message["edit"]["edit_seq"])
    stream.send = send
    story = threading.Thread(target=lambda: controller.submit_story_event("map lost", "map"))
    topic = threading.Thread(target=lambda: controller.update_plan(from_bar=0, topic="city"))
    try:
        story.start()
        assert entered.wait(2)
        topic.start()
        release.set()
        story.join(3)
        topic.join(3)
        assert delivered == [1, 2]
    finally:
        release.set()
        story.join(5)
        if topic.ident is not None:
            topic.join(5)
        controller.close()


def test_reset_between_controller_epoch_check_and_socket_capture_rejects_old_edit():
    from streammuse.infrastructure.rap.lyric_stream_client import LyricStreamConnection
    class Socket:
        def __init__(self):
            self.sent = []
        def send(self, message):
            self.sent.append(message)
        def close(self):
            pass
    connection = LyricStreamConnection("http://test")
    controller = _controller(connection, _Source(), [])
    controller._started = True
    connection._capabilities = {"story_context_v1"}
    connection._session_id = controller.session_id
    connection._socket = Socket()
    newer = Socket()
    send_for_session = connection.send_for_session
    def reset_before_capture(session_id, message):
        controller.reset()
        with connection._lock:
            connection._session_id, connection._socket = controller.session_id, newer
        send_for_session(session_id, message)
    connection.send_for_session = reset_before_capture
    try:
        with pytest.raises(ConnectionError, match="session"):
            controller.submit_story_event("map lost", "map")
        assert newer.sent == []
        assert controller.story_snapshot()["events"] == []
        assert controller._pending_story_edits == set()
    finally:
        controller.close()

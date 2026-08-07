"""Tests for the minimal ordered rap event stream."""

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.domain.rap import RapEventType


def test_publisher_assigns_one_ordered_sequence_to_every_sink() -> None:
    left = []
    right = []
    publisher = RapEventPublisher(
        "session-1",
        utc_now=lambda: "2026-08-07T00:00:00+00:00",
        monotonic_ns=lambda: 123,
    )
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(left.append, right.append))
    dispatcher.start()

    publisher.emit(RapEventType.BAR_RESERVED, bar=0, tick=0, payload={"source": "fallback"})
    publisher.emit(RapEventType.BAR_FROZEN, bar=0, tick=0, payload={"fallback": True})
    dispatcher.flush_and_close()

    assert [event.sequence for event in left] == [1, 2]
    assert left == right
    assert left[0].session_id == "session-1"

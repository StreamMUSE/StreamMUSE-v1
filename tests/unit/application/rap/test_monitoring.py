"""Tests for canonical, ordered rap event monitoring."""

from __future__ import annotations

from threading import Event, Thread

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.domain.rap import RapEvent, RapEventType


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


class _FirstPutBlockingQueue:
    """Makes the old sequence-then-unlocked-put race deterministic."""

    def __init__(self) -> None:
        from queue import SimpleQueue

        self._queue: SimpleQueue[object] = SimpleQueue()
        self.first_put_entered = Event()
        self.release_first_put = Event()

    def put(self, item: object) -> None:
        if getattr(item, "sequence", None) == 1:
            self.first_put_entered.set()
            assert self.release_first_put.wait(timeout=1)
        self._queue.put(item)

    def get(self) -> object:
        return self._queue.get()

    def get_nowait(self) -> object:
        return self._queue.get_nowait()


def test_publisher_assigns_sequence_and_enqueues_atomically() -> None:
    publisher = RapEventPublisher("session-1")
    queue = _FirstPutBlockingQueue()
    publisher.queue = queue  # type: ignore[assignment]
    received: list[RapEvent] = []
    dispatcher = RapEventDispatcher(queue, sinks=(received.append,))
    dispatcher.start()

    first = Thread(target=lambda: publisher.emit(RapEventType.BAR_RESERVED, bar=0))
    second = Thread(target=lambda: publisher.emit(RapEventType.BAR_RESERVED, bar=1))
    first.start()
    assert queue.first_put_entered.wait(timeout=1)
    second.start()
    queue.release_first_put.set()
    first.join(timeout=1)
    second.join(timeout=1)
    dispatcher.flush_and_close()

    assert not first.is_alive()
    assert not second.is_alive()
    assert [event.sequence for event in received] == [1, 2]


def test_dispatcher_disables_failed_sink_and_records_presentation_error() -> None:
    received: list[RapEvent] = []

    def failing_sink(_: RapEvent) -> None:
        raise RuntimeError("terminal down")

    publisher = RapEventPublisher(
        "session-1",
        utc_now=lambda: "2026-08-07T00:00:00+00:00",
        monotonic_ns=lambda: 123,
    )
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(failing_sink, received.append))
    dispatcher.start()
    publisher.emit(RapEventType.TICK, bar=0, tick=0)
    dispatcher.flush_and_close()

    assert [event.event_type for event in received] == [RapEventType.TICK, RapEventType.PRESENTATION_ERROR]
    assert received[1].sequence == 2
    assert received[1].payload == {
        "error_type": "RuntimeError",
        "error_message": "terminal down",
        "failed_event_type": "tick",
        "sink": "failing_sink",
    }


def _event(
    sequence: int,
    event_type: RapEventType,
    *,
    bar: int | None = None,
    tick: int | None = None,
    request_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> RapEvent:
    return RapEvent(
        session_id="session-1",
        sequence=sequence,
        event_type=event_type,
        utc_time="2026-08-07T00:00:00+00:00",
        monotonic_ns=sequence,
        bar=bar,
        tick=tick,
        request_id=request_id,
        payload=payload or {},
    )


def test_state_projector_exposes_a_deep_serializable_snapshot() -> None:
    projector = RapStateProjector()
    events = (
        _event(1, RapEventType.BAR_RESERVED, bar=2, payload={"topic": "space", "template_id": "nine"}),
        _event(2, RapEventType.BAR_PLANNING_STARTED, bar=2, tick=32, request_id="r2", payload={"topic": "space", "template_id": "nine"}),
        _event(3, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=2, tick=33, request_id="r2", payload={"latency_ms": 12.5}),
        _event(4, RapEventType.CANDIDATE_EVALUATED, bar=2, request_id="r2", payload={"candidate_id": "c1", "valid": True}),
        _event(5, RapEventType.BAR_FROZEN, bar=2, tick=64, request_id="r2", payload={"text": "space line", "fallback": True, "fallback_reason": "deadline_miss"}),
        _event(6, RapEventType.FALLBACK_ACTIVATED, bar=2, request_id="r2", payload={"fallback_reason": "deadline_miss"}),
        _event(7, RapEventType.TICK, bar=2, tick=65),
        _event(8, RapEventType.SYLLABLE_EMITTED, bar=2, tick=65, payload={"label": "space", "jitter_ms": 0.25}),
    )
    for event in events:
        projector.apply(event)

    snapshot = projector.snapshot()

    assert snapshot["current_tick"] == 65
    assert snapshot["current_segment"] == {"bar": 2, "topic": "space", "template_id": "nine"}
    assert snapshot["pending_request"] is None
    assert snapshot["candidates"]["c1"]["valid"] is True
    assert snapshot["frozen_bars"]["2"]["text"] == "space line"
    assert snapshot["emitted_syllables"] == [{"bar": 2, "tick": 65, "label": "space", "jitter_ms": 0.25}]
    assert snapshot["latencies"]["generation_latency_ms"] == {"count": 1, "total": 12.5, "min": 12.5, "max": 12.5}
    assert snapshot["fallbacks"] == {"count": 1, "by_reason": {"deadline_miss": 1}}

    snapshot["frozen_bars"]["2"]["text"] = "mutated"
    assert projector.snapshot()["frozen_bars"]["2"]["text"] == "space line"

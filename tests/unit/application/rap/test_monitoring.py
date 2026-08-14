"""Tests for canonical, ordered rap event monitoring."""

from __future__ import annotations

import json
from threading import Event, Thread
from time import sleep

import pytest

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.infrastructure.rap.recorder import derive_summary


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
        _event(1, RapEventType.SESSION_STARTED, payload={"tempo_bpm": 92.0, "generator": "local_chat"}),
        _event(2, RapEventType.BAR_RESERVED, bar=2, payload={"topic": "space", "template_id": "nine"}),
        _event(3, RapEventType.BAR_PLANNING_STARTED, bar=2, tick=32, request_id="r2", payload={"topic": "space", "template_id": "nine", "context_lines": ["prior line"], "flow": {"slots": [{"tick_in_bar": 0}]}}),
        _event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=2, tick=33, request_id="r2", payload={"latency_ms": 12.5, "prompt": [{"role": "user", "content": "exact prompt"}], "raw_response": "raw line"}),
        _event(5, RapEventType.CANDIDATE_EVALUATED, bar=2, request_id="r2", payload={"candidate_id": "c1", "valid": True}),
        _event(6, RapEventType.BAR_FROZEN, bar=2, tick=64, request_id="r2", payload={"text": "space line", "fallback": True, "fallback_reason": "deadline_miss"}),
        _event(7, RapEventType.FALLBACK_ACTIVATED, bar=2, request_id="r2", payload={"fallback_reason": "deadline_miss"}),
        _event(8, RapEventType.TICK, bar=2, tick=65),
        _event(9, RapEventType.SYLLABLE_EMITTED, bar=2, tick=65, payload={"label": "space", "observation_delay_ms": 0.25}),
        _event(10, RapEventType.SESSION_STOPPED),
    )
    for event in events:
        projector.apply(event)

    snapshot = projector.snapshot()

    assert snapshot["current_tick"] == 65
    assert snapshot["session_metadata"]["tempo_bpm"] == 92.0
    assert snapshot["stopped"] is True
    assert snapshot["recent_events"][-1]["event_type"] == "session_stopped"
    assert snapshot["current_segment"] == {"bar": 2, "topic": "space", "template_id": "nine"}
    assert snapshot["pending_request"] is None
    assert snapshot["latest_request"]["context_lines"] == ["prior line"]
    assert snapshot["latest_request"]["flow"]["slots"][0]["tick_in_bar"] == 0
    assert snapshot["latest_batch"]["prompt"][0]["content"] == "exact prompt"
    assert snapshot["latest_batch"]["raw_response"] == "raw line"
    assert snapshot["candidates"]["c1"]["valid"] is True
    assert snapshot["frozen_bars"]["2"]["text"] == "space line"
    assert snapshot["emitted_syllables"] == [
        {"bar": 2, "tick": 65, "label": "space", "observation_delay_ms": 0.25}
    ]
    assert snapshot["latencies"]["generation_latency_ms"] == {"count": 1, "total": 12.5, "min": 12.5, "max": 12.5}
    assert snapshot["fallbacks"] == {"count": 1, "by_reason": {"deadline_miss": 1}}

    snapshot["frozen_bars"]["2"]["text"] = "mutated"
    assert projector.snapshot()["frozen_bars"]["2"]["text"] == "space line"


def test_state_projector_cumulative_metrics_match_recorder_derivation() -> None:
    events = (
        _event(1, RapEventType.SESSION_STARTED, payload={"repetition_window_bars": 2}),
        _event(2, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0"),
        _event(
            3,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            bar=0,
            request_id="r0",
            payload={"candidate_count": 2, "latency_ms": 10.0, "deadline_slack_ms": 20.0, "late": False},
        ),
        _event(
            4,
            RapEventType.CANDIDATE_EVALUATED,
            bar=0,
            request_id="r0",
            payload={
                "candidate_id": "c0",
                "valid": True,
                "word_analysis_sources": [
                    {"word": "clean", "source": "cmudict_first_pronunciation"},
                    {"word": "orbit", "source": "vowel_group_heuristic"},
                ],
            },
        ),
        _event(5, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c1", "valid": False}),
        _event(6, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "clean orbit now", "fallback": False}),
        _event(7, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="r1"),
        _event(
            8,
            RapEventType.CANDIDATE_BATCH_RECEIVED,
            bar=1,
            request_id="r1",
            payload={
                "candidate_count": 0,
                "latency_ms": 30.0,
                "deadline_slack_ms": -5.0,
                "late": True,
                "error_type": "generation_error",
            },
        ),
        _event(9, RapEventType.GENERATION_FAILED, bar=1, request_id="r1", payload={"error_type": "generation_error"}),
        _event(10, RapEventType.BAR_FROZEN, bar=1, request_id="r1", payload={"text": "orbit now returns", "fallback": True}),
        _event(11, RapEventType.SYLLABLE_EMITTED, bar=0, tick=0, payload={"observation_delay_ms": 1.0}),
        _event(12, RapEventType.SYLLABLE_EMITTED, bar=1, tick=16, payload={"observation_delay_ms": 3.0}),
    )
    projector = RapStateProjector(max_recent_bars=1, max_emitted_syllables=1, max_candidates=1, max_recent_events=1)
    for event in events:
        projector.apply(event)

    snapshot = projector.snapshot()
    summary = derive_summary(events)

    assert snapshot["research_metrics"]["metrics"] == summary["metrics"]
    assert snapshot["research_metrics"]["latencies"] == summary["latencies"]


def test_flush_coordinates_with_an_inflight_emit_and_rejects_later_external_events() -> None:
    entered_clock = Event()
    release_clock = Event()
    seen: list[RapEvent] = []

    def blocking_utc() -> str:
        entered_clock.set()
        assert release_clock.wait(timeout=1)
        return "2026-08-07T00:00:00+00:00"

    publisher = RapEventPublisher("session-1", utc_now=blocking_utc)
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(seen.append,))
    dispatcher.start()
    emitter = Thread(target=lambda: publisher.emit(RapEventType.TICK, bar=0, tick=0))
    closer = Thread(target=dispatcher.flush_and_close)
    emitter.start()
    assert entered_clock.wait(timeout=1)
    closer.start()
    sleep(0.02)
    assert closer.is_alive()
    release_clock.set()
    emitter.join(timeout=1)
    closer.join(timeout=1)

    assert [event.sequence for event in seen] == [1]
    with pytest.raises(RuntimeError, match="closed"):
        publisher.emit(RapEventType.TICK, bar=0, tick=1)


def test_publisher_detaches_nested_payloads_before_dispatch() -> None:
    payload = {"nested": {"value": 1}}
    seen: list[RapEvent] = []
    publisher = RapEventPublisher("session-1")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(seen.append,))
    dispatcher.start()
    publisher.emit(RapEventType.TICK, payload=payload)
    payload["nested"]["value"] = 99
    dispatcher.flush_and_close()

    assert seen[0].payload == {"nested": {"value": 1}}


def test_state_projector_bounds_live_windows_and_tracks_request_failures() -> None:
    projector = RapStateProjector(max_recent_bars=1, max_emitted_syllables=1, max_candidates=1, max_recent_events=2)
    projector.apply(_event(1, RapEventType.BAR_RESERVED, bar=0, payload={"topic": "now", "template_id": "one"}))
    projector.apply(_event(2, RapEventType.BAR_RESERVED, bar=1, payload={"topic": "future", "template_id": "two"}))
    assert projector.snapshot()["current_segment"] is None
    projector.apply(_event(3, RapEventType.TICK, bar=0, tick=0))
    projector.apply(_event(4, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0"))
    projector.apply(_event(5, RapEventType.GENERATION_FAILED, bar=0, request_id="r0"))
    projector.apply(_event(6, RapEventType.CANDIDATE_EVALUATED, request_id="r0", payload={"candidate_id": "old"}))
    projector.apply(_event(7, RapEventType.CANDIDATE_EVALUATED, request_id="r0", payload={"candidate_id": "new"}))
    projector.apply(_event(8, RapEventType.BAR_FROZEN, bar=0, payload={"text": "old"}))
    projector.apply(_event(9, RapEventType.BAR_FROZEN, bar=1, payload={"text": "new"}))
    projector.apply(_event(10, RapEventType.SYLLABLE_EMITTED, bar=0, tick=0, payload={"label": "old"}))
    projector.apply(_event(11, RapEventType.SYLLABLE_EMITTED, bar=1, tick=16, payload={"label": "new"}))

    snapshot = projector.snapshot()
    assert snapshot["current_segment"] == {"bar": 0, "topic": "now", "template_id": "one"}
    assert snapshot["pending_request"] is None
    assert list(snapshot["candidates"]) == ["new"]
    assert list(snapshot["frozen_bars"]) == ["1"]
    assert snapshot["emitted_syllables"] == [{"bar": 1, "tick": 16, "label": "new"}]
    assert [event["sequence"] for event in snapshot["recent_events"]] == [10, 11]
    json.dumps(snapshot)


def test_state_projector_retains_terminal_dashboard_state_across_freeze_and_reconnect() -> None:
    projector = RapStateProjector()
    projector.apply(
        _event(
            1,
            RapEventType.BAR_RESERVED,
            bar=0,
            payload={"text": "fallback line", "fallback": True, "source": "prevalidated_fallback"},
        )
    )
    projector.apply(
        _event(
            2,
            RapEventType.BAR_PLANNING_STARTED,
            bar=0,
            request_id="r0",
            payload={"topic": "demo"},
        )
    )
    projector.apply(
        _event(
            3,
            RapEventType.BAR_REPLACED,
            bar=0,
            request_id="r0",
            payload={"text": "selected line", "fallback": False, "source": "local_chat", "total_score": 0.875},
        )
    )
    projector.apply(
        _event(
            4,
            RapEventType.BAR_FROZEN,
            bar=0,
            request_id="r0",
            payload={"text": "selected line", "fallback": False, "source": "local_chat"},
        )
    )
    projector.apply(
        _event(
            5,
            RapEventType.BAR_RESERVED,
            bar=1,
            payload={"text": "next safe line", "fallback": True, "source": "prevalidated_fallback"},
        )
    )
    projector.apply(_event(6, RapEventType.TICK, bar=0, tick=4, payload={"beat": 1, "tick_in_beat": 0}))
    projector.apply(
        _event(
            7,
            RapEventType.SYLLABLE_EMITTED,
            bar=0,
            tick=4,
            payload={"label": "selected", "stress": 1, "subdivision": 0, "observation_delay_ms": 0.25},
        )
    )
    projector.apply(
        _event(
            8,
            RapEventType.GENERATION_FAILED,
            bar=1,
            request_id="r1",
            payload={"error_type": "timeout", "error_message": "model missed deadline"},
        )
    )

    snapshot = projector.snapshot()

    assert snapshot["bars"]["0"]["frozen"] is True
    assert snapshot["bars"]["0"]["total_score"] == 0.875
    assert snapshot["frozen_bars"]["0"]["total_score"] == 0.875
    assert snapshot["bars"]["1"]["text"] == "next safe line"
    assert snapshot["current_syllable"] == {
        "bar": 0,
        "tick": 4,
        "label": "selected",
        "stress": 1,
        "subdivision": 0,
        "observation_delay_ms": 0.25,
    }
    assert snapshot["last_error"]["event_type"] == "generation_failed"
    assert snapshot["last_error"]["payload"]["error_message"] == "model missed deadline"
    json.dumps(snapshot)


def test_state_projector_replaces_candidates_when_a_new_request_starts() -> None:
    projector = RapStateProjector()
    projector.apply(_event(1, RapEventType.CANDIDATE_EVALUATED, request_id="old", payload={"candidate_id": "old"}))
    projector.apply(_event(2, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="new"))
    projector.apply(_event(3, RapEventType.CANDIDATE_EVALUATED, request_id="new", payload={"candidate_id": "new"}))

    assert list(projector.snapshot()["candidates"]) == ["new"]


def test_state_projector_ignores_stale_evaluations_and_retains_current_future_segments() -> None:
    projector = RapStateProjector(max_recent_bars=1)
    for bar in range(20):
        projector.apply(_event(bar + 1, RapEventType.BAR_RESERVED, bar=bar, payload={"topic": str(bar), "template_id": "one"}))
    projector.apply(_event(21, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="new"))
    projector.apply(_event(22, RapEventType.CANDIDATE_EVALUATED, request_id="old", payload={"candidate_id": "old"}))
    projector.apply(_event(23, RapEventType.CANDIDATE_EVALUATED, request_id="new", payload={"candidate_id": "new"}))
    projector.apply(_event(24, RapEventType.TICK, bar=0, tick=0))

    snapshot = projector.snapshot()
    assert list(snapshot["candidates"]) == ["new"]
    assert snapshot["current_segment"] == {"bar": 0, "topic": "0", "template_id": "one"}


def test_rap_package_exports_preserve_existing_and_monitoring_public_apis() -> None:
    from streammuse.application.rap import RollingRapController, align_exact
    from streammuse.application.rap import RapEventDispatcher as ExportedDispatcher
    from streammuse.infrastructure.rap import PhraseBankGenerator, RapSessionRecorder

    assert RollingRapController is not None
    assert align_exact is not None
    assert ExportedDispatcher is RapEventDispatcher
    assert PhraseBankGenerator is not None
    assert RapSessionRecorder is not None


def test_state_projector_tracks_audio_state_warnings_and_latency_aggregates() -> None:
    projector = RapStateProjector()
    events = (
        _event(1, RapEventType.AUDIO_RENDER_COMPLETED, bar=2, payload={"synthesis_latency_ms": 11.0, "render_latency_ms": 48.0}),
        _event(
            2,
            RapEventType.BAR_AUDIO_COMMITTED,
            bar=2,
            payload={"queue_depth": 3, "buffered_seconds": 12.0, "deadline_slack_ms": 250.0, "device": "Mac speakers"},
        ),
        _event(
            3,
            RapEventType.PRONUNCIATION_FALLBACK,
            bar=2,
            payload={"word": "StreamMUSE", "action": "espeak_g2p", "source": "espeak_g2p"},
        ),
        _event(4, RapEventType.TIMING_PRESSURE, bar=2, payload={"slot_index": 7, "word": "trans", "compression_ratio": 1.45}),
        _event(5, RapEventType.AUDIO_UNDERRUN, payload={"queue_depth": 0}),
        _event(6, RapEventType.BAR_PLAYBACK_STARTED, bar=2, payload={"absolute_frame": 384_000, "queue_depth": 3, "buffered_seconds": 12.0}),
    )
    for event in events:
        projector.apply(event)

    state = projector.snapshot()

    assert state["audio"] == {
        "state": "running",
        "current_bar": 2,
        "render_state": "committed",
        "render_bar": 2,
        "queue_depth": 3,
        "buffered_seconds": 12.0,
        "underruns": 1,
        "device": "Mac speakers",
        "recording_path": None,
        "absolute_frame": 384_000,
    }
    assert [warning["word"] for warning in state["audio_warnings"][:2]] == ["StreamMUSE", "trans"]
    assert state["latencies"]["synthesis_latency_ms"]["count"] == 1
    assert state["latencies"]["bar_render_latency_ms"]["max"] == 48.0
    assert state["latencies"]["audio_commit_slack_ms"]["total"] == 250.0


def test_future_render_events_do_not_overwrite_authoritative_playback_lifecycle() -> None:
    projector = RapStateProjector()
    for event in (
        _event(1, RapEventType.SESSION_STARTED, payload={"playback_state": "running"}),
        _event(2, RapEventType.BAR_PLAYBACK_STARTED, bar=1, payload={"absolute_frame": 192_000}),
        _event(3, RapEventType.AUDIO_RENDER_COMPLETED, bar=4, payload={"render_latency_ms": 12.0}),
        _event(4, RapEventType.BAR_AUDIO_READY, bar=4, payload={}),
        _event(5, RapEventType.BAR_AUDIO_COMMITTED, bar=3, payload={"deadline_slack_ms": 40.0}),
    ):
        projector.apply(event)

    audio = projector.snapshot()["audio"]
    assert audio["state"] == "running"
    assert audio["current_bar"] == 1
    assert audio["render_state"] == "committed"
    assert audio["render_bar"] == 3


def test_state_projector_bounds_audio_warnings_and_resets_audio_state() -> None:
    projector = RapStateProjector()
    for sequence in range(1, 131):
        projector.apply(
            _event(
                sequence,
                RapEventType.PRONUNCIATION_FALLBACK,
                bar=0,
                payload={"word": f"word-{sequence}"},
            )
        )
    state = projector.snapshot()

    assert len(state["audio_warnings"]) == 128
    assert state["audio_warnings"][0]["word"] == "word-3"
    projector.apply(_event(131, RapEventType.SESSION_RESET, payload={"playback_state": "stopped"}))
    state = projector.snapshot()

    assert state["audio_warnings"] == []
    assert state["audio"]["state"] == "stopped"
    assert state["audio"]["current_bar"] is None
    assert state["audio"]["render_state"] == "idle"
    assert state["audio"]["render_bar"] is None


def test_session_reset_clears_the_entire_projected_research_epoch() -> None:
    projector = RapStateProjector()
    events = (
        _event(1, RapEventType.SESSION_STARTED, payload={"tempo_bpm": 92.0, "audio_device": "Studio output"}),
        _event(2, RapEventType.BAR_RESERVED, bar=0, payload={"topic": "orbit"}),
        _event(3, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0", payload={"topic": "orbit"}),
        _event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 1, "latency_ms": 10.0}),
        _event(5, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True}),
        _event(6, RapEventType.FALLBACK_ACTIVATED, bar=0, payload={"fallback_reason": "deadline_miss"}),
        _event(7, RapEventType.SYLLABLE_EMITTED, bar=0, tick=0, payload={"label": "orbit", "observation_delay_ms": 1.0}),
        _event(8, RapEventType.AUDIO_RENDER_COMPLETED, bar=0, payload={"synthesis_latency_ms": 4.0, "render_latency_ms": 8.0}),
        _event(9, RapEventType.PRONUNCIATION_FALLBACK, bar=0, payload={"word": "orbit"}),
        _event(10, RapEventType.GENERATION_FAILED, bar=0, request_id="r0", payload={"error_message": "late"}),
        _event(11, RapEventType.SESSION_RESET, payload={"playback_state": "stopped"}),
    )
    for event in events:
        projector.apply(event)

    state = projector.snapshot()

    assert state["session_metadata"] == {"tempo_bpm": 92.0, "audio_device": "Studio output"}
    assert state["bars"] == {}
    assert state["frozen_bars"] == {}
    assert state["candidates"] == {}
    assert state["current_tick"] is None
    assert state["current_playback"] is None
    assert state["current_syllable"] is None
    assert state["current_segment"] is None
    assert state["pending_request"] is None
    assert state["latest_request"] is None
    assert state["latest_batch"] is None
    assert state["last_error"] is None
    assert state["emitted_syllables"] == []
    assert state["fallbacks"] == {"count": 0, "by_reason": {}}
    assert state["audio_warnings"] == []
    assert state["audio"]["state"] == "stopped"
    assert state["research_metrics"]["metrics"]["candidate_validity"]["denominator"] == 0
    assert all(value["count"] == 0 for value in state["latencies"].values())


def test_render_latency_uses_only_render_completed_for_one_real_bar_sequence() -> None:
    projector = RapStateProjector()
    for event in (
        _event(1, RapEventType.AUDIO_RENDER_COMPLETED, bar=0, payload={"synthesis_latency_ms": 12.0, "render_latency_ms": 48.0}),
        _event(2, RapEventType.BAR_AUDIO_READY, bar=0, payload={"render_latency_ms": 48.0}),
        _event(3, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"render_latency_ms": 48.0, "deadline_slack_ms": 250.0}),
    ):
        projector.apply(event)

    assert projector.snapshot()["latencies"]["bar_render_latency_ms"] == {
        "count": 1,
        "total": 48.0,
        "min": 48.0,
        "max": 48.0,
    }


def test_syllable_observation_delay_is_not_projected_as_physical_jitter() -> None:
    projector = RapStateProjector()
    projector.apply(
        _event(
            1,
            RapEventType.SYLLABLE_EMITTED,
            bar=0,
            tick=2,
            payload={"stress": 1, "subdivision": 2, "observation_delay_ms": 12.5},
        )
    )

    state = projector.snapshot()
    assert state["current_syllable"]["stress"] == 1
    assert state["current_syllable"]["subdivision"] == 2
    assert state["latencies"]["syllable_observation_delay_ms"]["total"] == 12.5
    assert "emission_jitter_ms" not in state["latencies"]

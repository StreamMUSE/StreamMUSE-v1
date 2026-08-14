"""Tests for the standalone rap tick loop."""

from collections import UserDict
from types import MappingProxyType

import pytest

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher
from streammuse.application.rap.runtime import RapAudioDemoDependencies, RapDemoDependencies, RapTickLoop
from streammuse.domain.rap import PlaybackState
from streammuse.domain.timing import Tempo


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_tick_loop_uses_absolute_monotonic_deadlines_without_drift() -> None:
    clock = FakeClock()
    ticks: list[int] = []
    loop = RapTickLoop(Tempo(120.0, 4, 4), on_tick=ticks.append, clock=clock, sleep=clock.sleep)

    loop.run(max_ticks=5)

    assert ticks == [0, 1, 2, 3, 4]
    assert clock.sleeps == pytest.approx([0.125, 0.125, 0.125, 0.125])


def test_tick_loop_compensates_for_callback_time_instead_of_accumulating_drift() -> None:
    clock = FakeClock()

    def on_tick(_tick: int) -> None:
        clock.now += 0.025

    loop = RapTickLoop(Tempo(120.0, 4, 4), on_tick=on_tick, clock=clock, sleep=clock.sleep)
    loop.run(max_ticks=3)

    assert clock.sleeps == pytest.approx([0.1, 0.1])


def test_demo_start_uses_the_configured_web_runtime_bar_limit(tmp_path, monkeypatch) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=())
    dispatcher.start()
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path,
        configured_max_bars=3,
    )
    observed: list[int] = []
    monkeypatch.setattr(dependencies, "run", lambda *, max_bars: observed.append(max_bars))

    dependencies.start()

    assert observed == [3]


def test_demo_session_start_records_resolved_repetition_window(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    events = []
    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(events.append,))
    dispatcher.start()
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4),
        Controller(),
        publisher,
        dispatcher,
        TickLoop(),
        tmp_path,
        repetition_window_bars=7,
    )
    dependencies.run(max_bars=1)

    assert events[0].payload["repetition_window_bars"] == 7


def test_demo_closes_dispatcher_and_recorder_when_controller_start_fails(tmp_path) -> None:
    calls: list[str] = []

    class Controller:
        def start(self) -> None:
            calls.append("start")
            raise RuntimeError("cannot start")

        def close(self) -> None:
            calls.append("controller_close")

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            raise AssertionError("tick loop must not run")

        def stop(self) -> None:
            calls.append("tick_stop")

    class Recorder:
        def __call__(self, event) -> None:
            calls.append(event.event_type.value)

        def close(self) -> None:
            calls.append("recorder_close")

    recorder = Recorder()
    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(recorder,))
    dispatcher.start()
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path,
        recorder=recorder,
    )

    with pytest.raises(RuntimeError, match="cannot start"):
        dependencies.run(max_bars=1)

    assert "session_started" in calls
    assert calls.index("tick_stop") < calls.index("controller_close")
    assert calls.index("session_stopped") < calls.index("recorder_close")
    assert calls.count("recorder_close") == 1


@pytest.mark.parametrize(
    "failing_phase",
    ("tick_stop", "controller_close", "session_stopped", "dispatcher_close"),
)
def test_demo_close_attempts_every_teardown_phase_once_after_failure(tmp_path, failing_phase: str) -> None:
    calls: list[str] = []
    failure = RuntimeError(f"{failing_phase} failed")

    def record(phase: str) -> None:
        calls.append(phase)
        if phase == failing_phase:
            raise failure

    class TickLoop:
        def stop(self) -> None:
            record("tick_stop")

    class Controller:
        def close(self) -> None:
            record("controller_close")

    class Publisher:
        def emit(self, _event_type, *, payload) -> None:
            record("session_stopped")

    class Dispatcher:
        def flush_and_close(self) -> None:
            record("dispatcher_close")

    class Recorder:
        def close(self) -> None:
            record("recorder_close")

    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4),
        Controller(),
        Publisher(),
        Dispatcher(),
        TickLoop(),
        tmp_path,
        recorder=Recorder(),
    )

    with pytest.raises(RuntimeError) as raised:
        dependencies.close()

    assert raised.value is failure
    assert calls == [
        "tick_stop",
        "controller_close",
        "session_stopped",
        "dispatcher_close",
        "recorder_close",
    ]

    dependencies.close()
    assert calls.count("tick_stop") == 1
    assert calls.count("controller_close") == 1
    assert calls.count("session_stopped") == 1
    assert calls.count("dispatcher_close") == 1
    assert calls.count("recorder_close") == 1


def test_demo_session_metadata_is_copied_and_cannot_override_canonical_values(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    metadata = {"scenario_id": "default_research_demo", "generator": "local_chat", "tempo_bpm": 1.0}
    events = []
    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(events.append,))
    dispatcher.start()
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path,
        repetition_window_bars=7, session_metadata=metadata,
    )
    metadata["scenario_id"] = "mutated"
    dependencies.run(max_bars=3)

    assert dependencies.session_metadata["scenario_id"] == "default_research_demo"
    assert events[0].payload["scenario_id"] == "default_research_demo"
    assert events[0].payload["tempo_bpm"] == 120.0
    assert events[0].payload["ticks_per_beat"] == 4
    assert events[0].payload["beats_per_bar"] == 4
    assert events[0].payload["max_bars"] == 3


def test_demo_dependencies_reject_nonpositive_repetition_window(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=())
    with pytest.raises(ValueError, match="repetition_window_bars"):
        RapDemoDependencies(Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path, 0)


def test_demo_session_metadata_recursively_freezes_mapping_proxy_and_user_dict(tmp_path) -> None:
    class Controller:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    class TickLoop:
        def run(self, max_ticks: int | None = None) -> None:
            return None

        def stop(self) -> None:
            return None

    nested = UserDict({"lines": [{"text": "original"}]})
    metadata = MappingProxyType({"nested": [nested, (MappingProxyType({"mode": "split"}),)]})
    publisher = RapEventPublisher("session")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=())
    dependencies = RapDemoDependencies(
        Tempo(120.0, 4, 4), Controller(), publisher, dispatcher, TickLoop(), tmp_path, session_metadata=metadata,
    )
    nested["lines"][0]["text"] = "mutated"

    assert dependencies.session_metadata["nested"][0]["lines"][0]["text"] == "original"
    assert dependencies.session_metadata["nested"][1][0]["mode"] == "split"


def test_audio_dependency_lifecycle_keeps_components_open_until_permanent_close(tmp_path) -> None:
    calls: list[str] = []

    class Controller:
        def __init__(self, playback) -> None:
            self.playback = playback
            self.starts = 0

        def start(self) -> None:
            calls.append("controller_start")
            self.starts += 1
            if self.starts == 1:
                self.playback.state = PlaybackState.PRIMING

        def resume_audio(self, bar: int) -> None:
            calls.append(f"controller_resume_{bar}")
            self.playback.state = PlaybackState.PRIMING

        def resume_after_stop(self) -> None:
            calls.append("controller_resume_after_stop")

        def request_stop(self, *, successor_bar: int) -> None:
            calls.append(f"controller_stop_{successor_bar}")

        def reset(self) -> None:
            calls.append("controller_reset")

        def close(self) -> None:
            calls.append("controller_close")

    class Playback:
        def __init__(self) -> None:
            self.state = PlaybackState.STOPPED
            self.current_tick: int | None = None
            self.next_start_bar = 0
            self.stop_successor_bar = 1

        def start(self) -> None:
            calls.append("playback_start")
            self.state = PlaybackState.RUNNING
            self.current_tick = 15

        def request_stop(self) -> None:
            calls.append("playback_stop")
            self.state = PlaybackState.STOP_REQUESTED

        def wait(self, timeout: float | None = None) -> None:
            calls.append("playback_wait")
            self.state = PlaybackState.STOPPED
            self.next_start_bar = 1

        def reset(self) -> None:
            calls.append("playback_reset")
            assert self.state == PlaybackState.STOPPED

        def close(self) -> None:
            calls.append("playback_close")
            self.state = PlaybackState.CLOSED

    class Coordinator:
        def close(self) -> None:
            calls.append("coordinator_close")

    class Publisher:
        def emit(self, *_args, **_kwargs) -> None:
            calls.append("session_stopped")

    class Dispatcher:
        def flush_and_close(self) -> None:
            calls.append("dispatcher_close")

    class Recorder:
        def close(self) -> None:
            calls.append("recorder_close")

    playback = Playback()
    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=Controller(playback),
        coordinator=Coordinator(),
        playback=playback,
        publisher=Publisher(),
        dispatcher=Dispatcher(),
        session_dir=tmp_path,
        recorder=Recorder(),
        configured_max_bars=1,
    )

    dependencies.start()
    dependencies.start()
    dependencies.reset()

    assert calls.count("controller_stop_1") == 2
    assert calls.count("playback_stop") == 2
    assert "controller_resume_1" in calls
    assert "controller_resume_after_stop" in calls

    assert calls.count("controller_close") == 0
    assert calls.count("coordinator_close") == 0
    assert calls.count("playback_close") == 0
    assert calls.count("dispatcher_close") == 0
    assert calls.count("recorder_close") == 0

    dependencies.close()

    assert calls.count("controller_close") == 1
    assert calls.count("coordinator_close") == 1
    assert calls.count("playback_close") == 1
    assert calls.count("dispatcher_close") == 1
    assert calls.count("recorder_close") == 1

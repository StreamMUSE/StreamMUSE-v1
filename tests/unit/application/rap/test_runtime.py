"""Tests for the standalone rap tick loop."""

from collections import UserDict
from threading import Event, Thread
from time import monotonic, sleep
from types import MappingProxyType, SimpleNamespace

import pytest

from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.runtime import RapAudioDemoDependencies, RapDemoDependencies, RapTickLoop
from streammuse.domain.rap import PlaybackState, RapEventType
from streammuse.domain.timing import Tempo
from streammuse.presentation.rap_demo.terminal_state import TerminalRapStateProjector


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

        def reset(self) -> int:
            calls.append("controller_reset")
            return 1

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
            dependencies.request_stop()
            self.state = PlaybackState.STOPPED
            self.next_start_bar = 1

        def reset(self, *, coordinator_epoch: int) -> None:
            calls.append("playback_reset")
            assert self.state == PlaybackState.STOPPED
            assert coordinator_epoch == 1

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
        configured_max_bars=0,
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


@pytest.mark.parametrize("loop", (False, True))
@pytest.mark.parametrize("controller_limit", (None, 2, 8))
def test_remote_finite_cap_restarts_automatically_even_when_scenario_has_more_bars(
    tmp_path,
    loop: bool,
    controller_limit: int | None,
) -> None:
    calls: list[object] = []

    class Playback:
        state = PlaybackState.STOPPED
        current_tick: int | None = None
        next_start_bar = 0
        stop_successor_bar = 2

        def start(self) -> None:
            calls.append("playback_start")
            self.state = PlaybackState.RUNNING
            self.current_tick = 31

        def request_stop(self) -> int:
            calls.append("playback_stop")
            self.state = PlaybackState.STOP_REQUESTED
            return 2

        def wait(self, timeout: float | None = None) -> None:
            self.state = PlaybackState.STOPPED
            self.next_start_bar = 2

        def reset(self, *, coordinator_epoch: int) -> None:
            calls.append(("playback_reset", coordinator_epoch))
            self.next_start_bar = 0
            self.current_tick = None

        def close(self) -> None:
            self.state = PlaybackState.CLOSED

    playback = Playback()

    class Controller:
        scenario = SimpleNamespace(loop=loop, total_bars=8)
        terminal_bar_limit = controller_limit

        def start(self) -> None:
            calls.append("controller_start")
            playback.state = PlaybackState.PRIMING

        def request_stop(self, *, successor_bar: int | None) -> None:
            calls.append(("controller_stop", successor_bar))

        def reset(self) -> int:
            calls.append("controller_reset")
            return 7

        def close(self) -> None:
            calls.append("controller_close")

    class Publisher:
        def emit(self, *_args, **_kwargs) -> None:
            return None

    class Dispatcher:
        def flush_and_close(self) -> None:
            return None

    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(120.0, 4, 4),
        controller=Controller(),
        coordinator=None,
        playback=playback,
        publisher=Publisher(),
        dispatcher=Dispatcher(),
        session_dir=tmp_path,
        configured_max_bars=2,
    )

    dependencies.start()

    assert dependencies.restart_requires_reset is True
    assert ("controller_stop", None) in calls
    dependencies.start()
    assert calls.count("controller_reset") == 1
    assert calls.count("controller_start") == 2
    assert ("playback_reset", 7) in calls
    dependencies.reset()
    assert dependencies.restart_requires_reset is False
    assert ("playback_reset", 7) in calls


def test_audio_dependency_allows_controller_owned_remote_strategy_without_coordinator(tmp_path) -> None:
    calls: list[str] = []

    class Component:
        def close(self) -> None:
            calls.append("close")

    class Dispatcher:
        def flush_and_close(self) -> None:
            calls.append("close")

    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=Component(),
        coordinator=None,
        playback=Component(),
        publisher=object(),
        dispatcher=Dispatcher(),
        session_dir=tmp_path,
    )

    dependencies.close()

    assert calls == ["close", "close", "close"]


def test_audio_dependency_arms_playback_before_a_blocked_controller_stop(tmp_path) -> None:
    class Controller:
        def __init__(self) -> None:
            self.entered = Event()
            self.release = Event()

        def request_stop(self, *, successor_bar: int) -> None:
            assert successor_bar == 2
            self.entered.set()
            assert self.release.wait(timeout=1.0)

    class Playback:
        state = PlaybackState.RUNNING
        stop_successor_bar = 1

        def __init__(self) -> None:
            self.stop_entered = Event()

        def request_stop(self) -> int:
            self.stop_entered.set()
            self.state = PlaybackState.STOP_REQUESTED
            return 2

    controller = Controller()
    playback = Playback()
    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=controller,
        coordinator=object(),
        playback=playback,
        publisher=object(),
        dispatcher=object(),
        session_dir=tmp_path,
    )
    worker = Thread(target=dependencies.request_stop)
    worker.start()

    try:
        assert playback.stop_entered.wait(timeout=1.0)
        assert controller.entered.wait(timeout=1.0)
    finally:
        controller.release.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()


def test_stop_during_startup_cancels_without_starting_audio(tmp_path) -> None:
    entered, release = Event(), Event()
    calls = []

    class Playback:
        state = PlaybackState.STOPPED
        current_tick = None
        next_start_bar = 0
        def start(self):
            calls.append("audio_started")
        def request_stop(self):
            self.state = PlaybackState.STOPPED
            return 0
        def reset(self, *, coordinator_epoch):
            self.state = PlaybackState.STOPPED
            calls.append(("playback_reset", coordinator_epoch))

    playback = Playback()
    class Controller:
        def start(self):
            entered.set()
            assert release.wait(1)
            playback.state = PlaybackState.PRIMING
        def request_stop(self, *, successor_bar):
            assert successor_bar is None
            release.set()
        def reset(self):
            calls.append("controller_reset")
            return 1

    runtime = RapAudioDemoDependencies(
        tempo=Tempo(60, 4, 4), controller=Controller(), coordinator=None,
        playback=playback, publisher=SimpleNamespace(emit=lambda *a, **k: None),
        dispatcher=object(), session_dir=tmp_path,
    )
    worker = Thread(target=runtime.start)
    worker.start()
    try:
        assert entered.wait(1)
        assert runtime.control_state == PlaybackState.PRIMING
        runtime.request_stop()
        worker.join(1)
        assert not worker.is_alive()
        assert "audio_started" not in calls
        assert ("playback_reset", 1) in calls
        assert runtime.control_state == PlaybackState.STOPPED
    finally:
        release.set()
        worker.join(1)


def test_runtime_reset_ignores_blocked_old_epoch_audio_events_in_monitor_and_terminal(tmp_path) -> None:
    class Controller:
        def __init__(self) -> None:
            self.epoch = 0

        def reset(self) -> int:
            self.epoch += 1
            return self.epoch

        def close(self) -> None:
            return None

    class Playback:
        state = PlaybackState.STOPPED

        def reset(self, *, coordinator_epoch: int) -> None:
            publisher.emit(
                RapEventType.SESSION_RESET,
                payload={"playback_state": "stopped", "coordinator_epoch": coordinator_epoch},
            )

        def close(self) -> None:
            self.state = PlaybackState.CLOSED

    class Coordinator:
        def close(self) -> None:
            return None

    monitor = RapStateProjector()
    terminal = TerminalRapStateProjector()
    received = []
    publisher = RapEventPublisher("reset-race")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(monitor, terminal, received.append))
    dispatcher.start()
    release_old_worker = Event()
    old_worker_ready = Event()

    def publish_old_completion() -> None:
        old_worker_ready.set()
        assert release_old_worker.wait(timeout=1.0)
        publisher.emit(
            RapEventType.AUDIO_RENDER_COMPLETED,
            bar=0,
            payload={"coordinator_epoch": 0, "synthesis_latency_ms": 5.0, "render_latency_ms": 9.0},
        )
        publisher.emit(
            RapEventType.PRONUNCIATION_FALLBACK,
            bar=0,
            payload={"coordinator_epoch": 0, "word": "stale"},
        )

    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=Controller(),
        coordinator=Coordinator(),
        playback=Playback(),
        publisher=publisher,
        dispatcher=dispatcher,
        session_dir=tmp_path,
    )
    worker = Thread(target=publish_old_completion)
    worker.start()
    assert old_worker_ready.wait(timeout=1.0)
    try:
        dependencies.reset()
        release_old_worker.set()
        worker.join(timeout=1.0)
        _wait_for_events(received, 3)

        reset_state = monitor.snapshot()
        assert reset_state["coordinator_epoch"] == 1
        assert reset_state["stopped"] is True
        assert reset_state["bars"] == {}
        assert reset_state["audio"]["state"] == "stopped"
        assert reset_state["audio_warnings"] == []
        assert reset_state["latencies"]["synthesis_latency_ms"]["count"] == 0
        assert terminal.state.stopped is True
        assert terminal.state.bars == {}
        assert terminal.state.audio["state"] == "stopped"
        assert terminal.state.audio_warnings == ()

        publisher.emit(
            RapEventType.AUDIO_RENDER_COMPLETED,
            bar=0,
            payload={"coordinator_epoch": 1, "synthesis_latency_ms": 7.0, "render_latency_ms": 11.0},
        )
        publisher.emit(
            RapEventType.PRONUNCIATION_FALLBACK,
            bar=0,
            payload={"coordinator_epoch": 1, "word": "current"},
        )
        _wait_for_events(received, 5)

        current_state = monitor.snapshot()
        assert current_state["latencies"]["synthesis_latency_ms"]["total"] == 7.0
        assert current_state["audio_warnings"][-1]["word"] == "current"
        assert terminal.state.audio["state"] == "stopped"
        assert terminal.state.audio["render_state"] == "rendering"
        assert terminal.state.audio_warnings[-1]["word"] == "current"
    finally:
        release_old_worker.set()
        worker.join(timeout=1.0)
        dependencies.close()


def test_runtime_reset_quiesces_old_tick_delivery_before_controller_epoch_reset(tmp_path) -> None:
    old_tick_finished = Event()
    quiesce_entered = Event()
    allow_old_tick_to_finish = Event()
    controller_reset = Event()
    order: list[str] = []

    class Controller:
        def reset(self) -> int:
            order.append("controller_reset")
            assert old_tick_finished.is_set()
            controller_reset.set()
            return 1

        def close(self) -> None:
            return None

    class Playback:
        state = PlaybackState.STOPPED

        def quiesce_for_reset(self) -> None:
            order.append("playback_quiesce")
            quiesce_entered.set()
            assert allow_old_tick_to_finish.wait(timeout=1.0)
            old_tick_finished.set()

        def reset(self, *, coordinator_epoch: int) -> None:
            order.append("playback_reset")
            assert coordinator_epoch == 1

        def close(self) -> None:
            self.state = PlaybackState.CLOSED

    class Coordinator:
        def close(self) -> None:
            return None

    class Dispatcher:
        def flush_and_close(self) -> None:
            return None

    dependencies = RapAudioDemoDependencies(
        tempo=Tempo(60.0, 4, 4),
        controller=Controller(),
        coordinator=Coordinator(),
        playback=Playback(),
        publisher=object(),
        dispatcher=Dispatcher(),
        session_dir=tmp_path,
    )
    worker = Thread(target=dependencies.reset)
    worker.start()
    try:
        assert quiesce_entered.wait(timeout=1.0)
        assert not controller_reset.is_set()
        allow_old_tick_to_finish.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive()
        assert order == ["playback_quiesce", "controller_reset", "playback_reset"]
    finally:
        allow_old_tick_to_finish.set()
        worker.join(timeout=1.0)
        dependencies.close()


def _wait_for_events(events: list[object], count: int) -> None:
    deadline = monotonic() + 1.0
    while len(events) < count and monotonic() < deadline:
        sleep(0.005)
    assert len(events) >= count


def test_session_start_reports_current_audio_settings_not_launch_defaults(tmp_path):
    from streammuse.application.rap.settings import RapSettings

    runtime, _ = _hardening_runtime(tmp_path)
    runtime.controller_factory = lambda values, epoch: None
    events = []
    runtime.publisher = SimpleNamespace(emit=lambda kind, **kwargs: events.append((kind, kwargs)))
    runtime.settings_service = SimpleNamespace(values=RapSettings(
        alignment_policy="all_onsets_stress_r3", chord_loop="major_cycle", chord_gain=0.4,
        drum_gain=0.6, vocal_gain=0.7,
    ))
    runtime.start()
    payload = events[0][1]["payload"]
    assert payload["audio"]["local_warp"]["policy"] == "all_onsets_stress_r3"
    assert payload["audio"]["mix"]["chord_loop"] == "major_cycle"
    assert payload["audio"]["mix"]["chord_gain"] == 0.4
    assert payload["audio"]["mix"]["drum_gain"] == 0.6


def _hardening_runtime(tmp_path):
    from streammuse.domain.rap.timeline import RapTimeline

    class Playback:
        state = PlaybackState.STOPPED
        next_start_bar = 0
        current_tick = None
        timeline = RapTimeline(Tempo(60, 4, 4))

        def reset(self, *, coordinator_epoch):
            assert self.state == PlaybackState.STOPPED
            self.next_start_bar = 0
            self.current_tick = None
            self.epoch = coordinator_epoch

        def set_timeline(self, timeline):
            self.timeline = timeline

        def request_stop(self):
            self.state = PlaybackState.STOPPED
            return 0

        def start(self):
            self.state = PlaybackState.RUNNING

        def wait(self, timeout=None):
            self.state = PlaybackState.STOPPED

    playback = Playback()

    class Controller:
        def __init__(self, bpm=60, epoch=0):
            self.initial = RapTimeline(Tempo(bpm, 4, 4))
            self.timeline = self.initial
            self.coordinator_epoch = epoch
            self.started = False
            self.closed = False

        def start(self):
            if self.started:
                return
            self.started = True
            playback.state = PlaybackState.PRIMING

        def reset(self):
            self.started = False
            self.coordinator_epoch += 1
            self.timeline = self.initial
            return self.coordinator_epoch

        def update_tempo(self, bpm):
            self.timeline = self.timeline.with_tempo(bpm, 3)
            playback.set_timeline(self.timeline)
            return {"effective_from_bar": 3}

        def update_mix(self, config):
            self.mix = config
            return {"effective_from_bar": 3}

        def request_stop(self, *, successor_bar):
            pass

        def close(self):
            self.closed = True

    runtime = RapAudioDemoDependencies(
        tempo=Tempo(60, 4, 4), controller=Controller(), coordinator=None,
        playback=playback, publisher=SimpleNamespace(emit=lambda *a, **k: None),
        dispatcher=object(), session_dir=tmp_path,
    )
    return runtime, Controller


def test_device_selection_updates_persistent_start_metadata_and_event(tmp_path):
    runtime, _ = _hardening_runtime(tmp_path)
    events = []
    runtime.publisher = SimpleNamespace(emit=lambda kind, **kwargs: events.append((kind, kwargs)))
    runtime.record_audio_device("Monitor speakers")
    assert runtime.session_metadata["audio"]["audio_device"] == "Monitor speakers"
    assert events[-1][0] == RapEventType.AUDIO_SETTINGS_CHANGED
    runtime.start()
    started = [payload for kind, payload in events if kind == RapEventType.SESSION_STARTED]
    assert started[0]["payload"]["audio"]["audio_device"] == "Monitor speakers"


def test_reset_reinstalls_controller_timeline_without_settings(tmp_path):
    runtime, _ = _hardening_runtime(tmp_path)
    runtime.controller.update_tempo(90)
    runtime.reset()
    assert runtime.playback.timeline.snapshot(4, 48_000).bpm == 60
    assert runtime.playback.timeline is runtime.controller.timeline


def test_canceled_start_reinstalls_controller_timeline(tmp_path):
    runtime, _ = _hardening_runtime(tmp_path)
    runtime.controller.update_tempo(90)
    start = runtime.controller.start

    def cancel_start():
        start()
        runtime.request_stop()

    runtime.controller.start = cancel_start
    runtime.start()
    assert runtime.control_state == PlaybackState.STOPPED
    assert runtime.playback.timeline is runtime.controller.timeline
    assert runtime.playback.timeline.snapshot(4, 48_000).bpm == 60


@pytest.mark.parametrize("cancel_start", [False, True])
def test_reset_rebuilds_accepted_settings_at_bar_zero(tmp_path, cancel_start):
    from streammuse.application.rap.audio_settings import install_audio_settings
    from streammuse.application.rap.settings import RapSettings

    runtime, Controller = _hardening_runtime(tmp_path)
    old = runtime.controller

    def factory(settings, epoch):
        replacement = Controller(settings.tempo_bpm, epoch)
        replacement.gains = (settings.drum_gain, settings.vocal_gain)
        return replacement

    service = install_audio_settings(
        runtime, RapSettings(), sink=object(), controller_factory=factory,
        capabilities_probe=lambda: {"tempo_updates_v1"},
    )
    runtime.playback.state = PlaybackState.RUNNING
    runtime.update_settings({"tempo_bpm": 90})
    runtime.update_settings({"drum_gain": 0.3, "vocal_gain": 0.4})
    runtime.playback.state = PlaybackState.STOPPED
    if cancel_start:
        def cancel():
            runtime.playback.state = PlaybackState.PRIMING
            runtime.request_stop()
        old.start = cancel
        runtime.start()
    else:
        runtime.reset()
    assert runtime.controller is not old
    assert old.closed
    assert runtime.controller.timeline.snapshot(0, 48_000).bpm == 90
    assert runtime.playback.timeline is runtime.controller.timeline
    assert runtime.controller.gains == (0.3, 0.4)
    assert service.values.tempo_bpm == 90


@pytest.mark.parametrize("recovery", ["explicit", "canceled", "failed"])
def test_reset_rebases_accepted_settings_metadata_for_new_epoch(tmp_path, recovery):
    from streammuse.application.rap.audio_settings import install_audio_settings
    from streammuse.application.rap.settings import RapSettings

    runtime, Controller = _hardening_runtime(tmp_path)
    update_tempo = runtime.controller.update_tempo
    runtime.controller.update_tempo = lambda bpm: {
        **update_tempo(bpm), "revision": 17, "coordinator_epoch": 0,
    }
    install_audio_settings(
        runtime, RapSettings(), sink=object(), capabilities_probe=lambda: {"tempo_updates_v1"},
        controller_factory=lambda settings, epoch: Controller(settings.tempo_bpm, epoch),
    )
    runtime.playback.state = PlaybackState.RUNNING
    accepted = runtime.update_settings({"tempo_bpm": 90})
    assert accepted["effective_from_bar"] == 3
    runtime.playback.state = PlaybackState.STOPPED

    def start():
        runtime.playback.state = PlaybackState.PRIMING
        if recovery == "failed":
            raise RuntimeError("startup failed")
        runtime.request_stop()

    if recovery == "explicit":
        runtime.reset()
    else:
        runtime.controller.start = start
        if recovery == "failed":
            with pytest.raises(RuntimeError, match="startup failed"):
                runtime.start()
        else:
            runtime.start()

    snapshot = runtime.settings_state()
    assert runtime.controller.timeline.snapshot(0, 48_000).bpm == 90
    assert snapshot["values"]["tempo_bpm"] == 90
    assert snapshot["effective_from_bar"] == 0
    assert snapshot["coordinator_epoch"] == runtime.controller.coordinator_epoch == 1
    assert "revision" not in snapshot


def test_reset_rebases_acceptance_without_waiting_for_settings_discovery(tmp_path):
    from streammuse.application.rap.settings import RapSettings, RapSettingsService

    runtime, Controller = _hardening_runtime(tmp_path)
    discover, entered, release, reset_done = Event(), Event(), Event(), Event()
    snapshots, errors = [], []

    def capabilities():
        if discover.is_set():
            entered.set()
            assert release.wait(2)
        return {"tempo_bpm": True}

    runtime.settings_service = RapSettingsService(
        RapSettings(), state=lambda: runtime.control_state.value, capabilities=capabilities,
        apply=lambda current, updated, changed: runtime.controller.update_tempo(updated.tempo_bpm),
    )
    runtime.controller_factory = lambda settings, epoch: Controller(settings.tempo_bpm, epoch)
    runtime.update_settings({"tempo_bpm": 90})
    discover.set()

    def snapshot():
        try:
            snapshots.append(runtime.settings_state())
        except BaseException as error:
            errors.append(error)

    def reset():
        try:
            runtime.reset()
            reset_done.set()
        except BaseException as error:
            errors.append(error)

    reader, resetter = Thread(target=snapshot), Thread(target=reset)
    reader.start()
    try:
        assert entered.wait(1)
        resetter.start()
        assert reset_done.wait(0.25), "reset waited behind settings discovery"
    finally:
        release.set()
        reader.join(2)
        if resetter.ident is not None:
            resetter.join(2)
    assert not reader.is_alive() and not resetter.is_alive()
    assert errors == []
    assert snapshots[0]["effective_from_bar"] == 0
    assert snapshots[0]["coordinator_epoch"] == 1


@pytest.mark.parametrize("failure_phase", ["controller", "playback", "publication"])
def test_failed_start_is_stopped_and_retry_uses_fresh_controller_state(tmp_path, failure_phase):
    runtime, _ = _hardening_runtime(tmp_path)
    owner = {"controller": runtime.controller, "playback": runtime.playback, "publication": runtime.publisher}[failure_phase]
    name = "emit" if failure_phase == "publication" else "start"
    original = getattr(owner, name)

    def fail(*args, **kwargs):
        if failure_phase == "controller":
            runtime.controller.started = True
        raise RuntimeError("startup failed")

    setattr(owner, name, fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        runtime.start()
    setattr(owner, name, original)
    assert runtime.control_state == PlaybackState.STOPPED
    runtime.start()
    assert runtime.controller.started
    assert runtime.control_state == PlaybackState.STOPPED


def test_failed_replacement_restores_timeline_and_requires_recovery(tmp_path):
    from streammuse.domain.rap.timeline import RapTimeline

    runtime, _ = _hardening_runtime(tmp_path)
    old = runtime.controller

    def fail(epoch):
        runtime.playback.set_timeline(RapTimeline(Tempo(100, 4, 4)))
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        runtime.replace_controller(fail)
    assert runtime.controller is old
    assert not old.closed
    assert runtime.playback.timeline is old.timeline
    assert runtime.restart_requires_reset


def test_retired_controller_close_failure_does_not_reject_accepted_settings(tmp_path, caplog):
    from streammuse.application.rap.audio_settings import install_audio_settings
    from streammuse.application.rap.settings import RapSettings

    runtime, Controller = _hardening_runtime(tmp_path)

    def fail_close():
        raise OSError("retired stream unavailable")

    runtime.controller.close = fail_close
    service = install_audio_settings(
        runtime, RapSettings(), sink=object(),
        controller_factory=lambda settings, epoch: Controller(settings.tempo_bpm, epoch),
        capabilities_probe=lambda: {"tempo_updates_v1"},
    )
    result = runtime.update_settings({"tempo_bpm": 100})
    assert result["values"]["tempo_bpm"] == 100
    assert service.values.tempo_bpm == 100
    assert runtime.controller.timeline.snapshot(0, 48_000).bpm == 100
    assert "retired stream unavailable" in caplog.text


def test_settings_network_wait_does_not_block_stop_or_allow_reset_race(tmp_path):
    runtime, _ = _hardening_runtime(tmp_path)
    entered, release, stopped = Event(), Event(), Event()

    def update(payload):
        entered.set()
        assert release.wait(2)
        return {}

    runtime.settings_service = SimpleNamespace(update=update)
    runtime.playback.state = PlaybackState.RUNNING
    runtime.controller.request_stop = lambda **kw: stopped.set()
    worker = Thread(target=lambda: runtime.update_settings({"tempo_bpm": 90}))
    worker.start()
    try:
        assert entered.wait(1)
        runtime.request_stop()
        assert stopped.is_set()
        with pytest.raises(RuntimeError, match="settings"):
            runtime.reset()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


def test_stop_at_sink_start_arms_playback_even_before_startup_returns(tmp_path):
    runtime, _ = _hardening_runtime(tmp_path)
    stopped = []
    stop = runtime.playback.request_stop

    def stop_playback():
        stopped.append("playback")
        return stop()

    def start_and_stop():
        runtime.playback.state = PlaybackState.RUNNING
        runtime.request_stop()

    runtime.playback.request_stop = stop_playback
    runtime.playback.start = start_and_stop
    runtime.start()
    assert stopped == ["playback"]
    assert runtime.control_state == PlaybackState.STOPPED


def _blocked_device_runtime(tmp_path, *, fail_start=False):
    from streammuse.application.rap.playback import RapPlaybackService
    from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar
    from streammuse.infrastructure.rap.audio_output import NullAudioSink

    entered, release = Event(), Event()
    runtime, _ = _hardening_runtime(tmp_path)
    audio_format = AudioFormat()

    class Sink(NullAudioSink):
        starts = 0

        def start(self):
            self.starts += 1
            entered.set()
            assert release.wait(3)
            if fail_start:
                raise OSError("device startup failed")
            return super().start()

    sink = Sink(audio_format=audio_format)
    playback = RapPlaybackService(
        tempo=runtime.tempo, sink=sink, publisher=None, on_tick=lambda tick: None,
        timeline=runtime.controller.timeline,
    )
    prepared = PreparedRapBar(
        bar=0, text="", source="test", fallback_reason=None, scheduled=(),
        audio=PcmAudio(audio_format, 192_000, bytes(192_000 * 2 * 4)),
        diagnostics=(), warnings=(), render_latency_ms=0,
        timing=runtime.controller.timeline.snapshot(0, audio_format.sample_rate_hz),
    )
    runtime.playback = playback
    runtime.controller.start = lambda: playback.enqueue(prepared)
    runtime.dispatcher = SimpleNamespace(flush_and_close=lambda: None)
    return runtime, sink, entered, release


@pytest.mark.parametrize("fail_start", [False, True])
def test_external_stop_cancels_blocked_device_start_without_waiting_for_device(tmp_path, fail_start):
    runtime, sink, entered, release = _blocked_device_runtime(tmp_path, fail_start=fail_start)
    stop_done = Event()
    errors = []

    def start():
        try:
            runtime.start()
        except BaseException as error:
            errors.append(error)

    def stop():
        try:
            runtime.request_stop()
            stop_done.set()
        except BaseException as error:
            errors.append(error)

    starter, stopper = Thread(target=start), Thread(target=stop)
    starter.start()
    try:
        assert entered.wait(1)
        stopper.start()
        assert stop_done.wait(0.25), "Stop waited for blocking device startup"
        assert runtime.control_state == PlaybackState.STOP_REQUESTED
        with pytest.raises(RuntimeError, match="preparing"):
            runtime.reset()
        runtime.start()
        assert sink.starts == 1
    finally:
        release.set()
        starter.join(2)
        if stopper.ident is not None:
            stopper.join(2)
        runtime.close()
    assert not starter.is_alive() and not stopper.is_alive()
    assert runtime.controller.coordinator_epoch == 1
    assert runtime.playback.next_start_bar == 0
    if fail_start:
        assert len(errors) == 1 and str(errors[0]) == "device startup failed"
    else:
        assert errors == []


def test_close_fences_runtime_while_device_start_is_blocked(tmp_path):
    runtime, _, entered, release = _blocked_device_runtime(tmp_path)
    closing = Event()
    errors = []
    close_playback = runtime.playback.close

    def close():
        closing.set()
        close_playback()

    runtime.playback.close = close

    def run(action):
        try:
            action()
        except BaseException as error:
            errors.append(error)

    starter = Thread(target=lambda: run(runtime.start))
    closer = Thread(target=lambda: run(runtime.close))
    starter.start()
    try:
        assert entered.wait(1)
        closer.start()
        assert closing.wait(0.25), "Close could not fence blocked device startup"
    finally:
        release.set()
        starter.join(2)
        if closer.ident is not None:
            closer.join(2)
    assert not starter.is_alive() and not closer.is_alive()
    assert errors == []
    assert runtime.control_state == PlaybackState.CLOSED
    assert runtime.controller.coordinator_epoch == 0


def test_cli_factory_reset_prepares_bar_zero_with_latest_tempo_and_mix(tmp_path):
    from streammuse.domain.rap.audio import AudioFormat, PcmAudio
    from streammuse.infrastructure.rap.audio_output import NullAudioSink
    from streammuse.presentation.rap_demo.cli import RapAudioFactories, build_demo, build_parser

    prepared = []

    class Sink(NullAudioSink):
        def enqueue(self, bar):
            prepared.append(bar)
            super().enqueue(bar)

    class AudioFactories(RapAudioFactories):
        def create_synthesizer(self):
            return SimpleNamespace(synthesize=lambda request: SimpleNamespace(
                audio=PcmAudio(AudioFormat(), 16, bytes(16 * 2 * 4)),
                pronunciation_source="test", renderer_phonemes=(), synthesis_latency_ms=0,
                warnings=(),
            ))

        def create_time_stretcher(self):
            return SimpleNamespace()

        def create_phrase_warper(self, *, policy):
            return SimpleNamespace(policy=policy, rubberband_version="test")

        def create_lyric_stream(self, *, base_url):
            return SimpleNamespace(
                open=lambda *args: None, send=lambda message: None, close=lambda: None,
                get_capabilities=lambda: {"tempo_updates_v1", "bounded_turns_v1"},
            )

        def create_bar_audio_source(self, *, base_url, opus_codec):
            return SimpleNamespace(close=lambda: None)

        def create_sink(self, *, output, audio_format, audio_file, audio_device):
            return Sink(audio_format=audio_format)

    args = build_parser().parse_args([
        "--rap-audio-renderer", "moss_aligned_remote", "--rap-protocol", "v3",
        "--audio-output", "wav", "--log-dir", str(tmp_path), "--tempo", "60",
        "--max-bars", "0", "--rap-render-startup-timeout", "0.01",
    ])
    runtime = build_demo(args, audio_factories=AudioFactories())
    try:
        runtime.update_settings({"tempo_bpm": 90})
        runtime.update_settings({"drum_gain": 0.3, "vocal_gain": 0.4})
        runtime.reset()
        runtime.controller.start()
        assert prepared[0].bar == 0
        assert prepared[0].timing.bpm == 90
        assert prepared[0].audio.frame_count == 128_000
        assert prepared[0].mix_config.drum_gain == 0.3
        assert prepared[0].mix_config.vocal_gain == 0.4
        assert runtime.playback.timeline is runtime.controller.timeline
        assert runtime.controller.coordinator_epoch == 2
    finally:
        runtime.close()

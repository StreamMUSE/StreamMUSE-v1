from types import SimpleNamespace
from threading import Event, Thread

import pytest

from streammuse.application.rap.inference import InferenceService
from streammuse.application.rap.inference.runtime_bridge import install_inference
from streammuse.application.rap.runtime import RapAudioDemoDependencies
from streammuse.domain.rap import PlaybackState
from streammuse.domain.timing import Tempo


class Backend:
    capabilities = {"commit_quantum_ticks": 1, "min_prefix_ticks": 1}

    def submit(self, *args):
        pass

    def cancel(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path):
    playback = SimpleNamespace(state=PlaybackState.STOPPED, current_tick=0, next_start_bar=0,
                               stop_successor_bar=1, reset=lambda **kw: None, close=lambda: None)

    def stop():
        playback.state = PlaybackState.STOP_REQUESTED
        return 1

    playback.request_stop = stop
    controller = SimpleNamespace(reset=lambda: 1, request_stop=lambda **kw: None, close=lambda: None)
    runtime = RapAudioDemoDependencies(Tempo(60, 4, 4), controller, None, playback,
        SimpleNamespace(emit=lambda *a, **kw: None), SimpleNamespace(flush_and_close=lambda: None), tmp_path)
    service = InferenceService(session_id="one-session", backend=Backend(), sample_rate=100, channels=1)
    install_inference(runtime, service)
    yield runtime
    runtime.close()


def test_legacy_activity_prevents_inference_configuration_and_start(runtime):
    runtime.playback.state = PlaybackState.RUNNING
    with pytest.raises(RuntimeError, match="legacy"):
        runtime.inference.configure({"bpm": 90})
    with pytest.raises(RuntimeError, match="legacy"):
        runtime.inference.start("tab")
    snap = runtime.inference.snapshot()
    assert snap["legacy_active"] is True
    assert snap["active_mode"] == "legacy"
    assert snap["session_id"] == "one-session"


def test_stop_from_inference_page_stops_legacy_audio(runtime):
    runtime.playback.state = PlaybackState.RUNNING
    runtime.inference.stop()
    assert runtime.playback.state == PlaybackState.STOP_REQUESTED


def test_legacy_stop_and_reset_route_to_inference_owner(runtime):
    runtime.inference.configure({"mode": "steering"})
    runtime.inference.start()
    assert runtime.control_state == PlaybackState.PRIMING
    runtime.request_stop()
    assert runtime.inference.snapshot()["state"] == "stopped"
    epoch = runtime.inference.snapshot()["epoch"]
    runtime.reset()
    assert runtime.inference.snapshot()["epoch"] > epoch


def test_legacy_start_cannot_compete_with_running_inference(runtime):
    runtime.inference.start("tab")
    with pytest.raises(RuntimeError, match="inference"):
        runtime.start()
    assert runtime.inference.snapshot()["phase"] == "USER"


def test_stop_during_device_open_still_fences_legacy_start(runtime):
    entered, release = Event(), Event()
    suspended = []
    class Output:
        def start(self, render):
            entered.set()
            release.wait(2)
        def suspend(self):
            suspended.append(True)
        def close(self):
            pass
    runtime.inference.service.output = Output()
    runtime.inference.configure({"mode": "steering"})
    thread = Thread(target=runtime.inference.start)
    thread.start()
    try:
        assert entered.wait(1)
        runtime.request_stop()
        with pytest.raises(RuntimeError, match="inference"):
            runtime.start()
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert suspended
    assert runtime.inference.snapshot()["state"] == "stopped"


def test_stopped_audio_settings_suspend_outside_gate_before_apply(runtime):
    actions = []
    class Output:
        def start(self, render):
            pass
        def suspend(self):
            assert not runtime._lifecycle_lock._is_owned()
            actions.append("suspend")
        def close(self):
            pass
    def update(payload, *, group):
        assert actions == ["suspend"]
        actions.append("settings")
        return {"applied": True}
    runtime.settings_service = SimpleNamespace(values=SimpleNamespace(mode="continuous"),
        update=update, reset_acceptance=lambda *args, **kwargs: None)
    runtime.inference.service.output = Output()
    runtime.inference.configure({"mode": "steering"})
    runtime.inference.start()
    assert not runtime.inference.snapshot()["capabilities"]["audio_controls_supported"]
    runtime.inference.stop()
    assert runtime.inference.snapshot()["capabilities"]["audio_controls_supported"]
    assert runtime.update_settings({"drum_gain": 0.5}, group="local") == {"applied": True}
    assert actions == ["suspend", "settings"]


@pytest.mark.parametrize("command", ["snapshot", "heartbeat", "tap", "steer"])
def test_bridge_projection_and_controls_copy_outside_shared_callback_gate(runtime, monkeypatch, command):
    bridge = runtime.inference
    if command == "steer":
        bridge.configure({"mode": "steering"})
        bridge.start()
    else:
        bridge.start("tab")
    snapshot = bridge.snapshot()
    payloads = {"heartbeat": {"owner_id": "tab"}, "steer": {"topic": "new"},
                "tap": {"owner_id": "tab", "session_id": snapshot["session_id"],
                        "epoch": snapshot["epoch"], "cycle_id": snapshot["cycle_id"],
                        "event_id": "once", "phase_elapsed_ms": 0}}
    from streammuse.application.rap.inference import runtime_bridge as module
    original = module.copy.deepcopy
    entered, release, played = Event(), Event(), Event()
    def slow_copy(value, *args, **kwargs):
        if isinstance(value, dict) and "requests" in value:
            entered.set()
            release.wait(2)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(module.copy, "deepcopy", slow_copy)
    def invoke():
        method = getattr(bridge, command)
        method() if command == "snapshot" else method(payloads[command])
    request = Thread(target=invoke)
    callback = Thread(target=lambda: (bridge.service.audio_callback(1), played.set()))
    request.start()
    try:
        assert entered.wait(1)
        callback.start()
        assert played.wait(0.2)
    finally:
        release.set()
        request.join(2)
        callback.join(2)

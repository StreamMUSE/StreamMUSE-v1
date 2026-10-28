from types import SimpleNamespace

import pytest

from streammuse.application.rap.audio_settings import install_audio_settings
from streammuse.application.rap.settings import RapSettings
from streammuse.domain.rap.audio import PlaybackState


def test_alignment_policy_updates_local_preparation_without_rebuild_or_remote_capability():
    changes = []
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=SimpleNamespace(
        update_alignment_policy=lambda policy: changes.append(policy) or {"effective_from_bar": 3}))
    runtime.replace_controller = lambda factory: pytest.fail("alignment must not rebuild")
    service = install_audio_settings(
        runtime, RapSettings(), sink=object(),
        controller_factory=lambda settings, epoch: (settings.alignment_policy, epoch),
    )
    result = service.update({"alignment_policy": "all_onsets_stress_r3"})
    assert changes == ["all_onsets_stress_r3"]
    assert result["effective_from_bar"] == 3
    assert not result.get("session_restarted", False)


def test_local_gain_update_never_calls_remote_capability_probe():
    calls, live = [], []
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=SimpleNamespace(
        update_mix=lambda config: calls.append(config) or {"effective_from_bar": 4}))
    sink = SimpleNamespace(set_live_gains=lambda **values: live.append(values) or {"ramp_ms": 10.0})

    def probe():
        pytest.fail("local update waited on remote discovery")

    service = install_audio_settings(runtime, RapSettings(), sink=sink, capabilities_probe=probe)
    result = service.update({"drum_gain": 0.2}, group="local")
    assert live == [{"drum_gain": 0.2, "vocal_gain": 0.8, "chord_gain": 0.2}]
    assert calls[0].drum_gain == 0.2
    assert result["application"]["drum_gain"] == "immediate"


def test_nondevice_backend_does_not_claim_immediate_gains_from_method_presence():
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=SimpleNamespace(update_mix=lambda config: {}))
    sink = SimpleNamespace(set_live_gains=lambda **values: {},
                           output_device_status=lambda: {"supports_live_gains": False, "backend": "timed"})
    service = install_audio_settings(runtime, RapSettings(), sink=sink)
    assert service.snapshot()["application"]["drum_gain"] == "bar-boundary"


def test_stopped_tempo_rebuild_preserves_local_change_during_remote_discovery():
    from threading import Event, Thread

    entered, release = Event(), Event()
    rebuilt, failures = [], []
    controller = SimpleNamespace(update_tempo=lambda bpm: {}, update_alignment_policy=lambda policy: {})
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=controller)
    runtime.replace_controller = lambda factory: rebuilt.append(factory(2))

    def probe():
        entered.set()
        assert release.wait(2)
        return {"tempo_updates_v1"}

    service = install_audio_settings(
        runtime, RapSettings(), sink=object(), capabilities_probe=probe,
        controller_factory=lambda settings, epoch: settings,
    )

    def change_tempo():
        try:
            service.update({"tempo_bpm": 80})
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=change_tempo)
    thread.start()
    try:
        assert entered.wait(0.5)
        service.update({"alignment_policy": "all_onsets_r3"})
        runtime.control_state = PlaybackState.STOPPED
    finally:
        release.set()
        thread.join(2)
    assert failures == []
    assert rebuilt[0].alignment_policy == "all_onsets_r3"
    assert rebuilt[0].tempo_bpm == 80
    assert service.values == rebuilt[0]


def test_chord_settings_update_mix_at_bar_boundary():
    mixes = []
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=SimpleNamespace(
        update_mix=lambda mix: mixes.append(mix) or {"effective_from_bar": 4}))
    service = install_audio_settings(runtime, RapSettings(), sink=object())
    result = service.update({"chord_loop": "minor_cycle", "chord_gain": 0.3})
    assert mixes[0].chord_loop == "minor_cycle"
    assert mixes[0].chord_gain == 0.3
    assert result["effective_from_bar"] == 4


@pytest.mark.parametrize("patch", [{"alignment_policy": "magic"}, {"chord_loop": "bad"}, {"chord_gain": -1}])
def test_invalid_audio_options_have_no_side_effects(patch):
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=object())
    service = install_audio_settings(runtime, RapSettings(), sink=object())
    with pytest.raises(ValueError):
        service.update(patch)


def test_settings_enumerate_stable_device_identity_and_apply_selection():
    selected = []
    recorded = []
    device = SimpleNamespace(to_payload=lambda: {"index": 9, "name": "Speakers", "host_api": "Core Audio", "available": True})
    sink = SimpleNamespace(
        list_output_devices=lambda: (device,), validate_output_device=lambda item: item,
        set_output_device=lambda item: selected.append(item),
    )
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=object(), record_audio_device=recorded.append)
    service = install_audio_settings(runtime, RapSettings(), sink=sink)
    snapshot = service.snapshot()
    identifier = snapshot["devices"][0]["id"]
    assert identifier != "9"
    service.update({"output_device": identifier})
    assert selected == [{"host_api": "Core Audio", "name": "Speakers"}]
    assert recorded == ["Speakers"]


def test_live_tempo_updates_current_controller_and_preserves_ack():
    calls = []
    controller = SimpleNamespace(update_tempo=lambda bpm: calls.append(bpm) or {"effective_from_bar": 5, "revision": 1})
    runtime = SimpleNamespace(control_state=PlaybackState.RUNNING, controller=controller)
    service = install_audio_settings(runtime, RapSettings(), sink=object(), capabilities_probe=lambda: {"tempo_updates_v1"})
    result = service.update({"tempo_bpm": 80})
    assert calls == [80]
    assert result["effective_from_bar"] == 5


def test_stopped_mode_switch_builds_distinct_controller_and_resets_session():
    changes = []
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=object())
    runtime.replace_controller = lambda factory: changes.append(factory(4))
    service = install_audio_settings(
        runtime, RapSettings(), sink=object(), capabilities_probe=lambda: {"bounded_turns_v1"},
        controller_factory=lambda settings, epoch: (settings.mode, settings.response_bars, epoch),
    )
    result = service.update({"mode": "alternating", "response_bars": 3})
    assert changes == [("alternating", 3, 4)]
    assert result["session_restarted"] is True


def test_stopped_bulk_rebuild_updates_retained_sink_gain_targets():
    rebuilt, live = [], []
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=SimpleNamespace(
        update_mix=lambda config: {"effective_from_bar": 0}))
    runtime.replace_controller = lambda factory: rebuilt.append(factory(4))
    sink = SimpleNamespace(set_live_gains=lambda **values: live.append(values) or values)
    service = install_audio_settings(
        runtime, RapSettings(), sink=sink, capabilities_probe=lambda: {"bounded_turns_v1"},
        controller_factory=lambda settings, epoch: settings,
    )
    service.update({"drum_gain": 0.3}, group="local")
    result = service.update({"mode": "alternating", "drum_gain": 0.9})
    assert rebuilt[-1].drum_gain == service.values.drum_gain == 0.9
    assert live[-1] == {"drum_gain": 0.9, "vocal_gain": 0.8, "chord_gain": 0.2}
    assert result["live_mix"]["drum_gain"] == 0.9


def test_failed_capability_probe_leaves_controls_disabled_without_fake_support():
    def missing():
        raise OSError("service offline")
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=SimpleNamespace(update_tempo=lambda bpm: {}))
    service = install_audio_settings(runtime, RapSettings(), sink=object(), capabilities_probe=missing)
    assert service.snapshot()["capabilities"]["tempo_bpm"] is False
    with pytest.raises(NotImplementedError):
        service.update({"tempo_bpm": 70})


@pytest.mark.parametrize("patch", [
    {"tempo_bpm": 80, "vocal_gain": 0.4},
    {"tempo_bpm": 80, "output_device": "speakers"},
])
def test_independent_settings_operations_are_rejected_before_side_effects(patch):
    calls = []
    controller = SimpleNamespace(
        update_tempo=lambda bpm: calls.append("tempo") or {},
        update_mix=lambda config: calls.append("mix") or {},
    )
    sink = SimpleNamespace(
        validate_output_device=lambda value: value,
        set_output_device=lambda value: calls.append("device"),
        list_output_devices=lambda: [],
    )
    runtime = SimpleNamespace(control_state=PlaybackState.STOPPED, controller=controller)
    service = install_audio_settings(
        runtime, RapSettings(), sink=sink, capabilities_probe=lambda: {"tempo_updates_v1"},
    )
    with pytest.raises(ValueError, match="separat"):
        service.update(patch)
    assert calls == []
    assert service.values == RapSettings()


def test_committed_mix_survives_installed_device_capability_discovery_failure():
    calls = 0
    applied = []

    def status():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("transient status failure")
        return {"supports_device_selection": True}

    runtime = SimpleNamespace(
        control_state=PlaybackState.RUNNING,
        controller=SimpleNamespace(update_mix=lambda config: applied.append(config) or {"effective_from_bar": 3}),
    )
    sink = SimpleNamespace(
        output_device_status=status, set_output_device=lambda value: None,
        list_output_devices=lambda: [],
    )
    service = install_audio_settings(runtime, RapSettings(), sink=sink)

    result = service.update({"vocal_gain": 0.4})

    assert result["values"]["vocal_gain"] == 0.4
    assert applied[0].vocal_gain == 0.4
    assert result["effective_from_bar"] == 3
    assert result["device_error"] == "transient status failure"
    recovered = service.snapshot()
    assert recovered["capabilities"]["output_device"] is True
    assert "device_error" not in recovered

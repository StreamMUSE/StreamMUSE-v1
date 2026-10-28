"""The live acceptance harness must distinguish device and timed playback."""

import pytest

from scripts.experiments.rap_interaction_acceptance import Acceptance, require_device_output


def test_live_acceptance_requires_an_active_device_stream():
    require_device_output({"backend": "sounddevice", "audible": True, "error": None})


@pytest.mark.parametrize("status", [
    {"backend": "timed", "audible": False},
    {"backend": "sounddevice", "audible": False},
    {"backend": "sounddevice", "audible": True, "error": "device disconnected"},
    {},
])
def test_silent_or_failed_output_cannot_pass_live_acceptance(status):
    with pytest.raises(AssertionError, match="active device"):
        require_device_output(status)


def test_turn_waits_for_running_playback_not_only_an_idle_controller():
    class ReadinessChecked(Exception):
        pass

    acceptance = object.__new__(Acceptance)

    def check_readiness(path, predicate):
        assert path == "/api/control/interaction"
        assert not predicate({"phase": "DRUMS_IDLE", "state": "priming", "running": True})
        assert not predicate({"phase": "DRUMS_IDLE", "state": "stopped", "running": False})
        assert predicate({"phase": "DRUMS_IDLE", "state": "running", "running": True})
        raise ReadinessChecked

    acceptance.wait = check_readiness
    with pytest.raises(ReadinessChecked):
        acceptance.turn(0, [])

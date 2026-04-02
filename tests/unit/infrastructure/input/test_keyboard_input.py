import pytest
import queue

from streammuse.domain.musical import EventType
from streammuse.infrastructure.input.keyboard import DEFAULT_KEY_TO_PITCH, KeyboardInput


def test_keyboard_mapping_matches_legacy_subset():
    # A couple high-signal checks for the legacy map used in archive/current_system.
    assert DEFAULT_KEY_TO_PITCH["z"] == 60
    assert DEFAULT_KEY_TO_PITCH["s"] == 61
    assert DEFAULT_KEY_TO_PITCH["m"] == 71
    assert DEFAULT_KEY_TO_PITCH[";"] == 75


def test_keyboard_input_emits_note_on_off_for_key():
    src = KeyboardInput()

    src._handle_key_down("z")
    ev1 = src._events.get_nowait()
    assert ev1.event_type == EventType.NOTE_ON
    assert ev1.pitch == 60
    assert ev1.tick == 0

    src._handle_key_up("z")
    ev2 = src._events.get_nowait()
    assert ev2.event_type == EventType.NOTE_OFF
    assert ev2.pitch == 60
    assert ev2.velocity == 0
    assert ev2.tick == 0


def test_keyboard_input_ignores_repeat_keydown_until_release():
    src = KeyboardInput()

    src._handle_key_down("z")
    _ = src._events.get_nowait()
    src._handle_key_down("z")
    with pytest.raises(queue.Empty):
        # no second event produced
        src._events.get_nowait()

    src._handle_key_up("z")
    _ = src._events.get_nowait()

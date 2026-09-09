"""Same-tick input order must agree across rolls, tokens and history state."""

from copy import deepcopy

import numpy as np
import pytest

from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import PianoMusicTokenizer
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import LekaiPromptEngine


PITCH = 70
INDEX = PITCH - 21


def on(tick, pitch=PITCH):
    return {"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 100}


def off(tick, pitch=PITCH):
    return {"type": "note_off", "pitch": pitch, "tick": tick, "velocity": 0}


@pytest.mark.parametrize("tick", [0, 3, 4, 5, 31, 32, 33])
def test_zero_tick_tap_has_no_onset_sustain_or_carried_state(tick):
    events = [on(tick), off(tick)]
    original = deepcopy(events)
    converter = MidiConverter(ticks_per_beat=4)
    backend = LekaiHttpBackend()
    prompt = LekaiPromptEngine()

    assert not converter.events_to_pianoroll(events, 0, 36).any()
    assert prompt._active_pitches_before_tick(events, tick + 1) == set()
    assert backend._active_pitches_before_tick(events, tick + 1) == set()
    assert backend._advance_active_pitches(events, 0, 36, set()) == set()
    assert backend._trim_event_history(events, tick + 1) == []
    assert backend._trim_event_history(events, tick) == events
    assert events == original


def test_retrigger_at_boundary_keeps_new_onset_and_sustain():
    events = [on(0), off(4), on(4), off(6)]
    converter = MidiConverter(ticks_per_beat=4)
    backend = LekaiHttpBackend()
    prompt = LekaiPromptEngine()
    full = converter.events_to_pianoroll(events, 0, 8)
    np.testing.assert_array_equal(full[0, INDEX], [1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(full[1, INDEX], [1, 0, 0, 0, 1, 0, 0, 0])
    assert prompt._active_pitches_before_tick(events, 5) == {PITCH}
    assert backend._active_pitches_before_tick(events, 5) == {PITCH}
    assert backend._trim_event_history(events, 5) == [on(4), off(6)]
    first = converter.events_to_pianoroll(events, 0, 4)
    active = backend._advance_active_pitches(events, 0, 4, set())
    assert active == {PITCH}
    second = converter.events_to_pianoroll(events, 4, 8, active_pitches=active)
    np.testing.assert_array_equal(np.concatenate([first, second], axis=2), full)
    assert backend._advance_active_pitches(events, 4, 8, active) == set()


def test_zero_tick_retrigger_closes_previous_note_without_resurrecting_it():
    events = [on(0), on(4), off(4)]
    roll = MidiConverter(ticks_per_beat=4).events_to_pianoroll(events, 0, 8)
    np.testing.assert_array_equal(roll[0, INDEX], [1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(roll[1, INDEX], [1, 0, 0, 0, 0, 0, 0, 0])
    assert LekaiHttpBackend()._trim_event_history(events, 5) == []


def test_same_tick_tap_then_real_onset_keeps_the_last_onset():
    events = [on(4), off(4), on(4), off(6)]
    roll = MidiConverter(ticks_per_beat=4).events_to_pianoroll(events, 4, 8)
    np.testing.assert_array_equal(roll[0, INDEX], [1, 1, 0, 0])
    np.testing.assert_array_equal(roll[1, INDEX], [1, 0, 0, 0])
    backend = LekaiHttpBackend()
    assert backend._active_pitches_before_tick(events, 5) == {PITCH}
    assert backend._trim_event_history(events, 5) == [on(4), off(6)]


def test_single_tick_note_crosses_beat_boundary_without_future_note_off_leak():
    events = [on(3), off(4)]
    converter = MidiConverter(ticks_per_beat=4)
    backend = LekaiHttpBackend()
    first = converter.events_to_pianoroll(events, 0, 4)
    np.testing.assert_array_equal(first[0, INDEX], [0, 0, 0, 1])
    np.testing.assert_array_equal(first[1, INDEX], [0, 0, 0, 1])
    active = backend._advance_active_pitches(events, 0, 4, set())
    assert active == {PITCH}
    assert backend._active_pitches_before_tick(events, 4) == {PITCH}
    assert not converter.events_to_pianoroll(events, 4, 8, active_pitches=active).any()
    assert backend._advance_active_pitches(events, 4, 8, active) == set()


def test_chronological_sort_preserves_ties_and_other_pitches():
    events = [off(7, 60), on(5), off(5), on(0, 60)]
    original = deepcopy(events)
    roll = MidiConverter(ticks_per_beat=4).events_to_pianoroll(events, 0, 8)
    assert not roll[:, INDEX].any()
    np.testing.assert_array_equal(roll[0, 60 - 21], [1, 1, 1, 1, 1, 1, 1, 0])
    assert roll[1, 60 - 21, 0] == 1
    assert events == original


@pytest.mark.parametrize("prompt_start", [0, 32])
def test_prompt_tokens_ignore_closed_zero_tick_taps_inside_and_before_window(prompt_start):
    engine = LekaiPromptEngine()
    events = [on(5), off(5), on(33, 61), off(33, 61)]
    actual, _, _ = engine._build_melody_prompt_tokens(events, prompt_start, 32, bpm=80)
    expected, _, _ = engine._build_melody_prompt_tokens([], prompt_start, 32, bpm=80)
    assert actual.tolist() == expected.tolist()


def test_continuation_tokens_and_history_ignore_closed_zero_tick_taps():
    backend = LekaiHttpBackend()
    backend._converter = MidiConverter(ticks_per_beat=4)
    backend._tokenizer = PianoMusicTokenizer()
    events = [on(5), off(5)]
    actual, active = backend._encode_beat_tokens(events, 4, set(), end_marker=171)
    expected, _ = backend._encode_beat_tokens([], 4, set(), end_marker=171)
    assert actual.tolist() == expected.tolist()
    assert active == set()
    assert backend._active_pitches_before_tick(events, 8) == set()
    assert backend._trim_event_history(events, 8) == []

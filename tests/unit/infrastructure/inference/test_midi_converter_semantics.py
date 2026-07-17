from __future__ import annotations

import numpy as np
import pytest

from streammuse.infrastructure.inference.lekai_http_backend import (
    TIMESTEPS_PER_BEAT,
    decode_part1_token_trace,
)
from streammuse.infrastructure.inference.lekai_model.inference_adapter import (
    beats_to_pianoroll,
)
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.lekai_model.my_tokenizer import (
    PianoRollTokenizer,
)


PITCH = 60
PITCH_INDEX = PITCH - 21


def _roll(sustain: list[int], onset: list[int]) -> np.ndarray:
    assert len(sustain) == len(onset)
    roll = np.zeros((2, 88, len(sustain)), dtype=np.uint8)
    roll[0, PITCH_INDEX] = sustain
    roll[1, PITCH_INDEX] = onset
    return roll


def _event_tuples(events: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    return [
        (str(event["type"]), int(event["pitch"]), int(event["tick"]))
        for event in events
    ]


@pytest.mark.parametrize(
    "sustain",
    (
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1],
    ),
)
def test_inactive_sustain_only_cells_are_semantically_inert(sustain):
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll(sustain, [0] * len(sustain)),
        start_tick=40,
        active_pitches=set(),
    )

    assert events == []
    assert active == set()


def test_sustain_only_prefix_does_not_synthesize_note_off_before_real_onset():
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([1, 1, 1, 1], [0, 0, 1, 0]),
        start_tick=100,
        active_pitches=set(),
    )

    assert _event_tuples(events) == [("note_on", PITCH, 102)]
    assert active == {PITCH}


@pytest.mark.parametrize(
    ("sustain", "onset", "initial", "expected", "expected_active"),
    (
        ([0, 0, 0, 0], [0, 0, 0, 0], {PITCH}, [("note_off", PITCH, 8)], set()),
        ([1, 1, 0, 0], [0, 0, 0, 0], {PITCH}, [("note_off", PITCH, 10)], set()),
        (
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            {PITCH},
            [("note_off", PITCH, 10), ("note_on", PITCH, 10)],
            {PITCH},
        ),
        (
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            set(),
            [("note_on", PITCH, 8), ("note_off", PITCH, 10)],
            set(),
        ),
    ),
)
def test_real_semantic_notes_keep_boundary_fall_and_retrigger_behavior(
    sustain, onset, initial, expected, expected_active
):
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll(sustain, onset),
        start_tick=8,
        active_pitches=initial,
    )

    assert _event_tuples(events) == expected
    assert active == expected_active


def test_single_tick_onset_has_one_tick_duration_when_closed_at_end():
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([0], [1]),
        start_tick=12,
        close_at_end=True,
        active_pitches=set(),
    )

    assert _event_tuples(events) == [
        ("note_on", PITCH, 12),
        ("note_off", PITCH, 13),
    ]
    assert active == set()


@pytest.mark.parametrize(
    ("close_at_end", "expected", "expected_active"),
    (
        (False, [], {PITCH}),
        (True, [("note_off", PITCH, 12)], set()),
    ),
)
def test_zero_tick_window_preserves_or_closes_real_carry(
    close_at_end, expected, expected_active
):
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([], []),
        start_tick=12,
        close_at_end=close_at_end,
        active_pitches={PITCH},
    )

    assert _event_tuples(events) == expected
    assert active == expected_active


def test_single_tick_onset_carries_to_next_window_when_not_closed():
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([0], [1]),
        start_tick=12,
        close_at_end=False,
        active_pitches=set(),
    )

    assert _event_tuples(events) == [("note_on", PITCH, 12)]
    assert active == {PITCH}


def test_real_single_tick_carry_is_closed_only_at_requested_window_end():
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([1], [0]),
        start_tick=12,
        close_at_end=True,
        active_pitches={PITCH},
    )

    assert _event_tuples(events) == [("note_off", PITCH, 13)]
    assert active == set()


def test_consecutive_onsets_retrigger_the_semantically_active_note():
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)

    events, active = converter.pianoroll_to_events(
        _roll([0, 0], [1, 1]),
        start_tick=20,
        active_pitches=set(),
    )

    assert _event_tuples(events) == [
        ("note_on", PITCH, 20),
        ("note_off", PITCH, 21),
        ("note_on", PITCH, 21),
    ]
    assert active == {PITCH}


def test_real_sustain_only_token_does_not_create_ghost_active_across_beats():
    records = [
        {
            "target_beat": 41,
            "start_tick": 164,
            "raw_tokens": [115, 40, 171],
            "boundary_tokens": [],
        },
        {
            "target_beat": 42,
            "start_tick": 168,
            "raw_tokens": [169],
            "boundary_tokens": [],
        },
    ]

    assert decode_part1_token_trace(records) == []


def test_real_mixed_token_keeps_only_onset_rooted_events():
    records = [
        {
            "target_beat": 53,
            "start_tick": 212,
            "raw_tokens": [118, 49, 84, 41, 85, 48, 84, 2, 171],
            "boundary_tokens": [],
        }
    ]

    events = decode_part1_token_trace(records)

    assert _event_tuples(events) == [
        ("note_on", 58, 213),
        ("note_on", 65, 213),
        ("note_off", 65, 215),
        ("note_on", 61, 215),
        ("note_on", 68, 215),
    ]


def test_true_carry_in_still_closes_on_empty_real_token():
    records = [
        {
            "target_beat": 0,
            "start_tick": 0,
            "raw_tokens": [169],
            "boundary_tokens": [],
        }
    ]

    events = decode_part1_token_trace(records, initial_active_pitches=[55])

    assert _event_tuples(events) == [("note_off", 55, 0)]


def test_wire_decode_matches_independent_onset_rooted_offline_roll():
    raw_beats = [
        [115, 40, 171],
        [169],
        [118, 49, 84, 41, 85, 48, 84, 2, 171],
        [111, 22, 88, 27, 84, 40, 88, 27, 171],
    ]
    tokenizer = PianoRollTokenizer(patch_h=1, patch_w=4)
    converter = MidiConverter(ticks_per_beat=TIMESTEPS_PER_BEAT)
    decoded_roll = beats_to_pianoroll(
        raw_beats,
        tokenizer=tokenizer,
        timesteps_per_beat=TIMESTEPS_PER_BEAT,
    )
    notes = converter.pianoroll_to_notes(decoded_roll)
    offline_roll = converter.notes_to_pianoroll(
        notes,
        max_tick=decoded_roll.shape[2],
    )
    records = [
        {
            "target_beat": index,
            "start_tick": index * TIMESTEPS_PER_BEAT,
            "raw_tokens": tokens,
            "boundary_tokens": [],
        }
        for index, tokens in enumerate(raw_beats)
    ]
    wire_events = decode_part1_token_trace(records)
    wire_roll = converter.events_to_pianoroll(
        wire_events,
        start_tick=0,
        end_tick=decoded_roll.shape[2],
    )

    np.testing.assert_array_equal(wire_roll, offline_roll)

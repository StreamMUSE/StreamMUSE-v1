import pytest

from streammuse.infrastructure.inference.lekai_prompt_continuation.catchup_state import CatchUpState


def test_prompt_running_user_continues_then_continuation_needs_gap_plus_one():
    state = CatchUpState()

    state.observe_melody_beats(8)
    state.observe_melody_beats(3)
    state.accept_prompt_accompaniment(8)

    assert state.melody_history_beats == 11
    assert state.accompaniment_history_beats == 8
    assert state.beats_needed_for_playback() == 4
    assert state.is_history_aligned() is False
    assert state.is_playback_ready() is False


def test_equal_history_lengths_are_not_playback_ready_by_default():
    state = CatchUpState(melody_history_beats=11, accompaniment_history_beats=11)

    assert state.is_history_aligned() is True
    assert state.beats_needed_for_playback() == 1
    assert state.is_playback_ready() is False


def test_next_accompaniment_beat_makes_playback_ready():
    state = CatchUpState(melody_history_beats=11, accompaniment_history_beats=8)

    state.accept_continuation_beats(3)
    assert state.is_history_aligned() is True
    assert state.is_playback_ready() is False

    state.accept_continuation_beats(1)
    assert state.accompaniment_history_beats == 12
    assert state.beats_needed_for_playback() == 0
    assert state.is_playback_ready() is True


def test_custom_lookahead_can_require_more_buffer():
    state = CatchUpState(
        melody_history_beats=11,
        accompaniment_history_beats=12,
        playable_lookahead_beats=2,
    )

    assert state.is_history_aligned() is True
    assert state.beats_needed_for_playback() == 1
    assert state.is_playback_ready() is False


def test_snapshot_contains_protocol_fields():
    state = CatchUpState(melody_history_beats=11, accompaniment_history_beats=12)

    assert state.snapshot() == {
        "melody_history_beats": 11,
        "accompaniment_history_beats": 12,
        "playable_lookahead_beats": 1,
        "target_playable_accompaniment_beats": 12,
        "beats_needed_for_playback": 0,
        "is_history_aligned": True,
        "is_playback_ready": True,
    }


def test_set_history_lengths_replaces_current_lengths():
    state = CatchUpState(melody_history_beats=11, accompaniment_history_beats=12)

    state.set_history_lengths(melody_beats=8, accompaniment_beats=8)

    assert state.melody_history_beats == 8
    assert state.accompaniment_history_beats == 8
    assert state.beats_needed_for_playback() == 1


def test_reset_clears_lengths_but_keeps_lookahead():
    state = CatchUpState(
        melody_history_beats=11,
        accompaniment_history_beats=12,
        playable_lookahead_beats=2,
    )

    state.reset()

    assert state.melody_history_beats == 0
    assert state.accompaniment_history_beats == 0
    assert state.playable_lookahead_beats == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"melody_history_beats": -1},
        {"accompaniment_history_beats": -1},
        {"playable_lookahead_beats": -1},
    ],
)
def test_negative_initial_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CatchUpState(**kwargs)


def test_negative_updates_are_rejected():
    state = CatchUpState()

    with pytest.raises(ValueError):
        state.observe_melody_beats(-1)
    with pytest.raises(ValueError):
        state.accept_accompaniment_beats(-1)
    with pytest.raises(ValueError):
        state.set_history_lengths(melody_beats=-1, accompaniment_beats=0)

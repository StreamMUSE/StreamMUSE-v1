import numpy as np

from streammuse.infrastructure.inference.lekai_model.Token2Midi import (
    _delay_minus_one_playable_part1_beats,
    process_delay_minus_one_part1_to_pianoroll,
)


def test_delay_minus_one_mapping_distinguishes_boundary_and_playable_bar():
    part0_beats = [
        [173],
        [255],
        [143, 170],
        [143, 170],
        [143, 170],
        [143, 170],
        [255],
        [143, 170],
    ]
    part1_beats = [
        [255],
        [169],
        [255],
        [81, 2, 171],
        [169],
        [255],
        [258, 82, 2, 171],
        [83, 2, 171],
    ]

    playable = _delay_minus_one_playable_part1_beats(part0_beats, part1_beats)

    assert playable == [
        [169],
        [255],
        [81, 2, 171],
        [169],
        [82, 2, 171],
        [83, 2, 171],
    ]


def test_delay_minus_one_pianoroll_keeps_one_time_slice_per_playable_beat():
    part0_beats = [
        [173],
        [255],
        [143, 170],
        [143, 170],
        [143, 170],
        [143, 170],
        [255],
        [143, 170],
    ]
    part1_beats = [
        [255],
        [169],
        [255],
        [81, 2, 171],
        [169],
        [255],
        [258, 82, 2, 171],
        [83, 2, 171],
    ]

    pianoroll = process_delay_minus_one_part1_to_pianoroll(part0_beats, part1_beats)

    assert pianoroll.shape == (2, 88, 24)
    assert np.count_nonzero(pianoroll[:, :, 0:8]) == 0
    assert np.count_nonzero(pianoroll[:, :, 8:12]) > 0
    assert np.count_nonzero(pianoroll[:, :, 12:16]) == 0
    assert np.count_nonzero(pianoroll[:, :, 16:24]) > 0

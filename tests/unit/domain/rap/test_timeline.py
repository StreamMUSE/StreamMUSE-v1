from dataclasses import FrozenInstanceError

import pytest

from streammuse.application.rap.audio_rendering import bar_frame_count, bar_start_frame, tick_frame_in_bar
from streammuse.domain.rap.audio import AudioFormat
from streammuse.domain.timing import Tempo


def timeline(bpm=92):
    from streammuse.domain.rap import timeline as module
    return module.RapTimeline(Tempo(bpm, 4, 4))


def test_static_snapshots_match_existing_rounding_for_every_tick():
    clock = timeline()
    tempo, fmt = Tempo(92, 4, 4), AudioFormat()
    for bar in range(100):
        snapshot = clock.snapshot(bar, fmt.sample_rate_hz)
        assert snapshot.start_frame == bar_start_frame(bar, tempo, fmt)
        assert snapshot.frame_count == bar_frame_count(bar, tempo, fmt)
        assert snapshot.tick_offsets == tuple(
            tick_frame_in_bar(bar, tick, tempo, fmt) for tick in range(16)
        ) + (snapshot.frame_count,)


@pytest.mark.parametrize('bpm', [30, 60, 120, 180])
def test_future_tempo_preserves_past_snapshots_and_boundary(bpm):
    original = timeline(90)
    changed = original.with_tempo(bpm, from_bar=4)
    assert changed.revision == 1
    assert tuple(changed.snapshot(bar, 48000) for bar in range(4)) == tuple(
        original.snapshot(bar, 48000) for bar in range(4)
    )
    assert changed.snapshot(4, 48000).start_frame == original.snapshot(4, 48000).start_frame
    assert changed.snapshot(4, 48000).bpm == bpm
    assert changed.snapshot(4, 48000).frame_count == round(240 / bpm * 48000)


def test_repeated_edits_replace_future_segments_without_changing_old_snapshots():
    initial = timeline(90)
    first = initial.with_tempo(120, from_bar=3)
    second = first.with_tempo(60, from_bar=5)
    replacement = second.with_tempo(100, from_bar=4)
    assert replacement.snapshot(3, 48000) == first.snapshot(3, 48000)
    assert replacement.snapshot(5, 48000).bpm == 100
    assert second.snapshot(5, 48000).bpm == 60
    frozen = replacement.snapshot(4, 48000)
    with pytest.raises(FrozenInstanceError):
        frozen.bpm = 90
    assert len(frozen.tick_offsets) == 17
    assert replacement.tick_to_seconds(64) == pytest.approx(first.tick_to_seconds(64))


@pytest.mark.parametrize('bpm', [0, 29, 181, float('nan'), float('inf'), True])
def test_invalid_live_bpm_is_rejected(bpm):
    with pytest.raises(ValueError):
        timeline().with_tempo(bpm, from_bar=1)


def test_audio_helpers_accept_full_snapshot_and_timeline():
    clock = timeline().with_tempo(120, from_bar=3)
    fmt = AudioFormat()
    snapshot = clock.snapshot(4, fmt.sample_rate_hz)
    for timing in (clock, snapshot):
        assert bar_start_frame(4, timing, fmt) == snapshot.start_frame
        assert bar_frame_count(4, timing, fmt) == snapshot.frame_count
        assert tick_frame_in_bar(4, 7, timing, fmt) == snapshot.tick_offsets[7]

from pathlib import Path

import pretty_midi

from experiments.lekai_sampling_monotony.analyze_task2_outputs import _output_features


def _write_repeated_bars(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody = pretty_midi.Instrument(program=0, name="Melody")
    accompaniment = pretty_midi.Instrument(program=0, name="Accompaniment")
    beat_seconds = 0.5
    for beat in (8, 12):
        accompaniment.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=60,
                start=beat * beat_seconds,
                end=(beat + 1) * beat_seconds,
            )
        )
    midi.instruments.extend([melody, accompaniment])
    midi.write(str(path))


def test_repetition_metrics_ignore_prompt_and_detect_identical_bars(tmp_path):
    midi_path = tmp_path / "output.mid"
    _write_repeated_bars(midi_path)

    result = _output_features(midi_path, bpm=120, total_steps=64, prefix_beats=8)

    assert result["acc_note_count"] == 2
    assert result["evaluated_bar_count"] == 2
    assert result["unique_bar_ratio"] == 0.5
    assert result["adjacent_exact_bar_repeat_rate"] == 1.0
    assert result["adjacent_exact_nonempty_bar_repeat_rate"] == 1.0
    assert result["max_identical_bar_run"] == 2
    assert result["max_identical_nonempty_bar_run"] == 2
    assert result["max_empty_bar_run"] == 0
    assert result["long_note_rate_gt_8_beats"] == 0.0
    assert result["piece_end_note_count"] == 1
    assert type(result["piece_end_note_count"]) is int
    assert result["piece_end_note_rate"] == 0.5

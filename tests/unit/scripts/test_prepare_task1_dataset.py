from __future__ import annotations

from collections import Counter

import numpy as np
import pretty_midi
import pytest

from experiments.lekai_melody_timing_robustness.prepare_task1_dataset import (
    perturb_onsets_on_model_grid,
    validate_prepared_npz,
)
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter


def _write_melody(path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=480)
    piano = pretty_midi.Instrument(program=0, name="Melody")
    for index, pitch in enumerate((60, 62, 64, 65)):
        piano.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=index * 0.5,
                end=index * 0.5 + 0.25,
            )
        )
    midi.instruments.append(piano)
    midi.write(str(path))


def _shape_counter(path):
    converter = MidiConverter(ticks_per_beat=4)
    midi, _ = converter.load_midi(str(path))
    notes, _ = converter.midi_to_notes(midi)
    return Counter((n["pitch"], n["duration"], n["program"]) for n in notes)


def test_grid_perturbation_survives_4tpb_and_preserves_duration(tmp_path):
    source = tmp_path / "source.mid"
    output = tmp_path / "perturbed.mid"
    _write_melody(source)

    report = perturb_onsets_on_model_grid(
        source,
        output,
        selection_rate=0.5,
        shift_steps=1,
        seed=7,
    )

    assert output.is_file()
    assert report["selected_note_count"] == 2
    assert report["effective_changed_note_records"] == 2
    assert report["pianoroll_changed_cells"] > 0
    assert report["pianoroll_equal"] is False
    assert _shape_counter(source) == _shape_counter(output)


def test_grid_perturbation_is_reproducible(tmp_path):
    source = tmp_path / "source.mid"
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"
    _write_melody(source)

    first_report = perturb_onsets_on_model_grid(
        source, first, selection_rate=1.0, shift_steps=1, seed=2
    )
    second_report = perturb_onsets_on_model_grid(
        source, second, selection_rate=1.0, shift_steps=1, seed=2
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["intended_delta_step_counts"] == second_report["intended_delta_step_counts"]


def _write_npz(path, measure):
    np.savez(
        path,
        metadata={"time_signature_idx": 4, "bpm": 120, "num_measures": 1},
        measure_0=measure,
    )


def test_npz_validation_requires_only_melody_to_change(tmp_path):
    baseline = np.zeros((4, 88, 16), dtype=np.float32)
    baseline[2, 40, 0:4] = 1
    perturbed = baseline.copy()
    perturbed[1, 39, 1] = 1
    baseline_path = tmp_path / "baseline.npz"
    perturbed_path = tmp_path / "perturbed.npz"
    _write_npz(baseline_path, baseline)
    _write_npz(perturbed_path, perturbed)
    rows = [
        {"piece_id": "x", "condition": "baseline", "npz": str(baseline_path)},
        {"piece_id": "x", "condition": "onset", "npz": str(perturbed_path)},
    ]

    validate_prepared_npz(rows)

    assert rows[1]["npz_melody_changed_cells"] == 1
    assert rows[1]["npz_accompaniment_changed_cells"] == 0


def test_npz_validation_rejects_accompaniment_changes(tmp_path):
    baseline = np.zeros((4, 88, 16), dtype=np.float32)
    perturbed = baseline.copy()
    perturbed[1, 39, 1] = 1
    perturbed[3, 40, 1] = 1
    baseline_path = tmp_path / "baseline.npz"
    perturbed_path = tmp_path / "perturbed.npz"
    _write_npz(baseline_path, baseline)
    _write_npz(perturbed_path, perturbed)
    rows = [
        {"piece_id": "x", "condition": "baseline", "npz": str(baseline_path)},
        {"piece_id": "x", "condition": "onset", "npz": str(perturbed_path)},
    ]

    with pytest.raises(RuntimeError, match="accompaniment changed"):
        validate_prepared_npz(rows)

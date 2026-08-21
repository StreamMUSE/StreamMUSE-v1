import csv
import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from experiments.lekai_failcase_analysis.extract_features import extract_feature_tables
from experiments.lekai_failcase_analysis.music_features import (
    BeatNote,
    TrackSelectionError,
    hard_input_features,
    instrument_to_beat_notes,
    js_divergence,
    load_prompt_accompaniment,
    melody_features,
    quantize_beat,
    select_source_melody,
    window_features,
)


def _write_midi(path: Path, tracks, tempo=60.0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=480)
    for name, notes in tracks:
        instrument = pretty_midi.Instrument(program=0, name=name)
        for start, end, pitch in notes:
            instrument.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end)
            )
        midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_quantization_matches_python_round_and_midi_beat_conversion(tmp_path):
    assert quantize_beat(0.125) == 0.0
    assert quantize_beat(0.375) == 0.5
    assert quantize_beat(0.625) == 0.5

    path = tmp_path / "tempo.mid"
    _write_midi(path, [("Melody", [(1.0, 1.24, 60)])], tempo=60.0)
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = instrument_to_beat_notes(midi, midi.instruments[0])
    assert notes == [BeatNote(start=1.0, end=1.25, pitch=60)]


def test_leading_blank_quantized_features_and_entropy():
    notes = [
        BeatNote(2.0, 2.25, 60),
        BeatNote(2.0, 2.5, 64),
        BeatNote(2.25, 2.5, 61),
        BeatNote(2.5, 2.75, 62),
        BeatNote(2.75, 3.0, 63),
    ]
    hard = hard_input_features(notes)
    assert hard["first8_note_count"] == 5
    assert hard["first8_leading_blank_beats"] == 2.0
    assert hard["first8_chord_onset_ratio"] == 0.25
    assert hard["first8_polyphonic_active_ratio"] > 0

    features = window_features(notes, 8.0)
    assert features["onset_phase_entropy"] == pytest.approx(1.0)
    assert features["active_coverage"] == pytest.approx(1.0 / 8.0)
    assert features["one_step_duration_rate"] == pytest.approx(4.0 / 5.0)


def test_js_divergence_endpoints_and_prefix_full_features():
    assert js_divergence(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert np.isnan(js_divergence(None, np.array([1.0, 0.0])))

    notes = [BeatNote(0.0, 1.0, 60), BeatNote(8.0, 9.0, 61)]
    features = melody_features(notes)
    assert features["full_horizon_beats"] == 9.0
    assert features["prefix_full_pitch_class_js_divergence"] > 0


def test_track_selection_is_semantic_and_empty_prompt_is_valid(tmp_path):
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(
        [
            pretty_midi.Instrument(0, name="Accompaniment"),
            pretty_midi.Instrument(0, name="mELoDy"),
        ]
    )
    assert select_source_melody(midi).name == "mELoDy"

    ambiguous = pretty_midi.PrettyMIDI()
    ambiguous.instruments.extend(
        [pretty_midi.Instrument(0, name="Piano A"), pretty_midi.Instrument(0, name="Piano B")]
    )
    with pytest.raises(TrackSelectionError, match="expected one non-drum"):
        select_source_melody(ambiguous)

    empty_path = tmp_path / "empty_prompt.mid"
    _write_midi(empty_path, [("Melody", [(0.0, 0.5, 60)])])
    name, notes, present = load_prompt_accompaniment(empty_path)
    assert (name, notes, present) == ("", [], False)

    acc_only_path = tmp_path / "acc_only.mid"
    _write_midi(acc_only_path, [("aCCOMPaniment", [(0.0, 0.5, 48)])])
    name, notes, present = load_prompt_accompaniment(acc_only_path)
    assert name == "aCCOMPaniment"
    assert present is True
    assert len(notes) == 1


def test_end_to_end_tables_preserve_empty_prompt_semantics(tmp_path):
    bundle = tmp_path / "bundle"
    piece_dir = bundle / "listening_by_style" / "anime" / "piece_a"
    _write_midi(piece_dir / "00_source_melody.mid", [("Melody", [(1.0, 1.5, 60), (2.0, 2.5, 62)])])
    _write_midi(piece_dir / "empty.mid", [("Melody", [(0.0, 0.5, 60)])])
    _write_midi(piece_dir / "acc.mid", [("Accompaniment", [(0.0, 0.5, 48), (4.0, 4.5, 48)])])

    manifest_fields = ["style", "piece", "seed", "condition", "status", "prompt_midi"]
    _write_csv(
        bundle / "run_manifest.csv",
        manifest_fields,
        [
            {
                "style": "anime", "piece": "piece_a", "seed": "0", "condition": "single_n1",
                "status": "ok", "prompt_midi": "listening_by_style/anime/piece_a/empty.mid",
            },
            {
                "style": "anime", "piece": "piece_a", "seed": "0", "condition": "rule_s_n5",
                "status": "ok", "prompt_midi": "listening_by_style/anime/piece_a/acc.mid",
            },
        ],
    )
    label_fields = [
        "style", "piece", "npz_time_signature", "meter_status", "seed", "condition",
        "score_raw", "fail_type_raw", "quality_floor", "quality_mean", "issue_any",
        "severe_fail", "label_status", "include_main", "exclusion_reason", "comments",
    ]
    base = {
        "style": "anime", "piece": "piece_a", "npz_time_signature": "4/4",
        "meter_status": "include_4_4", "seed": "0", "score_raw": "4",
        "fail_type_raw": "", "quality_floor": "4.0", "quality_mean": "4.0",
        "issue_any": "False", "severe_fail": "False", "label_status": "rated",
        "include_main": "True", "exclusion_reason": "", "comments": "",
    }
    _write_csv(
        tmp_path / "labels.csv",
        label_fields,
        [dict(base, condition="single_n1"), dict(base, condition="rule_s_n5")],
    )

    audit = extract_feature_tables(bundle, tmp_path / "labels.csv", tmp_path / "out")
    with (tmp_path / "out" / "prompt_features.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    empty = next(row for row in rows if row["condition"] == "single_n1")
    assert empty["prompt_extraction_status"] == "empty_prompt"
    assert empty["prompt_has_accompaniment"] == "0"
    assert empty["prompt_note_count"] == "0.0"
    assert empty["prompt_active_coverage"] == "0.0"
    assert empty["prompt_empty_beat_rate"] == "1.0"
    assert empty["prompt_average_voice_number"] == "0.0"
    assert empty["prompt_duration_median"] == ""
    assert empty["prompt_pitch_class_entropy"] == ""
    assert empty["prompt_groove_consistency"] == ""
    assert audit["track_and_file_status_counts"]["empty_prompt"] == 1
    assert audit["main_track_and_file_status_counts"]["empty_prompt"] == 1
    assert audit["track_and_file_status_counts"]["missing_prompt_midi"] == 0
    assert audit["row_counts"] == {
        "labels_runs": 2,
        "manifest_runs": 2,
        "melody_features": 1,
        "prompt_features": 2,
        "main_melody_features": 1,
        "main_prompt_features": 2,
    }
    assert json.loads((tmp_path / "out" / "feature_audit.json").read_text()) == audit

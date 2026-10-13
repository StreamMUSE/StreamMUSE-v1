from __future__ import annotations

import json

import pretty_midi

from experiments.lekai_melody_timing_robustness.analyze_task1_outputs import analyze


def _write_output(path, acc_pitch):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    melody = pretty_midi.Instrument(program=0, name="Melody")
    melody.notes.append(pretty_midi.Note(velocity=64, pitch=72, start=0.0, end=0.5))
    accompaniment = pretty_midi.Instrument(program=0, name="Accompaniment")
    accompaniment.notes.append(
        pretty_midi.Note(velocity=64, pitch=acc_pitch, start=0.0, end=0.5)
    )
    midi.instruments.extend((melody, accompaniment))
    midi.write(str(path))


def test_analysis_maps_outputs_and_compares_accompaniment(tmp_path):
    baseline_midi = tmp_path / "baseline.mid"
    changed_midi = tmp_path / "changed.mid"
    _write_output(baseline_midi, 48)
    _write_output(changed_midi, 50)
    prepared_manifest = tmp_path / "prepared.json"
    prepared_manifest.write_text(
        json.dumps(
            [
                {
                    "piece_id": "x",
                    "condition": "baseline",
                    "selection_rate_requested": 0.0,
                    "npz": "/data/x__baseline.npz",
                },
                {
                    "piece_id": "x",
                    "condition": "onset_tick1_pct10_seed0",
                    "selection_rate_requested": 0.1,
                    "npz": "/data/x__onset_tick1_pct10_seed0.npz",
                },
            ]
        ),
        encoding="utf-8",
    )
    inference_manifest = tmp_path / "inference.json"
    inference_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "file_name": "x__baseline.npz",
                        "gen_midi_path": str(baseline_midi),
                        "metadata": {"total_steps": 4},
                    },
                    {
                        "file_name": "x__onset_tick1_pct10_seed0.npz",
                        "gen_midi_path": str(changed_midi),
                        "metadata": {"total_steps": 4},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows, summary = analyze(prepared_manifest, inference_manifest, tmp_path / "analysis")

    assert rows[0]["acc_exact_match_to_baseline"] is True
    assert rows[1]["acc_exact_match_to_baseline"] is False
    assert rows[1]["acc_pianoroll_cell_jaccard"] == 0.0
    assert summary["perturbed_exact_match_count"] == 0
    assert (tmp_path / "analysis" / "listening_by_piece" / "x" / "x__baseline.mid").is_file()

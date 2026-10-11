import numpy as np

from experiments.lekai_cross_style.build_style_candidates import (
    _metadata_text,
    _npz_stats,
    _term_hits,
)


def test_metadata_text_separates_titles_and_creators():
    record = {
        "movement_title": "Clair de Lune",
        "creators": {"composer": ["Claude Debussy"]},
        "credits": [
            {"type": "subtitle", "words": ["Piano Solo"]},
            {"type": "arranger", "words": ["Example Arranger"]},
        ],
    }

    titles, creators = _metadata_text(record)

    assert titles == ["Clair de Lune", "Piano Solo"]
    assert creators == ["Claude Debussy", "Example Arranger"]
    assert _term_hits(" | ".join(creators), ["debussy"]) == ["debussy"]


def test_npz_stats_reads_four_channel_note_onsets(tmp_path):
    measure = np.zeros((4, 88, 16), dtype=np.uint8)
    measure[1, 40, 0] = 1
    measure[3, 36, 0] = 1
    measure[3, 40, 4] = 1
    path = tmp_path / "piece.npz"
    np.savez_compressed(
        path,
        measure_0=measure,
        metadata=np.array(
            {
                "time_signature": "4/4",
                "bpm": 120.0,
                "num_measures": 1,
                "num_channels": 4,
                "total_length": 16,
            },
            dtype=object,
        ),
    )

    result = _npz_stats(path)

    assert result["shape_valid"] is True
    assert result["melody_note_count"] == 1
    assert result["accompaniment_note_count"] == 2
    assert result["melody_density_per_beat"] == 0.25
    assert result["accompaniment_density_per_beat"] == 0.5

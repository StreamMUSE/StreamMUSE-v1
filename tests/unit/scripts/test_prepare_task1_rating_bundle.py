import csv
from pathlib import Path

from experiments.lekai_melody_timing_robustness.prepare_rating_bundle import main


def test_prepare_rating_bundle_uses_short_ordered_names(tmp_path, monkeypatch):
    analysis = tmp_path / "analysis"
    listening = analysis / "listening_by_piece"
    rows = []
    conditions = ["baseline"] + [
        f"onset_tick1_pct{rate}_seed{seed}"
        for rate in (10, 25, 50)
        for seed in range(3)
    ]
    for piece_id in ("43038", "3410421", "2581346"):
        piece_dir = listening / piece_id
        piece_dir.mkdir(parents=True)
        for condition in conditions:
            midi = piece_dir / f"{piece_id}__{condition}.mid"
            midi.write_bytes(b"midi")
            rows.append(
                {
                    "comments": "",
                    "condition": condition,
                    "listening_midi": str(midi.relative_to(analysis)),
                    "melody_accompaniment_fit_rating": "",
                    "musical_quality_rating": "",
                    "noticeable_degradation": "",
                    "piece_id": piece_id,
                    "seed": "",
                    "selection_rate_requested": "",
                }
            )
    with (analysis / "ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "bundle"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_rating_bundle.py",
            "--analysis-dir",
            str(analysis),
            "--output-root",
            str(output),
        ],
    )
    main()

    assert (output / "01_43038" / "00_base.mid").is_file()
    assert (output / "01_43038" / "09_p50_s2.mid").is_file()
    rating_rows = list(csv.DictReader((output / "ratings.csv").open(encoding="utf-8")))
    assert len(rating_rows) == 30
    assert rating_rows[0]["midi"] == str(Path("01_43038") / "00_base.mid")
    assert rating_rows[-1]["midi"] == str(Path("03_2581346") / "09_p50_s2.mid")

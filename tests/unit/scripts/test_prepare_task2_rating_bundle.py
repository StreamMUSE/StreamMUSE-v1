import csv
from pathlib import Path

from experiments.lekai_sampling_monotony.prepare_rating_bundle import main


def test_prepare_task2_rating_bundle_uses_short_ordered_names(tmp_path, monkeypatch):
    analysis = tmp_path / "analysis"
    listening = analysis / "listening_by_piece"
    rows = []
    conditions = [("greedy", 0)] + [("sampling", seed) for seed in range(5)]
    for piece_id in ("43038", "3410421", "2581346"):
        piece_dir = listening / piece_id
        piece_dir.mkdir(parents=True)
        for mode, seed in conditions:
            midi = piece_dir / f"{piece_id}__{mode}__seed{seed}.mid"
            midi.write_bytes(b"midi")
            rows.append(
                {
                    "comments": "",
                    "stability_rating_1_low_5_high": "",
                    "listening_midi": str(midi.relative_to(analysis)),
                    "melody_accompaniment_fit_1_low_5_high": "",
                    "mode": mode,
                    "variety_rating_1_low_5_high": "",
                    "musical_quality_rating_1_low_5_high": "",
                    "piece_id": piece_id,
                    "seed": seed,
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

    assert (output / "01_43038" / "00_greedy.mid").is_file()
    assert (output / "01_43038" / "05_sample4.mid").is_file()
    rating_rows = list(csv.DictReader((output / "ratings.csv").open(encoding="utf-8")))
    assert len(rating_rows) == 18
    assert rating_rows[0]["midi"] == str(Path("01_43038") / "00_greedy.mid")
    assert rating_rows[-1]["midi"] == str(Path("03_2581346") / "05_sample4.mid")
    assert "variety_1_low_5_high" in rating_rows[0]
    assert "stability_1_low_5_high" in rating_rows[0]

import csv
import json

from experiments.lekai_cross_style.prepare_task3_inputs import main


def test_prepare_task3_inputs_materializes_frozen_selection(tmp_path, monkeypatch):
    shortlist = tmp_path / "shortlist.csv"
    fields = [
        "style",
        "piece_id",
        "title",
        "creator",
        "primary_evidence",
        "bpm",
        "num_measures",
        "melody_density_per_beat",
        "accompaniment_density_per_beat",
    ]
    with shortlist.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "style": "classical",
                "piece_id": "1",
                "title": "Example",
                "creator": "Composer",
                "primary_evidence": "composer",
                "bpm": "120",
                "num_measures": "64",
                "melody_density_per_beat": "1.0",
                "accompaniment_density_per_beat": "2.0",
            }
        )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"classical": ["1"]}), encoding="utf-8")
    npz_root = tmp_path / "npz"
    npz_root.mkdir()
    (npz_root / "1.npz").write_bytes(b"test-npz")
    output_root = tmp_path / "prepared"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_task3_inputs.py",
            "--shortlist",
            str(shortlist),
            "--selection",
            str(selection),
            "--npz-root",
            str(npz_root),
            "--output-root",
            str(output_root),
        ],
    )

    main()

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["piece_id"] == "1"
    assert (output_root / "input_npz" / "classical__1.npz").read_bytes() == b"test-npz"
    assert json.loads((output_root / "indices.json").read_text()) == [0]


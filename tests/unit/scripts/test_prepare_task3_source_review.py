import json

from experiments.lekai_cross_style.prepare_source_review import main


def test_prepare_source_review_maps_frozen_piece_to_gt(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "style": "classical",
                    "piece_id": "1",
                    "title": "Example",
                    "creator": "Composer",
                    "primary_evidence": "composer",
                }
            ]
        ),
        encoding="utf-8",
    )
    source_root = tmp_path / "source"
    source_midi = source_root / "classical" / "1__source_gt.mid"
    source_midi.parent.mkdir(parents=True)
    source_midi.write_bytes(b"midi")
    output_csv = tmp_path / "source_review.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_source_review.py",
            "--prepared-manifest",
            str(manifest),
            "--source-gt-root",
            str(source_root),
            "--output-csv",
            str(output_csv),
        ],
    )

    assert main() == 0
    text = output_csv.read_text(encoding="utf-8")
    assert "classical" in text
    assert "metadata_evidence" in text
    assert "style_label_ok" not in text
    assert "1__source_gt.mid" in text

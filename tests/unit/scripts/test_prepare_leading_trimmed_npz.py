from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from experiments.lekai_failcase_dataset_v2.prepare_leading_trimmed_npz import (
    load_npz_roll,
    prepare_record,
    prepare_selection,
)


def _write_npz(path, full_roll, *, widths=None, metadata_extra=None):
    if widths is None:
        widths = [16] * (full_roll.shape[2] // 16)
        if full_roll.shape[2] % 16:
            widths.append(full_roll.shape[2] % 16)
    assert sum(widths) == full_roll.shape[2]
    metadata = {
        "time_signature": "4/4",
        "resolution": 16,
        "num_measures": len(widths),
        "total_length": int(full_roll.shape[2]),
        "valid_measures": [2, 7],
    }
    metadata.update(metadata_extra or {})
    payload = {"metadata": metadata}
    start = 0
    for index, width in enumerate(widths):
        payload[f"measure_{index}"] = full_roll[:, :, start : start + width]
        start += width
    np.savez(path, **payload)


def _record(piece_id, source):
    return {
        "order": 1,
        "style": "pop_contemporary",
        "id": str(piece_id),
        "title": "Synthetic",
        "npz": {"path": str(source)},
    }


def _load_output(path):
    metadata, full_roll, _ = load_npz_roll(path)
    return metadata, full_roll


def test_melody_onset_at_step_4_outputs_exact_source_suffix(tmp_path):
    source_roll = np.zeros((4, 88, 32), dtype=np.uint8)
    source_roll[1, 39, 4] = 1
    source_roll[0, 39, 4:8] = 1
    source_roll[3, 30, 18] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)

    row = prepare_record(_record("101", source), output_root=tmp_path / "output")
    metadata, output_roll = _load_output(tmp_path / "output" / "input_npz" / "101.npz")

    assert row["offset_steps"] == 4
    assert row["exact_suffix_verified"] is True
    assert np.array_equal(output_roll, source_roll[:, :, 4:])
    assert metadata["valid_measures"] == [2, 7]


def test_accompaniment_before_melody_offset_is_discarded(tmp_path):
    source_roll = np.zeros((4, 88, 16), dtype=np.uint8)
    source_roll[3, 12, 2] = 9
    source_roll[2, 12, 2:4] = 9
    source_roll[1, 40, 4] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)

    prepare_record(_record("102", source), output_root=tmp_path / "output")
    _, output_roll = _load_output(tmp_path / "output" / "input_npz" / "102.npz")

    assert np.array_equal(output_roll, source_roll[:, :, 4:])
    assert 9 not in output_roll


def test_internal_and_trailing_rests_are_preserved_elementwise(tmp_path):
    source_roll = np.zeros((4, 88, 32), dtype=np.uint8)
    source_roll[1, 35, 3] = 1
    source_roll[0, 35, 3:6] = 1
    source_roll[3, 20, 15] = 1
    source_roll[2, 20, 15:18] = 1
    source_roll[1, 42, 23] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)

    prepare_record(_record("103", source), output_root=tmp_path / "output")
    _, output_roll = _load_output(tmp_path / "output" / "input_npz" / "103.npz")

    expected = source_roll[:, :, 3:]
    assert np.array_equal(output_roll, expected)
    assert np.count_nonzero(output_roll[:, :, 3:12]) == 0
    assert np.count_nonzero(output_roll[:, :, -8:]) == 0


def test_final_measure_is_partial_and_never_padded(tmp_path):
    source_roll = np.zeros((4, 88, 32), dtype=np.uint8)
    source_roll[1, 30, 4] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)

    prepare_record(_record("104", source), output_root=tmp_path / "output")
    output = tmp_path / "output" / "input_npz" / "104.npz"
    with np.load(output, allow_pickle=True) as archive:
        assert archive["measure_0"].shape == (4, 88, 16)
        assert archive["measure_1"].shape == (4, 88, 12)
        assert "measure_2" not in archive.files
        assert archive["metadata"].item()["total_length"] == 28


@pytest.mark.parametrize("offset_steps", [3, 8])
def test_pickup_partial_first_measure_preserves_exact_flattened_suffix(
    tmp_path, offset_steps
):
    source_roll = np.zeros((4, 88, 40), dtype=np.uint8)
    source_roll[3, 18, 1] = 1
    source_roll[1, 45, offset_steps] = 1
    source_roll[0, 45, offset_steps : offset_steps + 3] = 1
    source_roll[3, 24, 25] = 1
    source = tmp_path / f"pickup_{offset_steps}.npz"
    _write_npz(source, source_roll, widths=[8, 16, 16])

    prepare_record(
        _record(f"pickup{offset_steps}", source),
        output_root=tmp_path / f"output_{offset_steps}",
    )
    output = (
        tmp_path
        / f"output_{offset_steps}"
        / "input_npz"
        / f"pickup{offset_steps}.npz"
    )
    _, output_roll = _load_output(output)

    assert np.array_equal(output_roll, source_roll[:, :, offset_steps:])


def test_offset_zero_is_a_byte_exact_copy(tmp_path):
    source_roll = np.zeros((4, 88, 16), dtype=np.uint8)
    source_roll[1, 44, 0] = 1
    source_roll[0, 44, 0:3] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)
    source_bytes = source.read_bytes()

    row = prepare_record(_record("105", source), output_root=tmp_path / "output")
    output = tmp_path / "output" / "input_npz" / "105.npz"

    assert row["offset_steps"] == 0
    assert output.read_bytes() == source_bytes
    assert row["source_sha256"] == row["output_sha256"]


def test_missing_melody_onset_raises(tmp_path):
    source_roll = np.zeros((4, 88, 16), dtype=np.uint8)
    source_roll[3, 20, 2] = 1
    source = tmp_path / "source.npz"
    _write_npz(source, source_roll)

    with pytest.raises(ValueError, match="no Melody onset"):
        prepare_record(_record("106", source), output_root=tmp_path / "output")


def test_output_uses_id_filename_and_writes_manifests(tmp_path):
    npz_root = tmp_path / "npz_root"
    npz_root.mkdir()
    source_roll = np.zeros((4, 88, 16), dtype=np.uint8)
    source_roll[1, 50, 2] = 1
    _write_npz(npz_root / "987654.npz", source_roll)
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "order": 7,
                "style": "anime",
                "id": "987654",
                "title": "Example",
                "npz": {"path": "/wrong/remote/path.npz"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "dataset"

    rows = prepare_selection(selection, output_root=output_root, npz_root=npz_root)

    assert (output_root / "input_npz" / "987654.npz").is_file()
    assert not (output_root / "input_npz" / "987654_0.npz").exists()
    json_rows = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    with (output_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "987654"
    assert json_rows[0]["exact_suffix_verified"] is True
    assert csv_rows[0]["source_sha256"]
    assert csv_rows[0]["output_sha256"]

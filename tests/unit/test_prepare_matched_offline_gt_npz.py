from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import mido
import numpy as np
import pytest


@pytest.fixture(scope="module")
def script() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    name = "_test_prepare_matched_offline_gt_npz"
    spec = importlib.util.spec_from_file_location(
        name, root / "scripts" / "prepare_matched_offline_gt_npz.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prepare_matched_offline_gt_npz.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _track(
    name: str,
    notes: list[tuple[int, int, int]],
    *,
    ticks_per_beat: int = 480,
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    include_timing: bool,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if include_timing:
        track.append(
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
        )
        track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=time_signature[0],
                denominator=time_signature[1],
                time=0,
            )
        )
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for onset_step, offset_step, pitch in notes:
        onset = onset_step * ticks_per_beat // 4
        offset = offset_step * ticks_per_beat // 4
        events.append(
            (
                onset,
                1,
                mido.Message("note_on", note=pitch, velocity=80, time=0),
            )
        )
        events.append(
            (
                offset,
                0,
                mido.Message("note_off", note=pitch, velocity=0, time=0),
            )
        )
    previous = 0
    for tick, _priority, message in sorted(events, key=lambda item: item[:2]):
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def _write_midi(
    path: Path,
    tracks: list[tuple[str, list[tuple[int, int, int]]]],
    *,
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    for index, (name, notes) in enumerate(tracks):
        midi.tracks.append(
            _track(
                name,
                notes,
                bpm=bpm,
                time_signature=time_signature,
                include_timing=index == 0,
            )
        )
    midi.save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_melody(notes: list[tuple[int, int, int]]) -> str:
    payload = {
        "bpm": 120,
        "notes": [
            [onset, offset, pitch, 80]
            for onset, offset, pitch in sorted(
                notes, key=lambda value: (value[0], value[2], value[1])
            )
        ],
        "schema_version": 1,
        "steps_per_beat": 4,
        "track_name": "Melody",
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest(
    root: Path,
    *,
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    gt_tracks: list[tuple[str, list[tuple[int, int, int]]]] | None = None,
) -> tuple[Path, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    root.mkdir(parents=True, exist_ok=True)
    piece = root / "01_piece-a"
    piece.mkdir()
    melody_notes = [(0, 4, 60), (32, 36, 62), (124, 128, 64)]
    accompaniment_notes = [
        (2, 8, 48),
        (32, 36, 50),
        (126, 132, 52),
        (128, 132, 55),
    ]
    melody = piece / "melody_120bpm.mid"
    gt = piece / "gt_120bpm.mid"
    _write_midi(
        melody,
        [("Melody", melody_notes)],
        bpm=bpm,
        time_signature=time_signature,
    )
    _write_midi(
        gt,
        gt_tracks
        if gt_tracks is not None
        else [("Melody", melody_notes), ("Accompaniment", accompaniment_notes)],
        bpm=bpm,
        time_signature=time_signature,
    )
    old_npz = piece / "old_source.npz"
    old_npz.write_bytes(b"old-source")
    manifest = root / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_contract": "test-contract",
                "samples": [
                    {
                        "order": 1,
                        "piece_id": "piece-a",
                        "source_npz": "01_piece-a/old_source.npz",
                        "source_npz_sha256": _sha256(old_npz),
                        "melody_midi": "01_piece-a/melody_120bpm.mid",
                        "melody_midi_sha256": _sha256(melody),
                        "gt_midi": "01_piece-a/gt_120bpm.mid",
                        "gt_midi_sha256": _sha256(gt),
                        "canonical_melody_input_sha256": _canonical_melody(
                            melody_notes
                        ),
                        "identity_marker": "must-survive",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, melody_notes, accompaniment_notes


def _run(script: ModuleType, manifest: Path, output: Path) -> dict:
    return script.prepare_matched_offline_gt_npz(
        cohort_manifest=manifest, output_dir=output
    )


def test_named_tracks_boundary_clipping_metadata_and_exact_gates(
    script: ModuleType, tmp_path: Path
) -> None:
    manifest, _melody, _accompaniment = _manifest(tmp_path / "source")
    output = tmp_path / "derived"

    result = _run(script, manifest, output)

    row = result["samples"][0]
    assert row["identity_marker"] == "must-survive"
    assert row["melody_midi"] == "01_piece-a/melody_120bpm.mid"
    assert row["gt_midi"] == "01_piece-a/gt_120bpm.mid"
    assert (output / row["melody_midi"]).read_bytes() == (
        manifest.parent / row["melody_midi"]
    ).read_bytes()
    npz = output / row["source_npz"]
    assert npz.name == "source.npz"
    assert row["source_npz_sha256"] == script.file_sha256(npz)
    assert row["source_npz_sha256"] != json.loads(manifest.read_text())["samples"][
        0
    ]["source_npz_sha256"]

    with np.load(npz, allow_pickle=True) as payload:
        metadata = payload["metadata"].item()
        measures = [payload[f"measure_{index}"] for index in range(8)]
    assert metadata["time_signature"] == "4/4"
    assert metadata["time_signature_idx"] == 0
    assert metadata["bpm"] == 120.0
    assert metadata["num_measures"] == 8
    assert metadata["num_channels"] == 4
    assert metadata["resolution"] == 16
    assert metadata["total_length"] == 128
    assert metadata["is_continuation"] is False
    assert all(measure.shape == (4, 88, 16) for measure in measures)
    full = np.concatenate(measures, axis=2)
    assert full.shape == (4, 88, 128)

    # Named tracks remain separate. The accompaniment crossing step 128 is
    # clipped, while the onset exactly at step 128 is outside the window.
    assert full[1, 60 - 21, 0] == 1
    assert full[3, 48 - 21, 2] == 1
    assert full[3, 52 - 21, 126] == 1
    assert full[2, 52 - 21, 126:128].tolist() == [1, 1]
    assert int(full[3, 55 - 21].sum()) == 0
    assert int(full[1, 48 - 21].sum()) == 0

    audit_json = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    audit = audit_json["samples"][0]
    assert audit["roll_shape"] == "[4, 88, 128]"
    assert audit["measure_shape"] == "[4, 88, 16]"
    assert audit["melody_midi_npz_roll_exact"] is True
    assert audit["accompaniment_midi_npz_roll_exact"] is True
    assert audit["melody_tokenizer_roundtrip_exact"] is True
    assert audit["accompaniment_tokenizer_roundtrip_exact"] is True
    assert audit["melody_postjoin_geometry_exact"] is True
    assert audit["accompaniment_postjoin_geometry_exact"] is True
    assert audit["reserved_patch_token_collisions"] == 0
    with (output / "audit.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["source_npz_sha256"] == row["source_npz_sha256"]
    assert script.file_sha256(output / "cohort_manifest.json") in (
        output / "cohort_manifest.json.sha256"
    ).read_text(encoding="ascii")


def test_gt_physical_track_order_is_resolved_by_name(
    script: ModuleType, tmp_path: Path
) -> None:
    melody_notes = [(0, 4, 60), (32, 36, 62), (124, 128, 64)]
    accompaniment_notes = [
        (2, 8, 48),
        (32, 36, 50),
        (126, 132, 52),
        (128, 132, 55),
    ]
    manifest, _, _ = _manifest(
        tmp_path / "source-swapped",
        gt_tracks=[
            ("Accompaniment", accompaniment_notes),
            ("Melody", melody_notes),
        ],
    )
    output = tmp_path / "derived-swapped"

    result = _run(script, manifest, output)

    with np.load(
        output / result["samples"][0]["source_npz"], allow_pickle=True
    ) as payload:
        full = np.concatenate(
            [payload[f"measure_{index}"] for index in range(8)], axis=2
        )
    assert full[1, 60 - 21, 0] == 1
    assert full[3, 48 - 21, 2] == 1
    assert int(full[3, 60 - 21].sum()) == 0
    assert int(full[1, 48 - 21].sum()) == 0


def test_existing_nonempty_output_is_refused_without_changes(
    script: ModuleType, tmp_path: Path
) -> None:
    manifest, _, _ = _manifest(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _run(script, manifest, output)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("tempo", "120 BPM"),
        ("time_signature", "4/4"),
        ("tracks", "physical non-drum tracks"),
    ),
)
def test_malformed_timing_and_tracks_fail_atomically(
    script: ModuleType, tmp_path: Path, mutation: str, message: str
) -> None:
    kwargs = {}
    if mutation == "tempo":
        kwargs["bpm"] = 100.0
    elif mutation == "time_signature":
        kwargs["time_signature"] = (3, 4)
    else:
        melody = [(0, 4, 60), (32, 36, 62)]
        accompaniment = [(2, 8, 48), (32, 36, 50)]
        kwargs["gt_tracks"] = [
            ("Melody", melody),
            ("Piano", accompaniment),
        ]
    manifest, _, _ = _manifest(tmp_path / f"source-{mutation}", **kwargs)
    output = tmp_path / f"output-{mutation}"

    with pytest.raises(ValueError, match=message):
        _run(script, manifest, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.staging-*"))


def test_hash_mismatch_and_nonzero_trim_anchor_are_rejected(
    script: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source-hash"
    manifest, _, _ = _manifest(source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["gt_midi_sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _run(script, manifest, tmp_path / "hash-output")

    source = tmp_path / "source-anchor"
    manifest, _, accompaniment = _manifest(source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    melody = source / payload["samples"][0]["melody_midi"]
    gt = source / payload["samples"][0]["gt_midi"]
    shifted = [(4, 8, 60), (32, 36, 62)]
    _write_midi(melody, [("Melody", shifted)])
    _write_midi(gt, [("Melody", shifted), ("Accompaniment", accompaniment)])
    payload["samples"][0]["melody_midi_sha256"] = _sha256(melody)
    payload["samples"][0]["gt_midi_sha256"] = _sha256(gt)
    payload["samples"][0]["canonical_melody_input_sha256"] = _canonical_melody(
        shifted
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="start at tick 0"):
        _run(script, manifest, tmp_path / "anchor-output")


def test_output_is_deterministic(script: ModuleType, tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"

    _run(script, manifest, first)
    _run(script, manifest, second)

    first_files = {
        path.relative_to(first).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_adjacent_same_pitch_retrigger_fails_exact_geometry_gate(
    script: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source-retrigger"
    manifest, _, accompaniment = _manifest(source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    melody = source / payload["samples"][0]["melody_midi"]
    gt = source / payload["samples"][0]["gt_midi"]
    melody_notes = [(0, 4, 60), (116, 117, 87), (117, 118, 87)]
    _write_midi(melody, [("Melody", melody_notes)])
    _write_midi(
        gt,
        [("Melody", melody_notes), ("Accompaniment", accompaniment)],
    )
    payload["samples"][0]["melody_midi_sha256"] = _sha256(melody)
    payload["samples"][0]["gt_midi_sha256"] = _sha256(gt)
    payload["samples"][0]["canonical_melody_input_sha256"] = _canonical_melody(
        melody_notes
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "retrigger-output"

    with pytest.raises(ValueError, match="not exactly representable"):
        _run(script, manifest, output)

    assert not output.exists()


def test_reserved_patch_token_collision_is_rejected(script: ModuleType) -> None:
    measures = [np.zeros((4, 88, 16), dtype=np.uint8) for _ in range(8)]
    # Sustain-only pattern 0,1,1,1 maps to reserved patch token 13.
    measures[0][0, 39, 1:4] = 1
    metadata = {
        "time_signature": "4/4",
        "time_signature_idx": 0,
        "bpm": 120.0,
        "num_measures": 8,
        "num_channels": 4,
        "resolution": 16,
        "total_length": 128,
        "is_continuation": False,
    }

    with pytest.raises(ValueError, match="reserved patch-token collision"):
        script._tokenizer_roundtrip(measures, metadata)

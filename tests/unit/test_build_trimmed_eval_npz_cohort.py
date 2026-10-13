from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import mido
import numpy as np
import pytest


@pytest.fixture
def builder() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    name = "_streammuse_test_build_trimmed_eval_npz_cohort"
    spec = importlib.util.spec_from_file_location(
        name, root / "scripts" / "build_trimmed_eval_npz_cohort.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _write_midi(path: Path, tracks: dict[str, list[tuple[int, int, int]]]) -> None:
    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat // 4
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(meta)
    for name, notes in tracks.items():
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        timeline = []
        for index, (pitch, onset, offset) in enumerate(notes):
            timeline.append((onset, 1, pitch, index, "on"))
            timeline.append((offset, 0, pitch, index, "off"))
        previous = 0
        for step, _priority, pitch, _index, kind in sorted(timeline):
            absolute = step * ticks_per_step
            if kind == "on":
                message = mido.Message("note_on", note=pitch, velocity=80, time=0)
            else:
                message = mido.Message("note_off", note=pitch, velocity=0, time=0)
            message.time = absolute - previous
            previous = absolute
            track.append(message)
        midi.tracks.append(track)
    midi.save(path)


def _write_npz(path: Path, roll: np.ndarray, *, valid_steps: int) -> None:
    assert roll.shape[2] % 16 == 0
    num_measures = roll.shape[2] // 16
    metadata = {
        "time_signature": "4/4",
        "time_signature_idx": 0,
        "key_signature_idx": 0,
        "key_signature": "C major",
        "bpm": 120.0,
        "tempo_text": "moderato",
        "num_measures": num_measures,
        "resolution": 16,
        "total_length": valid_steps,
        "num_parts": 2,
        "num_channels": 4,
        "original_measures": num_measures,
        "valid_measures": list(range((valid_steps + 15) // 16)),
    }
    payload = {"metadata": metadata}
    for index in range(num_measures):
        payload[f"measure_{index}"] = roll[:, :, index * 16 : (index + 1) * 16]
    np.savez_compressed(path, **payload)


def _roll(
    valid_steps: int,
    melody: list[tuple[int, int, int]],
    accompaniment: list[tuple[int, int, int]],
) -> np.ndarray:
    serialized = ((valid_steps + 15) // 16) * 16
    roll = np.zeros((4, 88, serialized), dtype=np.uint8)
    for sustain, onset, notes in ((0, 1, melody), (2, 3, accompaniment)):
        for pitch, start, end in notes:
            pitch_index = pitch - 21
            roll[sustain, pitch_index, start:end] = 1
            roll[onset, pitch_index, start] = 1
    return roll


def _case(
    builder: ModuleType,
    root: Path,
    *,
    cutoff_steps: int,
    source_valid_steps: int = 20,
    source_melody: list[tuple[int, int, int]] | None = None,
    source_acc: list[tuple[int, int, int]] | None = None,
    gt_melody: list[tuple[int, int, int]] | None = None,
    gt_acc: list[tuple[int, int, int]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    root.mkdir()
    source_npz = root / "source.npz"
    melody_midi = root / "melody.mid"
    gt_midi = root / "gt.mid"
    source_melody = source_melody or [(60, cutoff_steps, cutoff_steps + 3)]
    source_acc = source_acc or [(48, cutoff_steps, cutoff_steps + 4)]
    gt_melody = gt_melody or [
        (pitch, start - cutoff_steps, end - cutoff_steps)
        for pitch, start, end in source_melody
        if end > cutoff_steps
    ]
    gt_acc = gt_acc or [
        (pitch, max(0, start - cutoff_steps), end - cutoff_steps)
        for pitch, start, end in source_acc
        if end > cutoff_steps
    ]
    source_roll = _roll(source_valid_steps, source_melody, source_acc)
    _write_npz(source_npz, source_roll, valid_steps=source_valid_steps)
    _write_midi(melody_midi, {"Melody": gt_melody})
    _write_midi(gt_midi, {"Melody": gt_melody, "Accompaniment": gt_acc})
    manifest = root / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "order": 1,
                        "piece_id": "piece-01",
                        "source_npz": source_npz.name,
                        "source_npz_sha256": builder.file_sha256(source_npz),
                        "melody_midi": melody_midi.name,
                        "melody_midi_sha256": builder.file_sha256(melody_midi),
                        "gt_midi": gt_midi.name,
                        "gt_midi_sha256": builder.file_sha256(gt_midi),
                        "leading_trim_beats": cutoff_steps / 4,
                        "num_measures": source_roll.shape[2] // 16,
                        "num_steps": source_valid_steps,
                        "midi_artifacts_shifted_only": True,
                    }
                ],
                "leading_rest_preprocessing": {"source_npz_shifted": False},
            }
        ),
        encoding="utf-8",
    )
    return manifest, source_npz, melody_midi, gt_midi


def _load_output(manifest: dict, output_dir: Path) -> tuple[dict, np.ndarray, Path]:
    row = manifest["samples"][0]
    path = (output_dir / row["source_npz"]).resolve()
    with np.load(path, allow_pickle=True) as archive:
        metadata = archive["metadata"].item()
        roll = np.concatenate(
            [archive[f"measure_{index}"] for index in range(metadata["num_measures"])],
            axis=2,
        )
    return metadata, roll, path


def test_zero_offset_writes_independent_npz_and_relative_midi_provenance(
    builder: ModuleType, tmp_path: Path
) -> None:
    manifest_path, source_npz, melody_midi, _gt_midi = _case(
        builder, tmp_path / "input", cutoff_steps=0, source_valid_steps=16
    )
    source_bytes = source_npz.read_bytes()
    output_dir = tmp_path / "output"

    result = builder.build_cohort(
        cohort_manifest=manifest_path, output_dir=output_dir
    )
    metadata, output_roll, output_npz = _load_output(result, output_dir)

    assert output_npz != source_npz.resolve()
    assert output_npz.is_file()
    assert source_npz.read_bytes() == source_bytes
    assert output_roll.shape == (4, 88, 16)
    assert metadata["trimmed_eval_npz_provenance"]["cutoff_steps"] == 0
    assert (output_dir / result["samples"][0]["melody_midi"]).resolve() == melody_midi
    assert (output_dir / "npz_trim_audit.csv").is_file()


def test_non_measure_aligned_offset_pads_only_tail(
    builder: ModuleType, tmp_path: Path
) -> None:
    manifest_path, *_ = _case(
        builder, tmp_path / "input", cutoff_steps=2, source_valid_steps=20
    )
    output_dir = tmp_path / "output"

    result = builder.build_cohort(
        cohort_manifest=manifest_path, output_dir=output_dir
    )
    metadata, roll, _path = _load_output(result, output_dir)

    assert metadata["total_length"] == 18
    assert metadata["num_measures"] == 2
    assert metadata["valid_measures"] == [0, 1]
    assert roll.shape[2] == 32
    assert not np.any(roll[:, :, 18:])
    assert result["samples"][0]["num_steps"] == 18


def test_cross_cutoff_sustain_rehydrates_onset_at_zero(
    builder: ModuleType, tmp_path: Path
) -> None:
    manifest_path, *_ = _case(
        builder,
        tmp_path / "input",
        cutoff_steps=4,
        source_valid_steps=20,
        source_acc=[(48, 2, 7)],
        gt_acc=[(48, 0, 3)],
    )
    output_dir = tmp_path / "output"

    result = builder.build_cohort(
        cohort_manifest=manifest_path, output_dir=output_dir
    )
    metadata, roll, _path = _load_output(result, output_dir)

    assert roll[2, 48 - 21, 0:3].tolist() == [1, 1, 1]
    assert roll[3, 48 - 21, 0] == 1
    assert metadata["trimmed_eval_npz_provenance"]["rehydrated_onset_count"] == 1


@pytest.mark.parametrize("failure", ["hash", "path"])
def test_bad_source_hash_or_path_fails_closed(
    builder: ModuleType, tmp_path: Path, failure: str
) -> None:
    manifest_path, *_ = _case(builder, tmp_path / "input", cutoff_steps=0)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if failure == "hash":
        payload["samples"][0]["source_npz_sha256"] = "0" * 64
        expected = "SHA-256 mismatch"
    else:
        payload["samples"][0]["source_npz"] = "missing.npz"
        expected = "not found"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        builder.build_cohort(
            cohort_manifest=manifest_path, output_dir=tmp_path / "output"
        )


def test_midi_npz_mismatch_fails_closed(builder: ModuleType, tmp_path: Path) -> None:
    manifest_path, _source, _melody, gt_midi = _case(
        builder, tmp_path / "input", cutoff_steps=2
    )
    _write_midi(
        gt_midi,
        {"Melody": [(61, 0, 3)], "Accompaniment": [(48, 0, 4)]},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["samples"][0]["gt_midi_sha256"] = builder.file_sha256(gt_midi)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="MIDI/NPZ mismatch"):
        builder.build_cohort(
            cohort_manifest=manifest_path, output_dir=tmp_path / "output"
        )


def test_nonempty_output_directory_is_refused(
    builder: ModuleType, tmp_path: Path
) -> None:
    manifest_path, *_ = _case(builder, tmp_path / "input", cutoff_steps=0)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        builder.build_cohort(cohort_manifest=manifest_path, output_dir=output)

    assert sentinel.read_text(encoding="utf-8") == "keep"

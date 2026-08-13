from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import mido
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


midi_to_npz = _load("midi_to_npz_under_test", "midi_to_npz.py")
perturb = _load("perturb_melody_for_npz_test", "perturb_melody.py")


def _write_pair(root: Path) -> tuple[Path, Path]:
    mel_dir = root / "source" / "mel"
    acc_dir = root / "source" / "acc"
    for directory, pitches in ((mel_dir, (60, 64)), (acc_dir, (48, 52))):
        directory.mkdir(parents=True)
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        meta.append(
            mido.MetaMessage(
                "time_signature", numerator=4, denominator=4,
                clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0
            )
        )
        meta.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(meta)
        track = mido.MidiTrack()
        track.append(mido.Message("program_change", channel=0, program=0, time=0))
        tick = 0
        for pitch in pitches:
            track.append(mido.Message("note_on", channel=0, note=pitch, velocity=80, time=0))
            track.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=480))
            tick += 480
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(track)
        midi.save(directory / "song.mid")
    return mel_dir, acc_dir


def _campaign(tmp_path: Path, name: str = "campaign") -> tuple[Path, Path]:
    mel_dir, acc_dir = _write_pair(tmp_path)
    root = tmp_path / name
    perturb.generate_campaign(
        mel_dir=mel_dir,
        acc_dir=acc_dir,
        output_root=root,
        expected_song_count=None,
    )
    return root, root / "input_manifest.json"


def _write_delayed_pair(root: Path) -> tuple[Path, Path]:
    mel_dir = root / "delayed" / "mel"
    acc_dir = root / "delayed" / "acc"
    mel_dir.mkdir(parents=True)
    acc_dir.mkdir(parents=True)

    melody = mido.MidiFile(type=1, ticks_per_beat=480)
    melody_track = mido.MidiTrack()
    melody_track.append(mido.Message("note_on", note=60, velocity=80, time=960))
    melody_track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    melody.tracks.append(melody_track)
    melody.save(mel_dir / "song.mid")

    accompaniment = mido.MidiFile(type=1, ticks_per_beat=480)
    accompaniment_track = mido.MidiTrack()
    accompaniment_track.append(mido.Message("note_on", note=48, velocity=80, time=0))
    accompaniment_track.append(mido.Message("note_off", note=48, velocity=0, time=480))
    accompaniment_track.append(mido.Message("note_on", note=52, velocity=80, time=480))
    accompaniment_track.append(mido.Message("note_off", note=52, velocity=0, time=480))
    accompaniment.tracks.append(accompaniment_track)
    accompaniment.save(acc_dir / "song.mid")
    return mel_dir, acc_dir


def test_strict_manifest_conversion_updates_hashes_and_passes_all_roll_gates(
    tmp_path: Path,
) -> None:
    root, manifest_path = _campaign(tmp_path)
    summary_path = root / "strict-summary.json"
    summary = midi_to_npz.convert_expected_manifest(
        manifest_path,
        summary_path=summary_path,
    )
    assert summary["status"] == "ok"
    assert summary["converted"] == summary["expected"] == 8
    assert summary["skipped"] == 0
    assert summary["exact_stem_set"] is True
    assert all(row["roll_gate"]["differing_cells"] == 0 for row in summary["results"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {entry["stem"] for entry in manifest["entries"]} == set(manifest["exact_stems"])
    assert all(len(entry["npz"]["sha256"]) == 64 for entry in manifest["entries"])
    assert len(list((root / "npz").glob("*.npz"))) == 8
    assert (root / "input_manifest.json.sha256").is_file()

    with pytest.raises(FileExistsError, match="fresh NPZ directory"):
        midi_to_npz.convert_expected_manifest(manifest_path)


def test_strict_conversion_rejects_extra_stem_before_writing_npz(tmp_path: Path) -> None:
    root, manifest_path = _campaign(tmp_path)
    source = next((root / "mel").glob("*.mid"))
    shutil.copyfile(source, root / "mel" / "orphan.mid")
    with pytest.raises(ValueError, match="stem set mismatch"):
        midi_to_npz.convert_expected_manifest(manifest_path)
    assert not list((root / "npz").glob("*.npz"))


def test_strict_conversion_rejects_input_hash_mismatch(tmp_path: Path) -> None:
    root, manifest_path = _campaign(tmp_path)
    accompaniment = next((root / "acc").glob("*.mid"))
    accompaniment.write_bytes(accompaniment.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        midi_to_npz.convert_expected_manifest(manifest_path)


def test_roll_gate_detects_corrupt_npz_cell(tmp_path: Path) -> None:
    mel_dir, acc_dir = _write_pair(tmp_path)
    npz_path = tmp_path / "song.npz"
    assert midi_to_npz.midi_pair_to_npz(
        str(mel_dir / "song.mid"),
        str(acc_dir / "song.mid"),
        str(npz_path),
    )
    with np.load(npz_path, allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["measure_0"] = payload["measure_0"].copy()
    payload["measure_0"][0, 60 - 21, 0] = 0
    np.savez(npz_path, **payload)
    with pytest.raises(AssertionError, match="roll mismatch"):
        midi_to_npz.verify_midi_npz_roll(mel_dir / "song.mid", npz_path)


def test_trim_leading_rest_uses_melody_offset_for_both_tracks(tmp_path: Path) -> None:
    mel_dir, acc_dir = _write_delayed_pair(tmp_path)
    npz_path = tmp_path / "trimmed.npz"

    assert midi_to_npz.midi_pair_to_npz(
        str(mel_dir / "song.mid"),
        str(acc_dir / "song.mid"),
        str(npz_path),
        trim_leading_rest=True,
    )

    with np.load(npz_path, allow_pickle=True) as archive:
        metadata = archive["metadata"].item()
        measures = [
            archive[f"measure_{index}"]
            for index in range(int(metadata["num_measures"]))
        ]
    roll = np.concatenate(measures, axis=2)
    assert metadata["leading_offset_ticks"] == 8
    assert metadata["trim_leading_rest"] is True
    assert np.flatnonzero(roll[1].any(axis=0)).tolist() == [0]
    assert np.flatnonzero(roll[3].any(axis=0)).tolist() == [0]


def test_npz_loader_rejects_missing_measure_key(tmp_path: Path) -> None:
    npz_path = tmp_path / "bad.npz"
    np.savez(
        npz_path,
        metadata={
            "time_signature_idx": 4,
            "bpm": 120,
            "num_measures": 2,
            "is_continuation": False,
        },
        measure_0=np.zeros((4, 88, 16), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="measure keys mismatch"):
        midi_to_npz.load_npz_melody_roll(npz_path)

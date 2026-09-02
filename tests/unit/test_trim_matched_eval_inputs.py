from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import mido
import pytest


def _load_script(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trim_script() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    name = "_streammuse_test_trim_matched_eval_inputs"
    module = _load_script(root / "scripts" / "trim_matched_eval_inputs.py", name)
    yield module
    sys.modules.pop(name, None)


def _note_track(
    name: str,
    notes: list[tuple[int, int, int]],
    *,
    include_timing: bool,
    velocity: int = 80,
) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if include_timing:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        track.append(
            mido.MetaMessage(
                "time_signature", numerator=4, denominator=4, time=0
            )
        )
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    events: list[tuple[int, int, mido.Message]] = []
    for onset, offset, pitch in notes:
        events.append(
            (
                onset,
                1,
                mido.Message(
                    "note_on", channel=0, note=pitch, velocity=velocity, time=0
                ),
            )
        )
        events.append(
            (
                offset,
                0,
                mido.Message(
                    "note_off", channel=0, note=pitch, velocity=0, time=0
                ),
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
    *,
    melody_notes: list[tuple[int, int, int]],
    accompaniment_notes: list[tuple[int, int, int]] | None = None,
    ticks_per_beat: int = 480,
    melody_velocity: int = 80,
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    midi.tracks.append(
        _note_track(
            "Melody",
            melody_notes,
            include_timing=True,
            velocity=melody_velocity,
        )
    )
    if accompaniment_notes is not None:
        midi.tracks.append(
            _note_track(
                "Accompaniment", accompaniment_notes, include_timing=False
            )
        )
    midi.save(path)


def _track_notes(path: Path, track_name: str) -> list[tuple[int, int, int]]:
    midi = mido.MidiFile(path)
    track = next(
        track
        for track in midi.tracks
        if any(
            message.type == "track_name" and message.name == track_name
            for message in track
        )
    )
    active: dict[tuple[int, int], list[int]] = {}
    notes: list[tuple[int, int, int]] = []
    absolute = 0
    for message in track:
        absolute += int(message.time)
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append(absolute)
        elif message.type in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            onset = active[key].pop(0)
            notes.append((onset, absolute, message.note))
    return sorted(notes)


def _builder_canonical_hash(
    notes: list[tuple[int, int, int]], *, ticks_per_beat: int = 480
) -> str:
    ticks_per_step = ticks_per_beat // 4
    canonical_notes = [
        [onset // ticks_per_step, offset // ticks_per_step, pitch, 80]
        for onset, offset, pitch in sorted(notes, key=lambda note: (note[0], note[2], note[1]))
    ]
    payload = {
        "bpm": 120,
        "notes": canonical_notes,
        "schema_version": 1,
        "steps_per_beat": 4,
        "track_name": "Melody",
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_manifest(
    script: ModuleType,
    root: Path,
    *,
    gt_ticks_per_beat: int = 480,
    melody_notes: list[tuple[int, int, int]] | None = None,
) -> Path:
    source_npz = root / "source.npz"
    source_npz.write_bytes(b"source-npz")
    melody = root / "melody.mid"
    gt = root / "gt.mid"
    melody_notes = melody_notes or [(960, 1440, 60), (1920, 2400, 62)]
    _write_midi(melody, melody_notes=melody_notes, melody_velocity=37)
    _write_midi(
        gt,
        melody_notes=melody_notes,
        accompaniment_notes=[
            (0, 480, 48),
            (480, 1440, 50),
            (1920, 2400, 52),
        ],
        ticks_per_beat=gt_ticks_per_beat,
        melody_velocity=37,
    )
    manifest = root / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "order": 1,
                        "piece_id": "piece-01",
                        "source_npz": source_npz.name,
                        "source_npz_sha256": script.file_sha256(source_npz),
                        "melody_midi": melody.name,
                        "melody_midi_sha256": script.file_sha256(melody),
                        "gt_midi": gt.name,
                        "gt_midi_sha256": script.file_sha256(gt),
                        "canonical_melody_input_sha256": _builder_canonical_hash(
                            melody_notes
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_prepare_trims_melody_and_gt_with_note_aware_cutoff(
    trim_script: ModuleType, tmp_path: Path
) -> None:
    manifest = _write_manifest(trim_script, tmp_path)
    output = tmp_path / "trimmed"

    result = trim_script.prepare_trimmed_cohort(
        cohort_manifest=manifest, output_dir=output
    )

    row = result["samples"][0]
    trimmed_melody = (output / row["melody_midi"]).resolve()
    trimmed_gt = (output / row["gt_midi"]).resolve()
    assert _track_notes(trimmed_melody, "Melody") == [
        (0, 480, 60),
        (960, 1440, 62),
    ]
    assert _track_notes(trimmed_gt, "Melody") == [
        (0, 480, 60),
        (960, 1440, 62),
    ]
    assert _track_notes(trimmed_gt, "Accompaniment") == [
        (0, 480, 50),
        (960, 1440, 52),
    ]
    assert row["leading_trim_ticks"] == 960
    assert row["leading_trim_beats"] == 2.0
    assert row["midi_artifacts_shifted_only"] is True
    preprocessing = result["leading_rest_preprocessing"]
    assert preprocessing["shifted_artifacts"] == ["melody_midi", "gt_midi"]
    assert preprocessing["source_npz_shifted"] is False
    assert trim_script.file_sha256(trimmed_gt) == row["gt_midi_sha256"]
    assert (output / row["source_npz"]).resolve() == (tmp_path / "source.npz")

    with (output / "trim_audit.csv").open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    assert audit[0]["gt_dropped_pre_cutoff_note_count"] == "1"
    assert audit[0]["gt_rehydrated_cross_cutoff_note_count"] == "1"
    assert audit[0]["first_melody_note_tick_after"] == "0"
    assert audit[0]["leading_trim_ticks"] == "960"
    assert audit[0]["midi_artifacts_shifted_only"] == "True"
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "offline NPZ control" in readme
    assert "intentionally not shifted" in readme

    root = Path(__file__).resolve().parents[2]
    runner_name = "_streammuse_test_trimmed_manifest_runner"
    runner = _load_script(root / "scripts" / "run_matched_system_eval.py", runner_name)
    try:
        pieces = runner.load_cohort_manifest(output / "cohort_manifest.json")
    finally:
        sys.modules.pop(runner_name, None)
    assert pieces[0].midi_path == trimmed_melody
    assert pieces[0].melody_input_sha256 == row["melody_midi_sha256"]


def test_canonical_hash_matches_builder_and_ignores_actual_midi_velocity(
    trim_script: ModuleType,
) -> None:
    notes = (
        trim_script.MidiNote(0, 60, 480, 37, 0),
        trim_script.MidiNote(480, 64, 960, 101, 0),
    )

    assert trim_script.canonical_melody_sha256(
        notes, ticks_per_beat=480
    ) == _builder_canonical_hash([(0, 480, 60), (480, 960, 64)])


def test_zero_offset_midi_outputs_are_byte_preserving(
    trim_script: ModuleType, tmp_path: Path
) -> None:
    manifest = _write_manifest(
        trim_script,
        tmp_path,
        melody_notes=[(0, 480, 60), (960, 1440, 62)],
    )
    source = json.loads(manifest.read_text(encoding="utf-8"))["samples"][0]
    output = tmp_path / "zero-output"
    output.mkdir()

    result = trim_script.prepare_trimmed_cohort(
        cohort_manifest=manifest, output_dir=output
    )

    row = result["samples"][0]
    assert row["leading_trim_ticks"] == 0
    assert row["melody_midi_sha256"] == source["melody_midi_sha256"]
    assert row["gt_midi_sha256"] == source["gt_midi_sha256"]


def test_empty_melody_is_rejected(trim_script: ModuleType, tmp_path: Path) -> None:
    manifest = _write_manifest(trim_script, tmp_path)
    empty = mido.MidiFile(type=1, ticks_per_beat=480)
    empty.tracks.append(_note_track("Melody", [], include_timing=True))
    melody_path = tmp_path / "melody.mid"
    empty.save(melody_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["melody_midi_sha256"] = trim_script.file_sha256(
        melody_path
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no Melody note-on"):
        trim_script.prepare_trimmed_cohort(
            cohort_manifest=manifest, output_dir=tmp_path / "empty-output"
        )


def test_trim_preserves_midi_timing_and_track_metadata(
    trim_script: ModuleType, tmp_path: Path
) -> None:
    manifest = _write_manifest(trim_script, tmp_path)
    output = tmp_path / "metadata-output"
    result = trim_script.prepare_trimmed_cohort(
        cohort_manifest=manifest, output_dir=output
    )
    row = result["samples"][0]
    source = mido.MidiFile(tmp_path / "gt.mid")
    trimmed = mido.MidiFile(output / row["gt_midi"])

    assert trimmed.ticks_per_beat == source.ticks_per_beat
    assert [message.tempo for track in trimmed.tracks for message in track if message.type == "set_tempo"] == [
        message.tempo
        for track in source.tracks
        for message in track
        if message.type == "set_tempo"
    ]
    assert [
        (message.numerator, message.denominator)
        for track in trimmed.tracks
        for message in track
        if message.type == "time_signature"
    ] == [
        (message.numerator, message.denominator)
        for track in source.tracks
        for message in track
        if message.type == "time_signature"
    ]
    assert [
        message.name
        for track in trimmed.tracks
        for message in track
        if message.type == "track_name"
    ] == ["Melody", "Accompaniment"]


def test_prepare_rejects_gt_with_different_ticks_per_beat(
    trim_script: ModuleType, tmp_path: Path
) -> None:
    manifest = _write_manifest(trim_script, tmp_path, gt_ticks_per_beat=960)

    with pytest.raises(ValueError, match="ticks_per_beat differs"):
        trim_script.prepare_trimmed_cohort(
            cohort_manifest=manifest, output_dir=tmp_path / "output"
        )


def test_prepare_rejects_hash_mismatch_and_nonempty_output(
    trim_script: ModuleType, tmp_path: Path
) -> None:
    manifest = _write_manifest(trim_script, tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["samples"][0]["melody_midi_sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        trim_script.prepare_trimmed_cohort(
            cohort_manifest=manifest, output_dir=tmp_path / "hash-output"
        )

    manifest = _write_manifest(trim_script, tmp_path)
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        trim_script.prepare_trimmed_cohort(
            cohort_manifest=manifest, output_dir=output
        )

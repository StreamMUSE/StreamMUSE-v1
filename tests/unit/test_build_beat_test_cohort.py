from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import mido
import numpy as np
import pytest

from scripts import build_beat_test_cohort as cohort


def _roll(*, steps: int = 128, time_signature_idx=0) -> tuple[dict, list[np.ndarray]]:
    assert steps % 16 == 0
    full = np.zeros((4, 88, steps), dtype=np.uint8)
    for channel, pitch, onset in (
        (1, 39, 0),
        (3, 27, 4),
        (1, 41, 40),
        (3, 29, 44),
    ):
        full[channel, pitch, onset] = 1
        full[channel - 1, pitch, onset : onset + 3] = 1
    measures = [full[:, :, start : start + 16] for start in range(0, steps, 16)]
    metadata = {
        "num_channels": 4,
        "num_measures": len(measures),
        "time_signature_idx": time_signature_idx,
    }
    return metadata, measures


def _write_npz(path: Path, *, steps: int = 128, time_signature_idx=0) -> None:
    metadata, measures = _roll(steps=steps, time_signature_idx=time_signature_idx)
    payload = {f"measure_{index}": measure for index, measure in enumerate(measures)}
    payload["metadata"] = metadata
    np.savez_compressed(path, **payload)


def _test_payload(entries: list[str], *, seed: int, ratio: float) -> str:
    indices = np.arange(len(entries))
    np.random.RandomState(seed).shuffle(indices)
    test_size = int(len(entries) * ratio)
    return "\n".join(entries[int(index)] for index in indices[-test_size:])


def test_legacy_split_matches_training_rng_and_hashes_order() -> None:
    entries = [f"{index:03d}.npz" for index in range(20)]
    payload = _test_payload(entries, seed=42, ratio=0.25)
    contract = cohort.SplitContract(
        total_files=20,
        test_files=5,
        split_seed=42,
        test_ratio=0.25,
        test_list_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )

    selected, actual_payload = cohort.legacy_test_entries(entries, contract)

    assert actual_payload == payload
    assert selected == payload.splitlines()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("time_signature", "time_signature_not_all_4_4_idx0"),
        ("short", "total_steps_lt_128"),
        ("missing_late_melody", "mel_onsets_32_128_empty"),
        ("missing_early_acc", "acc_onsets_0_32_empty"),
    ],
)
def test_candidate_eligibility_reports_frozen_exclusion_reasons(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    path = tmp_path / "candidate.npz"
    metadata, measures = _roll(
        steps=112 if mutation == "short" else 128,
        time_signature_idx=[0, 1] if mutation == "time_signature" else 0,
    )
    full = np.concatenate(measures, axis=2)
    if mutation == "missing_late_melody":
        full[1, :, 32:128] = 0
    if mutation == "missing_early_acc":
        full[3, :, 0:32] = 0
    measures = [full[:, :, start : start + 16] for start in range(0, full.shape[2], 16)]
    metadata["num_measures"] = len(measures)
    np.savez_compressed(
        path,
        metadata=metadata,
        **{f"measure_{index}": value for index, value in enumerate(measures)},
    )

    result = cohort.inspect_candidate(path)

    assert result.eligible is False
    assert reason in result.reasons


def test_midi_export_is_120_bpm_named_and_note_hash_is_event_based(tmp_path: Path) -> None:
    _metadata, measures = _roll()
    full = np.concatenate(measures, axis=2)
    melody = cohort.pianoroll_note_events(full[:2])
    accompaniment = cohort.pianoroll_note_events(full[2:])
    melody_path = tmp_path / "melody.mid"
    gt_path = tmp_path / "gt.mid"

    cohort.write_midi(melody_path, [("Melody", melody)])
    cohort.write_midi(gt_path, [("Melody", melody), ("Accompaniment", accompaniment)])

    melody_midi = mido.MidiFile(melody_path)
    gt_midi = mido.MidiFile(gt_path)
    assert melody_midi.ticks_per_beat == 480
    assert [track.name for track in melody_midi.tracks] == ["Melody"]
    assert [track.name for track in gt_midi.tracks] == ["Melody", "Accompaniment"]
    tempos = [msg.tempo for track in gt_midi.tracks for msg in track if msg.type == "set_tempo"]
    assert tempos == [mido.bpm2tempo(120)]
    assert cohort.melody_input_sha256(melody) == cohort.melody_input_sha256(list(melody))
    changed = list(melody) + [cohort.NoteEvent(100, 60, 101)]
    assert cohort.melody_input_sha256(melody) != cohort.melody_input_sha256(changed)


def test_full_builder_writes_audit_artifacts_and_count_prefix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    entries = [f"{index:03d}.npz" for index in range(20)]
    payload = _test_payload(entries, seed=42, ratio=0.5)
    test_entries = payload.splitlines()
    selection_seed = 7
    permutation = np.random.RandomState(selection_seed).permutation(
        len(test_entries)
    )
    first_checked = test_entries[int(permutation[0])]
    for entry in entries:
        _write_npz(
            data_dir / entry,
            time_signature_idx=1 if entry == first_checked else 0,
        )
    with (data_dir / ".lengths_cache.pkl").open("wb") as handle:
        pickle.dump({"data_files": entries}, handle)
    contract = cohort.SplitContract(
        total_files=20,
        test_files=10,
        split_seed=42,
        test_ratio=0.5,
        test_list_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )
    output_dir = tmp_path / "cohort"

    manifest = cohort.build_cohort(
        data_dir=data_dir,
        output_dir=output_dir,
        count=2,
        selection_seed=selection_seed,
        contract=contract,
        expected_first_ids=(),
    )

    assert len(manifest["samples"]) == 2
    with (output_dir / "candidate_audit.csv").open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 3
    assert audit[0]["eligible"] == "False"
    assert "time_signature_not_all_4_4_idx0" in audit[0]["exclusion_reasons"]
    assert [row["selected_order"] for row in audit[1:]] == ["1", "2"]
    for row in manifest["samples"]:
        piece_dir = output_dir / f"{int(row['order']):02d}_{row['piece_id']}"
        assert (piece_dir / "source.npz").is_file()
        assert (piece_dir / "melody_120bpm.mid").is_file()
        assert (piece_dir / "gt_120bpm.mid").is_file()
        assert len(row["canonical_melody_input_sha256"]) == 64
    frozen = (output_dir / "test_split.txt").read_text(encoding="utf-8")
    assert hashlib.sha256(frozen.encode()).hexdigest() == contract.test_list_sha256
    assert cohort.sha256_file(output_dir / "test_split.txt") == contract.test_list_sha256
    assert json.loads((output_dir / "cohort_manifest.json").read_text())["split"][
        "test_files"
    ] == 10


def test_explicit_exclusion_is_audited_and_next_eligible_candidate_fills_count(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    entries = [f"{index:03d}.npz" for index in range(20)]
    payload = _test_payload(entries, seed=42, ratio=0.5)
    test_entries = payload.splitlines()
    selection_seed = 7
    permutation = np.random.RandomState(selection_seed).permutation(len(test_entries))
    selection_ids = [Path(test_entries[int(index)]).stem for index in permutation]
    for entry in entries:
        _write_npz(data_dir / entry)
    with (data_dir / ".lengths_cache.pkl").open("wb") as handle:
        pickle.dump({"data_files": entries}, handle)
    contract = cohort.SplitContract(
        total_files=20,
        test_files=10,
        split_seed=42,
        test_ratio=0.5,
        test_list_sha256=hashlib.sha256(payload.encode()).hexdigest(),
    )
    output_dir = tmp_path / "cohort"

    manifest = cohort.build_cohort(
        data_dir=data_dir,
        output_dir=output_dir,
        count=2,
        selection_seed=selection_seed,
        contract=contract,
        expected_first_ids=(),
        exclude_piece_ids=[selection_ids[0]],
    )

    assert [row["piece_id"] for row in manifest["samples"]] == selection_ids[1:3]
    assert len(manifest["samples"]) == 2
    with (output_dir / "candidate_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        audit = list(csv.DictReader(handle))
    assert audit[0]["piece_id"] == selection_ids[0]
    assert audit[0]["eligible"] == "False"
    assert audit[0]["selected_order"] == ""
    assert audit[0]["exclusion_reasons"] == cohort.EXPLICIT_EXCLUSION_REASON
    assert [row["selected_order"] for row in audit[1:]] == ["1", "2"]


def test_default_contract_freezes_existing_first_five() -> None:
    assert cohort.EXPECTED_FIRST_FIVE == (
        "5509144",
        "6123939",
        "5563373",
        "3810536",
        "5820238",
    )

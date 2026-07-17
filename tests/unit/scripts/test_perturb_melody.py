from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path

import mido
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "perturb_melody.py"
_SPEC = importlib.util.spec_from_file_location("perturb_melody_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
perturb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = perturb
_SPEC.loader.exec_module(perturb)


def _write_midi(path: Path, *, accompaniment: bool = False, off_grid: bool = False) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="meta", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature", numerator=4, denominator=4, clocks_per_click=24,
            notated_32nd_notes_per_beat=8, time=0
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    events: list[tuple[int, int, mido.Message]] = [
        (0, 0, mido.MetaMessage("track_name", name="melody", time=0)),
        (0, 0, mido.Message("program_change", channel=0, program=5, time=0)),
        (120, 0, mido.Message("control_change", channel=0, control=64, value=127, time=0)),
    ]
    if accompaniment:
        note_specs = [(48, 0, 1920, 0)]
    else:
        first_start = 60 if off_grid else 0
        note_specs = [
            (21, first_start, 480, 0),
            (108, 480, 960, 0),
            (60, 960, 1440, 0),
            (60, 1080, 1440, 0),
            (62, 1440, 1920, 0),
            (10, 0, 240, 0),
            (40, 0, 240, 9),
        ]
    for pitch, start, end, channel in note_specs:
        events.append(
            (start, 2, mido.Message("note_on", channel=channel, note=pitch, velocity=80, time=0))
        )
        events.append(
            (end, 1, mido.Message("note_off", channel=channel, note=pitch, velocity=0, time=0))
        )
    if not accompaniment and not off_grid:
        # Invisible under PrettyMIDI/MidiConverter: a duplicate off and a dangling on.
        events.append((1920, 1, mido.Message("note_on", channel=0, note=62, velocity=0, time=0)))
        events.append((2040, 2, mido.Message("note_on", channel=0, note=64, velocity=70, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    track = mido.MidiTrack()
    previous = 0
    for tick, _priority, message in events:
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=240))
    midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


@pytest.fixture
def source_pair(tmp_path: Path) -> tuple[Path, Path]:
    mel_dir = tmp_path / "source" / "mel"
    acc_dir = tmp_path / "source" / "acc"
    _write_midi(mel_dir / "song.mid")
    _write_midi(acc_dir / "song.mid", accompaniment=True)
    return mel_dir, acc_dir


def _sidecar(root: Path, condition: str, seed: int | None = None) -> dict:
    suffix = condition if seed is None else f"{condition}__ps{seed}"
    path = root / "sidecars" / f"song__{suffix}.perturbation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_parser_matches_visible_universe_and_grid_gate(
    source_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    mel_dir, _acc_dir = source_pair
    parsed = perturb.parse_canonical_midi(mel_dir / "song.mid")
    assert len(parsed.notes) == 5
    assert parsed.stats == {
        "matched_model_visible": 5,
        "matched_drum": 1,
        "matched_out_of_range": 1,
        "dangling_note_on": 1,
        "spurious_note_off": 2,
        "off_grid_model_visible": 0,
    }
    assert len({note.source_note_id for note in parsed.notes}) == 5
    assert all(21 <= note.state.pitch <= 108 for note in parsed.notes)
    assert parsed.dropped_note_event_indices

    off_grid = tmp_path / "off-grid.mid"
    _write_midi(off_grid, off_grid=True)
    with pytest.raises(ValueError, match="off the PPQ/4 grid"):
        perturb.parse_canonical_midi(off_grid)


def test_keyed_prf_is_fixed_and_dose_selection_is_nested(
    source_pair: tuple[Path, Path],
) -> None:
    parsed = perturb.parse_canonical_midi(source_pair[0] / "song.mid")
    decision = perturb.build_latent_decisions(parsed, 2026071001)[parsed.notes[0].source_note_id]
    assert decision.pitch_score == pytest.approx(0.981375954082783)
    assert decision.onset_score == pytest.approx(0.1861897767045663)
    assert set(decision.pitch_candidates).issubset({-2, -1, 1, 2})
    assert decision.onset_candidates in ((-1, 1), (1, -1))

    decisions = perturb.build_latent_decisions(parsed, 2026071001)
    pitch_medium = {
        note_id for note_id, item in decisions.items() if item.pitch_score < 0.05
    }
    pitch_high = {
        note_id for note_id, item in decisions.items() if item.pitch_score < 0.20
    }
    onset_medium = {
        note_id for note_id, item in decisions.items() if item.onset_score < 0.15
    }
    onset_high = {
        note_id for note_id, item in decisions.items() if item.onset_score < 0.40
    }
    assert pitch_medium <= pitch_high
    assert onset_medium <= onset_high


def test_p_equals_one_preserves_bounds_duration_and_serializable_roll(
    source_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    parsed = perturb.parse_canonical_midi(source_pair[0] / "song.mid")
    condition = perturb.ConditionSpec("fixture", 1.0, 1.0, (11,))
    states, records, stats = perturb.apply_condition(
        parsed,
        condition=condition,
        perturb_seed=11,
        decisions=perturb.build_latent_decisions(parsed, 11),
    )
    assert stats["counts"]["selected_pitch"] == len(parsed.notes)
    assert stats["counts"]["selected_onset"] == len(parsed.notes)
    for note, record in zip(parsed.notes, records):
        final = perturb.NoteState(**record["final"])
        assert 21 <= final.pitch <= 108
        assert final.start_tick >= 0
        assert final.duration_ticks == note.state.duration_ticks

    output = tmp_path / "fixture-output.mid"
    perturb.write_states_to_midi(parsed, states, output)
    reparsed = perturb.parse_canonical_midi(output)
    horizon = max(state.end_tick for state in states.values())
    expected = perturb.states_to_roll(states.values(), horizon=horizon)
    actual = perturb.states_to_roll(
        (note.state for note in reparsed.notes), horizon=horizon
    )
    assert np.array_equal(expected, actual)


def test_collision_helpers_distinguish_native_overlap_and_new_collision() -> None:
    original = {
        "a": perturb.NoteState(60, 0, 4),
        "b": perturb.NoteState(60, 2, 4),
        "c": perturb.NoteState(62, 4, 8),
    }
    assert perturb._new_same_pitch_collision(
        note_id="a", candidate=original["a"], states=original, original_states=original
    ) == []
    assert perturb._new_same_pitch_collision(
        note_id="c",
        candidate=perturb.NoteState(60, 1, 5),
        states=original,
        original_states=original,
    ) == ["a", "b"]


def test_campaign_is_reproducible_replayable_and_factorially_audited(
    source_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    mel_dir, acc_dir = source_pair
    first = tmp_path / "campaign-a"
    second = tmp_path / "campaign-b"
    manifest_a = perturb.generate_campaign(
        mel_dir=mel_dir,
        acc_dir=acc_dir,
        output_root=first,
        expected_song_count=None,
    )
    manifest_b = perturb.generate_campaign(
        mel_dir=mel_dir,
        acc_dir=acc_dir,
        output_root=second,
        expected_song_count=None,
    )
    assert manifest_a["input_count"] == manifest_b["input_count"] == 8
    assert manifest_a["exact_stems"] == manifest_b["exact_stems"]
    assert len(list((first / "mel").glob("*.mid"))) == 8
    assert len(list((first / "acc").glob("*.mid"))) == 8
    assert len(list((first / "sidecars").glob("*.json"))) == 8

    for entry_a, entry_b in zip(manifest_a["entries"], manifest_b["entries"]):
        assert entry_a["output_midi"]["sha256"] == entry_b["output_midi"]["sha256"]
        assert entry_a["sidecar"]["sha256"] == entry_b["sidecar"]["sha256"]
        source = (first / entry_a["source_midi"]["path"]).resolve()
        sidecar = first / entry_a["sidecar"]["path"]
        perturb.verify_sidecar(source, sidecar)
        assert entry_a["analysis_end_tick"] == 16
        assert entry_a["last_input_note_off_tick"] >= 16
        assert entry_a["validation_horizon_ticks"] >= entry_a["last_input_note_off_tick"]
        assert entry_a["validation_horizon_ticks"] % 16 == 0

    sham = _sidecar(first, "sham")
    source_parsed = perturb.parse_canonical_midi(mel_dir / "song.mid")
    sham_parsed = perturb.parse_canonical_midi(first / "mel" / "song__sham.mid")
    assert np.array_equal(
        perturb.states_to_roll(note.state for note in source_parsed.notes),
        perturb.states_to_roll(note.state for note in sham_parsed.notes),
    )
    assert sham["metadata_policy"]["dangling_note_on"].startswith("drop")

    for seed in perturb.DEFAULT_PERTURB_SEEDS:
        arms = {
            condition: _sidecar(first, condition, seed)
            for condition in ("pitch", "onset", "both")
        }
        assert all(arm["factorial_pairing"]["latent_pairing_verified"] for arm in arms.values())
        by_arm = {
            name: {record["source_note_id"]: record for record in sidecar["notes"]}
            for name, sidecar in arms.items()
        }
        for note_id in by_arm["both"]:
            assert (
                by_arm["pitch"][note_id]["selection"]["pitch_score"]
                == by_arm["both"][note_id]["selection"]["pitch_score"]
            )
            assert (
                by_arm["onset"][note_id]["selection"]["onset_score"]
                == by_arm["both"][note_id]["selection"]["onset_score"]
            )
            assert (
                by_arm["pitch"][note_id]["candidate_order"]["pitch_offsets"]
                == by_arm["both"][note_id]["candidate_order"]["pitch_offsets"]
            )
            assert (
                by_arm["onset"][note_id]["candidate_order"]["onset_deltas"]
                == by_arm["both"][note_id]["candidate_order"]["onset_deltas"]
            )

    medium_pitch = _sidecar(first, "pitch", perturb.DEFAULT_HIGH_PSEED)
    high = _sidecar(first, "high", perturb.DEFAULT_HIGH_PSEED)
    medium_selected = {
        record["source_note_id"]
        for record in medium_pitch["notes"]
        if record["selection"]["pitch_selected"]
    }
    high_selected = {
        record["source_note_id"]
        for record in high["notes"]
        if record["selection"]["pitch_selected"]
    }
    assert medium_selected <= high_selected
    assert "factorial_pairing" not in high


def test_campaign_rejects_nonfresh_staging(source_pair: tuple[Path, Path], tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not fresh"):
        perturb.generate_campaign(
            mel_dir=source_pair[0],
            acc_dir=source_pair[1],
            output_root=output,
            expected_song_count=None,
        )


def test_cross_pythonhashseed_reproducibility(
    source_pair: tuple[Path, Path], tmp_path: Path
) -> None:
    roots = [tmp_path / "hash-seed-1", tmp_path / "hash-seed-999"]
    for python_hash_seed, output in zip(("1", "999"), roots):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = python_hash_seed
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "generate",
                "--mel-dir",
                str(source_pair[0]),
                "--acc-dir",
                str(source_pair[1]),
                "--output-root",
                str(output),
                "--expected-song-count",
                "1",
            ],
            check=True,
            cwd=_SCRIPT.parent.parent,
            env=env,
            capture_output=True,
            text=True,
        )

    def hashes(root: Path) -> dict[str, str]:
        paths = sorted((root / "mel").glob("*.mid")) + sorted(
            (root / "sidecars").glob("*.json")
        )
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }

    assert hashes(roots[0]) == hashes(roots[1])

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import mido

from streammuse.experiments.melody_robustness import (
    build_run_schedule,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    write_canonical_json,
    write_jsonl,
)
from streammuse.experiments.robustness_metrics import write_roll_midi
from streammuse.experiments.triangle_listening import build_triangle_control_roll
from streammuse.experiments.triangle_midi import build_formal_triangle_excerpt


def _write_theoretical_midi(
    path: Path, notes: list[tuple[int, int, int, int]]
) -> None:
    """Write logical-tick notes to the formal theoretical track."""

    timeline: list[tuple[int, int, mido.Message]] = []
    for start, end, pitch, velocity in notes:
        timeline.append(
            (start * 120, 1, mido.Message("note_on", note=pitch, velocity=velocity))
        )
        timeline.append(
            (end * 120, 0, mido.Message("note_off", note=pitch, velocity=0))
        )
    timeline.sort(key=lambda row: (row[0], row[1], row[2].note))
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    midi.tracks.append(meta)
    track = mido.MidiTrack()
    track.append(
        mido.MetaMessage("track_name", name="Theoretical Accompaniment", time=0)
    )
    previous = 0
    for absolute, _priority, message in timeline:
        message.time = absolute - previous
        track.append(message)
        previous = absolute
    midi.tracks.append(track)
    midi.save(path)


def _write_inferences(path: Path) -> None:
    token_payload = {"raw": [1, 2], "structural": [3]}
    accompaniment = [{"tick": 0, "pitch": 60, "velocity": 37}]
    raw_digest = hashlib.sha256(
        json.dumps(
            token_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output_digest = hashlib.sha256(
        json.dumps(
            accompaniment, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(
            [
                {
                    "response_data": {
                        "response_metadata": {
                            "raw_tokens": token_payload["raw"],
                            "structural_tokens": token_payload["structural"],
                            "raw_token_digest": raw_digest,
                            "output_event_digest": output_digest,
                        },
                        "accompaniment": accompaniment,
                    }
                }
            ],
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_formal_excerpt_preserves_velocity_and_reconstructs_active_notes(
    tmp_path: Path, load_script
) -> None:
    prepare = load_script("prepare_robustness_listening")
    source = tmp_path / "theoretical_model.mid"
    excerpt = tmp_path / "excerpt.mid"
    _write_theoretical_midi(
        source,
        [
            (2, 10, 60, 37),
            (8, 12, 64, 96),
        ],
    )

    roll, events = prepare._rebuild_formal_excerpt_midi(
        source,
        excerpt,
        start_model_tick=4,
        end_model_tick=12,
    )

    assert events == [
        {
            "start_model_tick": 0,
            "end_model_tick": 6,
            "pitch": 60,
            "velocity": 37,
        },
        {
            "start_model_tick": 4,
            "end_model_tick": 8,
            "pitch": 64,
            "velocity": 96,
        },
    ]
    assert (0, 60) in roll.onsets
    assert (5, 60) in roll.sustain
    assert prepare._canonical_midi_contract(
        excerpt, expected_end_model_tick=8
    ) == events


def test_objective_identity_includes_final_pair_wav_bytes(
    tmp_path: Path, load_script
) -> None:
    prepare = load_script("prepare_robustness_listening")
    events_a = [
        {
            "start_model_tick": 0,
            "end_model_tick": 4,
            "pitch": 60,
            "velocity": 80,
        }
    ]
    events_b = [
        {
            "start_model_tick": 0,
            "end_model_tick": 4,
            "pitch": 61,
            "velocity": 80,
        }
    ]
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.write_bytes(b"same-final-pcm-container")
    wav_b.write_bytes(b"same-final-pcm-container")

    assert prepare._triangle_objective_identity(events_a, events_a)
    assert prepare._triangle_objective_identity(
        events_a, events_b, wav_a=wav_a, wav_b=wav_b
    )
    wav_b.write_bytes(b"different-final-pcm-container")
    assert not prepare._triangle_objective_identity(
        events_a, events_b, wav_a=wav_a, wav_b=wav_b
    )

def test_retry_control_binding_resolves_c5_base_not_retry_question_ids(
    robustness_fixture, load_script
) -> None:
    prepare = load_script("prepare_robustness_listening")
    base_path = robustness_fixture.listening_manifest.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    retry = copy.deepcopy(base)
    retry["listening_attempt_id"] = "listening-attempt-002"
    retry["listening_attempt_number"] = 2
    retry["retry_lineage"] = {
        "base_selection_path": str(base_path),
        "base_selection_sha256": file_sha256(base_path),
    }
    for index, row in enumerate(retry["trials"], start=1):
        row["question_id"] = f"R{index:03d}"

    resolved_path, resolved, resolved_sha = prepare._triangle_control_base_selection(
        robustness_fixture.root / "retry-selection.json", retry
    )

    assert resolved_path == base_path
    assert resolved == base
    assert resolved_sha == file_sha256(base_path)


def _build_midi_only_package(
    *, tmp_path: Path, prepare, robustness_fixture, monkeypatch
) -> Path:
    config = json.loads(
        robustness_fixture.campaign_config.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        robustness_fixture.input_manifest.read_text(encoding="utf-8")
    )
    schedule = build_run_schedule(manifest, config)
    schedule_path = tmp_path / "formal_schedule.jsonl"
    write_jsonl(schedule_path, schedule)
    assert read_jsonl(schedule_path) == schedule

    output_root = tmp_path / "formal-output"
    attempt = output_root / "shared" / "attempt-001"
    attempt.mkdir(parents=True)
    theoretical = attempt / "theoretical_model.mid"
    inferences = attempt / "inferences.json"
    _write_theoretical_midi(theoretical, [(0, 240, 60, 37)])
    _write_inferences(inferences)
    verdict = {
        "content_valid": True,
        "operational_valid": True,
        "artifact_index": [
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (theoretical, inferences)
        ],
    }
    write_canonical_json(attempt / "verdict.json", verdict)

    binding = {
        "campaign_binding_sha256": "b" * 64,
        "qualification_result_sha256": config["qualification_result"]["sha256"],
    }
    campaign_audit_path = tmp_path / "campaign_audit.json"
    write_canonical_json(campaign_audit_path, {"fixture": True})
    config_sha = file_sha256(robustness_fixture.campaign_config)
    selection = json.loads(
        robustness_fixture.listening_manifest.read_text(encoding="utf-8")
    )
    control_records = []
    control_root = tmp_path / "analysis-controls"
    for index, trial in enumerate(
        [
            row
            for row in selection["trials"]
            if row["block"] == "known_different_control"
        ],
        start=1,
    ):
        formal = trial["sources"]["a"]
        synthetic = trial["sources"]["b"]
        control_dir = control_root / f"K{index:02d}"
        comparator = control_dir / "formal_comparator_excerpt.mid"
        synthetic_path = control_dir / "synthetic_known_different.mid"
        excerpt = trial["excerpt"]
        _roll, note_events = build_formal_triangle_excerpt(
            theoretical,
            comparator,
            start_model_tick=excerpt["start_model_tick"],
            end_model_tick=excerpt["end_model_tick"],
        )
        synthetic_roll, synthetic_velocity = build_triangle_control_roll(synthetic)
        write_roll_midi(
            synthetic_roll,
            synthetic_path,
            bpm=120,
            velocity=synthetic_velocity,
        )
        schedule_row = prepare._triangle_find_run(schedule, formal)
        control_records.append(
            {
                "campaign_config_sha256": config_sha,
                "semantic_id": trial["semantic_id"],
                "question_id": trial["question_id"],
                "selection_source_a_sha256": canonical_sha256(formal),
                "selection_source_b_sha256": canonical_sha256(synthetic),
                "selection_recipe_sha256": canonical_sha256(synthetic["recipe"]),
                "formal_run_id": schedule_row["run_id"],
                "formal_source_path": str(theoretical),
                "formal_source_sha256": file_sha256(theoretical),
                "formal_comparator_excerpt_path": str(comparator),
                "formal_comparator_excerpt_sha256": file_sha256(comparator),
                "formal_comparator_note_events_sha256": canonical_sha256(
                    note_events
                ),
                "synthetic_excerpt_path": str(synthetic_path),
                "synthetic_excerpt_sha256": file_sha256(synthetic_path),
                "synthetic_velocity": synthetic_velocity,
                "not_identical": True,
            }
        )
    control_report_path = tmp_path / "control_report.json"
    write_canonical_json(
        control_report_path,
        {
            "campaign_config_sha256": config_sha,
            "listening_known_different": {
                "selection_sha256": file_sha256(
                    robustness_fixture.listening_manifest
                ),
                "expected_count": 6,
                "actual_count": 6,
                "all_recipe_bound": True,
                "all_source_selectors_bound": True,
                "all_not_identical": True,
                "controls": control_records,
            },
        },
    )
    validated = (
        config,
        config_sha,
        schedule,
        robustness_fixture.input_manifest,
        [],
        binding,
        {"fixture": True},
    )
    monkeypatch.setattr(
        prepare,
        "_validated_triangle_build_inputs",
        lambda *_args, **_kwargs: validated,
    )
    monkeypatch.setattr(
        prepare,
        "_validated_campaign_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        prepare,
        "verify_attempt_verdict",
        lambda *_args, **_kwargs: (
            attempt,
            verdict,
            {theoretical.resolve(), inferences.resolve()},
        ),
    )

    package = tmp_path / "triangle-package"
    args = argparse.Namespace(
        selection=str(robustness_fixture.listening_manifest),
        config=str(robustness_fixture.campaign_config),
        config_sha256=config_sha,
        schedule=str(schedule_path),
        schedule_sha256=file_sha256(schedule_path),
        output_root=str(output_root),
        campaign_audit=str(campaign_audit_path),
        campaign_audit_sha256=file_sha256(campaign_audit_path),
        control_report=str(control_report_path),
        control_report_sha256=file_sha256(control_report_path),
        package_dir=str(package),
        midi_only=True,
        soundfont=None,
        sample_rate=44100,
        gain=0.5,
    )
    prepare.build_triangle_package(args)
    return package


def test_midi_only_package_rebuilds_synthetic_and_rejects_synchronized_tampering(
    tmp_path: Path,
    load_script,
    robustness_fixture,
    monkeypatch,
) -> None:
    prepare = load_script("prepare_robustness_listening")
    package = _build_midi_only_package(
        tmp_path=tmp_path,
        prepare=prepare,
        robustness_fixture=robustness_fixture,
        monkeypatch=monkeypatch,
    )
    baseline = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert baseline["valid"], baseline["errors"]
    assert baseline["control_report_sha256"] == file_sha256(
        tmp_path / "control_report.json"
    )
    semantic_extra = package / "blind" / "pitch-song-1.wav"
    semantic_extra.write_bytes(b"unindexed semantic preview")
    extra_file_audit = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert not extra_file_audit["valid"]
    assert any("unexpected files" in error for error in extra_file_audit["errors"])
    semantic_extra.unlink()



    key_path = package / "private" / "private_key.json"
    render_path = package / "render_manifest.json"
    original_key = key_path.read_bytes()
    original_render = render_path.read_bytes()
    key = json.loads(original_key)
    render = json.loads(original_render)
    objective_row = next(
        row
        for row in key["trials"]
        if row["block"] == "medium_primary" and row["objective_identity"] is True
    )
    objective_row["objective_identity"] = False
    write_canonical_json(key_path, key)
    render["private_key_sha256"] = file_sha256(key_path)
    write_canonical_json(render_path, render)
    identity_tamper = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert not identity_tamper["valid"]
    assert any(
        "objective_identity differs" in error for error in identity_tamper["errors"]
    ), identity_tamper["errors"]
    key_path.write_bytes(original_key)
    render_path.write_bytes(original_render)
    key = json.loads(original_key)
    render = json.loads(original_render)

    coverage_row = next(
        row for row in key["trials"] if row["block"] == "medium_primary"
    )
    coverage_row["coverage_driven"] = not coverage_row["coverage_driven"]
    coverage_row["coverage_collapse"] = not coverage_row["coverage_collapse"]
    coverage_row["coverage_ratios"] = {"a": 0.125, "b": 0.875}
    write_canonical_json(key_path, key)
    render["private_key_sha256"] = file_sha256(key_path)
    write_canonical_json(render_path, render)
    coverage_tamper = prepare.audit_triangle_package_dir(
        package, require_wav=False
    )
    assert not coverage_tamper["valid"]
    assert any(
        "coverage metadata differs" in error
        for error in coverage_tamper["errors"]
    ), coverage_tamper["errors"]
    key_path.write_bytes(original_key)
    render_path.write_bytes(original_render)
    key = json.loads(original_key)
    render = json.loads(original_render)

    control_report_path = Path(render["control_report_path"])
    original_control_report = control_report_path.read_bytes()
    control_report = json.loads(original_control_report)
    control_report["listening_known_different"]["controls"][0][
        "formal_comparator_excerpt_sha256"
    ] = "0" * 64
    write_canonical_json(control_report_path, control_report)
    synchronized_control_sha = file_sha256(control_report_path)
    key["control_report_sha256"] = synchronized_control_sha
    write_canonical_json(key_path, key)
    render["control_report_sha256"] = synchronized_control_sha
    render["private_key_sha256"] = file_sha256(key_path)
    write_canonical_json(render_path, render)
    control_tamper = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert not control_tamper["valid"]
    assert any(
        "control-report revalidation" in error
        for error in control_tamper["errors"]
    ), control_tamper["errors"]

    control_report_path.write_bytes(original_control_report)
    key_path.write_bytes(original_key)
    render_path.write_bytes(original_render)
    key = json.loads(original_key)
    render = json.loads(original_render)
    sources = [
        row[side]
        for row in key["trials"]
        for side in ("source_a", "source_b")
    ]
    synthetic = next(source for source in sources if source["kind"] == "synthetic_control")
    synthetic_events = prepare._canonical_midi_contract(
        Path(synthetic["excerpt_midi_path"]), expected_end_model_tick=64
    )
    assert synthetic_events
    assert {event["velocity"] for event in synthetic_events} == {96}

    formal = next(source for source in sources if source["kind"] == "formal")
    formal_path = Path(formal["excerpt_midi_path"])
    original_formal = formal_path.read_bytes()
    midi = mido.MidiFile(formal_path)
    changed = False
    for track in midi.tracks:
        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                message.velocity += 1
                changed = True
                break
        if changed:
            break
    assert changed
    midi.save(formal_path)
    tampered_events = prepare._canonical_midi_contract(
        formal_path, expected_end_model_tick=64
    )
    formal_id = formal["canonical_source_id"]
    for row in key["trials"]:
        for side in ("source_a", "source_b"):
            source = row[side]
            if source["canonical_source_id"] == formal_id:
                source["excerpt_midi_sha256"] = file_sha256(formal_path)
                source["excerpt_note_event_sha256"] = prepare._note_event_sha256(
                    tampered_events
                )
    write_canonical_json(key_path, key)
    render["private_key_sha256"] = file_sha256(key_path)
    write_canonical_json(render_path, render)
    tampered = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert not tampered["valid"]
    assert any(
        "independent rebuild" in error for error in tampered["errors"]
    ), tampered["errors"]

    # Restore the first attack, then prove that synchronizing a blind MIDI hash
    # cannot hide a program/bank contract change either.
    formal_path.write_bytes(original_formal)
    key_path.write_bytes(original_key)
    render_path.write_bytes(original_render)
    render = json.loads(original_render)
    presentation = render["trials"][0]["presentations"][0]
    blind_midi = (package / presentation["midi"]).resolve()
    midi = mido.MidiFile(blind_midi)
    program = next(
        message
        for track in midi.tracks
        for message in track
        if message.type == "program_change"
    )
    program.program = 5
    midi.save(blind_midi)
    presentation["midi_sha256"] = file_sha256(blind_midi)
    write_canonical_json(render_path, render)
    program_tamper = prepare.audit_triangle_package_dir(package, require_wav=False)
    assert not program_tamper["valid"]
    assert any(
        "program 0" in error for error in program_tamper["errors"]
    ), program_tamper["errors"]

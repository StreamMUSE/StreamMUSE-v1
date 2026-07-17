from __future__ import annotations

import argparse
import copy
import json
import threading
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from streammuse.experiments.melody_robustness import (
    canonical_sha256,
    file_sha256,
    write_canonical_json,
)
from streammuse.experiments.triangle_listening import (
    BLOCK_COUNTS,
    PRIMARY_CONDITIONS,
    TRIANGLE_LISTENING_ATTEMPT_ID,
    TRIANGLE_PATTERNS,
    append_response,
    append_sitting_event,
    build_triangle_selection_manifest,
    create_snapshot,
    derive_triangle_retry_seed,
    derive_triangle_retry_selection_manifest,
    export_generated_acc,
    progress_summary,
    summarize_unblinded,
    unblind_snapshot,
    validate_response_ledger,
    validate_sitting_ledger,
    validate_snapshot,
    validate_triangle_selection_manifest,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_package(
    tmp_path: Path,
    fixture,
    *,
    selection_path: Path | None = None,
    name: str = "triangle-workflow",
) -> tuple[Path, dict]:
    package = tmp_path / name
    (package / "blind").mkdir(parents=True)
    (package / "private").mkdir()
    selection_path = (
        fixture.listening_manifest.resolve()
        if selection_path is None
        else selection_path.resolve()
    )
    selection = _load(selection_path)
    key_rows = []
    for trial in selection["trials"]:
        source_a = {**trial["sources"]["a"], "source_empty": False}
        source_b = {**trial["sources"]["b"], "source_empty": False}
        key_rows.append(
            {
                "question_id": trial["question_id"],
                "semantic_id": trial["semantic_id"],
                "block": trial["block"],
                "condition": trial["condition"],
                "repeat_of": trial.get("repeat_of"),
                "repeat_distance": trial.get("repeat_distance"),
                "presentation_pattern": trial["presentation_pattern"],
                "correct_choice": trial["correct_choice"],
                "odd_position": trial["odd_position"],
                "source_a": source_a,
                "source_b": source_b,
                "pair_id": (
                    str(trial.get("repeat_of"))
                    if trial["block"] == "exact_repeat"
                    else trial["semantic_id"]
                ),
                "objective_identity": trial["block"] == "identity_catch",
                "coverage_driven": False,
                "coverage_collapse": False,
                "coverage_ratios": {"a": 1.0, "b": 1.0},
            }
        )
    write_canonical_json(
        package / "private" / "private_key.json",
        {
            "schema_version": (
                "streammuse.melody_robustness.listening_triangle_private_key.v2"
            ),
            "listening_attempt_id": selection["listening_attempt_id"],
            "retry_lineage": selection.get("retry_lineage"),
            "retry_lineage_sha256": selection.get("retry_lineage_sha256"),
            "selection_sha256": file_sha256(selection_path),
            "trials": key_rows,
        },
    )
    write_canonical_json(
        package / "render_manifest.json",
        {
            "schema_version": (
                "streammuse.melody_robustness.listening_triangle_render.v2"
            ),
            "listening_attempt_id": selection["listening_attempt_id"],
            "retry_lineage": selection.get("retry_lineage"),
            "retry_lineage_sha256": selection.get("retry_lineage_sha256"),
            "selection_path": str(selection_path),
            "selection_sha256": file_sha256(selection_path),
            "midi_only_development": False,
        },
    )
    write_canonical_json(
        package / "package_audit.json",
        {
            "schema_version": (
                "streammuse.melody_robustness.listening_triangle_audit.v2"
            ),
            "listening_attempt_id": selection["listening_attempt_id"],
            "retry_lineage": selection.get("retry_lineage"),
            "retry_lineage_sha256": selection.get("retry_lineage_sha256"),
            "valid": True,
            "accepted_final": True,
        },
    )
    (package / "blind" / "player.html").write_text("blind player", encoding="utf-8")
    return package, selection


def _answer(package: Path, trial: dict, *, sitting: str = "sitting-001") -> dict:
    _rows, _head, states = validate_sitting_ledger(package)
    if sitting not in states:
        append_sitting_event(
            package,
            event="start",
            sitting_id=sitting,
            device="test headphones",
            environment="quiet unit-test room",
        )
    return append_response(
        package,
        trial_id=trial["question_id"],
        odd_choice=trial["correct_choice"],
        confidence_1_to_5=4,
        sitting_id=sitting,
        difference_tags=("rhythm_timing",),
        note="",
        play_counts=(1, 1, 1),
        response_time_ms=250,
    )


def test_triangle_selection_is_exact_prefix_balanced_and_pattern_balanced(
    robustness_fixture,
):
    selection = _load(robustness_fixture.listening_manifest)
    manifest = _load(robustness_fixture.input_manifest)

    assert validate_triangle_selection_manifest(
        selection,
        manifest,
        manifest_path=robustness_fixture.input_manifest,
        verify_files=True,
    ) == selection
    assert selection["listening_attempt_id"] == TRIANGLE_LISTENING_ATTEMPT_ID
    assert Counter(row["block"] for row in selection["trials"]) == BLOCK_COUNTS

    coverage: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for trial in selection["trials"]:
        if trial["block"] == "medium_primary":
            source = trial["sources"]["b"]
            coverage[(source["song"], trial["condition"])].add(
                (source["perturb_seed"], source["sample_seed"])
            )
    assert set(coverage) == {
        (str(song), condition)
        for song in range(1, 6)
        for condition in PRIMARY_CONDITIONS
    }
    assert {len(pairs) for pairs in coverage.values()} == {4}

    for chunk_start in range(0, 95, 19):
        chunk = selection["trials"][chunk_start : chunk_start + 19]
        medium = [row for row in chunk if row["block"] == "medium_primary"]
        assert Counter(row["condition"] for row in medium) == {
            "pitch": 4,
            "onset": 4,
            "both": 4,
        }
        assert sorted(Counter(row["sources"]["a"]["song"] for row in chunk).values()) == [
            3,
            4,
            4,
            4,
            4,
        ]

    for condition in PRIMARY_CONDITIONS:
        rows = [
            row
            for row in selection["trials"]
            if row["block"] == "medium_primary" and row["condition"] == condition
        ]
        assert Counter(row["duplicated_source"] for row in rows) == {"a": 10, "b": 10}
        assert Counter(row["odd_position"] for row in rows) == {1: 6, 2: 7, 3: 7}
        pattern_counts = Counter(row["presentation_pattern"] for row in rows)
        assert set(pattern_counts) == set(TRIANGLE_PATTERNS)
        assert set(pattern_counts.values()) <= {3, 4}

    originals = {
        row["semantic_id"]: row
        for row in selection["trials"]
        if row["block"] != "exact_repeat"
    }
    repeats = [row for row in selection["trials"] if row["block"] == "exact_repeat"]
    assert len(repeats) == 8
    for repeat in repeats:
        original = originals[repeat["repeat_of"]]
        assert repeat["repeat_distance"] >= 8
        assert repeat["sources"] == original["sources"]
        assert repeat["duplicated_source"] == original["duplicated_source"]
        assert repeat["odd_position"] != original["odd_position"]


def test_triangle_selection_exact_rebuild_rejects_any_semantic_mutation(
    robustness_fixture,
):
    selection = _load(robustness_fixture.listening_manifest)
    manifest = _load(robustness_fixture.input_manifest)
    changed = copy.deepcopy(selection)
    changed["trials"][0]["presentation_pattern"] = "AAA"

    with pytest.raises(ValueError, match="selection manifest mismatch"):
        validate_triangle_selection_manifest(
            changed,
            manifest,
            manifest_path=robustness_fixture.input_manifest,
            verify_files=True,
        )


def test_invalid_response_never_writes_or_corrupts_ledger(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    valid = {
        "trial_id": selection["trials"][0]["question_id"],
        "odd_choice": "1",
        "confidence_1_to_5": 3,
        "sitting_id": "sitting-001",
        "difference_tags": (),
        "note": "",
        "play_counts": (1, 1, 1),
        "response_time_ms": 10,
    }
    with pytest.raises(ValueError, match="no structured sitting start"):
        append_response(package, **valid)
    assert not (package / "blind" / "response_ledger.jsonl").exists()
    append_sitting_event(
        package,
        event="start",
        sitting_id="sitting-001",
        device="test headphones",
        environment="quiet unit-test room",
    )
    invalid_overrides = [
        {"trial_id": "P001"},
        {"sitting_id": 7},
        {"note": 7},
        {"difference_tags": "density"},
        {"play_counts": (1, True, 1)},
        {"response_time_ms": -1},
        {"submitted_at": "2026-07-17T12:00:00"},
    ]
    for override in invalid_overrides:
        with pytest.raises(ValueError):
            append_response(package, **{**valid, **override})
        assert not (package / "blind" / "response_ledger.jsonl").exists()
        assert validate_response_ledger(package) == ([], None)


def test_sitting_ledger_direct_tamper_fails_hash_chain(
    robustness_fixture, tmp_path
):
    package, _selection = _workflow_package(
        tmp_path, robustness_fixture, name="sitting-ledger-tamper"
    )
    append_sitting_event(
        package,
        event="start",
        sitting_id="sitting-tamper",
        device="test headphones",
        environment="quiet room",
    )
    ledger = package / "blind" / "sitting_ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8"))
    row["note"] = "edited without rebuilding the chain"
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sitting ledger hash mismatch"):
        validate_sitting_ledger(package)


def test_sitting_duplicate_start_and_end_without_active_start_fail_atomically(
    robustness_fixture, tmp_path
):
    package, _selection = _workflow_package(
        tmp_path, robustness_fixture, name="invalid-sitting-transitions"
    )
    ledger = package / "blind" / "sitting_ledger.jsonl"
    with pytest.raises(ValueError, match="requires one active start"):
        append_sitting_event(
            package,
            event="end",
            sitting_id="never-started",
        )
    assert not ledger.exists()

    append_sitting_event(
        package,
        event="start",
        sitting_id="sitting-001",
        device="test headphones",
        environment="quiet room",
    )
    with pytest.raises(ValueError, match="already used"):
        append_sitting_event(
            package,
            event="start",
            sitting_id="sitting-001",
            device="test headphones",
            environment="quiet room",
        )
    assert len(validate_sitting_ledger(package)[0]) == 1

    append_sitting_event(
        package,
        event="end",
        sitting_id="sitting-001",
    )
    with pytest.raises(ValueError, match="requires one active start"):
        append_sitting_event(
            package,
            event="end",
            sitting_id="sitting-001",
        )
    assert len(validate_sitting_ledger(package)[0]) == 2


def test_concurrent_same_trial_responses_are_serialized_as_one_cas(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(
        tmp_path, robustness_fixture, name="concurrent-response-cas"
    )
    append_sitting_event(
        package,
        event="start",
        sitting_id="concurrent-sitting",
        device="test headphones",
        environment="quiet room",
    )
    barrier = threading.Barrier(3)
    successes: list[dict] = []
    failures: list[Exception] = []
    result_lock = threading.Lock()

    def submit() -> None:
        barrier.wait()
        try:
            result = append_response(
                package,
                trial_id=selection["trials"][0]["question_id"],
                odd_choice=selection["trials"][0]["correct_choice"],
                confidence_1_to_5=4,
                sitting_id="concurrent-sitting",
                play_counts=(1, 1, 1),
                response_time_ms=100,
            )
            with result_lock:
                successes.append(result)
        except Exception as exc:
            with result_lock:
                failures.append(exc)

    workers = [threading.Thread(target=submit) for _index in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    rows, head = validate_response_ledger(package)
    assert len(successes) == 1
    assert len(failures) == 1
    assert "next frozen trial is Q002" in str(failures[0])
    assert len(rows) == 1
    assert head == rows[0]["record_hash"]
    assert progress_summary(package)["answered_count"] == 1


def test_static_export_import_preserves_structured_sitting_events_idempotently(
    load_script, robustness_fixture, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    package, selection = _workflow_package(
        tmp_path, robustness_fixture, name="static-sitting-import"
    )
    sitting_id = "static-sitting-001"
    start_time = "2026-07-17T08:00:00+00:00"
    response_time = "2026-07-17T08:05:00+00:00"
    end_time = "2026-07-17T08:10:00+00:00"
    response_export = tmp_path / "triangle-responses.json"
    response_export.write_text(
        json.dumps(
            {
                "selection_sha256": file_sha256(
                    robustness_fixture.listening_manifest
                ),
                "sitting_events": [
                    {
                        "event": "start",
                        "sitting_id": sitting_id,
                        "device": "test headphones",
                        "environment": "quiet room",
                        "note": "static browser session",
                        "recorded_at": start_time,
                    },
                    {
                        "event": "end",
                        "sitting_id": sitting_id,
                        "note": "completed one question",
                        "anomalies": ["brief noise"],
                        "recorded_at": end_time,
                    },
                ],
                "responses": [
                    {
                        "trial_id": selection["trials"][0]["question_id"],
                        "odd_choice": selection["trials"][0]["correct_choice"],
                        "confidence_1_to_5": 4,
                        "difference_tags": ["rhythm_timing"],
                        "note": "",
                        "play_counts": [1, 1, 1],
                        "response_time_ms": 300,
                        "sitting_id": sitting_id,
                        "submitted_at": response_time,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        package_dir=str(package),
        responses=str(response_export),
    )

    listening.import_triangle_responses(args)
    response_rows, _response_head = validate_response_ledger(package)
    sitting_rows, _sitting_head, states = validate_sitting_ledger(package)
    assert len(response_rows) == 1
    assert [row["event"] for row in sitting_rows] == ["start", "end"]
    assert states[sitting_id]["start"]["recorded_at"] == start_time
    assert states[sitting_id]["end"]["recorded_at"] == end_time
    assert states[sitting_id]["end"]["anomalies"] == ["brief noise"]

    listening.import_triangle_responses(args)
    assert len(validate_response_ledger(package)[0]) == 1
    assert len(validate_sitting_ledger(package)[0]) == 2

def test_one_answer_pause_resume_snapshot_and_partial_unblind_boundary(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    assert progress_summary(package)["collection_status"] == "not_started"

    first = _answer(package, selection["trials"][0])
    assert first["progress"]["answered_count"] == 1
    assert first["progress"]["next_trial_id"] == "Q002"
    snapshot_one, sealed_one = create_snapshot(package)
    assert sealed_one["collection_status"] == "partial"
    assert sealed_one["answered_count"] == 1
    assert sealed_one["pending_count"] == 94
    assert create_snapshot(package)[0] == snapshot_one

    unblind_snapshot(package, snapshot_one)
    _summary_path, first_summary = summarize_unblinded(package, snapshot_one)
    assert first_summary["answered_count"] == 1
    assert first_summary["qc_status"] == "pending"
    assert first_summary["attempt_disposition"] == "in_progress_qc_pending"
    with pytest.raises(ValueError, match="full 95-response snapshot"):
        derive_triangle_retry_selection_manifest(package, snapshot_one)

    second = _answer(package, selection["trials"][1], sitting="sitting-002")
    assert second["response"]["blinding_phase"] == (
        "post_partial_unblind_exploratory"
    )
    resumed = progress_summary(package)
    assert resumed["answered_count"] == 2
    assert resumed["sitting_counts"] == {"sitting-001": 1, "sitting-002": 1}
    assert resumed["blinding_status"] == "partially_unblinded_during_collection"

    snapshot_two, _sealed_two = create_snapshot(package)
    unblind_snapshot(package, snapshot_two)
    _path, second_summary = summarize_unblinded(package, snapshot_two)
    assert second_summary["views"]["pre_unblind"]["answered"] == 1
    assert second_summary["views"]["post_unblind_exploratory"]["answered"] == 1
    assert second_summary["views"]["combined_descriptive"]["answered"] == 2


def test_generated_acc_export_reuses_first_authorization_after_later_partial_unblind(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(
        tmp_path, robustness_fixture, name="repeat-semantic-export"
    )
    generated_midi = tmp_path / "formal-theoretical.mid"
    excerpt_midi = tmp_path / "canonical-excerpt.mid"
    canonical_wav = tmp_path / "canonical-excerpt.wav"
    generated_midi.write_bytes(b"formal-generated-midi")
    excerpt_midi.write_bytes(b"canonical-excerpt-midi")
    canonical_wav.write_bytes(b"canonical-eight-second-wav")

    key_path = package / "private" / "private_key.json"
    key = _load(key_path)
    formal_count = 0
    for trial in key["trials"]:
        for side in ("source_a", "source_b"):
            source = trial[side]
            if source.get("kind") != "formal":
                continue
            formal_count += 1
            source.update(
                {
                    "source_path": str(generated_midi.resolve()),
                    "source_sha256": file_sha256(generated_midi),
                    "excerpt_midi_path": str(excerpt_midi.resolve()),
                    "excerpt_midi_sha256": file_sha256(excerpt_midi),
                    "canonical_wav_path": str(canonical_wav.resolve()),
                    "canonical_wav_sha256": file_sha256(canonical_wav),
                }
            )
    assert formal_count > 0
    write_canonical_json(key_path, key)

    _answer(package, selection["trials"][0])
    snapshot_one, _sealed_one = create_snapshot(package)
    unblind_snapshot(package, snapshot_one)
    summarize_unblinded(package, snapshot_one)
    csv_one, rows_one = export_generated_acc(package, snapshot_dir=snapshot_one)
    index_path = csv_one.with_name("generated_acc_index.json")
    first_index = _load(index_path)
    first_index_bytes = index_path.read_bytes()
    assert first_index["export_authorizing_snapshot_id"] == snapshot_one.name
    assert first_index["answered_count"] == 1

    _answer(package, selection["trials"][1], sitting="sitting-002")
    snapshot_two, _sealed_two = create_snapshot(package)
    unblind_snapshot(package, snapshot_two)
    summarize_unblinded(package, snapshot_two)
    csv_two, rows_two = export_generated_acc(package, snapshot_dir=snapshot_two)

    assert csv_two == csv_one
    assert rows_two == rows_one
    assert index_path.read_bytes() == first_index_bytes
    assert _load(index_path)["export_authorizing_snapshot_id"] == snapshot_one.name

    tampered = _load(index_path)
    tampered["answered_count"] = 2
    write_canonical_json(index_path, tampered)
    with pytest.raises(ValueError, match="authorizing snapshot answered_count drifted"):
        export_generated_acc(package, snapshot_dir=snapshot_two)

def test_snapshot_rehashed_edit_still_fails_against_durable_ledger_prefix(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    _answer(package, selection["trials"][0])
    snapshot, sealed = create_snapshot(package)
    forged = copy.deepcopy(sealed)
    forged["responses"][0]["note"] = "edited after sealing"
    response = forged["responses"][0]
    body = {key: value for key, value in response.items() if key != "record_hash"}
    response["record_hash"] = canonical_sha256(body)
    forged["ledger_head_hash"] = response["record_hash"]
    forged_id = f"snapshot-001-{response['record_hash'][:12]}"
    forged["snapshot_id"] = forged_id
    forged_dir = snapshot.parent / forged_id
    forged_dir.mkdir()
    write_canonical_json(forged_dir / "sealed_responses.json", forged)

    with pytest.raises(ValueError, match="durable-ledger prefix"):
        validate_snapshot(package, forged_dir)


def test_unblind_state_deletion_or_rebinding_fails_closed(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    _answer(package, selection["trials"][0])
    snapshot, _sealed = create_snapshot(package)
    unblind_snapshot(package, snapshot)
    state_path = package / "unblind_state.json"
    sidecar = package / "unblind_state.json.sha256"
    state_bytes = state_path.read_bytes()
    sidecar_bytes = sidecar.read_bytes()

    state_path.unlink()
    with pytest.raises(ValueError, match="unblind_state is missing"):
        validate_response_ledger(package)
    state_path.write_bytes(state_bytes)
    sidecar.write_bytes(sidecar_bytes)

    tampered = _load(state_path)
    tampered["first_snapshot_answered_count"] = 2
    write_canonical_json(state_path, tampered)
    with pytest.raises(ValueError, match="snapshot bindings mismatch"):
        validate_response_ledger(package)


def test_n94_and_n95_snapshots_are_valid_and_full_qc_decisions_are_gated(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    for trial in selection["trials"][:94]:
        _answer(package, trial)
    snapshot_94, sealed_94 = create_snapshot(package)
    assert sealed_94["collection_status"] == "partial"
    assert sealed_94["answered_count"] == 94
    assert sealed_94["pending_count"] == 1

    _answer(package, selection["trials"][94])
    snapshot_95, sealed_95 = create_snapshot(package)
    assert sealed_95["collection_status"] == "full"
    assert sealed_95["answered_count"] == 95
    unblind_snapshot(package, snapshot_95)
    _summary_path, summary = summarize_unblinded(package, snapshot_95)
    assert summary["collection_status"] == "full"
    assert summary["qc_status"] == "pass"
    assert summary["attempt_disposition"] == "eligible_for_preregistered_decisions"
    assert all(
        result["decision"]
        == "confirmed discriminable in this fixed listener/package"
        for result in summary["conditions"].values()
    )
    with pytest.raises(ValueError, match="qc_status=fail"):
        derive_triangle_retry_selection_manifest(package, snapshot_95)
    assert (package / "full" / "unblinded_scores.json").is_file()
    assert (package / "full" / "discrimination_summary.json").is_file()
    assert validate_snapshot(package, snapshot_94)["answered_count"] == 94


def test_partial_summary_identity_and_operational_counts_use_answered_rows_only(
    robustness_fixture, tmp_path
):
    package, selection = _workflow_package(
        tmp_path, robustness_fixture, name="partial-identities"
    )
    key_path = package / "private" / "private_key.json"
    key = _load(key_path)
    for side in ("source_a", "source_b"):
        key["trials"][0][side].update(
            {
                "operational_valid": False if side == "source_a" else True,
                "raw_token_payload_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "excerpt_midi_sha256": "c" * 64,
                "rendered_pair_wav_sha256": "d" * 64,
            }
        )
    write_canonical_json(key_path, key)
    _answer(package, selection["trials"][0])
    snapshot, _sealed = create_snapshot(package)
    unblind_snapshot(package, snapshot)
    _summary_path, summary = summarize_unblinded(package, snapshot)
    answered = summary["views"]["combined_descriptive"]
    assert answered["answered"] == 1
    assert answered["operational_invalid_either"] == 1
    assert answered["raw_token_identity"] == 1
    assert answered["theoretical_midi_identity"] == 1
    assert answered["canonical_midi_identity"] == 1
    assert answered["rendered_wav_identity"] == 1


def test_full_qc_failure_derives_deterministic_attempt_002_and_empty_new_ledger(
    load_script, robustness_fixture, tmp_path
):
    failed_package, failed_selection = _workflow_package(
        tmp_path, robustness_fixture, name="failed-attempt-001"
    )
    append_sitting_event(
        failed_package,
        event="start",
        sitting_id="failed-sitting",
        device="test headphones",
        environment="quiet unit-test room",
    )
    for trial in failed_selection["trials"]:
        append_response(
            failed_package,
            trial_id=trial["question_id"],
            odd_choice="no_difference",
            confidence_1_to_5=2,
            sitting_id="failed-sitting",
            play_counts=(1, 1, 1),
            response_time_ms=10,
        )
    failed_snapshot, _sealed = create_snapshot(failed_package)
    unblind_snapshot(failed_package, failed_snapshot)
    _summary_path, failed_summary = summarize_unblinded(
        failed_package, failed_snapshot
    )
    assert failed_summary["qc_status"] == "fail"
    assert failed_summary["retry_required"] is True

    retry, authorization_path, authorization = (
        derive_triangle_retry_selection_manifest(failed_package, failed_snapshot)
    )
    repeated, repeated_authorization_path, repeated_authorization = (
        derive_triangle_retry_selection_manifest(failed_package, failed_snapshot)
    )
    assert retry == repeated
    assert authorization_path == repeated_authorization_path
    assert authorization == repeated_authorization
    assert retry["listening_attempt_id"] == "listening-attempt-002"
    expected_seed, expected_seed_sha = derive_triangle_retry_seed(
        base_blind_seed=failed_selection["base_blind_order_seed"],
        attempt_number=2,
    )
    assert retry["effective_blind_order_seed"] == expected_seed
    assert retry["effective_blind_order_seed_sha256"] == expected_seed_sha
    assert retry["retry_lineage"]["previous_qc_status"] == "fail"
    assert retry["retry_lineage"]["retry_authorization_path"] == str(
        authorization_path
    )

    def semantics(selection: dict) -> dict[str, dict]:
        ignored = {
            "presentation_pattern",
            "duplicated_source",
            "correct_choice",
            "odd_position",
            "global_order_index",
            "question_id",
            "repeat_distance",
        }
        return {
            row["semantic_id"]: {key: value for key, value in row.items() if key not in ignored}
            for row in selection["trials"]
        }

    assert semantics(retry) == semantics(failed_selection)
    assert [row["semantic_id"] for row in retry["trials"]] != [
        row["semantic_id"] for row in failed_selection["trials"]
    ]
    retry_path = tmp_path / "attempt-002-selection.json"
    listening = load_script("prepare_robustness_listening")
    listening.derive_triangle_retry(
        argparse.Namespace(
            failed_package=str(failed_package),
            failed_snapshot=str(failed_snapshot),
            output=str(retry_path),
        )
    )
    assert _load(retry_path) == retry
    assert validate_triangle_selection_manifest(
        retry,
        _load(robustness_fixture.input_manifest),
        manifest_path=robustness_fixture.input_manifest,
        verify_files=True,
    ) == retry

    new_package, _new_selection = _workflow_package(
        tmp_path,
        robustness_fixture,
        selection_path=retry_path,
        name="fresh-attempt-002",
    )
    assert progress_summary(new_package)["collection_status"] == "not_started"
    assert progress_summary(new_package)["answered_count"] == 0
    assert validate_response_ledger(new_package) == ([], None)
    assert not (new_package / "blind" / "response_ledger.jsonl").exists()
    assert validate_sitting_ledger(new_package) == ([], None, {})
    assert not (new_package / "blind" / "sitting_ledger.jsonl").exists()

    # Even a self-consistent rewrite of one failed artifact is rejected because
    # attempt-002 binds the original summary hash through authorization/lineage.
    summary_path = failed_snapshot / "partial_discrimination_summary.json"
    changed = _load(summary_path)
    changed["limitations"].append("tampered after retry authorization")
    write_canonical_json(summary_path, changed)
    with pytest.raises(ValueError, match="summary|authorization|failed-artifact"):
        validate_triangle_selection_manifest(
            retry,
            _load(robustness_fixture.input_manifest),
            manifest_path=robustness_fixture.input_manifest,
            verify_files=True,
        )


def test_local_http_post_is_durable_before_progress_advances(
    load_script, robustness_fixture, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    package, selection = _workflow_package(tmp_path, robustness_fixture)
    player = listening._triangle_player_html(
        {"question_prompt": "blind", "practice_trials": [], "trials": []},
        "a" * 64,
    )
    assert 'id="device"' in player and 'id="environment"' in player
    assert "/api/sitting/start" in player and "/api/sitting/end" in player
    assert "function stopAllAudio()" in player
    assert "stopAllAudio();const a=" in player
    server = listening.make_triangle_server(
        package, host="127.0.0.1", port=0, quiet=True, require_final=True
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    sitting_id = "browser-sitting-001"
    start_payload = json.dumps(
        {
            "sitting_id": sitting_id,
            "device": "test headphones",
            "environment": "quiet test room",
        }
    ).encode("utf-8")
    payload = json.dumps(
        {
            "trial_id": selection["trials"][0]["question_id"],
            "odd_choice": selection["trials"][0]["correct_choice"],
            "confidence_1_to_5": 4,
            "difference_tags": [],
            "note": "",
            "play_counts": [1, 1, 1],
            "response_time_ms": 123,
            "sitting_id": sitting_id,
        }
    ).encode("utf-8")
    try:
        semantic_extra = package / "blind" / "condition-map.json"
        semantic_extra.write_text('{"Q001":"pitch"}', encoding="utf-8")
        with pytest.raises(urllib.error.HTTPError) as blocked:
            urllib.request.urlopen(
                f"http://{host}:{port}/condition-map.json", timeout=5
            )
        assert blocked.value.code == 404

        start_request = urllib.request.Request(
            f"http://{host}:{port}/api/sitting/start",
            data=start_payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(start_request, timeout=5) as response:
            start_result = json.loads(response.read())
        assert start_result["sitting_event"]["event"] == "start"
        request = urllib.request.Request(
            f"http://{host}:{port}/api/response",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["progress"]["answered_count"] == 1
        with urllib.request.urlopen(f"http://{host}:{port}/api/progress", timeout=5) as response:
            progress = json.loads(response.read())
        assert progress["answered_count"] == 1
        rows, head = validate_response_ledger(package)
        assert len(rows) == 1
        assert head == result["response"]["record_hash"]
        assert _load(package / "blind" / "progress_state.json") == progress
        end_request = urllib.request.Request(
            f"http://{host}:{port}/api/sitting/end",
            data=json.dumps(
                {"sitting_id": sitting_id, "anomalies": ["brief external noise"]}
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(end_request, timeout=5) as response:
            end_result = json.loads(response.read())
        assert end_result["sitting_event"]["event"] == "end"
        sitting_rows, sitting_head, sitting_states = validate_sitting_ledger(package)
        assert len(sitting_rows) == 2
        assert sitting_head == end_result["sitting_event"]["record_hash"]
        assert sitting_states[sitting_id]["end"]["anomalies"] == [
            "brief external noise"
        ]
        with pytest.raises(ValueError, match="after the sitting was ended"):
            append_response(
                package,
                trial_id=selection["trials"][1]["question_id"],
                odd_choice=selection["trials"][1]["correct_choice"],
                confidence_1_to_5=3,
                sitting_id=sitting_id,
                play_counts=(1, 1, 1),
                response_time_ms=10,
            )
        assert len(validate_response_ledger(package)[0]) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

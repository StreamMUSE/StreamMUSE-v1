from __future__ import annotations

import copy
from collections import Counter

import pytest

from streammuse.experiments.melody_robustness import (
    ATTESTATION_BUNDLE_SCHEMA_VERSION,
    CAMPAIGN_SCHEMA_VERSION,
    CONDITION_TABLE,
    RUN_SCHEMA_VERSION,
    SEEDS,
    build_run_schedule,
    default_campaign_config,
    require_exact_artifacts,
    validate_campaign_config,
    validate_input_manifest,
)


SONGS = tuple(str(index) for index in range(1, 6))


def _attestation_records() -> dict:
    return {
        "schema_version": ATTESTATION_BUNDLE_SCHEMA_VERSION,
        "code_identity": {
            "path": "campaign/code_identity.json",
            "sha256": "1" * 64,
        },
        "checkpoint_identity": {
            "path": "campaign/checkpoint_identity.json",
            "sha256": "2" * 64,
        },
        "environment": {
            "path": "campaign/environment.json",
            "sha256": "3" * 64,
        },
        "qualification_spec": {
            "path": "campaign/qualification_spec.json",
            "sha256": "4" * 64,
        },
    }


def _raw_campaign_config() -> dict:
    config = default_campaign_config(
        code_identity="a" * 40,
        checkpoint_path="models/test.safetensors",
        checkpoint_sha256="c" * 64,
        input_manifest_path="campaign/input_manifest.json",
        input_manifest_sha256="d" * 64,
        attestation=_attestation_records(),
        playback_tempo=60,
        tail_beats=24,
    )
    config["qualification"] = {
        "dense_song": "2",
        "tail_songs": ["2", "4"],
        "sample_seed": SEEDS["sample"][0],
        "perturb_seed": SEEDS["perturb"][0],
        "tempo_candidates": [60, 30],
        "tail_candidates": [8, 16, 24],
        "decision_order": ["determinism", "static_input_gate", "tempo", "tail"],
    }
    config["status"] = "qualified_frozen"
    config["qualification_candidate"] = {
        "path": "campaign/qualification_config.json",
        "sha256": "e" * 64,
    }
    config["qualification_result"] = {
        "path": "campaign/qualification_result.json",
        "sha256": "f" * 64,
    }
    config["listening"]["selection_manifest_path"] = "campaign/listening.json"
    config["listening"]["selection_manifest_sha256"] = "b" * 64
    config["listening"]["renderer_identity"] = {
        "path": "campaign/triangle_renderer_identity.json",
        "sha256": "9" * 64,
    }
    return config


def _campaign_config() -> dict:
    # Keep unrelated mutation/fail-closed cases isolated even if production regresses to
    # returning references to its module-level dictionaries.
    return copy.deepcopy(_raw_campaign_config())


def _input_manifest() -> dict:
    entries = []
    for song in SONGS:
        for condition, spec in CONDITION_TABLE.items():
            if condition == "sham":
                perturb_seeds = [None]
            elif condition == "high":
                perturb_seeds = [SEEDS["high_perturb"]]
            else:
                perturb_seeds = list(SEEDS["perturb"])
            assert len(perturb_seeds) == spec["pseed_count"]
            for perturb_seed in perturb_seeds:
                suffix = "none" if perturb_seed is None else str(perturb_seed)
                entries.append(
                    {
                        "stem": f"{song}__{condition}__ps-{suffix}",
                        "song": song,
                        "source_stem": song,
                        "condition": condition,
                        "perturb_seed": perturb_seed,
                        "pitch_probability": spec["pitch_probability"],
                        "onset_probability": spec["onset_probability"],
                    }
                )
    assert len(entries) == 40
    return {
        "entries": entries,
        "input_count": 40,
        "exact_stems": sorted(entry["stem"] for entry in entries),
        "perturb_seeds": list(SEEDS["perturb"]),
        "high_pseed": SEEDS["high_perturb"],
    }


def test_frozen_constants_match_the_registered_campaign_seeds_and_doses():
    assert SEEDS == {
        "perturb": [2026071001, 2026071002],
        "sample": [2026071101, 2026071102],
        "high_perturb": 2026071001,
        "run_order": 2026071201,
        "bootstrap": 2026071301,
        "blind_order": 2026071401,
    }
    assert CONDITION_TABLE == {
        "sham": {"pitch_probability": 0.0, "onset_probability": 0.0, "pseed_count": 1},
        "pitch": {"pitch_probability": 0.05, "onset_probability": 0.0, "pseed_count": 2},
        "onset": {"pitch_probability": 0.0, "onset_probability": 0.15, "pseed_count": 2},
        "both": {"pitch_probability": 0.05, "onset_probability": 0.15, "pseed_count": 2},
        "high": {"pitch_probability": 0.20, "onset_probability": 0.40, "pseed_count": 1},
    }


def test_valid_40_entry_manifest_builds_reproducible_160_run_schedule():
    manifest = _input_manifest()
    config = _campaign_config()

    schedule = build_run_schedule(manifest, config)
    repeated = build_run_schedule(copy.deepcopy(manifest), copy.deepcopy(config))

    assert schedule == repeated
    assert len(schedule) == 160
    assert len({row["run_id"] for row in schedule}) == 160
    assert [row["schedule_index"] for row in schedule] == list(range(160))
    assert {row["schema_version"] for row in schedule} == {RUN_SCHEMA_VERSION}
    assert {row["sample_seed"] for row in schedule} == set(SEEDS["sample"])
    assert all(row["perturb_seed"] == row["input"]["perturb_seed"] for row in schedule)

    # Run one pipeline at a time so an interleaved schedule cannot accidentally launch
    # offline and realtime model owners concurrently.  Conditions remain shuffled within
    # each deterministic 80-row block to avoid time/order confounding.
    assert {row["pipeline"] for row in schedule[:80]} == {"offline"}
    assert {row["pipeline"] for row in schedule[80:]} == {"rt"}
    for block in (schedule[:80], schedule[80:]):
        conditions = [row["condition"] for row in block]
        assert conditions != sorted(conditions)
        assert sum(left != right for left, right in zip(conditions, conditions[1:])) >= 20

    assert Counter(row["pipeline"] for row in schedule) == {"offline": 80, "rt": 80}
    assert Counter(row["condition"] for row in schedule) == {
        "sham": 20,
        "pitch": 40,
        "onset": 40,
        "both": 40,
        "high": 20,
    }
    per_input = Counter(row["input_stem"] for row in schedule)
    assert set(per_input.values()) == {4}  # two pipelines x two sample seeds
    for stem in per_input:
        assert {
            (row["pipeline"], row["sample_seed"])
            for row in schedule
            if row["input_stem"] == stem
        } == {
            (pipeline, sample_seed)
            for pipeline in ("offline", "rt")
            for sample_seed in SEEDS["sample"]
        }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="wrong"),
        lambda value: value.update(status="draft"),
        lambda value: value["seeds"]["sample"].__setitem__(0, 999),
        lambda value: value["conditions"]["both"].update(onset_probability=0.25),
        lambda value: value["run_counts"].update(model_runs=159),
        lambda value: value["runtime"].update(playback_tempo=45),
        lambda value: value["runtime"].update(tail_beats=12),
        lambda value: value["runtime"].update(deterministic_boundary_generation=False),
        lambda value: value["sampling"].update(top_k=49),
        lambda value: value["perturbation"].update(collision_attempts=4),
        lambda value: value["validity"]["retry"].update(content_failure_max_retries=2),
        lambda value: value["estimands"].update(primary_quality="changed"),
        lambda value: value["listening"].update(render_bpm=60),
        lambda value: value["checkpoint"].update(sha256=""),
        lambda value: value["input_manifest"].update(sha256=""),
        lambda value: value.pop("attestation"),
        lambda value: value["attestation"].pop("environment"),
        lambda value: value.pop("qualification_result"),
        lambda value: value["qualification"].update(tempo_candidates=[60]),
        lambda value: value["qualification"].update(dense_song="1"),
        lambda value: value["qualification"].update(tail_songs=["4", "2"]),
        lambda value: value["qualification"].update(tail_songs=["2", "3"]),
    ],
)
def test_campaign_config_rejects_mutated_frozen_fields(mutate):
    config = _campaign_config()
    mutate(config)

    with pytest.raises(ValueError):
        validate_campaign_config(config)


def test_default_campaign_configs_do_not_alias_each_other_or_global_constants():
    expected_seeds = copy.deepcopy(SEEDS)
    expected_conditions = copy.deepcopy(CONDITION_TABLE)
    first = _raw_campaign_config()
    second = _raw_campaign_config()

    try:
        first["seeds"]["sample"][0] = -1
        first["conditions"]["both"]["pitch_probability"] = 1.0

        assert SEEDS == expected_seeds
        assert CONDITION_TABLE == expected_conditions
        assert second["seeds"] == expected_seeds
        assert second["conditions"] == expected_conditions
        validate_campaign_config(second)
    finally:
        SEEDS.clear()
        SEEDS.update(expected_seeds)
        CONDITION_TABLE.clear()
        CONDITION_TABLE.update(expected_conditions)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda entries: entries.pop(), "exactly 40"),
        (lambda entries: entries[1].update(stem=entries[0]["stem"]), "unique"),
        (lambda entries: entries[0].update(condition="unknown"), "unknown condition"),
        (lambda entries: entries[0].update(song=None, source_stem=None), "song"),
        (lambda entries: entries[1].update(perturb_seed=999), "perturb.?seed"),
        (
            lambda entries: entries[2].update(perturb_seed=entries[1]["perturb_seed"]),
            "expected perturb seeds",
        ),
        (lambda entries: entries[0].update(perturb_seed=SEEDS["perturb"][0]), "sham"),
        (lambda entries: entries[7].update(perturb_seed=SEEDS["perturb"][1]), "high"),
        (lambda entries: entries[1].update(pitch_probability=0.75), "pitch_probability"),
        (lambda entries: entries[1].update(onset_probability=0.75), "onset_probability"),
    ],
)
def test_input_manifest_rejects_cardinality_identity_seed_and_dose_mutations(mutation, message):
    manifest = _input_manifest()
    mutation(manifest["entries"])

    with pytest.raises(ValueError, match=message):
        validate_input_manifest(manifest)


def test_input_manifest_rejects_a_missing_song_even_if_it_forms_an_eight_entry_group():
    manifest = _input_manifest()
    for entry in manifest["entries"]:
        if entry["song"] == SONGS[-1]:
            entry["song"] = None
            entry["source_stem"] = None

    with pytest.raises(ValueError, match="song"):
        validate_input_manifest(manifest)


def test_exact_artifact_gate_accepts_only_the_declared_file_set(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mid").write_bytes(b"midi")

    require_exact_artifacts(tmp_path, ["a.json", "nested/b.mid"])

    (tmp_path / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="artifact set mismatch"):
        require_exact_artifacts(tmp_path, ["a.json", "nested/b.mid"])

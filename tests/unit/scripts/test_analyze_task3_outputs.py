from experiments.lekai_cross_style.analyze_task3_outputs import (
    _aggregate_by_style_and_decoder,
    _validate_manifest,
)


def test_validate_task3_manifest_accepts_frozen_sampling_setting():
    fingerprint = "frozen"
    manifest = {
        "seed": 1,
        "selection_fingerprint_sha1": fingerprint,
        "items": [{} for _ in range(20)],
        "generation": {
            "gt_prefix_beats": 8,
            "temperature": 1.1,
            "top_k": 0,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "use_fp16": True,
            "export_gt_midi": True,
        },
    }

    _validate_manifest(
        manifest,
        decoder="sampling",
        seed=1,
        expected_fingerprint=fingerprint,
    )


def test_validate_task3_manifest_accepts_greedy_control():
    manifest = {
        "seed": 0,
        "selection_fingerprint_sha1": "frozen",
        "items": [{} for _ in range(20)],
        "generation": {
            "gt_prefix_beats": 8,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "use_fp16": True,
            "export_gt_midi": True,
        },
    }

    _validate_manifest(
        manifest,
        decoder="greedy",
        seed=0,
        expected_fingerprint="frozen",
    )


def test_aggregate_task3_metrics_keeps_style_as_unit():
    common = {
        "acc_note_density_per_beat": 1.0,
        "pitch_class_entropy": 2.0,
        "dominant_pitch_class_rate": 0.5,
        "median_duration_beats": 1.0,
        "long_note_rate_gt_8_beats": 0.0,
        "piece_end_note_rate": 0.0,
        "empty_bar_rate": 0.25,
        "unique_nonempty_bar_ratio": 0.75,
        "adjacent_exact_nonempty_bar_repeat_rate": 0.1,
        "max_empty_bar_run": 2,
        "output_empty": False,
    }
    rows = [
        {
            **common,
            "style": "classical",
            "decoder": "sampling",
            "piece_id": "1",
            "seed": seed,
        }
        for seed in range(3)
    ] + [
        {
            **common,
            "style": "pop",
            "decoder": "sampling",
            "piece_id": "2",
            "seed": seed,
        }
        for seed in range(3)
    ]

    result = _aggregate_by_style_and_decoder(rows)

    assert [row["style"] for row in result] == ["classical", "pop"]
    assert all(row["decoder"] == "sampling" for row in result)
    assert all(row["sample_count"] == 3 for row in result)
    assert all(row["piece_count"] == 1 for row in result)
    assert all(row["seed_count"] == 3 for row in result)

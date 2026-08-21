"""Tests for JSON-ready rap monitoring payloads."""

from __future__ import annotations

import json

from streammuse.application.rap.monitoring_payloads import (
    bounded_chunk_event_payload,
    flow_template_payload,
    scheduled_syllables_payload,
)
from streammuse.domain.rap import BeatSlot, ScheduledSyllable, Syllable
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def test_flow_template_payload_preserves_every_alignment_field() -> None:
    payload = flow_template_payload(BUILTIN_TEMPLATES.get("baseline_syncopated_9"))

    assert payload["template_id"] == "baseline_syncopated_9"
    assert payload["ticks_per_beat"] == 4
    assert payload["beats_per_bar"] == 4
    assert [slot["tick_in_bar"] for slot in payload["slots"]] == [0, 2, 3, 5, 7, 8, 10, 13, 15]
    assert payload["slots"][-1]["boundary_strength"] == 3
    assert payload["slots"][-1]["rhyme_group"] == "A"
    json.dumps(payload)


def test_scheduled_syllables_payload_uses_relative_alignment_fields() -> None:
    scheduled = (
        ScheduledSyllable(
            BeatSlot(bar=2, tick=35, beat=0, tick_in_beat=3, accent=0.8, slot_index=4),
            Syllable("cosmic", 1, 2, 1),
        ),
    )

    assert scheduled_syllables_payload(scheduled, bar=2) == [
        {
            "slot_index": 4,
            "tick_in_bar": 3,
            "target_stress": 0.8,
            "label": ".",
            "word": "cosmic",
            "stress": 1,
            "stressed": True,
        }
    ]


def test_chunk_event_payload_keeps_bounded_research_evidence_without_artifact_bodies() -> None:
    payload = {
        "state": "returned",
        "renderer_decision": "moss_aligned_remote",
        "chunk_index": 4,
        "bars": [8, 9, 10],
        "selected_lines": ["First selected line", "Second selected line", "unbounded third line"],
        "flows": [
            {
                "template_id": "flow-a",
                "name": "First flow",
                "ticks_per_beat": 4,
                "beats_per_bar": 4,
                "slots": [
                    {"tick_in_bar": 0, "target_stress": 1.0, "duration_ticks": 2},
                    {"tick_in_bar": 2, "target_stress": 0.25, "duration_ticks": 1},
                ],
                "character_spans": [{"start": 0, "end": 4}],
            },
            {
                "template_id": "flow-b",
                "name": "Second flow",
                "ticks_per_beat": 4,
                "beats_per_bar": 4,
                "slots": [{"tick_in_bar": 1, "target_stress": 0.8, "duration_ticks": 2}],
            },
        ],
        "prompt_summary": "system: clean rap | user: two exact schedules",
        "context_lines": ["Committed one", "Committed two"],
        "request_budget_ms": 5_000,
        "elapsed_ms": 4_820.5,
        "deadline_slack_ms": 179.5,
        "transfer": {"total_ms": 21.5, "response_bytes": 262_144},
        "mac_validation_mix_ms": 7.25,
        "hashes": {"package_sha256": "package-hash"},
        "artifact_refs": {
            "manifest": "/h200/request-4/manifest.json",
            "candidate_ledger": "/h200/request-4/candidates.jsonl",
        },
        "failure_reason": None,
        "manifest": {
            "selected_bars": [
                {
                    "bar": 8,
                    "text": "First selected line",
                    "score": 0.91,
                    "diagnostics": {
                        "component_scores": {"stress_alignment": 0.88, "topic_coverage": 0.74}
                    },
                    "scheduled": [{"full": "schedule ledger"}],
                },
                {
                    "bar": 9,
                    "text": "Second selected line",
                    "score": 0.87,
                    "diagnostics": {"component_scores": {"continuity": 0.82}},
                },
            ],
            "vocal_sha256": "vocal-hash",
            "diagnostics": {
                "candidate_stats": {
                    "requested_count": 64,
                    "parseable_count": 60,
                    "valid_count": 11,
                    "selectable_count": 6,
                    "top_candidates": [{"full": "candidate ledger"}],
                    "rejections": [{"full": "rejection ledger"}],
                },
                "stage_timings_ms": {
                    "generation": 1_100.0,
                    "evaluation": 90.0,
                    "moss": 2_500.0,
                    "aligner": 45.0,
                    "warp": 180.0,
                    "packaging": 12.0,
                    "total": 3_927.0,
                },
                "alignment_diagnostics": {
                    "method": "mms_forced_alignment",
                    "confidence": 0.94,
                    "fallback_counts": {"transcript_proportional": 1, "unmatched_words": 1},
                    "source_anchors": [0.1, 0.3, 0.5],
                    "target_anchors": [0.0, 0.25, 0.5],
                    "character_spans": [{"start": 0, "end": 4}],
                    "local_warp_ratios": [0.82, 1.21],
                },
                "warnings": ["stretch_ratio_high:1.21", "alignment_fallback:one_word"],
            },
        },
        "raw_wav": b"RIFF-not-for-the-event-bus",
        "candidate_ledger": [{"unbounded": True}],
        "character_spans": [{"start": 0, "end": 4}],
    }

    bounded = bounded_chunk_event_payload(payload)

    assert bounded["bars"] == [8, 9]
    assert bounded["selected_lines"] == ["First selected line", "Second selected line"]
    assert [flow["template_id"] for flow in bounded["flows"]] == ["flow-a", "flow-b"]
    assert bounded["flows"][0]["slot_stress_schedule"] == "t0@1.00, t2@0.25"
    assert bounded["candidate_counts"] == {
        "requested": 64,
        "parseable": 60,
        "valid": 11,
        "selectable": 6,
    }
    assert bounded["selected_scores"] == [
        {
            "bar": 8,
            "total": 0.91,
            "component_scores": {"stress_alignment": 0.88, "topic_coverage": 0.74},
        },
        {"bar": 9, "total": 0.87, "component_scores": {"continuity": 0.82}},
    ]
    assert bounded["prompt_summary"] == "system: clean rap | user: two exact schedules"
    assert bounded["context_lines"] == ["Committed one", "Committed two"]
    assert bounded["stage_timings_ms"] == {
        "generation": 1_100.0,
        "evaluation": 90.0,
        "moss": 2_500.0,
        "aligner": 45.0,
        "r3": 180.0,
        "package": 12.0,
        "transfer": 21.5,
        "mac": 7.25,
        "total": 3_927.0,
    }
    assert bounded["alignment"] == {
        "method": "mms_forced_alignment",
        "confidence": 0.94,
        "fallback_counts": {"transcript_proportional": 1, "unmatched_words": 1},
    }
    assert bounded["stretch_warnings"] == ["stretch_ratio_high:1.21"]
    assert bounded["hashes"] == {"package_sha256": "package-hash", "vocal_sha256": "vocal-hash"}
    assert bounded["artifact_refs"]["candidate_ledger"].endswith("candidates.jsonl")
    assert bounded["transfer_bytes"] == 262_144
    encoded = json.dumps(bounded)
    for forbidden in ("RIFF-not-for-the-event-bus", '"unbounded"', "character_spans", "source_anchors", "target_anchors"):
        assert forbidden not in encoded

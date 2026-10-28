"""Tests for JSON-ready rap monitoring payloads."""

from __future__ import annotations

import json
from dataclasses import replace
from collections.abc import Iterator, Mapping

import pytest

from streammuse.application.rap.monitoring_payloads import (
    MAX_BOUNDED_CHUNK_EVENT_BYTES,
    bounded_chunk_event_payload,
    flow_template_payload,
    scheduled_syllables_payload,
)
from streammuse.domain.rap import (
    BeatSlot,
    REMOTE_CHUNK_ARTIFACT_IDS,
    ScheduledSyllable,
    Syllable,
)
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


@pytest.mark.parametrize(("chord_loop", "chord_gain"), [("off", 0.0), ("minor_cycle", 0.25), ("major_cycle", 0.4)])
def test_prepared_timing_payload_distinguishes_warp_estimate_cursor_and_unavailable_acoustics(chord_loop, chord_gain):
    from streammuse.application.rap.monitoring_payloads import prepared_bar_timing_payload
    from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig, RenderedOnsetEstimate

    prepared = PreparedRapBar(
        0, "one", "moss", None, (), PcmAudio(AudioFormat(), 48_000, bytes(48_000 * 8)), (), (), 1.0,
        mix_config=RapMixConfig(revision=2, chord_loop=chord_loop, chord_gain=chord_gain),
        onset_estimates=(RenderedOnsetEstimate(0, 12_000, 14_400, 0.1, 0.7, "source_alignment_warp_estimate"),),
    )
    payload = prepared_bar_timing_payload(prepared, playback_frame_in_bar=24_000)

    assert payload["rendered_estimate_status"] == "available"
    assert payload["requested_onsets"][0]["sample"] == 12_000
    estimate = payload["rendered_onset_estimates"][0]
    assert estimate["estimated_sample"] == 14_400
    assert estimate["drift_ms"] == 50.0
    assert estimate["confidence"] == 0.7
    assert estimate["method"] == "source_alignment_warp_estimate"
    assert payload["playback_cursor"] == {"sample": 24_000, "seconds": 0.5, "kind": "device_or_software_clock", "is_acoustic_observation": False}
    assert payload["acoustic_onset_status"] == "unavailable"
    assert payload["acoustic_onsets"] is None
    assert payload["mix"]["revision"] == 2
    assert payload["mix"].get("chord_loop") == chord_loop
    assert payload["mix"].get("chord_gain") == chord_gain
    assert "PCM" not in json.dumps(payload)

    unavailable = prepared_bar_timing_payload(replace(prepared, onset_estimates=()))
    assert unavailable["rendered_estimate_status"] == "unavailable"
    assert unavailable["rendered_onset_estimates"] == []
    assert unavailable["playback_cursor"] is None


def test_prepared_timing_payload_keeps_missing_estimates_and_rejects_nonfinite_values():
    from streammuse.application.rap.monitoring_payloads import prepared_bar_timing_payload
    from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RenderedOnsetEstimate

    prepared = PreparedRapBar(
        0, "one", "test", None, (), PcmAudio(AudioFormat(), 1, bytes(8)), (), (), 1.0,
        onset_estimates=(RenderedOnsetEstimate(0, 0, unavailable_reason="no alignment"), RenderedOnsetEstimate(1, 1, float("nan"))),
    )
    payload = prepared_bar_timing_payload(prepared)
    assert payload["rendered_estimate_status"] == "unavailable"
    assert payload["rendered_onset_estimates"][0]["unavailable_reason"] == "no alignment"
    assert all(item["estimated_sample"] is None for item in payload["rendered_onset_estimates"])
    assert all(item["drift_ms"] is None for item in payload["rendered_onset_estimates"])
    json.dumps(payload, allow_nan=False)


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
            "display_label": "-mic",
            "display_index": 1,
            "display_count": 2,
            "stress": 1,
            "stressed": True,
        }
    ]


@pytest.mark.parametrize(("word", "count", "labels"), [
    ("whisper", 2, ["whis-", "-per"]),
    ("beat", 1, ["beat"]),
    ("banana", 3, ["ba-", "-na-", "-na"]),
    ("can't", 1, ["can't"]),
    ("zorbax", 2, ["zor-", "-bax"]),
    ("whisper", 1, ["whisper"]),
])
def test_scheduled_display_fragments_leave_words_phonemes_and_labels_unchanged(word, count, labels):
    scheduled = tuple(
        ScheduledSyllable(
            BeatSlot(2, 32 + index * 2, 0, index * 2, 0.8, slot_index=index),
            Syllable(word, index, count, 1, phonemes=("W", "IH1")),
        )
        for index in range(count)
    )

    payload = scheduled_syllables_payload(scheduled, bar=2)

    assert [item.get("display_label") for item in payload] == labels
    assert [item["display_index"] for item in payload] == list(range(count))
    assert [item["display_count"] for item in payload] == [count] * count
    assert [item["word"] for item in payload] == [word] * count
    assert [item["label"] for item in payload] == [word] + ["."] * (count - 1)
    assert [item["tick_in_bar"] for item in payload] == list(range(0, count * 2, 2))
    assert all(item.syllable.phonemes == ("W", "IH1") for item in scheduled)
    json.dumps(payload)


@pytest.mark.parametrize(("word", "count"), [("whisper", 3), ("banana", 2), ("rhythmx", 3), ("o'clock", 2)])
def test_display_count_mismatch_keeps_all_letters_and_scheduled_slots(word, count):
    scheduled = tuple(
        ScheduledSyllable(BeatSlot(0, index, 0, index, 0.5), Syllable(word, index, count, 0))
        for index in range(count)
    )
    payload = scheduled_syllables_payload(scheduled, bar=0)
    labels = [item.get("display_label") for item in payload]

    assert all(isinstance(label, str) and label.strip("-") for label in labels)
    assert "".join(label.strip("-") for label in labels) == word
    assert len(labels) == count
    assert labels[0].endswith("-")
    assert labels[-1].startswith("-")
    assert scheduled_syllables_payload(scheduled, bar=0) == payload


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


class _ScanBombMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "safe":
            return 1
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "safe"
        for index in range(8):
            yield f"extra-{index}"
        raise AssertionError("bounded projection scanned past its item limit")

    def __len__(self) -> int:
        return 10**100_000


class _ReadBombMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"hostile read: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("hostile iteration")

    def __len__(self) -> int:
        raise AssertionError("hostile length")


def test_chunk_event_payload_handles_hostile_numbers_and_mappings_without_raising() -> None:
    huge = 10**100_000
    payload = {
        "chunk_index": huge,
        "bars": [huge, 2, 3],
        "request_budget_ms": huge,
        "elapsed_ms": huge,
        "deadline_slack_ms": -huge,
        "stage_timings_ms": {"total": huge},
        "alignment": {
            "confidence": huge,
            "fallback_counts": _ScanBombMapping(),
        },
        "hashes": _ScanBombMapping(),
        "artifact_refs": _ScanBombMapping(),
    }

    bounded = bounded_chunk_event_payload(payload)

    assert bounded["chunk_index"] is None
    assert bounded["bars"] == [2]
    assert bounded["request_budget_ms"] is None
    assert bounded["elapsed_ms"] is None
    assert bounded["deadline_slack_ms"] is None
    assert bounded["stage_timings_ms"] == {}
    assert bounded["alignment"]["confidence"] is None
    assert len(json.dumps(bounded).encode("utf-8")) <= MAX_BOUNDED_CHUNK_EVENT_BYTES

    read_bomb = bounded_chunk_event_payload(_ReadBombMapping())
    assert read_bomb["state"] is None
    assert len(json.dumps(read_bomb).encode("utf-8")) <= MAX_BOUNDED_CHUNK_EVENT_BYTES


def test_chunk_event_payload_preserves_all_versioned_artifact_references() -> None:
    bounded = bounded_chunk_event_payload(
        {"artifact_refs": REMOTE_CHUNK_ARTIFACT_IDS}
    )

    assert bounded["artifact_refs"] == dict(REMOTE_CHUNK_ARTIFACT_IDS)


def test_chunk_event_payload_preserves_zero_budget_and_rejects_negative_durations() -> None:
    bounded = bounded_chunk_event_payload(
        {
            "request_budget_ms": 0,
            "elapsed_ms": -1,
            "deadline_slack_ms": -5,
            "stage_timings_ms": {"generation": -2},
            "transfer": {"total_ms": -3},
            "mac_validation_mix_ms": -4,
            "manifest": {
                "diagnostics": {"accepted_request_budget_ms": 5_000},
            },
        }
    )

    assert bounded["request_budget_ms"] == 0
    assert bounded["elapsed_ms"] is None
    assert bounded["deadline_slack_ms"] == -5
    assert bounded["stage_timings_ms"] == {}


def test_chunk_event_payload_enforces_a_serialized_byte_ceiling_and_bounded_depth() -> None:
    long_text = "x" * 100_000
    payload = {
        "state": long_text,
        "renderer_decision": long_text,
        "selected_lines": [long_text] * 100,
        "flows": [
            {
                "template_id": long_text,
                "name": long_text,
                "slots": [
                    {
                        "tick_in_bar": index,
                        "target_stress": 1.0,
                        "rhyme_group": long_text,
                    }
                    for index in range(1_000)
                ],
            }
        ] * 100,
        "selected_scores": [
            {"component_scores": {f"{long_text}-{index}": 1.0 for index in range(100)}}
        ] * 100,
        "prompt_summary": long_text,
        "context_lines": [long_text] * 100,
        "warnings": [long_text] * 100,
        "stretch_warnings": [long_text] * 100,
        "hashes": {f"{long_text}-{index}": long_text for index in range(100)},
        "artifact_refs": {f"{long_text}-{index}": long_text for index in range(100)},
    }

    bounded = bounded_chunk_event_payload(payload)
    encoded = json.dumps(bounded, separators=(",", ":")).encode("utf-8")

    assert len(encoded) <= MAX_BOUNDED_CHUNK_EVENT_BYTES
    assert max(_json_depth(bounded)) <= 5


def _json_depth(value: object, depth: int = 0) -> list[int]:
    if isinstance(value, dict):
        return [depth, *[item for child in value.values() for item in _json_depth(child, depth + 1)]]
    if isinstance(value, list):
        return [depth, *[item for child in value for item in _json_depth(child, depth + 1)]]
    return [depth]

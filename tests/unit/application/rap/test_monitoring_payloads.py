"""Tests for JSON-ready rap monitoring payloads."""

from __future__ import annotations

import json

from streammuse.application.rap.monitoring_payloads import flow_template_payload, scheduled_syllables_payload
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

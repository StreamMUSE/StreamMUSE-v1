from __future__ import annotations

from streammuse.domain.debug.canonical import (
    canonical_event_payloads,
    hash_jsonable,
    summarize_token_sequence,
)


def test_event_payload_hash_is_stable_for_equivalent_ordering() -> None:
    left = [
        {"type": "note_on", "pitch": 64, "tick": 4, "velocity": 80},
        {"type": "note_off", "pitch": 64, "tick": 8, "velocity": 0},
    ]
    right = [
        {"velocity": 0, "tick": 8, "pitch": 64, "type": "note_off"},
        {"velocity": 80, "tick": 4, "pitch": 64, "type": "note_on"},
    ]

    assert hash_jsonable(canonical_event_payloads(left)) == hash_jsonable(
        canonical_event_payloads(right)
    )


def test_token_summary_reports_first_mismatch() -> None:
    summary = summarize_token_sequence([1, 2, 3, 4], other=[1, 2, 9, 4])

    assert summary["length"] == 4
    assert summary["other_length"] == 4
    assert summary["first_mismatch"] == {"position": 2, "left": 3, "right": 9}

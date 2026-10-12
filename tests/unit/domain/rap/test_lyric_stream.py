from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from streammuse.domain.rap import (
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.domain.rap.lyric_stream import (
    BarAudioManifest,
    BarLyricRecord,
    PlanEdit,
)


def _flow(template_id: str = "two-slot") -> FlowTemplate:
    return FlowTemplate(
        template_id=template_id,
        name=template_id,
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=4, target_stress=1.0),
            FlowSlot(tick_in_bar=8, duration_ticks=4, target_stress=0.5, rhyme_group="A"),
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def _record(**overrides) -> BarLyricRecord:
    flow = overrides.pop("flow_template", _flow())
    bar = overrides.pop("bar", 5)
    scheduled = tuple(
        ScheduledSyllable(slot, Syllable(word, 0, 1, 1, ("AH1",), "test"))
        for slot, word in zip(materialize_flow(flow, bar), ("deep", "sea"), strict=True)
    )
    values = dict(
        session_id="session-1",
        bar=bar,
        revision=0,
        topic="deep sea",
        flow_template=flow,
        text="deep sea",
        scheduled=scheduled,
        score=0.8,
        provenance={"model": "qwen", "candidates": 32},
    )
    values.update(overrides)
    return BarLyricRecord(**values)


def test_lyric_record_round_trips_and_its_id_covers_revision_and_content() -> None:
    record = _record()

    restored = BarLyricRecord.from_payload(json.loads(json.dumps(record.to_payload())))

    assert restored == record
    assert restored.lyric_id == record.lyric_id
    assert _record(revision=1).lyric_id != record.lyric_id
    assert _record(topic="space").lyric_id != record.lyric_id
    # Score and provenance describe the choice; they are not part of the line's identity.
    assert _record(score=0.1, provenance={}).lyric_id == record.lyric_id


def test_lyric_record_rejects_a_forged_id_and_a_schedule_for_another_bar() -> None:
    payload = _record().to_payload()
    payload["lyric_id"] = hashlib.sha256(b"forged").hexdigest()
    with pytest.raises(ValueError, match="does not match"):
        BarLyricRecord.from_payload(payload)

    record = _record()
    with pytest.raises(ValueError, match="fill the flow's slots"):
        replace(record, bar=6)


def test_plan_edit_requires_a_change_and_round_trips() -> None:
    edit = PlanEdit(edit_seq=3, from_bar=12, topic="space", flow_template=_flow("other"))

    assert PlanEdit.from_payload(json.loads(json.dumps(edit.to_payload()))) == edit
    with pytest.raises(ValueError, match="change the topic or the flow"):
        PlanEdit(edit_seq=1, from_bar=2)


def _manifest(**overrides) -> BarAudioManifest:
    values = dict(
        lyric_id="a" * 64,
        bar=5,
        tempo_bpm=90.0,
        frame_count=66_000,
        source_sha256="b" * 64,
        source_onsets=(0.1, 1.4),
        onset_confidence=(0.9, 0.8),
        stage_timings_ms={"moss": 500.0, "aligner": 20.0},
        diagnostics={"fallback_counts": {"word": 0}},
    )
    values.update(overrides)
    return BarAudioManifest(**values)


def test_bar_audio_manifest_round_trips() -> None:
    manifest = _manifest()

    assert BarAudioManifest.from_payload(json.loads(manifest.canonical_json_bytes())) == manifest


@pytest.mark.parametrize(
    "overrides",
    (
        {"lyric_id": "not-a-hash"},
        {"frame_count": 0},
        {"frame_count": 24_000 * 16},
        {"source_onsets": (0.1,), "onset_confidence": (0.9, 0.8)},
        {"source_onsets": (0.1, 9.0)},
        {"onset_confidence": (0.9, 1.5)},
        {"stage_timings_ms": {"moss": -1.0}},
    ),
)
def test_bar_audio_manifest_rejects_inconsistent_values(overrides) -> None:
    with pytest.raises(ValueError):
        _manifest(**overrides)

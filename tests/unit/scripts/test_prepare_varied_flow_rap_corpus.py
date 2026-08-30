from __future__ import annotations

from pathlib import Path

import pytest


def _source_record(bar: int) -> dict:
    stresses = (1, 0, 1, 1, 0, 1, 0, 1, 1)
    words = [f"word{bar}_{index}" for index in range(9)]
    return {
        "bar": bar,
        "song_index": 4,
        "title": "Lights After Midnight",
        "topic": "city nights",
        "text": " ".join(words),
        "normalized_text": " ".join(words),
        "syllable_count": 9,
        "template_id": "baseline_straight_9",
        "schedule": [
            {
                "absolute_tick": (bar * 16) + index,
                "analysis_source": "fixture",
                "lexical_stress": stress,
                "phonemes": ["AA1" if stress else "AH0"],
                "seconds_from_song_start": ((bar * 16) + index) / 6,
                "slot_index": index,
                "syllable_label": word,
                "target_sample_in_bar": index * 8_000,
                "target_stress": float(stress),
                "tick_in_bar": index,
                "word": word,
            }
            for index, (word, stress) in enumerate(zip(words, stresses, strict=True))
        ],
    }


def test_varied_flow_assignment_uses_every_template_per_block_without_adjacent_repeats(load_script) -> None:
    module = load_script("prepare_varied_flow_rap_corpus")
    records = tuple(_source_record(bar) for bar in range(24))
    templates = module.build_varied_templates()

    assignments = module.assign_templates(records, templates=templates, song_index=4)

    assert len(templates) == 12
    assert len({tuple(slot.tick_in_bar for slot in template.slots) for template in templates}) == 12
    assert len({tuple(slot.target_stress for slot in template.slots) for template in templates}) >= 6
    assert len(assignments) == len(records)
    assert len({item.template.template_id for item in assignments[:12]}) == 12
    assert len({item.template.template_id for item in assignments[12:24]}) == 12
    assert all(
        left.template.template_id != right.template.template_id
        for left, right in zip(assignments, assignments[1:])
    )
    assert all(0.0 <= item.stress_alignment <= 1.0 for item in assignments)


def test_retimed_records_build_exact_two_bar_requests(load_script) -> None:
    module = load_script("prepare_varied_flow_rap_corpus")
    records = tuple(_source_record(bar) for bar in range(24))
    templates = module.build_varied_templates()
    assignments = module.assign_templates(records, templates=templates, song_index=4)

    retimed = tuple(
        module.retime_record(record, assignment)
        for record, assignment in zip(records, assignments, strict=True)
    )
    requests = module.build_requests("04_city_nights", retimed, templates=templates)

    assert len(requests) == 12
    assert all(len(request.syllables) == 18 for request in requests)
    assert [request.chunk_index for request in requests] == list(range(12))
    for record, assignment in zip(retimed, assignments, strict=True):
        assert record["template_id"] == assignment.template.template_id
        assert [item["tick_in_bar"] for item in record["schedule"]] == [
            slot.tick_in_bar for slot in assignment.template.slots
        ]
        assert [item["target_stress"] for item in record["schedule"]] == [
            slot.target_stress for slot in assignment.template.slots
        ]


def test_varied_flow_assignment_rejects_non_nine_syllable_source(load_script) -> None:
    module = load_script("prepare_varied_flow_rap_corpus")
    record = _source_record(0)
    record["schedule"] = record["schedule"][:-1]

    with pytest.raises(ValueError, match="exactly nine scheduled syllables"):
        module.assign_templates(
            (record,),
            templates=module.build_varied_templates(),
            song_index=4,
        )

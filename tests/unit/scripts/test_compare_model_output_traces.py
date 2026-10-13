from __future__ import annotations

from scripts.compare_model_output_traces import compare_model_output_records
from streammuse.infrastructure.inference.model_output_trace import token_digest


def _record(beat: int | None, output: list[int], prompt: list[int]) -> dict:
    return {
        "target_beat": beat,
        "output_tokens": output,
        "output_digest": token_digest(output),
        "prompt_digest": token_digest(prompt),
    }


def test_output_match_passes_even_when_prompt_digests_differ():
    report = compare_model_output_records(
        [_record(3, [81, 2, 171], [1, 2])],
        [_record(3, [81, 2, 171], [9, 8])],
    )

    assert report["passed"] is True
    assert report["exact_match_beats"] == [3]
    assert report["prompt_digest_mismatch_count"] == 1
    assert report["prompt_digest_affects_pass"] is False


def test_output_mismatch_reports_exact_beat_and_tokens():
    report = compare_model_output_records(
        [_record(4, [169], [1])],
        [_record(4, [81, 3, 171], [1])],
    )

    assert report["passed"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["target_beat"] == 4
    assert report["mismatches"][0]["streammuse_output_tokens"] == [[169]]

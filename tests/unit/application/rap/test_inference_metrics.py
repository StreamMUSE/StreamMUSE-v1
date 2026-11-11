import pytest

from streammuse.application.rap.inference.metrics import coverage_summary, latency_summary


def test_latency_summary_preserves_unavailable_and_reports_sample_count():
    assert latency_summary([None, float("nan"), float("inf"), True, "12"]) == {
        "count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None,
    }
    result = latency_summary([0, 100, 200, None])
    assert result == {"count": 3, "p50_ms": 100, "p95_ms": 190, "max_ms": 200}


def test_coverage_uses_duration_not_packet_counts_and_excludes_wait():
    result = coverage_summary(eligible_ms=8000, missing_ms=1000, stale_ms=2000,
                              fallback_ms=500, wall_ms=16000)
    assert result["missing_rate"] == 0.125
    assert result["stale_rate"] == 0.25
    assert result["fallback_rate"] == 0.0625
    assert result["on_time_musical_s_per_wall_s"] == 7 / 16
    assert result["on_time_scope"] == "nonmissing audio; not semantic correctness or new-context compliance"


def test_empty_coverage_is_unavailable_not_perfect_success():
    result = coverage_summary(eligible_ms=0, missing_ms=0, stale_ms=0,
                              fallback_ms=0, wall_ms=0)
    assert result["missing_rate"] is None
    assert result["on_time_musical_s_per_wall_s"] is None


@pytest.mark.parametrize("field,value", [("missing_ms", -1), ("eligible_ms", float("nan")),
                                        ("wall_ms", True), ("missing_ms", 8001)])
def test_invalid_coverage_counters_are_not_silently_clamped(field, value):
    inputs = dict(eligible_ms=8000, missing_ms=1000, stale_ms=0, fallback_ms=0, wall_ms=9000)
    inputs[field] = value
    with pytest.raises(ValueError):
        coverage_summary(**inputs)

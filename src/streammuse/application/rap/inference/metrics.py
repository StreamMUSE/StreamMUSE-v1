"""Pure measurement formulas; missing instrumentation is never an observed zero."""

import math


def latency_summary(values):
    samples = sorted(float(value) for value in values
                     if type(value) in (int, float) and math.isfinite(value) and value >= 0)

    def percentile(fraction):
        if not samples:
            return None
        position = (len(samples) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    return {"count": len(samples), "p50_ms": percentile(0.5), "p95_ms": percentile(0.95),
            "max_ms": samples[-1] if samples else None}


def coverage_summary(*, eligible_ms, missing_ms, stale_ms, fallback_ms, wall_ms):
    values = dict(eligible_ms=eligible_ms, missing_ms=missing_ms, stale_ms=stale_ms,
                  fallback_ms=fallback_ms, wall_ms=wall_ms)
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative duration")
    for name in ("missing_ms", "stale_ms", "fallback_ms"):
        if values[name] > eligible_ms:
            raise ValueError(f"{name} exceeds eligible musical coverage")
    return {**values,
            "missing_rate": missing_ms / eligible_ms if eligible_ms else None,
            "stale_rate": stale_ms / eligible_ms if eligible_ms else None,
            "fallback_rate": fallback_ms / eligible_ms if eligible_ms else None,
            "on_time_musical_s_per_wall_s": (eligible_ms - missing_ms) / wall_ms if wall_ms else None,
            "on_time_scope": "nonmissing audio; not semantic correctness or new-context compliance"}

"""CPU-only, single completed-bar history experiment. No model integration."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import median


class InputError(ValueError):
    """The transaction cannot safely be evaluated."""


def _require(condition, message):
    if not condition:
        raise InputError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum,
             name + " must be an integer >= " + str(minimum))
    return value


def _object(value, required, optional=()):
    _require(type(value) is dict, "expected an object")
    _require(set(required) <= value.keys(), "missing fields: " + str(set(required) - value.keys()))
    _require(value.keys() <= set(required) | set(optional), "unknown fields in object")


def _notes(value, cutoff, name):
    _require(type(value) is list, name + " must be a list")
    for note in value:
        _object(note, ("onset", "end", "pitch"))
        onset = _integer(note["onset"], name + ".onset")
        end = _integer(note["end"], name + ".end")
        pitch = _integer(note["pitch"], name + ".pitch")
        _require(onset < end, name + " requires onset < end")
        _require(pitch <= 127, name + " pitch must be <= 127")
        _require(end <= cutoff, name + " contains future or unfinished note data")
    return value


def bar_stats(notes, start, end):
    """Descriptive statistics only; overlap coverage is the interval union."""
    overlapping = [n for n in notes if n["onset"] < end and n["end"] > start]
    intervals = sorted((max(start, n["onset"]), min(end, n["end"]))
                       for n in overlapping)
    covered = 0
    right = start
    for left, stop in intervals:
        covered += max(0, stop - max(left, right))
        right = max(right, stop)
    durations = [stop - left for left, stop in intervals]
    pitches = [n["pitch"] for n in overlapping]
    return {
        "onsets": sum(start <= n["onset"] < end for n in notes),
        "overlap_coverage_steps": covered,
        "overlap_coverage_fraction": covered / (end - start),
        "overlap_duration_median_steps": median(durations) if durations else None,
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "pitch_range_semitones": max(pitches) - min(pitches) if pitches else None,
    }


def run_transaction(document):
    """Validate the entire input, then return an independent provenance record."""
    data = deepcopy(document)
    _object(data, ("steps_per_beat", "bar_start_step", "bar_length_steps",
                   "observed_until", "melody_notes", "history_notes",
                   "played_notes", "policy", "decisions"),
            ("repair_candidate", "repair_fallback"))
    _require(type(data["steps_per_beat"]) is int and data["steps_per_beat"] == 4,
             "steps_per_beat must be integer 4")
    start = _integer(data["bar_start_step"], "bar_start_step")
    length = _integer(data["bar_length_steps"], "bar_length_steps", 1)
    end = start + length
    cutoff = _integer(data["observed_until"], "observed_until")
    _require(cutoff == end, "observed_until must equal this transaction's bar end")
    _require(data["policy"] in ("baseline", "drop", "repair_history"), "unknown policy")
    decisions = data["decisions"]
    _require(type(decisions) is list and len(decisions) == 1,
             "exactly one decision required; duplicate/overlapping boundaries forbidden")
    decision = decisions[0]
    _object(decision, ("bar_start_step", "boundary_step", "decision", "source"))
    _require(_integer(decision["bar_start_step"], "decision.bar_start_step") == start
             and _integer(decision["boundary_step"], "decision.boundary_step") == end,
             "decision must match the explicit bar and boundary")
    _require(decision["decision"] in ("accept", "reject"), "unknown decision")
    _require(type(decision["source"]) is str and decision["source"].strip(),
             "decision source must be a nonempty string")
    melody = _notes(data["melody_notes"], cutoff, "melody_notes")
    history = _notes(data["history_notes"], cutoff, "history_notes")
    played = _notes(data["played_notes"], cutoff, "played_notes")
    fallback = data.get("repair_fallback")
    if data["policy"] == "repair_history" or "repair_fallback" in data:
        _require(fallback in ("keep", "drop", "error"),
                 "repair_fallback must explicitly be keep, drop, or error")
    candidate = data.get("repair_candidate")
    if candidate is not None:
        _object(candidate, ("notes", "source", "decision"))
        _require(candidate["decision"] in ("accept", "reject"),
                 "candidate requires an explicit external accept/reject decision")
        _require(type(candidate["source"]) is str and candidate["source"].strip(),
                 "candidate source must be a nonempty string")
        _notes(candidate["notes"], cutoff, "repair_candidate.notes")
        _require(all(start <= n["onset"] < n["end"] <= end for n in candidate["notes"]),
                 "candidate notes must be wholly inside the selected bar")

    action = "keep"
    reason = "baseline" if data["policy"] == "baseline" else "accepted"
    if decision["decision"] == "reject" and data["policy"] != "baseline":
        action, reason = "drop", "external_rejection"
        if data["policy"] == "repair_history":
            if candidate is not None and candidate["decision"] == "accept":
                action, reason = "replace", "externally_accepted_candidate"
            else:
                unavailable = "missing" if candidate is None else "rejected"
                _require(fallback != "error", "repair candidate " + unavailable + "; fallback=error")
                action, reason = fallback, unavailable + "_candidate_fallback_" + fallback

    result = deepcopy(history)
    if action != "keep":
        # Removing an overlapping sustain would also change context outside this bar.
        overlaps = [n for n in history if n["onset"] < end and n["end"] > start]
        _require(all(start <= n["onset"] and n["end"] <= end for n in overlaps),
                 "history replacement/drop cannot cross a bar boundary")
        result = [deepcopy(n) for n in history
                  if not (n["onset"] < end and n["end"] > start)]
        if action == "replace":
            result.extend(deepcopy(candidate["notes"]))

    original_stats = bar_stats(history, start, end)
    melody_stats = bar_stats(melody, start, end)
    numerator = original_stats["overlap_duration_median_steps"]
    denominator = melody_stats["overlap_duration_median_steps"]
    return {
        "schema_version": 1,
        "input_provenance": data,
        "scope": "HISTORY_ONLY",
        "conceptual_boundary_step": end,
        "repair_timing_simulated": False,
        "action": action,
        "reason": reason,
        "original_note_history": deepcopy(history),
        "resulting_note_history": result,
        "future_context_notes": deepcopy(result),
        "playback_before": deepcopy(played),
        "playback_after": deepcopy(played),
        "statistics": {
            "original_history_bar": original_stats,
            "melody_bar": melody_stats,
            "resulting_history_bar": bar_stats(result, start, end),
            "original_history_to_melody_duration_median_ratio":
                numerator / denominator if numerator is not None and denominator is not None else None,
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise InputError("non-finite JSON number: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="explicit single-bar JSON transaction")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        result = run_transaction(data)
    except (OSError, ValueError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

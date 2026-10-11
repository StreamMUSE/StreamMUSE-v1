"""Controlled snapshot resampling, not a realtime scheduler or quality gate."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

try:
    from .harness import InputError, _integer, _unique_object, _invalid_constant, run_transaction
except ImportError:
    from harness import InputError, _integer, _unique_object, _invalid_constant, run_transaction


def notes_to_events(notes):
    events = []
    for note in notes:
        events.extend([
            {"type": "note_on", "tick": note["onset"], "pitch": note["pitch"], "velocity": 100},
            {"type": "note_off", "tick": note["end"], "pitch": note["pitch"], "velocity": 0},
        ])
    # This note-level probe deliberately canonicalizes ties: off before on.
    return sorted(events, key=lambda e: (e["tick"], e["type"] == "note_on"))


def closed_candidate_notes(events, start, end):
    """Never invent a release for a model-generated active note."""
    active, notes = {}, []
    last_tick = start
    for event in events:
        if not isinstance(event, dict):
            raise InputError("candidate event must be an object")
        tick = _integer(event.get("tick"), "candidate tick")
        pitch = _integer(event.get("pitch"), "candidate pitch")
        if pitch > 127 or tick < last_tick or not start <= tick <= end:
            raise InputError("candidate pitch/time/order out of bounds")
        last_tick = tick
        if event.get("type") == "note_on":
            velocity = _integer(event.get("velocity"), "candidate velocity", 1)
            if velocity > 127 or tick == end or pitch in active:
                raise InputError("candidate onset invalid or same-pitch overlap")
            active[pitch] = tick
        elif event.get("type") == "note_off":
            if pitch not in active or active[pitch] == tick:
                raise InputError("candidate release unpaired or zero-length")
            notes.append({"onset": active.pop(pitch), "end": tick, "pitch": pitch})
        else:
            raise InputError("candidate type must be note_on or note_off")
    if active:
        raise InputError("candidate has active notes at boundary; no synthetic note-off added")
    return sorted(notes, key=lambda n: (n["onset"], n["pitch"]))


def splice_candidate_events(prefix, candidate, start, end, initial_active=None):
    """Keep the immutable pre-bar prefix; carry genuine open notes into the next call."""
    retained = [deepcopy(e) for e in prefix if e["tick"] < start]
    active = {}

    def consume(event, lower, upper, last):
        if type(event) is not dict:
            raise InputError("event must be an object")
        tick = _integer(event.get("tick"), "event tick")
        pitch = _integer(event.get("pitch"), "event pitch")
        if pitch > 127 or tick < last or not lower <= tick <= upper:
            raise InputError("event pitch/time/order invalid")
        value = deepcopy(event)
        if value.get("type") == "note_on":
            # The backend's wire projection uses 100 when velocity is omitted.
            velocity = _integer(value.get("velocity", 100), "velocity", 1)
            if velocity > 127 or pitch in active or tick == end:
                raise InputError("invalid/repeated onset")
            value["velocity"] = velocity
            active[pitch] = tick
        elif value.get("type") == "note_off":
            if pitch not in active or active[pitch] == tick:
                raise InputError("unpaired or zero-length release")
            active.pop(pitch)
        else:
            raise InputError("unsupported event type")
        return value, tick

    previous = 0
    for event in retained:
        _, previous = consume(event, 0, start - 1, previous)
    initial = sorted(active)
    if initial_active is not None and initial != sorted(initial_active):
        raise InputError("candidate trace initial active pitches disagree with history prefix")
    normalized = []
    previous = start
    for event in candidate:
        value, previous = consume(event, start, end, previous)
        normalized.append(value)
    return {
        "scope": "HISTORY_ONLY", "representation": "events_with_active_tail",
        "prefix_events": retained, "candidate_events": normalized,
        "resulting_history_events": retained + normalized,
        "initial_active_pitches": initial,
        "active_pitches_at_end": sorted(active),
        "synthetic_note_off_count": 0,
        "boundary_rule": "retain prefix tick<start; replacement owns events at start",
    }


def preflight(document, repair_seed, continuation_seed):
    _integer(repair_seed, "repair_seed")
    _integer(continuation_seed, "continuation_seed")
    if "repair_candidate" in document:
        raise InputError("remove supplied repair_candidate; this probe generates its own")
    if document.get("policy") != "repair_history":
        raise InputError("snapshot probe requires policy=repair_history")
    checked = deepcopy(document)
    checked["policy"] = "baseline"
    run_transaction(checked)
    start = document["bar_start_step"]
    length = document["bar_length_steps"]
    if start % 4 or length % 4:
        raise InputError("model probe requires whole quarter-note beats (4 steps)")
    if document["decisions"][0]["decision"] != "reject":
        raise InputError("probe requires an explicit fixed intervention target")
    # Fail before any model allocation when a replacement would erase a carry.
    for note in document["history_notes"]:
        if note["onset"] < start < note["end"]:
            raise InputError("history sustains across repair start; unsupported")
    return document


def run_probe(document, probe, *, repair_seed, continuation_seed, apply_repair_history=False):
    preflight(document, repair_seed, continuation_seed)
    original = deepcopy(document)
    start = original["bar_start_step"]
    end = original["observed_until"]
    melody = notes_to_events(original["melody_notes"])
    history = notes_to_events(original["history_notes"])
    prefix = [event for event in history if event["tick"] < start]
    baseline = probe.generate(melody_events=deepcopy(melody), accompaniment_events=history,
                              target_start_step=end, length_steps=4, observed_until=end,
                              seed=continuation_seed)
    repair = probe.generate(melody_events=deepcopy(melody), accompaniment_events=prefix,
                            target_start_step=start, length_steps=end-start, observed_until=end,
                            seed=repair_seed)
    result = {
        "experiment": "isolated_snapshot_bar_resampling",
        "realtime_simulation": False, "automatic_quality_gate": False,
        "input": original, "repair_seed": repair_seed, "continuation_seed": continuation_seed,
        "event_conversion": "velocity=100; note_off before note_on at same tick; not live-event replay",
        "baseline_next_beat": baseline, "repair_bar": repair,
        "playback_before": deepcopy(original["played_notes"]),
        "playback_after": deepcopy(original["played_notes"]),
        "actual_live_cache_updated": False,
    }
    try:
        changed = splice_candidate_events(
            prefix, repair["events"], start, end,
            repair.get("metadata", {}).get("token_decode_initial_active_pitches"))
    except InputError as exc:
        result.update(status="candidate_boundary_unsupported", reason=str(exc))
        return result
    result["candidate_event_state"] = changed
    if not apply_repair_history:
        result["status"] = "candidate_recorded_only"
        return result
    changed["decision_source"] = "explicit fixed intervention; NOT evidence of better musical quality"
    result["history_transaction"] = changed
    result["repaired_next_beat"] = probe.generate(
        melody_events=deepcopy(melody),
        accompaniment_events=deepcopy(changed["resulting_history_events"]),
        target_start_step=end, length_steps=4, observed_until=end, seed=continuation_seed)
    result["status"] = "paired_probe_complete"
    return result


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-seed", type=int, required=True)
    parser.add_argument("--continuation-seed", type=int, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--apply-repair-history", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        document, config = read_json(args.input), read_json(args.config)
        preflight(document, args.repair_seed, args.continuation_seed)
        try:
            from .model_probe import SnapshotProbe
        except ImportError:
            from model_probe import SnapshotProbe
        with SnapshotProbe(config, allow_gpu=args.allow_gpu) as probe:
            result = run_probe(document, probe, repair_seed=args.repair_seed,
                               continuation_seed=args.continuation_seed,
                               apply_repair_history=args.apply_repair_history)
        result["config"] = config
        result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, "error: " + str(exc) + "\n")


if __name__ == "__main__":
    main()

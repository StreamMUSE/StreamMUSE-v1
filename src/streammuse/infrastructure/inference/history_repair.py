"""Strict event-only validation for opt-in, fixed-bar history transactions.

No sampling, playback reconciliation, guard replay, or synthesized releases.
Events use half-open [start, end) ticks, matching continuation prompt carry.
"""

from copy import deepcopy


class HistoryRepairError(ValueError):
    """A stable rejection code, suitable for coordinator audit records."""


def _replay(events, start, end, initial_active=(), duplicate_release_before=0):
    if type(events) is not list:
        raise HistoryRepairError("invalid_events")
    active = {pitch: start - 1 for pitch in initial_active}
    previous = start
    released = {}
    for event in events:
        if type(event) is not dict or not {"type", "pitch", "tick"} <= event.keys():
            raise HistoryRepairError("invalid_events")
        if event.keys() - {"type", "pitch", "tick", "velocity"}:
            raise HistoryRepairError("unsupported_event_fields")
        tick, pitch = event["tick"], event["pitch"]
        if (type(tick) is not int or not start <= tick < end or tick < previous
                or type(pitch) is not int or not 21 <= pitch <= 108):
            raise HistoryRepairError("invalid_event_range_or_order")
        previous = tick
        kind = event["type"]
        velocity = event.get("velocity", 100 if kind == "note_on" else 0)
        if type(velocity) is not int or not 0 <= velocity <= 127:
            raise HistoryRepairError("invalid_velocity")
        if kind == "note_on":
            if not velocity or pitch in active:
                raise HistoryRepairError("invalid_or_repeated_onset")
            active[pitch] = tick
            released.pop(pitch, None)
        elif kind == "note_off":
            # Existing Prompt handoff may repeat a successfully closed release.
            # Only tolerate it in immutable history, never in a candidate bar.
            if (pitch not in active and tick < duplicate_release_before
                    and released.get(pitch) == tick):
                continue
            if pitch not in active or active[pitch] == tick:
                raise HistoryRepairError("unpaired_or_zero_length_release")
            del active[pitch]
            released[pitch] = tick
        else:
            raise HistoryRepairError("unsupported_event_type")
    return set(active)


def prepare_history_repair(history, candidate, start, end, active_pitches, completed_until):
    """Return detached replacement; require identical live decoder carry.

    Cross-start sustains use the immutable prefix carry. End-tick releases
    belong to the unchanged suffix, not to the replacement candidate.
    A matching open tail is supported; its actual played onset is not rewritten.
    """
    original = deepcopy(history)
    replacement = deepcopy(candidate)
    # Validate before indexing so malformed history fails closed too.
    final_active = _replay(original, 0, completed_until, duplicate_release_before=start)
    prefix = [event for event in original if event["tick"] < start]
    initial_active = _replay(prefix, 0, start, duplicate_release_before=start)
    if final_active != set(active_pitches):
        raise HistoryRepairError("backend_active_state_mismatch")
    original_bar = [event for event in original if start <= event["tick"] < end]
    original_tail = _replay(original_bar, start, end, initial_active)
    candidate_tail = _replay(replacement, start, end, initial_active)
    if candidate_tail != original_tail:
        raise HistoryRepairError("active_tail_mismatch")
    suffix = [event for event in original if event["tick"] >= end]
    result = prefix + replacement + suffix
    if _replay(result, 0, completed_until, duplicate_release_before=start) != final_active:
        raise HistoryRepairError("active_suffix_mismatch")
    return result

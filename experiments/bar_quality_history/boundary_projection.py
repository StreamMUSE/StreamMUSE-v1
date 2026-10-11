"""Optional partial repair: retain original notes crossing either boundary.

Never changes a note's pitch, onset, offset, or velocity. Candidate notes must
be wholly contained in the half-open bar and cannot overlap protected notes.
The resulting events are NOT the unmodified sampled token decode.
"""
from copy import deepcopy

from streammuse.infrastructure.inference.history_repair import (
    HistoryRepairError, _replay, prepare_history_repair,
)


def _pairs(events, initial_active=()):
    active = {p: None for p in initial_active}
    pairs = []
    for i, event in enumerate(events):
        pitch = event['pitch']
        if event['type'] == 'note_on':
            active[pitch] = i
        elif pitch in active:
            pairs.append((pitch, active.pop(pitch), i))
    pairs.extend((pitch, index, None) for pitch, index in active.items())
    return pairs


def preserve_cross_boundary(ticket, candidate):
    start, end = ticket['start_tick'], ticket['end_tick']
    completed = ticket['completed_until']
    history = deepcopy(ticket['accompaniment_events'])
    sampled = deepcopy(candidate['events'])
    final = _replay(history, 0, completed, duplicate_release_before=start)
    prefix = [e for e in history if e['tick'] < start]
    initial = _replay(prefix, 0, start, duplicate_release_before=start)
    original_bar = [e for e in history if start <= e['tick'] < end]
    prepare_history_repair(history, original_bar, start, end, final, completed)
    _replay(sampled, start, end, initial)
    protected, keep_indices = [], set()
    for pitch, on, off in _pairs(history):
        if on is None:
            raise HistoryRepairError('missing_original_onset')
        onset = history[on]['tick']
        offset = history[off]['tick'] if off is not None else float('inf')
        if onset < end and offset >= start and (onset < start or offset >= end):
            protected.append((pitch, onset, offset))
            keep_indices.update(i for i in (on, off) if i is not None and start <= history[i]['tick'] < end)
    retained = [history[i] for i in sorted(keep_indices)]
    selected, dropped_crossing, dropped_overlap = [], 0, 0
    for pitch, on, off in _pairs(sampled, initial):
        if on is None or off is None:
            dropped_crossing += 1
            continue
        onset, offset = sampled[on]['tick'], sampled[off]['tick']
        if any(pitch == p and onset < finish and begin < offset for p, begin, finish in protected):
            dropped_overlap += 1
            continue
        selected.extend((sampled[on], sampled[off]))
    if not selected:
        raise HistoryRepairError('no_compatible_interior_notes')
    # Release before retrigger at an equal tick; no event time is modified.
    projected = sorted(retained + selected, key=lambda e: (e['tick'], e['type'] == 'note_on', e['pitch']))
    prepare_history_repair(history, projected, start, end, final, completed)
    if projected == original_bar:
        raise HistoryRepairError('projection_has_no_change')
    result = deepcopy(candidate)
    result['events'] = projected
    # Original sampled token trace stays in provenance, not labeled as projected.
    result['token_decode_beats'] = []
    result.setdefault('provenance', {})['boundary_projection'] = dict(
        policy='preserve_cross_boundary', sampled_events=sampled,
        sampled_token_decode_beats=deepcopy(candidate['token_decode_beats']),
        preserved_original_events=retained, selected_interior_notes=len(selected) // 2,
        discarded_crossing_notes=dropped_crossing, discarded_overlapping_notes=dropped_overlap,
        projected_events_are_original_token_decode=False,
    )
    return result

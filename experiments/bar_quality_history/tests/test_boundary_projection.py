from copy import deepcopy
import unittest

from experiments.bar_quality_history.boundary_projection import preserve_cross_boundary
from streammuse.infrastructure.inference.history_repair import HistoryRepairError


def on(tick, pitch=60):
    return dict(type='note_on', pitch=pitch, tick=tick, velocity=90)


def off(tick, pitch=60):
    return dict(type='note_off', pitch=pitch, tick=tick, velocity=0)


def ticket(history):
    return dict(start_tick=16, end_tick=32, completed_until=40, accompaniment_events=history)


def candidate(events):
    return dict(events=events, token_decode_beats=[{'raw_tokens': [1, 2]}], provenance={})


class BoundaryProjectionTest(unittest.TestCase):
    def test_preserve_both_boundaries_without_mutating_sources(self):
        t = ticket([on(12), off(20), on(22, 65), off(24, 65), on(28, 62), off(36, 62)])
        c = candidate([off(16), on(18), off(22), on(20, 67), off(24, 67), on(28, 68)])
        # Events supplied to the worker must already be time ordered.
        c['events'].sort(key=lambda e: e['tick'])
        old_t, old_c = deepcopy(t), deepcopy(c)
        r = preserve_cross_boundary(t, c)
        self.assertEqual(r['events'], [off(20), on(20, 67), off(24, 67), on(28, 62)])
        self.assertEqual(t, old_t)
        self.assertEqual(c, old_c)
        self.assertEqual(r['token_decode_beats'], [])
        p = r['provenance']['boundary_projection']
        self.assertEqual(p['selected_interior_notes'], 1)
        self.assertEqual(p['discarded_overlapping_notes'], 1)
        self.assertEqual(p['discarded_crossing_notes'], 2)
        self.assertEqual(p['sampled_events'], c['events'])

    def test_release_at_end_is_preserved_in_original_suffix(self):
        t = ticket([on(28, 62), off(32, 62)])
        r = preserve_cross_boundary(t, candidate([on(16, 65), off(20, 65)]))
        self.assertEqual(r['events'], [on(16, 65), off(20, 65), on(28, 62)])

    def test_same_pitch_touching_but_not_overlapping(self):
        t = ticket([on(12), off(16), on(28), off(36)])
        r = preserve_cross_boundary(t, candidate([off(16), on(16), off(28)]))
        self.assertEqual(r['events'], [off(16), on(16), off(28), on(28)])

    def test_candidate_overlap_with_right_boundary_is_removed(self):
        t = ticket([on(28), off(36)])
        r = preserve_cross_boundary(t, candidate([on(20), on(20, 65), off(24, 65), off(30)]))
        self.assertEqual(r['events'], [on(20, 65), off(24, 65), on(28)])

    def test_zero_compatible_notes_rejected(self):
        with self.assertRaisesRegex(HistoryRepairError, 'no_compatible_interior_notes'):
            preserve_cross_boundary(ticket([]), candidate([on(28)]))

    def test_no_change_rejected(self):
        events = [on(16), off(20)]
        with self.assertRaisesRegex(HistoryRepairError, 'projection_has_no_change'):
            preserve_cross_boundary(ticket(events), candidate(events))

    def test_malformed_candidate_not_silently_filtered(self):
        for events in ([off(16)], [on(16), off(16)], [on(16), on(20)],
                       [on(20), off(19)], [on(16), off(32)]):
            with self.subTest(events=events), self.assertRaises(HistoryRepairError):
                preserve_cross_boundary(ticket([]), candidate(events))

    def test_original_prefix_duplicates_preserved(self):
        t = ticket([on(0), off(4), off(4)])
        self.assertEqual(preserve_cross_boundary(t, candidate([on(16, 65), off(20, 65)]))['events'],
                         [on(16, 65), off(20, 65)])


if __name__ == '__main__':
    unittest.main()

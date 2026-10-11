import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.bar_quality_history.analyze_offline_paired import describe, clipped
from experiments.bar_quality_history.run_offline_paired import export_midi


def note(pitch, start, end):
    return SimpleNamespace(pitch=pitch, start=start, end=end, velocity=80)


class OfflineMetricTests(unittest.TestCase):
    def test_empty_retained(self):
        r = describe([], 0, 8, .25)
        self.assertEqual(r['sounding_time_coverage'], 0)
        self.assertEqual(r['nonempty'], 0)
        self.assertIsNone(r['adjacent_exact_onset_pattern_repeat'])

    def test_union_not_sum_and_clipping(self):
        r = describe([note(60, -1, 2), note(64, 1, 3), note(67, 5, 9)], 0, 8, .25)
        self.assertAlmostEqual(r['sounding_time_coverage'], .75)
        self.assertEqual(r['onset_count'], 2)

    def test_repetition_excludes_silent_pairs(self):
        r = describe([note(60, 0, .5), note(60, 1, 1.5), note(62, 3, 3.5)], 0, 4, .25)
        self.assertEqual(r['adjacent_nonempty_pairs'], 1)
        self.assertEqual(r['adjacent_exact_onset_pattern_repeat'], 1)

    def test_primary_window_clips_carried_note(self):
        r = clipped([note(60, 0, 3), note(62, 4, 5)], 2, 4)
        self.assertEqual(len(r), 1)
        self.assertEqual((r[0].start, r[0].end), (2, 3))

    def test_export_terminal_closure_does_not_change_events(self):
        import mido
        events = [{'type': 'note_on', 'pitch': 60, 'tick': 1, 'velocity': 80}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.mid'
            audit = export_midi(path, [], events, 8, 90)
            self.assertEqual(len(events), 1)
            self.assertEqual(audit['Accompaniment']['terminal_note_off_pitches'], [60])
            absolute, found = 0, []
            for e in mido.MidiFile(path).tracks[2]:
                absolute += e.time
                if e.type in ('note_on', 'note_off'):
                    found.append((e.type, absolute // 120))
            self.assertEqual(found, [('note_on', 1), ('note_off', 8)])


if __name__ == '__main__':
    unittest.main()

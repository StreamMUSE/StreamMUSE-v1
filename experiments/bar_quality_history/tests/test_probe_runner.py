from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import InputError
from probe_runner import closed_candidate_notes, notes_to_events, run_probe, splice_candidate_events


class RecordingProbe:
    def __init__(self, open_candidate=False):
        self.calls = []
        self.open_candidate = open_candidate

    def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        tick = kwargs["target_start_step"]
        events = [{"type": "note_on", "tick": tick, "pitch": 60, "velocity": 90}]
        if not (tick == 16 and self.open_candidate):
            events.append({"type": "note_off", "tick": tick + 3, "pitch": 60})
        return {"events": events, "elapsed_seconds": 0.0, "metadata": {}, "runtime": {"mock": True}}


class ProbeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example.synthetic.json").read_text())
        self.data.pop("repair_candidate")

    def run_case(self, probe=None, apply=True):
        return run_probe(self.data, probe or RecordingProbe(), repair_seed=71,
                         continuation_seed=99, apply_repair_history=apply)

    def test_three_calls_use_independent_repair_and_matched_continuation_seed(self):
        probe = RecordingProbe()
        before = deepcopy(self.data)
        result = self.run_case(probe)
        self.assertEqual(result["status"], "paired_probe_complete")
        self.assertEqual([c["seed"] for c in probe.calls], [99, 71, 99])
        self.assertEqual([c["target_start_step"] for c in probe.calls], [32, 16, 32])
        self.assertEqual([c["length_steps"] for c in probe.calls], [4, 16, 4])
        self.assertEqual(probe.calls[0]["melody_events"], probe.calls[2]["melody_events"])
        self.assertNotEqual(probe.calls[0]["accompaniment_events"], probe.calls[2]["accompaniment_events"])
        self.assertEqual(result["playback_before"], result["playback_after"])
        self.assertEqual(self.data, before)

    def test_observe_does_not_apply_candidate(self):
        probe = RecordingProbe()
        result = self.run_case(probe, apply=False)
        self.assertEqual(len(probe.calls), 2)
        self.assertEqual(result["status"], "candidate_recorded_only")
        self.assertNotIn("repaired_next_beat", result)

    def test_runner_through_snapshot_adapter_with_fake_backend(self):
        from test_model_probe import FakeBackend, config
        from experiments.bar_quality_history.model_probe import SnapshotProbe
        backends = []

        class BoundaryFakeBackend(FakeBackend):
            def generate(self, **kwargs):
                events, timing = super().generate(**kwargs)
                if kwargs["generation_start_tick"] == 16:
                    events.insert(0, {"type": "note_off", "tick": 16, "pitch": 48})
                return events, timing

        def factory(settings):
            backend = BoundaryFakeBackend(settings)
            backends.append(backend)
            return backend

        with SnapshotProbe(config(), backend_factory=factory) as probe:
            result = self.run_case(probe)
        self.assertEqual(result["status"], "paired_probe_complete")
        self.assertEqual(len(backends), 1)
        self.assertTrue(backends[0].closed)
        self.assertEqual([value for name, value in backends[0].calls if name == "reset"], [99, 71, 99])
        self.assertEqual(result["repaired_next_beat"]["runtime"]["backend_kind"], "fake")

    def test_unclosed_candidate_carries_without_inventing_note_off(self):
        probe = RecordingProbe(open_candidate=True)
        result = self.run_case(probe)
        self.assertEqual(result["status"], "paired_probe_complete")
        self.assertEqual(len(probe.calls), 3)
        self.assertIn(60, result["history_transaction"]["active_pitches_at_end"])
        self.assertEqual(result["history_transaction"]["synthetic_note_off_count"], 0)
        self.assertEqual(len(result["repair_bar"]["events"]), 1)

    def test_boundary_release_consumes_real_prefix_active_note_once(self):
        prefix = notes_to_events([{"onset": 12, "end": 16, "pitch": 48}])
        candidate = [{"type": "note_off", "tick": 16, "pitch": 48},
                     {"type": "note_on", "tick": 16, "pitch": 60}]
        result = splice_candidate_events(prefix, candidate, 16, 32, [48])
        self.assertEqual(result["active_pitches_at_end"], [60])
        releases = [e for e in result["resulting_history_events"] if e["type"] == "note_off"]
        self.assertEqual(releases, [candidate[0]])
        self.assertEqual(result["candidate_events"][1]["velocity"], 100)

    def test_boundary_splice_rejects_orphan_or_incorrect_initial_state(self):
        with self.assertRaises(InputError):
            splice_candidate_events([], [{"type": "note_off", "tick": 16, "pitch": 48}], 16, 32)
        with self.assertRaises(InputError):
            splice_candidate_events([], [], 16, 32, [48])

    def test_event_splice_leaves_prefix_and_inputs_unchanged(self):
        prefix = notes_to_events([{"onset": 0, "end": 8, "pitch": 48}])
        candidate = [{"type": "note_on", "tick": 16, "pitch": 60, "velocity": 90}]
        original = deepcopy((prefix, candidate))
        result = splice_candidate_events(prefix, candidate, 16, 32)
        self.assertEqual(result["prefix_events"], prefix)
        result["candidate_events"][0]["pitch"] = 61
        self.assertEqual((prefix, candidate), original)

    def test_preflight_before_generation(self):
        for field, value in (("bar_length_steps", 15), ("melody_notes", [{"onset": 30, "end": 33, "pitch": 60}])):
            original = deepcopy(self.data)
            self.data[field] = value
            probe = RecordingProbe()
            with self.assertRaises(InputError):
                self.run_case(probe)
            self.assertEqual(probe.calls, [])
            self.data = original

    def test_carry_history_rejected_before_generation(self):
        self.data["history_notes"][0]["end"] = 17
        probe = RecordingProbe()
        with self.assertRaises(InputError):
            self.run_case(probe)
        self.assertEqual(probe.calls, [])

    def test_supplied_candidate_not_mistaken_for_generated(self):
        self.data["repair_candidate"] = None
        with self.assertRaises(InputError):
            self.run_case()

    def test_candidate_validation(self):
        invalid = [
            [{"type": "note_off", "tick": 17, "pitch": 60}],
            [{"type": "note_on", "tick": 32, "pitch": 60, "velocity": 90}],
            [{"type": "note_on", "tick": 17.5, "pitch": 60, "velocity": 90}],
            [{"type": "note_on", "tick": 17, "pitch": 60, "velocity": 0}],
            [{"type": "pedal", "tick": 17, "pitch": 60}],
        ]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(InputError):
                closed_candidate_notes(events, 16, 32)

    def test_release_at_boundary_and_canonical_ties(self):
        notes = [{"onset": 16, "end": 20, "pitch": 60}, {"onset": 20, "end": 32, "pitch": 60}]
        events = notes_to_events(notes)
        self.assertEqual([e["type"] for e in events if e["tick"] == 20], ["note_off", "note_on"])
        self.assertEqual(closed_candidate_notes(events, 16, 32), notes)


if __name__ == "__main__":
    unittest.main()

import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bar_history_harness", ROOT / "harness.py")
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def note(onset, end, pitch=60):
    return {"onset": onset, "end": end, "pitch": pitch}


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "example.synthetic.json").read_text(encoding="utf-8"))

    def run_case(self, **updates):
        data = deepcopy(self.data)
        data.update(updates)
        return harness.run_transaction(data)

    def test_baseline_unchanged_even_when_rejected(self):
        out = self.run_case(policy="baseline")
        self.assertEqual(out["resulting_note_history"], self.data["history_notes"])
        self.assertEqual(out["action"], "keep")
        self.assertEqual(out["scope"], "HISTORY_ONLY")
        self.assertEqual(out["conceptual_boundary_step"], 32)
        self.assertFalse(out["repair_timing_simulated"])

    def test_drop_only_removes_context_in_bar(self):
        out = self.run_case(policy="drop")
        self.assertEqual(out["resulting_note_history"], [note(8, 16, 48)])
        self.assertEqual(out["future_context_notes"], out["resulting_note_history"])
        self.assertEqual(out["playback_after"], self.data["played_notes"])

    def test_repair_preserves_played_and_original_and_input(self):
        before = deepcopy(self.data)
        out = harness.run_transaction(self.data)
        self.assertEqual(self.data, before)
        self.assertEqual(out["original_note_history"], before["history_notes"])
        self.assertEqual(out["resulting_note_history"], [note(8, 16, 48), note(16, 20)])
        self.assertEqual(out["playback_before"], out["playback_after"])
        out["resulting_note_history"][0]["pitch"] = 1
        self.assertEqual(out["playback_after"], before["played_notes"])
        self.assertEqual(out["original_note_history"], before["history_notes"])
        self.assertEqual(out["input_provenance"], before)

    def test_accept_keeps_history_for_all_policies(self):
        self.data["decisions"][0]["decision"] = "accept"
        for policy in ("baseline", "drop", "repair_history"):
            with self.subTest(policy=policy):
                self.assertEqual(self.run_case(policy=policy)["resulting_note_history"],
                                 self.data["history_notes"])

    def test_future_and_unfinished_notes_rejected_even_baseline(self):
        for field in ("melody_notes", "history_notes", "played_notes"):
            for notes in ([note(32, 33)], [note(31, 33)]):
                with self.subTest(field=field, notes=notes):
                    with self.assertRaises(harness.InputError):
                        self.run_case(**{field: notes, "policy": "baseline"})
        self.data["repair_candidate"]["notes"] = [note(32, 33)]
        with self.assertRaises(harness.InputError):
            self.run_case(policy="baseline")

    def test_crossing_history_fails_closed_for_mutation(self):
        self.data["history_notes"] = [note(15, 20)]
        for policy in ("drop", "repair_history"):
            with self.subTest(policy=policy), self.assertRaises(harness.InputError):
                self.run_case(policy=policy)
        self.assertEqual(self.run_case(policy="baseline")["action"], "keep")

    def test_crossing_candidate_rejected(self):
        for notes in ([note(15, 20)], [note(30, 33)]):
            self.data["repair_candidate"]["notes"] = notes
            with self.subTest(notes=notes), self.assertRaises(harness.InputError):
                self.run_case()

    def test_boundary_inclusion_and_union_coverage(self):
        notes = [note(8, 16), note(16, 20, 60), note(18, 32, 67)]
        out = self.run_case(history_notes=notes, policy="drop")
        stats = out["statistics"]["original_history_bar"]
        self.assertEqual(out["resulting_note_history"], [note(8, 16)])
        self.assertEqual(stats["onsets"], 2)
        self.assertEqual(stats["overlap_coverage_steps"], 16)
        self.assertEqual(stats["overlap_coverage_fraction"], 1)
        self.assertEqual(stats["overlap_duration_median_steps"], 9)
        self.assertEqual(stats["pitch_range_semitones"], 7)
        self.assertEqual(out["statistics"]["original_history_to_melody_duration_median_ratio"], 2.25)

    def test_empty_stats_and_empty_candidate_are_explicit(self):
        self.data["repair_candidate"]["notes"] = []
        out = self.run_case(melody_notes=[])
        self.assertEqual(out["action"], "replace")
        self.assertIsNone(out["statistics"]["original_history_to_melody_duration_median_ratio"])
        self.assertIsNone(out["statistics"]["resulting_history_bar"]["pitch_range_semitones"])

    def test_missing_candidate_fallbacks(self):
        self.data.pop("repair_candidate")
        for fallback in ("keep", "drop"):
            out = self.run_case(repair_fallback=fallback)
            self.assertEqual(out["action"], fallback)
            self.assertEqual(out["reason"], "missing_candidate_fallback_" + fallback)
        with self.assertRaises(harness.InputError):
            self.run_case(repair_fallback="error")
        self.data.pop("repair_fallback")
        with self.assertRaises(harness.InputError):
            self.run_case()

    def test_rejected_candidate_uses_explicit_fallback(self):
        self.data["repair_candidate"]["decision"] = "reject"
        for fallback in ("keep", "drop"):
            out = self.run_case(repair_fallback=fallback)
            self.assertEqual(out["action"], fallback)
            self.assertEqual(out["reason"], "rejected_candidate_fallback_" + fallback)
            expected = self.data["history_notes"] if fallback == "keep" else [note(8, 16, 48)]
            self.assertEqual(out["resulting_note_history"], expected)
            self.assertEqual(out["playback_after"], self.data["played_notes"])
        with self.assertRaises(harness.InputError):
            self.run_case(repair_fallback="error")

    def test_candidate_decision_required_and_valid(self):
        for value in (None, True, "approved", []):
            self.data["repair_candidate"]["decision"] = value
            with self.subTest(value=value), self.assertRaises(harness.InputError):
                self.run_case()
        self.data["repair_candidate"].pop("decision")
        with self.assertRaises(harness.InputError):
            self.run_case()

    def test_invalid_notes(self):
        for field in ("onset", "end", "pitch"):
            for value in (True, "17", 17.5, 17.0, None, -1):
                bad = note(16, 20)
                bad[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(harness.InputError):
                    self.run_case(history_notes=[bad])
        for bad in (note(20, 20), note(21, 20), note(16, 20, 128), {}, []):
            with self.subTest(note=bad), self.assertRaises(harness.InputError):
                self.run_case(history_notes=[bad])

    def test_invalid_schema_and_boundaries(self):
        for updates in ({"observed_until": 33}, {"observed_until": True},
                        {"bar_length_steps": 0}, {"bar_length_steps": "16"},
                        {"steps_per_beat": 4.0}, {"policy": "auto"},
                        {"decisions": []}, {"decisions": self.data["decisions"] * 2},
                        {"extra": 1}):
            with self.subTest(updates=updates), self.assertRaises(harness.InputError):
                self.run_case(**updates)
        for field in ("bar_start_step", "boundary_step"):
            data = deepcopy(self.data)
            data["decisions"][0][field] += 1
            with self.assertRaises(harness.InputError):
                harness.run_transaction(data)
        self.data.pop("bar_length_steps")
        with self.assertRaises(harness.InputError):
            self.run_case()

    def test_cli_json_success_and_fail_closed(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            harness.main([str(ROOT / "example.synthetic.json")])
        self.assertEqual(json.loads(stdout.getvalue())["action"], "replace")
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{', '[]'):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                path.write_text(raw, encoding="utf-8")
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        harness.main([str(path)])
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

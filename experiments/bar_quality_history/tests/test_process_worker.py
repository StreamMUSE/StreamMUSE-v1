"""Worker envelope tests without starting processes or importing torch."""

from copy import deepcopy
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.bar_quality_history import process_worker as module


class Probe:
    def generate(self, **kwargs):
        self.arguments = kwargs
        return {"events": [], "metadata": {"token_decode_beats": []}}


class WorkerTests(unittest.TestCase):
    def test_close_only_terminates_owned_worker_after_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = module.ProcessRepairWorker.__new__(module.ProcessRepairWorker)
            worker._closed = False
            worker.audit_dir = Path(folder)
            child = Mock(pid=123)
            child.is_alive.side_effect = [True, False]
            worker._pool = Mock()
            worker._pool._processes = {123: child}
            worker.close(timeout=0)
            worker._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
            child.terminate.assert_called_once_with()
            child.kill.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / "shutdown.json").read_text())[
                "forced_termination_pids"], [123])
            worker.close(timeout=0)

    def test_config_mismatch_rejected_before_submission(self):
        worker = module.ProcessRepairWorker.__new__(module.ProcessRepairWorker)
        worker._closed = False
        worker._pending = None
        worker._ready_future = Future()
        worker._ready_future.set_result(True)
        worker._pool = Mock()
        worker.config = {"bpm": 120}
        with self.assertRaisesRegex(ValueError, "bpm"):
            worker.submit({"generation_config": {"bpm": 80}})
        worker._pool.submit.assert_not_called()

    def test_worker_audit_survives_discard_or_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(module, "_AUDIT_DIR", Path(folder)):
            with patch.object(module, "_generate", return_value={"events": []}):
                self.assertEqual(module._generate_audited({"end_tick": 32}, 101), {"events": []})
            with patch.object(module, "_generate", side_effect=ValueError("invalid")):
                with self.assertRaises(ValueError):
                    module._generate_audited({"end_tick": 32}, 101)
            records = [json.loads(p.read_text()) for p in Path(folder).glob("candidate_*.json")]
            self.assertEqual(len(records), 2)
            self.assertTrue(any("error" in r for r in records))
            self.assertTrue(any("candidate" in r for r in records))

    def test_no_implicit_gpu_or_mock_model(self):
        with self.assertRaises(ValueError):
            module.ProcessRepairWorker(config={"mode": "real"}, gpu="2", seed=0, audit_dir="unused")
        with self.assertRaises(ValueError):
            module.ProcessRepairWorker(config={"mode": "real"}, gpu="0", seed=0, audit_dir="unused", allow_gpu=True)
        with self.assertRaises(ValueError):
            module.ProcessRepairWorker(config={"mode": "mock"}, gpu="2", seed=0, audit_dir="unused", allow_gpu=True)

    def test_prefix_boundary_and_observation_filter(self):
        probe = Probe()
        ticket = {"start_tick": 16, "end_tick": 32,
                  "melody_events": [
                      {"type": "note_on", "tick": 20, "pitch": 60},
                      {"type": "note_off", "tick": 32, "pitch": 60},
                      {"type": "note_on", "tick": 32, "pitch": 62},
                  ],
                  "accompaniment_events": [
                      {"type": "note_on", "tick": 0, "pitch": 48},
                      {"type": "note_off", "tick": 15, "pitch": 48},
                      {"type": "note_on", "tick": 16, "pitch": 50},
                  ]}
        original = deepcopy(ticket)
        with patch.object(module, "_PROBE", probe):
            result = module._generate(ticket, 101)
        self.assertEqual(ticket, original)
        self.assertEqual(len(probe.arguments["melody_events"]), 2)
        self.assertEqual(len(probe.arguments["accompaniment_events"]), 2)
        self.assertEqual(probe.arguments["seed"], 101)
        self.assertEqual(probe.arguments["observed_until"], 32)
        self.assertEqual(result["events"], [])


if __name__ == "__main__":
    unittest.main()

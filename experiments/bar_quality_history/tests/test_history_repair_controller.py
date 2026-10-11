"""CPU-only scheduling tests; no torch/model import or background jobs."""

from concurrent.futures import Future
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Controller = load("src/streammuse/infrastructure/inference/history_repair_controller.py",
                  "repair_controller_test").FixedBarHistoryRepair


class Backend:
    def __init__(self):
        self.calls = []
        self.begin_status = "ready"

    def begin_history_repair(self, start, end):
        self.calls.append(("begin", start, end))
        return {"status": self.begin_status, "ticket": {"start_tick": start, "end_tick": end}}

    def apply_history_repair(self, ticket, candidate):
        self.calls.append(("apply", ticket, candidate))
        return {"status": "applied"}

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return ([{"type": "note_on", "tick": 16, "pitch": 60}], {"time": 1.0})

    def history_repair_completed_until(self):
        return 20

    def reset_session(self, seed):
        return {"seed": seed}

    def set_session_generation_config(self, **kwargs):
        pass

    def inject_history(self, **kwargs):
        return kwargs


class Worker:
    def __init__(self):
        self.future = Future()
        self.calls = []

    def submit(self, ticket):
        self.calls.append(deepcopy(ticket))
        return self.future


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend, self.worker = Backend(), Worker()
        self.controller = Controller(start_tick=0, end_tick=16, worker=self.worker)

    def test_explicit_bar_validation(self):
        for start, end in ((0, 4), (4, 20), (-16, 0), (True, 17)):
            with self.assertRaises(ValueError):
                Controller(start_tick=start, end_tick=end, worker=self.worker)

    def test_one_submission_at_boundary(self):
        for end in (4, 8, 12, 16, 20, 32):
            self.controller.after_generate(self.backend, end)
        self.assertEqual(len(self.worker.calls), 1)

    def test_pending_does_not_wait_or_apply(self):
        self.controller.after_generate(self.backend, 20)
        self.worker.future.result = lambda: self.fail("waited for pending candidate")
        self.controller.before_generate(self.backend)
        self.assertEqual(self.backend.calls[0][0], "begin")
        self.assertEqual(len(self.backend.calls), 1)

    def test_completed_applies_once(self):
        self.controller.after_generate(self.backend, 20)
        self.worker.future.set_result({"events": []})
        self.controller.before_generate(self.backend)
        self.controller.before_generate(self.backend)
        self.assertEqual([x[0] for x in self.backend.calls], ["begin", "apply"])

    def test_failed_worker_preserves_history(self):
        self.controller.after_generate(self.backend, 20)
        self.worker.future.set_exception(RuntimeError("worker failed"))
        self.controller.before_generate(self.backend)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.controller.records[-1]["status"], "failed")

    def test_reset_discards_running_result_and_does_not_rearm(self):
        self.controller.after_generate(self.backend, 20)
        self.worker.future.set_running_or_notify_cancel()
        self.controller.invalidate("reset")
        self.worker.future.set_result({"events": []})
        self.controller.before_generate(self.backend)
        self.controller.after_generate(self.backend, 20)
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(len(self.backend.calls), 1)

    def test_rejected_begin_does_not_submit(self):
        self.backend.begin_status = "rejected"
        self.controller.after_generate(self.backend, 20)
        self.assertEqual(self.worker.calls, [])

    def test_missed_boundary_not_retried(self):
        self.controller.after_generate(self.backend, 24)
        self.controller.after_generate(self.backend, 20)
        self.assertEqual(self.worker.calls, [])

    def test_unknown_frontier_does_not_consume_attempt(self):
        self.controller.after_generate(self.backend, None)
        self.controller.after_generate(self.backend, 16)
        self.assertEqual(self.worker.calls, [])
        self.controller.after_generate(self.backend, 20)
        self.assertEqual(len(self.worker.calls), 1)

    def test_detach_completed_candidate_records_without_apply(self):
        self.controller.after_generate(self.backend, 20)
        self.worker.future.set_result({"events": []})
        self.controller.invalidate("detached")
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.controller.records[-1]["status"], "completed_discarded")

    def test_engine_default_path_and_reset_invalidation(self):
        name = "streammuse.infrastructure.inference.lekai_http_backend"
        stub = types.ModuleType(name)
        stub.BackendRuntimeConfig = object
        stub.EventPayload = dict
        stub.TimingPayload = dict
        stub.LekaiHttpBackend = Backend
        with patch.dict(sys.modules, {name: stub}):
            module = load("src/streammuse/infrastructure/inference/lekai_prompt_continuation/continuation_engine.py",
                          "repair_engine_test")
        engine = module.LekaiContinuationEngine(backend=self.backend)
        request = dict(melody_events=[], generation_start_tick=16,
                       generation_length_frames=4, generation_interval_ticks=4,
                       prompt_length_ticks=128, inference_mode="sliding_window",
                       model_name="lekai", checkpoint_path=None, bpm=120)
        original = engine.generate(**request)
        self.assertEqual([x[0] for x in self.backend.calls], ["generate"])
        engine.attach_history_repair(self.controller)
        self.assertEqual(engine.generate(**request), original)
        engine.reset_session(9)
        self.assertTrue(self.worker.future.cancelled())
        self.assertEqual(engine.generate(**request), original)


if __name__ == "__main__":
    unittest.main()

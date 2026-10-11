"""Opt-in, one-shot repair orchestration. Never waits for model work."""

from copy import deepcopy
import time


class FixedBarHistoryRepair:
    """Worker.submit(ticket) must return a Future without running inference inline.

    The worker must be prestarted and process-isolated from the live model. Its
    ownership remains with the caller; reset cancels/ignores work without joining.
    """

    def __init__(self, *, start_tick, end_tick, worker):
        if (type(start_tick) is not int or type(end_tick) is not int
                or start_tick < 0 or start_tick % 16 or end_tick != start_tick + 16):
            raise ValueError("first integration supports one explicit 4/4 bar")
        self.start_tick, self.end_tick = start_tick, end_tick
        self.worker = worker
        self.records = []
        self._pending = None
        self._attempted = False

    def _record(self, status, **fields):
        self.records.append({"status": status, "recorded_at": time.perf_counter(), **fields})

    def invalidate(self, reason):
        if self._pending is not None:
            _, future = self._pending
            if future.done():
                try:
                    self._record("completed_discarded", reason=reason,
                                 candidate=deepcopy(future.result()))
                except Exception as exc:
                    self._record("failed_discarded", reason=reason, error=str(exc))
            else:
                self._record("discarded", reason=reason, cancelled=future.cancel(),
                             later_outcome="worker_audit")
            self._pending = None
        else:
            self._record("retired", reason=reason)
        # A controller belongs to one session. Re-arm explicitly with a new one.
        self._attempted = True

    def before_generate(self, backend):
        if self._pending is None:
            return
        ticket, future = self._pending
        if not future.done():
            return
        self._pending = None
        try:
            candidate = future.result()
            result = backend.apply_history_repair(ticket, candidate)
            application = (backend.history_repair_audit_snapshot()
                           if result.get("status") == "applied"
                           and hasattr(backend, "history_repair_audit_snapshot") else [])
            self._record("completed", result=deepcopy(result), candidate=deepcopy(candidate),
                         application_audit=deepcopy(application))
        except Exception as exc:
            self._record("failed", stage="apply", error=str(exc))

    def after_generate(self, backend, generation_end_tick):
        # ACC-first generation observes previous beats. Wait one further beat;
        # backend trace must independently confirm its Melody encoding horizon.
        trigger_tick = self.end_tick + 4
        if self._attempted or generation_end_tick is None or generation_end_tick < trigger_tick:
            return
        self._attempted = True
        if generation_end_tick != trigger_tick:
            self._record("skipped", reason="target_boundary_passed")
            return
        try:
            result = backend.begin_history_repair(self.start_tick, self.end_tick)
            if result.get("status") != "ready":
                self._record("skipped", result=deepcopy(result))
                return
            ticket = deepcopy(result["ticket"])
            future = self.worker.submit(deepcopy(ticket))
            self._pending = (ticket, future)
            self._record("submitted", start_tick=self.start_tick, end_tick=self.end_tick,
                         ticket=deepcopy(ticket))
        except Exception as exc:
            self._record("failed", stage="submit", error=str(exc))

    def audit_snapshot(self):
        return deepcopy(self.records)

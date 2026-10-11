"""Explicit spawned-process model worker for the opt-in live experiment.

Warm up and await ready BEFORE attaching a controller. GPU availability and
permission must be checked externally. Importing this module loads no model.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import json
import time
import uuid


_PROBE = None
_AUDIT_DIR = None


def _initialize(config, gpu, audit_dir):
    global _PROBE, _AUDIT_DIR
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    _AUDIT_DIR = Path(audit_dir)
    os.environ["LEKAI_RT_LOG_DIR"] = str(_AUDIT_DIR / "generation")
    from experiments.bar_quality_history.model_probe import SnapshotProbe
    _PROBE = SnapshotProbe(config, allow_gpu=True)
    _PROBE.__enter__()


def _ready():
    return True


def _generate(ticket, seed):
    start, end = ticket["start_tick"], ticket["end_tick"]

    def wire(events):
        return [dict(type=e["type"], pitch=int(e["pitch"]), tick=int(e["tick"]),
                     velocity=int(e.get("velocity", 100))) for e in events]

    result = _PROBE.generate(
        melody_events=wire([e for e in ticket["melody_events"]
                            if e["tick"] < end or
                            (e["tick"] == end and e["type"] == "note_off")]),
        accompaniment_events=wire([e for e in ticket["accompaniment_events"]
                                   if e["tick"] < start]),
        target_start_step=start, length_steps=end - start,
        observed_until=end, seed=seed,
        constraint_snapshot=ticket.get("constraint_snapshot"),
    )
    return {"events": result["events"],
            "token_decode_beats": result["metadata"]["token_decode_beats"],
            "provenance": result,
            "guard_semantics": "candidate restores bar-start snapshot; live actual-output guard is preserved"}


def _generate_audited(ticket, seed, boundary_policy='strict'):
    started = time.perf_counter()
    record = {"ticket": ticket, "repair_seed": seed, "started_at": started, "boundary_policy": boundary_policy}
    try:
        result = _generate(ticket, seed)
        if boundary_policy == 'preserve_cross_boundary':
            from experiments.bar_quality_history.boundary_projection import preserve_cross_boundary
            record['unprojected_candidate'] = deepcopy(result)
            result = preserve_cross_boundary(ticket, result)
        record["candidate"] = result
        return result
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["completed_at"] = time.perf_counter()
        # Preserve even a result later discarded by session invalidation/CAS.
        with (_AUDIT_DIR / f"candidate_{uuid.uuid4().hex}.json").open("x", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False, allow_nan=False, indent=2)


class ProcessRepairWorker:
    def __init__(self, *, config, gpu, seed, audit_dir, allow_gpu=False, boundary_policy='strict'):
        if allow_gpu is not True or str(gpu) not in ("1", "2", "3"):
            raise ValueError("explicit permission and currently authorized GPU 1/2/3 required")
        if config.get("mode") != "real":
            raise ValueError("process worker requires explicit real model configuration")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("explicit nonnegative repair seed required")
        self.seed = seed
        if boundary_policy not in ('strict', 'preserve_cross_boundary'):
            raise ValueError('unknown boundary policy')
        self.boundary_policy = boundary_policy
        self.config = deepcopy(config)
        self.audit_dir = Path(audit_dir).resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=False)
        self._pool = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(self.config, str(gpu), str(self.audit_dir)),
        )
        self._ready_future = self._pool.submit(_ready)
        self._pending = None
        self._closed = False

    def ready(self):
        return self._ready_future.done() and self._ready_future.result() is True

    def submit(self, ticket):
        if self._closed or not self.ready():
            raise RuntimeError("repair worker not ready")
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("repair worker busy; keep original history")
        source = ticket.get("generation_config", {})
        if source.get("empty_token_guard_enabled") and ticket.get("constraint_snapshot") is None:
            raise ValueError("guarded repair requires bar-start constraint snapshot")
        for key in ("bpm", "dtype", "context_beats", "temperature", "top_p", "top_k",
                    "repetition_penalty", "time_signature_index",
                    "tonal_constraint_enabled", "empty_token_guard_enabled"):
            if key not in source or source[key] != self.config[key]:
                raise ValueError(f"repair worker configuration mismatch: {key}")
        if (not source.get("checkpoint_path") or
                Path(source["checkpoint_path"]).resolve() !=
                Path(self.config["checkpoint_path"]).resolve()):
            raise ValueError("repair worker checkpoint mismatch")
        self._pending = self._pool.submit(_generate_audited, deepcopy(ticket), self.seed, self.boundary_policy)
        return self._pending

    def close(self, timeout=5.0):
        """Bounded teardown of ONLY this worker; call outside the playback loop."""
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        if self._closed:
            return
        self._closed = True
        # Python 3.10 lacks Executor.terminate_workers(). Capture only processes
        # owned by this executor before shutdown clears its internal registry.
        processes = list((self._pool._processes or {}).values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + timeout
        terminated = []
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                terminated.append(process.pid)
                process.terminate()
                process.join(2.0)
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
        with (self.audit_dir / "shutdown.json").open("x", encoding="utf-8") as output:
            json.dump({"forced_termination_pids": terminated,
                       "pending_candidate_may_be_incomplete": bool(terminated)}, output)

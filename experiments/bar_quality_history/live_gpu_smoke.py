"""Paced real-model integration test, not the HTTP/playback ISR evaluator."""

import argparse
from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)


def midi_events(path, track):
    import mido
    midi = mido.MidiFile(path)
    tick, result = 0, []
    for event in midi.tracks[track]:
        tick += event.time
        if event.type not in ("note_on", "note_off"):
            continue
        if tick * 4 % midi.ticks_per_beat:
            raise ValueError("source MIDI not on exact four-step grid")
        result.append(dict(type="note_off" if event.type == "note_off" or not event.velocity
                           else "note_on", tick=tick * 4 // midi.ticks_per_beat,
                           pitch=event.note, velocity=event.velocity))
    return result


class IdentityWorker:
    """Positive plumbing control, NOT a regenerated candidate."""
    def submit(self, ticket):
        future = Future()
        future.set_result({"events": [e for e in ticket["accompaniment_events"]
                                      if ticket["start_tick"] <= e["tick"] < ticket["end_tick"]],
                           "token_decode_beats": [],
                           "provenance": {"control": "identity_original_generated_events"}})
        return future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--melody", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=["1", "2", "3"], required=True)
    parser.add_argument("--playback-bpm", type=int, default=90)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("explicit GPU binding must match")
    config = json.loads(args.config.read_text())
    original_config = deepcopy(config)
    config["empty_token_guard_enabled"] = False
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "config.json", {"effective": config, "source": original_config,
         "change": "empty guard explicitly OFF for first live history integration",
         "playback_bpm": args.playback_bpm,
         "scope": "paced backend integration; no HTTP/MIDI sink or ISR claim"})
    melody = midi_events(args.melody, 1)
    history = midi_events(args.history, 2)
    save(args.output / "sources.json", [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                       for p in (args.melody, args.history)])
    os.environ["LEKAI_RT_LOG_DIR"] = str(args.output / "main_generation")
    os.environ["LEKAI_HISTORY_REPAIR_ENABLED"] = "1"
    from experiments.bar_quality_history.model_probe import SnapshotProbe, _verify_runtime
    from experiments.bar_quality_history.process_worker import ProcessRepairWorker
    from streammuse.infrastructure.inference.history_repair_controller import FixedBarHistoryRepair
    from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import LekaiContinuationEngine

    worker = ProcessRepairWorker(config=config, gpu=args.gpu, seed=101,
                                 audit_dir=args.output / "worker", allow_gpu=True)
    probe = SnapshotProbe(config, allow_gpu=True)
    summaries = []
    try:
        deadline = time.monotonic() + 180
        while not worker.ready():
            if time.monotonic() > deadline:
                raise TimeoutError("worker startup exceeded 180 seconds")
            time.sleep(0.1)
        probe.__enter__()
        with probe._environment():
            # In-process engine calls, not HTTP requests with session envelopes.
            os.environ["LEKAI_REQUIRE_SESSION"] = "0"
            backend = probe._backend
            _verify_runtime(backend.runtime_info(), config)
            engine = LekaiContinuationEngine(backend=backend)
            for seed, condition in [(s, c) for s in (0, 1, 2) for c in ("baseline", "repair")] + [(0, "identity_control")]:
                engine.attach_history_repair(None)
                engine.reset_session(seed)
                engine.set_session_generation_config(**{k: config[k] for k in (
                    "temperature", "top_p", "top_k", "repetition_penalty")})
                engine.inject_history([e for e in melody if e["tick"] < 32],
                                      [e for e in history if e["tick"] < 32], 32)
                controller = None
                if condition != "baseline":
                    controller = FixedBarHistoryRepair(start_tick=32, end_tick=48,
                                  worker=worker if condition == "repair" else IdentityWorker())
                    engine.attach_history_repair(controller)
                rows, returned = [], []
                origin = time.perf_counter()
                previous_boundary = 32
                for start in range(32, 80, 4):
                    due = origin + (start - 32) / 4 * 60 / args.playback_bpm
                    time.sleep(max(0, due - time.perf_counter()))
                    increment = [e for e in melody if previous_boundary <= e["tick"] < start]
                    previous_boundary = start
                    backend.input_token_calls.clear()
                    began = time.perf_counter()
                    events, timings = engine.generate(
                        melody_events=increment, generation_start_tick=start,
                        generation_length_frames=4, generation_interval_ticks=4,
                        prompt_length_ticks=config["context_beats"] * 4,
                        inference_mode="sliding_window", model_name="lekai",
                        checkpoint_path=config["checkpoint_path"], bpm=config["bpm"])
                    ended = time.perf_counter()
                    metadata = backend.generation_metadata_snapshot()[-1]
                    returned.extend(deepcopy(events))
                    rows.append({"start_tick": start, "elapsed_seconds": ended - began,
                                 "started_after_due_seconds": began - due,
                                 "completed_after_due_seconds": ended - due,
                                 "within_one_beat_budget": ended <= due + 60 / args.playback_bpm,
                                 "events": deepcopy(events), "timings": timings,
                                 "input_tokens": deepcopy(backend.input_token_calls),
                                 "metadata": metadata})
                applied = backend.history_repair_audit_snapshot()
                engine.attach_history_repair(None)
                record = {"seed": seed, "condition": condition, "rows": rows,
                          "controller_audit": controller.audit_snapshot() if controller else [],
                          "applied_repairs": applied, "returned_events": returned,
                          "runtime": backend.runtime_info()}
                save(args.output / f"{condition}_s{seed}.json", record)
                summary = {"seed": seed, "condition": condition, "applied": len(applied),
                           "mean_request_seconds": sum(r["elapsed_seconds"] for r in rows) / len(rows),
                           "within_beat_budget": sum(r["within_one_beat_budget"] for r in rows),
                           "requests": len(rows), "audit": [
                               {k: r[k] for k in ("status", "reason", "result", "error") if k in r}
                               for r in record["controller_audit"]]}
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            save(args.output / "summary.json", summaries)
    finally:
        probe.close()
        worker.close()


if __name__ == "__main__":
    main()

"""Audit-only MIDI-to-fixed-event input using the existing Realtime tick hook."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

import mido

import plain_lekai_device_replay_acceptance as audit
from human6_timed_input_replay import CASES
from summarize_human_input_timing import MISSING_CASES
from streammuse.application.config.models import (
    ApplicationConfig, InferenceConfig, InputConfig, OutputConfig, TempoConfig,
)
from streammuse.application.runtime.builder import RuntimeSessionBuilder
from streammuse.domain.musical import EventType, MusicalEvent


def load_midi_events(path):
    midi = mido.MidiFile(path)
    tracks = [(i, t) for i, t in enumerate(midi.tracks)
              if any(m.type in ("note_on", "note_off") for m in t)]
    if len(tracks) != 1:
        raise ValueError("Expected exactly one note-bearing Melody track")
    index, track = tracks[0]
    events, absolute = [], 0
    for message_index, msg in enumerate(track):
        absolute += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        tick = Fraction(absolute * 4, midi.ticks_per_beat)
        if tick.denominator != 1:
            raise ValueError(f"Off-grid event at MIDI tick {absolute}: {tick} system ticks")
        event_type = "note_on" if msg.type == "note_on" and msg.velocity else "note_off"
        events.append({"sequence": len(events) + 1, "tick": int(tick), "pitch": msg.note,
                       "event_type": event_type, "velocity": msg.velocity, "channel": msg.channel,
                       "midi_message_index": message_index, "midi_absolute_tick": absolute,
                       "midi_message_type": msg.type})
    if not events:
        raise ValueError("Empty Melody MIDI")
    return events, {"midi_ppq": midi.ticks_per_beat, "track_index": index,
                    "track_name": track.name, "event_count": len(events),
                    "last_event_tick": events[-1]["tick"], "midi_sha256": audit.sha(Path(path).read_bytes())}


class FixedTickInput:
    """Inject only the current tick before the existing service drains its queue."""

    def __init__(self, events):
        self.schedule = defaultdict(list)
        for event in events:
            self.schedule[event["tick"]].append(event)
        self.service = None
        self.delivered = []
        self.last_tick = -1
        self.error = None

    def start(self):
        if self.service is None:
            raise RuntimeError("Fixed input is not attached to the runtime")

    def on_tick(self, tick):
        try:
            if tick != self.last_tick + 1:
                raise ValueError("Nonconsecutive service ticks")
            self.last_tick = tick
            for row in self.schedule.get(tick, []):
                event = MusicalEvent(tick=row["tick"], pitch=row["pitch"],
                                     event_type=EventType(row["event_type"]),
                                     velocity=row["velocity"], channel=row["channel"], source="user")
                self.service._event_q.put(event)
                with self.service._melody_history_lock:
                    self.service._melody_history.append(event)
                self.delivered.append({**row, "injected_at_logical_tick": tick})
        except Exception as exc:
            self.error = repr(exc)
            raise

    def close(self):
        pass


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=False)
    cases = []
    for piece, date, session in sorted(CASES + MISSING_CASES):
        source = args.source_root / date / session / "prompt_continuation_replay_melody.mid"
        events, metadata = load_midi_events(source)
        dest = args.root / "inputs" / piece
        dest.mkdir(parents=True)
        shutil.copy2(source, dest / "melody.mid")
        audit.save(dest / "fixed_events.json", events)
        audit.save(dest / "midi_metadata.json", {**metadata, "source": str(source)})
        cases.append({"piece": piece, "source_session": session, **metadata,
                      "stop_tick": ((metadata["last_event_tick"] + 3) // 4) * 4 + 16})
    audit.save(args.root / "plan.json", {
        "system_ref": "ec843a79", "condition": "plain_lekai_fixed_midi_ticks",
        "seed": 1051154023138951872, "cases": cases,
        "scope": "Two new same-version runs per MIDI; NOT equality with historical live output",
        "playback_bpm": 90, "model_bpm": 80, "temperature": 1.1,
        "top_p": 0.95, "top_k": 50, "repetition_penalty": 1,
        "input_rule": "Native MIDI ticks * 4 / PPQ; require integers; preserve track order and velocity",
        "visibility_rule": "At service tick t, inject only events labeled t before the existing queue drain",
        "real_time_clock": True, "pause_clock_for_inference": False,
    })
    print(json.dumps(cases, indent=2))


def client(args):
    plan = audit.read(args.root / "plan.json")
    case = next(c for c in plan["cases"] if c["piece"] == args.piece)
    inputs = args.root / "inputs" / args.piece
    events, metadata = load_midi_events(inputs / "melody.mid")
    assert metadata["midi_sha256"] == case["midi_sha256"]
    assert events == audit.read(inputs / "fixed_events.json")
    out = args.root / "cases" / args.piece / f"repeat_{args.repeat}"
    out.mkdir(parents=True, exist_ok=False)
    reset = audit.api(args.server, "/debug/reset_session", {"seed": plan["seed"]})
    assert reset["effective_seed"] == plan["seed"]
    audit.save(out / "session_initialized.json", reset)
    audit.save(out / "identity.json", audit.identity(plan["system_ref"]))
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90, ticks_per_beat=4, beats_per_bar=4),
        input=InputConfig(type="list"),
        output=OutputConfig(type="session", inference_log_detail="full", session_artifact_tier="debug"),
        inference=InferenceConfig(type="http", model_name="lekai", inference_mode="sliding_window",
            server_generate_url=args.server.rstrip("/") + "/generate_accompaniment",
            model_condition_bpm=80, generation_interval_ticks=4, generation_length_frames=4,
            temperature=1.1, top_p=0.95, top_k=50, repetition_penalty=1.0),
        continuation_mode="standard", count_in_beats=4,
    )
    audit.save(out / "configuration.json", asdict(config))
    observer = FixedTickInput(events)
    runtime = RuntimeSessionBuilder(config=config, log_dir=str(out / "logs"),
                                    tick_observer_factory=lambda tempo: observer).build_cli()
    observer.service = runtime.service
    audit.save(out / "input_contract.json", {
        "source_midi_sha256": case["midi_sha256"], "clock": "service logical tick",
        "input_adapter": "empty ListInput plus audit-only FixedTickInput observer",
        "wallclock_requantization": False, "future_events_preloaded_into_service": False,
        "stop_tick": case["stop_tick"], "expected_events": len(events),
    })
    try:
        runtime.start(run_stop_tick=case["stop_tick"])
        while runtime.running:
            time.sleep(0.1)
    finally:
        runtime.stop()
        runtime.cleanup()
        audit.save(out / "delivered_events.json", observer.delivered)
    assert observer.error is None, observer.error
    assert observer.delivered == [{**r, "injected_at_logical_tick": r["tick"]} for r in events]
    audit.save(out / "status.json", {"status": "completed", "session": str(runtime.session_dir),
                                      "delivered_events": len(observer.delivered)})


def analyze_piece(root, piece):
    plan = audit.read(root / "plan.json")
    case = next(c for c in plan["cases"] if c["piece"] == piece)
    calls = audit.rows(root / "server/backend_calls.jsonl")
    samples = audit.rows(root / "server/sampling_observer.jsonl")
    repeats = []
    for repeat in (1, 2):
        out = root / "cases" / piece / f"repeat_{repeat}"
        status = audit.read(out / "status.json")
        assert status["status"] == "completed"
        reset = audit.read(out / "session_initialized.json")
        assert reset["effective_seed"] == plan["seed"]
        own = [r for r in calls if r["session_id"] == reset["session_id"]]
        assert own and all(r["effective_seed"] == plan["seed"] for r in own)
        mapped = {r["request"]["generation_start_tick"]: r for r in own}
        assert len(mapped) == len(own)
        sampler = [{k: r[k] for k in ("prompt_tokens", "sampling", "samples", "returned_tokens")}
                   for r in samples if r["session_id"] == reset["session_id"]]
        assert sampler
        session = next((out / "logs").glob("*/session_*"))
        repeats.append({"calls": mapped, "samples": sampler,
                        "delivered": audit.read(out / "delivered_events.json"),
                        "playback": audit.midi_notes(session / "combined.mid")})
    a, b = repeats
    common = sorted(set(a["calls"]) & set(b["calls"]))
    fields = ["part0_roll_sha256", "part0_tokens", "prompt_tokens", "raw_tokens", "structural_tokens"]
    different = {k: [] for k in fields + ["input_events", "raw_output"]}
    for tick in common:
        x, y = a["calls"][tick], b["calls"][tick]
        for field in fields:
            if x["trace"][field] != y["trace"][field]:
                different[field].append(tick)
        if x["request"]["melody_events"] != y["request"]["melody_events"]:
            different["input_events"].append(tick)
        if x["raw_output"] != y["raw_output"]:
            different["raw_output"].append(tick)
    frozen = audit.read(root / "inputs" / piece / "fixed_events.json")
    expected = list(range(4, case["stop_tick"] + 1, 4))
    if any(e["tick"] == 0 for e in frozen):
        expected.insert(0, 0)
    missing = [sorted(set(expected) - set(r["calls"])) for r in repeats]
    unexpected = [sorted(set(r["calls"]) - set(expected)) for r in repeats]
    contract_exact = not any(missing) and not any(unexpected)
    result = {"piece": piece, "expected_generation_calls": len(expected),
        "generation_calls": [len(a["calls"]), len(b["calls"])],
        "missing_generation_ticks": missing, "unexpected_generation_ticks": unexpected,
        "delivered_events_exact": a["delivered"] == b["delivered"],
        "mismatched_generation_ticks": different,
        "equal_call_counts": {k: len(common) - len(v) for k, v in different.items()},
        "sampler_exact": a["samples"] == b["samples"],
        "backend_exact": contract_exact and not any(different.values()),
        "raw_note_on": [sum(e["type"] == "note_on" for c in r["calls"].values()
                            for e in c["raw_output"]) for r in repeats],
        "playback_exact": a["playback"] == b["playback"],
        "playback_notes": [len(a["playback"]), len(b["playback"])],
        "common_playback_notes": sum((Counter(a["playback"]) & Counter(b["playback"])).values())}
    audit.save(root / "cases" / piece / "comparison.json", result)
    return result


def run(args):
    plan = audit.read(args.root / "plan.json")
    selected = args.pieces or [plan["cases"][0]["piece"]]
    assert set(selected) <= {c["piece"] for c in plan["cases"]}
    progress = {"status": "running", "pieces": [], "selected": selected}
    audit.save(args.root / "progress.json", progress)
    for piece in selected:
        entry = {"piece": piece, "status": "running", "repeats_completed": 0}
        progress["pieces"].append(entry)
        try:
            for repeat in (1, 2):
                audit.save(args.root / "progress.json", progress)
                print(f"START {piece} repeat {repeat}", flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "client", "--root", str(args.root),
                           "--piece", piece, "--repeat", str(repeat), "--server", args.server]
                env = {k: v for k, v in os.environ.items() if not k.startswith("LEKAI_")}
                env.update(CUDA_VISIBLE_DEVICES="", LEKAI_EFFECTIVE_SEED=str(plan["seed"]))
                log = args.root / f"{piece}_repeat{repeat}.log"
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                entry["repeats_completed"] += 1
            comparison = analyze_piece(args.root, piece)
            entry.update(status="completed", comparison=comparison)
            if not comparison["backend_exact"] or not comparison["sampler_exact"]:
                print(json.dumps(comparison), flush=True)
                if not args.continue_on_difference:
                    progress["status"] = "stopped_on_difference"
                    audit.save(args.root / "progress.json", progress)
                    return
        except BaseException as exc:
            entry.update(status="failed", error=repr(exc))
            progress["status"] = "failed"
            audit.save(args.root / "progress.json", progress)
            raise
        audit.save(args.root / "progress.json", progress)
    has_difference = any(not all(e["comparison"][key] for key in
                                 ("delivered_events_exact", "backend_exact", "sampler_exact", "playback_exact"))
                         for e in progress["pieces"])
    progress["status"] = "completed_with_differences" if has_difference else "completed"
    audit.save(args.root / "progress.json", progress)
    print(json.dumps(progress), flush=True)


def launch(args):
    if not (args.root / "plan.json").is_file() or (args.root / "server").exists():
        raise RuntimeError("Need prepared inputs and an unused server output directory")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if processes.strip():
        raise RuntimeError("GPU already has compute processes")
    url = urlsplit(args.server)
    if url.hostname != "127.0.0.1":
        raise ValueError("This same-host launcher requires localhost")
    with socket.socket() as port_check:
        port_check.bind((url.hostname, url.port))
    checkpoint = Path(args.checkpoint)
    assert audit.sha(checkpoint.read_bytes()) == "d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271"
    snapshot = args.root / "executed_scripts"
    snapshot.mkdir()
    for name in (Path(__file__).name, "plain_lekai_device_replay_acceptance.py", "device_replay_acceptance.py",
                 "human6_timed_input_replay.py", "summarize_human_input_timing.py"):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    command = [sys.executable, "-u", str(Path(__file__).with_name("plain_lekai_device_replay_acceptance.py")),
               "server", "--out", str(args.root / "server"), "--gpu", "0", "--host", url.hostname,
               "--port", str(url.port), "--checkpoint-path", str(checkpoint), "--system-ref", "ec843a79"]
    with (args.root / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Backend exited during startup")
                try:
                    info = audit.api(args.server, "/runtime_info")
                    assert info["has_real_model"]
                    print("Real backend ready", flush=True)
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("Backend not ready")
            run(args)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "client", "run", "launch"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--piece")
    parser.add_argument("--pieces", nargs="+")
    parser.add_argument("--continue-on-difference", action="store_true",
                        help="Record mismatches and finish remaining pieces without retrying failed comparisons")
    parser.add_argument("--repeat", type=int, choices=[1, 2])
    parser.add_argument("--server", default="http://127.0.0.1:18846")
    parser.add_argument("--checkpoint", default="/home/xiaosongma/models/streammuse_prompt_continuation/continuation/model.safetensors")
    args = parser.parse_args()
    {"prepare": prepare, "client": client, "run": run, "launch": launch}[args.mode](args)

"""Observe ordinary Realtime A/B/C without changing service or model behavior."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

from device_replay_acceptance import (
    CONT_CKPT, REPO, api, identity, observer_setup, read, save, sha,
    source_events, wait_for_first_frame,
)


def extract_melody(source, destination):
    import mido
    midi = mido.MidiFile(source)
    melody = [t for t in midi.tracks if t.name == "Melody"]
    if len(melody) != 1:
        raise ValueError(f"Expected one Melody track, got {[t.name for t in midi.tracks]}")
    result = mido.MidiFile(type=1, ticks_per_beat=midi.ticks_per_beat)
    result.tracks = [t.copy() for t in midi.tracks
                     if t is melody[0] or not any(m.type in ("note_on", "note_off") for m in t)]
    result.save(destination)
    save(Path(destination).with_suffix(".json"), {
        "source": str(source), "source_sha256": sha(Path(source).read_bytes()),
        "method": "Copy exported Melody and metadata tracks; no retiming, no re-encoding",
        "midi_sha256": sha(Path(destination).read_bytes()),
    })


def midi_notes(path, track_name="Accompaniment"):
    """Compare exact file positions, avoiding rounded microsecond tempo errors."""
    import mido
    midi = mido.MidiFile(path)
    tracks = [t for t in midi.tracks if t.name == track_name]
    if len(tracks) != 1:
        raise ValueError(f"Expected one {track_name} track in {path}")
    active, notes, tick = {}, [], 0
    for msg in tracks[0]:
        tick += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            if key in active:
                raise ValueError(f"Unclosed retrigger in exported MIDI: {path}, {key}")
            active[key] = (tick, msg.velocity)
        else:
            if key not in active:
                raise ValueError(f"Orphan note_off in exported MIDI: {path}, {key}")
            start, velocity = active.pop(key)
            notes.append((msg.note, float(Fraction(start * 4, midi.ticks_per_beat)),
                          float(Fraction(tick * 4, midi.ticks_per_beat)), velocity))
    if active:
        raise ValueError(f"Unclosed notes in exported MIDI: {path}")
    return sorted(notes)


def timed_messages(path):
    """Retain application-arrival time and sequence; never derive time from tick."""
    import mido
    records = rows(Path(path))
    if not records:
        raise ValueError("Empty timed input")
    previous_time = -1.0
    for index, record in enumerate(records, start=1):
        elapsed = float(record["elapsed_seconds"])
        assert record["event_sequence"] == index
        assert elapsed >= previous_time and elapsed >= 0
        assert record["clock_domain"] == "service_now"
        assert float(record["bpm"]) == 90 and int(record["ticks_per_beat"]) == 4
        assert float(record["snap_forward_fraction"]) == 0.4
        assert int(record["channel"]) == 0
        assert abs(elapsed - (record["application_received_time_s"] - record["timeline_start_time_s"])) < 1e-6
        previous_time = elapsed
    return records, [(float(r["elapsed_seconds"]), mido.Message(
        r["event_type"], note=int(r["pitch"]), velocity=int(r["velocity"]),
        channel=int(r["channel"]), time=0)) for r in records]


def timeline_anchor(session):
    first = next(r for r in rows(session / "system_trace.jsonl")
                 if r["record_type"] == "frame_deadline" and r["tick"] == 0)
    assert first["clock_domain"] == "service_now"
    return float(first["nominal_tick_time_s"])


def server(args):
    checkpoint = str(Path(args.checkpoint_path).resolve())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    args.out.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith("LEKAI_"):
            del os.environ[key]
    env = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "LEKAI_DEVICE": "cuda", "LEKAI_DTYPE": "float16",
        "LEKAI_CHECKPOINT_PATH": checkpoint, "LEKAI_DISABLE_FALLBACK": "1",
        "LEKAI_ENABLE_DEBUG_RESET": "true", "LEKAI_RT_SEED": str(args.seed),
        "LEKAI_RT_TEMPERATURE": "1.1", "LEKAI_RT_TOP_P": "0.95", "LEKAI_RT_TOP_K": "50",
        "LEKAI_RT_REPETITION_PENALTY": "1", "LEKAI_DEFAULT_BPM": "80",
        "LEKAI_PROMPT_CONTEXT_BEATS": "32", "LEKAI_HISTORY_MAX_TICKS": "128",
        "LEKAI_CONTINUATION_TONAL_CONSTRAINT": "0", "LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD": "0",
        "LEKAI_DETERMINISTIC_BOUNDARY_GENERATION": "0",
        "LEKAI_SERVER_HOST": args.host, "LEKAI_SERVER_PORT": str(args.port),
        "LEKAI_RT_LOG_DIR": str(args.out / "generation"),
    }
    os.environ.update(env)
    sys.path.insert(0, str(REPO / "src"))
    save(args.out / "identity.json", identity(args.system_ref))
    save(args.out / "environment.json", env)
    save(args.out / "pid.json", {"pid": os.getpid()})
    observer_setup(args.out)

    from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
    original = LekaiHttpBackend.generate
    lock = threading.Lock()

    def observed(self, *positional, **kwargs):
        result = original(self, *positional, **kwargs)
        trace = dict(self._current_generation_trace)
        roll = trace.pop("part0_roll", None)
        if roll is not None:
            trace["part0_roll_shape"] = list(roll.shape)
            trace["part0_roll_dtype"] = str(roll.dtype)
            trace["part0_roll_base64"] = base64.b64encode(roll.tobytes()).decode()
            trace["part0_roll_sha256"] = sha(roll.tobytes())
        row = {"session_id": self._session_id, "session_epoch": self._session_epoch,
               "effective_seed": self._effective_seed, "request": kwargs,
               "raw_output": result[0], "timings": result[1], "trace": trace}
        with lock, (args.out / "backend_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        return result

    LekaiHttpBackend.generate = observed
    import torch
    import transformers
    save(args.out / "libraries.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "transformers_file": transformers.__file__,
        "gpu": torch.cuda.get_device_name(0),
        "scope": "Observers synchronize tensors. Not a latency benchmark or physical audio test.",
    })
    from streammuse.infrastructure.inference import server_lekai
    runtime = server_lekai.backend.runtime_info()
    save(args.out / "loaded_runtime.json", runtime)
    assert runtime["has_real_model"] and runtime["checkpoint_path"] == checkpoint
    server_lekai.main()


def one_client(args, label, midi=None):
    import mido
    out = args.out / label
    out.mkdir(parents=True, exist_ok=False)
    reset = api(args.server, "/debug/reset_session", {"seed": args.seed})
    assert reset["effective_seed"] == args.seed
    save(out / "session_initialized.json", reset)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LEKAI_")}
    env.update({"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "",
                "LEKAI_EFFECTIVE_SEED": str(args.seed)})
    command = [sys.executable, "-m", "streammuse.presentation.cli.cli",
        "--tempo", "90", "--model-condition-bpm", "80", "--ticks-per-beat", "4", "--beats-per-bar", "4",
        "--model-name", "lekai", "--continuation-mode", "standard", "--inference-mode", "sliding_window",
        "--server-url", args.server.rstrip("/") + "/generate_accompaniment",
        "--temperature", "1.1", "--top-p", "0.95", "--top-k", "50", "--repetition-penalty", "1",
        "--count-in-beats", "4", "--input-snap-forward-fraction", "0.4",
        "--generation-interval-ticks", "4", "--generation-length-frames", "4",
        "--run-stop-tick", str(args.stop_tick), "--output-type", "session", "--session-artifact-tier", "debug",
        "--log-dir", str(out / "logs"), "--log-input-quantization", "--inference-log-detail", "full"]
    sender = None
    process = None
    try:
        if midi is None:
            name = "codex_plain_acceptance_" + uuid.uuid4().hex[:8]
            sender = mido.open_output(name, virtual=True)
            ports = [p for p in mido.get_input_names() if name in p]
            assert len(ports) == 1
            command += ["--input-mode", "midi_device", "--midi-device-name", ports[0]]
        else:
            command += ["--input-mode", "midi_file", "--midi-file-path", str(midi)]
        save(out / "command.json", command)
        save(out / "runtime_before.json", api(args.server, "/runtime_info"))
        with (out / "client.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            session = wait_for_first_frame(out / "logs", process)
            if sender is not None:
                if args.timed_input is not None:
                    records, planned = timed_messages(args.timed_input)
                    start = timeline_anchor(session)
                    now = time.time
                    if now() >= start + planned[0][0]:
                        raise RuntimeError("First source event deadline passed before sender was ready")
                    save(out / "sender_anchor.json", {
                        "clock": "time.time / service_now", "timeline_start_time_s": start,
                        "source": str(args.timed_input), "event_count": len(records),
                        "source_sha256": sha(args.timed_input.read_bytes()),
                        "retimed": False, "trimmed": False,
                    })
                else:
                    planned = source_events(REPO / "prompts/old_input/mel/001.mid")
                    start = time.monotonic() + 0.03
                    now = time.monotonic
                with (out / "sent_messages.jsonl").open("w", encoding="utf-8") as sent:
                    for sequence, (seconds, msg) in enumerate(planned, start=1):
                        while now() < start + seconds:
                            if process.poll() is not None:
                                raise RuntimeError("CLI exited during virtual MIDI input")
                            time.sleep(min(0.001, max(0, start + seconds - now())))
                        actual = now()
                        sender.send(msg)
                        sent.write(json.dumps({"event_sequence": sequence, "planned_seconds": seconds,
                            "actual_relative_seconds": actual - start, "message": msg.dict()}) + "\n")
            if process.wait(timeout=args.stop_tick / 6 + 90):
                raise RuntimeError(f"CLI failed; inspect {out / 'client.log'}")
        save(out / "runtime_after.json", api(args.server, "/runtime_info"))
        save(out / "status.json", {"status": "completed", "session": str(session)})
        return session
    finally:
        if sender is not None:
            sender.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def clients(args):
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "identity.json", identity(args.system_ref))
    runtime = api(args.server, "/runtime_info")
    assert runtime["has_real_model"] and runtime["checkpoint_path"] == CONT_CKPT
    save(args.out / "server_before.json", runtime)
    source = args.timed_input or REPO / "prompts/old_input/mel/001.mid"
    save(args.out / "contract.json", {
        "source": str(source), "source_sha256": sha(source.read_bytes()),
        "source_window": ("Complete recorded application-arrival timeline, unchanged"
                          if args.timed_input else "First 24 beats relative to first note_on; close held notes at end"),
        "playback_bpm": 90, "model_bpm": 80, "count_in_beats": 4,
        "generation_interval_ticks": 4, "generation_length_frames": 4,
        "stop_tick_exclusive": args.stop_tick,
        "seed": args.seed, "prompt_model": False, "constraints": False,
        "system_ref": args.system_ref, "automatic_virtual_device": True,
        "human_or_audio_hardware_validation": False, "latency_benchmark": False,
    })
    a = one_client(args, "A_virtual_device")
    melody = args.out / "A_exported_melody.mid"
    extract_melody(a / "combined.mid", melody)
    b = one_client(args, "B_midi_replay", melody)
    sessions = {"A": str(a), "B": str(b), "seed": args.seed}
    if not args.skip_repeat:
        sessions["C"] = str(one_client(args, "C_midi_replay_repeat", melody))
    save(args.out / "sessions.json", sessions)
    save(args.out / "status.json", {"status": "completed", "comparison_pending": True})
    print(json.dumps(read(args.out / "sessions.json")), flush=True)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(args):
    root = args.out
    backend_dir = args.backend_dir or root / "server"
    calls = rows(backend_dir / "backend_calls.jsonl")
    samples = rows(backend_dir / "sampling_observer.jsonl")
    client = root / "client"
    labels = {"A": "A_virtual_device", "B": "B_midi_replay", "C": "C_midi_replay_repeat"}
    labels = {k: v for k, v in labels.items() if (client / v / "status.json").is_file()}
    assert "A" in labels and "B" in labels
    comparisons = [("A", "B")] + ([("B", "C")] if "C" in labels else [])
    datasets = {}
    for label, dirname in labels.items():
        reset = read(client / dirname / "session_initialized.json")
        session = next((client / dirname / "logs").glob("*/session_*"))
        own = [r for r in calls if r["session_id"] == reset["session_id"]]
        assert own, label
        assert all(r["effective_seed"] == reset["effective_seed"] for r in own)
        mapped = {r["request"]["generation_start_tick"]: r for r in own}
        assert len(mapped) == len(own), "Repeated generation tick must be inspected"
        own_samples = [r for r in samples if r["session_id"] == reset["session_id"]]
        assert own_samples and all(r["seed"] == reset["effective_seed"] for r in own_samples)
        datasets[label] = {"calls": mapped, "samples": own_samples, "session": session}
    result = {"counts": {}, "comparisons": {}}
    for label, d in datasets.items():
        result["counts"][label] = {
            "generation_calls": len(d["calls"]), "sampling_calls": len(d["samples"]),
            "sampling_steps": sum(len(r["samples"]) for r in d["samples"]),
            "raw_note_on": sum(e["type"] == "note_on" for r in d["calls"].values() for e in r["raw_output"]),
            "quantization_events": len(rows(d["session"] / "input_quantization_trace.jsonl")),
        }
    for left, right in comparisons:
        a, b = datasets[left], datasets[right]
        common = sorted(set(a["calls"]) & set(b["calls"]))
        fields = ["part0_roll_sha256", "part0_tokens", "prompt_tokens", "raw_tokens", "structural_tokens"]
        mismatches = {key: [] for key in fields + ["raw_output", "input_events"]}
        for tick in common:
            x, y = a["calls"][tick], b["calls"][tick]
            for key in fields:
                if key not in x["trace"] or key not in y["trace"]:
                    raise ValueError(f"Missing {key} at {tick}")
                if x["trace"][key] != y["trace"][key]:
                    mismatches[key].append(tick)
            if x["raw_output"] != y["raw_output"]:
                mismatches["raw_output"].append(tick)
            if x["request"]["melody_events"] != y["request"]["melody_events"]:
                mismatches["input_events"].append(tick)
        def clean_sampler(data):
            return [{k: r[k] for k in ("prompt_tokens", "sampling", "samples", "returned_tokens")}
                    for r in data]
        result["comparisons"][left + "_vs_" + right] = {
            "common_generation_ticks": common,
            "left_only": sorted(set(a["calls"]) - set(b["calls"])),
            "right_only": sorted(set(b["calls"]) - set(a["calls"])),
            "mismatched_ticks": mismatches,
            "equal_call_counts": {k: len(common) - len(v) for k, v in mismatches.items()},
            "sampler_exact": clean_sampler(a["samples"]) == clean_sampler(b["samples"]),
        }
    for label, d in datasets.items():
        d["playback_notes"] = midi_notes(d["session"] / "combined.mid")
        result["counts"][label]["playback_notes"] = len(d["playback_notes"])
        schedule = rows(d["session"] / "model_schedule_trace.jsonl")
        result["counts"][label]["schedule_policies"] = dict(Counter(r["policy"] for r in schedule))
        inferences = read(d["session"] / "inferences.json")
        result["counts"][label]["inference_log_calls"] = len(inferences)
        raw = [{"generation_start_tick": t, "events": r["raw_output"]}
               for t, r in sorted(d["calls"].items())]
        save(root / f"{label}_backend_raw_by_request.json", raw)
    for left, right in comparisons:
        a, b = datasets[left]["playback_notes"], datasets[right]["playback_notes"]
        result["comparisons"][left + "_vs_" + right]["playback"] = {
            "exact": a == b, "common_notes": sum((Counter(a) & Counter(b)).values()),
            "left_only": list((Counter(a) - Counter(b)).elements()),
            "right_only": list((Counter(b) - Counter(a)).elements()),
        }
    save(root / "analysis.json", result)
    listening = root / "listening"
    listening.mkdir(exist_ok=True)
    shutil.copy2(client / "A_exported_melody.mid", listening / "00_exported_melody.mid")
    for label, d in datasets.items():
        shutil.copy2(d["session"] / "combined.mid", listening / f"{label}_playback.mid")
    print(json.dumps({"counts": result["counts"], "comparisons": {
        name: {k: v for k, v in report.items() if k != "playback"} |
              {"playback": {k: v for k, v in report["playback"].items()
                            if k not in ("left_only", "right_only")}}
        for name, report in result["comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "clients", "analyze"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--system-ref", default="ec843a79")
    parser.add_argument("--seed", type=int, default=1051154023138951872)
    parser.add_argument("--gpu", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--port", type=int, default=18844)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint-path", type=Path, default=Path(CONT_CKPT))
    parser.add_argument("--server", default="http://10.127.30.168:18844")
    parser.add_argument("--timed-input", type=Path)
    parser.add_argument("--stop-tick", type=int, default=160)
    parser.add_argument("--skip-repeat", action="store_true")
    parser.add_argument("--backend-dir", type=Path)
    args = parser.parse_args()
    {"server": server, "clients": clients, "analyze": analyze}[args.mode](args)

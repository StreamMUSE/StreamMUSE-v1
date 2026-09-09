"""Isolated virtual-MIDI / exported-MIDI acceptance test; no system edits."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "2303f341d13a9a6b415335afc69c839d33a8f5f9"
PROMPT_CKPT = "/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors"
CONT_CKPT = "/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors"


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def api(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def identity():
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", BASE_COMMIT, "--", "src", "transformers"])
    if diff:
        raise RuntimeError("System source differs from the pinned trusted base")
    files = ["src/streammuse/infrastructure/input/midi_device.py",
             "src/streammuse/infrastructure/input/midi_file.py",
             "src/streammuse/infrastructure/inference/lekai_http_backend.py",
             "src/streammuse/application/services/prompt_continuation_realtime_service.py"]
    return {"repo": str(REPO), "host": platform.node(), "python": sys.version,
            "head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "base": BASE_COMMIT, "system_source_unchanged": True,
            "source_sha256": {p: sha((REPO / p).read_bytes()) for p in files}}


def observer_setup(out):
    import torch
    from streammuse.infrastructure.inference import lekai_http_backend as backend_module
    from streammuse.infrastructure.inference.lekai_model import generation_utils

    state = threading.local()
    write_lock = threading.Lock()
    original_sample = generation_utils.sample_token
    original_generate = backend_module.LekaiHttpBackend._generate_part1_tokens_from_prompt

    def sampled(logits, *args, **kwargs):
        context = getattr(state, "context", None)
        generator = kwargs.get("generator")
        if context is None or generator is None:
            return original_sample(logits, *args, **kwargs)
        before = generator.get_state().cpu().numpy().tobytes()
        raw_logits = logits.detach().cpu().contiguous()
        raw_bytes = raw_logits.numpy().tobytes()
        token = original_sample(logits, *args, **kwargs)
        after = generator.get_state().cpu().numpy().tobytes()
        context["samples"].append({
            "index": len(context["samples"]) + 1,
            "rng_before_sha256": sha(before), "rng_after_sha256": sha(after),
            "rng_before_base64": base64.b64encode(before).decode(),
            "logits_sha256": sha(raw_bytes), "logits_dtype": str(raw_logits.dtype),
            "logits_shape": list(raw_logits.shape),
            "logits_base64": base64.b64encode(raw_bytes).decode(),
            "token": int(token.item()),
        })
        return token

    def generated(self, prompt_tokens, **kwargs):
        context = {"session_epoch": self._session_epoch, "session_id": self._session_id,
                   "seed": self._effective_seed, "prompt_tokens": prompt_tokens.tolist(),
                   "sampling": {k: sorted(v) if isinstance(v, set) else v for k, v in kwargs.items()},
                   "samples": [], "started_monotonic": time.monotonic()}
        state.context = context
        try:
            result = original_generate(self, prompt_tokens, **kwargs)
            context["returned_tokens"] = result
            assert result == [x["token"] for x in context["samples"]]
            return result
        except Exception as exc:
            context["error"] = repr(exc)
            raise
        finally:
            state.context = None
            context["finished_monotonic"] = time.monotonic()
            with write_lock:
                with (out / "sampling_observer.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(context) + "\n")

    generation_utils.sample_token = sampled
    backend_module.LekaiHttpBackend._generate_part1_tokens_from_prompt = generated


def server(args):
    args.out.mkdir(parents=True, exist_ok=False)
    env = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "LEKAI_DEVICE": "cuda", "LEKAI_DTYPE": "float16",
        "LEKAI_PROMPT_CHECKPOINT_PATH": PROMPT_CKPT, "LEKAI_CONTINUATION_CHECKPOINT_PATH": CONT_CKPT,
        "LEKAI_PROMPT_CONTINUATION_ENGINE": "standard", "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS": "1",
        "LEKAI_DISABLE_FALLBACK": "1", "LEKAI_PROMPT_SELECTION_MODE": "rule_s_if_else",
        "LEKAI_PROMPT_BATCH_CANDIDATES": "10", "LEKAI_PROMPT_CONTEXT_BEATS": "32",
        "LEKAI_HISTORY_MAX_TICKS": "128", "LEKAI_DEFAULT_BPM": "80", "LEKAI_PROMPT_BPM": "80",
        "LEKAI_CONTINUATION_TONAL_CONSTRAINT": "1", "LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD": "1",
        "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
        "LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY": "0",
        "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
        "LEKAI_DETERMINISTIC_BOUNDARY_GENERATION": "0",
        "LEKAI_SERVER_HOST": "0.0.0.0", "LEKAI_SERVER_PORT": str(args.port),
        "LEKAI_RT_LOG_DIR": str(args.out / "generation"),
    }
    for prefix in ("LEKAI_PROMPT", "LEKAI_RT"):
        env.update({prefix + "_TEMPERATURE": "1.1", prefix + "_TOP_P": "0.95",
                    prefix + "_TOP_K": "50", prefix + "_REPETITION_PENALTY": "1.0"})
    for key in list(os.environ):
        if key.startswith("LEKAI_"):
            del os.environ[key]
    os.environ.update(env)
    os.environ["STREAMMUSE_CODE_SHA"] = identity()["head"]
    sys.path.insert(0, str(REPO / "src"))
    save(args.out / "identity.json", identity())
    save(args.out / "environment.json", env)
    save(args.out / "pid.json", {"pid": os.getpid()})
    observer_setup(args.out)
    import torch
    import transformers
    save(args.out / "libraries.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "transformers_file": transformers.__file__,
        "gpu": torch.cuda.get_device_name(0), "sampling_observer": True,
        "scope": "Diagnostic only: observer synchronizes tensors; not a latency benchmark."})
    from streammuse.infrastructure.inference import server_lekai
    save(args.out / "loaded_runtime.json", server_lekai.prompt_continuation_backend.runtime_info())
    server_lekai.main()


def source_events(path, beats=24):
    import mido
    midi = mido.MidiFile(path)
    items, tick = [], 0
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        if msg.type in ("note_on", "note_off"):
            items.append((tick, msg))
    start = next(t for t, m in items if m.type == "note_on" and m.velocity > 0)
    stop = start + beats * midi.ticks_per_beat
    active = set()
    result = []
    for t, msg in items:
        if not start <= t < stop:
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            active.add(key)
        else:
            active.discard(key)
        result.append(((t-start) / midi.ticks_per_beat * (60/90), msg.copy(time=0)))
    for channel, note in sorted(active):
        result.append((beats * 60/90, mido.Message("note_off", channel=channel, note=note, velocity=0)))
    return result


def wait_for_first_frame(logs, process):
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("CLI exited before the first frame; inspect client.log")
        for p in logs.rglob("system_trace.jsonl"):
            if '"frame_deadline"' in p.read_text(encoding="utf-8"):
                return p.parent
        time.sleep(0.02)
    raise TimeoutError("CLI did not reach its first frame")


def one_client(args, label, midi=None, seed=None, source=None):
    import mido
    out = args.out / label
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LEKAI_")}
    env.update({"PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""})
    env.update({"LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
                "LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY": "0",
                "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0"})
    # The bare CLI adopts a session; Web Start normally performs this API call.
    initialized = api(args.server, "/prompt_continuation/session/initialize",
                      {} if seed is None else {"prompt_seed": seed, "continuation_seed": seed})
    save(out / "session_initialized.json", initialized)
    for field, name in {"prompt_requested_seed":"LEKAI_PROMPT_REQUESTED_SEED",
        "prompt_effective_seed":"LEKAI_PROMPT_EFFECTIVE_SEED",
        "continuation_requested_seed":"LEKAI_CONTINUATION_REQUESTED_SEED",
        "continuation_effective_seed":"LEKAI_CONTINUATION_EFFECTIVE_SEED",
        "session_id":"LEKAI_PROMPT_SESSION_ID", "session_epoch":"LEKAI_PROMPT_SESSION_EPOCH"}.items():
        env[name] = str(initialized[field])
    command = [sys.executable, "-m", "streammuse.presentation.cli.cli",
        "--tempo", "90", "--model-condition-bpm", "80", "--ticks-per-beat", "4", "--beats-per-bar", "4",
        "--model-name", "lekai", "--continuation-mode", "prompt_continuation",
        "--server-url", args.server.rstrip("/") + "/generate_accompaniment",
        "--prompt-selection-mode", "rule_s_if_else", "--prompt-batch-candidates", "10",
        "--temperature", "1.1", "--top-p", "0.95", "--top-k", "50", "--repetition-penalty", "1",
        "--count-in-beats", "4", "--input-snap-forward-fraction", "0.4",
        "--generation-interval-ticks", "2", "--generation-length-frames", "20",
        "--prompt-length-ticks", "32", "--run-stop-tick", "160", "--output-type", "session",
        "--log-dir", str(out / "logs"), "--log-input-quantization", "--inference-log-detail", "full"]
    sender = None
    if midi is None:
        name = "codex_acceptance_" + uuid.uuid4().hex[:8]
        sender = mido.open_output(name, virtual=True)
        ports = [p for p in mido.get_input_names() if name in p]
        assert len(ports) == 1
        command += ["--input-mode", "midi_device", "--midi-device-name", ports[0]]
    else:
        command += ["--input-mode", "midi_file", "--midi-file-path", str(midi)]
    save(out / "command.json", command)
    save(out / "environment.json", {k:v for k,v in env.items()
                                    if k.startswith("LEKAI_") or k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")})
    process = None
    try:
        with (out / "client.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            session = wait_for_first_frame(out / "logs", process)
            if sender is not None:
                planned = source_events(source)
                start = time.monotonic() + 0.03
                with (out / "sent_messages.jsonl").open("w", encoding="utf-8") as sent:
                    for seconds, msg in planned:
                        while time.monotonic() < start + seconds:
                            if process.poll() is not None:
                                raise RuntimeError("CLI exited while sending MIDI")
                            time.sleep(min(0.003, max(0, start + seconds - time.monotonic())))
                        actual = time.monotonic()
                        sender.send(msg)
                        sent.write(json.dumps({"planned_seconds": seconds, "actual_relative_seconds": actual-start,
                                               "message": msg.dict()}) + "\n")
            code = process.wait(timeout=90)
            if code:
                raise RuntimeError(f"CLI failed with exit {code}: {out}")
        save(out / "post_cli_audit.json", api(args.server, "/prompt_continuation/replay_audit"))
        save(out / "status.json", {"status":"completed", "session":str(session)})
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
    save(args.out / "identity.json", identity())
    runtime = api(args.server, "/prompt_continuation/runtime_info")
    save(args.out / "server_before.json", runtime)
    assert runtime["prompt_has_real_model"] and runtime["has_real_model"]
    assert runtime["prompt_selection_mode"] == "rule_s_if_else"
    source = REPO / "prompts/old_input/mel/001.mid"
    save(args.out / "contract.json", {"source":str(source), "source_sha256":sha(source.read_bytes()),
        "source_window":"first 24 beats relative to first note_on; close held notes at window end",
        "source_velocity":"preserved", "automatic_device_input":True, "new_human_performance":False,
        "server":args.server, "device_run_seed":"generated by existing session initialize API",
        "replay_seed":"copied from device run", "system_code_unchanged":True,
        "hardware_audio_test":False, "latency_benchmark":False})
    a = one_client(args, "A_virtual_device", source=source)
    seed_record = read(a / "prompt_continuation_session_seed.json")
    seed = int(seed_record["continuation_effective_seed"])
    assert seed == int(seed_record["prompt_effective_seed"])
    midi = a / "prompt_continuation_replay_melody.mid"
    assert midi.is_file()
    b = one_client(args, "B_midi_replay", midi=midi, seed=seed)
    c = one_client(args, "C_midi_replay_repeat", midi=midi, seed=seed)
    save(args.out / "sessions.json", {"A":str(a), "B":str(b), "C":str(c), "seed":seed})
    for label, left, right in [("A_vs_B",a,b), ("B_vs_C",b,c)]:
        proc = subprocess.run([sys.executable, str(REPO / "scripts/compare_realtime_replay_exact.py"),
            str(left), str(right), "--output", str(args.out / f"{label}_strict.json")],
            capture_output=True, text=True)
        save(args.out / f"{label}_comparator_process.json", {"exit_code":proc.returncode,"stderr":proc.stderr})
        if proc.returncode not in (0,1,2):
            raise RuntimeError(proc.stderr)
    save(args.out / "status.json", {"status":"completed", "strict_comparison_results_must_be_read":True})
    print(json.dumps(read(args.out / "sessions.json")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "clients"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=[0,1,2], default=0)
    parser.add_argument("--port", type=int, default=18791)
    parser.add_argument("--server", default="http://10.127.30.168:18791")
    args = parser.parse_args()
    if args.mode == "server":
        server(args)
    else:
        clients(args)


if __name__ == "__main__":
    main()

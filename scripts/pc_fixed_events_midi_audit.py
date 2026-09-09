"""Audit P+C raw equality without changing production model or scheduler code."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import device_replay_acceptance as common
from fixed_tick_midi_replay import load_midi_events
from plain_lekai_device_replay_acceptance import midi_notes

SYSTEM_REF = "ec843a79"
SEED_FIELDS = {
    "prompt_requested_seed": "LEKAI_PROMPT_REQUESTED_SEED",
    "prompt_effective_seed": "LEKAI_PROMPT_EFFECTIVE_SEED",
    "continuation_requested_seed": "LEKAI_CONTINUATION_REQUESTED_SEED",
    "continuation_effective_seed": "LEKAI_CONTINUATION_EFFECTIVE_SEED",
    "session_id": "LEKAI_PROMPT_SESSION_ID",
    "session_epoch": "LEKAI_PROMPT_SESSION_EPOCH",
}


class FixedPromptInput:
    """Audit adapter: current-tick events enter the existing two input buffers."""

    def __init__(self, service, events):
        self.service = service
        self.events = events
        self.tick = 0
        self.next_event = 0
        self.delivered = []
        self.original_drain = service._drain_user_events

    def drain(self):
        from streammuse.domain.musical import MusicalEvent, EventType
        with self.service._input_window_lock:
            while self.next_event < len(self.events):
                row = self.events[self.next_event]
                if row["tick"] > self.tick:
                    break
                if row["tick"] != self.tick:
                    raise RuntimeError("Fixed input skipped an event tick")
                event = MusicalEvent(tick=row["tick"], pitch=row["pitch"],
                    event_type=EventType(row["event_type"]), velocity=row["velocity"],
                    channel=row["channel"], source="user")
                self.service._input_window_events.append(event)
                self.service._event_q.put(event)
                self.delivered.append({**row, "injected_at_logical_tick": self.tick})
                self.next_event += 1
        self.tick += 1
        return self.original_drain()


def install_raw_observer(out):
    from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
    original = LekaiHttpBackend.generate
    lock = threading.Lock()

    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        trace = dict(self._current_generation_trace)
        roll = trace.pop("part0_roll", None)
        if roll is not None:
            trace["part0_roll_sha256"] = common.sha(roll.tobytes())
            trace["part0_roll_shape"] = list(roll.shape)
        row = {"session_id": self._session_id, "effective_seed": self._effective_seed,
               "request": kwargs, "trace": trace, "raw_output": result[0]}
        with lock, (out / "backend_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        return result

    LekaiHttpBackend.generate = observed


def server(args):
    # Reuse the existing P+C server launcher, replacing its expensive logits
    # observer with a raw-output observer only. Sampling itself is untouched.
    common.observer_setup = install_raw_observer
    original_save = common.save

    def save_server_metadata(path, value):
        if Path(path).name == "libraries.json":
            value = {**value, "sampling_observer": False,
                     "raw_output_observer": True,
                     "scope": "Post-call raw logging only; not a latency benchmark."}
        original_save(path, value)

    common.save = save_server_metadata
    common.server(args)


def prepare(args):
    old = common.read(args.reference_plan)
    args.root.mkdir(parents=True, exist_ok=False)
    for case in old["cases"]:
        matches = list(args.source_root.glob(
            f"*/{case['source_session']}/prompt_continuation_replay_melody.mid"))
        if len(matches) != 1:
            raise ValueError((case["piece"], matches))
        events, metadata = load_midi_events(matches[0])
        assert metadata["midi_sha256"] == case["midi_sha256"]
        dest = args.root / "inputs" / case["piece"]
        dest.mkdir(parents=True)
        shutil.copy2(matches[0], dest / "melody.mid")
        common.save(dest / "fixed_events.json", events)
    plan = {**old, "condition": "pc_rule_if_else_fixed_events_vs_midi_file",
        "system_ref": SYSTEM_REF, "prompt_selection": "rule_s_if_else",
        "prompt_candidates": 10, "constraints": {"tonal": True, "empty_token_guard": True},
        "prompt_ticks": 32, "generation_interval_ticks": 4, "generation_length_frames": 4,
        "repeats": 2, "primary_endpoint": "per-call Backend raw and accumulated raw history",
        "scope": "Both input paths rerun on the same H200 per piece; not historical live output",
        "input_adapter": "Fixed current-tick injection before P+C queue drain; no wallclock requantization",
        "mismatch_policy": "Record differences and continue; never retry to obtain equality",
        "latency_benchmark": False}
    common.save(args.root / "plan.json", plan)
    print(json.dumps({"pieces": len(plan["cases"]), "runs": len(plan["cases"]) * 4}))


def client(args):
    from streammuse.application.config.models import (
        ApplicationConfig, TempoConfig, InputConfig, OutputConfig, InferenceConfig)
    from streammuse.application.runtime.builder import RuntimeSessionBuilder
    plan = common.read(args.root / "plan.json")
    case = next(x for x in plan["cases"] if x["piece"] == args.piece)
    out = args.root / "cases" / args.piece / f"{args.path_mode}_repeat_{args.repeat}"
    out.mkdir(parents=True, exist_ok=False)
    midi = args.root / "inputs" / args.piece / "melody.mid"
    events, meta = load_midi_events(midi)
    assert meta["midi_sha256"] == case["midi_sha256"]
    reset = common.api(args.server, "/prompt_continuation/session/initialize",
        {"prompt_seed": plan["seed"], "continuation_seed": plan["seed"]})
    assert reset["prompt_effective_seed"] == reset["continuation_effective_seed"] == plan["seed"]
    common.save(out / "session_initialized.json", reset)
    os.environ.update({name: str(reset[field]) for field, name in SEED_FIELDS.items()})
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=90, ticks_per_beat=4, beats_per_bar=4),
        input=InputConfig(type="list"),
        output=OutputConfig(type="session", inference_log_detail="full", session_artifact_tier="debug"),
        inference=InferenceConfig(type="http", model_name="lekai",
            server_generate_url=args.server + "/generate_accompaniment", model_condition_bpm=80,
            generation_interval_ticks=4, generation_length_frames=4, prompt_length_ticks=32,
            prompt_selection_mode="rule_s_if_else", prompt_batch_candidates=10,
            temperature=1.1, top_p=.95, top_k=50, repetition_penalty=1),
        continuation_mode="prompt_continuation", count_in_beats=4)
    if args.path_mode == "fixed":
        common.save(out / "configuration.json", asdict(config))
        runtime = RuntimeSessionBuilder(config=config, log_dir=str(out / "logs")).build_cli()
        adapter = FixedPromptInput(runtime.service, events)
        runtime.service._drain_user_events = adapter.drain
        try:
            runtime.start(run_stop_tick=case["stop_tick"])
            deadline = time.monotonic() + case["stop_tick"] / 6 + 120
            while runtime.running:
                if time.monotonic() > deadline:
                    raise TimeoutError("Fixed client did not stop")
                time.sleep(.05)
        finally:
            runtime.stop()
            runtime.cleanup()
            common.save(out / "delivered_events.json", adapter.delivered)
        assert adapter.delivered == [{**r, "injected_at_logical_tick": r["tick"]} for r in events]
        session = runtime.session_dir
    else:
        command = [sys.executable, "-m", "streammuse.presentation.cli.cli",
            "--tempo", "90", "--model-condition-bpm", "80", "--ticks-per-beat", "4", "--beats-per-bar", "4",
            "--model-name", "lekai", "--continuation-mode", "prompt_continuation",
            "--server-url", args.server + "/generate_accompaniment",
            "--prompt-selection-mode", "rule_s_if_else", "--prompt-batch-candidates", "10",
            "--temperature", "1.1", "--top-p", ".95", "--top-k", "50", "--repetition-penalty", "1",
            "--count-in-beats", "4", "--input-snap-forward-fraction", ".4",
            "--generation-interval-ticks", "4", "--generation-length-frames", "4", "--prompt-length-ticks", "32",
            "--run-stop-tick", str(case["stop_tick"]), "--output-type", "session",
            "--session-artifact-tier", "debug", "--log-dir", str(out / "logs"),
            "--log-input-quantization", "--inference-log-detail", "full",
            "--input-mode", "midi_file", "--midi-file-path", str(midi)]
        common.save(out / "command.json", command)
        with (out / "cli.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=common.REPO, stdout=log, stderr=subprocess.STDOUT,
                timeout=case["stop_tick"] / 6 + 120, check=True)
        sessions = list((out / "logs").glob("*/session_*"))
        assert len(sessions) == 1
        session = sessions[0]
    common.save(out / "status.json", {"status": "completed", "session": str(session)})


def read_run(out, server_dir):
    status = common.read(out / "status.json")
    assert status["status"] == "completed"
    session = Path(status["session"])
    reset = common.read(out / "session_initialized.json")
    calls = [json.loads(line) for line in (server_dir / "backend_calls.jsonl").read_text().splitlines()
             if line.strip()]
    calls = [r for r in calls if r["session_id"] == reset["session_id"]]
    assert calls and all(r["effective_seed"] == reset["continuation_effective_seed"] for r in calls)
    mapped = {r["request"]["generation_start_tick"]: r for r in calls}
    assert len(mapped) == len(calls), "Repeated generation ticks require separate diagnosis"
    trace = common.read(session / "prompt_continuation_model_trace.json")
    runtime = trace["runtime_info"]
    assert not runtime["scheduler_is_failed"], runtime.get("scheduler_error")
    requests = [json.loads(line) for line in (session / "prompt_continuation_replay_requests.jsonl").read_text().splitlines() if line.strip()]
    assert requests and all(not r.get("error") and isinstance(r.get("acknowledgement"), dict) for r in requests)
    prompt = trace["prompt_generation_log"]
    assert prompt.get("selection_mode") == "rule_s_if_else", prompt.keys()
    assert prompt.get("candidate_count") == 10
    return {"calls": mapped, "prompt": prompt, "trace_complete": trace["trace_capture_complete"],
        "raw_history": common.read(session / "prompt_continuation_raw_history.json"),
        "prompt_history": common.read(session / "prompt_continuation_prompt_history.json"),
        "requests": requests, "playback": midi_notes(session / "combined.mid")}


def compare(a, b):
    from compare_realtime_replay_exact import _prompt_output_evidence
    common_ticks = sorted(set(a["calls"]) & set(b["calls"]))
    only_a = sorted(set(a["calls"]) - set(b["calls"]))
    only_b = sorted(set(b["calls"]) - set(a["calls"]))
    bad = {k: [] for k in ("prompt_tokens", "raw_tokens", "structural_tokens", "raw_output")}
    examples = {}
    for tick in common_ticks:
        for k in bad:
            l = a["calls"][tick]
            r = b["calls"][tick]
            x, y = (l[k], r[k]) if k == "raw_output" else (l["trace"][k], r["trace"][k])
            if x != y:
                bad[k].append(tick)
                examples.setdefault(k, {"tick": tick, "left": x, "right": y})
    complete = a["trace_complete"] and b["trace_complete"]
    same_calls = not only_a and not only_b
    prompt_output = _prompt_output_evidence(a["prompt"]) == _prompt_output_evidence(b["prompt"])
    result = {"calls": [len(a["calls"]), len(b["calls"])], "common_calls": len(common_ticks),
        "only_left_ticks": only_a, "only_right_ticks": only_b, "trace_complete": complete,
        "mismatched_generation_ticks": bad, "first_difference_examples": examples,
        "prompt_input_exact": a["prompt"]["prompt_tokens"] == b["prompt"]["prompt_tokens"],
        "prompt_selection_exact": prompt_output, "prompt_history_exact": a["prompt_history"] == b["prompt_history"],
        "model_inputs_exact": complete and same_calls and not bad["prompt_tokens"]
                              and a["prompt"]["prompt_tokens"] == b["prompt"]["prompt_tokens"],
        "raw_tokens_exact": complete and same_calls and not bad["raw_tokens"] and not bad["structural_tokens"],
        "raw_output_exact": complete and same_calls and not bad["raw_output"],
        "raw_history_exact": complete and a["raw_history"] == b["raw_history"],
        "raw_note_on": [sum(e.get("event_type", e.get("type")) == "note_on" for e in d["raw_history"]) for d in (a,b)],
        "playback_exact": a["playback"] == b["playback"],
        "playback_notes": [len(d["playback"]) for d in (a,b)]}
    result["primary_raw_exact"] = all(result[k] for k in ("raw_tokens_exact", "raw_output_exact", "raw_history_exact", "prompt_history_exact"))
    return result


def worker(args):
    assert args.gpu in (0, 1, 2)
    used = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert int(used.strip()) < 500, f"GPU {args.gpu} is busy"
    plan = common.read(args.root / "plan.json")
    common.identity(SYSTEM_REF)
    own = args.root / f"worker{args.gpu}"
    own.mkdir(exist_ok=False)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port=s.getsockname()[1]
    url=f"http://127.0.0.1:{port}"
    command=[sys.executable,'-u',__file__,'server','--out',str(own/'server'),'--gpu',str(args.gpu),'--port',str(port)]
    progress={"status":"running","gpu":args.gpu,"pieces":[]}
    progress_path=args.root/f'progress_gpu{args.gpu}.json'
    common.save(progress_path,progress)
    process=None
    try:
        with (own/'server.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('Backend exited')
                try:
                    info=common.api(url,'/prompt_continuation/runtime_info')
                    assert info['has_real_model'] and info['prompt_has_real_model']
                    assert info['prompt_selection_mode']=='rule_s_if_else'
                    assert info['tonal_constraint_enabled'] and info['empty_token_guard_enabled']
                    common.save(own/'loaded_runtime.json',info)
                    break
                except OSError: time.sleep(1)
            else: raise TimeoutError('Backend startup')
            for index,case in enumerate(plan['cases']):
                if index%3!=args.gpu: continue
                entry={'piece':case['piece'],'status':'running','completed_runs':0}
                progress['pieces'].append(entry)
                common.save(progress_path,progress)
                runs={}
                for repeat in (1,2):
                    for mode in ('fixed','midi'):
                        label=f'{mode}_repeat_{repeat}'
                        print('START',case['piece'],label,flush=True)
                        env={k:v for k,v in os.environ.items() if not k.startswith('LEKAI_')}
                        env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(common.REPO/'src'))
                        for name in ('RECOVER_LATE_EVENTS','BOUND_LATE_RECOVERY','REHYDRATE_ACTIVE_NOTES'):
                            env['LEKAI_PROMPT_CONTINUATION_'+name]='0'
                        log_path=own/f"{case['piece']}_{label}.log"
                        with log_path.open('w',encoding='utf-8') as c_log:
                            subprocess.run([sys.executable,'-u',__file__,'client','--root',str(args.root),
                                '--piece',case['piece'],'--path-mode',mode,'--repeat',str(repeat),'--server',url],
                                cwd=common.REPO,env=env,stdout=c_log,stderr=subprocess.STDOUT,
                                timeout=case['stop_tick']/6+180,check=True)
                        out=args.root/'cases'/case['piece']/label
                        runs[label]=read_run(out,own/'server')
                        entry['completed_runs']+=1
                        common.save(progress_path,progress)
                    comparison=compare(runs[f'fixed_repeat_{repeat}'],runs[f'midi_repeat_{repeat}'])
                    common.save(args.root/'cases'/case['piece']/f'comparison_repeat_{repeat}.json',comparison)
                    print('COMPARE',case['piece'],repeat,{k:comparison[k] for k in ('calls','trace_complete','primary_raw_exact','model_inputs_exact','playback_exact')},flush=True)
                entry.update(status='completed',raw_exact=all(common.read(args.root/'cases'/case['piece']/f'comparison_repeat_{n}.json')['primary_raw_exact'] for n in (1,2)))
                for mode in ('fixed','midi'):
                    common.save(args.root/'cases'/case['piece']/f'{mode}_self_repeat.json',compare(runs[f'{mode}_repeat_1'],runs[f'{mode}_repeat_2']))
                common.save(progress_path,progress)
        progress['status']='completed'
    except BaseException as exc:
        progress.update(status='failed',error=repr(exc))
        raise
    finally:
        common.save(progress_path,progress)
        if process is not None:
            if process.poll() is None: process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference-plan',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p=sub.add_parser('server')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--port',type=int,required=True)
    p.set_defaults(system_ref=SYSTEM_REF)
    p=sub.add_parser('client')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--piece',required=True)
    p.add_argument('--path-mode',choices=['fixed','midi'],required=True)
    p.add_argument('--repeat',type=int,required=True)
    p.add_argument('--server',required=True)
    p=sub.add_parser('worker')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    globals()[args.action](args)


if __name__=='__main__':
    main()

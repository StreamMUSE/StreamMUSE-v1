#!/usr/bin/env python3
"""Reproducible rap scheduling measurements; no product behavior is modified."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import fields, is_dataclass
import hashlib
import heapq
import io
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
import wave


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    result = {}
    for condition, items in groups.items():
        elapsed = [row["elapsed_ms"] for row in items if row.get("elapsed_ms") is not None]
        valid = [row["valid_count"] for row in items if row.get("valid_count") is not None]
        result[condition] = {
            "rounds": len(items),
            "success_rate": sum(bool(row["success"]) for row in items) / len(items),
            "elapsed_ms_observed_n": len(elapsed),
            "valid_count_observed_n": len(valid),
            "elapsed_ms_median": statistics.median(elapsed) if elapsed else None,
            "elapsed_ms_max": max(elapsed) if elapsed else None,
            "valid_count_mean": statistics.mean(valid) if valid else None,
        }
    return result


def stage_summary(rows):
    result = summarize(rows)
    for condition, summary in result.items():
        measures = defaultdict(list)
        for row in rows:
            if row["condition"] != condition:
                continue
            timings = row.get("stage_timings_ms", {})
            for key, value in timings.items():
                measures[key].append(value)
            if all(key in timings for key in ("render_total", "moss", "aligner")):
                measures["render_residual"].append(timings["render_total"] - timings["moss"] - timings["aligner"])
        summary["stages"] = {key: {"observed_n": len(values), "median_ms": statistics.median(values),
                                    "min_ms": min(values), "max_ms": max(values)} for key, values in measures.items()}
    return result


def run_analysis(args):
    root = Path(args.raw_root)
    remote = root / "h200-snapshot/evidence"
    primary_inputs = {
        "budget_h200": remote / "budget-16-32-64-local/rounds.jsonl",
        "budget_mac": root / "budget-16-32-64-mac/rounds.jsonl",
        "concurrency": remote / "stages-concurrency/bars.json",
        "edits": remote / "stages-edit-churn-corrected/bars.json",
        "preparation": root / "mac-preparation-measured/rounds.json",
        "mac_stream": root / "mac-v3-two-bars/session/rounds.json",
    }
    budget = [json.loads(line) for line in (remote / "budget-16-32-64-local/rounds.jsonl").read_text().splitlines()]
    mac_budget = [json.loads(line) for line in (root / "budget-16-32-64-mac/rounds.jsonl").read_text().splitlines()]
    concurrency = json.loads((remote / "stages-concurrency/bars.json").read_text())
    edits = json.loads((remote / "stages-edit-churn-corrected/bars.json").read_text())
    preparation = json.loads((root / "mac-preparation-measured/rounds.json").read_text())
    mac_stream = json.loads((root / "mac-v3-two-bars/session/rounds.json").read_text())
    for row in mac_stream:
        row["condition"] = "mac_two_bars"
    conditions = {condition: [row for row in concurrency + edits + mac_stream if row["condition"] == condition]
                  for condition in ("sequential", "parallel", "no_edits", "one_edit", "burst_edits", "mac_two_bars")}
    grid = []
    for condition, rows in conditions.items():
        for bpm in (60, 90, 120, 150, 180):
            for lead in (1, 2, 3):
                grid.append({"condition": condition, "bpm": bpm, "lookahead_bars": lead,
                             **deadline_sensitivity(rows, bpm=bpm, lookahead_bars=lead, guard_ticks=1)})
    pcm_audit = []
    for directory in (remote / "stages-concurrency", remote / "stages-edit-churn-corrected", root / "mac-v3-two-bars"):
        for wav in sorted(directory.rglob("bar-*.wav")):
            pcm_audit.append({"path": str(wav), **inspect_pcm(wav.read_bytes())})
    summary = {
        "schema_version": "streammuse.scheduling_evidence.summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_inputs": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for name, path in primary_inputs.items()},
        "measurement_counts": {"writer_attempts": len(budget) + len(mac_budget),
                               "source_bars": len(concurrency) + len(edits) + len(mac_stream),
                               "preparations": len(preparation), "counterfactual_cells": len(grid)},
        "budget_h200": summarize(budget), "budget_mac": summarize(mac_budget),
        "budget_paired_deltas_ms": {f"n{n}-n16": paired_deltas(budget, "n16", f"n{n}", "elapsed_ms") for n in (32, 64)},
        "stages": stage_summary(concurrency), "edits": stage_summary(edits),
        "edit_first_bar_ms": {condition: [row["elapsed_ms"] for row in rows if row["bar"] == 0]
                              for condition, rows in conditions.items() if condition in ("no_edits", "one_edit", "burst_edits")},
        "preparation": summarize(preparation), "source_pcm_audit": pcm_audit,
        "deadline_counterfactuals": grid,
        "counterfactual_assumptions": [
            "Observed source-ready elapsed times held fixed across tempo and lead; no new model executions",
            "Original bar deadlines retained after edits; no replay is counted as fresh output",
            "Deadline for bar j is ((lookahead+j)*16-1)*60000/(4*bpm) ms",
            "Client warp/mix, enqueue and acoustic playback are not included in source-ready deadlines",
            "MOSS render_total residual includes lock wait and uninstrumented overhead, not a pure measured queue span",
        ],
        "excluded_runs": {"budget-no-rescue": "frozen-mapping serialization bug; interrupted",
                          "stages-edit-churn": "harness confused per-bar revision with plan edit sequence",
                          "mac-preparation": "invalid Tempo constructor before any measurement"},
        "inference_limits": ["No significance claims or fitted distributions", "No quality-listening evaluation",
                             "No beat/tick generation measurement", "No device/acoustic onset measurement",
                             "API seeds reduce variation but do not guarantee bitwise reproducibility"],
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix(".json"), summary)
    plot_budget_evidence(budget, prefix.with_suffix(".png"))
    plot_structural_model(prefix.parent / "2026-10-22-scheduling-stage-model")
    print(json.dumps({"summary": str(prefix.with_suffix(".json")), "figure": str(prefix.with_suffix(".png")),
                      "audited_non_silent_source_wavs": len(pcm_audit)}, indent=2))


def plot_budget_evidence(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["condition"]] = row
    conditions = ["n16", "n32", "n64"]
    for axis, metric, title, ylabel, color in (
        (axes[0], "valid_count", "Candidate yield", "Valid candidates per attempt", "#087f8c"),
        (axes[1], "elapsed_ms", "H200-local writer latency", "Writer wall time (ms)", "#b54a35"),
    ):
        for pair in pairs.values():
            values = [pair[condition][metric] for condition in conditions]
            axis.plot(range(3), values, color="#b9bec4", linewidth=0.8, alpha=0.65, zorder=1)
            axis.scatter(range(3), values, color=color, s=20, alpha=0.65, zorder=2)
        medians = [statistics.median(pair[condition][metric] for pair in pairs.values()) for condition in conditions]
        axis.plot(range(3), medians, color="#161b22", linewidth=2, marker="D", markersize=6, label="Median", zorder=3)
        axis.set_xticks(range(3), [f"{condition[1:]} candidates\nn={sum(row['condition'] == condition for row in rows)}" for condition in conditions])
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=12, loc="left", pad=12)
        axis.set_xlim(-0.2, 2.2)
        axis.set_ylim(bottom=-0.4 if metric == "valid_count" else 0)
        axis.grid(axis="y", color="#e4e7eb", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper left" if metric == "valid_count" else "upper right")
    fig.suptitle("One-wave budget trade-off: paired real Qwen2.5-7B requests", x=0.07, ha="left", fontsize=14)
    fig.text(0.07, 0.025, "12 workload/seed pairs per budget; thin lines match pairs. No rescue. All attempts retained, including cold CMU initialization.\nNo confidence intervals or significance claims. Validity is a structural gate, not a listening-quality rating.", fontsize=8, color="#444b53")
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.91))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_structural_model(prefix):
    """Draw structural dependencies; positions and widths are not measured costs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(14, 8.4), facecolor="white")
    ax = fig.add_axes((0.045, 0.08, 0.92, 0.80))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.text(0.045, 0.95, "Parameterized stage DAG and deadline model", fontsize=19, weight="bold")
    fig.text(0.045, 0.915, "Structural relationships, not fitted laws. Measured semantic output is one bar; beat/tick output remains theoretical.", fontsize=10, color="#454c55")

    nodes = [
        (0.0, "Observe / release", "G6: input watermark A\nunit, phase, context", "#e7ecef"),
        (2.0, "Write / select", "G3: budget N, rescues\nservice Cw; LLM", "#cfe9e7"),
        (4.0, "TTS / align", "G2: Cs + Ca\neffective slots k", "#cfe9e7"),
        (6.0, "Transfer", "G5: Cx, payload\nroute / connection", "#e6e5f3"),
        (8.0, "Warp / mix", "G5: Cp\nflow / tempo / mix", "#e6e5f3"),
        (10.0, "Eligible / commit", "G1: deadline D\nG7: version / fallback", "#f6e6cb"),
        (12.0, "Sink / playback", "G6: sample placement\nnot measured here", "#e7ecef"),
    ]
    for index, (x, title, body, color) in enumerate(nodes):
        ax.add_patch(Rectangle((x, 5.45), 1.75, 1.18, facecolor=color, edgecolor="#52616b", linewidth=0.8))
        ax.text(x + 0.875, 6.34, title, ha="center", va="center", fontsize=9.3, weight="bold")
        ax.text(x + 0.875, 5.85, body, ha="center", va="center", fontsize=8.7, linespacing=1.6)
        if index < len(nodes) - 1:
            ax.annotate("", xy=(x + 1.99, 6.02), xytext=(x + 1.76, 6.02),
                        arrowprops={"arrowstyle": "->", "color": "#34434d", "lw": 1.2})

    ax.text(0, 7.6, "G4: edits carry distinct plan sequence, content revision and timeline revision", fontsize=11, weight="bold", color="#a64232")
    ax.plot([2.88, 10.88], [7.15, 7.15], color="#a64232", linestyle="--", linewidth=1)
    for x in (2.88, 4.88, 8.88, 10.88):
        ax.annotate("", xy=(x, 6.68), xytext=(x, 7.15),
                    arrowprops={"arrowstyle": "->", "color": "#a64232", "linestyle": "--", "lw": 1})
    ax.text(0, 7.30, "Invalidate only actual dependencies", fontsize=9, color="#a64232")
    ax.text(0, 5.10, "Cross-bar dependency: selected text(j) -> writer(j+1); rendering may overlap, but future input is unavailable.", fontsize=9.5, color="#087f8c")
    ax.text(0, 4.72, "G7 keys include producer/config versions: source = text / voice / model / conditioning; prepared adds flow / timeline / format / mix.", fontsize=9.5)
    ax.text(0, 4.34, "Key equality is necessary, not sufficient: readiness, integrity and session/epoch/fidelity policy must also permit reuse.", fontsize=9.5)
    ax.text(0, 3.84, "R = max(policy release, input availability, predecessor finishes); S >= R under capacity; F = S + C; Q = S - R", fontsize=10.3, weight="bold")

    ax.text(0, 3.2, "Illustrative critical-path order (not a measured trace; widths are not to scale)", fontsize=11, weight="bold")
    spans = [(0.5, 1.1, "Cw", "#cfe9e7", None), (1.6, 1.2, "resource wait", "#f2f2f2", "///"),
             (2.8, 3.3, "Cs + Ca", "#cfe9e7", None), (6.1, 1.0, "Cx", "#e6e5f3", None),
             (7.1, 1.4, "Cp", "#e6e5f3", None)]
    for x, width, label, color, hatch in spans:
        ax.add_patch(Rectangle((x, 2.08), width, 0.48, facecolor=color, edgecolor="#64717b", hatch=hatch, linewidth=0.7))
        ax.text(x + width / 2, 2.32, label, ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(13.75, 1.88), xytext=(0, 1.88), arrowprops={"arrowstyle": "->", "color": "#34434d"})
    for x, label, color in ((0.5, "input A / release R", "#34434d"), (8.5, "prepared finish F", "#34434d"),
                            (10.35, "deadline D", "#a64232"), (12.3, "intended playback P", "#34434d")):
        ax.plot([x, x], [1.78, 2.75], color=color, linestyle="--", linewidth=1)
        ax.text(x, 1.45, label, ha="center" if x > 1 else "left", fontsize=9, color=color)
    ax.annotate("", xy=(12.3, 2.95), xytext=(10.35, 2.95), arrowprops={"arrowstyle": "<->", "color": "#a64232"})
    ax.text(11.325, 3.10, "delivery margin M", ha="center", fontsize=9, color="#a64232")
    ax.text(0, 0.9, "G1: D = P - M; slack(t) = D - t - remaining critical path. Fresh success needs F <= D AND current eligibility.", fontsize=10.5)
    ax.text(0, 0.45, "Timeline T is seconds; stage/deadline times are ms: P_ms = 1000 * (T(beat) - request_time_s). Samples = round(fs * (T - t0)).", fontsize=9.7)
    fig.text(0.045, 0.035, "Capacity and cache-key relations are explicit assumptions/contracts. Device timing, true cancellation cost and finer output units are unmeasured.", fontsize=9, color="#454c55")
    for extension in ("svg", "png"):
        fig.savefig(Path(prefix).with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def json_value(value):
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def paired_deltas(rows, baseline, treatment, metric):
    pairs = defaultdict(dict)
    for row in rows:
        condition = row["condition"]
        if condition not in (baseline, treatment):
            continue
        pair = pairs[row["pair_id"]]
        if condition in pair:
            raise ValueError("duplicate condition within pair")
        pair[condition] = row.get(metric)
    return [pair[treatment] - pair[baseline] for _, pair in sorted(pairs.items())
            if pair.get(baseline) is not None and pair.get(treatment) is not None]


def deadline_sensitivity(rows, *, bpm, lookahead_bars, guard_ticks):
    if any(not math.isfinite(value) for value in (bpm, lookahead_bars, guard_ticks)) or bpm <= 0 or lookahead_bars <= 0 or not 0 <= guard_ticks < lookahead_bars * 16:
        raise ValueError("invalid deadline configuration")
    deadline = (lookahead_bars * 16 - guard_ticks) * 60000 / (4 * bpm)
    outcomes = []
    for row in rows:
        bar = row.get("bar", 0)
        if not isinstance(bar, int) or isinstance(bar, bool) or bar < 0:
            raise ValueError("bar ordinal must be a non-negative integer")
        actual_deadline = deadline + bar * 240000 / bpm
        elapsed = row["elapsed_ms"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid observed elapsed time or bar")
        miss = not row["success"] or not row.get("latest_revision", True) or elapsed > actual_deadline
        outcomes.append({"bar": bar, "deadline_ms": actual_deadline,
                         "elapsed_ms": elapsed, "miss": miss, "lateness_ms": elapsed - actual_deadline})
    misses = sum(row["miss"] for row in outcomes)
    return {"evidence_kind": "trace_counterfactual", "deadline_ms": deadline,
            "rounds": len(rows), "misses": misses,
            "miss_rate": misses / len(rows) if rows else None, "outcomes": outcomes}


def replay_pipeline(stage_costs, *, render_slots=1, release_times_ms=None):
    """Counterfactual FIFO replay: one causal writer, independent render slots."""
    if not isinstance(render_slots, int) or render_slots < 1:
        raise ValueError("render_slots must be positive")
    stage_costs = list(stage_costs)
    releases = [0.0] * len(stage_costs) if release_times_ms is None else list(release_times_ms)
    if len(releases) != len(stage_costs) or any(not math.isfinite(v) or v < 0 for v in releases):
        raise ValueError("each job needs one finite non-negative release watermark")
    slots = [0.0] * render_slots
    writer_free = 0.0
    result = []
    for (writer_ms, render_ms), release in zip(stage_costs, releases):
        if any(not math.isfinite(v) or v < 0 for v in (writer_ms, render_ms)):
            raise ValueError("stage costs must be finite and non-negative")
        writer_start = max(writer_free, release)
        writer_free = writer_start + writer_ms
        render_start = max(writer_free, heapq.heappop(slots))
        finish = render_start + render_ms
        heapq.heappush(slots, finish)
        result.append({"writer_start_ms": writer_start, "writer_finish_ms": writer_free,
                       "render_start_ms": render_start, "finish_ms": finish})
    return result


def revision_outcome(*, start_ms, finish_ms, deadline_ms, edit_ms=None):
    """Snapshot at start, evaluate freshness at deadline, assume no preemption."""
    times = [start_ms, finish_ms, deadline_ms] + ([] if edit_ms is None else [edit_ms])
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("timestamps must be finite and non-negative")
    if finish_ms < start_ms:
        raise ValueError("finish precedes start")
    stale = edit_ms is not None and start_ms < edit_ms <= deadline_ms
    return {"ready": finish_ms <= deadline_ms, "latest_revision": not stale,
            "discarded_ms": finish_ms - start_ms if stale else 0}


def cache_compatible(artifact, current, *, dependencies):
    return all(key in artifact and key in current and artifact[key] == current[key] for key in dependencies)


def isolated_service_plan(name, argv, environment, *, root, old_root, repo):
    if name not in ("vllm", "vllm-live", "render"):
        raise ValueError("unknown service")
    argv, environment = list(argv), dict(environment)
    port, gpu = {"vllm": (18601, 0), "vllm-live": (18611, 1), "render": (18620, 2)}[name]
    argv[argv.index("--port") + 1] = str(port)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["VLLM_CACHE_ROOT"] = f"{root}/vllm-cache"
    if name == "render":
        argv[argv.index("--artifact-root") + 1] = f"{root}/render_artifacts"
        argv[argv.index("--vllm-url") + 1] = "http://127.0.0.1:18611/v1"
        environment["PYTHONPATH"] = f"{old_root}/deps:{repo}/src:{repo}"
        environment["HF_MODULES_CACHE"] = f"{root}/hf-modules"
    return argv, environment, port, gpu


def launch_service(args):
    """Reuse verified command construction only; never invoke old start/stop."""
    old_root = Path(args.old_deployment).resolve()
    root = Path(args.root).resolve()
    if root == old_root or old_root in root.parents:
        raise ValueError("new deployment must be separate from old deployment")
    spec = importlib.util.spec_from_file_location("previous_deployment", old_root / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv, changes, _ = module.command("vllm" if args.service == "vllm-live" else args.service)
    argv, changes, port, gpu = isolated_service_plan(
        args.service, argv, changes, root=str(root), old_root=str(old_root), repo=str(root / "repo"))
    root.mkdir(parents=True, exist_ok=True)
    pidfile = root / f"{args.service}.pid.json"
    if pidfile.exists():
        raise RuntimeError("isolated pidfile already exists; inspect before any relaunch")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    gpu_row = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if int(gpu_row.split(",")[-1].strip()) > 16:
        raise RuntimeError(f"GPU{gpu} is not empty; refusing launch: {gpu_row}")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(changes)
    with (root / f"{args.service}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(argv, cwd=root / "repo", env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = {"service": args.service, "pid": process.pid, "identity": module.proc_identity(process.pid),
              "started_at": datetime.now(timezone.utc).isoformat(), "argv": argv,
              "environment": changes, "port": port, "gpu_preflight": gpu_row,
              "cwd": str(root / "repo"), "log": str(root / f"{args.service}.log")}
    pidfile.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def can_stop(record, identity, *, uid):
    return record["identity"] == identity and identity["uid"] == uid and identity["pgid"] == record["pid"]


def stop_service(args):
    root = Path(args.root).resolve()
    pidfile = root / f"{args.service}.pid.json"
    record = json.loads(pidfile.read_text())
    pid = record["pid"]
    proc = Path("/proc") / str(pid)
    if proc.exists():
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        identity = {"pid": pid, "pgid": os.getpgid(pid), "uid": proc.stat().st_uid, "start_ticks": fields[19]}
        if not can_stop(record, identity, uid=os.getuid()):
            raise RuntimeError("process identity/ownership mismatch; no signal sent")
        os.killpg(pid, signal.SIGTERM)
        until = time.monotonic() + 30
        while proc.exists() and time.monotonic() < until:
            time.sleep(0.2)
        if proc.exists():
            raise RuntimeError("recorded process has not exited; no escalation sent")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pidfile.rename(root / f"{args.service}.stopped-{stamp}.pid.json")
    print(json.dumps({"service": args.service, "stopped_pid": pid, "record_archived": True}))


def measure_writer_round(generator, request, policy):
    from streammuse.application.rap.lyric_writer import SequentialLyricWriter, LyricWriteFailed
    from streammuse.domain.rap import ScoreWeights
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer

    waves = []

    class Recorder:
        def generate(self, candidate_request):
            begun = time.perf_counter()
            batch = generator.generate(candidate_request)
            waves.append({"requested_count": candidate_request.count,
                          "request_seed": candidate_request.seed,
                          "elapsed_ms": (time.perf_counter() - begun) * 1000,
                          "batch": json_value(batch)})
            return batch

    writer = SequentialLyricWriter(Recorder(), CmuProsodyAnalyzer(), ScoreWeights(), policy)
    started = time.perf_counter()
    row = {"success": False, "selected": None, "evaluation_ms": None,
           "error_type": None, "error": None}
    try:
        result = writer.write(request)
        ledger = [dict(item) for item in result.ledger]
        row.update(success=True, selected=result.record.to_payload(),
                   evaluation_ms=result.evaluation_ms, generation_ms=result.generation_ms)
    except LyricWriteFailed as error:
        ledger = [dict(item) for item in error.ledger]
        row.update(error_type=type(error).__name__, error=str(error),
                   generation_ms=sum(wave["elapsed_ms"] for wave in waves))
    row.update(elapsed_ms=(time.perf_counter() - started) * 1000,
               ledger=ledger, raw_waves=waves, waves=len(waves),
               requested_count=sum(wave["requested_count"] for wave in waves),
               valid_count=sum(item["status"] in ("valid", "selected") for item in ledger),
               selectable_count=sum(item["status"] in ("valid", "selected") and
                                    item.get("score", -1) >= policy.minimum_score for item in ledger))
    return row


def make_start_payload(*, topic, seed, template_id, bars, bpm):
    from streammuse.domain.rap.lyric_stream import LYRIC_STREAM_SCHEMA_VERSION
    from streammuse.domain.rap.remote_chunk import _flow_payload
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
    return {"type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
            "tempo_bpm": bpm, "start_bar": 0, "seed": seed, "loop": False,
            "lyric_lead": bars, "audio_lead": bars,
            "segments": [{"start_bar": 0, "bars": bars, "topic": topic,
                          "flow_template": _flow_payload(BUILTIN_TEMPLATES.get(template_id))}],
            "edits": [], "history": []}


def execute_jobs(jobs, *, workers):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: job(), jobs))
    return {"elapsed_ms": (time.perf_counter() - started) * 1000, "results": results}


def make_edit_schedule(offsets):
    if any(not math.isfinite(value) or value <= 0 for value in offsets) or list(offsets) != sorted(set(offsets)):
        raise ValueError("edits need increasing positive finite offsets")
    topics = ("ocean waves", "morning rain", "street drums")
    return [{"at_ms": seconds * 1000, "message": {"type": "update", "edit": {
        "edit_seq": index + 1, "from_bar": 0, "topic": topics[index % len(topics)], "flow_template": None}}}
        for index, seconds in enumerate(offsets)]


def inspect_pcm(source_wav):
    with wave.open(io.BytesIO(source_wav)) as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or frames == 0:
            raise ValueError("expected nonempty mono PCM16 source")
        samples = struct.unpack("<" + "h" * frames, wav.readframes(frames))
    peak = max(abs(sample) for sample in samples)
    if peak == 0:
        raise ValueError("model returned silent source audio")
    return {"frames": frames, "sample_rate_hz": rate, "source_duration_s": frames / rate,
            "peak_pcm16": peak, "rms_pcm16": math.sqrt(sum(sample * sample for sample in samples) / frames)}


def matches_latest(lyric, *, latest_id, topic):
    return lyric is not None and lyric["lyric_id"] == latest_id and lyric["topic"] == topic


def inspect_prepared(audio, *, expected_frames):
    import numpy as np
    if audio.frame_count != expected_frames:
        raise ValueError("prepared frame count differs from destination")
    samples = np.frombuffer(audio.data, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("prepared samples must be finite")
    peak = float(np.max(np.abs(samples), initial=0))
    if peak == 0:
        raise ValueError("research preparation returned silent audio")
    return {"frame_count": audio.frame_count, "peak": peak,
            "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))}


def run_preparation(args):
    import numpy as np
    from streammuse.application.rap.bar_audio import BarAudioPreparer
    from streammuse.domain.rap import AudioFormat
    from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
    from streammuse.domain.timing import Tempo
    from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
    from streammuse.infrastructure.rap.phrase_warp import RubberBandPhraseWarper, probe_rubberband_r3

    output = prepare_run(args)
    version = probe_rubberband_r3()
    warper = RubberBandPhraseWarper(policy="gentle_sparse_r3", rubberband_version=version)
    drums = ProceduralBoomBapRenderer(seed=20261022)
    fmt = AudioFormat()
    rows = []
    for manifest_path in sorted(Path(args.input).rglob("bar-*.manifest.json"))[:args.limit]:
        manifest = BarAudioManifest.from_payload(json.loads(manifest_path.read_text()))
        events = json.loads((manifest_path.parent / "events.json").read_text())
        payload = next(event["message"]["record"] for event in events
                       if event["message"]["type"] == "lyric" and event["message"]["record"]["lyric_id"] == manifest.lyric_id)
        record = BarLyricRecord.from_payload(payload)
        source = (manifest_path.parent / f"bar-{record.bar}.wav").read_bytes()
        preparer = BarAudioPreparer(warper=warper, drums=drums,
                                   tempo=Tempo(bpm=manifest.tempo_bpm, ticks_per_beat=4, beats_per_bar=4), audio_format=fmt)
        row = {"source_manifest": str(manifest_path), "lyric_id": record.lyric_id,
               "bar": record.bar, "bpm": manifest.tempo_bpm, "evidence_kind": "real_measurement",
               "condition": "mac_warp_mix", "first_preparation": not rows, "rubberband_version": version}
        begun = time.perf_counter()
        try:
            prepared = preparer.prepare(record, manifest, source)
            row["elapsed_ms"] = (time.perf_counter() - begun) * 1000
            expected = round((record.bar + 1) * 240 / manifest.tempo_bpm * fmt.sample_rate_hz) - round(record.bar * 240 / manifest.tempo_bpm * fmt.sample_rate_hz)
            row.update(inspect_prepared(prepared.audio, expected_frames=expected), success=True,
                       warp_ms=prepared.render_latency_ms)
            samples = np.frombuffer(prepared.audio.data, dtype=np.float32)
            with wave.open(str(output / f"prepared-{len(rows):03}.wav"), "wb") as wav:
                wav.setnchannels(fmt.channels)
                wav.setsampwidth(2)
                wav.setframerate(fmt.sample_rate_hz)
                wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        except Exception as error:
            row.update(success=False, elapsed_ms=(time.perf_counter() - begun) * 1000,
                       error_type=type(error).__name__, error=str(error))
        rows.append(row)
        write_json(output / "rounds.json", rows)
        print(json.dumps(row), flush=True)
    write_json(output / "summary.json", summarize(rows))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def prepare_run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__)]
    source_root = Path(__file__).resolve().parents[2] / "src"
    sources.extend(source_root.glob("streammuse/application/rap/lyric_*.py"))
    sources.extend(source_root / relative for relative in (
        "streammuse/application/rap/bar_audio.py", "streammuse/application/rap/chunk_audio.py",
        "streammuse/application/rap/audio_rendering.py", "streammuse/infrastructure/rap/phrase_warp.py",
        "streammuse/infrastructure/rap/drums.py", "streammuse/infrastructure/rap/generators.py",
        "streammuse/infrastructure/inference/local_chat_client.py") if (source_root / relative).is_file())
    write_json(output / "run.json", {
        "schema": "streammuse.scheduling_evidence.v1", "evidence_kind": "real_measurement",
        "started_at": datetime.now(timezone.utc).isoformat(), "args": vars(args),
        "argv": sys.argv, "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "clock": "client_host.perf_counter",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "limitations": ["No measured acoustic output", "No cross-host clock subtraction",
                        "Short paired experiment; no statistical significance claim"],
    })
    return output


def run_candidates(args):
    from streammuse.application.rap.lyric_writer import LyricWriteRequest, LyricWriterPolicy
    from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
    from streammuse.infrastructure.rap.generators import IndependentChoiceCandidateGenerator
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    output = prepare_run(args)
    rows = []
    topics = ("city lights", "ocean waves", "coding late", "train ride", "morning rain", "street drums")
    templates = ("baseline_straight_9", "baseline_syncopated_9", "baseline_staggered_9")
    with (output / "rounds.jsonl").open("w") as trace:
        for pair in range(args.pairs):
            seed = args.seed + pair
            counts = args.budgets if pair % 2 == 0 else list(reversed(args.budgets))
            for count in counts:
                config = LocalChatModelClientConfig(base_url=args.llm_url, model=args.model,
                                                    timeout_s=args.timeout, extra_payload={"seed": seed})
                client = LocalChatModelClient(config)
                request = LyricWriteRequest(
                    f"research-{pair}-n{count}", 0, 0, topics[pair % len(topics)],
                    BUILTIN_TEMPLATES.get(templates[pair % len(templates)]), (), seed)
                policy = LyricWriterPolicy(candidates=count, rescue_waves=args.rescue_waves,
                                           rescue_candidates=16, minimum_valid=3)
                try:
                    generator = IndependentChoiceCandidateGenerator(client, max_tokens_per_choice=32, temperature=1.0)
                    row = measure_writer_round(generator, request, policy)
                finally:
                    client.close()
                row.update(condition=f"n{count}", pair_id=f"pair{pair:03}",
                           evidence_kind="real_measurement", topic=request.topic,
                           template_id=request.flow_template.template_id, api_seed=seed,
                           api_seed_policy="fixed_within_attempt", policy=policy.to_payload())
                rows.append(row)
                trace.write(json.dumps(row, default=str, allow_nan=False) + "\n")
                trace.flush()
                print(json.dumps({key: row[key] for key in ("condition", "pair_id", "success", "elapsed_ms", "valid_count", "waves")}), flush=True)
    result = {"conditions": summarize(rows), "paired_latency_deltas_ms": {
        f"n{count}-n{args.budgets[0]}": paired_deltas(rows, f"n{args.budgets[0]}", f"n{count}", "elapsed_ms")
        for count in args.budgets[1:]}}
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


def run_stream_session(*, base_url, output, topic, seed, template_id, bars, bpm, timeout, edit_offsets=()):
    from websockets.sync.client import connect
    from streammuse.infrastructure.rap.bar_package import decode_bar_package

    output.mkdir(parents=True, exist_ok=False)
    session = "research-" + uuid.uuid4().hex
    start = make_start_payload(topic=topic, seed=seed, template_id=template_id, bars=bars, bpm=bpm)
    write_json(output / "start.json", start)
    edits = make_edit_schedule(edit_offsets)
    write_json(output / "edit_schedule.json", edits)
    next_edit = 0
    rows, events, lyrics, finished = [], [], {}, set()
    latest_ids = {}
    applied_edit_seq = 0
    begin = time.perf_counter()
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    try:
        with connect(ws_url + f"/v1/rap/sessions/{session}", open_timeout=10, close_timeout=5) as websocket:
            websocket.send(json.dumps(start))
            try:
                while len(finished) < bars:
                    remaining = timeout - (time.perf_counter() - begin)
                    if remaining <= 0:
                        raise TimeoutError("session deadline reached")
                    now_ms = (time.perf_counter() - begin) * 1000
                    while next_edit < len(edits) and now_ms >= edits[next_edit]["at_ms"]:
                        websocket.send(json.dumps(edits[next_edit]["message"]))
                        events.append({"elapsed_ms": (time.perf_counter() - begin) * 1000,
                                       "message": {"type": "research_update_sent", **edits[next_edit]}})
                        next_edit += 1
                    try:
                        message = json.loads(websocket.recv(timeout=min(remaining, 0.05) if next_edit < len(edits) else remaining))
                    except TimeoutError:
                        if next_edit < len(edits):
                            continue
                        raise
                    elapsed = (time.perf_counter() - begin) * 1000
                    events.append({"elapsed_ms": elapsed, "message": message})
                    write_json(output / "events.json", events)
                    if message["type"] == "lyric":
                        record = message["record"]
                        lyrics[record["lyric_id"]] = (elapsed, record)
                        latest_ids[record["bar"]] = record["lyric_id"]
                    if message["type"] == "superseded":
                        applied_edit_seq = max(applied_edit_seq, message["edit_seq"])
                    if message["type"] == "error":
                        print(json.dumps({"session": session, "server_error": message}), flush=True)
                    if message["type"] != "audio_ready" or message["bar"] in finished:
                        continue
                    bar, lyric_id = message["bar"], message["lyric_id"]
                    lyric_ms, lyric = lyrics.get(lyric_id, (None, None))
                    target_topic = edits[-1]["message"]["edit"]["topic"] if edits else topic
                    if next_edit < len(edits) or applied_edit_seq < len(edits) or not matches_latest(
                        lyric, latest_id=latest_ids.get(bar), topic=target_topic
                    ):
                        continue
                    url = f"{base_url}/v1/rap/sessions/{session}/bars/{bar}/audio?lyric_id={lyric_id}&wait_ms=0"
                    fetch_begin = time.perf_counter()
                    with urllib.request.urlopen(url, timeout=20) as response:
                        package = response.read()
                        headers = dict(response.headers)
                    http_ms = (time.perf_counter() - fetch_begin) * 1000
                    fetched = (time.perf_counter() - begin) * 1000
                    decoded = decode_bar_package(package, expected_lyric_id=lyric_id)
                    pcm = inspect_pcm(decoded.source_wav)
                    manifest = json.loads(decoded.manifest.canonical_json_bytes())
                    (output / f"bar-{bar}.zip").write_bytes(package)
                    (output / f"bar-{bar}.wav").write_bytes(decoded.source_wav)
                    write_json(output / f"bar-{bar}.manifest.json", manifest)
                    row = {"success": True, "bar": bar, "session_id": session,
                           "elapsed_ms": fetched, "lyric_ready_ms": lyric_ms,
                           "audio_ready_ms": elapsed,
                           "http_after_ready_ms": http_ms,
                           "pcm": pcm,
                           "package_bytes": len(package), "transport": decoded.transport,
                           "stage_timings_ms": manifest["stage_timings_ms"],
                           "manifest": manifest, "lyric": lyric, "headers": headers,
                           "edit_count": len(edits), "latest_revision": True}
                    rows.append(row)
                    finished.add(bar)
                    print(json.dumps({key: row[key] for key in ("session_id", "bar", "elapsed_ms", "stage_timings_ms")}), flush=True)
            finally:
                websocket.send(json.dumps({"type": "stop"}))
    except Exception as error:
        for bar in range(bars):
            if bar not in finished:
                rows.append({"success": False, "bar": bar, "session_id": session,
                             "elapsed_ms": (time.perf_counter() - begin) * 1000,
                             "error_type": type(error).__name__, "error": str(error)})
    write_json(output / "events.json", events)
    write_json(output / "rounds.json", rows)
    return rows


def run_stage_study(args):
    output = prepare_run(args)
    with urllib.request.urlopen(args.render_url + "/health", timeout=10) as response:
        write_json(output / "health.json", json.load(response))
    groups, all_rows = [], []
    conditions = ["sequential", "parallel"] if args.study == "concurrency" else ["no_edits", "one_edit", "burst_edits"]
    for pair in range(args.pairs):
        order = conditions if pair % 2 == 0 else list(reversed(conditions))
        for condition in order:
            edits = {"one_edit": (0.25,), "burst_edits": (0.25, 0.5, 0.75)}.get(condition, ())
            session_count = 2 if args.study == "concurrency" else 1
            jobs = []
            for ordinal in range(session_count):
                kwargs = dict(base_url=args.render_url, output=output / f"pair{pair}-{condition}-session{ordinal}",
                              topic=("city lights", "morning rain")[ordinal % 2], seed=args.seed + pair * 2 + ordinal,
                              template_id="baseline_straight_9", bars=args.bars, bpm=args.bpm,
                              timeout=args.timeout, edit_offsets=edits)
                jobs.append(lambda kwargs=kwargs: run_stream_session(**kwargs))
            measured = execute_jobs(jobs, workers=2 if condition == "parallel" else 1)
            rows = [row for session in measured["results"] for row in session]
            for row in rows:
                row.update(condition=condition, pair_id=f"pair{pair:03}", evidence_kind="real_measurement")
            all_rows.extend(rows)
            group = {"condition": condition, "pair_id": f"pair{pair:03}", "elapsed_ms": measured["elapsed_ms"],
                     "success": all(row["success"] for row in rows), "requested_bars": args.bars * session_count,
                     "successful_bars": sum(row["success"] for row in rows),
                     "controlled_concurrent_sessions": 2 if condition == "parallel" else 1}
            groups.append(group)
            write_json(output / "groups.json", groups)
            write_json(output / "bars.json", all_rows)
            print(json.dumps(group), flush=True)
    write_json(output / "summary.json", summarize(groups))


def run_stream(args):
    output = prepare_run(args)
    with urllib.request.urlopen(args.render_url + "/health", timeout=10) as response:
        write_json(output / "health.json", json.load(response))
    rows = run_stream_session(base_url=args.render_url, output=output / "session",
                              topic=args.topic, seed=args.seed, template_id=args.template,
                              bars=args.bars, bpm=args.bpm, timeout=args.timeout)
    write_json(output / "summary.json", {"successful_bars": sum(row["success"] for row in rows),
                                         "requested_bars": args.bars, "rows": rows})
    if not all(row["success"] for row in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("launch")
    launch.add_argument("service", choices=["vllm", "vllm-live", "render"])
    launch.add_argument("--root", required=True)
    launch.add_argument("--old-deployment", required=True)
    stop = commands.add_parser("stop")
    stop.add_argument("service", choices=["vllm", "vllm-live", "render"])
    stop.add_argument("--root", required=True)
    candidates = commands.add_parser("candidates")
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--llm-url", default="http://127.0.0.1:18601/v1")
    candidates.add_argument("--model", default="qwen-rap")
    candidates.add_argument("--pairs", type=int, default=12)
    candidates.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64])
    candidates.add_argument("--rescue-waves", type=int, default=0)
    candidates.add_argument("--seed", type=int, default=20261022)
    candidates.add_argument("--timeout", type=float, default=60)
    stream = commands.add_parser("stream")
    stream.add_argument("--output", required=True)
    stream.add_argument("--render-url", default="http://127.0.0.1:18620")
    stream.add_argument("--bars", type=int, default=2)
    stream.add_argument("--bpm", type=float, default=90)
    stream.add_argument("--topic", default="city lights")
    stream.add_argument("--template", default="baseline_straight_9")
    stream.add_argument("--seed", type=int, default=20261022)
    stream.add_argument("--timeout", type=float, default=180)
    study = commands.add_parser("stages")
    study.add_argument("--study", choices=["concurrency", "edits"], required=True)
    study.add_argument("--output", required=True)
    study.add_argument("--render-url", default="http://127.0.0.1:18620")
    study.add_argument("--pairs", type=int, default=3)
    study.add_argument("--bars", type=int, default=2)
    study.add_argument("--bpm", type=float, default=90)
    study.add_argument("--seed", type=int, default=20261022)
    study.add_argument("--timeout", type=float, default=60)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--input", required=True)
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--limit", type=int, default=12)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--raw-root", required=True)
    analysis.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    if args.command == "launch":
        launch_service(args)
    elif args.command == "stop":
        stop_service(args)
    elif args.command == "candidates":
        run_candidates(args)
    elif args.command == "stream":
        run_stream(args)
    elif args.command == "stages":
        run_stage_study(args)
    elif args.command == "prepare":
        run_preparation(args)
    elif args.command == "analyze":
        run_analysis(args)


if __name__ == "__main__":
    main()

"""Profile the SGLang-Omni MOSS render path (MOSS + MMS + R3) on a live server.

Subcommands
-----------
render  Run warmups + measured renders through the real adapter and renderer,
        with SGLang-Omni request-level event recording enabled and fine
        timers around renderer sub-steps. Writes records.jsonl.
torch   Capture a torch profiler trace (CPU + CUDA) inside the SGLang-Omni
        stage processes for a few requests.
sweep   Send raw /v1/audio/speech requests with varied token targets to fit
        latency = fixed + per_step * decode_steps.

The script never starts or stops servers; it expects SGLang-Omni on --url.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

DEFAULT_BENCH = ROOT / "logs" / "sglang_moss_benchmark_20260911"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class Timers:
    """Accumulate wall time per label for the current sample."""

    def __init__(self) -> None:
        self.local = threading.local()

    def reset(self) -> None:
        self.local.rows = {}

    def add(self, label: str, ms: float) -> None:
        rows = getattr(self.local, "rows", None)
        if rows is None:
            self.reset()
            rows = self.local.rows
        total, count = rows.get(label, (0.0, 0))
        rows[label] = (total + ms, count + 1)

    def snapshot(self) -> dict[str, dict[str, float]]:
        rows = getattr(self.local, "rows", {}) or {}
        return {k: {"ms": v[0], "calls": v[1]} for k, v in rows.items()}


TIMERS = Timers()


def wrap(owner, name: str, label: str) -> None:
    original = getattr(owner, name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TIMERS.add(label, (time.perf_counter() - started) * 1000.0)

    setattr(owner, name, timed)


def post_json(url: str, path: str, body: dict) -> dict:
    import httpx

    response = httpx.post(url + path, json=body, timeout=120)
    response.raise_for_status()
    return response.json() if response.content else {}


def sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_requests(bench: Path):
    from streammuse.experiments.rap_audio_protocols.contracts import (
        SyllableTarget,
        TwoBarRenderRequest,
    )

    result = []
    for item in json.loads((bench / "corpus.json").read_text()):
        item.pop("duration_seconds", None)
        item["syllables"] = tuple(
            SyllableTarget(**{**s, "phonemes": tuple(s["phonemes"])})
            for s in item["syllables"]
        )
        result.append(TwoBarRenderRequest(**item))
    return result


def make_synthesizer(args):
    from streammuse.infrastructure.rap.sglang_moss_tts import (
        SglangMossConfig,
        SglangMossSynthesizer,
    )

    model_dir = next((args.bench / "model").iterdir())
    synth = SglangMossSynthesizer(
        SglangMossConfig.from_files(
            base_url=args.url,
            model_id="OpenMOSS-Team/MOSS-TTS-v1.5",
            model_revision=model_dir.name,
            reference_audio_uri=(args.bench / "media" / "reference.wav").as_uri(),
            reference_audio_file=args.bench / "media" / "reference.wav",
            reference_text_file=args.bench / "reference.txt",
            runtime_config_sha256=sha(args.config),
            request_timeout_seconds=180,
        )
    )
    synth.probe()
    return synth


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


# ---------------------------------------------------------------------------
# render: full MOSS + MMS + R3 with fine timers and server request events
# ---------------------------------------------------------------------------

def install_render_timers() -> None:
    from streammuse.infrastructure.rap import mms_forced_alignment as mms
    from streammuse.infrastructure.rap import moss_aligned_phrase as phrase
    from streammuse.infrastructure.rap import time_stretch as ts

    wrap(mms.MmsForcedAligner, "align", "mms.align")
    for name in (
        "_load_source_audio",
        "_prepare_warp_input",
        "_resolve_warp_plan",
        "_write_json_atomic",
        "_encode_pcm16_wav",
        "_validate_pcm16_wav",
        "continuous_pitch_preserving_warp",
        "_complete_alignment_artifact",
    ):
        if hasattr(phrase, name):
            wrap(phrase, name, f"phrase.{name}")
    wrap(ts.RubberBandTimeMapStretcher, "_run_rubberband", "r3.subprocess")
    wrap(ts.RubberBandTimeMapStretcher, "stretch", "r3.stretch_total")
    # The production renderer uses the experiments time-map stretcher, which
    # shells out once per duration-fitting attempt.
    import subprocess

    from streammuse.experiments.rap_audio_protocols import warp as exp_warp

    wrap(exp_warp.RubberBandTimeMapStretcher, "__call__", "r3.timemap_call")
    wrap(subprocess, "run", "subprocess.run")
    # MMS internals, when present.
    for name in dir(mms.MmsForcedAligner):
        if name.startswith("_") and not name.startswith("__") and callable(
            getattr(mms.MmsForcedAligner, name)
        ):
            wrap(mms.MmsForcedAligner, name, f"mms.{name}")
    for name in ("map_syllable_onsets", "_load_inference_waveform"):
        if hasattr(mms, name):
            wrap(mms, name, f"mms.{name}")


def cmd_render(args) -> None:
    import torch

    from streammuse.application.rap.execution import SynthesisExecutionContext
    from streammuse.infrastructure.rap.mms_forced_alignment import MmsForcedAligner
    from streammuse.infrastructure.rap.moss_aligned_phrase import (
        MossAlignedPhraseRenderer,
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    install_render_timers()
    synth = make_synthesizer(args)
    aligner = MmsForcedAligner.load(device="cuda:0")
    renderer = MossAlignedPhraseRenderer(
        synthesizer=synth, aligner=aligner, rubberband_version="3.3.0 R3"
    )
    requests = load_requests(args.bench)
    jobs = [(True, i, requests[i % len(requests)]) for i in range(args.warmups)]
    jobs += [(False, i, r) for i, r in enumerate(requests[: args.limit])]

    event_dir = out / "server_events"
    profiler_on = False
    profile = cProfile.Profile() if args.cprofile else None
    with (out / "records.jsonl").open("w", buffering=1) as ledger:
        for warmup, index, request in jobs:
            if not warmup and not profiler_on and args.server_events:
                post_json(
                    args.url,
                    "/start_request_profile",
                    {"run_id": "render", "event_dir": str(event_dir)},
                )
                profiler_on = True
            path = out / (f"warmup-{index:02}" if warmup else f"sample-{index:02}")
            TIMERS.reset()
            torch.cuda.synchronize()
            started = time.perf_counter()
            if profile is not None and not warmup:
                profile.enable()
            result = renderer.render(
                request,
                path,
                execution=SynthesisExecutionContext.from_timeout(
                    240, correlation_id=f"profile-{warmup}-{index}"
                ),
            )
            if profile is not None and not warmup:
                profile.disable()
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - started) * 1000.0
            meta = result.moss_serving_metadata
            row = {
                "sample": index,
                "warmup": warmup,
                "text": request.text,
                "target_seconds": request.duration_seconds,
                "total_ms": total_ms,
                "stages_ms": dict(result.stage_timings_ms),
                "timers": TIMERS.snapshot(),
                "serving": dict(meta) if meta else None,
                "source_frames": wav_frames(path / "source.wav"),
                "final_frames": wav_frames(path / "vocal.wav"),
                "warp_policy": result.alignment_diagnostics.get("warp_policy")
                if hasattr(result.alignment_diagnostics, "get")
                else None,
            }
            ledger.write(json.dumps(row, default=str, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "sample": index,
                        "warmup": warmup,
                        "total_ms": round(total_ms, 1),
                        "moss_ms": round(row["stages_ms"]["moss"], 1),
                        "source_frames": row["source_frames"],
                    }
                ),
                flush=True,
            )
    if profiler_on:
        post_json(args.url, "/stop_request_profile", {"run_id": "render"})
    if profile is not None:
        profile.dump_stats(out / "client.cprofile")
        text = io.StringIO()
        pstats.Stats(profile, stream=text).sort_stats("cumulative").print_stats(60)
        (out / "client_cprofile_top.txt").write_text(text.getvalue())
    synth.close()


# ---------------------------------------------------------------------------
# torch: kernel-level trace inside SGLang-Omni stage processes
# ---------------------------------------------------------------------------

def cmd_torch(args) -> None:
    from streammuse.application.rap.execution import SynthesisExecutionContext

    args.out.mkdir(parents=True, exist_ok=True)
    synth = make_synthesizer(args)
    requests = load_requests(args.bench)
    for i in range(args.warmups):
        synth.synthesize(
            requests[i % len(requests)],
            args.out / f"warm-{i}.wav",
            execution=SynthesisExecutionContext.from_timeout(120, correlation_id=f"tw{i}"),
        )
    started = post_json(
        args.url,
        "/start_profile",
        {"run_id": args.run_id, "enable_torch": True},
    )
    print(json.dumps(started), flush=True)
    timings = []
    try:
        for i in range(args.limit):
            t0 = time.perf_counter()
            synth.synthesize(
                requests[i],
                args.out / f"trace-{i}.wav",
                execution=SynthesisExecutionContext.from_timeout(120, correlation_id=f"tt{i}"),
            )
            timings.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.3)  # visible gap between requests in the trace
    finally:
        stopped = post_json(args.url, "/stop_profile", {"run_id": args.run_id})
    print(json.dumps({"stopped": stopped, "request_ms": timings}), flush=True)
    (args.out / f"{args.run_id}.json").write_text(json.dumps({"request_ms": timings}))
    synth.close()


# ---------------------------------------------------------------------------
# sweep: latency versus requested length
# ---------------------------------------------------------------------------

def cmd_sweep(args) -> None:
    import httpx

    args.out.mkdir(parents=True, exist_ok=True)
    synth = make_synthesizer(args)
    requests = load_requests(args.bench)
    client = httpx.Client(base_url=args.url, timeout=180)
    rows = []
    targets = [int(x) for x in args.targets.split(",")]
    for rep in range(args.repeats + 1):
        for target in targets:
            for i in range(args.limit):
                payload = synth.build_request_payload(requests[i])
                payload["token_count"] = target
                payload["max_new_tokens"] = max(256, target * 3)
                t0 = time.perf_counter()
                response = client.post("/v1/audio/speech", json=payload)
                elapsed = (time.perf_counter() - t0) * 1000.0
                response.raise_for_status()
                with wave.open(io.BytesIO(response.content), "rb") as handle:
                    frames = handle.getnframes()
                    rate = handle.getframerate()
                row = {
                    "rep": rep,
                    "warmup": rep == 0,
                    "token_target": target,
                    "sample": i,
                    "ms": elapsed,
                    "frames": frames,
                    "seconds": frames / rate,
                    "codec_frames": round(frames / rate * 12.5),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    (args.out / "sweep.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    measured = [r for r in rows if not r["warmup"]]
    xs = [r["codec_frames"] for r in measured]
    ys = [r["ms"] for r in measured]
    slope, intercept = statistics.linear_regression(xs, ys)
    summary = {
        "n": len(measured),
        "ms_per_codec_frame": slope,
        "intercept_ms": intercept,
        "by_target": {
            t: {
                "median_ms": statistics.median(r["ms"] for r in measured if r["token_target"] == t),
                "median_codec_frames": statistics.median(
                    r["codec_frames"] for r in measured if r["token_target"] == t
                ),
            }
            for t in targets
        },
    }
    (args.out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    synth.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8030")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--config", type=Path, default=DEFAULT_BENCH / "moss_tts.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    render = sub.add_parser("render")
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--warmups", type=int, default=3)
    render.add_argument("--limit", type=int, default=20)
    render.add_argument("--server-events", action="store_true")
    render.add_argument("--cprofile", action="store_true")

    torch_cmd = sub.add_parser("torch")
    torch_cmd.add_argument("--out", type=Path, required=True)
    torch_cmd.add_argument("--run-id", default="torch_trace")
    torch_cmd.add_argument("--warmups", type=int, default=2)
    torch_cmd.add_argument("--limit", type=int, default=3)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--targets", default="25,50,67,100,150")
    sweep.add_argument("--limit", type=int, default=4)
    sweep.add_argument("--repeats", type=int, default=2)

    args = parser.parse_args()
    {"render": cmd_render, "torch": cmd_torch, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()

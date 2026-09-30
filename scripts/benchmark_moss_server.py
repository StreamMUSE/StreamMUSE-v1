"""Server-only, sequential MOSS backend pilot; not a production promotion gate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

LINES = (
    "Rocket blasts liftoff, silence breaks free.",
    "Shadows stretch beneath vast cosmic sea.",
    "City lights reflect the dreams we make.",
    "Every step can make the ground shake.",
    "Music moves us through the darkest night.",
    "Keep the rhythm steady, hold it tight.",
    "Turn the page and let the story flow.",
    "Plant the seeds and let the future grow.",
    "Chasing echoes down an empty street.",
    "Find the truth inside a broken beat.",
    "We can build a bridge across the rain.",
    "Let the sunrise wash away the pain.",
    "Silver trains arrive beneath the moon.",
    "All the voices find a common tune.",
    "Write the words that no one dares to say.",
    "Let the bass line lead us all the way.",
    "Through the storm we keep our heads high.",
    "Watch the sparks illuminate the sky.",
    "Heavy footsteps echo through the hall.",
    "We stand tall whenever shadows fall.",
    "Break the silence with a single rhyme.",
    "Make each moment count against the time.",
    "Take a breath and let the pressure go.",
    "Feel the pulse beneath the falling snow.",
    "Neon letters paint the midnight air.",
    "Hope lives in each new dream that we share.",
    "Morning traffic hums a brand new song.",
    "Keep on moving when the road is long.",
    "From the basement to the stage we rise.",
    "See the light reflected in the skies.",
    "Let the drum beats roll across the floor.",
    "Hear the crowd come back and ask for more.",
    "Each new setback teaches us to climb.",
    "Change the whole world one verse at a time.",
    "Build a signal from the static noise.",
    "Make a chorus out of each new voice.",
    "Send a message past the city walls.",
    "We will answer when the future calls.",
    "Keep the promise written in the sound.",
    "Let the final beat bring us around.",
)
REF_TEXT = "We asked over twenty different people, and they all said it was his."


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    content = json.dumps(plain(value), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"Refusing to overwrite different evidence: {path}")
    path.write_text(content)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
    from streammuse.experiments.rap_audio_protocols.contracts import (
        SyllableTarget, TwoBarRenderRequest,
    )

    analyzer = CmuProsodyAnalyzer()
    analyses = [analyzer.analyze(line) for line in LINES]
    bad = [(i, len(a.syllables), a.oov_words, LINES[i])
           for i, a in enumerate(analyses) if len(a.syllables) != 9 or a.oov_words]
    if bad:
        raise ValueError(f"Invalid benchmark bars: {bad}")
    args.root.mkdir(parents=True, exist_ok=True)
    requests = []
    for i in range(20):
        syllables = []
        for bar in range(2):
            ticks = (0, 2, 3, 5, 7, 8, 10, 13, 15) if (i + bar) % 2 else (
                0, 2, 4, 6, 8, 10, 12, 14, 15)
            for s, tick in zip(analyses[2 * i + bar].syllables, ticks):
                tick += bar * 16
                syllables.append(SyllableTarget(
                    word=s.word, index_in_word=s.index_in_word, phonemes=s.phonemes,
                    lexical_stress=s.stress, target_stress=float(s.stress > 0),
                    boundary_strength=2 if tick % 16 == 15 else 0,
                    absolute_tick=i * 32 + tick, tick_in_chunk=tick,
                    target_seconds=tick / 6,
                ))
        request = TwoBarRenderRequest(
            song_id="server-pilot-20", chunk_index=i, start_bar=i * 2,
            end_bar=i * 2 + 2, text=" ".join(LINES[2 * i:2 * i + 2]),
            syllables=tuple(syllables),
        )
        requests.append(request.to_payload())
    write_json(args.root / "corpus.json", requests)
    (args.root / "reference.txt").write_text(REF_TEXT)
    write_json(args.root / "protocol.json", {
        "scope": "server-only renderer, no Qwen/client/network-to-Mac/packaging",
        "order": ["A1", "B1", "B2", "A2"], "requests_per_block": 20,
        "warmups_per_block": args.warmups, "concurrency": 1, "gpu": args.gpu,
        "corpus_sha256": sha(args.root / "corpus.json"),
        "corpus_provenance": "Two historical fixture bars plus 38 curated pilot bars; not a production corpus",
        "reference_sha256": sha(args.reference), "reference_path": str(args.reference),
        "reference_transcript": REF_TEXT,
        "reference_source": "zhaochenyang20/seed-tts-eval-mini, common_voice_en_10119832",
        "model_snapshot": str(args.model), "model_revision": args.model.name,
        "seed_base": 20260816, "seed_pairing": "Same seed per lyric in every block; engines need not generate identical audio",
        "artifact_cache": False, "reference_cache": "baseline off, candidate upstream default on",
        "timing": "warmup excluded; full source WAV includes HTTP on candidate; total includes MMS/R3/final WAV",
    })
    print(f"Prepared {len(requests)} fixed two-bar requests", flush=True)


def load_requests(path):
    from streammuse.experiments.rap_audio_protocols.contracts import (
        SyllableTarget, TwoBarRenderRequest,
    )
    result = []
    for item in json.loads(path.read_text()):
        item.pop("duration_seconds", None)
        item["syllables"] = tuple(SyllableTarget(
            **{**s, "phonemes": tuple(s["phonemes"])}
        ) for s in item["syllables"])
        result.append(TwoBarRenderRequest(**item))
    return result


def audio_info(path):
    if not path.exists():
        return None
    import numpy as np
    from scipy.io import wavfile
    rate, samples = wavfile.read(path)
    scale = np.iinfo(samples.dtype).max if samples.dtype.kind in "iu" else 1
    values = samples.astype(np.float64) / scale
    return {"sample_rate": rate, "frames": len(samples),
            "seconds": len(samples) / rate, "sha256": sha(path),
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "peak": float(np.max(np.abs(values))), "finite": bool(np.isfinite(values).all())}


class RecordingSynthesizer:
    def __init__(self, inner):
        self.inner = inner
        self.last = {}

    def synthesize(self, request, output_wav, *, execution):
        started = time.perf_counter()
        self.last = {}
        result = self.inner.synthesize(request, output_wav, execution=execution)
        self.last = {"moss_ms": (time.perf_counter() - started) * 1000,
                     "serving": plain(result.serving_metadata.to_payload())}
        return result


def run_block(args):
    import torch
    from streammuse.infrastructure.rap.moss_tts import PersistentMossSynthesizer
    from streammuse.infrastructure.rap.sglang_moss_tts import SglangMossConfig, SglangMossSynthesizer
    from streammuse.infrastructure.rap.mms_forced_alignment import MmsForcedAligner
    from streammuse.infrastructure.rap.moss_aligned_phrase import MossAlignedPhraseRenderer
    from streammuse.application.rap.execution import SynthesisExecutionContext

    block = args.root / args.block
    block.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(block / "packages.json", packages)
    started = time.perf_counter()
    if args.backend == "inprocess":
        inner = PersistentMossSynthesizer.load(
            model_id=str(args.model), device="cuda:0", reference_wav=args.reference,
        )
    else:
        inner = SglangMossSynthesizer(SglangMossConfig.from_files(
            base_url=args.url, model_id=args.served_model or str(args.model),
            model_revision=args.model.name, reference_audio_uri=args.reference.as_uri(),
            reference_audio_file=args.reference, reference_text_file=args.root / "reference.txt",
            runtime_config_sha256=sha(args.config), request_timeout_seconds=180,
        ))
        inner.probe()
    load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    aligner = MmsForcedAligner.load(device="cuda:0")
    aligner_load_ms = (time.perf_counter() - started) * 1000
    recording = RecordingSynthesizer(inner)
    renderer = MossAlignedPhraseRenderer(synthesizer=recording, aligner=aligner,
                                        rubberband_version="3.3.0 R3")
    write_json(block / "startup.json", {
        "moss_load_or_probe_ms": load_ms, "aligner_load_ms": aligner_load_ms,
        "backend": args.backend, "torch_cuda": torch.version.cuda,
        "model": str(args.model), "gpu_name": torch.cuda.get_device_name(),
        "threads": torch.get_num_threads(),
    })
    requests = load_requests(args.root / "corpus.json")
    failed = 0
    stop = threading.Event()

    def monitor():
        with (block / "gpu.csv").open("w") as output:
            while not stop.is_set():
                result = subprocess.run([
                    "nvidia-smi", f"--id={args.gpu}",
                    "--query-gpu=timestamp,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, timeout=10)
                output.write(result.stdout)
                output.flush()
                stop.wait(0.5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        jobs = [(True, i, requests[i % len(requests)]) for i in range(args.warmups)]
        jobs += [(False, i, r) for i, r in enumerate(requests[:args.limit])]
        with (block / "records.jsonl").open("w", buffering=1) as ledger:
            for warmup, i, request in jobs:
                path = block / (f"warmup-{i:02}" if warmup else f"sample-{i:02}")
                row = {"block": args.block, "backend": args.backend, "sample": i,
                       "warmup": warmup, "request_sha256": request.sha256,
                       "text": request.text, "target_seconds": request.duration_seconds}
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    result = renderer.render(request, path, execution=SynthesisExecutionContext.from_timeout(
                        240, correlation_id=f"benchmark-{args.block}-{warmup}-{i}"))
                    torch.cuda.synchronize()
                    row.update(success=True, total_ms=(time.perf_counter() - started) * 1000,
                               stages_ms=plain(result.stage_timings_ms),
                               alignment=plain(result.alignment_diagnostics),
                               monitoring=plain(result.monitoring_summary), warnings=list(result.warnings))
                except Exception as exc:
                    failed += 1
                    row.update(success=False, total_ms=(time.perf_counter() - started) * 1000,
                               error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                row.update(recording.last)
                row["source_audio"] = audio_info(path / "source.wav")
                row["final_audio"] = audio_info(path / "vocal.wav")
                ledger.write(json.dumps(plain(row), sort_keys=True) + "\n")
                print(json.dumps({k: row.get(k) for k in (
                    "block", "sample", "warmup", "success", "moss_ms", "total_ms", "error")}), flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=12)
        close = getattr(inner, "close", None)
        if close:
            close()
    if failed:
        raise SystemExit(1)


def summarize(args):
    import numpy as np
    records = [json.loads(line) for label in ("A1", "B1", "B2", "A2")
               for line in (args.root / label / "records.jsonl").read_text().splitlines()]
    measured = [r for r in records if not r["warmup"]]
    summary = {"blocks": {}, "backends": {}, "paired": {}}
    def stats(rows):
        output = {"n": len(rows), "successes": sum(r["success"] for r in rows)}
        for metric in ("moss_ms", "total_ms"):
            values = [r[metric] for r in rows if r.get(metric) is not None and (
                metric == "moss_ms" or r["success"])]
            output[metric] = {"n": len(values), "median": float(np.median(values)),
                              "p95": float(np.percentile(values, 95))} if values else None
        output["source_seconds"] = [r["source_audio"]["seconds"] for r in rows if r["source_audio"]]
        return output
    for label in ("A1", "B1", "B2", "A2"):
        summary["blocks"][label] = stats([r for r in measured if r["block"] == label])
    for backend in ("inprocess", "sglang-omni"):
        summary["backends"][backend] = stats([r for r in measured if r["backend"] == backend])
    for metric in ("moss_ms", "total_ms"):
        paired = []
        for i in sorted({r["sample"] for r in measured}):
            rows = [r for r in measured if r["sample"] == i]
            a = [r[metric] for r in rows if r["backend"] == "inprocess" and r["success"] and metric in r]
            b = [r[metric] for r in rows if r["backend"] == "sglang-omni" and r["success"] and metric in r]
            if len(a) == len(b) == 2:
                paired.append({"sample": i, "a_ms": statistics.mean(a), "b_ms": statistics.mean(b),
                               "speedup": statistics.mean(a) / statistics.mean(b)})
        summary["paired"][metric] = paired
    write_json(args.root / "summary.json", summary)
    print(json.dumps(summary["backends"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "block", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--block")
    parser.add_argument("--backend", choices=("inprocess", "sglang-omni"))
    parser.add_argument("--url", default="http://127.0.0.1:8030")
    parser.add_argument("--served-model")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.reference:
        args.reference = args.reference.absolute()
    {"prepare": prepare, "block": run_block, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()

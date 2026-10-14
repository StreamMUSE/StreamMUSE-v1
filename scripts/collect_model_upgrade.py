"""Collect the model-upgrade results into one data file plus page-ready audio (plan 2026-10-14).

Reads, under ``<out>`` (default ``output/model_upgrade_20261014``):

- ``llm/<label>.json`` from ``compare_lyric_models.py`` (lyric writers);
- ``tts/<run>/quality.json`` and ``tts/<run>/warped/`` from the bar-unit comparison (TTS quality);
- ``tts/latency/<run>/records.json`` (TTS latency on an otherwise idle GPU);
- ``sessions/<combo>/<session>/`` from ``run_model_upgrade_sessions.sh`` (online v3 sessions).

Writes ``<out>/site/data.json`` and loudness-normalized MP3s under ``<out>/site/audio/``.

    uv run python scripts/collect_model_upgrade.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

LLMS = [
    ("qwen2.5-7b", "Qwen2.5-7B-Instruct", "v1.0's lyric model (2024)"),
    ("qwen3.8-27b-fp8", "Qwen3.8-27B (FP8)", "newest single-GPU dense model (2026-08)"),
    ("qwen3.6-27b-fp8", "Qwen3.6-27B (FP8)", "previous dense generation (2026-04)"),
    ("qwen3.6-35b-a3b-fp8", "Qwen3.6-35B-A3B (FP8)", "MoE, 3B active per token"),
    ("qwen3.5-9b", "Qwen3.5-9B", "newest small model (2026-02)"),
    ("qwen2.5-7b-hyphens", "Qwen2.5-7B + syllable hyphens", "A2 prompt variant"),
    ("qwen3.8-27b-fp8-hyphens", "Qwen3.8-27B + syllable hyphens", "A2 prompt variant"),
]
LYRIC_SAMPLES = ["S01", "S08", "S14"]
TTS = [
    # (run dir, unit, label, note, latency run dir)
    ("moss_v15", "phrase", "MOSS-TTS v1.5, two bars at once", "v0.x unit", None),
    ("moss_v15", "bars", "MOSS-TTS v1.5, each bar alone", "", "latency_moss_v15_production"),
    ("moss_v15", "continued", "MOSS-TTS v1.5 + continuation", "v1.0 today", "latency_moss_v15_production"),
    ("moss_local", "bars", "MOSS-TTS-Local v1.5, each bar alone", "", "latency_moss_local"),
    ("moss_local_continued", "continued", "MOSS-TTS-Local v1.5 + continuation", "ported patch", "moss_local_continued"),
    ("higgs", "bars", "Higgs TTS 3 (4B)", "Boson AI, non-commercial", "latency_higgs_bosonai_higgs-tts-3-4b"),
    ("fish_s2_pro", "bars", "Fish Audio S2 Pro", "research license", "latency_fish_s2_pro_fishaudio_s2-pro"),
    ("qwen3_tts", "bars", "Qwen3-TTS 1.7B Base", "Apache 2.0", "latency_qwen3_tts_Qwen_Qwen3-TTS-12Hz-1_7B-Base"),
    ("indextts", "bars", "IndexTTS-2.5", "own Python inference", "latency_indextts_IndexTeam_IndexTTS-2_5"),
]
TTS_ITEMS = [0, 4, 5, 12, 16, 18]
COMBOS = [
    ("qwen25-moss014", "Qwen2.5-7B + MOSS v1.5 (v1.0)"),
    ("qwen25-mosslocal", "Qwen2.5-7B + MOSS-Local"),
    ("qwen35-moss014", "Qwen3.5-9B + MOSS v1.5"),
    ("qwen35-mosslocal", "Qwen3.5-9B + MOSS-Local"),
    ("qwen38-moss014", "Qwen3.8-27B + MOSS v1.5, one GPU"),
    ("qwen38-mosslocal", "Qwen3.8-27B + MOSS-Local, one GPU"),
    ("qwen38-moss014-split", "Qwen3.8-27B (GPU 2) + MOSS v1.5"),
    ("qwen38-mosslocal-split", "Qwen3.8-27B (GPU 2) + MOSS-Local"),
]


def _encode(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "192k", str(target)], check=True)


def _latency(path: Path, unit: str) -> dict | None:
    records = path / "records.json"
    if not records.exists():
        return None
    values = [part["stage_timings_ms"]["moss"] for record in json.loads(records.read_text()) if record["unit"] == unit
              for index, part in enumerate(record["parts"]) if unit != "continued" or index == 1]
    if not values:
        return None
    ordered = sorted(values)
    return {"p50": round(statistics.median(values)), "p95": round(ordered[max(0, int(0.95 * len(ordered)) - 1)]),
            "n": len(values)}


def main() -> None:
    from analyze_lyric_stream_session import summarize

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "model_upgrade_20261014")
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()
    site = args.out / "site"
    data: dict[str, object] = {"llm": [], "tts": [], "tts_items": [], "sessions": []}

    for label, name, note in LLMS:
        path = args.out / "llm" / f"{label}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        lines = {s: [r["text"] for r in result["rows"] if r["sample"] == s] for s in LYRIC_SAMPLES}
        data["llm"].append({"label": label, "name": name, "note": note, "summary": result["summary"], "lines": lines})
    samples = {s["sample"]: s for s in json.loads((ROOT / "demo_page" / "samples.json").read_text())}
    data["lyric_samples"] = [samples[s] for s in LYRIC_SAMPLES]

    manifest = json.loads((args.out / "tts" / "moss_v15" / "warped" / "manifest.json").read_text())
    texts = {(e["round"], e["item"]): e["bar_texts"] for e in manifest}
    data["tts_items"] = [{"item": i, "bar_texts": texts[(0, i)]} for i in TTS_ITEMS]
    reference = args.out / "reference.wav"
    if not args.skip_audio and reference.exists():
        _encode(reference, site / "audio" / "reference.mp3")
    for run, unit, label, note, latency_run in TTS:
        quality = args.out / "tts" / run / "quality.json"
        if not quality.exists():
            continue
        summary = json.loads(quality.read_text())["summary"].get(unit)
        latency = _latency(args.out / "tts" / "latency" / latency_run, unit) if latency_run else None
        clips = {}
        for item in TTS_ITEMS:
            wav = args.out / "tts" / run / "warped" / f"r0-{item:02d}-{unit}.wav"
            if wav.exists():
                clips[str(item)] = f"audio/tts-{run}-{unit}-{item:02d}.mp3"
                if not args.skip_audio:
                    _encode(wav, site / clips[str(item)])
        data["tts"].append({"run": run, "unit": unit, "label": label, "note": note, "quality": summary,
                            "latency_ms": latency, "clips": clips})

    for combo, name in COMBOS:
        for tempo in (90, 120):
            folder = args.out / "sessions" / f"{combo}-{tempo}"
            sessions = sorted(p for p in folder.glob("rap-*") if (p / "events.jsonl").exists()) if folder.exists() else []
            if not sessions:
                continue
            session = sessions[-1]
            summary = summarize(session)
            audio = f"audio/session-{combo}-{tempo}.mp3"
            if not args.skip_audio:
                _encode(session / "mixed.wav", site / audio)
            events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
            lines = [{"bar": e["bar"], "text": e["payload"].get("text"), "choice": e["payload"].get("choice"),
                      "topic": e["payload"].get("topic")} for e in events if e["event_type"] == "bar_frozen"]
            data["sessions"].append({"combo": combo, "name": name, "tempo": tempo, "session": session.name,
                                     "summary": summary, "audio": audio, "lines": lines})

    site.mkdir(parents=True, exist_ok=True)
    (site / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"llm {len(data['llm'])}, tts {len(data['tts'])}, sessions {len(data['sessions'])} -> {site / 'data.json'}")


if __name__ == "__main__":
    main()

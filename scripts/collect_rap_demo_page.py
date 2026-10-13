"""Collect demo-page runs into one data file plus page-ready audio.

Reads ``<out>/runs.jsonl`` (written by ``run_rap_demo_page.py``), keeps the
last successful attempt of every (version, mode, sample), and writes
``<out>/site/data.json`` and ``<out>/site/audio/<version>-<mode>-<sample>.mp3``
(loudness-normalized to -16 LUFS, 192 kbps). Raw WAVs stay where they are.

Per bar it records the lyric line and where its audio came from:
``remote`` (server audio of the current line), ``remote_older`` (v1.0 only:
server audio of an earlier version of the line) or ``espeak`` (local
fallback), with the fallback reason. ``--server-outcomes`` (from
``scan_rap_render_outcomes.py`` on the H200) maps each remote request id to
what the server did with it; the v0.x client reports every non-200 answer and
every request it gave up on as a "transport failure", so the category comes
from the server side: ``timing`` (the server finished, or ran out of budget,
after the client's deadline), ``server_rejected`` (e.g. no valid lyric
candidate), ``client_rejected`` (the client's response checks failed) or
``other``.

    uv run python scripts/collect_rap_demo_page.py --out output/rap_demo_page_2026-10 \\
        --server-outcomes output/rap_demo_page_2026-10/server_outcomes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

TIMING_REASONS = {"deadline_miss", "late", "startup_timeout", "timeout"}
LATE_OUTCOMES = {"completed", "budget_exhausted"}


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(values), 1) if values else None


def _bars(events, server_outcomes):
    rejected = {}
    for event in events:
        if event["event_type"] in ("chunk_remote_rejected", "chunk_fallback_activated"):
            payload = event["payload"]
            for bar in payload.get("bars") or []:
                rejected.setdefault(bar, (event.get("request_id"), payload.get("failure_reason")))
    bars = []
    for event in events:
        if event["event_type"] != "bar_frozen":
            continue
        payload = event["payload"]
        choice = payload.get("choice")
        if choice is not None:  # v1.0
            source = {"remote_latest": "remote", "remote_older": "remote_older"}.get(choice, "espeak")
            reason = None if source == "remote" else choice
        else:  # v0.x
            source = "espeak" if payload.get("fallback") else "remote"
            reason = payload.get("fallback_reason") if source == "espeak" else None
        detail = category = None
        if source == "espeak":
            request_id, failure = rejected.get(event["bar"], (event.get("request_id"), None))
            outcome = server_outcomes.get(request_id or "")
            detail = failure
            if choice is not None or reason in TIMING_REASONS:
                category = "timing"  # v1.0 eSpeak always means the server audio was not there in time
            elif reason == "invalid":
                category = "client_rejected"
            elif outcome in LATE_OUTCOMES:
                category, detail = "timing", f"server {outcome}; the client had already given up"
            elif outcome and outcome != "incomplete":
                category, detail = "server_rejected", outcome
            else:
                category = "other"
        bars.append({
            "bar": event["bar"], "text": payload.get("text"), "source": source,
            "reason": reason, "detail": detail, "category": category,
            "continued": payload.get("continued"),
        })
    bars.sort(key=lambda item: item["bar"])
    # v0.x clients allow one request at a time: after a chunk times out its
    # request is still being torn down, so the next chunk fails at once without
    # reaching the server. That knock-on failure is a timing failure too.
    by_bar = {item["bar"]: item for item in bars}
    for item in bars:
        before = by_bar.get(item["bar"] - 2)
        if item["category"] == "other" and before is not None and before["category"] == "timing":
            item["category"], item["detail"] = "timing", "the previous chunk's late request was still open"
    return bars


def _stats(events, summary):
    timings = [e["payload"].get("stage_timings_ms") or {} for e in events if e["event_type"] == "chunk_remote_completed"]
    return {
        "underruns": (summary.get("audio") or {}).get("underruns"),
        "remote_units": len(timings),
        "server_moss_ms_p50": _median([t.get("moss") for t in timings]),
        "server_generation_ms_p50": _median([t.get("generation") for t in timings]),
        "remote_total_ms_p50": _median([t.get("total") for t in timings]),
        "transfer_ms_p50": _median([t.get("transfer") for t in timings]),
    }


def _encode(wav: Path, mp3: Path) -> None:
    mp3.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-outcomes", type=Path)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()
    server_outcomes = json.loads(args.server_outcomes.read_text()) if args.server_outcomes else {}
    latest = {}
    for line in (args.out / "runs.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["exit"] == 0 and len(record["session_dirs"]) == 1:
            latest[(record["version"], record["mode"], record["sample"])] = record
    site = args.out / "site"
    runs = []
    for (version, mode, sample), record in sorted(latest.items()):
        session = Path(record["session_dirs"][0])
        events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
        summary = json.loads((session / "summary.json").read_text())
        fake = json.loads((session / "fake_realtime.json").read_text()) if (session / "fake_realtime.json").exists() else None
        wav = session / "mixed.wav"
        audio = f"audio/{version}-{mode}-{sample}.mp3"
        if not args.skip_audio:
            _encode(wav, site / audio)
        bars = _bars(events, server_outcomes)
        runs.append({
            "version": version, "tag": record["tag"], "mode": mode, "sample": sample,
            "topic": record["topic"], "template_id": record["template_id"], "seed": record["seed"],
            "session": session.name, "rtt_seconds_p50": _median(record["rtt_seconds"]),
            "started_utc": record["started_utc"], "attempt": record["attempt"],
            "wav": str(wav), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "audio": audio,
            "bars": bars,
            "counts": {source: sum(1 for bar in bars if bar["source"] == source) for source in ("remote", "remote_older", "espeak")},
            "stats": _stats(events, summary),
            "fake_realtime": fake,
        })
    site.mkdir(parents=True, exist_ok=True)
    (site / "data.json").write_text(json.dumps({"runs": runs}, indent=1) + "\n")
    print(f"{len(runs)} runs -> {site / 'data.json'}")


if __name__ == "__main__":
    main()

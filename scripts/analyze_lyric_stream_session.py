"""Summarize a protocol v3 (lyric stream) demo session from its events.jsonl.

Reports per-bar commit choices (latest remote / older remote / eSpeak of the
streamed line / eSpeak of a local line), how far ahead of its commit point each
lyric and each remote audio was ready, fetch and Mac-side timings, revisions,
and playback underruns.

    uv run python scripts/analyze_lyric_stream_session.py <session-dir> [more...]
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def _pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[max(0, math.ceil(q * len(values)) - 1)], 1)


def summarize(session: Path) -> dict[str, object]:
    events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
    frozen = [e for e in events if e["event_type"] == "bar_frozen"]
    choices = Counter(e["payload"].get("choice") for e in frozen)
    lyric_lead = [e["payload"].get("lead_ms") for e in events if e["event_type"] == "bar_reserved" and e["payload"].get("source") == "lyric_stream"]
    remote_lead = [e["payload"].get("remote_ready_lead_ms") for e in frozen]
    completed = [e["payload"].get("stage_timings_ms", {}) for e in events if e["event_type"] == "chunk_remote_completed"]
    revisions = [e["payload"].get("revision") for e in frozen if e["payload"].get("revision") is not None]
    stale = sum(
        1 for e in frozen
        if e["payload"].get("revision") is not None
        and e["payload"].get("latest_revision") is not None
        and e["payload"]["revision"] < e["payload"]["latest_revision"]
    )
    summary = json.loads((session / "summary.json").read_text()) if (session / "summary.json").exists() else {}
    playback = {e["bar"]: e["monotonic_ns"] for e in events if e["event_type"] == "bar_playback_started"}
    reactions = []
    for edit in (e for e in events if e["event_type"] == "plan_edited"):
        payload = edit["payload"]
        target = [
            e for e in frozen
            if e["bar"] >= payload["effective_from_bar"]
            and (payload["topic"] is None or e["payload"].get("topic") == payload["topic"])
            and (payload["template_id"] is None or e["payload"].get("template_id") == payload["template_id"])
        ]
        first = min(target, key=lambda e: e["bar"]) if target else None
        between = [
            e["payload"].get("choice") for e in frozen
            if payload["effective_from_bar"] <= e["bar"] < (first["bar"] if first else 10**9)
        ]
        reactions.append({
            "edit_seq": payload["edit_seq"],
            "effective_from_bar": payload["effective_from_bar"],
            "first_new_bar": first["bar"] if first else None,
            "first_new_choice": first["payload"].get("choice") if first else None,
            "bars_before_change": len(between),
            "choices_before_change": between,
            "reaction_ms": (
                round((playback[first["bar"]] - edit["monotonic_ns"]) / 1e6) if first and first["bar"] in playback else None
            ),
        })
    return {
        "session": session.name,
        "bars": len(frozen),
        "choices": dict(choices),
        "remote_latest_rate": round(choices.get("remote_latest", 0) / max(1, len(frozen)), 3),
        "lyric_lead_ms": {"p5": _pct(lyric_lead, 0.05), "p50": _pct(lyric_lead, 0.5)},
        "remote_ready_lead_ms": {"min": _pct(remote_lead, 0.0001), "p5": _pct(remote_lead, 0.05), "p50": _pct(remote_lead, 0.5)},
        "fetch_transfer_ms": {"p50": _pct([t.get("transfer") for t in completed], 0.5), "p95": _pct([t.get("transfer") for t in completed], 0.95)},
        "mac_prepare_ms": {"p50": _pct([t.get("mac") for t in completed], 0.5), "p95": _pct([t.get("mac") for t in completed], 0.95)},
        "server_moss_ms": {"p50": _pct([t.get("server_moss") for t in completed], 0.5), "p95": _pct([t.get("server_moss") for t in completed], 0.95)},
        "continued_rate": (
            round(sum(t.get("continued", 0.0) for t in completed) / len(completed), 3) if completed else None
        ),
        "max_revision_committed": max(revisions) if revisions else None,
        "committed_older_revision": stale,
        "underruns": (summary.get("audio") or {}).get("underruns"),
        "presentation_errors": sum(1 for e in events if e["event_type"] == "presentation_error"),
        "edits": reactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.sessions:
        sessions = [path] if (path / "events.jsonl").exists() else sorted(p for p in path.iterdir() if (p / "events.jsonl").exists())
        for session in sessions:
            print(json.dumps(summarize(session), indent=1))


if __name__ == "__main__":
    main()

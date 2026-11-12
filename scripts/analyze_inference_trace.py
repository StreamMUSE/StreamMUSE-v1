"""Break an exported inference-page trace into per-request stage times.

Input: the JSON from ``GET /api/inference/export?format=json`` (the page's
"Export" button, or ``trace.json`` written by the acceptance scripts).

    uv run python scripts/analyze_inference_trace.py output/.../trace.json
    uv run python scripts/analyze_inference_trace.py trace.json --json report.json

For every request it reports, relative to the moment the request was released:

- when the first bar's lyric reached the Mac;
- when the first bar's audio was ready to play (fetched, decoded, warped, mixed);
- cyclic mode: when MODEL started, which is the WAIT the user experienced;
- the server's own render time and the Mac's preparation time for that bar;
- whether each bar was synthesized as a continuation of the line before it.

For every steering edit it reports how long until the new version was first
performed, which bar that was relative to the bar playing at the edit, and how
much old-version audio started playing after the edit (whole bars of the old
plan; the rest of the bar that was already playing is not counted here, it is
part of the page's total stale time).

All times are Mac monotonic milliseconds; server durations are the server's own
measurements and are never subtracted from Mac timestamps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _spread(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def _bar_of(output_id, request_id):
    prefix = request_id + ":bar:"
    return int(output_id[len(prefix):]) if output_id.startswith(prefix) and output_id[len(prefix):].isdigit() else None


def analyze(trace: dict) -> dict:
    """The report for one exported trace; see the module text for what each field means."""
    if trace.get("schema_version") != 1 or "events" not in trace:
        raise ValueError("not an inference trace export (schema_version 1 with an events list)")
    events = trace["events"]
    snapshot = trace.get("snapshot") or {}
    requests: dict[str, dict] = {}
    edits: list[dict] = []
    playback: list[dict] = []  # callback commitments, in order
    sessions = []

    for event in events:
        kind, at = event.get("type"), event.get("at_ms")
        if kind == "request_released":
            context = event["context"]
            requests[event["request_id"]] = {
                "request_id": event["request_id"], "mode": context.get("config", {}).get("mode"),
                "epoch": context["epoch"], "input_version": context["input_version"],
                "bpm": context["bpm"], "start_bar": context["start_tick"] // 16,
                "bars": context["duration_ticks"] // 16, "released_ms": at,
                "lyric_ms": {}, "ready_ms": {}, "stages": {}, "continuation": {},
                "wait_ms": None, "outcome": None, "retries": 0, "error": None,
            }
        elif kind in ("backend", "backend_late"):
            payload = event.get("payload") or {}
            row = requests.get(payload.get("request_id"))
            what = payload.get("type")
            if what == "steering_session":
                sessions.append({key: payload.get(key) for key in (
                    "remote_session_id", "start_bar", "lyric_lead", "audio_lead", "continue_from")})
            if row is None:
                continue
            if what == "candidate_counts" and payload.get("bar") not in row["lyric_ms"]:
                row["lyric_ms"][payload["bar"]] = at - row["released_ms"]
            elif what == "stage_timings":
                row["stages"][payload["bar"]] = payload.get("timings_ms", {})
                if "continuation" in payload:
                    row["continuation"][payload["bar"]] = payload["continuation"]
            elif what == "retry_warning":
                row["retries"] += 1
            elif what in ("complete", "canceled", "error") and row["outcome"] is None:
                row["outcome"] = what
                if what == "error":
                    row["error"] = payload.get("message")
        elif kind == "decoded_ready":
            row = requests.get(event.get("request_id"))
            bar = None if row is None else _bar_of(event.get("output_id", ""), row["request_id"])
            if bar is not None and bar not in row["ready_ms"]:
                row["ready_ms"][bar] = at - row["released_ms"]
        elif kind == "model_started":
            # The request this start answers: the newest one of the same epoch released before it.
            started = event.get("observed_callback_ms", at)
            candidates = [row for row in requests.values()
                          if row["epoch"] == event.get("epoch") and row["released_ms"] <= started
                          and row["wait_ms"] is None]
            if candidates:
                row = max(candidates, key=lambda item: item["released_ms"])
                row["wait_ms"] = started - row["released_ms"]
        elif kind == "committed":
            playback.append(event)
        elif kind == "steered":
            edits.append({"input_version": event.get("input_version"), "at_ms": at, "epoch": event.get("epoch"),
                          "changed": sorted(key for key in ("topic", "template_id") if key in event)})

    request_rows = []
    for row in requests.values():
        first = min(row["lyric_ms"], default=None) if row["lyric_ms"] else None
        first_ready = min(row["ready_ms"], default=None) if row["ready_ms"] else None
        stage = row["stages"].get(first_ready if first_ready is not None else 0, {})
        lyric = row["lyric_ms"].get(first) if first is not None else None
        ready = row["ready_ms"].get(first_ready) if first_ready is not None else None
        request_rows.append({
            "request_id": row["request_id"], "mode": row["mode"], "epoch": row["epoch"],
            "input_version": row["input_version"], "start_bar": row["start_bar"], "bars": row["bars"],
            "outcome": row["outcome"], "error": row["error"], "retries": row["retries"],
            "first_lyric_ms": lyric, "first_bar_ready_ms": ready,
            "lyric_to_ready_ms": ready - lyric if lyric is not None and ready is not None else None,
            "wait_ms": row["wait_ms"],
            "ready_to_start_ms": row["wait_ms"] - ready if row["wait_ms"] is not None and ready is not None else None,
            "all_bars_ready_ms": max(row["ready_ms"].values()) if len(row["ready_ms"]) == row["bars"] else None,
            "first_bar_fetch_ms": stage.get("client_fetch_decode_ms"),
            "first_bar_server_render_ms": stage.get("server_render_total_ms"),
            "first_bar_server_moss_ms": stage.get("server_moss_ms"),
            "first_bar_prepare_ms": stage.get("client_prepare_ms"),
            "continuation": {str(bar): value for bar, value in sorted(row["continuation"].items())},
        })

    # Steering edits: the first callback commitment that plays the edit's version.
    edit_rows = []
    for edit in edits:
        version, at = edit["input_version"], edit["at_ms"]
        bpm = next((row["bpm"] for row in requests.values() if row["input_version"] == version), None)
        before = [item for item in playback if item["at_ms"] <= at and item.get("epoch") == edit["epoch"]]
        after = [item for item in playback if item["at_ms"] > at and item.get("epoch") == edit["epoch"]]
        performed = next((item for item in after
                          if item.get("input_version") == version and item.get("state") == "committed"), None)
        playing_bar = position = None
        if before and bpm:
            last = before[-1]
            tick = last["start_tick"] + (at - last["committed_ms"]) * bpm * 4 / 60000
            playing_bar, position = int(tick // 16), (tick % 16) / 16
        stale_ticks = sum(item["duration_ticks"] for item in after
                          if item.get("state") == "stale"
                          and (performed is None or item["at_ms"] < performed["at_ms"]))
        edit_rows.append({
            "input_version": version, "changed": edit["changed"],
            "playing_bar": playing_bar,
            "position_in_bar": position,
            "half": None if position is None else ("first" if position < 0.5 else "second"),
            "latency_ms": performed["committed_ms"] - at if performed else None,
            "performed_bar": int(performed["start_tick"] // 16) if performed else None,
            "bars_later": (int(performed["start_tick"] // 16) - playing_bar
                           if performed and playing_bar is not None else None),
            "old_plan_started_after_edit_ms": stale_ticks * 60000 / bpm / 4 if bpm else None,
        })

    # Continuation rate: every bar with a server report except the very first bar of the trace,
    # which has nothing before it.
    reports = [(row["released_ms"], bar, value) for row in requests.values()
               for bar, value in row["continuation"].items()]
    reports.sort(key=lambda item: (item[0], item[1]))
    counted = reports[1:]
    reasons: dict[str, int] = {}
    for _, _, value in counted:
        if not value.get("used"):
            reason = str(value.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
    continuation = {
        "bars_reported": len(reports), "bars_counted": len(counted),
        "continued": sum(1 for _, _, value in counted if value.get("used")),
        "handed_over": sum(1 for _, _, value in counted if value.get("handoff")),
        "rate": (sum(1 for _, _, value in counted if value.get("used")) / len(counted)) if counted else None,
        "not_continued_reasons": reasons,
    }

    cycle = [row for row in request_rows if row["mode"] == "cycle" and row["wait_ms"] is not None]
    metrics = snapshot.get("metrics") or {}
    summary = {
        "requests": len(request_rows),
        "outcomes": {name: sum(1 for row in request_rows if row["outcome"] == name)
                     for name in ("complete", "canceled", "error")},
        "steering_adapter": (snapshot.get("capabilities") or {}).get("steering_adapter"),
        "cycle_wait_ms": _spread([row["wait_ms"] for row in cycle]),
        "cycle_first_lyric_ms": _spread([row["first_lyric_ms"] for row in cycle]),
        "cycle_lyric_to_ready_ms": _spread([row["lyric_to_ready_ms"] for row in cycle]),
        "cycle_ready_to_start_ms": _spread([row["ready_to_start_ms"] for row in cycle]),
        "cycle_first_bar_fetch_ms": _spread([row["first_bar_fetch_ms"] for row in cycle]),
        "cycle_first_bar_server_render_ms": _spread([row["first_bar_server_render_ms"] for row in cycle]),
        "cycle_first_bar_prepare_ms": _spread([row["first_bar_prepare_ms"] for row in cycle]),
        "steering_latency_ms": _spread([row["latency_ms"] for row in edit_rows]),
        "steering_bars_later": [row["bars_later"] for row in edit_rows],
        "continuation": continuation,
        "missing_duration_ms": metrics.get("missing_duration_ms"),
        "stale_duration_ms": metrics.get("stale_duration_ms"),
        "eligible_duration_ms": metrics.get("eligible_duration_ms"),
        "error_count": metrics.get("error_count"),
        "steering_sessions": sessions,
    }
    return {"summary": summary, "requests": request_rows, "edits": edit_rows}


def _ms(value):
    return "-" if value is None else f"{value:.0f}"


def _spread_text(value):
    if value is None:
        return "-"
    return f"median {value['median']:.0f} ms (min {value['min']:.0f}, max {value['max']:.0f}, n={value['n']})"


def render(report: dict) -> str:
    summary, lines = report["summary"], []
    outcomes = ", ".join(f"{count} {name}" for name, count in summary["outcomes"].items() if count)
    lines.append(f"requests: {summary['requests']} ({outcomes or 'no outcome recorded'})")
    if summary["steering_adapter"]:
        lines.append(f"steering adapter: {summary['steering_adapter']}")
    lines.append(f"eligible {_ms(summary['eligible_duration_ms'])} ms, missing {_ms(summary['missing_duration_ms'])} ms, "
                 f"stale {_ms(summary['stale_duration_ms'])} ms, errors {summary['error_count']}")
    if summary["cycle_wait_ms"]:
        lines += ["", "Cyclic turns (from request release)",
                  f"  first bar's lyric at Mac   {_spread_text(summary['cycle_first_lyric_ms'])}",
                  f"  then until audio ready     {_spread_text(summary['cycle_lyric_to_ready_ms'])}",
                  f"    of which fetch           {_spread_text(summary['cycle_first_bar_fetch_ms'])}",
                  f"    of which server render   {_spread_text(summary['cycle_first_bar_server_render_ms'])}",
                  f"    of which Mac preparation {_spread_text(summary['cycle_first_bar_prepare_ms'])}",
                  f"  then until MODEL starts    {_spread_text(summary['cycle_ready_to_start_ms'])}",
                  f"  WAIT                       {_spread_text(summary['cycle_wait_ms'])}"]
    if report["edits"]:
        lines += ["", "Steering edits",
                  "  version  bar  half    new version after  bars later  old-plan bars started after the edit"]
        for row in report["edits"]:
            lines.append(f"  {row['input_version']!s:>7}  {row['playing_bar']!s:>3}  {row['half'] or '-':6}  "
                         f"{_ms(row['latency_ms']):>14} ms  {row['bars_later']!s:>10}  "
                         f"{_ms(row['old_plan_started_after_edit_ms']):>10} ms")
        lines.append(f"  latency: {_spread_text(summary['steering_latency_ms'])}")
    continuation = summary["continuation"]
    if continuation["bars_reported"]:
        rate = "-" if continuation["rate"] is None else f"{continuation['rate'] * 100:.1f}%"
        lines += ["", f"Continuation: {continuation['continued']}/{continuation['bars_counted']} bars ({rate}) "
                      f"continued the line before; {continuation['handed_over']} across a session boundary",
                  f"  (the first bar of the trace is not counted) not continued: "
                  f"{continuation['not_continued_reasons'] or 'none'}"]
    else:
        lines += ["", "Continuation: the trace has no server continuation reports "
                      "(recorded from the stage 2 client onward)"]
    lines += ["", "Requests",
              "  request                              mode      bars  outcome   lyric  ready   WAIT  fetch  render"]
    for row in report["requests"]:
        label = row["request_id"].rsplit(":", 2)
        lines.append(f"  {':'.join(label[-2:]):36s} {row['mode'] or '-':8s} {row['start_bar']:>3}+{row['bars']:<2} "
                     f"{row['outcome'] or '-':9s} {_ms(row['first_lyric_ms']):>5}  {_ms(row['first_bar_ready_ms']):>5}  "
                     f"{_ms(row['wait_ms']):>5}  {_ms(row['first_bar_fetch_ms']):>5}  "
                     f"{_ms(row['first_bar_server_render_ms']):>6}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path, help="exported trace JSON")
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)
    try:
        report = analyze(json.loads(args.trace.read_text()))
    except (OSError, ValueError) as error:
        print(f"cannot analyze {args.trace}: {error}", file=sys.stderr)
        return 2
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

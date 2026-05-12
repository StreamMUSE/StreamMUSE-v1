"""Join per-event [EVT_T*] traces from server + client logs and compute lateness."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple


EVT_RE = re.compile(
    r"\[EVT_(T[0-6])\]\s+"
    r"tick=(?P<tick>-?\d+)\s+"
    r"pitch=(?P<pitch>-?\d+)\s+"
    r"type=(?P<type>\w+)\s+"
    r"wall=(?P<wall>[0-9.]+)"
)
SCHED_TICK_RE = re.compile(r"schedule_tick=(-?\d+)")
CUR_TICK_RE = re.compile(r"current_tick=(-?\d+)")

SESSION_RE = re.compile(
    r"\[SESSION_START\]\s+wall=(?P<wall>[0-9.]+)\s+"
    r"bpm=(?P<bpm>[0-9.]+)\s+"
    r"ticks_per_beat=(?P<tpb>\d+)"
)


def parse_log(path: str) -> tuple[dict, dict, dict]:
    """Return (stage_to_earliest_wall, sched_records, session_meta_dict)."""
    stage_to_first: Dict[str, Dict[Tuple[int, int, str], float]] = defaultdict(dict)
    sched_records: Dict[Tuple[int, int, str], Tuple[int, int]] = {}  # (sched_tick, current_tick)
    session: Dict[str, Optional[float]] = {"wall": None, "bpm": None, "tpb": None}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[SESSION_START]"):
                m = SESSION_RE.search(line)
                if m:
                    session["wall"] = float(m.group("wall"))
                    session["bpm"] = float(m.group("bpm"))
                    session["tpb"] = int(m.group("tpb"))
                continue
            m = EVT_RE.search(line)
            if not m:
                continue
            stage = m.group(1)
            key = (int(m.group("tick")), int(m.group("pitch")), m.group("type"))
            wall = float(m.group("wall"))
            d = stage_to_first[stage]
            if key not in d or wall < d[key]:
                d[key] = wall
            if stage == "T5":
                st_m = SCHED_TICK_RE.search(line)
                ct_m = CUR_TICK_RE.search(line)
                if st_m and ct_m and key not in sched_records:
                    sched_records[key] = (int(st_m.group(1)), int(ct_m.group(1)))
    return stage_to_first, sched_records, session


def merge_stages(*sources: dict) -> Dict[str, Dict[Tuple[int, int, str], float]]:
    merged: Dict[str, Dict[Tuple[int, int, str], float]] = defaultdict(dict)
    for src in sources:
        for stage, mapping in src.items():
            for key, wall in mapping.items():
                if key not in merged[stage] or wall < merged[stage][key]:
                    merged[stage][key] = wall
    return merged


def pct(xs: Iterable[float], q: float) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    i = int(q * (len(xs) - 1))
    return xs[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--client-log", required=True)
    ap.add_argument("--prompt-length-ticks", type=int, default=32)
    ap.add_argument("--bpm", type=float, default=None,
                    help="override bpm if SESSION_START not found")
    ap.add_argument("--ticks-per-beat", type=int, default=None,
                    help="override ticks_per_beat if SESSION_START not found")
    ap.add_argument("--session-start", type=float, default=None,
                    help="override session_start wall if SESSION_START not found")
    args = ap.parse_args()

    server, server_sched, server_meta = parse_log(args.server_log)
    client, client_sched, client_meta = parse_log(args.client_log)
    stages = merge_stages(server, client)
    sched_records = {**server_sched, **client_sched}

    session_wall = args.session_start if args.session_start is not None \
        else (server_meta["wall"] or client_meta["wall"])
    bpm = args.bpm if args.bpm is not None \
        else (server_meta["bpm"] or client_meta["bpm"])
    tpb = args.ticks_per_beat if args.ticks_per_beat is not None \
        else (server_meta["tpb"] or client_meta["tpb"])

    if session_wall is None or bpm is None or tpb is None:
        print("ERROR: SESSION_START not found in logs and no override provided.", file=sys.stderr)
        return 1

    seconds_per_tick = (60.0 / float(bpm)) / int(tpb)

    keys_all = set()
    for d in stages.values():
        keys_all.update(d.keys())
    note_on_keys = {k for k in keys_all if k[2] == "note_on"}

    def get(stage: str, key) -> Optional[float]:
        return stages.get(stage, {}).get(key)

    # Compute per-event metrics for note_on (T6 anchored).
    per_event = []
    for key in note_on_keys:
        rec = {st: get(st, key) for st in ("T0", "T1", "T2", "T3", "T4", "T5", "T6")}
        rec["tick"], rec["pitch"], rec["type"] = key
        rec["target_wall"] = session_wall + rec["tick"] * seconds_per_tick
        per_event.append(rec)

    emitted = [r for r in per_event if r["T6"] is not None]
    lateness_ms = [(r["T6"] - r["target_wall"]) * 1000.0 for r in emitted]

    drained = [r for r in per_event if r["T4"] is not None]
    drain_lateness_ms = [(r["T4"] - r["target_wall"]) * 1000.0 for r in drained]

    minted = [r for r in per_event if r["T0"] is not None]
    mint_lateness_ms = [(r["T0"] - r["target_wall"]) * 1000.0 for r in minted]

    def seg(a: str, b: str):
        vals = [(r[b] - r[a]) * 1000.0
                for r in per_event
                if r[a] is not None and r[b] is not None]
        return vals

    seg_gen_to_serialize = seg("T0", "T1")
    seg_network          = seg("T1", "T2")
    seg_client_q_dwell   = seg("T3", "T4")
    seg_sched            = seg("T4", "T5")
    seg_sink_emit        = seg("T5", "T6")
    seg_total            = seg("T0", "T6")

    def fmt(xs):
        if not xs:
            return "n/a"
        return f"p50={pct(xs,0.5):.1f} p95={pct(xs,0.95):.1f} max={max(xs):.1f}  n={len(xs)}"

    n_total = len(per_event)
    n_emitted = len(emitted)
    n_late_0 = sum(1 for x in lateness_ms if x > 0)
    n_late_50 = sum(1 for x in lateness_ms if x > 50)
    n_late_200 = sum(1 for x in lateness_ms if x > 200)

    prompt_lateness = [(r["T6"] - r["target_wall"]) * 1000.0
                       for r in emitted if r["tick"] < args.prompt_length_ticks]
    catchup_lateness = [(r["T6"] - r["target_wall"]) * 1000.0
                        for r in emitted if r["tick"] >= args.prompt_length_ticks]

    print("# Lateness analysis")
    print()
    print(f"session_start_wall = {session_wall:.6f}")
    print(f"bpm = {bpm}  ticks_per_beat = {tpb}  seconds_per_tick = {seconds_per_tick:.6f}")
    print(f"prompt_length_ticks = {args.prompt_length_ticks}")
    print()
    print("## Drain-time lateness (T4 - target_wall) — when client tick_loop drained event from _playable_q")
    print(f"  drained_total: {len(drain_lateness_ms)}")
    if drain_lateness_ms:
        late_pos = sum(1 for x in drain_lateness_ms if x > 0)
        print(f"  late_at_drain (>0ms): {late_pos}  ({late_pos/len(drain_lateness_ms):.1%})")
        print(f"  drain_lateness_ms: {fmt(drain_lateness_ms)}")
    print()
    print("## Mint-time lateness (T0 - target_wall) — when server scheduler first minted the event")
    if mint_lateness_ms:
        late_pos = sum(1 for x in mint_lateness_ms if x > 0)
        print(f"  late_at_mint (>0ms): {late_pos}  ({late_pos/len(mint_lateness_ms):.1%})")
        print(f"  mint_lateness_ms: {fmt(mint_lateness_ms)}")
    print()
    print("## Overall (emit-anchored; requires T6)")
    print(f"  events_total (unique note_on by (tick,pitch)): {n_total}")
    print(f"  events_with_T6 (actually emitted to fluidsynth): {n_emitted}")
    print(f"  events_late_>0ms:    {n_late_0}  ({(n_late_0/max(1,n_emitted)):.1%})")
    print(f"  events_late_>50ms:   {n_late_50}  ({(n_late_50/max(1,n_emitted)):.1%})")
    print(f"  events_late_>200ms:  {n_late_200}  ({(n_late_200/max(1,n_emitted)):.1%})")
    print(f"  lateness_ms: {fmt(lateness_ms)}")
    print()
    print("## Phase split")
    print(f"  prompt  (tick <  {args.prompt_length_ticks}): n={len(prompt_lateness)}  {fmt(prompt_lateness)}")
    print(f"  catchup (tick >= {args.prompt_length_ticks}): n={len(catchup_lateness)}  {fmt(catchup_lateness)}")
    print()
    print("## Segment medians (ms)")
    print(f"  T0->T1 gen_to_serialize: {fmt(seg_gen_to_serialize)}")
    print(f"  T1->T2 network:          {fmt(seg_network)}")
    print(f"  T3->T4 client_q_dwell:   {fmt(seg_client_q_dwell)}")
    print(f"  T4->T5 sched:            {fmt(seg_sched)}")
    print(f"  T5->T6 sink_emit:        {fmt(seg_sink_emit)}")
    print(f"  T0->T6 total_pipeline:   {fmt(seg_total)}")
    print()
    # Schedule relationship analysis
    print("## Schedule relationship (event.tick vs schedule_tick) — only note_on with T5")
    on_time = 0         # schedule_tick == event.tick (event in future or exactly now)
    future = 0          # schedule_tick == event.tick and event.tick > current_tick
    just_in_time = 0    # schedule_tick == event.tick and event.tick == current_tick
    bursted = 0         # schedule_tick == current_tick and event.tick < current_tick
    bursted_by_phase: Dict[str, int] = {"prompt": 0, "catchup": 0}
    burst_lateness: list = []
    for key, (sched, cur) in sched_records.items():
        if key[2] != "note_on":
            continue
        ev_tick = key[0]
        if ev_tick > cur:
            future += 1
            on_time += 1
        elif ev_tick == cur:
            just_in_time += 1
            on_time += 1
        else:
            bursted += 1
            burst_lateness.append(cur - ev_tick)
            phase = "prompt" if ev_tick < args.prompt_length_ticks else "catchup"
            bursted_by_phase[phase] += 1
    total = on_time + bursted
    print(f"  total_note_on_scheduled: {total}")
    print(f"  future       (sched_tick==event.tick > current_tick): {future}")
    print(f"  just_in_time (sched_tick==event.tick == current_tick): {just_in_time}")
    print(f"  bursted      (sched_tick==current_tick, event.tick < current_tick): {bursted}")
    print("    bursted by phase: prompt={p} catchup={c}".format(p=bursted_by_phase["prompt"], c=bursted_by_phase["catchup"]))
    if burst_lateness:
        burst_lateness.sort()
        print(f"    bursted late_ticks: p50={burst_lateness[len(burst_lateness)//2]} p95={burst_lateness[int(len(burst_lateness)*0.95)]} max={burst_lateness[-1]}")
    print()
    # Per-stage event coverage diagnostic.
    print("## Coverage by stage (events seen at each stage)")
    for st in ("T0", "T1", "T2", "T3", "T4", "T5", "T6"):
        d = stages.get(st, {})
        n = sum(1 for k in d if k[2] == "note_on")
        print(f"  {st}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

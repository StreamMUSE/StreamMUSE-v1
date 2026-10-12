"""Lyric-stream plan, phase 1: sequential per-bar writer vs the paired two-bar planner.

Writes the same bars of the default scenario twice against one live
OpenAI-compatible Qwen server:

- ``paired``: today's ChunkCandidatePlanner (two bars written independently,
  realtime_default policy, then the best pair);
- ``sequential``: SequentialLyricWriter (one bar at a time, ``--candidates`` per
  bar, couplet rhymes: odd bars rhyme with the bar before).

Reports the couplet rhyme rate (bar 2k+1's end rhyme against bar 2k: exact
tail or same vowels, the scorer's ``rhyme_quality`` >= 0.6), mean score and
components, rescue waves and Qwen time per bar.

    python scripts/compare_lyric_writers.py --vllm-url http://127.0.0.1:8001/v1 --bars 48 --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]


def main() -> None:
    from streammuse.application.rap.chunk_orchestration import ChunkCandidatePlanner, NoValidCandidates
    from streammuse.application.rap.lyric_writer import (
        LyricWriteFailed,
        LyricWriteRequest,
        LyricWriterPolicy,
        SequentialLyricWriter,
    )
    from streammuse.application.rap.scoring import rhyme_quality
    from streammuse.domain.rap import (
        RemoteCandidatePolicy,
        RemoteRapBarRequest,
        RemoteRapChunkRequest,
        ScoreWeights,
    )
    from streammuse.infrastructure.inference.local_chat_client import (
        LocalChatModelClient,
        LocalChatModelClientConfig,
    )
    from streammuse.infrastructure.rap.generators import IndependentChoiceCandidateGenerator
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
    from streammuse.infrastructure.rap.scenarios import default_scenario
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", default="qwen-rap")
    parser.add_argument("--bars", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenario = default_scenario()
    analyzer = CmuProsodyAnalyzer()
    config = LocalChatModelClientConfig(base_url=args.vllm_url, model=args.vllm_model, timeout_s=60.0)
    clients = [LocalChatModelClient(config) for _ in range(2)]
    generators = [IndependentChoiceCandidateGenerator(c, max_tokens_per_choice=32, temperature=1.0) for c in clients]

    def plan_for(bar):
        segment = scenario.segment_for_bar(bar)
        return segment.topic, BUILTIN_TEMPLATES.get(segment.template_id)

    # -- paired (today) ------------------------------------------------------------
    planner = ChunkCandidatePlanner(
        generators[0], analyzer, ScoreWeights(), bar_generators=(generators[0], generators[1])
    )
    paired = []
    failures = {"paired": 0, "sequential": 0}
    failure_reasons: dict[str, int] = {}
    context: list[str] = []
    for start in range(0, args.bars, 2):
        bars = tuple(RemoteRapBarRequest(bar, *plan_for(bar)) for bar in (start, start + 1))
        request = RemoteRapChunkRequest.create(
            session_id="paired", chunk_index=start // 2, bars=bars, tempo_bpm=scenario.tempo_bpm,
            remaining_budget_ms=60_000, policy=RemoteCandidatePolicy.realtime_default(),
            context_lines=tuple(context[-4:]), seed=args.seed + start,
        )
        started = time.perf_counter()
        try:
            plan = planner.plan(request)
        except NoValidCandidates:
            failures["paired"] += 2  # the runtime would play two fallback bars
            continue
        elapsed = (time.perf_counter() - started) * 1000.0
        for selected in plan.selected_bars:
            waves = {row.get("wave") for row in plan.candidate_ledger if row.get("bar") == selected.bar and row.get("wave") is not None}
            paired.append({
                "bar": selected.bar, "text": selected.text, "score": selected.score, "waves": len(waves),
                "generation_ms_per_bar": plan.stage_timings_ms["generation"] / 2, "wall_ms_per_bar": elapsed / 2,
            })
            context.append(selected.text)

    # -- sequential (v3) -----------------------------------------------------------
    writer = SequentialLyricWriter(
        generators[0], analyzer, ScoreWeights(), LyricWriterPolicy(candidates=args.candidates),
    )
    sequential = []
    history: list[str] = []
    previous = None
    for bar in range(args.bars):
        topic, template = plan_for(bar)
        started = time.perf_counter()
        try:
            written = writer.write(LyricWriteRequest(
                session_id="sequential", bar=bar, revision=0, topic=topic, flow_template=template,
                history=tuple(history[-4:]), seed=args.seed + bar,
                previous_line=previous.text if previous else None,
                previous_template=previous.flow_template if previous else None,
            ))
        except LyricWriteFailed as error:
            failures["sequential"] += 1
            for row in error.ledger:
                for reason in row.get("reasons") or ():
                    failure_reasons[str(reason).split(":")[0]] = failure_reasons.get(str(reason).split(":")[0], 0) + 1
            previous = None
            continue
        elapsed = (time.perf_counter() - started) * 1000.0
        record = written.record
        sequential.append({
            "bar": bar, "text": record.text, "score": record.score, "waves": written.waves,
            "generation_ms_per_bar": written.generation_ms, "wall_ms_per_bar": elapsed,
            "rhyme_with": record.provenance.get("rhyme_with"),
            "valid": record.provenance.get("candidates_valid"),
        })
        history.append(record.text)
        previous = record

    def rhyme_rate(rows):
        hits = []
        by_bar = {row["bar"]: row for row in rows}
        for bar in range(0, args.bars, 2):
            even, odd = by_bar.get(bar), by_bar.get(bar + 1)
            if even is None or odd is None:
                continue
            tail_a = tuple(analyzer.analyze(even["text"]).end_rhyme_tail)
            tail_b = tuple(analyzer.analyze(odd["text"]).end_rhyme_tail)
            hits.append(rhyme_quality(tail_b, tail_a) >= 0.6)
        return sum(hits) / max(1, len(hits))

    def summary(name, rows):
        return {
            "bars": len(rows),
            "failed_bars": failures[name],
            "rhyme_label_leaks": sum(1 for row in rows if row["text"].rstrip(" .,!?").endswith(" A")),
            "repeated_end_words": sum(
                1 for before, after in zip(rows, rows[1:])
                if after["bar"] == before["bar"] + 1
                and before["text"].lower().split()[-1].strip(".,!?") == after["text"].lower().split()[-1].strip(".,!?")
            ),
            "distinct_end_words": len({row["text"].lower().split()[-1].strip(".,!?") for row in rows}),
            "couplet_rhyme_rate": rhyme_rate(rows),
            "mean_score": statistics.mean(row["score"] for row in rows),
            "generation_ms_per_bar_median": statistics.median(row["generation_ms_per_bar"] for row in rows),
            "wall_ms_per_bar_median": statistics.median(row["wall_ms_per_bar"] for row in rows),
            "rescue_rate": sum(1 for row in rows if row["waves"] > 1) / len(rows),
        }

    result = {"paired": summary("paired", paired), "sequential": summary("sequential", sequential),
              "sequential_failure_reasons": failure_reasons, "rows": {"paired": paired, "sequential": sequential}}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("paired", "sequential", "sequential_failure_reasons")}, indent=1))
    for client in clients:
        client.close()


if __name__ == "__main__":
    main()

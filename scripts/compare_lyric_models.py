"""Compare lyric-writer LLMs on the demo-page scenarios (model-upgrade plan, A1).

Runs the v1.0 ``SequentialLyricWriter`` (32 candidates per bar, couplet
rhymes, rescue waves) against one OpenAI-compatible model and writes every
bar of the 15 demo-page samples (5 topics x 3 flow templates, 16 bars each).

Per model it reports the time of each generation request (first wave and
whole bar), the first-wave candidate pass rate (how many of the 32 lines pass
the syllable/stress/rhyme checks: the most direct measure of the model),
rejection reasons, rescue and failure rates, the couplet rhyme rate, distinct
end words and the mean score. Every written line is kept for listening pages.

    python scripts/compare_lyric_models.py --vllm-url http://127.0.0.1:8401/v1 --vllm-model cand \\
        --label qwen3.8-27b-fp8 --extra-payload '{"chat_template_kwargs": {"enable_thinking": false}}' \\
        --out <json>

Plan: developing-logs/plans/2026-10-14-rap-model-upgrade-plan.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]


def _pct(values, q):
    values = sorted(values)
    return round(values[max(0, math.ceil(q * len(values)) - 1)], 1) if values else None


class _TimedGenerator:
    """Records the wall time and wave of every generation request."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[dict[str, object]] = []

    def generate(self, request):
        started = time.perf_counter()
        batch = self._inner.generate(request)
        self.calls.append({
            "request_id": request.request_id, "count": request.count,
            "ms": (time.perf_counter() - started) * 1000.0,
            "received": len(batch.candidates), "error": batch.error if hasattr(batch, "error") else None,
        })
        return batch


def main() -> None:
    from streammuse.application.rap.lyric_writer import (
        LyricWriteFailed,
        LyricWriteRequest,
        LyricWriterPolicy,
        SequentialLyricWriter,
    )
    from streammuse.application.rap.scoring import rhyme_quality
    from streammuse.domain.rap import ScoreWeights
    from streammuse.infrastructure.inference.local_chat_client import (
        LocalChatModelClient,
        LocalChatModelClientConfig,
    )
    from streammuse.infrastructure.rap.generators import IndependentChoiceCandidateGenerator
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
    from streammuse.infrastructure.rap.scenarios import load_scenario
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--extra-payload", default="", help="JSON merged into every chat request")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--samples", default="", help="comma-separated sample ids (default: all)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    samples = json.loads((ROOT / "demo_page" / "samples.json").read_text())
    if args.samples:
        samples = [s for s in samples if s["sample"] in set(args.samples.split(","))]
    extra = json.loads(args.extra_payload) if args.extra_payload else None
    config = LocalChatModelClientConfig(base_url=args.vllm_url, model=args.vllm_model, timeout_s=120.0, extra_payload=extra)
    client = LocalChatModelClient(config)
    generator = _TimedGenerator(IndependentChoiceCandidateGenerator(client, max_tokens_per_choice=32, temperature=1.0))
    analyzer = CmuProsodyAnalyzer()
    writer = SequentialLyricWriter(generator, analyzer, ScoreWeights(), LyricWriterPolicy(candidates=args.candidates))

    rows = []
    reasons = Counter()
    first_wave = {"rows": 0, "passed": 0}
    for sample in samples:
        scenario = load_scenario(ROOT / "demo_page" / "scenarios" / f"{sample['sample']}.json")
        history: list[str] = []
        previous = None
        for bar in range(args.bars):
            segment = scenario.segment_for_bar(bar)
            template = BUILTIN_TEMPLATES.get(segment.template_id)
            calls_before = len(generator.calls)
            started = time.perf_counter()
            try:
                written = writer.write(LyricWriteRequest(
                    session_id=f"{args.label}-{sample['sample']}", bar=bar, revision=0, topic=segment.topic,
                    flow_template=template, history=tuple(history[-4:]), seed=sample["seed"] * 100 + bar,
                    previous_line=previous.text if previous else None,
                    previous_template=previous.flow_template if previous else None,
                ))
                ledger, record, failed = written.ledger, written.record, False
            except LyricWriteFailed as error:
                ledger, record, failed = error.ledger, None, True
            wall_ms = (time.perf_counter() - started) * 1000.0
            calls = generator.calls[calls_before:]
            for row in ledger:
                if row.get("wave") == 0 and row.get("status") not in ("generation_warning", "generation_error"):
                    first_wave["rows"] += 1
                    first_wave["passed"] += row.get("status") in ("valid", "selected")
                if row.get("status") not in ("valid", "selected", "generation_warning"):
                    for reason in row.get("reasons") or ():
                        reasons[str(reason).split(":")[0]] += 1
            rows.append({
                "sample": sample["sample"], "topic": segment.topic, "template_id": segment.template_id, "bar": bar,
                "text": record.text if record else None, "score": record.score if record else None,
                "failed": failed, "waves": len(calls), "wall_ms": wall_ms,
                "first_wave_ms": calls[0]["ms"] if calls else None,
                "rhyme_with": record.provenance.get("rhyme_with") if record else None,
            })
            if record is not None:
                history.append(record.text)
                previous = record
            else:
                previous = None

    def rhyme_rate():
        hits = []
        by_key = {(r["sample"], r["bar"]): r for r in rows}
        for (sample, bar), row in by_key.items():
            before = by_key.get((sample, bar - 1))
            if bar % 2 == 1 and row["text"] and before and before["text"] and row["template_id"] == before["template_id"]:
                hits.append(rhyme_quality(tuple(analyzer.analyze(row["text"]).end_rhyme_tail),
                                          tuple(analyzer.analyze(before["text"]).end_rhyme_tail)) >= 0.6)
        return round(sum(hits) / max(1, len(hits)), 3)

    written_rows = [r for r in rows if r["text"]]
    end_words = Counter(r["text"].lower().rstrip(" .,!?").split()[-1] for r in written_rows)
    per_sample_distinct = statistics.mean(
        len({r["text"].lower().rstrip(" .,!?").split()[-1] for r in written_rows if r["sample"] == s["sample"]})
        for s in samples
    )
    summary = {
        "label": args.label, "model": args.vllm_model, "bars": len(rows),
        "failed_bars": sum(r["failed"] for r in rows),
        "first_wave_pass_rate": round(first_wave["passed"] / max(1, first_wave["rows"]), 3),
        "rescue_rate": round(sum(1 for r in rows if r["waves"] > 1) / len(rows), 3),
        "first_wave_ms": {"p50": _pct([r["first_wave_ms"] for r in rows if r["first_wave_ms"]], 0.5),
                          "p95": _pct([r["first_wave_ms"] for r in rows if r["first_wave_ms"]], 0.95)},
        "bar_wall_ms": {"p50": _pct([r["wall_ms"] for r in rows], 0.5), "p95": _pct([r["wall_ms"] for r in rows], 0.95),
                        "max": round(max(r["wall_ms"] for r in rows), 1)},
        "couplet_rhyme_rate": rhyme_rate(),
        "distinct_end_words_per_16_bars": round(per_sample_distinct, 2),
        "most_common_end_words": end_words.most_common(8),
        "mean_score": round(statistics.mean(r["score"] for r in written_rows), 3) if written_rows else None,
        "rejection_reasons": dict(reasons.most_common(12)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    client.close()


if __name__ == "__main__":
    main()

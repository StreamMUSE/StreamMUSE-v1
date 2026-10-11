#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        records.append(payload)
    return records


def _records_by_beat(records: Iterable[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    ignored = 0
    for record in records:
        target_beat = record.get("target_beat")
        if target_beat is None:
            ignored += 1
            continue
        grouped.setdefault(int(target_beat), []).append(record)
    return grouped, ignored


def compare_model_output_records(
    streammuse_records: Iterable[dict[str, Any]],
    offline_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    stream_by_beat, stream_ignored = _records_by_beat(streammuse_records)
    offline_by_beat, offline_ignored = _records_by_beat(offline_records)
    offline_min_beat = min(offline_by_beat) if offline_by_beat else None
    offline_max_beat = max(offline_by_beat) if offline_by_beat else None
    stream_outside_offline_range = [
        beat
        for beat in sorted(stream_by_beat)
        if offline_min_beat is None
        or offline_max_beat is None
        or beat < offline_min_beat
        or beat > offline_max_beat
    ]
    compared_stream_beats = [
        beat for beat in sorted(stream_by_beat) if beat not in stream_outside_offline_range
    ]

    missing_in_offline: list[int] = []
    mismatches: list[dict[str, Any]] = []
    exact_match_beats: list[int] = []
    prompt_digest_match_count = 0
    prompt_digest_mismatch_count = 0

    for beat in compared_stream_beats:
        offline_group = offline_by_beat.get(beat)
        if not offline_group:
            missing_in_offline.append(beat)
            continue

        stream_digests = sorted({str(record.get("output_digest")) for record in stream_by_beat[beat]})
        offline_digests = sorted({str(record.get("output_digest")) for record in offline_group})
        if len(stream_digests) == 1 and stream_digests == offline_digests:
            exact_match_beats.append(beat)
        else:
            mismatches.append(
                {
                    "target_beat": beat,
                    "streammuse_output_digests": stream_digests,
                    "offline_output_digests": offline_digests,
                    "streammuse_output_tokens": [record.get("output_tokens", []) for record in stream_by_beat[beat]],
                    "offline_output_tokens": [record.get("output_tokens", []) for record in offline_group],
                }
            )

        stream_prompt = stream_by_beat[beat][0].get("prompt_digest")
        offline_prompt = offline_group[0].get("prompt_digest")
        if stream_prompt is not None and offline_prompt is not None:
            if stream_prompt == offline_prompt:
                prompt_digest_match_count += 1
            else:
                prompt_digest_mismatch_count += 1

    stream_beats = sorted(stream_by_beat)
    offline_beats = sorted(offline_by_beat)
    passed = bool(compared_stream_beats) and not missing_in_offline and not mismatches
    return {
        "passed": passed,
        "comparison_scope": "StreamMUSE target beats within the offline song range",
        "streammuse_record_count": sum(len(group) for group in stream_by_beat.values()),
        "offline_record_count": sum(len(group) for group in offline_by_beat.values()),
        "streammuse_target_beats": stream_beats,
        "streammuse_target_beats_compared": compared_stream_beats,
        "offline_target_beats": offline_beats,
        "streammuse_ignored_structural_records": stream_ignored,
        "offline_ignored_structural_records": offline_ignored,
        "streammuse_duplicate_target_beats": [
            beat for beat, group in stream_by_beat.items() if len(group) > 1
        ],
        "offline_duplicate_target_beats": [
            beat for beat, group in offline_by_beat.items() if len(group) > 1
        ],
        "exact_match_count": len(exact_match_beats),
        "exact_match_beats": exact_match_beats,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "missing_in_offline": missing_in_offline,
        "streammuse_beats_outside_offline_range": stream_outside_offline_range,
        "extra_offline_beats": sorted(set(offline_beats) - set(stream_beats)),
        "prompt_digest_match_count": prompt_digest_match_count,
        "prompt_digest_mismatch_count": prompt_digest_mismatch_count,
        "prompt_digest_affects_pass": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-beat StreamMUSE and offline model outputs")
    parser.add_argument("--streammuse-trace", type=Path, required=True)
    parser.add_argument("--offline-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_model_output_records(
        load_trace(args.streammuse_trace),
        load_trace(args.offline_trace),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Single-model ZipZapZop memory sweep.

This script runs the ProjectIsochron-style ZipZapZop task through StreamMUSE's
batch TaskRuntime and aggregates latency/accuracy across history windows,
temperatures, APC labels, and optional oracle-history controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from streammuse.domain.tasks import ZipZapZopTask
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.presentation.task.cli import run_task


@dataclass(frozen=True)
class HistoryCell:
    label: str
    history_limit: int


@dataclass(frozen=True)
class RunCell:
    cell_order_index: int
    temperature: float
    repeat: int
    history_label: str
    history_limit: int
    oracle: bool = False

    @property
    def config_label(self) -> str:
        return "all_oracle" if self.oracle else self.history_label


@dataclass(frozen=True)
class ModelSpecialConfig:
    max_tokens: int
    extra_payload: dict[str, Any] | None
    answer_mode: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single-model ZipZapZop memory sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--history-limits", default="0,8,32,all")
    parser.add_argument("--temperatures", default="0,0.7")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repeats-nonzero-temp", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--sweep-root", default=None)
    parser.add_argument("--pilot", action="store_true", help="Use fixed order and skip oracle add-on runs")
    parser.add_argument("--apc-label", choices=["on", "off"], default="on")
    parser.add_argument("--spec-label", default="none")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--no-oracle", action="store_true", help="Skip the all+greedy oracle-history add-on run")
    parser.add_argument("--allow-model-alias", action="store_true", help="Allow /models to expose a single alias instead of --model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.turns <= 0:
        raise SystemExit("--turns must be positive")
    if args.repeats_nonzero_temp <= 0:
        raise SystemExit("--repeats-nonzero-temp must be positive")

    sweep_root = Path(args.sweep_root or f"task_runs/zzz_memory_sweep_{time.strftime('%Y%m%d-%H%M%S')}").expanduser().resolve()
    runs_root = sweep_root / "runs"
    sweep_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    history_cells = parse_history_cells(args.history_limits, turns=args.turns)
    temperatures = parse_float_list(args.temperatures)
    special = model_special_config(args.model, explicit_max_tokens=args.max_tokens)
    schedule = build_schedule(
        history_cells,
        temperatures,
        repeats_nonzero_temp=int(args.repeats_nonzero_temp),
        pilot=bool(args.pilot),
        include_oracle=(not args.pilot and not args.no_oracle),
        turns=int(args.turns),
    )

    preflight: dict[str, Any] = {"skipped": bool(args.skip_preflight)}
    reset_probe: dict[str, Any] = {"skipped": bool(args.skip_preflight), "available": False}
    run_records: list[dict[str, Any]] = []
    per_turn_rows: list[dict[str, Any]] = []
    status_code = 0

    run_config = {
        "schema_version": 1,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "model_url": args.model_url,
        "turns": int(args.turns),
        "history_limits": [asdict(cell) for cell in history_cells],
        "temperatures": temperatures,
        "top_p": float(args.top_p),
        "repeats_nonzero_temp": int(args.repeats_nonzero_temp),
        "effective_max_tokens": int(special.max_tokens),
        "apc_label": args.apc_label,
        "spec_label": args.spec_label,
        "pilot": bool(args.pilot),
        "oracle_enabled": bool(not args.pilot and not args.no_oracle),
        "special_config": asdict(special),
        "schedule": [asdict(cell) for cell in schedule],
        "preflight": preflight,
        "reset_prefix_cache": reset_probe,
        "run_records": run_records,
    }
    write_json(sweep_root / "run_config.json", run_config)

    try:
        if not args.skip_preflight:
            preflight = preflight_check(args.model_url, args.model, allow_alias=bool(args.allow_model_alias))
            reset_probe = probe_reset_prefix_cache(args.model_url)
            run_config["preflight"] = preflight
            run_config["reset_prefix_cache"] = reset_probe
            write_json(sweep_root / "run_config.json", run_config)
        if not args.skip_warmup:
            warmup(args.model_url, args.model, special, timeout_s=float(args.timeout_s))
    except Exception as exc:  # noqa: BLE001 - user-facing CLI should keep the reason in config.
        status_code = 1
        run_config["fatal_error"] = describe_exception(exc)
        write_json(sweep_root / "run_config.json", run_config)
        write_outputs(sweep_root, per_turn_rows, run_records, args.turns)
        print(f"preflight/warmup failed: {exc}", file=sys.stderr)
        return status_code

    for cell in schedule:
        reset_info = reset_prefix_cache(args.model_url, reset_probe) if reset_probe.get("available") else dict(reset_probe)
        record = {
            "cell_order_index": int(cell.cell_order_index),
            "temperature": float(cell.temperature),
            "repeat": int(cell.repeat),
            "history_label": cell.history_label,
            "history_limit": int(cell.history_limit),
            "config": cell.config_label,
            "oracle": bool(cell.oracle),
            "top_p": float(args.top_p) if cell.temperature > 0 else None,
            "apc": args.apc_label,
            "spec_decode": args.spec_label,
            "reset_prefix_cache": reset_info,
            "status": "pending",
        }
        run_records.append(record)
        write_json(sweep_root / "run_config.json", {**run_config, "run_records": run_records})
        try:
            result = run_task(
                task_name="zip_zap_zop",
                runner_kind="offline_benchmark",
                max_turns=int(args.turns),
                model_url=args.model_url,
                model=args.model,
                timeout_s=float(args.timeout_s),
                max_tokens=int(special.max_tokens),
                temperature=float(cell.temperature),
                tick_rate_hz=1.0,
                deadline_ms=1_000_000.0,
                output_dir=str(runs_root),
                start_number=1,
                history_limit=int(cell.history_limit),
                top_p=float(args.top_p) if cell.temperature > 0 else None,
                extra_payload=special.extra_payload,
                oracle_history=bool(cell.oracle),
            )
            record.update(
                {
                    "status": "ok",
                    "run_dir": result.output_dir,
                    "turn_count": result.turn_count,
                    "valid_count": result.valid_count,
                    "invalid_count": result.invalid_count,
                    "deadline_miss_count": result.deadline_miss_count,
                }
            )
            rows = parse_run_dir(Path(result.output_dir), args.model, args.apc_label, args.spec_label, cell, special.answer_mode)
            per_turn_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "error", "error": describe_exception(exc)})
            status_code = 1
            if is_connection_error(exc):
                print(f"server connection failed during {cell.config_label}: {exc}", file=sys.stderr)
                break
            print(f"run failed for {cell.config_label}: {exc}", file=sys.stderr)
        finally:
            write_outputs(sweep_root, per_turn_rows, run_records, args.turns)
            write_json(sweep_root / "run_config.json", {**run_config, "run_records": run_records})

    final_config = {**run_config, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_records": run_records}
    write_json(sweep_root / "run_config.json", final_config)
    print(f"sweep output: {sweep_root}")
    return status_code


def parse_history_cells(raw: str, *, turns: int) -> list[HistoryCell]:
    cells: list[HistoryCell] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        if value.lower() == "all":
            cells.append(HistoryCell(label="all", history_limit=int(turns)))
        else:
            limit = int(value)
            if limit < 0:
                raise ValueError("history limits must be non-negative")
            label = "memoryless" if limit == 0 else f"recent-{limit}"
            cells.append(HistoryCell(label=label, history_limit=limit))
    if not cells:
        raise ValueError("at least one history limit is required")
    return cells


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one temperature is required")
    return values


def model_special_config(model: str, *, explicit_max_tokens: int | None) -> ModelSpecialConfig:
    lower = model.lower()
    extra_payload: dict[str, Any] | None = None
    max_tokens = 8 if explicit_max_tokens is None else int(explicit_max_tokens)
    answer_mode = "content"
    if model == "Qwen/Qwen3-8B" or model.startswith("Qwen/Qwen3.6-"):
        extra_payload = {"chat_template_kwargs": {"enable_thinking": False}}
    if "gpt-oss" in lower:
        max_tokens = 512 if explicit_max_tokens is None else int(explicit_max_tokens)
        answer_mode = "gpt_oss_final"
    return ModelSpecialConfig(max_tokens=max_tokens, extra_payload=extra_payload, answer_mode=answer_mode)


def build_schedule(
    history_cells: list[HistoryCell],
    temperatures: list[float],
    *,
    repeats_nonzero_temp: int,
    pilot: bool,
    include_oracle: bool,
    turns: int,
) -> list[RunCell]:
    schedule: list[RunCell] = []
    order_index = 0
    for temperature in temperatures:
        repeat_count = 1 if temperature == 0 else repeats_nonzero_temp
        for repeat in range(repeat_count):
            ordered = history_cells if pilot else balanced_history_order(history_cells, repeat)
            for cell in ordered:
                schedule.append(
                    RunCell(
                        cell_order_index=order_index,
                        temperature=float(temperature),
                        repeat=repeat,
                        history_label=cell.label,
                        history_limit=cell.history_limit,
                    )
                )
                order_index += 1
    if include_oracle:
        schedule.append(
            RunCell(
                cell_order_index=order_index,
                temperature=0.0,
                repeat=0,
                history_label="all",
                history_limit=int(turns),
                oracle=True,
            )
        )
    return schedule


def balanced_history_order(cells: list[HistoryCell], repeat: int) -> list[HistoryCell]:
    if repeat % 3 == 0:
        return list(cells)
    if repeat % 3 == 1:
        return list(reversed(cells))
    if len(cells) == 4:
        return [cells[1], cells[3], cells[0], cells[2]]
    offset = repeat % len(cells)
    return list(cells[offset:] + cells[:offset])


def preflight_check(model_url: str, model: str, *, allow_alias: bool) -> dict[str, Any]:
    url = f"{model_url.rstrip('/')}/models"
    started = time.perf_counter()
    response = requests.get(url, timeout=10.0)
    latency_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    payload = response.json()
    model_ids = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
    if model not in model_ids:
        if not (allow_alias and len(model_ids) == 1):
            raise RuntimeError(f"model {model!r} not exposed by {url}; available={model_ids}")
    return {"url": url, "available_models": model_ids, "latency_ms": latency_ms, "matched": model in model_ids}


def server_root_from_model_url(model_url: str) -> str:
    parsed = urlparse(model_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def probe_reset_prefix_cache(model_url: str) -> dict[str, Any]:
    url = f"{server_root_from_model_url(model_url)}/reset_prefix_cache"
    started = time.perf_counter()
    try:
        response = requests.post(url, timeout=5.0)
    except requests.RequestException as exc:
        return {"available": False, "url": url, "method": "unavailable", "error": str(exc)}
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "available": 200 <= int(response.status_code) < 300,
        "url": url,
        "method": "post" if 200 <= int(response.status_code) < 300 else "unavailable",
        "status_code": int(response.status_code),
        "latency_ms": latency_ms,
        "response_text": response.text[:500],
    }


def reset_prefix_cache(model_url: str, reset_probe: dict[str, Any]) -> dict[str, Any]:
    if not reset_probe.get("available"):
        return dict(reset_probe)
    url = str(reset_probe.get("url") or f"{server_root_from_model_url(model_url)}/reset_prefix_cache")
    started = time.perf_counter()
    try:
        response = requests.post(url, timeout=10.0)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "available": 200 <= int(response.status_code) < 300,
            "url": url,
            "method": "post",
            "status_code": int(response.status_code),
            "latency_ms": latency_ms,
            "response_text": response.text[:500],
        }
    except requests.RequestException as exc:
        return {"available": False, "url": url, "method": "post", "error": str(exc)}


def warmup(model_url: str, model: str, special: ModelSpecialConfig, *, timeout_s: float) -> None:
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url=model_url,
            model=model,
            timeout_s=timeout_s,
            top_p=None,
            extra_payload=special.extra_payload,
        )
    )
    try:
        for number in (1, 2):
            client.generate(
                [
                    {"role": "system", "content": ZipZapZopTask.rules_prompt()},
                    {"role": "user", "content": f"{number}:"},
                ],
                max_tokens=min(int(special.max_tokens), 16),
                temperature=0.0,
            )
    finally:
        client.close()


def parse_run_dir(
    run_dir: Path,
    model: str,
    apc_label: str,
    spec_label: str,
    cell: RunCell,
    answer_mode: str,
) -> list[dict[str, Any]]:
    trace_rows = load_jsonl(run_dir / "trace.jsonl")
    response_rows = {int(row.get("turn_id", -1)): row for row in load_jsonl(run_dir / "response_trace.jsonl")}
    per_turn: list[dict[str, Any]] = []
    for event in trace_rows:
        if event.get("stage") != "task_turn":
            continue
        summary = event.get("summary") if isinstance(event.get("summary"), dict) else {}
        metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
        turn_id = int(summary.get("turn_id", event.get("logical_tick", len(per_turn))))
        artifact = load_turn_artifact(run_dir, event)
        turn_payload = artifact.get("turn") if isinstance(artifact.get("turn"), dict) else {}
        turn_metadata = turn_payload.get("metadata") if isinstance(turn_payload.get("metadata"), dict) else {}
        response_row = response_rows.get(turn_id, {})
        prompt = response_row.get("prompt") or turn_payload.get("messages") or []
        response_text = str(response_row.get("response", artifact.get("model_response", "")) or "")
        expected = str(summary.get("expected_output") or turn_payload.get("expected_output") or "")
        latency_ms = optional_float(metrics.get("latency_ms", summary.get("latency_ms")))
        prompt_tokens = optional_int(metrics.get("prompt_tokens"))
        completion_tokens = optional_int(metrics.get("completion_tokens"))
        raw_response = artifact.get("model_response_raw") if isinstance(artifact.get("model_response_raw"), dict) else {}
        per_turn.append(
            {
                "model": model,
                "apc": apc_label,
                "spec_decode": spec_label,
                "temperature": cell.temperature,
                "config": cell.config_label,
                "history_limit": cell.history_limit,
                "repeat": cell.repeat,
                "cell_order_index": cell.cell_order_index,
                "run_dir": str(run_dir),
                "turn_id": turn_id,
                "number": turn_metadata.get("number"),
                "response": response_text,
                "expected": expected,
                "strict_valid": bool(summary.get("is_valid", False)),
                "normalized_valid": normalize_answer(response_text) == normalize_answer(expected),
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": tokens_per_second(completion_tokens, latency_ms),
                "answer_extract_status": answer_extract_status(answer_mode, raw_response, response_text),
                "oracle": cell.oracle,
                "prompt_hash": stable_hash(prompt),
            }
        )
    return per_turn


def load_turn_artifact(run_dir: Path, event: dict[str, Any]) -> dict[str, Any]:
    refs = event.get("output_refs")
    if not isinstance(refs, list) or not refs:
        return {}
    first = refs[0]
    if not isinstance(first, dict) or not first.get("path"):
        return {}
    path = run_dir / str(first["path"])
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_outputs(sweep_root: Path, per_turn_rows: list[dict[str, Any]], run_records: list[dict[str, Any]], turns: int) -> None:
    write_csv(sweep_root / "per_turn.csv", per_turn_rows)
    summary_rows = aggregate_rows(per_turn_rows, turns=turns)
    write_json(sweep_root / "sweep_summary.json", {"runs": run_records, "summary": summary_rows})
    (sweep_root / "sweep_summary.md").write_text(render_markdown_summary(summary_rows, run_records), encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]], *, turns: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["apc"],
            row["spec_decode"],
            row["temperature"],
            row["config"],
            row["oracle"],
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        run_groups: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            run_groups.setdefault(str(row["run_dir"]), []).append(row)
        strict_accs = [accuracy(run, "strict_valid") for run in run_groups.values()]
        norm_accs = [accuracy(run, "normalized_valid") for run in run_groups.values()]
        first_errors = [first_error_turn(run) for run in run_groups.values()]
        before_accs = [accuracy_before_first_error(run) for run in run_groups.values()]
        post_accs = [post_error_accuracy(run) for run in run_groups.values()]
        latencies = [float(row["latency_ms"]) for row in group_rows if row.get("latency_ms") is not None]
        prompt_tokens = [int(row["prompt_tokens"]) for row in group_rows if row.get("prompt_tokens") is not None]
        completion_tokens = [int(row["completion_tokens"]) for row in group_rows if row.get("completion_tokens") is not None]
        tps = [float(row["tokens_per_second"]) for row in group_rows if row.get("tokens_per_second") is not None]
        summaries.append(
            {
                "model": key[0],
                "apc": key[1],
                "spec_decode": key[2],
                "temperature": key[3],
                "config": key[4],
                "oracle": key[5],
                "n_runs": len(run_groups),
                "turns": len(group_rows),
                "strict_accuracy_mean": safe_mean(strict_accs),
                "strict_accuracy_std": safe_std(strict_accs),
                "normalized_accuracy_mean": safe_mean(norm_accs),
                "normalized_accuracy_std": safe_std(norm_accs),
                "first_error_turn_min": min([value for value in first_errors if value is not None], default=None),
                "first_error_turns": first_errors,
                "accuracy_before_first_error_mean": safe_mean([value for value in before_accs if value is not None]),
                "post_error_accuracy_mean": safe_mean([value for value in post_accs if value is not None]),
                "latency_mean_ms": safe_mean(latencies),
                "latency_median_ms": safe_median(latencies),
                "latency_p95_ms": percentile(latencies, 95),
                "latency_max_ms": max(latencies, default=None),
                "first25_latency_median_ms": segment_median(group_rows, 0, min(25, turns)),
                "middle_latency_median_ms": segment_median(group_rows, min(25, turns), max(turns - 25, min(25, turns))),
                "last25_latency_median_ms": segment_median(group_rows, max(turns - 25, 0), turns),
                "avg_prompt_tokens": safe_mean(prompt_tokens),
                "avg_completion_tokens": safe_mean(completion_tokens),
                "tokens_per_second_median": safe_median(tps),
            }
        )
    return summaries


def accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def first_error_turn(rows: list[dict[str, Any]]) -> int | None:
    ordered = sorted(rows, key=lambda row: int(row.get("turn_id", 0)))
    for row in ordered:
        if not bool(row.get("strict_valid")):
            return int(row.get("turn_id", 0))
    return None


def accuracy_before_first_error(rows: list[dict[str, Any]]) -> float | None:
    first = first_error_turn(rows)
    if first is None:
        return accuracy(rows, "strict_valid")
    before = [row for row in rows if int(row.get("turn_id", 0)) < first]
    return accuracy(before, "strict_valid") if before else None


def post_error_accuracy(rows: list[dict[str, Any]]) -> float | None:
    first = first_error_turn(rows)
    if first is None:
        return None
    after = [row for row in rows if int(row.get("turn_id", 0)) > first]
    return accuracy(after, "strict_valid") if after else None


def segment_median(rows: list[dict[str, Any]], start: int, end: int) -> float | None:
    values = [float(row["latency_ms"]) for row in rows if start <= int(row.get("turn_id", 0)) < end and row.get("latency_ms") is not None]
    return safe_median(values)


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().lower()
    quote_chars = "'\"`“”‘’"
    while len(text) >= 2 and text[0] in quote_chars and text[-1] in quote_chars:
        text = text[1:-1].strip()
    trailing = ".,!?:;。，！？：；"
    while text and text[-1] in trailing:
        text = text[:-1].strip()
    return " ".join(text.split())


def answer_extract_status(answer_mode: str, raw_response: dict[str, Any], response_text: str) -> str:
    if answer_mode != "gpt_oss_final":
        return "content"
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "raw_unavailable"
    first = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = first.get("finish_reason")
    if not str(response_text or "").strip():
        return "empty_final_length" if finish_reason == "length" else "empty_final"
    return "content_length_finish" if finish_reason == "length" else "content"


def tokens_per_second(completion_tokens: int | None, latency_ms: float | None) -> float | None:
    if completion_tokens is None or latency_ms is None or latency_ms <= 0:
        return None
    return float(completion_tokens) / float(latency_ms) * 1000.0


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def safe_std(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.stdev(cleaned) if len(cleaned) >= 2 else None


def safe_median(values: list[float | int | None]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.median(cleaned) if cleaned else None


def percentile(values: list[float | int | None], pct: float) -> float | None:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return None
    index = max(0, min(len(cleaned) - 1, math.ceil((pct / 100.0) * len(cleaned)) - 1))
    return cleaned[index]


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "apc",
        "spec_decode",
        "temperature",
        "config",
        "history_limit",
        "repeat",
        "cell_order_index",
        "run_dir",
        "turn_id",
        "number",
        "response",
        "expected",
        "strict_valid",
        "normalized_valid",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "tokens_per_second",
        "answer_extract_status",
        "oracle",
        "prompt_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")


def render_markdown_summary(summary_rows: list[dict[str, Any]], run_records: list[dict[str, Any]]) -> str:
    lines = ["# ZipZapZop Memory Sweep Summary", ""]
    lines.append("## Runs")
    lines.append("")
    ok = sum(1 for record in run_records if record.get("status") == "ok")
    errors = [record for record in run_records if record.get("status") == "error"]
    lines.append(f"- completed runs: {ok}")
    lines.append(f"- errored runs: {len(errors)}")
    if errors:
        lines.append("")
        for record in errors:
            lines.append(f"- error cell {record.get('cell_order_index')} {record.get('config')}: {record.get('error')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    headers = [
        "model",
        "apc",
        "spec",
        "temp",
        "config",
        "oracle",
        "runs",
        "strict_acc",
        "norm_acc",
        "first_err",
        "median_ms",
        "p95_ms",
        "avg_prompt_tok",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model")),
                    str(row.get("apc")),
                    str(row.get("spec_decode")),
                    fmt(row.get("temperature")),
                    str(row.get("config")),
                    str(row.get("oracle")),
                    str(row.get("n_runs")),
                    fmt(row.get("strict_accuracy_mean")),
                    fmt(row.get("normalized_accuracy_mean")),
                    str(row.get("first_error_turn_min")),
                    fmt(row.get("latency_median_ms")),
                    fmt(row.get("latency_p95_ms")),
                    fmt(row.get("avg_prompt_tokens")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def describe_exception(exc: BaseException) -> dict[str, Any]:
    return {"type": exc.__class__.__name__, "message": str(exc), "traceback": traceback.format_exc()}


def is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.RequestException):
        return True
    cause = getattr(exc, "__cause__", None)
    return isinstance(cause, requests.RequestException)


if __name__ == "__main__":
    raise SystemExit(main())

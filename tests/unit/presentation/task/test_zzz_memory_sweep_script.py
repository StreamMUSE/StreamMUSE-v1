from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[4] / "scripts" / "run_zzz_memory_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_zzz_memory_sweep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


def test_build_schedule_uses_balanced_order_and_oracle_tail() -> None:
    cells = sweep.parse_history_cells("0,8,32,all", turns=100)

    schedule = sweep.build_schedule(
        cells,
        [0.0, 0.7],
        repeats_nonzero_temp=3,
        pilot=False,
        include_oracle=True,
        turns=100,
    )

    normal_repeat_1 = [cell.history_label for cell in schedule if cell.temperature == 0.7 and cell.repeat == 1]
    normal_repeat_2 = [cell.history_label for cell in schedule if cell.temperature == 0.7 and cell.repeat == 2]
    assert normal_repeat_1 == ["all", "recent-32", "recent-8", "memoryless"]
    assert normal_repeat_2 == ["recent-8", "all", "memoryless", "recent-32"]
    assert schedule[-1].oracle is True
    assert schedule[-1].config_label == "all_oracle"


def test_normalize_answer_keeps_format_strict_but_removes_trailing_punctuation() -> None:
    assert sweep.normalize_answer('"Zip."') == "zip"
    assert sweep.normalize_answer(" ZipZapZop! ") == "zipzapzop"
    assert sweep.normalize_answer("Zip Zap") == "zip zap"


def test_aggregate_rows_reports_strict_and_normalized_accuracy() -> None:
    rows = [
        {
            "model": "m",
            "apc": "on",
            "spec_decode": "none",
            "temperature": 0.0,
            "config": "memoryless",
            "oracle": False,
            "run_dir": "run-a",
            "turn_id": 0,
            "strict_valid": True,
            "normalized_valid": True,
            "latency_ms": 10.0,
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "tokens_per_second": 100.0,
        },
        {
            "model": "m",
            "apc": "on",
            "spec_decode": "none",
            "temperature": 0.0,
            "config": "memoryless",
            "oracle": False,
            "run_dir": "run-a",
            "turn_id": 1,
            "strict_valid": False,
            "normalized_valid": True,
            "latency_ms": 20.0,
            "prompt_tokens": 6,
            "completion_tokens": 1,
            "tokens_per_second": 50.0,
        },
    ]

    summary = sweep.aggregate_rows(rows, turns=2)[0]

    assert summary["strict_accuracy_mean"] == 0.5
    assert summary["normalized_accuracy_mean"] == 1.0
    assert summary["first_error_turn_min"] == 1
    assert summary["latency_median_ms"] == 15.0

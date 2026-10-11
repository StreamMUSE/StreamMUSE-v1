from __future__ import annotations

import json

from streammuse.infrastructure.inference.model_output_trace import (
    ModelInputTracer,
    ModelOutputTracer,
    token_digest,
)


def test_model_output_tracer_writes_exact_tokens_and_prompt_digest(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = ModelOutputTracer(mode="offline", path=trace_path)

    tracer.record(
        output_tokens=[81, 4, 171],
        target_beat=7,
        input_file="song.npz",
        prompt_tokens=[257, 263, 269],
        metadata={"slot_kind": "beat"},
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["mode"] == "offline"
    assert record["target_beat"] == 7
    assert record["output_tokens"] == [81, 4, 171]
    assert record["output_digest"] == token_digest([81, 4, 171])
    assert record["prompt_digest"] == token_digest([257, 263, 269])


def test_model_output_tracer_is_noop_without_path(monkeypatch, tmp_path):
    monkeypatch.delenv("LEKAI_MODEL_OUTPUT_TRACE_PATH", raising=False)
    tracer = ModelOutputTracer(mode="streammuse_realtime")

    tracer.record(output_tokens=[169], target_beat=0, input_file="song.mid")

    assert not list(tmp_path.iterdir())


def test_model_input_tracer_writes_exact_pre_model_tokens(tmp_path):
    trace_path = tmp_path / "input.jsonl"
    tracer = ModelInputTracer(mode="streammuse_realtime", path=trace_path)

    tracer.record(
        input_tokens=[257, 263, 269, 255, 172],
        target_beat=4,
        input_file="song.mid",
        metadata={"attention_mask": None, "use_cache": True},
    )

    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["record_type"] == "model_input"
    assert record["target_beat"] == 4
    assert record["input_tokens"] == [257, 263, 269, 255, 172]
    assert record["input_shape"] == [1, 5]
    assert record["input_dtype"] == "torch.int64"
    assert record["input_digest"] == token_digest([257, 263, 269, 255, 172])

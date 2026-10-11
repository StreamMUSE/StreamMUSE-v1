from __future__ import annotations

import json

import torch

from scripts.run_lekai_offline_model_output_trace import install_model_output_hook
from streammuse.infrastructure.inference.model_output_trace import ModelInputTracer, ModelOutputTracer


class _FakeWrapper:
    def _generate_one_beat(self, generated, *args, **kwargs):
        return [81, 3, 171], torch.cat([generated, torch.tensor([[171]])], dim=1)


def test_hook_records_returned_beat_without_changing_it(tmp_path):
    wrapper = _FakeWrapper()
    trace_path = tmp_path / "offline.jsonl"
    tracer = ModelOutputTracer(mode="offline", path=trace_path)
    install_model_output_hook(wrapper, tracer, input_file="song.npz")
    prompt = torch.tensor([[257, 263, 269]])

    output, generated = wrapper._generate_one_beat(prompt, object(), 1.0, 1, 1.0, 1.0)

    assert output == [81, 3, 171]
    assert generated.tolist() == [[257, 263, 269, 171]]
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["target_beat"] == 0
    assert record["output_tokens"] == output


def test_hook_records_input_before_generation(tmp_path):
    wrapper = _FakeWrapper()
    output_trace = ModelOutputTracer(mode="offline", path=tmp_path / "output.jsonl")
    input_path = tmp_path / "input.jsonl"
    input_trace = ModelInputTracer(mode="offline", path=input_path)
    install_model_output_hook(
        wrapper,
        output_trace,
        input_file="song.npz",
        input_tracer=input_trace,
    )

    prompt = torch.tensor([[257, 263, 269]])
    wrapper._generate_one_beat(prompt, object(), 1.0, 1, 1.0, 1.0)

    record = json.loads(input_path.read_text(encoding="utf-8"))
    assert record["target_beat"] == 0
    assert record["input_tokens"] == [257, 263, 269]

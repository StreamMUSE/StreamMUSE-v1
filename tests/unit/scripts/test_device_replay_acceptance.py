import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

import torch

from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
from streammuse.infrastructure.inference.lekai_model import generation_utils


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("acceptance", ROOT / "scripts/device_replay_acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def test_observer_preserves_tokens_and_rng_state(tmp_path, monkeypatch):
    class Model:
        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 259))
            logits[:, -1, 170] = 100
            return SimpleNamespace(logits=logits, past_key_values=None)

    def backend():
        obj = object.__new__(LekaiHttpBackend)
        obj._model_adapter = SimpleNamespace(model=Model(), device="cpu", use_cache=False)
        obj._model_generation_lock = threading.Lock()
        obj._sample_generator = torch.Generator().manual_seed(123)
        obj._session_epoch = 1
        obj._session_id = "cpu-test"
        obj._effective_seed = 123
        return obj

    kwargs = dict(temperature=1.1, top_k=50, top_p=.95, repetition_penalty=1.)
    prompt = torch.tensor([257, 255, 172])
    reference = backend()
    expected = reference._generate_part1_tokens_from_prompt(prompt, **kwargs)
    expected_state = reference._sample_generator.get_state()
    monkeypatch.setattr(generation_utils, "sample_token", generation_utils.sample_token)
    monkeypatch.setattr(LekaiHttpBackend, "_generate_part1_tokens_from_prompt",
                        LekaiHttpBackend._generate_part1_tokens_from_prompt)
    acceptance.observer_setup(tmp_path)
    observed = backend()
    assert observed._generate_part1_tokens_from_prompt(prompt, **kwargs) == expected
    assert torch.equal(observed._sample_generator.get_state(), expected_state)
    record = acceptance.read(tmp_path / "sampling_observer.jsonl")
    assert record["returned_tokens"] == expected == [170]
    assert record["samples"][0]["rng_before_sha256"] != record["samples"][0]["rng_after_sha256"]


def test_source_window_is_bounded():
    source = ROOT / "prompts/old_input/mel/001.mid"
    events = acceptance.source_events(source)
    assert events
    assert events[0][0] == 0
    assert max(t for t, _ in events) <= 24 * 60 / 90
    assert all(a[0] <= b[0] for a, b in zip(events, events[1:]))

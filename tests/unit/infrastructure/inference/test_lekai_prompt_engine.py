import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
    LekaiPromptEngine,
)


def _count_markers(tokens, marker):
    return tokens.tolist().count(marker)


def test_prompt_engine_uses_time_signature_bar_length(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    engine = LekaiPromptEngine()

    tokens, _bpm, window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    vocab = engine._tokenizer.vocab
    assert window_ticks == 32
    assert _count_markers(tokens, vocab.bar_token_id) == 4
    assert _count_markers(tokens, vocab.beat_marker) == 8


def test_prompt_engine_defaults_to_common_time_bars(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "4")
    engine = LekaiPromptEngine()

    tokens, _bpm, window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    vocab = engine._tokenizer.vocab
    assert window_ticks == 32
    assert _count_markers(tokens, vocab.bar_token_id) == 2
    assert _count_markers(tokens, vocab.beat_marker) == 8


def test_prompt_engine_condition_length_defaults_to_prompt_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 32


def test_prompt_engine_condition_length_can_be_overridden_by_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    monkeypatch.setenv("LEKAI_PROMPT_CONDITION_BEATS", "4")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 16


def test_prompt_engine_batch_selection_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_SELECTION_MODE", raising=False)
    engine = LekaiPromptEngine()

    assert engine._selection_mode() == "single"
    assert engine.runtime_info()["batch_candidate_count"] == 1


def test_prompt_engine_rule_s_uses_frozen_generation_defaults(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s")
    monkeypatch.delenv("LEKAI_PROMPT_TEMPERATURE", raising=False)
    monkeypatch.delenv("LEKAI_PROMPT_TOP_K", raising=False)
    engine = LekaiPromptEngine()

    assert engine._generation_parameters("rule_s") == {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }
    assert engine.runtime_info()["batch_candidate_count"] == 5


def test_prompt_engine_calls_one_batch_generation(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "5")

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate_music_batch(self, **kwargs):
            self.calls.append(kwargs)
            return torch.arange(15, dtype=torch.long).reshape(5, 3)

    engine = LekaiPromptEngine()
    engine._model = FakeModel()
    engine._resolved_device = "cpu"

    output = engine._generate_token_batch(
        torch.tensor([1, 2], dtype=torch.long),
        candidate_count=5,
    )

    assert output.shape == (5, 3)
    assert len(engine._model.calls) == 1
    assert engine._model.calls[0]["batch_size"] == 5
    assert engine._model.calls[0]["temperature"] == 0.8
    assert engine._model.calls[0]["top_k"] == 50

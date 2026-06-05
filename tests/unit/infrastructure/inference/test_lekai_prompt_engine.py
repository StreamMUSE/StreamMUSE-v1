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

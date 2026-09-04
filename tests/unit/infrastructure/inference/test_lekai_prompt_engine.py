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


def test_prompt_engine_explicit_session_bpm_overrides_environment(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "70")
    engine = LekaiPromptEngine()

    tokens, effective_bpm, _window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
        bpm=220,
    )

    assert effective_bpm == 220
    assert tokens.tolist()[2] == engine._tokenizer.encode_bpm(220)


def test_prompt_engine_omitted_session_bpm_preserves_environment_fallback(monkeypatch):
    monkeypatch.setenv("LEKAI_DEFAULT_BPM", "70")
    monkeypatch.setenv("LEKAI_PROMPT_BPM", "80")
    engine = LekaiPromptEngine()

    _tokens, effective_bpm, _window_ticks = engine._build_melody_prompt_tokens(
        melody_events=[],
        prompt_start_tick=0,
        prompt_length_ticks=32,
    )

    assert effective_bpm == 80


def test_prompt_engine_condition_length_defaults_to_prompt_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 32


def test_prompt_engine_condition_length_can_be_overridden_by_beats(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", "2")
    monkeypatch.setenv("LEKAI_PROMPT_CONDITION_BEATS", "4")
    engine = LekaiPromptEngine()

    assert engine._prompt_condition_length_ticks(32) == 16


def test_prompt_engine_defaults_to_stanley_single_candidate_path(monkeypatch):
    monkeypatch.delenv("LEKAI_PROMPT_SELECTION_MODE", raising=False)
    engine = LekaiPromptEngine()

    info = engine.runtime_info()

    assert info["selection_mode"] == "single"
    assert info["batch_candidate_count"] == 1
    assert engine._generation_parameters("single") == {
        "temperature": 1.1,
        "top_k": 0,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }


def test_prompt_engine_exposes_paired_batch_selection_modes(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "batch_first")
    monkeypatch.setenv("LEKAI_PROMPT_BATCH_CANDIDATES", "5")
    engine = LekaiPromptEngine()

    assert engine.runtime_info()["batch_candidate_count"] == 5
    assert engine._generation_parameters("batch_first")["top_k"] == 50

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s")
    assert engine.runtime_info()["selection_mode"] == "rule_s"


def test_prompt_engine_accepts_rule_s_v3_aliases(monkeypatch):
    engine = LekaiPromptEngine()

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule_s_v3")
    assert engine.runtime_info()["selection_mode"] == "rule_s_v3"

    monkeypatch.setenv("LEKAI_PROMPT_SELECTION_MODE", "rule-s-v3")
    assert engine.runtime_info()["selection_mode"] == "rule_s_v3"


def test_prompt_engine_session_seed_overrides_environment_and_clears_diagnostics(monkeypatch):
    monkeypatch.setenv("LEKAI_PROMPT_SEED", "12")
    engine = LekaiPromptEngine()
    engine._last_prompt_token_ids = [1, 2]
    engine._last_generated_token_ids = [1, 2, 3]
    engine._last_new_token_ids = [3]
    engine._last_generated_acc_beats = 8
    engine._last_generation_metadata = {"selection_mode": "rule_s"}

    actual_seed = engine.reset_session(34)

    assert actual_seed == 34
    assert engine.runtime_info()["sample_seed"] == 34
    assert engine.last_generation_log()["prompt_tokens"] == []
    assert engine.last_generation_log()["generated_tokens"] == []
    assert engine.last_generation_log()["generated_acc_beats"] == 0

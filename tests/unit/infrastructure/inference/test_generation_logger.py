import json
from pathlib import Path

from streammuse.infrastructure.inference.generation_logger import GenerationLogger, compare_logs


def _log(logger, *, input_file="example.mid", prompt_tokens=(1, 2, 3), **overrides):
    kwargs = dict(
        input_file=input_file,
        generation_start_tick=16,
        generation_length_frames=4,
        prompt_tokens=list(prompt_tokens),
        temperature=0.8,
        top_k=1,
        top_p=0.95,
        repetition_penalty=1.2,
        melody_events=[{"type": "note_on", "pitch": 60, "tick": 0}],
        accompaniment_events=[{"type": "note_on", "pitch": 48, "tick": 16}],
    )
    kwargs.update(overrides)
    return logger.log_generation(**kwargs)


def test_generation_logger_persists_diagnostics(tmp_path):
    logger = GenerationLogger(str(tmp_path), mode="fake_rt")

    path = logger.log_generation(
        input_file="example.mid",
        generation_start_tick=4,
        generation_length_frames=4,
        prompt_tokens=[1, 2, 3],
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        melody_events=[],
        accompaniment_events=[],
        diagnostics={"beat_diagnostics": [{"target_beat": 1}]},
    )

    payload = json.loads(Path(path).read_text())
    assert payload["diagnostics"] == {
        "beat_diagnostics": [{"target_beat": 1}]
    }


def test_generation_logger_writes_required_fields_and_summary(tmp_path):
    logger = GenerationLogger(str(tmp_path), mode="fake_rt")

    path = _log(logger, prompt_tokens=[1, 4, 5, 173, 255], bpm=120, notes="test generation")

    payload = json.loads(Path(path).read_text())
    for field in (
        "mode",
        "timestamp",
        "input_file",
        "generation_start_tick",
        "prompt_tokens",
        "prompt_token_count",
        "temperature",
        "top_k",
        "melody_events",
        "accompaniment_events",
    ):
        assert field in payload
    assert payload["mode"] == "fake_rt"
    assert payload["prompt_tokens"] == [1, 4, 5, 173, 255]
    assert Path(logger.save_summary()).exists()


def test_compare_logs_matches_identical_prompts_and_reports_mismatches(tmp_path):
    # Log files are named {mode}_gen_{start_tick}.json, so each logger gets its own directory.
    tokens = [1, 4, 5, 173, 255, 255, 10, 20, 30, 170]
    fake_rt = _log(GenerationLogger(str(tmp_path / "fake_rt"), mode="fake_rt"), input_file="a.mid", prompt_tokens=tokens)
    offline = _log(GenerationLogger(str(tmp_path / "offline"), mode="offline"), input_file="b.mid", prompt_tokens=tokens)
    different = _log(
        GenerationLogger(str(tmp_path / "offline_different"), mode="offline"),
        input_file="c.mid",
        prompt_tokens=[1, 4, 5, 999, 255],
        melody_events=[],
        accompaniment_events=[],
    )

    assert compare_logs(fake_rt, offline)["prompt_comparison"]["tokens_match"] is True

    mismatch = compare_logs(fake_rt, different)["prompt_comparison"]
    assert mismatch["tokens_match"] is False
    assert mismatch["mismatches"]

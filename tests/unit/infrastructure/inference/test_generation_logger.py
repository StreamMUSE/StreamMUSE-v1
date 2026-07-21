import json
from pathlib import Path

from streammuse.infrastructure.inference.generation_logger import GenerationLogger


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

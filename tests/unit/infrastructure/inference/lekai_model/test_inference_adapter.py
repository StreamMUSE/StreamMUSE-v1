import numpy as np

from streammuse.infrastructure.inference.lekai_model.inference_adapter import beats_to_pianoroll
from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer


def _build_tokenizer() -> PianoRollTokenizer:
    return PianoRollTokenizer(
        patch_h=1,
        patch_w=4,
        marker_offset=81,
        measures_length=88,
        end_marker_part0=170,
        end_marker_part1=171,
        empty_marker=169,
        img_h=88,
    )


def test_beats_to_pianoroll_empty_list_returns_empty_time_axis():
    tokenizer = _build_tokenizer()

    out = beats_to_pianoroll([], tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 0)


def test_beats_to_pianoroll_handles_empty_inner_list():
    tokenizer = _build_tokenizer()

    out = beats_to_pianoroll([[]], tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 4)
    assert np.count_nonzero(out) == 0


def test_beats_to_pianoroll_bar_token_produces_empty_beat():
    tokenizer = _build_tokenizer()

    out = beats_to_pianoroll([[255]], tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 4)
    assert np.count_nonzero(out) == 0


def test_beats_to_pianoroll_empty_marker_produces_empty_beat():
    tokenizer = _build_tokenizer()

    out = beats_to_pianoroll([[169]], tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 4)
    assert np.count_nonzero(out) == 0


def test_beats_to_pianoroll_roundtrip_real_beat():
    tokenizer = _build_tokenizer()

    beat = np.zeros((2, 88, 4), dtype=np.float32)
    beat[0, 60, 0:4] = 1.0
    beat[1, 60, 0] = 1.0

    tokens = tokenizer.image_to_patch_tokens(beat, strict_mode=True)
    compressed = tokenizer.compress_tokens(tokens, end_marker=171)

    out = beats_to_pianoroll([compressed.tolist()], tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 4)
    assert np.array_equal(out, beat)

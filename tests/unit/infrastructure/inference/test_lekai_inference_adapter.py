import numpy as np

from streammuse.infrastructure.inference.lekai_model.PianoDataset import (
    process_measure_with_beat_interleaving,
)
from streammuse.infrastructure.inference.lekai_model.inference_adapter import (
    beats_to_pianoroll,
)
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


def test_tokenizer_roundtrip_preserves_time_length():
    tokenizer = _build_tokenizer()

    # Two beats (8 timesteps) accompaniment pianoroll.
    pianoroll = np.zeros((2, 88, 8), dtype=np.float32)
    pianoroll[0, 60, 0:4] = 1.0
    pianoroll[1, 60, 0] = 1.0
    pianoroll[0, 64, 4:8] = 1.0
    pianoroll[1, 64, 4] = 1.0

    encoded = tokenizer.encode(pianoroll)
    decoded = tokenizer.decode(encoded)

    assert decoded.shape == pianoroll.shape


def test_beats_to_pianoroll_handles_empty_beats_without_crashing():
    tokenizer = _build_tokenizer()

    # Includes: bar-only beat, sparse beat, and malformed empty beat list.
    beat_tokens = [[255], [250, 171], []]

    pianoroll = beats_to_pianoroll(beat_tokens, tokenizer, timesteps_per_beat=4)

    assert pianoroll.shape == (2, 88, 12)
    assert np.isfinite(pianoroll).all()


def test_process_and_decode_beats_preserve_total_length():
    tokenizer = _build_tokenizer()

    # Build one measure with 3 beats of content to test interleaving path.
    measure = np.zeros((4, 88, 12), dtype=np.float32)
    measure[2, 55, 0:4] = 1.0
    measure[3, 55, 0] = 1.0
    measure[2, 57, 4:8] = 1.0
    measure[3, 57, 4] = 1.0
    measure[2, 59, 8:12] = 1.0
    measure[3, 59, 8] = 1.0

    _, part1_beats = process_measure_with_beat_interleaving(
        measure,
        tokenizer,
        timesteps_per_beat=4,
    )

    part1_beat_tokens = [beat.tolist() for beat in part1_beats]
    reconstructed = beats_to_pianoroll(part1_beat_tokens, tokenizer, timesteps_per_beat=4)

    assert reconstructed.shape[2] == len(part1_beat_tokens) * 4

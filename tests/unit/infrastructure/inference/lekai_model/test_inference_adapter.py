import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_model.inference_adapter import PianoLLaMAAdapter, beats_to_pianoroll
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


def test_beats_to_pianoroll_handles_mixed_empty_beats_without_crashing():
    tokenizer = _build_tokenizer()

    # Includes: bar-only beat, sparse beat, and malformed empty beat list.
    beat_tokens = [[255], [250, 171], []]

    out = beats_to_pianoroll(beat_tokens, tokenizer, timesteps_per_beat=4)

    assert out.shape == (2, 88, 12)
    assert np.isfinite(out).all()


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


def test_generate_from_beats_allows_empty_part0_prompt(monkeypatch):
    class _FakeModel:
        def eval(self):
            return None

        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            _ = past_key_values, use_cache

            class _Output:
                def __init__(self, seq_len: int):
                    self.logits = torch.zeros((1, seq_len, 300), dtype=torch.float32)
                    self.past_key_values = None

            return _Output(input_ids.shape[1])

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.generation_utils.sample_token",
        lambda *args, **kwargs: torch.tensor([[169]], dtype=torch.long),
    )

    adapter = PianoLLaMAAdapter(
        model=_FakeModel(),
        tokenizer=_build_tokenizer(),
        device="cpu",
    )

    beats = adapter.generate_from_beats(
        part0_beats=[],
        num_beats_to_generate=2,
    )

    assert len(beats) == 2
    assert all(len(beat) >= 1 for beat in beats)

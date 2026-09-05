import pytest
import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.model import (
    PianoLLaMA,
)


class _FakeCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["input_ids"].clone()


def _prompt_model() -> PianoLLaMA:
    model = PianoLLaMA.__new__(PianoLLaMA)
    torch.nn.Module.__init__(model)
    model.model = _FakeCausalModel()
    model.eos_token_id = 3
    model.pad_token_id = 0
    return model


def test_generate_music_batch_repeats_condition_in_one_generate_call():
    model = _prompt_model()

    condition = torch.tensor([4, 8, 9], dtype=torch.long)
    output = model.generate_music_batch(
        condition,
        batch_size=5,
        device="cpu",
        max_length=16,
    )

    assert len(model.model.calls) == 1
    input_ids = model.model.calls[0]["input_ids"]
    assert input_ids.shape == (5, 3)
    assert all(torch.equal(row, condition) for row in input_ids)
    assert torch.equal(output, input_ids)


@pytest.mark.parametrize("batch_size", [None, 3])
def test_prompt_generation_temperature_zero_uses_true_greedy(batch_size):
    model = _prompt_model()
    condition = torch.tensor([4, 8, 9], dtype=torch.long)

    if batch_size is None:
        model.generate_music(
            condition,
            device="cpu",
            temperature=0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
        )
    else:
        model.generate_music_batch(
            condition,
            batch_size=batch_size,
            device="cpu",
            temperature=0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    kwargs = model.model.calls[0]
    assert kwargs["do_sample"] is False
    assert kwargs["repetition_penalty"] == 1.1
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.parametrize("batch_size", [None, 3])
def test_prompt_generation_nonzero_temperature_preserves_sampling(batch_size):
    model = _prompt_model()
    condition = torch.tensor([4, 8, 9], dtype=torch.long)
    expected = {
        "temperature": 1.1,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    }

    if batch_size is None:
        model.generate_music(condition, device="cpu", **expected)
    else:
        model.generate_music_batch(
            condition,
            batch_size=batch_size,
            device="cpu",
            **expected,
        )

    kwargs = model.model.calls[0]
    assert kwargs["do_sample"] is True
    for name, value in expected.items():
        assert kwargs[name] == value


def test_generate_music_batch_rejects_nonpositive_batch_size():
    model = PianoLLaMA.__new__(PianoLLaMA)
    torch.nn.Module.__init__(model)

    try:
        model.generate_music_batch(torch.tensor([1]), batch_size=0, device="cpu")
    except ValueError as exc:
        assert "batch_size must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

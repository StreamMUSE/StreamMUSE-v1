import torch

from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.model import (
    PianoLLaMA,
)


def test_generate_music_batch_repeats_condition_in_one_generate_call():
    class FakeCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["input_ids"].clone()

    model = PianoLLaMA.__new__(PianoLLaMA)
    torch.nn.Module.__init__(model)
    model.model = FakeCausalModel()
    model.eos_token_id = 3
    model.pad_token_id = 0

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


def test_generate_music_batch_rejects_nonpositive_batch_size():
    model = PianoLLaMA.__new__(PianoLLaMA)
    torch.nn.Module.__init__(model)

    try:
        model.generate_music_batch(torch.tensor([1]), batch_size=0, device="cpu")
    except ValueError as exc:
        assert "batch_size must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

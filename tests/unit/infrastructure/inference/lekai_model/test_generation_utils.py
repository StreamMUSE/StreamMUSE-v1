from __future__ import annotations

import torch

from streammuse.infrastructure.inference.lekai_model.generation_utils import sample_token


def _draw_trace(seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.tensor([[0.1, 0.3, 0.2, 0.4]], dtype=torch.float32)
    return [
        int(
            sample_token(
                logits.clone(),
                temperature=0.8,
                top_k=4,
                top_p=1.0,
                generator=generator,
            ).item()
        )
        for _ in range(32)
    ]


def test_explicit_generator_reproduces_full_sampling_trace():
    assert _draw_trace(20260710) == _draw_trace(20260710)


def test_explicit_generator_does_not_consume_process_global_rng():
    torch.manual_seed(1234)
    expected_next_global_draw = torch.rand(8)

    torch.manual_seed(1234)
    sample_token(
        torch.zeros((1, 5), dtype=torch.float32),
        temperature=1.0,
        top_k=5,
        top_p=1.0,
        generator=torch.Generator(device="cpu").manual_seed(99),
    )
    actual_next_global_draw = torch.rand(8)

    assert torch.equal(actual_next_global_draw, expected_next_global_draw)


def test_top_p_filter_handles_each_batch_row():
    result = sample_token(
        torch.tensor([[10.0, 0.0, -1.0], [-1.0, 0.0, 10.0]]),
        temperature=1.0,
        top_k=3,
        top_p=0.5,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )

    assert result.tolist() == [[0], [2]]


def test_greedy_sampling_does_not_advance_explicit_generator():
    generator = torch.Generator(device="cpu").manual_seed(42)
    before = generator.get_state().clone()

    result = sample_token(
        torch.tensor([[0.0, 2.0, 1.0]]),
        temperature=0.0,
        generator=generator,
    )

    assert result.tolist() == [[1]]
    assert torch.equal(generator.get_state(), before)

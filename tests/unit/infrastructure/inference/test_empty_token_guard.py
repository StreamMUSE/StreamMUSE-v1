from types import SimpleNamespace

import pytest
import torch

from streammuse.infrastructure.inference.empty_token_guard import EmptyTokenGuard
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend


def arm(guard):
    for tick in range(0, 16, 4):
        assert not guard.before_beat(tick)
        trace = guard.observe(tick, [169, 170])
        assert trace["triggered_for_next_beat"] == (tick == 12)


def test_exactly_eight_following_beats_then_can_retrigger():
    guard = EmptyTokenGuard()
    arm(guard)
    for tick in range(16, 48, 4):
        assert guard.before_beat(tick)
        trace = guard.observe(tick, [170])
        assert trace["remaining_after"] == (44 - tick) // 4
    assert not guard.before_beat(48)
    for tick in range(48, 64, 4):
        guard.observe(tick, [169, 170])
    assert guard.before_beat(64)


@pytest.mark.parametrize("tokens", [[170], [81, 40, 170], [81, 67, 170], [169]])
def test_only_explicit_empty_beats_count(tokens):
    guard = EmptyTokenGuard()
    for tick in range(0, 12, 4):
        guard.observe(tick, [169, 170])
    trace = guard.observe(12, tokens)
    assert not trace["explicit_empty_beat"]
    assert guard.consecutive_empty == 0
    assert not guard.before_beat(16)


@pytest.mark.parametrize("tick", [12, 20, 0])
def test_discontinuous_or_replayed_tick_resets_guard(tick):
    guard = EmptyTokenGuard()
    arm(guard)
    assert not guard.before_beat(tick)
    assert guard.remaining_beats == guard.consecutive_empty == 0


def test_guard_checks_raw_token_invariant():
    guard = EmptyTokenGuard()
    arm(guard)
    with pytest.raises(RuntimeError, match="EMPTY token escaped"):
        guard.observe(16, [169, 170])


@pytest.mark.parametrize("method", ["reset_session", "clear_history", "inject_history"])
def test_backend_lifecycle_resets_guard(method):
    backend = LekaiHttpBackend()
    arm(backend._empty_token_guard)
    kwargs = dict(melody_events=[], accompaniment_events=[], injection_length_ticks=32)
    if method == "reset_session":
        kwargs = {"seed": 123}
    elif method == "clear_history":
        kwargs = {}
    getattr(backend, method)(**kwargs)
    assert not backend._empty_token_guard.before_beat(16)


@pytest.mark.parametrize("key", [None, set(), {0, 2, 4, 5, 7, 9, 11}])
@pytest.mark.parametrize("temperature,top_k,top_p", [(0.0, 0, 1.0), (1.1, 50, 0.95)])
def test_mask_blocks_empty_but_does_not_block_acc_end(key, temperature, top_k, top_p):
    backend = LekaiHttpBackend()

    def model(input_ids, **kwargs):
        logits = torch.full((1, input_ids.shape[1], 300), -float("inf"))
        logits[:, -1, 169] = 100.0
        logits[:, -1, 170] = 0.0 if input_ids[0, -1] != 169 else 200.0
        return SimpleNamespace(logits=logits, past_key_values=None)

    backend._model_adapter = SimpleNamespace(model=model, device="cpu", use_cache=False)
    kwargs = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                  repetition_penalty=1.0, tonal_pitch_classes=key)
    prompt = torch.tensor([257, 259, 264, 173])
    assert backend._generate_part1_tokens_from_prompt(prompt, **kwargs) == [169, 170]
    assert backend._generate_part1_tokens_from_prompt(
        prompt, block_empty_token=True, **kwargs,
    ) == [170]


@pytest.mark.parametrize("value", [float("nan"), -float("inf")])
@pytest.mark.parametrize("key", [None, set()])
def test_invalid_logits_never_restore_empty(value, key):
    backend = LekaiHttpBackend()
    backend._model_adapter = SimpleNamespace(device="cpu", use_cache=False, model=lambda **kw:
        SimpleNamespace(logits=torch.full((1, 1, 300), value), past_key_values=None))
    assert backend._generate_part1_tokens_from_prompt(
        torch.tensor([257, 259, 264, 173]), temperature=1.1, top_k=50, top_p=0.95,
        repetition_penalty=1.0, tonal_pitch_classes=key, block_empty_token=True,
    ) == [170]


def test_guard_does_not_force_an_onset_or_modify_other_finite_logits(monkeypatch):
    backend = LekaiHttpBackend()
    sequence = iter([84, 40, 170])  # C2 PIT followed by a sustain-only PAT.
    expected = torch.arange(300, dtype=torch.float32).unsqueeze(0)
    backend._model_adapter = SimpleNamespace(device="cpu", use_cache=False, model=lambda **kw:
        SimpleNamespace(logits=expected.unsqueeze(0).clone(), past_key_values=None))

    def sample(logits, **kwargs):
        assert torch.isneginf(logits[0, 169])
        other = torch.arange(300) != 169
        assert torch.equal(logits[:, other], expected[:, other])
        return torch.tensor([[next(sequence)]])

    monkeypatch.setattr(
        "streammuse.infrastructure.inference.lekai_model.generation_utils.sample_token", sample,
    )
    assert backend._generate_part1_tokens_from_prompt(
        torch.tensor([257, 259, 264, 173]), temperature=1.1, top_k=50, top_p=0.95,
        repetition_penalty=1.0, block_empty_token=True,
    ) == [84, 40, 170]

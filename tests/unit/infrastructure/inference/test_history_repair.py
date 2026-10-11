"""CPU mocks only: live-state ownership and fixed-bar transaction contract."""

from concurrent.futures import Future
from copy import deepcopy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend


def on(tick, pitch=60):
    return {"type": "note_on", "tick": tick, "pitch": pitch, "velocity": 90}


def off(tick, pitch=60):
    return {"type": "note_off", "tick": tick, "pitch": pitch, "velocity": 0}


def test_immutable_prefix_duplicate_release_is_preserved():
    from streammuse.infrastructure.inference.history_repair import prepare_history_repair
    history = [on(28), off(32), off(32), on(64), off(68)]
    replacement = [on(64, 62), off(68, 62)]
    assert prepare_history_repair(history, replacement, 64, 80, set(), 84) == history[:3] + replacement


@pytest.mark.parametrize('history,candidate', [
    ([off(32)], []),
    ([on(28), off(32), off(33)], []),
    ([on(32), off(32)], []),
    ([on(64), off(68), off(68)], []),
    ([], [on(64), off(68), off(68)]),
])
def test_duplicate_release_exception_remains_narrow(history, candidate):
    from streammuse.infrastructure.inference.history_repair import prepare_history_repair, HistoryRepairError
    with pytest.raises(HistoryRepairError):
        prepare_history_repair(history, candidate, 64, 80, set(), 84)


def generate(backend, start=16, length=16, melody=None):
    return backend.generate(
        melody_events=melody or [], generation_start_tick=start,
        generation_length_frames=length, generation_interval_ticks=4,
        prompt_length_ticks=64, inference_mode="sliding_window", model_name="lekai",
        checkpoint_path=None, session_id=backend._session_id,
        session_epoch=backend._session_epoch, request_id=f"request-{start}",
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setenv("LEKAI_RT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD", "0")
    monkeypatch.setenv("LEKAI_CONTINUATION_TONAL_CONSTRAINT", "1")
    monkeypatch.setenv("LEKAI_TIME_SIGNATURE_INDEX", "4")
    monkeypatch.setenv("LEKAI_MEASURE_BEATS", "4")
    b = LekaiHttpBackend(history_repair_enabled=True)
    b.reset_session(17)
    monkeypatch.setattr(b, "_has_real_model", lambda: True)

    def model(**kwargs):
        start = kwargs["generation_start_tick"]
        end = start + kwargs["generation_length_frames"]
        beats = list(range(start // 4, end // 4))
        b._accompaniment_token_history.update({beat: [120, 170] for beat in beats})
        b._current_generation_trace = {
            "part0_roll": np.zeros((2, 88, end - 4), dtype=np.uint8),
            "part0_roll_end_tick": end - 4,
            "raw_tokens": [120, 170] * len(beats),
            "token_decode_beats": [
                {"target_beat": beat, "start_tick": beat * 4,
                 "raw_tokens": [120, 170], "boundary_tokens": []} for beat in beats
            ],
        }
        return [on(start), off(end - 1)]

    monkeypatch.setattr(b, "_generate_with_model", model)
    generate(b, 0, 16, [on(0, 72), off(3, 72)])
    generate(b, melody=[on(16, 72), off(20, 72), on(40, 75)])
    generate(b, 32, 4)
    b._accompaniment_bar_token_history = {0: [255], 4: [255], 8: [255]}
    yield b
    b._boundary_executor.shutdown(wait=True, cancel_futures=True)


def ticket(b):
    result = b.begin_history_repair(16, 32)
    assert result["status"] == "ready", result
    return result["ticket"]


def candidate(events=None):
    return {"events": [on(18, 64), off(29, 64)] if events is None else events,
            "token_decode_beats": [], "provenance": {"worker": "mock"},
            "guard_semantics": "isolated_reset_state"}


def test_identical_configuration_does_not_invalidate_ticket(backend):
    t = ticket(backend)
    backend.configure(deepcopy(backend._runtime_config))
    backend.set_session_generation_config()
    assert backend.apply_history_repair(t, candidate())["status"] == "applied"


def test_apply_preserves_live_state_and_original_audit(backend, monkeypatch):
    b = backend
    t = ticket(b)
    before = b._history_repair_state()
    rng = b._sample_generator.get_state().clone()
    metadata = b.generation_metadata_snapshot()
    trace = deepcopy(b._current_generation_trace)
    played = b._accompaniment_history
    played_before = deepcopy(played)
    guard, tonal, generator = b._empty_token_guard, b._tonal_tracker, b._sample_generator
    for method in ("inject_history", "reset_session", "_generate_part1_tokens_from_prompt",
                   "_resolve_pending_boundary_generations"):
        monkeypatch.setattr(b, method, lambda *a, **k: pytest.fail("forbidden live operation"))
    monkeypatch.setattr(guard, "reset", lambda: pytest.fail("guard reset"))
    monkeypatch.setattr(tonal, "reset", lambda: pytest.fail("tonal reset"))
    # Test-only monkeypatch attributes are not a state mutation for this assertion.
    b._history_repair_ticket = (deepcopy(t), b._history_repair_state())
    c = candidate()
    result = b.apply_history_repair(t, c)
    assert result == {"status": "applied", "reason": "applied",
                      "history_revision": t["history_revision"] + 1,
                      "invalidated_token_beats": [4, 5, 6, 7]}
    assert b._accompaniment_history == played_before[:2] + c["events"] + played_before[4:]
    assert played == played_before
    assert b._melody_history == before["melody"]
    assert b._input_digest_history == before["input"]
    assert b._sample_generator is generator and torch.equal(rng, generator.get_state())
    assert b._empty_token_guard is guard and b._tonal_tracker is tonal
    assert b._active_pitches == set(before["active"])
    assert b.generation_metadata_snapshot() == metadata
    assert np.array_equal(b._current_generation_trace["part0_roll"], trace["part0_roll"])
    assert {k: v for k, v in b._current_generation_trace.items() if k != "part0_roll"} == {
        k: v for k, v in trace.items() if k != "part0_roll"}
    assert b._accompaniment_token_history == {k: v for k, v in before["tokens"].items() if k not in range(4, 8)}
    assert b._accompaniment_bar_token_history == {0: [255], 8: [255]}
    audit = b.history_repair_audit_snapshot()
    assert audit[0]["ticket"]["accompaniment_events"] == played_before
    assert audit[0]["original_bar_tokens"] == {k: [120, 170] for k in range(4, 8)}
    c["events"].clear()
    audit[0]["candidate"]["events"].clear()
    assert len(b.history_repair_audit_snapshot()[0]["candidate"]["events"]) == 2
    assert b.apply_history_repair(t, candidate())["reason"] == "stale_or_invalid_ticket"


def test_ticket_detached_and_melody_cutoff_not_completeness(backend):
    t = ticket(backend)
    assert t["observed_until"] == 32
    assert t["melody_observation"] == "received_events_only"
    assert all(e["tick"] < 32 for e in t["melody_events"])
    assert any(e["tick"] == 40 for e in backend._melody_history)
    t["accompaniment_events"].clear()
    assert backend._accompaniment_history
    assert backend.apply_history_repair(t, candidate())["status"] == "rejected"


@pytest.mark.parametrize("mutation", ["generate", "failed_generate", "reset", "inject", "clear", "sampling"])
def test_mutation_invalidates_ticket(backend, monkeypatch, mutation):
    t = ticket(backend)
    if mutation == "generate":
        generate(backend, 36, 4)
    elif mutation == "failed_generate":
        def fail(**kwargs):
            raise RuntimeError("model failure")
        monkeypatch.setattr(backend, "_generate_with_model", fail)
        with pytest.raises(RuntimeError):
            generate(backend, 36, 4)
    elif mutation == "reset":
        backend.reset_session(17)
    elif mutation == "inject":
        backend.inject_history([], [], 32)
    elif mutation == "clear":
        backend.clear_history()
    else:
        backend.set_session_generation_config(temperature=0.2)
    assert backend.apply_history_repair(t, candidate())["reason"] == "stale_or_invalid_ticket"


@pytest.mark.parametrize("field", ["_melody_history", "_accompaniment_history", "_input_digest_history"])
def test_snapshot_cas_detects_aliased_mutation(backend, field):
    t = ticket(backend)
    getattr(backend, field)[0]["velocity"] = 70
    before = deepcopy(backend._accompaniment_history)
    assert backend.apply_history_repair(t, candidate())["reason"] == "snapshot_changed"
    assert backend._accompaniment_history == before


@pytest.mark.parametrize("events,reason", [
    ([on(18)], "active_tail_mismatch"),
    ([off(18)], "unpaired_or_zero_length_release"),
    ([on(18), on(19)], "invalid_or_repeated_onset"),
    ([on(18), off(18)], "unpaired_or_zero_length_release"),
    ([on(15), off(20)], "invalid_event_range_or_order"),
    ([on(18), off(32)], "invalid_event_range_or_order"),
    ([on(20), off(19)], "invalid_event_range_or_order"),
    ([on(18, True)], "invalid_event_range_or_order"),
    ([on(18.0)], "invalid_event_range_or_order"),
    ([dict(on(18), velocity=0)], "invalid_or_repeated_onset"),
    ([dict(on(18), channel=0)], "unsupported_event_fields"),
])
def test_unsafe_candidate_is_atomic_rejection(backend, events, reason):
    t = ticket(backend)
    state = backend._history_repair_state()
    rng = backend._sample_generator.get_state().clone()
    assert backend.apply_history_repair(t, candidate(events))["reason"] == reason
    assert backend._history_repair_state() == state
    assert torch.equal(backend._sample_generator.get_state(), rng)
    assert backend.history_repair_audit_snapshot() == []


def test_matching_open_tail_only(backend):
    backend._accompaniment_history = [on(0), off(15), on(16), off(34)]
    t = ticket(backend)
    assert t["active_pitches_at_end"] == [60]
    assert backend.apply_history_repair(t, candidate([on(18, 64)]))["reason"] == "active_tail_mismatch"
    assert backend.apply_history_repair(t, candidate([on(18)]))["status"] == "applied"
    assert backend._active_pitches == set()
    assert backend._active_pitches_before_tick(backend._accompaniment_history, 32) == {60}
    assert backend._accompaniment_history[-1] == off(34)


def test_cross_start_sustain_uses_immutable_prefix_carry(backend):
    backend._accompaniment_history = [on(10), off(20)]
    t = ticket(backend)
    assert t["initial_active_pitches"] == [60]
    assert backend.apply_history_repair(t, candidate([]))["reason"] == "active_tail_mismatch"
    assert backend.apply_history_repair(t, candidate([off(17), on(18, 64), off(29, 64)]))["status"] == "applied"
    assert backend._accompaniment_history[0] == on(10)
    assert backend._active_pitches == set()


def test_cross_start_carry_cannot_invent_release(backend):
    backend._accompaniment_history = [on(10), off(20)]
    t = ticket(backend)
    result = backend.apply_history_repair(t, candidate([off(16, 64), off(20)]))
    assert result["reason"] == "unpaired_or_zero_length_release"


@pytest.mark.parametrize("start,end", [(32, 48), (16, 48), (17, 33), (True, 17), (16.0, 32)])
def test_invalid_or_nonlatest_bar(backend, start, end):
    assert backend.begin_history_repair(start, end)["status"] == "rejected"


def test_enabled_guard_requires_bar_start_snapshot(backend, monkeypatch):
    monkeypatch.setenv("LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD", "1")
    before = deepcopy(vars(backend._empty_token_guard))
    assert backend.begin_history_repair(16, 32)["reason"] == "missing_bar_start_constraint_snapshot"
    assert vars(backend._empty_token_guard) == before


def test_guard_chronology_is_preserved_after_repair(backend, monkeypatch):
    monkeypatch.setenv("LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD", "1")
    backend._history_repair_constraint_snapshots[16] = {
        "empty_guard": {"consecutive_empty": 0, "remaining_beats": 6, "last_tick": 12},
        "tonal_tracker": deepcopy(vars(backend._tonal_tracker)),
    }
    backend._empty_token_guard.remaining_beats = 2
    backend._empty_token_guard.last_tick = 32
    before = backend._empty_token_guard.snapshot()
    t = ticket(backend)
    assert t["constraint_snapshot"]["empty_guard"]["remaining_beats"] == 6
    assert backend.apply_history_repair(t, candidate())["status"] == "applied"
    assert backend._empty_token_guard.snapshot() == before
    assert backend._empty_token_guard.before_beat(36) is True
    after = backend._empty_token_guard.observe(36, [120, 170])
    assert after["remaining_after"] == 1


def test_trimmed_or_incomplete_history_rejected(backend):
    backend._history_repair_retained_from = 17
    assert backend.begin_history_repair(16, 32)["reason"] == "history_trimmed"
    backend._history_repair_retained_from = 0
    backend._history_repair_generated_beats.remove(4)
    assert backend.begin_history_repair(16, 32)["reason"] == "unsupported_generation_history"


def test_pending_future_never_drained(backend):
    t = ticket(backend)
    future = Future()
    backend._pending_boundary_generations[8] = future
    try:
        assert backend.begin_history_repair(16, 32)["reason"] == "pending_boundary_generation"
        assert backend.apply_history_repair(t, candidate())["reason"] == "pending_boundary_generation"
        assert not future.done()
    finally:
        backend._pending_boundary_generations.clear()


@pytest.mark.parametrize("lock_name", ["_session_gate", "_model_generation_lock"])
def test_busy_is_nonblocking(backend, lock_name):
    t = ticket(backend)
    locked, release = threading.Event(), threading.Event()
    def hold():
        with getattr(backend, lock_name):
            locked.set()
            release.wait(5)
    thread = threading.Thread(target=hold)
    thread.start()
    assert locked.wait(1)
    try:
        assert backend.begin_history_repair(16, 32)["reason"] == "backend_busy"
        assert backend.apply_history_repair(t, candidate())["reason"] == "backend_busy"
        assert not release.is_set()
    finally:
        release.set()
        thread.join(2)


def test_default_off_and_explicit_override(monkeypatch, tmp_path):
    monkeypatch.delenv("LEKAI_HISTORY_REPAIR_ENABLED", raising=False)
    monkeypatch.setenv("LEKAI_RT_LOG_DIR", str(tmp_path))
    b = LekaiHttpBackend()
    assert b.begin_history_repair(0, 16)["reason"] == "disabled"
    assert b.apply_history_repair({}, candidate())["reason"] == "disabled"
    monkeypatch.setenv("LEKAI_HISTORY_REPAIR_ENABLED", "1")
    assert LekaiHttpBackend()._history_repair_enabled
    assert not LekaiHttpBackend(history_repair_enabled=False)._history_repair_enabled
    with pytest.raises(ValueError):
        LekaiHttpBackend(history_repair_enabled="true")


def test_actual_frontier_and_melody_fence_required(backend):
    backend._history_repair_completed_until = 32
    assert backend.begin_history_repair(16, 32)["reason"] == "bar_not_interior_to_completed_history"
    backend._history_repair_completed_until = 36
    for fence in (None, 28):
        backend._history_repair_melody_fence = fence
        assert backend.begin_history_repair(16, 32)["reason"] == "melody_fence_unavailable_or_insufficient"


def test_generation_config_snapshot(backend):
    config = ticket(backend)["generation_config"]
    assert config["tonal_constraint_enabled"] is True
    assert config["empty_token_guard_enabled"] is False
    assert config["time_signature_index"] == 4
    assert all(key in config for key in ("checkpoint_path", "bpm", "context_beats",
                                        "temperature", "top_k", "top_p", "repetition_penalty"))


def test_suffix_release_exactly_at_boundary_preserved(backend):
    backend._accompaniment_history = [on(0), off(15), on(16), off(32)]
    t = ticket(backend)
    assert backend.apply_history_repair(t, candidate([on(20)]))["status"] == "applied"
    assert backend._accompaniment_history[-1] == off(32)


def test_next_standard_prompt_reencodes_repaired_events(backend, monkeypatch):
    from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import PianoMusicTokenizer
    from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter

    b = backend
    b._tokenizer = PianoMusicTokenizer()
    b._converter = MidiConverter(ticks_per_beat=4)
    b._model_adapter = SimpleNamespace(model=object(), tokenizer=b._tokenizer)
    original_tokens, _ = b._encode_beat_tokens(b._accompaniment_history, 16, set(), 170)
    assert b.apply_history_repair(ticket(b), candidate())["status"] == "applied"
    repaired_history = deepcopy(b._accompaniment_history)
    calls = []
    encode = b._encode_beat_tokens

    def record(**kwargs):
        result = encode(**kwargs)
        calls.append((kwargs, result[0].tolist()))
        return result

    monkeypatch.setattr(b, "_encode_beat_tokens", record)
    monkeypatch.setattr(b, "_generate_with_model", LekaiHttpBackend._generate_with_model.__get__(b))
    monkeypatch.setattr(b, "_generate_part1_tokens_from_prompt", lambda *a, **k: [169, 170])
    monkeypatch.setattr(b._logger, "log_generation", lambda **k: None)
    generate(b, 36, 4)
    repaired_tokens = next(tokens for args, tokens in calls
                           if args["beat_start_tick"] == 16 and args["end_marker"] == 170)
    assert repaired_tokens != original_tokens.tolist()
    assert all(args["events"][:len(repaired_history)] == repaired_history
               for args, _ in calls if args["end_marker"] == 170)


def test_new_ticket_supersedes_old(backend):
    old = ticket(backend)
    current = ticket(backend)
    assert backend.apply_history_repair(old, candidate())["reason"] == "stale_or_invalid_ticket"
    assert backend.apply_history_repair(current, candidate())["status"] == "applied"


@pytest.mark.parametrize("value", [{}, {"events": [], "token_decode_beats": None},
                                   {"events": [], "token_decode_beats": [], "unknown": 1},
                                   {"events": [], "token_decode_beats": [float("nan")]}])
def test_invalid_envelope_rejected_without_mutation(backend, value):
    t = ticket(backend)
    state = backend._history_repair_state()
    assert backend.apply_history_repair(t, value)["status"] == "rejected"
    assert backend._history_repair_state() == state

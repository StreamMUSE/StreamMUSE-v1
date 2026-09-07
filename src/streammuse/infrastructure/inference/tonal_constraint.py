"""Cumulative causal key estimation, updated every two bars, and sampling mask."""
from __future__ import annotations

import numpy as np
import torch

MAX_TONAL_NOTE_TICKS = 16  # Four quarter-note beats; evidence only, not playback.


def infer_key_with_nan_fallback(evidence: np.ndarray) -> dict:
    """Use the original best-scoring key, except with fewer than three PCs."""
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import infer_tonal_key

    observed_count = int(np.count_nonzero(evidence))
    key = infer_tonal_key(evidence) if observed_count >= 3 else None
    return {
        "key": key,
        "key_state": "determined" if key is not None else "NaN",
        "constraint_active": key is not None,
        "reason": ("estimated_from_cumulative_melody" if key is not None else
                   "insufficient_evidence_unconstrained"),
        "observed_pitch_class_count": observed_count,
    }


def melody_evidence(events, start_tick: int, end_tick: int) -> np.ndarray:
    """Duration in the causal window, including notes held into the window.

    Ignore all events at/after end_tick, including future NOTE_OFF information.
    Each NOTE_ON contributes at most 16 ticks over its entire lifetime, even
    across later estimates. A retrigger starts a new independently capped note.
    """
    evidence = np.zeros(12, dtype=np.float64)
    active: dict[int, int] = {}
    cursor = start_tick

    def accumulate(until_tick):
        for sounding, onset in active.items():
            evidence[sounding % 12] += max(
                0, min(until_tick, onset + MAX_TONAL_NOTE_TICKS) - max(cursor, onset),
            )
    # Stable tick-only sorting preserves observed order at ties. Reordering
    # NOTE_OFF before NOTE_ON turns a quantized zero-length tap into a ghost
    # sustain for the rest of the session.
    for event in sorted(events, key=lambda e: int(e.get("tick", 0))):
        tick = int(event.get("tick", 0))
        if tick >= end_tick:
            break
        pitch = int(event.get("pitch", -1))
        kind = event.get("type")
        if not 21 <= pitch <= 108 or kind not in {"note_on", "note_off"}:
            continue
        if tick > cursor:
            accumulate(tick)
            cursor = tick
        velocity = event.get("velocity")
        if kind == "note_on" and (velocity is None or int(velocity) > 0):
            active[pitch] = tick
        else:
            active.pop(pitch, None)
    accumulate(end_tick)
    return evidence


class TwoBarKeyTracker:
    """Estimate cumulatively on the last beat; apply at the next boundary.

    The first continuation decision is bootstrapped from available Prompt
    melody. All received melody since tick 0 contributes, without decay or
    a rolling cutoff. A final beat's unseen melody is never used.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._boundary = None
        self._key = None
        self._decision = None
        self._pending = None

    def update(self, events, generation_tick: int, beats_per_bar: int) -> dict:
        span = 2 * int(beats_per_bar) * 4
        if span <= 0:
            raise ValueError("beats_per_bar must be positive")
        boundary = int(generation_tick) // span * span
        if self._boundary != (boundary, span):
            if self._pending is not None and self._pending[0] == (boundary, span):
                self._decision = self._pending[1]
                self._key = self._decision["key"]
                self._pending = None
            else:
                # Bootstrap, or a caller that skipped the pre-estimation beat.
                self._decision = self._estimate(events, boundary, boundary, span)
                self._key = self._decision["key"]
            self._boundary = (boundary, span)
        if int(generation_tick) == boundary + span - 4:
            next_boundary = boundary + span
            if self._pending is None or self._pending[0] != (next_boundary, span):
                self._pending = ((next_boundary, span), self._estimate(
                    events, next_boundary, int(generation_tick), span,
                ))
        return dict(self._decision)

    def _estimate(self, events, boundary: int, observed_end: int, span: int) -> dict:
        evidence = melody_evidence(events, 0, observed_end)
        assessment = infer_key_with_nan_fallback(evidence)
        key = assessment["key"]
        return {
            **assessment,
            "boundary_tick": boundary,
            "evidence_scope": "cumulative_from_session_start",
            "max_note_duration_ticks": MAX_TONAL_NOTE_TICKS,
            "window_start_tick": 0,
            "window_end_tick": observed_end,
            "estimated_at_tick": observed_end,
            "update_interval_ticks": span,
            "pitch_class_duration_evidence": evidence.tolist(),
            "key": dict(key) if key else None,
            # None means no pitch constraint; [] would forbid every pitch.
            "allowed_pitch_classes": list(key["in_key_pitch_classes"]) if key else None,
        }


class TonalBeatMask:
    """Constrain PIT/PAT pairs, not fixed token IDs, for one accompaniment beat.

    PIT=81+relative pitch index, PAT=1..80, EMPTY=169, ACC_END=170.
    Disallow illegal/duplicate pitch positions and preserve a legal empty exit.
    Both onset and sustain at an out-of-key pitch are blocked.
    """

    def __init__(self, allowed_pitch_classes):
        self.allowed = frozenset(int(pc) for pc in allowed_pitch_classes)
        if not self.allowed <= set(range(12)):
            raise ValueError("pitch classes must be in 0..11")
        self.position = 0
        self.has_pitch = False
        self.state = "pitch"

    def legal_ids(self, step: int) -> list[int]:
        if self.state == "pattern":
            return list(range(1, 81))
        if self.state == "end" or step >= 98:
            return [170]
        pitches = [81 + delta for delta in range(1 if self.has_pitch else 0, 88 - self.position)
                   if (21 + self.position + delta) % 12 in self.allowed]
        return pitches + ([170] if self.has_pitch else [169, 170])

    def apply(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        legal = self.legal_ids(step)
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[legal] = False
        result = logits.clone().masked_fill(mask, -float("inf"))
        result = result.masked_fill(~torch.isfinite(result), -float("inf"))
        # Never relax the key constraint, even if all legal logits are invalid.
        fallback = 67 if self.state == "pattern" else (169 if 169 in legal else 170)
        invalid = ~torch.isfinite(result).any(dim=-1)
        # Stay on device: avoid a CPU/GPU synchronization just to test `any`.
        result[:, fallback] = torch.where(
            invalid, torch.zeros_like(result[:, fallback]), result[:, fallback],
        )
        return result

    def accept(self, token: int):
        if self.state == "pattern":
            if not 1 <= token <= 80:
                raise ValueError("Expected PAT token")
            self.state = "pitch"
        elif 81 <= token <= 168:
            self.position += token - 81
            self.has_pitch = True
            if self.position >= 88 or (21 + self.position) % 12 not in self.allowed:
                raise ValueError("Out-of-key PIT token escaped sampling mask")
            self.state = "pattern"
        elif token in {169, 170}:
            self.state = "end"
        else:
            raise ValueError("Invalid accompaniment token")

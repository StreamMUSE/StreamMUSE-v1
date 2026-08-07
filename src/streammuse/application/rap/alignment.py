"""Deterministic scheduling of estimated lyric syllables onto beat slots."""

from __future__ import annotations

from collections.abc import Sequence

from streammuse.domain.rap import AlignedLine, BeatSlot, ProsodyAnalysis, ScheduledSyllable, Syllable, analyse_syllables


def align_exact(analysis: ProsodyAnalysis, slots: Sequence[BeatSlot]) -> tuple[ScheduledSyllable, ...]:
    """Schedule one already-analyzed syllable per slot in source order."""
    if len(analysis.syllables) != len(slots):
        raise ValueError(f"exact alignment requires {len(slots)} syllables, got {len(analysis.syllables)}")
    return tuple(
        ScheduledSyllable(slot=slot, syllable=syllable)
        for syllable, slot in zip(analysis.syllables, slots, strict=True)
    )


def align_legacy_flexible(text: str, slots: Sequence[BeatSlot]) -> AlignedLine:
    """Assign a line to strictly increasing slots, favoring accents and flow."""
    syllables = analyse_syllables(text)
    if not slots:
        raise ValueError("slots must not be empty")
    if not syllables:
        return AlignedLine(text=text, syllables=(), events=(), score=-1000.0)

    overflow_count = max(0, len(syllables) - len(slots))
    if overflow_count:
        return AlignedLine(
            text=text,
            syllables=syllables,
            events=(),
            score=-1000.0 - float(overflow_count),
            overflow_count=overflow_count,
        )

    assignments, score = _best_slot_path(syllables, slots)
    events = tuple(
        ScheduledSyllable(slot=slots[slot_index], syllable=syllable)
        for syllable, slot_index in zip(syllables, assignments, strict=True)
    )
    density_penalty = (len(slots) - len(syllables)) * 0.20
    return AlignedLine(
        text=text,
        syllables=syllables,
        events=events,
        score=score - density_penalty,
    )


def align_text_to_slots(text: str, slots: Sequence[BeatSlot]) -> AlignedLine:
    """Preserve the original flexible alignment API for existing callers."""
    return align_legacy_flexible(text, slots)


def choose_best_line(candidates: Sequence[str], slots: Sequence[BeatSlot]) -> AlignedLine:
    """Return the highest-scoring candidate, preserving source order on ties."""
    if not candidates:
        raise ValueError("candidates must not be empty")

    best_line: AlignedLine | None = None
    for text in candidates:
        line = align_text_to_slots(text, slots)
        if best_line is None or line.score > best_line.score:
            best_line = line
    if best_line is None:
        raise RuntimeError("candidate selection did not produce a line")
    return best_line


def _best_slot_path(syllables: Sequence[Syllable], slots: Sequence[BeatSlot]) -> tuple[tuple[int, ...], float]:
    """Find the best monotonic path; first syllable intentionally owns beat one."""
    total_syllables = len(syllables)
    total_slots = len(slots)
    states: dict[int, tuple[float, tuple[int, ...]]] = {
        0: (_slot_score(syllables[0], slots[0], syllable_index=0, total_syllables=total_syllables, total_slots=total_slots), (0,))
    }

    for syllable_index, syllable in enumerate(syllables[1:], start=1):
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        remaining_syllables = total_syllables - syllable_index - 1
        for slot_index in range(syllable_index, total_slots - remaining_syllables):
            best_for_slot: tuple[float, tuple[int, ...]] | None = None
            for previous_slot, (previous_score, previous_path) in states.items():
                if previous_slot >= slot_index:
                    continue
                score = previous_score + _slot_score(
                    syllable,
                    slots[slot_index],
                    syllable_index=syllable_index,
                    total_syllables=total_syllables,
                    total_slots=total_slots,
                )
                if best_for_slot is None or score > best_for_slot[0]:
                    best_for_slot = (score, (*previous_path, slot_index))
            if best_for_slot is not None:
                next_states[slot_index] = best_for_slot
        states = next_states

    return max(states.values(), key=lambda item: item[0])[1], max(states.values(), key=lambda item: item[0])[0]


def _slot_score(
    syllable: Syllable,
    slot: BeatSlot,
    *,
    syllable_index: int,
    total_syllables: int,
    total_slots: int,
) -> float:
    ideal_slot = 0.0 if total_syllables == 1 else syllable_index * (total_slots - 1) / (total_syllables - 1)
    stress_multiplier = 1.50 if syllable.stressed else 1.00
    onset_bonus = 0.10 if slot.tick_in_beat == 0 else 0.0
    progression_penalty = 0.10 * abs((slot.tick % total_slots) - ideal_slot)
    return slot.accent * stress_multiplier + onset_bonus - progression_penalty

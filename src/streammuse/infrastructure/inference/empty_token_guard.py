"""Four consecutive explicit EMPTY beats block EMPTY in the next eight beats."""


class EmptyTokenGuard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.consecutive_empty = 0
        self.remaining_beats = 0
        self.last_tick = None

    def before_beat(self, tick: int) -> bool:
        # A jump/replay is not a consecutive beat sequence.
        if self.last_tick is not None and tick != self.last_tick + 4:
            self.reset()
        return self.remaining_beats > 0

    def observe(self, tick: int, raw_tokens: list[int]) -> dict:
        blocked = self.before_beat(tick)
        before = self.remaining_beats
        if blocked and 169 in raw_tokens:
            raise RuntimeError("EMPTY token escaped the eight-beat sampling guard")
        # Deliberately do not classify sustain, NOTE_ON counts, or synthesized
        # EMPTY tokens from postprocessing an ACC_END-only model output.
        empty = list(raw_tokens) == [169, 170]
        self.consecutive_empty = self.consecutive_empty + 1 if empty else 0
        if blocked:
            self.remaining_beats -= 1
        triggered = self.consecutive_empty == 4
        if triggered:
            self.remaining_beats = 8
            self.consecutive_empty = 0
        self.last_tick = tick
        return {
            "generation_start_tick": tick,
            "explicit_empty_beat": empty,
            "empty_token_blocked": blocked,
            "triggered_for_next_beat": triggered,
            "remaining_before": before,
            "remaining_after": self.remaining_beats,
            "consecutive_empty_after": self.consecutive_empty,
        }

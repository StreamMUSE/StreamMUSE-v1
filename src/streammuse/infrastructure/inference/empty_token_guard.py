"""Four consecutive explicit EMPTY beats block EMPTY in the next eight beats."""


class EmptyTokenGuard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.consecutive_empty = 0
        self.remaining_beats = 0
        self.last_tick = None

    def snapshot(self):
        return dict(consecutive_empty=self.consecutive_empty,
                    remaining_beats=self.remaining_beats, last_tick=self.last_tick)

    def restore(self, state):
        if type(state) is not dict or set(state) != {"consecutive_empty", "remaining_beats", "last_tick"}:
            raise ValueError("invalid EMPTY guard snapshot")
        count, remaining, tick = state["consecutive_empty"], state["remaining_beats"], state["last_tick"]
        if (type(count) is not int or not 0 <= count <= 3 or type(remaining) is not int
                or not 0 <= remaining <= 8 or
                (tick is not None and (type(tick) is not int or tick < 0 or tick % 4))):
            raise ValueError("invalid EMPTY guard snapshot values")
        self.consecutive_empty, self.remaining_beats, self.last_tick = count, remaining, tick

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

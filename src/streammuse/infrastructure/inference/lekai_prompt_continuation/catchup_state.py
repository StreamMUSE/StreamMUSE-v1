"""Beat-level catch-up state for Lekai prompt-continuation inference.

This module deliberately has no model, tokenizer, or HTTP dependencies. It only
answers: given melody/accompaniment history lengths, how many accompaniment
beats are required before playback is safe?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CatchUpState:
    """Track melody/accompaniment progress in beats.

    Playback is ready only when accompaniment contains one beat beyond the
    observed melody history. Equal lengths mean the accompaniment has caught up
    to history, but there is not yet a next beat to play.
    """

    melody_history_beats: int = 0
    accompaniment_history_beats: int = 0
    playable_lookahead_beats: int = 1

    def __post_init__(self) -> None:
        self._validate_non_negative("melody_history_beats", self.melody_history_beats)
        self._validate_non_negative("accompaniment_history_beats", self.accompaniment_history_beats)
        self._validate_non_negative("playable_lookahead_beats", self.playable_lookahead_beats)

    @staticmethod
    def _validate_non_negative(name: str, value: int) -> None:
        if int(value) < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    def observe_melody_beats(self, beats: int) -> None:
        self._validate_non_negative("beats", beats)
        self.melody_history_beats += int(beats)

    def accept_prompt_accompaniment(self, beats: int) -> None:
        self.accept_accompaniment_beats(beats)

    def accept_continuation_beats(self, beats: int) -> None:
        self.accept_accompaniment_beats(beats)

    def accept_accompaniment_beats(self, beats: int) -> None:
        self._validate_non_negative("beats", beats)
        self.accompaniment_history_beats += int(beats)

    def set_history_lengths(self, *, melody_beats: int, accompaniment_beats: int) -> None:
        self._validate_non_negative("melody_beats", melody_beats)
        self._validate_non_negative("accompaniment_beats", accompaniment_beats)
        self.melody_history_beats = int(melody_beats)
        self.accompaniment_history_beats = int(accompaniment_beats)

    def reset(self) -> None:
        self.melody_history_beats = 0
        self.accompaniment_history_beats = 0

    def target_playable_accompaniment_beats(self) -> int:
        return self.melody_history_beats + self.playable_lookahead_beats

    def beats_needed_for_playback(self) -> int:
        return max(0, self.target_playable_accompaniment_beats() - self.accompaniment_history_beats)

    def is_history_aligned(self) -> bool:
        return self.accompaniment_history_beats >= self.melody_history_beats

    def is_playback_ready(self) -> bool:
        return self.beats_needed_for_playback() == 0

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "melody_history_beats": int(self.melody_history_beats),
            "accompaniment_history_beats": int(self.accompaniment_history_beats),
            "playable_lookahead_beats": int(self.playable_lookahead_beats),
            "target_playable_accompaniment_beats": int(self.target_playable_accompaniment_beats()),
            "beats_needed_for_playback": int(self.beats_needed_for_playback()),
            "is_history_aligned": bool(self.is_history_aligned()),
            "is_playback_ready": bool(self.is_playback_ready()),
        }

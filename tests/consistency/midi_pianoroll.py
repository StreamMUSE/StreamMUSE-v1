"""Pianoroll-level accompaniment comparison for the realtime/offline consistency test.

Why not compare raw MIDI note (tick, pitch, type) events?
    The realtime path (RealTimeMusicService -> combined.mid) re-triggers a sustained
    pitch at every beat boundary (note_off + note_on each beat), while the offline path
    (run_lekai_offline.py -> *_generated.mid) keeps it as a single long note. The two are
    the *same pianoroll* but differ at the note_on/note_off bookkeeping level, so a raw
    event comparison reports spurious mismatches (~78% in manual Phase 0 testing).

What we compare instead:
    Each accompaniment note is expanded into the set of (beat, pitch) cells it sounds
    through. Two outputs are consistent iff their (beat, pitch) active sets are equal
    within a common beat window. This normalizes away the sustained-vs-retriggered
    representation difference. (Verified Phase 0: song 4 -> 100% match.)

Windowing:
    The melody MIDI is usually shorter than the realtime --max-ticks, so realtime keeps
    generating past the song end (with no melody to condition on) while offline stops at
    the data end. Always truncate the comparison to a common ``max_beat`` (the melody's
    last beat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mido

ACCOMPANIMENT_TRACK_KEYWORDS = ("Accompaniment", "Part1")


def _note_intervals(
    midi_path: str | Path,
    track_keywords: tuple[str, ...],
) -> tuple[list[tuple[int, int, int]], int]:
    """Return (start_tick, end_tick, pitch) intervals for matching tracks, plus ticks_per_beat.

    Unclosed note_on events (hanging notes) are closed at the file's max tick so they still
    contribute to the active set rather than being silently dropped.
    """
    mid = mido.MidiFile(str(midi_path))
    intervals: list[tuple[int, int, int]] = []
    max_tick = 0
    for track in mid.tracks:
        name = ""
        abs_tick = 0
        open_notes: dict[int, list[int]] = {}
        for msg in track:
            abs_tick += msg.time
            max_tick = max(max_tick, abs_tick)
            if msg.type == "track_name":
                name = str(getattr(msg, "name", ""))
                continue
            if not any(k in name for k in track_keywords):
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault(msg.note, []).append(abs_tick)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                starts = open_notes.get(msg.note)
                if starts:
                    start = starts.pop(0)
                    intervals.append((start, abs_tick, msg.note))
        # Close any hanging notes at the end of the track.
        for pitch, starts in open_notes.items():
            for start in starts:
                intervals.append((start, abs_tick, pitch))
    return intervals, mid.ticks_per_beat


def accompaniment_active_cells(
    midi_path: str | Path,
    *,
    max_beat: int | None = None,
    track_keywords: tuple[str, ...] = ACCOMPANIMENT_TRACK_KEYWORDS,
) -> set[tuple[int, int]]:
    """Return the set of (beat, pitch) cells the accompaniment sounds through.

    A note spanning [start_tick, end_tick) marks every beat it overlaps as active for its
    pitch. If ``max_beat`` is given, only beats ``< max_beat`` are kept (common window).
    """
    intervals, ticks_per_beat = _note_intervals(midi_path, track_keywords)
    active: set[tuple[int, int]] = set()
    for start, end, pitch in intervals:
        start_beat = start // ticks_per_beat
        end_beat = max(start_beat, (end - 1) // ticks_per_beat)
        for beat in range(start_beat, end_beat + 1):
            if max_beat is not None and beat >= max_beat:
                continue
            active.add((beat, pitch))
    return active


@dataclass
class PianorollComparison:
    realtime_count: int
    offline_count: int
    matched: int
    only_in_realtime: list[tuple[int, int]] = field(default_factory=list)
    only_in_offline: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return not self.only_in_realtime and not self.only_in_offline

    @property
    def match_rate(self) -> float:
        union = self.realtime_count + self.offline_count - self.matched
        return 100.0 if union == 0 else round(self.matched / union * 100, 2)

    def summary(self, *, head: int = 10) -> str:
        lines = [
            f"realtime cells={self.realtime_count}, offline cells={self.offline_count}, "
            f"matched={self.matched}, match_rate={self.match_rate}%",
        ]
        if self.only_in_realtime:
            lines.append(f"  only in realtime (first {head}): {self.only_in_realtime[:head]}")
        if self.only_in_offline:
            lines.append(f"  only in offline  (first {head}): {self.only_in_offline[:head]}")
        return "\n".join(lines)


def compare_accompaniment(
    realtime_midi: str | Path,
    offline_midi: str | Path,
    *,
    max_beat: int | None = None,
) -> PianorollComparison:
    """Compare two accompaniment outputs at the pianoroll (beat, pitch) level."""
    rt = accompaniment_active_cells(realtime_midi, max_beat=max_beat)
    off = accompaniment_active_cells(offline_midi, max_beat=max_beat)
    return PianorollComparison(
        realtime_count=len(rt),
        offline_count=len(off),
        matched=len(rt & off),
        only_in_realtime=sorted(rt - off),
        only_in_offline=sorted(off - rt),
    )

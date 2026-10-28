"""Small hand-authored chord catalog, independent of MCFlow or lyric templates."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ChordLoop:
    """One MIDI-pitch voicing per bar, starting at absolute bar zero."""

    name: str
    chords: tuple[tuple[int, ...], ...]


CHORD_LOOPS: Mapping[str, ChordLoop] = MappingProxyType({
    "minor_cycle": ChordLoop(
        name="Am-F-C-G",
        chords=((45, 52, 57, 60), (41, 48, 53, 57), (48, 55, 60, 64), (43, 50, 55, 59)),
    ),
    "major_cycle": ChordLoop(
        name="C-G-Am-F",
        chords=((48, 55, 60, 64), (43, 50, 55, 59), (45, 52, 57, 60), (41, 48, 53, 57)),
    ),
})
CHORD_LOOP_IDS = ("off", "minor_cycle", "major_cycle")

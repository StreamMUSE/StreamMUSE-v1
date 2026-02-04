"""
Clean MIDI utilities with no model dependencies

This module provides MIDI file reading utilities without requiring
transformers or any model dependencies.
"""

from preprocess.xf_midi import XFMidi


def midi_to_note(midi_path: str, min_pitch: int = 0, max_pitch: int = 127,
                 beat_div: int = 4, program: int = None, max_tick: int = None):
    """
    Read MIDI file and convert to note list format.

    Args:
        midi_path: Path to MIDI file
        min_pitch: Minimum pitch to include (0-127)
        max_pitch: Maximum pitch to include (0-127)
        beat_div: Beats per quarter note (default 4 = 16th notes)
        program: Filter by specific MIDI program/instrument (None = all)
        max_tick: Maximum tick to read (None = entire file)

    Returns:
        tuple: (notes, resolution, max_tick)
            notes: List of dicts with 'pitch', 'tick', 'duration'
            resolution: MIDI ticks per beat
            max_tick: Total ticks in file
    """
    midi = XFMidi(midi_path, constant_tempo=60.0 / beat_div)

    if max_tick is None:
        max_tick = int(midi.get_end_time())

    notes = []
    for inst in midi.instruments:
        # Filter by program if specified
        if program is not None and inst.program != program:
            continue

        for note in inst.notes:
            if min_pitch <= note.pitch <= max_pitch:
                start_tick = int(round(note.start))
                end_tick = int(round(note.end))

                if start_tick >= 0 and end_tick < max_tick:
                    notes.append({
                        'pitch': note.pitch,
                        'tick': start_tick,
                        'duration': end_tick - start_tick
                    })

    # Sort by tick
    notes.sort(key=lambda x: x['tick'])

    return notes, midi.resolution, max_tick

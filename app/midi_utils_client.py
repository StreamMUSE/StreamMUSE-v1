"""
Client-side MIDI utilities using mido (no server dependencies)

This module provides MIDI file reading utilities for the web client
without requiring preprocess or model dependencies.
"""

import mido


def midi_to_note_client(midi_path: str, min_pitch: int = 0, max_pitch: int = 127,
                        beat_div: int = 4, program: int = None, max_tick: int = None):
    """
    Read MIDI file and convert to note list format using mido.
    
    Args:
        midi_path: Path to MIDI file
        min_pitch: Minimum pitch to include (0-127)
        max_pitch: Maximum pitch to include (0-127)
        beat_div: Ticks per beat for output (default 4)
        program: Filter by specific MIDI program/instrument (None = all)
        max_tick: Maximum tick to read (None = entire file)
    
    Returns:
        tuple: (notes, resolution, actual_max_tick)
            notes: List of dicts with 'pitch', 'tick', 'duration'
            resolution: MIDI ticks per beat in original file
            actual_max_tick: Total ticks in converted output
    """
    midi_file = mido.MidiFile(midi_path)
    resolution = midi_file.ticks_per_beat
    
    ticks_per_output_tick = resolution / beat_div
    
    notes = []
    active_notes = {}
    
    for track in midi_file.tracks:
        current_tick = 0
        current_program = 0
        
        for msg in track:
            current_tick += msg.time
            
            if msg.type == 'program_change':
                current_program = msg.program
            
            elif msg.type == 'note_on' and msg.velocity > 0:
                if program is not None and current_program != program:
                    continue
                if min_pitch <= msg.note <= max_pitch:
                    output_tick = int(round(current_tick / ticks_per_output_tick))
                    active_notes[(msg.channel, msg.note)] = output_tick
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start_tick = active_notes[key]
                    output_tick = int(round(current_tick / ticks_per_output_tick))
                    duration = max(1, output_tick - start_tick)
                    
                    notes.append({
                        'pitch': msg.note,
                        'tick': start_tick,
                        'duration': duration
                    })
                    del active_notes[key]
    
    notes.sort(key=lambda x: (x['tick'], x['pitch']))
    
    if notes:
        actual_max_tick = max(n['tick'] + n['duration'] for n in notes)
    else:
        actual_max_tick = 0
    
    if max_tick is not None:
        notes = [n for n in notes if n['tick'] < max_tick]
        actual_max_tick = min(actual_max_tick, max_tick)
    
    return notes, resolution, actual_max_tick

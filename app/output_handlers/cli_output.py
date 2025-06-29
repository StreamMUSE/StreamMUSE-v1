"""
This is the CLI output handler for the StreamMUSE end to end system.
"""

from collections import deque
import shutil
import time

class CLIOutputHandler:
    """
    Handles displaying a persistent log with updating status lines at the bottom.
    """

    def __init__(self, ticks_per_beat: int, beats_per_bar: int, log_displays_count: int = 10):
        """
        Initialize the CLI output handler.
        """
        self.ticks_per_beat = ticks_per_beat
        self.beats_per_bar = beats_per_bar
        self.log_displays_count = log_displays_count
        self.status_line_count = 4
        self.total_managed_lines = self.status_line_count + self.log_displays_count

        # Use deque for maintaining fixed size history
        self.log_history = deque(maxlen=100)
        self.last_model_output_str = "None"
        self.is_first_display = True
        self.all_inference_times = []

    def _midi_to_note_name(self, pitch: int) -> str:
        """
        Converts a MIDI pitch number (0-127) to its scientific pitch notation.
        e.g., 60 -> "C4", 69 -> "A4", 38 -> "D2"
        """
        if not 0 <= pitch <= 127:
            return "N/A"
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        return f"{note_names[note_index]}{octave}"
    
    def update_and_display(self, tick_count, music_info, user_notes_played, model_notes_played):
        """
        Update the display with the latest log and status lines.
        """
        user_notes_played_str = ', '.join([self._midi_to_note_name(note) for note in user_notes_played])
        model_notes_played_str = ', '.join([self._midi_to_note_name(note) for note in model_notes_played])

        bar = music_info.get("bar", 0)
        beat = music_info.get("beat", 0)
        tick = (tick_count % self.ticks_per_beat) + 1
        log_entry = f'[{bar:03d}.{beat}.{tick}]'
        
        if user_notes_played_str:
            log_entry += f" USER: [{user_notes_played_str}]"
        if model_notes_played_str:
            log_entry += f" | MODEL: [{model_notes_played_str}]"
        
        # Placeholder for no note events
        if not user_notes_played_str and not model_notes_played_str:
            log_entry += " ..."

        # Update log history
        self.log_history.append(log_entry)

        # Line 1: Inputs during this tick
        line_inputs = f'INPUTS (current tick): {user_notes_played_str or 'None'}'

        # Line 2: Model output
        if music_info.get('inference_triggered'):
            future_events = ...
        

"""
Handles saving the performance to a MIDI file.
"""

import pretty_midi
import os
import time


class MidiFileHandler:
    """
    Collects user and model notes during a session and saves them to a MIDI file.
    """

    def __init__(self, tempo: float, ticks_per_beat: int):
        """
        Initializes the MIDI file handler.

        Args:
            tempo (float): The tempo of the performance in beats per minute.
            ticks_per_beat (int): The number of ticks per beat.
        """
        self.seconds_per_tick = (60.0 / tempo) / ticks_per_beat
        self.midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Track 0: User Melody (Acoustic Grand Piano, program 0)
        # For experiment convenience, we still use "Guitar" as the instrument name.
        self.user_instrument = pretty_midi.Instrument(program=0, name="Guitar")
        # Track 1: Model Accompaniment (Acoustic Grand Piano, program 0)
        # For experiment convenience, we still use "Piano" as the instrument name.
        self.model_instrument = pretty_midi.Instrument(program=0, name="Piano")

        self.midi_data.instruments.append(self.user_instrument)
        self.midi_data.instruments.append(self.model_instrument)

    def add_user_note(self, note: dict):
        """
        Adds a user-played note to the MIDI file.

        Args:
            note (dict): A dictionary with 'pitch', 'tick', and 'duration' keys.
        """
        start_time = note["tick"] * self.seconds_per_tick
        end_time = start_time + (note["duration"] * self.seconds_per_tick)
        midi_note = pretty_midi.Note(
            velocity=100,  # Use a standard velocity for user notes in the log
            pitch=note["pitch"],
            start=start_time,
            end=end_time,
        )
        self.user_instrument.notes.append(midi_note)

    def add_model_note(self, note: dict):
        """
        Adds a model-generated note to the MIDI file.

        Args:
            note (dict): A dictionary with 'pitch', 'tick', and 'duration' keys.
        """
        start_time = note["tick"] * self.seconds_per_tick
        end_time = start_time + (note["duration"] * self.seconds_per_tick)
        midi_note = pretty_midi.Note(
            velocity=80,  # Use a slightly different velocity for model notes
            pitch=note["pitch"],
            start=start_time,
            end=end_time,
        )
        self.model_instrument.notes.append(midi_note)

    def save_to_midi(
        self, session_log_dir: str, log_file_prefix="performance", midi_file_name=None
    ):
        """
        Saves the collected notes to a timestamped MIDI file in the session directory.
        """
        if not self.user_instrument.notes and not self.model_instrument.notes:
            print("\nNo notes were played, MIDI file will not be saved.")
            return

        if midi_file_name is None:
            midi_file_name = f"{log_file_prefix}.mid"
            log_filepath = os.path.join(session_log_dir, midi_file_name)
        else:
            midi_file_name = f"{midi_file_name}.mid"
            log_filepath = f"{session_log_dir}/{midi_file_name}"

        try:
            self.midi_data.write(log_filepath)
            print(f"Performance MIDI file saved to: {log_filepath}")
        except IOError as e:
            print(f"\nError saving MIDI file: {e}")

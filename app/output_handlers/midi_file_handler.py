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

        # Event-based recording state.
        # We convert note_on/note_off streams into pretty_midi.Note objects.
        # Key: pitch -> {'start_time': float, 'velocity': int}
        self._active_user_notes = {}
        self._active_model_notes = {}

    def _note_start_time(self, note_or_event: dict) -> float:
        return float(note_or_event["tick"]) * self.seconds_per_tick

    def _handle_event(self, note_or_event: dict, instrument: pretty_midi.Instrument, active_map: dict, default_velocity: int):
        """Handle both legacy duration-notes and event-based note_on/note_off."""
        if "tick" not in note_or_event or "pitch" not in note_or_event:
            return

        pitch = int(note_or_event["pitch"])
        start_time = self._note_start_time(note_or_event)

        # Legacy note format: {'pitch','tick','duration'}
        if "duration" in note_or_event and note_or_event.get("duration") is not None:
            duration = int(note_or_event["duration"])
            if duration <= 0:
                return
            end_time = start_time + duration * self.seconds_per_tick
            vel = int(note_or_event.get("velocity", default_velocity))
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=vel,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
            )
            return

        event_type = note_or_event.get("type")

        # If no explicit type is provided, we can't infer event semantics.
        if event_type not in ("note_on", "note_off"):
            return

        if event_type == "note_on":
            # If retrigger occurs without prior note_off, close previous note.
            if pitch in active_map:
                prev = active_map.pop(pitch)
                prev_start = float(prev["start_time"])
                prev_vel = int(prev["velocity"])
                if start_time > prev_start:
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=prev_vel,
                            pitch=pitch,
                            start=prev_start,
                            end=start_time,
                        )
                    )

            vel = int(note_or_event.get("velocity", default_velocity))
            active_map[pitch] = {"start_time": start_time, "velocity": vel}
            return

        # note_off
        prev = active_map.pop(pitch, None)
        if prev is None:
            return
        prev_start = float(prev["start_time"])
        prev_vel = int(prev["velocity"])
        if start_time <= prev_start:
            return
        instrument.notes.append(
            pretty_midi.Note(
                velocity=prev_vel,
                pitch=pitch,
                start=prev_start,
                end=start_time,
            )
        )

    def add_user_note(self, note: dict):
        """
        Adds a user-played note to the MIDI file.

        Args:
            note (dict):
                - Legacy note format: {'pitch', 'tick', 'duration', ...}
                - Event format: {'type': 'note_on'|'note_off', 'pitch', 'tick', 'velocity'?, ...}
        """
        self._handle_event(
            note_or_event=note,
            instrument=self.user_instrument,
            active_map=self._active_user_notes,
            default_velocity=100,
        )

    def add_model_note(self, note: dict):
        """
        Adds a model-generated note to the MIDI file.

        Args:
            note (dict):
                - Legacy note format: {'pitch', 'tick', 'duration', ...}
                - Event format: {'type': 'note_on'|'note_off', 'pitch', 'tick', 'velocity'?, ...}
        """
        self._handle_event(
            note_or_event=note,
            instrument=self.model_instrument,
            active_map=self._active_model_notes,
            default_velocity=80,
        )

    def finalize(self):
        """Close any still-active notes at the end of a session."""

        def flush(active_map: dict, instrument: pretty_midi.Instrument):
            if not instrument.notes:
                end_time = 0.0
            else:
                end_time = max(n.end for n in instrument.notes)
            for pitch, info in list(active_map.items()):
                start_time = float(info["start_time"])
                vel = int(info["velocity"])
                if end_time > start_time:
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=vel,
                            pitch=int(pitch),
                            start=start_time,
                            end=end_time,
                        )
                    )
                active_map.pop(pitch, None)

        flush(self._active_user_notes, self.user_instrument)
        flush(self._active_model_notes, self.model_instrument)

    def save_to_midi(
        self, session_log_dir: str, log_file_prefix="performance", midi_file_name=None
    ):
        """
        Saves the collected notes to a timestamped MIDI file in the session directory.
        """
        # DEBUG: Print statistics before finalize
        print(f"  [DEBUG MIDI] Before finalize:")
        print(f"    User notes recorded: {len(self.user_instrument.notes)}")
        print(f"    Model notes recorded: {len(self.model_instrument.notes)}")
        print(f"    Active user notes (missing note_off): {len(self._active_user_notes)}")
        print(f"    Active model notes (missing note_off): {len(self._active_model_notes)}")
        
        self.finalize()
        
        # DEBUG: Print statistics after finalize
        print(f"  [DEBUG MIDI] After finalize:")
        print(f"    User notes total: {len(self.user_instrument.notes)}")
        print(f"    Model notes total: {len(self.model_instrument.notes)}")
        
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

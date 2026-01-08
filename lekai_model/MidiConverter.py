import pretty_midi
import numpy as np


class MidiConverter:
    def __init__(self, ticks_per_beat=4):
        self.ticks_per_beat = ticks_per_beat

    def load_midi(self, midi_path):
        """
        Load MIDI file and extract metadata.
        """
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI: {e}")
            return None, None

        # Get BPM
        bpm = 120.0
        if pm.get_tempo_changes()[1].size > 0:
            bpm = pm.get_tempo_changes()[1][0]

        # Get Time Signature
        time_sig = (4, 4)
        if pm.time_signature_changes:
            ts = pm.time_signature_changes[0]
            time_sig = (ts.numerator, ts.denominator)

        metadata = {"bpm": bpm, "time_sig": time_sig, "resolution": pm.resolution}

        return pm, metadata

    def midi_to_notes(self, pm, filter_drums=True):
        """
        Convert PrettyMIDI object to list of quantized notes.
        Returns: notes, max_tick
        """
        if pm is None:
            return [], 0

        # Calculate quantization factor
        # We want to map seconds to ticks based on self.ticks_per_beat
        # But PrettyMIDI works in seconds.
        # We need to use the BPM to convert seconds to beats, then beats to ticks.

        # Note: This simple quantization assumes constant BPM for the whole file
        # or uses the initial BPM. For variable BPM, we should use pm.time_to_tick
        # but pm.time_to_tick uses the file's resolution (PPQ).
        # We want our own resolution (ticks_per_beat).

        # Strategy:
        # 1. Get note start/end in seconds.
        # 2. Convert seconds to beats using the tempo map.
        # 3. Convert beats to ticks (beats * ticks_per_beat).

        notes = []
        max_tick = 0

        # Get tempo changes
        tempo_change_times, tempi = pm.get_tempo_changes()

        def time_to_beat(time_sec):
            # Find the tempo active at time_sec
            # This is a simplified version, pretty_midi doesn't have a direct time_to_beat
            # We can integrate.

            # If no tempo changes or simple case
            if len(tempo_change_times) == 0:
                return time_sec * (120.0 / 60.0)

            beat = 0.0
            last_time = 0.0
            last_tempo = 120.0

            for i, t in enumerate(tempo_change_times):
                if t > time_sec:
                    break
                # Add beats from last_time to t
                beat += (t - last_time) * (last_tempo / 60.0)
                last_time = t
                last_tempo = tempi[i]

            # Add remaining
            beat += (time_sec - last_time) * (last_tempo / 60.0)
            return beat

        for inst in pm.instruments:
            if filter_drums and inst.is_drum:
                continue

            for note in inst.notes:
                start_beat = time_to_beat(note.start)
                end_beat = time_to_beat(note.end)

                start_tick = int(round(start_beat * self.ticks_per_beat))
                end_tick = int(round(end_beat * self.ticks_per_beat))
                duration = max(1, end_tick - start_tick)

                if start_tick < 0:
                    continue

                notes.append(
                    {
                        "pitch": note.pitch,
                        "tick": start_tick,
                        "duration": duration,
                        "velocity": note.velocity,
                        "program": inst.program,
                        "is_drum": inst.is_drum,
                    }
                )
                max_tick = max(max_tick, end_tick)

        notes.sort(key=lambda x: x["tick"])
        return notes, max_tick

    def notes_to_pianoroll(self, notes, max_tick=None):
        """
        Convert notes list to (2, 88, T) piano roll.
        Channel 0: Sustain
        Channel 1: Onset
        """
        if max_tick is None:
            if not notes:
                return np.zeros((2, 88, 0), dtype=np.uint8)
            max_tick = max(n["tick"] + n["duration"] for n in notes)

        pianoroll = np.zeros((2, 88, max_tick), dtype=np.uint8)

        if not notes:
            return pianoroll

        for note in notes:
            pitch_idx = note["pitch"] - 21  # MIDI 21-108
            if not (0 <= pitch_idx < 88):
                continue

            start = int(note["tick"])
            duration = int(note["duration"])
            end = min(start + duration, max_tick)

            if start >= max_tick:
                continue

            # Channel 0: Sustain
            pianoroll[0, pitch_idx, start:end] = 1
            # Channel 1: Onset
            pianoroll[1, pitch_idx, start] = 1

        return pianoroll

    def pianoroll_to_notes(self, pianoroll):
        """
        Convert (2, 88, T) piano roll back to notes list.
        """
        notes = []
        sustain = pianoroll[0]
        onset = pianoroll[1]
        max_tick = pianoroll.shape[2]

        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            # Find onsets
            onset_indices = np.where(onset[pitch_idx] > 0)[0]
            for start in onset_indices:
                # Find end of sustain
                end = start + 1
                while end < max_tick and sustain[pitch_idx, end] > 0:
                    end += 1

                duration = end - start
                notes.append(
                    {
                        "pitch": pitch,
                        "tick": start,
                        "duration": duration,
                        "velocity": 80,  # Default
                    }
                )

        notes.sort(key=lambda x: x["tick"])
        return notes

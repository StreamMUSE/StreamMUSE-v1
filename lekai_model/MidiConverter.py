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

    def events_to_pianoroll(self, events, start_tick, end_tick, active_pitches=None):
        """Convert event stream into a pianoroll for a specific tick window.

        Args:
            events (List[dict]): Each event:
                {"type": "note_on"|"note_off", "pitch": int, "tick": int}
                Tick is absolute on the engine's tick grid.
            start_tick (int): Inclusive window start (absolute tick).
            end_tick (int): Exclusive window end (absolute tick). Output length is
                `end_tick - start_tick`.
            active_pitches (Iterable[int], optional): Pitches that are already
                active strictly **before** start_tick (i.e., sustained into this
                window). These will be sustained starting at window-relative tick 0
                until ended by a note_off event inside the window.

        Returns:
            np.ndarray: pianoroll of shape (2, 88, end_tick-start_tick), dtype uint8.
        """

        start_tick = int(start_tick)
        end_tick = int(end_tick)
        if end_tick < start_tick:
            raise ValueError(f"end_tick ({end_tick}) must be >= start_tick ({start_tick})")

        T = end_tick - start_tick

        if active_pitches is None:
            active_pitches = set()
        else:
            active_pitches = set(active_pitches)

        pianoroll = np.zeros((2, 88, T), dtype=np.uint8)
        if T == 0:
            return pianoroll

        # Helper to map pitch to 0..87 index
        def _pidx(pitch):
            return int(pitch) - 21

        # Sustain from window start for pitches that were already active before start_tick
        for p in list(active_pitches):
            idx = _pidx(p)
            if 0 <= idx < 88:
                pianoroll[0, idx, 0:T] = 1

        # Filter to window then shift tick to window-relative
        evt_sorted = []
        for e in events:
            if not isinstance(e, dict):
                continue
            if e.get("type") not in {"note_on", "note_off"}:
                continue
            if "pitch" not in e or "tick" not in e:
                continue
            t_abs = int(e["tick"])
            if not (start_tick <= t_abs < end_tick):
                continue
            evt_sorted.append(
                {
                    "type": e.get("type"),
                    "pitch": int(e["pitch"]),
                    "tick": t_abs - start_tick,
                }
            )

        # Sort events by time; process note_off before note_on if same tick
        evt_sorted.sort(key=lambda e: (int(e.get("tick", 0)), 0 if e.get("type") == "note_off" else 1))

        # Track active pitches within the window (initialized from carry-in)
        active = set(active_pitches)

        for e in evt_sorted:
            et = e.get("type")
            p = e.get("pitch")
            t = int(e.get("tick", 0))
            if t < 0 or t >= T:
                continue
            idx = _pidx(p)
            if not (0 <= idx < 88):
                continue

            if et == "note_off":
                if p in active:
                    # Turn sustain off starting at tick t (relative)
                    pianoroll[0, idx, t:T] = 0
                    active.discard(p)

            else:  # note_on
                # Start (or retrigger) sustain from t and mark onset at t.
                pianoroll[0, idx, t:T] = 1
                pianoroll[1, idx, t] = 1
                active.add(p)

        return pianoroll

    def pianoroll_to_events(self, pianoroll, start_tick=0):
        """Convert (2, 88, T) piano roll back to note_on/note_off events.

        We interpret:
        - onset channel (1) as note_on.
        - sustain channel (0) falling edge (1->0) as note_off.
        - sustain length with no falling edge produces no note_off (caller may
          treat as still active after the window).

        Args:
            pianoroll (np.ndarray): shape (2, 88, T)
            start_tick (int): absolute tick offset to add to emitted events.

        Returns:
            List[dict]: events sorted by tick.
        """

        events = []
        sustain = pianoroll[0]
        onset = pianoroll[1]
        T = pianoroll.shape[2]

        for pitch_idx in range(88):
            pitch = pitch_idx + 21

            # note_on: any onset==1
            on_ticks = np.where(onset[pitch_idx] > 0)[0]
            for t in on_ticks:
                events.append({"type": "note_on", "pitch": int(pitch), "tick": int(start_tick + t)})

            # note_off: falling edges in sustain
            if T <= 1:
                continue
            s = sustain[pitch_idx].astype(np.int8)
            # falling edge at t means s[t-1]==1 and s[t]==0
            fall = np.where((s[:-1] == 1) & (s[1:] == 0))[0]
            for t0 in fall:
                t = t0 + 1
                events.append({"type": "note_off", "pitch": int(pitch), "tick": int(start_tick + t)})

        events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1))
        return events

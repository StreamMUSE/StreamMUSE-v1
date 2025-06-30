"""
This is the audio output handler for the StreamMUSE end to end system.
"""

import mido
import time

METRONOME_CHANNEL = 9
METRONOME_PITCH_FIRST = 76
METRONOME_PITCH_OTHER = 77
METRONOME_VELOCITY_FIRST = 100
METRONOME_VELOCITY_OTHER = 70
PIANO_CHANNEL = 0

class AudioOutputHandler:
    def __init__(self, port_name: str = None):
        """
        Initialize the audio output handler and the MIDI port.
        """
        print("\n--- Available MIDI Output Ports ---")
        try:
            output_names = mido.get_output_names()
            if not output_names:
                print("  No MIDI output ports found.")
            else:
                for name in output_names:
                    print(f"  - '{name}'")
        except Exception as e:
            print(f"  Could not list MIDI ports: {e}")
        print("-------------------------------------\n")
        
        self.port = None
        try:
            self.port = mido.open_output(port_name)
            print(f"Successfully opened MIDI output port: '{self.port.name}'")
            # Set instrument to Acoustic Grand Piano on channel 0
            self.port.send(mido.Message('program_change', channel=PIANO_CHANNEL, program=0))
            print(f"Set instrument on channel {PIANO_CHANNEL} to Acoustic Grand Piano.")
        except (OSError, IOError, mido.MidoError) as e:
            print(f"Warning: Could not open MIDI output port '{port_name}': {e}")
            print("Sound will not be played.")

    def on(self, pitch, vel, channel=PIANO_CHANNEL):
        """
        Play a note.
        """
        if self.port:
            msg = mido.Message('note_on', note=pitch, velocity=vel, channel=channel)
            self.port.send(msg)

    def off(self, pitch, channel=PIANO_CHANNEL):
        """
        Stop a note.
        """
        if self.port:
            msg = mido.Message('note_off', note=pitch, velocity=0, channel=channel)
            self.port.send(msg)

    def metro_first(self):
        """
        Play the first metronome note of each bar.
        """
        self.on(
            METRONOME_PITCH_FIRST,
            vel=METRONOME_VELOCITY_FIRST,
            channel=METRONOME_CHANNEL,
        )
        # Send a note_off immediately for a clean "click"
        time.sleep(0.01) # Small delay to ensure synth registers note_on
        self.off(METRONOME_PITCH_FIRST, channel=METRONOME_CHANNEL)

    def metro_other(self):
        """
        Play the other metronome notes of each bar.
        """
        self.on(
            METRONOME_PITCH_OTHER,
            vel=METRONOME_VELOCITY_OTHER,
            channel=METRONOME_CHANNEL,
        )
        # Send a note_off immediately for a clean "click"
        time.sleep(0.01) # Small delay to ensure synth registers note_on
        self.off(METRONOME_PITCH_OTHER, channel=METRONOME_CHANNEL)

    def close(self):
        """
        Close the audio output handler.
        """
        if self.port:
            self.port.reset()
            self.port.close()
        print("Audio output handler closed")

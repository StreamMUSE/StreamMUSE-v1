"""
This file contains the input handlers for the StreamMUSE client,
including MIDI device input and computer keyboard input.
"""

import time
import mido
from queue import Queue
from pynput import keyboard

# --- MIDI Input Handler ---
def read_midi_input(event_queue: Queue, device_name: str = None):
    """
    Worker function for reading MIDI input from a connected device (separate thread).
    """
    try:
        with mido.open_input(device_name) as port:
            print(f"Listening for MIDI input on '{port.name}'...")
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    event = {"type": "note_on", "pitch": msg.note, "velocity": msg.velocity, "time": time.time()}
                    event_queue.put(event)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    event = {"type": "note_off", "pitch": msg.note, "velocity": 0, "time": time.time()}
                    event_queue.put(event)
    except (OSError, IOError, mido.MidoError) as e:
        print(f"\nError opening MIDI input port: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        event_queue.put(None)


# --- Keyboard Input Handler ---
KEY_TO_PITCH = {
    # White keys (bottom row)
    'z': 60, 'x': 62, 'c': 64, 'v': 65, 'b': 67, 'n': 69, 'm': 71,
    ',': 72, '.': 74, '/': 76,
    # Black keys (top row)
    's': 61, 'd': 63, 'g': 66, 'h': 68, 'j': 70, 'l': 73, ';': 75,
}
VELOCITY = 100
pressed_keys = set()

def _on_press(key, event_queue):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key not in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.add(char_key)
            event = {"type": "note_on", "pitch": pitch, "velocity": VELOCITY, "time": time.time()}
            event_queue.put(event)
    except AttributeError:
        pass # Ignore special keys

def _on_release(key, event_queue):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.remove(char_key)
            event = {"type": "note_off", "pitch": pitch, "velocity": 0, "time": time.time()}
            event_queue.put(event)
    except AttributeError:
        if key == keyboard.Key.esc:
            # Stop listener
            event_queue.put(None)
            return False

def read_keyboard_input(event_queue: Queue):
    """
    Worker function for reading computer keyboard input (separate thread).
    Maps keyboard keys to MIDI notes.
    """
    print("Listening for computer keyboard input...")
    print("Mapped keys: 'zxcvbnm,' and 'sdghjl;'")
    print("Press 'ESC' to exit.")
    
    with keyboard.Listener(
            on_press=lambda key: _on_press(key, event_queue),
            on_release=lambda key: _on_release(key, event_queue)) as listener:
        listener.join() 
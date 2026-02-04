"""
This file contains the input handlers for the StreamMUSE client,
including MIDI device input, computer keyboard input, and MIDI file simulation.
"""

import time
import mido
from queue import Queue
import sys
import os

# Add the app directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from midi_utils import midi_to_note
except ImportError:
    from midi_utils_client import midi_to_note_client as midi_to_note

# --- MIDI Input Handler ---
def read_midi_input(event_queue: Queue, device_name: str = None, audio_handler=None, melody_channel: int = 0):
    """
    Worker function for reading MIDI input from a connected device (separate thread).
    If audio_handler is provided, plays notes immediately for low-latency feedback.
    """
    port = None
    try:
        port = mido.open_input(device_name)
        print(f"Listening for MIDI input on '{port.name}'...")
        
        # Use polling instead of blocking iteration
        while True:
            # Check for stop signal first
            if not event_queue.empty():
                try:
                    peek = event_queue.get_nowait()
                    if peek is None:
                        print("[DEBUG] MIDI input received stop signal")
                        break
                    else:
                        # Put it back if it's not None
                        event_queue.put(peek)
                except:
                    pass
            
            # Poll for MIDI messages with timeout
            msg = port.poll()
            if msg is None:
                # No message, sleep briefly and check stop signal again
                time.sleep(0.001)  # 1ms sleep to prevent busy-waiting
                continue
            
            if msg.type == 'note_on' and msg.velocity > 0:
                event = {"type": "note_on", "pitch": msg.note, "velocity": msg.velocity, "time": time.time()}
                event_queue.put(event)
                if audio_handler:
                    audio_handler.on(msg.note, msg.velocity, channel=melody_channel)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                event = {"type": "note_off", "pitch": msg.note, "velocity": 0, "time": time.time()}
                event_queue.put(event)
                if audio_handler:
                    audio_handler.off(msg.note, channel=melody_channel)
                    
    except (OSError, IOError, mido.MidoError) as e:
        print(f"\nError opening MIDI input port: {e}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nUnexpected error in MIDI input: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if port:
            print("[DEBUG] Closing MIDI input port...")
            try:
                port.close()
            except:
                pass
        print("[DEBUG] MIDI input thread exiting")
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

def _on_press(key, event_queue, keyboard, audio_handler=None, melody_channel=0):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key not in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.add(char_key)
            event = {"type": "note_on", "pitch": pitch, "velocity": VELOCITY, "time": time.time()}
            event_queue.put(event)
            if audio_handler:
                audio_handler.on(pitch, VELOCITY, channel=melody_channel)
    except AttributeError:
        pass # Ignore special keys

def _on_release(key, event_queue, keyboard, audio_handler=None, melody_channel=0):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.remove(char_key)
            event = {
                "type": "note_off",
                "pitch": pitch,
                "velocity": 0,
                "time": time.time(),
            }
            event_queue.put(event)
    except AttributeError:
        if key == keyboard.Key.esc:
            # Stop listener
            event_queue.put(None)

def read_keyboard_input(event_queue: Queue, audio_handler=None, melody_channel: int = 0):
    """
    Worker function for reading computer keyboard input (separate thread).
    Maps keyboard keys to MIDI notes.
    """
    try:
        from pynput import keyboard
    except ImportError as e:
        print(f"Error: Cannot import pynput for keyboard input: {e}")
        print("This feature requires a graphical environment and the pynput library.")
        print("Try using MIDI device input or MIDI file input instead.")
        event_queue.put(None)
        return
    
    print("Listening for computer keyboard input...")
    print("Mapped keys: 'zxcvbnm,' and 'sdghjl;'")
    print("Press 'ESC' to exit.")
    
    try:
        with keyboard.Listener(
                on_press=lambda key: _on_press(key, event_queue, keyboard, audio_handler, melody_channel),
                on_release=lambda key: _on_release(key, event_queue, keyboard, audio_handler, melody_channel)) as listener:
            listener.join()
    except Exception as e:
        print(f"Error starting keyboard listener: {e}")
        print("This feature requires a graphical environment.")
        event_queue.put(None)


# --- MIDI File Input Handler ---
def read_midi_file_input(
    event_queue: Queue,
    midi_file_path: str,
    current_tick_ref: dict,  # {'current_tick': value} - shared reference
    main_loop_tempo: float,
    main_loop_ticks_per_beat: int,
    delay_ticks: int = 0,
    skip_ticks: int = 0,  # Skip how many ticks, for injection
):
    """
    MIDI file input → event stream (note_on / note_off), no duration in events.

    Args:
        event_queue: Queue to put note events into
        midi_file_path: Path to the MIDI file
        current_tick_ref: Shared reference to current tick count from main loop
        main_loop_tempo: Tempo of the main loop (BPM)
        main_loop_ticks_per_beat: Ticks per beat in main loop
        delay_ticks: Number of ticks to delay before starting playback
        skip_ticks: Number of ticks to skip from the beginning of the MIDI file, it's for injection
    """
    try:
        print(f"Loading MIDI file: {midi_file_path}")

        beat_div = main_loop_ticks_per_beat

        notes, midi_resolution, max_tick = midi_to_note(
            midi_file_path,
            beat_div=beat_div,
            program=None
        )

        if not notes:
            print("No notes found in MIDI file")
            event_queue.put(None)
            return

        print(f"Loaded {len(notes)} notes from MIDI file")

        # Filter out the first skip_ticks and rebase tick to start from 0
        filtered_notes = []
        for note in notes:
            note_start = note["tick"]
            note_end = note["tick"] + note["duration"]

            # fully before skip -> ignore
            if note_end <= skip_ticks:
                continue

            adjusted = note.copy()

            # If the note starts before skip but ends after, clamp start to 0 and shorten duration
            if note_start < skip_ticks:
                adjusted["tick"] = 0
                adjusted["duration"] = note_end - skip_ticks
            else:
                adjusted["tick"] = note_start - skip_ticks

            filtered_notes.append(adjusted)

        if not filtered_notes:
            print(f"No notes found after skipping first {skip_ticks} ticks")
            event_queue.put(None)
            return

        print(
            f"After skipping first {skip_ticks} ticks: {len(filtered_notes)} notes remaining"
        )

        tick_schedule = {}
        start_offset_tick = delay_ticks

        if skip_ticks > 0:
            print(f"Skipped first {skip_ticks} ticks of MIDI file")
        if delay_ticks > 0:
            print(f"Delayed start by {delay_ticks} ticks")

        # Schedule note_on and note_off events (no duration field)
        for note in filtered_notes:
            onset_tick = note["tick"] + start_offset_tick
            offset_tick = onset_tick + int(note["duration"])

            # note_on
            tick_schedule.setdefault(onset_tick, []).append(
                {
                    "type": "note_on",
                    "pitch": int(note["pitch"]),
                    "velocity": int(note.get("velocity", 64)),
                    "tick": onset_tick,  # Include the scheduled tick for accurate recording
                    "time": time.time(),  # overwritten when sent
                }
            )

            # note_off
            tick_schedule.setdefault(offset_tick, []).append(
                {
                    "type": "note_off",
                    "pitch": int(note["pitch"]),
                    "velocity": 0,
                    "tick": offset_tick,  # Include the scheduled tick for accurate recording
                    "time": time.time(),  # overwritten when sent
                }
            )

        if tick_schedule:
            scheduled_ticks = sorted(tick_schedule.keys())
            print(
                f"Scheduled events from tick {scheduled_ticks[0]} to {scheduled_ticks[-1]} "
                f"(total events={sum(len(v) for v in tick_schedule.values())})"
            )

        last_tick = -1
        max_scheduled_tick = max(tick_schedule.keys()) if tick_schedule else 0

        while True:
            current_tick = current_tick_ref.get("current_tick", 0)

            if current_tick != last_tick and current_tick in tick_schedule:
                events = tick_schedule[current_tick]
                for event in events:
                    event["time"] = time.time()
                    event_queue.put(event)
                del tick_schedule[current_tick]

            last_tick = current_tick

            if not tick_schedule:
                print("MIDI file playback completed")
                break

            if current_tick > max_scheduled_tick + 50:
                print("MIDI file playback completed (main loop advanced past all events)")
                break

            time.sleep(0.001)

    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("MIDI file input handler finished, main loop continues") 
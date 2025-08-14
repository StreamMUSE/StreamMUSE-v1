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
from midi_input_script import midi_to_note

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

def _on_press(key, event_queue, keyboard):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key not in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.add(char_key)
            event = {"type": "note_on", "pitch": pitch, "velocity": VELOCITY, "time": time.time()}
            event_queue.put(event)
    except AttributeError:
        pass # Ignore special keys

def _on_release(key, event_queue, keyboard):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key in pressed_keys:
            pressed_keys.remove(char_key)
    except AttributeError:
        if key == keyboard.Key.esc:
            # Stop listener
            event_queue.put(None)

def read_keyboard_input(event_queue: Queue):
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
                on_press=lambda key: _on_press(key, event_queue, keyboard),
                on_release=lambda key: _on_release(key, event_queue, keyboard)) as listener:
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
    use_original_duration: bool = True,
    default_duration_ticks: int = 2
):
    """
    Worker function for reading MIDI file input and simulating user input (separate thread).
    
    Uses the existing midi_to_note function to parse MIDI files and convert them to 
    real-time note events that match the main loop's timing.
    
    Args:
        event_queue: Queue to put note events into
        midi_file_path: Path to the MIDI file
        current_tick_ref: Shared reference to current tick count from main loop
        main_loop_tempo: Tempo of the main loop (BPM)
        main_loop_ticks_per_beat: Ticks per beat in main loop
        delay_ticks: Number of ticks to delay before starting playback
        use_original_duration: If True, use original MIDI durations; if False, use fixed duration
        default_duration_ticks: Fixed duration when use_original_duration is False
    """
    try:
        print(f"Loading MIDI file: {midi_file_path}")
        
        # Use existing midi_to_note function to parse the MIDI file
        # The beat_div parameter controls the tempo conversion
        # beat_div=4 means quarter note = 1 beat (standard)
        beat_div = main_loop_ticks_per_beat
        
        # Parse MIDI file - this handles tempo conversion automatically
        notes, midi_resolution, max_tick = midi_to_note(
            midi_file_path, 
            beat_div=beat_div,
            program=None  # Accept all instruments, we'll filter for melody later
        )
        
        if not notes:
            print("No notes found in MIDI file")
            event_queue.put(None)
            return
        
        # Find melody track by selecting notes from the track with the most activity
        # Group notes by their timing characteristics to identify the main melody line
        print(f"Loaded {len(notes)} notes from MIDI file")
        
        # Create tick-indexed schedule with delay offset
        # No tempo conversion needed - the main loop's tempo will control playback speed
        tick_schedule = {}
        start_offset_tick = delay_ticks
        
        print(f"Scheduling notes with delay offset: {delay_ticks} ticks")
        
        for note in notes:
            # Use original tick timing + delay offset
            scheduled_tick = note['tick'] + start_offset_tick
            
            # Handle duration
            if use_original_duration:
                duration = note['duration']
            else:
                duration = default_duration_ticks
            
            # Schedule note_on event
            if scheduled_tick not in tick_schedule:
                tick_schedule[scheduled_tick] = []
            
            # Create event in the same format as other input handlers
            tick_schedule[scheduled_tick].append({
                'type': 'note_on',
                'pitch': note['pitch'],
                'velocity': 64,  # Default velocity for MIDI file notes
                'duration': duration,
                'time': time.time()  # Will be updated when actually sent
            })
        
        scheduled_ticks = sorted(tick_schedule.keys())
        print(f"Scheduled {len(notes)} notes from tick {scheduled_ticks[0]} to {scheduled_ticks[-1]}")
        if delay_ticks > 0:
            print(f"Delayed start by {delay_ticks} ticks")
        
        # Main playback loop - wait for the right tick and send events
        last_tick = -1
        while True:
            current_tick = current_tick_ref.get('current_tick', 0)
            
            # Check if we have events for this tick
            if current_tick != last_tick and current_tick in tick_schedule:
                events = tick_schedule[current_tick]
                for event in events:
                    # Update timestamp to current time
                    event['time'] = time.time()
                    event_queue.put(event)
                
                # Clean up processed events
                del tick_schedule[current_tick]
            
            last_tick = current_tick
            
            # Check if we're done (no more events scheduled)
            if not tick_schedule:
                print("MIDI file playback completed")
                break
            
            # Check if we've passed all scheduled events by a safe margin
            if tick_schedule and current_tick > max(tick_schedule.keys()) + 50:
                print("MIDI file playback completed (main loop advanced past all events)")
                break
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
            
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # MIDI file playback finished, but don't signal end of input
        # The main loop should continue running
        print("MIDI file input handler finished, main loop continues") 
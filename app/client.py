"""
This is the client side for the StreamMUSE end to end system.
"""

import sys
import os
import time
import requests
import mido
import threading
from queue import Queue
import argparse

from output_handlers.cli_output import CLIOutputHandler
from output_handlers.audio_output import AudioOutputHandler

def _midi_to_note_name(pitch: int) -> str:
    """
    Converts a MIDI pitch number (0-127) to its scientific pitch notation.
    e.g., 60 -> "C4", 69 -> "A4"
    """
    if not 0 <= pitch <= 127:
        return "N/A"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note_index = pitch % 12
    return f"{note_names[note_index]}{octave}"

def read_midi_input(event_queue: Queue, device_name: str = None):
    """
    Worker function for reading MIDI input (separate thread)
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
    except (OSError, IOError) as e:
        print(f"\nError opening MIDI input port: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        event_queue.put(None)

def inference_worker(request_queue: Queue, response_queue: Queue, server_url: str):
    """
    Worker function for sending requests to the server and receiving responses.
    """
    while True:
        request_data = request_queue.get()
        if request_data is None:
            break
        start_time = time.perf_counter()
        try:
            response = requests.post(server_url, json=request_data)
            response.raise_for_status()
            future_events = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error contacting server: {e}")
            future_events = []
        end_time = time.perf_counter()
        round_trip_time = end_time - start_time
        response_queue.put((future_events, round_trip_time))

def tick_loop(
    event_queue: Queue, 
    inference_request_queue: Queue, 
    inference_response_queue: Queue,
    output_handler: CLIOutputHandler, 
    audio_output_handler: AudioOutputHandler, 
    tempo: float, 
    ticks_per_beat: int, 
    beats_per_bar: int, 
    user_input_history: list,
    metronome_enabled: bool
):
    """
    Main tick loop for the client. (main thread)
    """
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    notes_played_since_last_request = []

    all_user_events = []
    all_model_events = []

    while True:
        tick_count += 1
        user_events_this_tick = []
        model_events_this_tick = []

        # Get all user input from queue and play it
        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                inference_request_queue.put(None)
                return
            
            user_input_history.append(event)
            user_events_this_tick.append(event)

            if event['type'] == 'note_on':
                audio_output_handler.on(event['pitch'], event['velocity'])
            elif event['type'] == 'note_off':
                audio_output_handler.off(event['pitch'])
        
        if len(user_events_this_tick) > 0:
            all_user_events.extend(user_events_this_tick)

        user_notes_for_display = [e for e in user_events_this_tick if e['type'] == 'note_on']
                    



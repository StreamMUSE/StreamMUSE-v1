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
from output_handlers.midi_file_handler import MidiFileHandler
from output_handlers.json_log_handler import JsonLogHandler
from input_handlers.input_handler import read_midi_input, read_keyboard_input

# --- Constants ---
DEFAULT_NOTE_DURATION_TICKS = 4
LATENCY_OFFSET_TICKS = 1

def inference_worker(request_queue: Queue, response_queue: Queue, server_url: str):
    """
    Worker function for sending requests to the server and receiving responses.
    """
    while True:
        queue_item = request_queue.get()
        if queue_item is None:
            break
        
        request_data, full_request_dict = queue_item

        # The request_data is now expected to be a dictionary for the InferenceRequest model
        start_time = time.perf_counter()
        try:
            response = requests.post(server_url, json=request_data)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error contacting server: {e}")
            response_json = None # Indicate failure
        
        end_time = time.perf_counter()
        round_trip_time = end_time - start_time
        
        # Pass the full response and timing info back
        response_queue.put((response_json, round_trip_time, full_request_dict))

def tick_loop(
    event_queue: Queue, 
    inference_request_queue: Queue, 
    inference_response_queue: Queue,
    output_handler: CLIOutputHandler, 
    audio_output_handler: AudioOutputHandler, 
    midi_file_handler: MidiFileHandler,
    json_log_handler: JsonLogHandler,
    tempo: float, 
    ticks_per_beat: int, 
    beats_per_bar: int, 
    all_timing_data: list, # Pass list in to be mutated
    metronome_enabled: bool
):
    """
    Main tick loop for the client. (main thread)
    """
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    
    # New state variables
    active_notes = {} # For tracking note durations if we need to in the future
    notes_for_next_request = []
    last_inference_timings = {} # To persist timing info for display
    ticks_per_bar = ticks_per_beat * beats_per_bar

    # --- Main Loop ---
    while True:
        tick_count += 1
        
        # --- 1. Process User Input ---
        user_notes_this_tick = []
        timings_this_tick = {} # Process timings on a per-tick basis

        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                inference_request_queue.put(None)
                return
            
            # --- Note Quantization ---
            if event['type'] == 'note_on':
                # For now, we use a fixed duration as requested.
                # In the future, we could track note_off here.
                quantized_note = {
                    "pitch": event['pitch'],
                    "tick": tick_count,
                    "duration": DEFAULT_NOTE_DURATION_TICKS
                }
                notes_for_next_request.append(quantized_note)
                user_notes_this_tick.append(quantized_note) # Add to tick-specific list
                midi_file_handler.add_user_note(quantized_note) # Log user note
                audio_output_handler.on(event['pitch'], event['velocity'])
            elif event['type'] == 'note_off':
                audio_output_handler.off(event['pitch'])
        
        # --- 2. Handle Inference Responses ---
        while not inference_response_queue.empty():
            response_data, round_trip_time, request_data = inference_response_queue.get()
            
            if response_data:
                # --- Log the complete inference event ---
                json_log_handler.log_inference_event(request_data, response_data)

                # --- Tick Consistency Filter ---
                generation_start_tick = response_data['generation_start_tick']
                newly_generated_notes = response_data['accompaniment']

                # --- Clear stale notes from the previous generation ---
                # This ensures that if a new response arrives before the old one is
                # fully played out, we replace the future notes with the new ones.
                ticks_to_clear = [t for t in playback_schedule if t >= generation_start_tick]
                for t in ticks_to_clear:
                    # In the current design, any scheduled event is a model-generated note_on.
                    # A more complex design might require tagging events with their source.
                    del playback_schedule[t]
                
                # --- Schedule new notes ---
                for note in newly_generated_notes:
                    if note['tick'] >= tick_count:
                        if note['tick'] not in playback_schedule:
                            playback_schedule[note['tick']] = []
                        playback_schedule[note['tick']].append(note)
                
                # Store timings for display, making them persistent
                last_inference_timings = response_data['timings']
                last_inference_timings['round_trip_time'] = round_trip_time
                all_timing_data.append(last_inference_timings) # Add full dict to benchmark list

        # --- 3. Trigger New Inference (Latency-Aware) ---
        is_trigger_tick = (tick_count % ticks_per_bar) == (ticks_per_bar - LATENCY_OFFSET_TICKS)
        
        if is_trigger_tick and notes_for_next_request:
            # The model should start generating from the beginning of the *next* bar.
            current_bar_start_tick = (tick_count // ticks_per_bar) * ticks_per_bar
            next_bar_start_tick = current_bar_start_tick + ticks_per_bar

            request_data = {
                "melody_notes": notes_for_next_request,
                "generation_start_tick": next_bar_start_tick
            }
            inference_request_queue.put((request_data, request_data)) # Pass it twice for logging
            notes_for_next_request = [] # Clear the buffer

        # --- 4. Play Scheduled Notes ---
        notes_to_play_this_tick = []
        notes_to_stop_this_tick = []
        
        # Check the schedule for events supposed to happen on the current tick
        scheduled_events = playback_schedule.pop(tick_count, [])
        for event in scheduled_events:
            if event.get('type') == 'note_off':
                notes_to_stop_this_tick.append(event)
            else: # It's a note_on
                notes_to_play_this_tick.append(event)
        
        # Process note-offs first
        for event in notes_to_stop_this_tick:
            audio_output_handler.off(event['pitch'])

        # Process note-ons and schedule their corresponding note-offs
        for event in notes_to_play_this_tick:
            audio_output_handler.on(event['pitch'], 100) # Use a fixed velocity for generated notes
            midi_file_handler.add_model_note(event) # Log model note
            
            note_off_tick = tick_count + event['duration']
            if note_off_tick not in playback_schedule:
                playback_schedule[note_off_tick] = []
            
            playback_schedule[note_off_tick].append({**event, 'type': 'note_off'})

        # --- 5. Metronome ---
        if metronome_enabled:
            is_beat_tick = (tick_count % ticks_per_beat) == 0
            if is_beat_tick:
                beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
                if beat_in_bar == 0:
                    audio_output_handler.metro_first()
                else:
                    audio_output_handler.metro_other()

        # --- 6. Update Display ---
        bar_count = tick_count // ticks_per_bar
        beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
        
        pending_user_notes_display = [n['pitch'] for n in notes_for_next_request]
        user_notes_this_tick_display = [n['pitch'] for n in user_notes_this_tick]
        model_notes_for_display = [n['pitch'] for n in notes_to_play_this_tick]

        music_info = {
            "bar": bar_count,
            "beat": beat_in_bar,
            "ticks_per_beat": ticks_per_beat,
            "inference_triggered": is_trigger_tick and bool(notes_for_next_request),
            "all_timing_data": all_timing_data
        }
        music_info.update(last_inference_timings)
        
        output_handler.update_and_display(
            tick_count,
            music_info,
            user_notes_this_tick_display,
            model_notes_for_display,
            pending_user_notes_display
        )

        # --- 7. Sleep ---
        time.sleep(seconds_per_tick)

def main():
    SERVER_URL = "http://localhost:8000/generate_accompaniment"
    TEMPO = 120.0
    TICKS_PER_BEAT = 4
    BEATS_PER_BAR = 4

    parser = argparse.ArgumentParser(description="StreamMUSE Client")
    parser.add_argument("--server_url", type=str, default=SERVER_URL)
    parser.add_argument("--tempo", type=float, default=TEMPO)
    parser.add_argument("--ticks_per_beat", type=int, default=TICKS_PER_BEAT)
    parser.add_argument("--beats_per_bar", type=int, default=BEATS_PER_BAR)
    parser.add_argument("--log_lines", type=int, default=10)
    parser.add_argument("--metronome", action="store_true", help="Enable an audible MIDI metronome click.")
    parser.add_argument("--midi_output_name", type=str, default=None, help="Specify the MIDI output port name.")
    parser.add_argument("--midi_input_name", type=str, default=None, help="Specify the MIDI input port name.")
    parser.add_argument("--use-keyboard-input", action="store_true", help="Use the computer keyboard as MIDI input.")
    args = parser.parse_args()

    # --- Create Session Log Directory ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_log_dir = os.path.join("app", "logs", f"session_{timestamp}")
    os.makedirs(session_log_dir, exist_ok=True)

    event_queue = Queue()
    inference_request_queue = Queue()
    inference_response_queue = Queue()
    audio_output_handler = AudioOutputHandler(args.midi_output_name)
    output_handler = CLIOutputHandler(args.log_lines)
    midi_file_handler = MidiFileHandler(args.tempo, args.ticks_per_beat)
    json_log_handler = JsonLogHandler()
    all_timing_data = [] # Initialize list in main scope

    if args.use_keyboard_input:
        input_thread = threading.Thread(target=read_keyboard_input, args=(event_queue,), daemon=True)
    else:
        # A check to see if MIDI input is available.
        try:
            if not mido.get_input_names():
                print("No MIDI input devices found. Please connect a MIDI device or use --use-keyboard-input.")
                return
            midi_input_name = args.midi_input_name or mido.get_input_names()[0]
        except Exception as e:
            print(f"Could not list MIDI devices: {e}")
            return

        input_thread = threading.Thread(
            target=read_midi_input,
            args=(event_queue, midi_input_name),
            daemon=True
        )

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(inference_request_queue, inference_response_queue, args.server_url),
        daemon=True
    )
    music_pacer_thread = threading.Thread(
        target=tick_loop,
        args=(
            event_queue,
            inference_request_queue,
            inference_response_queue,
            output_handler,
            audio_output_handler,
            midi_file_handler,
            json_log_handler,
            args.tempo,
            args.ticks_per_beat,
            args.beats_per_bar,
            all_timing_data,
            args.metronome),
        daemon=True
    )

    print('Starting StreamMUSE Client')
    print(f'Connecting to server at {args.server_url}')

    try:
        input_thread.start()
        inference_thread.start()
        music_pacer_thread.start()
        
        # Keep the main thread alive to catch KeyboardInterrupt
        while input_thread.is_alive() and music_pacer_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\r\nCtrl+C detected. Exiting application.")
    finally:
        print("\n--- Saving all session logs ---")
        # Pass the benchmark data to be saved
        output_handler.save_log_on_exit(session_log_dir, all_timing_data)
        midi_file_handler.save_to_midi(session_log_dir)
        json_log_handler.save_logs(session_log_dir)
        audio_output_handler.close()

if __name__ == "__main__":
    main()
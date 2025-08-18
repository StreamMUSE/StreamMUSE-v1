"""
StreamMUSE Client Application

This is the client side for the StreamMUSE end-to-end real-time music generation system.
It provides multiple input modes (MIDI devices, computer keyboard, MIDI files) and
communicates with a StreamMUSE server to generate musical accompaniment in real-time.

Usage:
    python client.py [options]
    
Input modes:
    - MIDI device (default): Uses connected MIDI input device
    - Keyboard: python client.py --use-keyboard-input
    - MIDI file: python client.py --midi-file-input path/to/file.mid
"""

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
from input_handlers.input_handler import read_midi_input, read_keyboard_input, read_midi_file_input

# --- Configuration ---
class StreamMUSEConfig:
    """Configuration settings for StreamMUSE client"""
    
    # Network
    DEFAULT_SERVER_URL = "http://localhost:8000/generate_accompaniment"
    
    # Musical timing
    DEFAULT_TEMPO = 90.0
    DEFAULT_TICKS_PER_BEAT = 4
    DEFAULT_BEATS_PER_BAR = 4
    DEFAULT_GENERATION_INTERVAL_TICKS = 2
    
    # Note handling
    DEFAULT_NOTE_DURATION_TICKS = 2
    LATENCY_OFFSET_TICKS = 2
    DEFAULT_ACCOMPANIMENT_VELOCITY = 50
    
    # Display
    DEFAULT_LOG_LINES = 10
    
    # MIDI File Input
    DEFAULT_MIDI_FILE_DELAY_TICKS = 0
    
    @staticmethod
    def validate_args(args):
        """Validate command line arguments."""
        if args.accompaniment_velocity < 0 or args.accompaniment_velocity > 127:
            print("Error: accompaniment-velocity must be between 0 and 127")
            return False
        
        if args.tempo <= 0:
            print("Error: tempo must be positive")
            return False
            
        if args.ticks_per_beat <= 0:
            print("Error: ticks_per_beat must be positive")
            return False
            
        if args.beats_per_bar <= 0:
            print("Error: beats_per_bar must be positive")
            return False
            
        if args.generation_interval_ticks <= 0:
            print("Error: generation_interval_ticks must be positive")
            return False
            
        if args.midi_file_delay_ticks < 0:
            print("Error: midi_file_delay_ticks must be non-negative")
            return False
        
        return True

# --- Constants ---
DEFAULT_NOTE_DURATION_TICKS = StreamMUSEConfig.DEFAULT_NOTE_DURATION_TICKS
LATENCY_OFFSET_TICKS = StreamMUSEConfig.LATENCY_OFFSET_TICKS

def inference_worker(request_queue: Queue, response_queue: Queue, server_url: str):
    """
    Worker function for sending requests to the server and receiving responses.
    """
    while True:
        queue_item = request_queue.get()
        if queue_item is None:
            break
        
        request_data, full_request_dict = queue_item

        # Add timestamp right before sending
        client_send_time = time.perf_counter()
        request_data['client_request_send_time'] = client_send_time
        full_request_dict['client_request_send_time'] = client_send_time # For logging

        start_time = client_send_time
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
    metronome_enabled: bool,
    generation_interval_ticks: int,
    current_tick_ref: dict = None  # Optional shared tick reference for MIDI file input
):
    """
    Main tick loop for the client. (main thread)
    """
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    
    # New state variables
    notes_for_next_request = []
    last_inference_timings = {} # To persist timing info for display
    ticks_per_bar = ticks_per_beat * beats_per_bar

    # --- Main Loop ---
    while True:
        tick_count += 1
        
        # Update shared tick reference for MIDI file input
        if current_tick_ref is not None:
            current_tick_ref['current_tick'] = tick_count
        
        # --- 1. Process User Input ---
        user_notes_this_tick = []

        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                inference_request_queue.put(None)
                return
            
            # --- Note Quantization & Audio Playback ---
            if event['type'] == 'note_on':
                # 1. Quantize the note for the inference engine request.
                # All user notes are given a fixed duration for the model.
                quantized_note = {
                    "pitch": event['pitch'],
                    "tick": tick_count,
                    "duration": DEFAULT_NOTE_DURATION_TICKS
                }
                notes_for_next_request.append(quantized_note)
                user_notes_this_tick.append(quantized_note)
                midi_file_handler.add_user_note(quantized_note)

                # 2. Play the note immediately for audio feedback.
                audio_output_handler.on(event['pitch'], event['velocity'])

            elif event['type'] == 'note_off':
                # Pass the note_off event directly to the audio handler
                audio_output_handler.off(event['pitch'])
        
        # --- 2. Handle Inference Responses ---
        while not inference_response_queue.empty():
            response_data, round_trip_time, request_data = inference_response_queue.get()
            
            if response_data:
                # --- Log the complete inference event ---
                json_log_handler.log_inference_event(request_data, response_data)

                # --- Calculate and store all timing information ---
                timings = response_data['timings']
                timings['round_trip_time'] = round_trip_time

                # Calculate server processing duration (this is accurate as it uses one clock)
                server_arrival_time = timings['request_arrival_time']
                server_response_time = timings['response_output_time']
                server_processing_duration = server_response_time - server_arrival_time
                timings['server_processing_duration'] = server_processing_duration

                # Calculate total network latency (accurate)
                # This is the time spent on the network for both the request and response.
                timings['total_network_latency'] = round_trip_time - server_processing_duration

                all_timing_data.append(timings)
                
                # --- Tick Consistency Filter ---
                newly_generated_notes = response_data['accompaniment']

                # --- Clear stale notes from the previous generation ---
                # This ensures that if a new response arrives before the old one is
                # fully played out, we only replace future model-generated notes.
                # User-played note_offs are preserved.
                if newly_generated_notes:
                    # Find the first tick where the new generation actually places a note.
                    # This prevents clearing old notes if there's a gap before the new music starts.
                    first_new_note_tick = min(note['tick'] for note in newly_generated_notes)

                    ticks_to_clean = [t for t in playback_schedule if t >= first_new_note_tick]
                    for tick in ticks_to_clean:
                        # Filter out events sourced from the model, keep user events
                        playback_schedule[tick] = [
                            event for event in playback_schedule[tick] if event.get("source") != "model"
                        ]
                        # If the tick is now empty, remove it from the schedule
                        if not playback_schedule[tick]:
                            del playback_schedule[tick]
                
                # --- Schedule new notes ---
                for note in newly_generated_notes:
                    if note['tick'] >= tick_count:
                        if note['tick'] not in playback_schedule:
                            playback_schedule[note['tick']] = []
                        # Tag as a model-originated event
                        playback_schedule[note['tick']].append({**note, "source": "model"})
                
                # Store timings for display, making them persistent
                last_inference_timings = timings

        # --- 3. Trigger New Inference (Latency-Aware) ---
        is_trigger_tick = (tick_count % generation_interval_ticks) == (generation_interval_ticks - LATENCY_OFFSET_TICKS)
        
        if is_trigger_tick:# and notes_for_next_request:
            # The model should start generating from the beginning of the *next* generation interval.
            # current_interval_start_tick = (tick_count // generation_interval_ticks) * generation_interval_ticks
            # next_interval_start_tick = current_interval_start_tick + generation_interval_ticks
            next_interval_start_tick = tick_count + 1

            request_data = {
                "melody_notes": notes_for_next_request,
                "generation_start_tick": next_interval_start_tick
            }
            inference_request_queue.put((request_data, request_data.copy())) # Pass a copy for logging
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
            # This loop only processes model-generated notes.
            audio_output_handler.on(event['pitch'], audio_output_handler.accompaniment_velocity)
            midi_file_handler.add_model_note(event)
            
            note_off_tick = tick_count + event['duration']
            if note_off_tick not in playback_schedule:
                playback_schedule[note_off_tick] = []
            
            # The source tag is preserved from the original event
            playback_schedule[note_off_tick].append({**event, 'type': 'note_off', "source": "model"})

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
            "all_timing_data": all_timing_data,
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
    config = StreamMUSEConfig()
    
    parser = argparse.ArgumentParser(
        description="StreamMUSE Client - Real-time music generation with AI accompaniment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Input Modes:
            Default:     Use connected MIDI device
            Keyboard:    --use-keyboard-input
            MIDI file:   --midi-file-input path/to/file.mid

            Examples:
            %(prog)s
            %(prog)s --use-keyboard-input --tempo 120
            %(prog)s --midi-file-input song.mid --midi-file-delay-ticks 8
        """
    )
    
    # Network arguments
    parser.add_argument("--server_url", type=str, default=config.DEFAULT_SERVER_URL,
                       help="URL of the StreamMUSE server")
    
    # Musical timing arguments
    parser.add_argument("--tempo", type=float, default=config.DEFAULT_TEMPO,
                       help="Tempo in BPM")
    parser.add_argument("--ticks_per_beat", type=int, default=config.DEFAULT_TICKS_PER_BEAT,
                       help="Number of ticks per beat")
    parser.add_argument("--beats_per_bar", type=int, default=config.DEFAULT_BEATS_PER_BAR,
                       help="Number of beats per bar")
    parser.add_argument("--generation_interval_ticks", type=int, default=config.DEFAULT_GENERATION_INTERVAL_TICKS,
                       help="Number of ticks between generation requests")
    
    # Display arguments
    parser.add_argument("--log_lines", type=int, default=config.DEFAULT_LOG_LINES,
                       help="Number of log lines to display")
    parser.add_argument("--metronome", action="store_true", 
                       help="Enable audible MIDI metronome click")
    
    # MIDI I/O arguments
    parser.add_argument("--midi_output_name", type=str, default=None,
                       help="Specify MIDI output port name")
    parser.add_argument("--midi_input_name", type=str, default=None,
                       help="Specify MIDI input port name")
    parser.add_argument("--accompaniment-velocity", type=int, default=config.DEFAULT_ACCOMPANIMENT_VELOCITY,
                       help="MIDI velocity for generated accompaniment notes (0-127)")
    
    # Input mode arguments
    parser.add_argument("--use-keyboard-input", action="store_true",
                       help="Use computer keyboard as MIDI input")
    parser.add_argument("--midi-file-input", type=str, default=None,
                       help="Path to MIDI file to simulate user input")
    parser.add_argument("--midi-file-delay-ticks", type=int, default=config.DEFAULT_MIDI_FILE_DELAY_TICKS,
                       help="Number of ticks to delay before MIDI file starts playing")
    parser.add_argument("--midi-file-use-original-duration", action="store_true",
                       help="Use original MIDI note durations instead of fixed duration")
    
    args = parser.parse_args()

    # --- Validate Arguments ---
    if not StreamMUSEConfig.validate_args(args):
        return

    # --- Create Session Log Directory ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_log_dir = os.path.join("app", "logs", f"session_{timestamp}")
    os.makedirs(session_log_dir, exist_ok=True)

    event_queue = Queue()
    inference_request_queue = Queue()
    inference_response_queue = Queue()
    audio_output_handler = AudioOutputHandler(
        port_name=args.midi_output_name, 
        accompaniment_velocity=args.accompaniment_velocity
    )
    output_handler = CLIOutputHandler(args.log_lines)
    midi_file_handler = MidiFileHandler(args.tempo, args.ticks_per_beat)
    json_log_handler = JsonLogHandler()
    all_timing_data = [] # Initialize list in main scope

    # Create shared reference for current tick count (for MIDI file input)
    current_tick_ref = {'current_tick': 0}

    if args.midi_file_input:
        # MIDI file input mode
        print(f"Using MIDI file input: {args.midi_file_input}")
        if args.midi_file_delay_ticks > 0:
            print(f"MIDI file will start after {args.midi_file_delay_ticks} ticks delay")
        
        input_thread = threading.Thread(
            target=read_midi_file_input,
            args=(
                event_queue,
                args.midi_file_input,
                current_tick_ref,
                args.tempo,
                args.ticks_per_beat,
                args.midi_file_delay_ticks,
                args.midi_file_use_original_duration,
                DEFAULT_NOTE_DURATION_TICKS
            ),
            daemon=True
        )
    elif args.use_keyboard_input:
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
            args.metronome,
            args.generation_interval_ticks,
            current_tick_ref),
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
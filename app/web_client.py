"""
StreamMUSE Web Client Server

FastAPI server that:
1. Serves the web UI static files
2. Provides WebSocket endpoint for real-time updates
3. Manages client lifecycle (start/stop/restart)
4. Bridges existing client logic to browser

Usage:
    python web_client.py
    # Open http://localhost:8080 in browser
"""

import os
import sys
import asyncio
import threading
import time
import argparse
from queue import Queue
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from output_handlers.websocket_output import WebSocketOutputHandler
from output_handlers.audio_output import AudioOutputHandler
from output_handlers.cli_output import CLIOutputHandler
from output_handlers.midi_file_handler import MidiFileHandler
from output_handlers.json_log_handler import JsonLogHandler
from input_handlers.input_handler import (
    read_midi_input,
    read_keyboard_input,
    read_midi_file_input,
)

try:
    from key_detection import detect_key_lightweight, detect_key_music21
    from prompt_library import PromptLibrary
    LISTENING_MODE_AVAILABLE = True
    print("[INIT] Listening mode dependencies loaded successfully")
except ImportError as e:
    LISTENING_MODE_AVAILABLE = False
    detect_key_lightweight = None
    detect_key_music21 = None
    PromptLibrary = None
    print(f"[INIT] Listening mode dependencies NOT available: {e}")


# Helper function to convert duration-based notes to event-stream format
def notes_to_events(notes: list) -> list:
    """
    Convert duration-based notes to event-stream format.

    Input: [{"pitch": 60, "tick": 0, "duration": 4}, ...]
    Output: [{"type": "note_on", "pitch": 60, "tick": 0},
             {"type": "note_off", "pitch": 60, "tick": 4}, ...]
    """
    events = []
    for n in notes:
        pitch = n["pitch"]
        tick = n["tick"]
        duration = n.get("duration", 1)
        events.append({"type": "note_on", "pitch": pitch, "tick": tick})
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration})
    # Sort by tick, note_off before note_on if same tick
    events.sort(key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1))
    return events


class ClientConfig(BaseModel):
    server_url: str = "http://localhost:8988/generate_accompaniment"

    tempo: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
    # NOTE: Engine generates per-beat, so interval should equal ticks_per_beat
    generation_interval_ticks: int = 4
    generation_length_per_request: int = 5

    # Deprecated: event-stream protocol, kept for legacy compatibility
    note_duration_ticks: int = 2
    accompaniment_velocity: int = 50
    melody_channel: int = 0
    accompaniment_channel: int = 0

    midi_file_delay_ticks: int = 0
    listening_duration_ticks: int = 0

    input_mode: str = "keyboard"
    midi_file_path: Optional[str] = None
    midi_input_name: Optional[str] = None
    midi_output_name: Optional[str] = None
    metronome: bool = True

    listening_mode: str = "auto"  # "auto" or "manual"
    manual_prompt_path: Optional[str] = None
    key_detection_method: str = "lightweight"
    prompt_dir: Optional[str] = None

    # Recording options
    record_session: bool = True
    save_json_log: bool = True


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


def inject_notes_to_server(
    server_url: str,
    melody_notes: list,
    accompaniment_notes: list,
    injection_length_ticks: int,
    midi_file_handler=None,
) -> bool:
    import requests

    injection_url = server_url.replace("/generate_accompaniment", "/inject_notes")
    status_url = server_url.replace("/generate_accompaniment", "/injection_status")

    try:
        # Convert duration-based notes to event-stream format *only if needed*.
        #
        # IMPORTANT: In this codebase, user melody collected during listening mode is
        # already event-stream (`{"type": "note_on"|"note_off", ...}`), while prompts
        # loaded from MIDI are usually duration-notes (`{"pitch","tick","duration"}`).
        #
        # If we re-convert event-stream melody with `notes_to_events`, we'd destroy
        # real note lengths (everything becomes duration=1 tick).
        def ensure_event_stream(items: list) -> list:
            if not items:
                return []
            # Heuristic: event-stream entries carry a "type" field.
            if isinstance(items[0], dict) and "type" in items[0]:
                return items
            return notes_to_events(items)

        melody_events = ensure_event_stream(melody_notes)
        accompaniment_events = ensure_event_stream(accompaniment_notes)

        request_data = {
            "melody_notes": melody_events,
            "accompaniment_notes": accompaniment_events,
            "injection_length_ticks": injection_length_ticks
        }
        print(f"Injecting {len(melody_events)} melody events and {len(accompaniment_events)} accompaniment events...")
        response = requests.post(injection_url, json=request_data, timeout=5.0)
        response.raise_for_status()
        result = response.json()

        if result["success"]:
            print(f"Injection successful: {result['message']}")

            # Query injection status to get the actual injected notes
            try:
                status_response = requests.get(status_url, timeout=5.0)
                status_response.raise_for_status()
                injection_state = status_response.json()

                if injection_state.get("is_injected") and midi_file_handler:
                    print("Recording injected notes to MIDI file...")

                    # Get melody and accompaniment notes from injection state
                    injected_melody = injection_state.get("melody_notes", [])
                    injected_acc = injection_state.get("accompaniment_notes", [])

                    # Record melody events (already in event-stream format from server)
                    for event in injected_melody:
                        midi_file_handler.add_user_note(event)

                    # Convert accompaniment notes (duration-based from server) to events
                    acc_events = notes_to_events(injected_acc)
                    for event in acc_events:
                        midi_file_handler.add_model_note(event)

                    print(f"✓ Recorded {len(injected_melody)} melody events and {len(acc_events)} acc events to MIDI")

            except Exception as e:
                print(f"Warning: Could not query injection status or record notes: {e}")

            return True
        else:
            print(f"Injection failed: {result['message']}")
            return False
    except Exception as e:
        print(f"Injection request failed: {e}")
        return False


def clear_history_on_server(server_url: str):
    """Best-effort clearing of server-side history/injection state."""
    import requests
    clear_url = server_url.replace("/generate_accompaniment", "/clear_history")
    try:
        resp = requests.post(clear_url, timeout=3.0)
        resp.raise_for_status()
        print(f"[DEBUG] Server history cleared via {clear_url}")
    except Exception as e:
        # Don't block client start if server is unreachable
        print(f"[WARN] Could not clear server history ({clear_url}): {e}")


def manual_injection_mode_worker(
    collected_melody_notes: list,
    listening_duration_ticks: int,
    manual_prompt_path: str,
    server_url: str,
    result_queue: Queue,
    midi_file_handler=None,
):
    """
    Manual injection worker - loads a user-selected accompaniment file and injects it.

    Args:
        collected_melody_notes: Notes collected during listening period
        listening_duration_ticks: Duration of listening period
        manual_prompt_path: Path to the accompaniment MIDI file
        server_url: Server URL
        result_queue: Queue to put result (True/False) when done
        midi_file_handler: Optional MIDI file handler for recording
    """
    try:
        print(f"\n{'='*60}")
        print(f"MANUAL INJECTION MODE - Loading {manual_prompt_path}")
        print(f"{'='*60}")

        start_time = time.perf_counter()

        # Load accompaniment notes from the selected file
        from midi_utils import midi_to_note

        load_start_time = time.perf_counter()
        accompaniment_notes, _, _ = midi_to_note(manual_prompt_path, max_tick=listening_duration_ticks)

        # Filter to injection length and add program field
        accompaniment_notes = [
            {**n, 'program': 1} if 'program' not in n else n
            for n in accompaniment_notes if n['tick'] < listening_duration_ticks
        ]
        load_time = time.perf_counter() - load_start_time

        if not accompaniment_notes:
            print("No accompaniment notes loaded from file")
            result_queue.put(False)
            return

        print(f"Loaded {len(accompaniment_notes)} accompaniment notes (took {load_time*1000:.1f}ms)")

        # Inject to server
        inject_start_time = time.perf_counter()
        success = inject_notes_to_server(
            server_url,
            collected_melody_notes,
            accompaniment_notes,
            listening_duration_ticks,
            midi_file_handler=midi_file_handler,
        )
        inject_time = time.perf_counter() - inject_start_time

        total_time = time.perf_counter() - start_time

        if success:
            print(f"Injection complete (took {inject_time*1000:.1f}ms)")
            print(f"TOTAL PROCESSING TIME: {total_time*1000:.1f}ms")
            print(f"{'='*60}\n")
            result_queue.put(True)
        else:
            print(f"Injection failed")
            result_queue.put(False)

    except Exception as e:
        print(f"Manual injection worker error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(False)


def listening_mode_worker(
    collected_melody_notes: list,
    listening_duration_ticks: int,
    prompt_library: PromptLibrary,
    server_url: str,
    key_detection_method: str,
    result_queue: Queue,
    midi_file_handler=None,
):
    """
    Listening mode worker - detects key and selects appropriate prompt.

    Args:
        collected_melody_notes: Notes collected during listening period
        listening_duration_ticks: Duration of listening period
        prompt_library: Prompt library instance
        server_url: Server URL
        key_detection_method: "lightweight" or "music21"
        result_queue: Queue to put result (True/False) when done
        midi_file_handler: Optional MIDI file handler for recording
    """
    try:
        print(f"\n{'='*60}")
        print(f"LISTENING MODE WORKER - Analyzing {len(collected_melody_notes)} notes")
        print(f"{'='*60}")

        start_time = time.perf_counter()
        if key_detection_method == "music21":
            detected_key = detect_key_music21(collected_melody_notes)
        else:
            detected_key = detect_key_lightweight(collected_melody_notes)
        key_detection_time = time.perf_counter() - start_time

        print(f"Detected key: {detected_key} (took {key_detection_time*1000:.1f}ms)")

        selected_prompt = prompt_library.select_prompt(detected_key, strategy="random")

        if not selected_prompt:
            print("No prompt available, continuing without injection")
            result_queue.put(False)
            return

        print(f"Selected prompt: {selected_prompt.get('name', 'unknown')}")

        load_start_time = time.perf_counter()
        _, accompaniment_notes = prompt_library.load_prompt_notes(
            selected_prompt,
            max_ticks=listening_duration_ticks,
            load_melody=False,
            load_accompaniment=True
        )
        load_time = time.perf_counter() - load_start_time

        if not accompaniment_notes:
            print("No accompaniment notes loaded from prompt")
            result_queue.put(False)
            return

        print(f"Loaded {len(accompaniment_notes)} accompaniment notes (took {load_time*1000:.1f}ms)")

        inject_start_time = time.perf_counter()
        success = inject_notes_to_server(
            server_url,
            collected_melody_notes,
            accompaniment_notes,
            listening_duration_ticks,
            midi_file_handler=midi_file_handler,
        )
        inject_time = time.perf_counter() - inject_start_time

        total_time = time.perf_counter() - start_time

        if success:
            print(f"Injection complete (took {inject_time*1000:.1f}ms)")
            print(f"TOTAL PROCESSING TIME: {total_time*1000:.1f}ms")
            print(f"{'='*60}\n")
            result_queue.put(True)
        else:
            print(f"Injection failed")
            result_queue.put(False)

    except Exception as e:
        print(f"Listening mode worker error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(False)


class ClientManager:
    """Manages the StreamMUSE client lifecycle."""

    # Match client_lekai.py defaults
    DEFAULT_NOTE_DURATION_TICKS = 2
    LATENCY_OFFSET_TICKS = 2
    GRACE_PERIOD_TICKS = 8

    def __init__(self, ws_handler: WebSocketOutputHandler):
        self.ws_handler = ws_handler
        self.config = ClientConfig()
        self.is_running = False
        self.stop_event = threading.Event()

        self.event_queue: Optional[Queue] = None
        self.inference_request_queue: Optional[Queue] = None
        self.inference_response_queue: Optional[Queue] = None

        self.input_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.tick_thread: Optional[threading.Thread] = None

        self.audio_output_handler: Optional[AudioOutputHandler] = None
        self.midi_file_handler: Optional[MidiFileHandler] = None
        self.json_log_handler: Optional[JsonLogHandler] = None
        self.all_timing_data = []
        self.tick_history = []
        self.session_log_dir: Optional[str] = None
    
    def start(self, config: Optional[ClientConfig] = None):
        """Start the client with given config."""
        if self.is_running:
            return False
        
        if config:
            self.config = config
        
        print(f"Starting with config: server_url={self.config.server_url}, generation_length_per_request={self.config.generation_length_per_request}, listening_duration_ticks={self.config.listening_duration_ticks}, prompt_dir={self.config.prompt_dir}")
        
        # Clear server-side history/injection state so each session starts clean.
        clear_history_on_server(self.config.server_url)

        self.stop_event.clear()
        self.event_queue = Queue()
        self.inference_request_queue = Queue()
        self.inference_response_queue = Queue()
        self.all_timing_data = []
        self.tick_history = []

        try:
            self.audio_output_handler = AudioOutputHandler(
                port_name=self.config.midi_output_name,
                accompaniment_velocity=self.config.accompaniment_velocity,
            )
        except Exception as e:
            print(f"Warning: Could not initialize audio output: {e}")
            self.audio_output_handler = None

        # Initialize MIDI and JSON logging
        if self.config.record_session:
            self.midi_file_handler = MidiFileHandler(self.config.tempo, self.config.ticks_per_beat)
            print("[INIT] MIDI file recording enabled")
        else:
            self.midi_file_handler = None

        if self.config.save_json_log:
            self.json_log_handler = JsonLogHandler()
            print("[INIT] JSON logging enabled")
        else:
            self.json_log_handler = None

        # Create session log directory
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_dir = f"logs/web_session_{timestamp}"
        os.makedirs(self.session_log_dir, exist_ok=True)
        print(f"[INIT] Session logs will be saved to: {self.session_log_dir}")
        
        current_tick_ref = {"current_tick": 0}
        
        if self.config.input_mode == "file" and self.config.midi_file_path:
            print(f"[DEBUG] Starting MIDI file input: {self.config.midi_file_path}")
            self.input_thread = threading.Thread(
                target=read_midi_file_input,
                args=(
                    self.event_queue,
                    self.config.midi_file_path,
                    current_tick_ref,
                    self.config.tempo,
                    self.config.ticks_per_beat,
                    self.config.midi_file_delay_ticks,
                    0,  # skip_ticks (for injection offset)
                    self.audio_output_handler,  # audio_handler
                    self.config.melody_channel,  # melody_channel
                ),
                daemon=True,
            )
        elif self.config.input_mode == "midi":
            print(f"[DEBUG] Starting MIDI device input: device_name='{self.config.midi_input_name}'")
            self.input_thread = threading.Thread(
                target=read_midi_input,
                args=(self.event_queue, self.config.midi_input_name, self.audio_output_handler, self.config.melody_channel),
                daemon=True,
            )
        else:
            print(f"[DEBUG] Starting keyboard input")
            self.input_thread = threading.Thread(
                target=read_keyboard_input,
                args=(self.event_queue, self.audio_output_handler, self.config.melody_channel),
                daemon=True,
            )
        
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )
        
        self.tick_thread = threading.Thread(
            target=self._tick_loop,
            args=(current_tick_ref,),
            daemon=True,
        )
        
        self.input_thread.start()
        self.inference_thread.start()
        self.tick_thread.start()
        
        self.is_running = True
        self.ws_handler.send_status("running", "Client started")
        self.ws_handler.send_config(self.config.model_dump())
        
        return True
    
    def stop(self):
        """Stop the client."""
        if not self.is_running:
            return False
        
        print("[DEBUG] Stopping client...")
        self.stop_event.set()
        
        # Send stop signal to all queues
        if self.event_queue:
            self.event_queue.put(None)
        if self.inference_request_queue:
            self.inference_request_queue.put(None)
        
        # Wait for threads to finish - input thread first to release MIDI port
        if self.input_thread and self.input_thread.is_alive():
            print("[DEBUG] Waiting for input thread to stop...")
            self.input_thread.join(timeout=2.0)
            if self.input_thread.is_alive():
                print("[WARNING] Input thread did not stop cleanly")
        
        if self.inference_thread and self.inference_thread.is_alive():
            print("[DEBUG] Waiting for inference thread to stop...")
            self.inference_thread.join(timeout=2.0)
        
        if self.tick_thread and self.tick_thread.is_alive():
            print("[DEBUG] Waiting for tick thread to stop...")
            self.tick_thread.join(timeout=2.0)
        
        # Close audio output
        if self.audio_output_handler:
            print("[DEBUG] Closing audio output handler...")
            self.audio_output_handler.close()
            self.audio_output_handler = None

        # Save session logs
        if self.session_log_dir:
            print(f"\n--- Saving session logs to {self.session_log_dir} ---")

            # Save MIDI file
            if self.midi_file_handler:
                try:
                    self.midi_file_handler.save_to_midi(self.session_log_dir)
                    print(f"✓ MIDI file saved")
                except Exception as e:
                    print(f"✗ Failed to save MIDI file: {e}")

            # Save JSON logs
            if self.json_log_handler:
                try:
                    self.json_log_handler.save_logs(self.session_log_dir)
                    print(f"✓ JSON logs saved")
                except Exception as e:
                    print(f"✗ Failed to save JSON logs: {e}")

            # Save tick history
            try:
                import json
                tick_history_path = os.path.join(self.session_log_dir, "tick_history.json")
                with open(tick_history_path, "w") as f:
                    json.dump(self.tick_history, f, indent=2)
                print(f"✓ Tick history saved ({len(self.tick_history)} ticks)")
            except Exception as e:
                print(f"✗ Failed to save tick history: {e}")

            # Save timing data summary
            if self.all_timing_data:
                try:
                    import json
                    timing_path = os.path.join(self.session_log_dir, "timing_summary.json")
                    with open(timing_path, "w") as f:
                        json.dump({
                            "count": len(self.all_timing_data),
                            "timings": self.all_timing_data
                        }, f, indent=2)
                    print(f"✓ Timing data saved ({len(self.all_timing_data)} requests)")
                except Exception as e:
                    print(f"✗ Failed to save timing data: {e}")

        # Clear references
        self.input_thread = None
        self.inference_thread = None
        self.tick_thread = None
        
        self.is_running = False
        self.ws_handler.send_status("stopped", "Client stopped")
        
        print("[DEBUG] Client stopped successfully")
        return True
    
    def restart(self, config: Optional[ClientConfig] = None):
        """Restart the client."""
        self.stop()
        time.sleep(0.5)
        return self.start(config)
    
    def _inference_worker(self):
        """Worker thread for sending requests to server."""
        import requests

        while not self.stop_event.is_set():
            try:
                queue_item = self.inference_request_queue.get(timeout=0.1)
            except Exception:
                continue

            if queue_item is None:
                break

            request_data, full_request_dict = queue_item

            # Melody notes are already in event-stream format (like client_lekai)
            # No conversion needed
            if "melody_notes" in request_data and request_data["melody_notes"]:
                print(f"[DEBUG] _inference_worker: Sending {len(request_data['melody_notes'])} melody events")
                print(f"[DEBUG]   Melody events: {request_data['melody_notes']}")
            else:
                print(f"[DEBUG] _inference_worker: No melody notes to send")

            client_send_time = time.perf_counter()
            request_data["client_request_send_time"] = client_send_time
            full_request_dict["client_request_send_time"] = client_send_time

            start_time = client_send_time
            try:
                response = requests.post(
                    self.config.server_url,
                    json=request_data,
                    timeout=5.0
                )
                response.raise_for_status()
                response_json = response.json()
            except Exception as e:
                print(f"Error contacting server: {e}")
                response_json = None

            end_time = time.perf_counter()
            round_trip_time = end_time - start_time

            self.inference_response_queue.put((response_json, round_trip_time, full_request_dict))
    
    def _tick_loop(self, current_tick_ref: dict):
        """Main tick loop."""
        seconds_per_tick = (60.0 / self.config.tempo) / self.config.ticks_per_beat
        tick_count = -1
        playback_schedule = {}
        number_of_hit = 0
        total_backup_level = 0
        
        notes_for_next_request = []
        last_inference_timings = {}
        ticks_per_bar = self.config.ticks_per_beat * self.config.beats_per_bar
        
        listening_mode_active = self.config.listening_duration_ticks > 0 and LISTENING_MODE_AVAILABLE
        listening_mode_completed = False
        listening_worker_thread = None
        listening_worker_result_queue = Queue()
        prompt_library = None
        
        print(f"[DEBUG] listening_duration_ticks={self.config.listening_duration_ticks}, LISTENING_MODE_AVAILABLE={LISTENING_MODE_AVAILABLE}, listening_mode_active={listening_mode_active}")
        
        if self.config.listening_duration_ticks > 0 and not LISTENING_MODE_AVAILABLE:
            print("Warning: Listening mode requested but dependencies not available (key_detection, prompt_library)")
            print("Install required dependencies or run from the proper environment")
        
        if listening_mode_active:
            print(f"\n{'='*60}")
            print(f"LISTENING MODE ACTIVE")
            print(f"Will collect user input for {self.config.listening_duration_ticks} ticks")
            print(f"Then process key detection for {self.GRACE_PERIOD_TICKS} ticks (grace period)")
            print(f"Real-time generation starts at tick {self.config.listening_duration_ticks + self.GRACE_PERIOD_TICKS}")
            print(f"{'='*60}\n")
            
            if self.config.prompt_dir:
                try:
                    prompt_library = PromptLibrary(self.config.prompt_dir)
                except Exception as e:
                    print(f"Warning: Could not initialize prompt library: {e}")
                    listening_mode_active = False
            else:
                print("Warning: No prompt_dir specified, listening mode disabled")
                listening_mode_active = False
        
        self.ws_handler.send_status("running", "Tick loop started")

        while not self.stop_event.is_set():
            tick_count += 1
            current_tick_ref["current_tick"] = tick_count

            bar_count = tick_count // ticks_per_bar
            beat_in_bar = (tick_count % ticks_per_bar) // self.config.ticks_per_beat

            self.ws_handler.send_tick(tick_count, bar_count, beat_in_bar)

            # --- 0. Trigger Inference at tick=0 (FIRST, before anything else) ---
            is_tick_zero = tick_count == 0
            is_trigger_tick = False  # Will be set later for tick=3,7,11,...

            # During listening mode (before completion), suppress inference triggers.
            suppress_inference = listening_mode_active and not listening_mode_completed

            if is_tick_zero and not suppress_inference:
                # Trigger inference at the very start, gen_start_tick = 0
                generation_start_tick = 0
                request_data = {
                    "melody_notes": notes_for_next_request,
                    "generation_start_tick": generation_start_tick,
                    "generation_length_frames": self.config.generation_length_per_request,
                }
                print(f"[DEBUG] Tick {tick_count}: Sending INITIAL inference request, gen_start={generation_start_tick}")
                self.inference_request_queue.put((request_data, request_data.copy()))
                notes_for_next_request = []
                is_trigger_tick = True

            time.sleep(seconds_per_tick * 0.1)

            # --- 1. Process User Input ---
            user_notes_this_tick = []

            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                except Exception:
                    break

                if event is None:
                    self.stop_event.set()
                    break

                # --- Note Quantization & Audio Playback ---
                # Use the event's tick if available (from MIDI file input), otherwise use current tick_count
                event_tick = event.get("tick", tick_count)

                if event["type"] == "note_on":
                    # 1. Quantize the note for the inference engine request.
                    quantized_note = {
                        "type": "note_on",
                        "pitch": event["pitch"],
                        "tick": event_tick,
                    }
                    # Preserve velocity if present (for MIDI recording)
                    if "velocity" in event:
                        quantized_note["velocity"] = event["velocity"]
                    notes_for_next_request.append(quantized_note)
                    user_notes_this_tick.append(quantized_note)

                    # Log user note to MIDI file
                    if self.midi_file_handler:
                        self.midi_file_handler.add_user_note(quantized_note)

                    print(f"[DEBUG] Tick {tick_count}: Added note_on pitch={event['pitch']} tick={event_tick} to notes_for_next_request (total={len(notes_for_next_request)})")

                    self.ws_handler.send_note_on(
                        pitch=event["pitch"],
                        velocity=event["velocity"],
                        tick=event_tick,
                        duration=self.DEFAULT_NOTE_DURATION_TICKS,
                        source="user"
                    )
                    # Audio playback moved to input thread for lower latency (already handled)

                elif event["type"] == "note_off":
                    quantized_note = {
                        "type": "note_off",
                        "pitch": event["pitch"],
                        "tick": event_tick,
                    }
                    notes_for_next_request.append(quantized_note)
                    user_notes_this_tick.append(quantized_note)

                    # Log user note_off to MIDI file
                    if self.midi_file_handler:
                        self.midi_file_handler.add_user_note(quantized_note)

                    self.ws_handler.send_note_off(
                        pitch=event["pitch"],
                        tick=event_tick,
                        source="user"
                    )
                    # Audio note_off handled in input thread

            # --- 2. Handle Inference Responses ---
            while not self.inference_response_queue.empty():
                try:
                    response_data, round_trip_time, request_data = self.inference_response_queue.get_nowait()
                except Exception:
                    break
                
                if response_data:
                    # --- Log the complete inference event ---
                    if self.json_log_handler:
                        self.json_log_handler.log_inference_event(request_data, response_data)

                    # --- Calculate and store all timing information ---
                    timings = response_data["timings"]
                    timings["round_trip_time"] = round_trip_time

                    # Calculate server processing duration (this is accurate as it uses one clock)
                    server_arrival_time = timings["request_arrival_time"]
                    server_response_time = timings["response_output_time"]
                    server_processing_duration = server_response_time - server_arrival_time
                    timings["server_processing_duration"] = server_processing_duration

                    # Calculate total network latency (accurate)
                    # This is the time spent on the network for both the request and response.
                    timings["total_network_latency"] = round_trip_time - server_processing_duration

                    self.all_timing_data.append(timings)

                    # Send stats to websocket (UI-specific)
                    self.ws_handler.send_stats(
                        round_trip_ms=round_trip_time * 1000,
                        server_process_ms=server_processing_duration * 1000,
                        network_latency_ms=timings["total_network_latency"] * 1000
                    )

                    # --- Process Generated Events ---
                    # Engine now returns event-stream format (note_on/note_off)
                    newly_generated_notes = response_data["accompaniment"]

                    # DEBUG: Log what server returned
                    note_on_count = sum(1 for n in newly_generated_notes if n.get("type") == "note_on")
                    note_off_count = sum(1 for n in newly_generated_notes if n.get("type") == "note_off")
                    if newly_generated_notes:
                        print(f"  [DEBUG] Server returned {len(newly_generated_notes)} events: {note_on_count} note_on, {note_off_count} note_off")

                    gen_start = None
                    if isinstance(request_data, dict):
                        gen_start = request_data.get("generation_start_tick")

                    # Assign backup_level based on tick offset from generation start
                    for n in newly_generated_notes:
                        if gen_start is not None:
                            n["backup_level"] = max(0, int(n["tick"] - gen_start))
                        else:
                            n["backup_level"] = 0

                    # --- Clear stale notes from the previous generation ---
                    # This ensures that if a new response arrives before the old one is
                    # fully played out, we only replace future model-generated notes.
                    # User-played note_offs are preserved.
                    if newly_generated_notes:
                        # Find the first tick where the new generation actually places a note.
                        # This prevents clearing old notes if there's a gap before the new music starts.
                        first_new_note_tick = min(note["tick"] for note in newly_generated_notes)

                        ticks_to_clean = [t for t in playback_schedule if t >= first_new_note_tick]
                        for tick in ticks_to_clean:
                            # Filter out events sourced from the model, keep user events
                            playback_schedule[tick] = [
                                event for event in playback_schedule[tick]
                                if event.get("source") != "model"
                            ]
                            # If the tick is now empty, remove it from the schedule
                            if not playback_schedule[tick]:
                                del playback_schedule[tick]

                    # --- Schedule new notes ---
                    for note in newly_generated_notes:
                        if note["tick"] >= tick_count:
                            if note["tick"] not in playback_schedule:
                                playback_schedule[note["tick"]] = []

                            # Server now returns explicit event type. For safety (older
                            # servers), default to note_on if absent.
                            normalized_note = dict(note)
                            normalized_note.setdefault("type", "note_on")

                            # Tag as a model-originated event
                            playback_schedule[note["tick"]].append({**normalized_note, "source": "model"})

                            # Send to websocket (only note_on events shown in UI) - UI-specific
                            if normalized_note.get("type") == "note_on":
                                self.ws_handler.send_note_on(
                                    pitch=note["pitch"],
                                    velocity=self.config.accompaniment_velocity,
                                    tick=note["tick"],
                                    duration=note.get("duration", 4),
                                    source="model",
                                    backup_level=note.get("backup_level", 0)
                                )

                    # Store timings for display, making them persistent
                    last_inference_timings = timings
            
            if listening_mode_active and not listening_mode_completed:
                if tick_count == self.config.listening_duration_ticks:
                    print(f"\n{'='*60}")
                    print(f"LISTENING PERIOD COMPLETE at tick {tick_count}")
                    print(f"Collected {len(notes_for_next_request)} melody notes")
                    
                    # Check if manual or auto mode
                    if self.config.listening_mode == "manual":
                        print(f"Manual injection mode - using selected prompt")
                        print(f"{'='*60}")
                        
                        if self.config.manual_prompt_path and self.config.server_url:
                            listening_worker_thread = threading.Thread(
                                target=manual_injection_mode_worker,
                                args=(
                                    notes_for_next_request.copy(),
                                    self.config.listening_duration_ticks,
                                    self.config.manual_prompt_path,
                                    self.config.server_url,
                                    listening_worker_result_queue,
                                    self.midi_file_handler,
                                ),
                                daemon=True
                            )
                            listening_worker_thread.start()
                        else:
                            print("No manual prompt selected or server URL missing, skipping injection")
                            listening_mode_completed = True
                            notes_for_next_request = []
                    else:
                        print(f"Auto mode - spawning background worker for key detection...")
                        print(f"{'='*60}")
                        
                        if prompt_library and self.config.server_url:
                            listening_worker_thread = threading.Thread(
                                target=listening_mode_worker,
                                args=(
                                    notes_for_next_request.copy(),
                                    self.config.listening_duration_ticks,
                                    prompt_library,
                                    self.config.server_url,
                                    self.config.key_detection_method,
                                    listening_worker_result_queue,
                                    self.midi_file_handler,
                                ),
                                daemon=True
                            )
                            listening_worker_thread.start()
                        else:
                            print("Prompt library not initialized, skipping injection")
                            listening_mode_completed = True
                            notes_for_next_request = []

                elif tick_count == self.config.listening_duration_ticks + self.GRACE_PERIOD_TICKS:
                    print(f"\n{'='*60}")
                    print(f"GRACE PERIOD COMPLETE at tick {tick_count}")
                    print(f"Checking worker status...")
                    print(f"{'='*60}")

                    if not listening_worker_result_queue.empty():
                        success = listening_worker_result_queue.get()
                        if success:
                            print("Listening mode injection successful!")
                        else:
                            print("Listening mode injection failed, continuing without prompt")
                    else:
                        print("Warning: Key detection took longer than grace period")
                        if listening_worker_thread and listening_worker_thread.is_alive():
                            listening_worker_thread.join(timeout=2.0)
                        if not listening_worker_result_queue.empty():
                            success = listening_worker_result_queue.get()
                            print(f"Worker completed with result: {success}")
                        else:
                            print("Worker timed out, continuing without prompt")

                    listening_mode_completed = True

                if not listening_mode_completed:
                    if self.config.metronome and self.audio_output_handler:
                        is_beat_tick = (tick_count % self.config.ticks_per_beat) == 0
                        if is_beat_tick:
                            beat_in_bar_metro = (tick_count % ticks_per_bar) // self.config.ticks_per_beat
                            if beat_in_bar_metro == 0:
                                self.audio_output_handler.metro_first()
                            else:
                                self.audio_output_handler.metro_other()

                    time.sleep(seconds_per_tick)
                    continue

            # --- 3. (Moved to end of loop for tick=3,7,11,...) ---
            # Inference triggering logic for tick=0 is handled at the beginning.
            # For tick=3,7,11,... (ticks_per_beat - 1), we trigger at the end of the loop.

            # --- 4. Play Scheduled Notes ---
            notes_to_play_this_tick = []
            notes_to_stop_this_tick = []

            # Check the schedule for events supposed to happen on the current tick
            scheduled_events = playback_schedule.pop(tick_count, [])
            is_hit = False
            this_backup_level = 0
            for event in scheduled_events:
                if event.get("source") == "model":
                    is_hit = True
                    this_backup_level = event.get("backup_level", 0)
                    if event.get("is_placeholder", False):
                        continue  # Skip placeholder notes
                if event.get("type") == "note_off":
                    notes_to_stop_this_tick.append(event)
                else:  # It's a note_on (or other non-note_off event)
                    notes_to_play_this_tick.append(event)
            if is_hit:
                number_of_hit += 1
                total_backup_level += this_backup_level
            try:  # add recording of tick history
                tick_record = {
                    "tick": tick_count,
                    "is_hit": bool(is_hit),
                    "backup_level": int(this_backup_level),
                    "num_model_notes": len(notes_to_play_this_tick),
                    "num_user_notes": len(user_notes_this_tick),
                }
                self.tick_history.append(tick_record)
            except Exception:
                # Don't let recording failure affect main loop
                pass

            # Process note-offs first
            for event in notes_to_stop_this_tick:
                if self.audio_output_handler:
                    self.audio_output_handler.off(event["pitch"])
                # Record model note_off events to MIDI file
                if event.get("source") == "model" and self.midi_file_handler:
                    # Ensure tick is set to current tick_count for accurate recording
                    # Preserve all event fields (type, pitch, velocity, etc.)
                    event_for_midi = dict(event)
                    event_for_midi["tick"] = tick_count
                    # Ensure type is set (should already be "note_off")
                    if "type" not in event_for_midi:
                        event_for_midi["type"] = "note_off"
                    self.midi_file_handler.add_model_note(event_for_midi)
                # Send to websocket (UI-specific)
                self.ws_handler.send_note_off(event["pitch"], tick_count, "model")

            # Process note-ons and schedule their corresponding note-offs
            for event in notes_to_play_this_tick:
                # This loop only processes model-generated notes.
                if event.get("type") != "note_on":
                    # Ignore non-note events (e.g. placeholders or future extensions)
                    continue
                if self.audio_output_handler:
                    self.audio_output_handler.on(event["pitch"], self.config.accompaniment_velocity)
                # Record to MIDI with current tick for accurate timing
                if self.midi_file_handler:
                    # Preserve all event fzields (type, pitch, velocity, backup_level, etc.)
                    event_for_midi = dict(event)
                    event_for_midi["tick"] = tick_count
                    # Ensure type is set (should already be "note_on")
                    if "type" not in event_for_midi:
                        event_for_midi["type"] = "note_on"
                    # Ensure velocity is set (use accompaniment_velocity if not present)
                    if "velocity" not in event_for_midi:
                        event_for_midi["velocity"] = self.config.accompaniment_velocity
                    self.midi_file_handler.add_model_note(event_for_midi)

                # Event-stream mode: note_off should arrive explicitly from server.
                # We keep *optional* legacy compatibility: if duration exists, still schedule.
                dur = event.get("duration")
                if dur is not None:
                    note_off_tick = tick_count + int(dur)
                    if note_off_tick not in playback_schedule:
                        playback_schedule[note_off_tick] = []
                    playback_schedule[note_off_tick].append({**event, "type": "note_off", "source": "model"})

            # --- 5. Metronome ---
            if self.config.metronome and self.audio_output_handler:
                is_beat_tick = (tick_count % self.config.ticks_per_beat) == 0
                if is_beat_tick:
                    beat_in_bar = (tick_count % ticks_per_bar) // self.config.ticks_per_beat
                    if beat_in_bar == 0:
                        self.audio_output_handler.metro_first()
                    else:
                        self.audio_output_handler.metro_other()

            # --- 6. Update Display (UI-specific - websocket stats) ---
            if tick_count > 0 and tick_count % 16 == 0:
                hit_rate = number_of_hit / tick_count if tick_count > 0 else 0
                avg_backup = total_backup_level / number_of_hit if number_of_hit > 0 else 0
                self.ws_handler.send_stats(
                    hit_rate=hit_rate,
                    avg_backup_level=avg_backup,
                    total_hits=number_of_hit,
                    total_ticks=tick_count
                )

            # --- 7. Trigger Inference at tick=3,7,11,... (ticks_per_beat - 1) at the end of loop ---
            # This gives the server maximum time to process before the next beat starts.
            # At tick=3, we trigger generation for beat 1 (gen_start_tick=4)
            # At tick=7, we trigger generation for beat 2 (gen_start_tick=8)
            # etc.
            if (
                not suppress_inference
                and not is_tick_zero
                and (tick_count % self.config.ticks_per_beat) == (self.config.ticks_per_beat - 1)
            ):
                # Trigger at the last tick of each beat (except tick 0 which was handled at the start)
                generation_start_tick = tick_count + 1  # Next beat's start tick
                request_data = {
                    "melody_notes": notes_for_next_request,
                    "generation_start_tick": generation_start_tick,
                    "generation_length_frames": self.config.generation_length_per_request,
                }
                print(f"[DEBUG] Tick {tick_count}: Sending inference request, gen_start={generation_start_tick}, melody_notes_count={len(notes_for_next_request)}")
                if notes_for_next_request:
                    print(f"[DEBUG]   Melody notes: {notes_for_next_request}")
                self.inference_request_queue.put((request_data, request_data.copy()))
                notes_for_next_request = []
                is_trigger_tick = True

            time.sleep(seconds_per_tick * 0.9)
        
        self.ws_handler.send_status("stopped", "Tick loop ended")


ws_handler = WebSocketOutputHandler()
connection_manager = ConnectionManager()
client_manager = ClientManager(ws_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcast_task = asyncio.create_task(ws_broadcast_loop())
    yield
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    client_manager.stop()


app = FastAPI(title="StreamMUSE Web Client", lifespan=lifespan)


async def ws_broadcast_loop():
    """Async loop to broadcast messages from queue to WebSocket clients."""
    while True:
        messages = ws_handler.get_pending_messages()
        for msg in messages:
            await connection_manager.broadcast(msg)
        await asyncio.sleep(0.01)


web_ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_ui")


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(web_ui_dir, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    ws_handler.send_status("connected", "WebSocket connected")
    ws_handler.send_config(client_manager.config.model_dump())
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "keyboard_input" and client_manager.event_queue:
                    pitch = msg["pitch"]
                    velocity = msg.get("velocity", 100)
                    if msg.get("event") == "note_on":
                        client_manager.event_queue.put({
                            "type": "note_on",
                            "pitch": pitch,
                            "velocity": velocity
                        })
                        if client_manager.audio_output_handler:
                            client_manager.audio_output_handler.on(pitch, velocity, channel=client_manager.config.melody_channel)
                    elif msg.get("event") == "note_off":
                        client_manager.event_queue.put({
                            "type": "note_off",
                            "pitch": pitch
                        })
                        if client_manager.audio_output_handler:
                            client_manager.audio_output_handler.off(pitch, channel=client_manager.config.melody_channel)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.post("/api/start")
async def start_client(config: Optional[ClientConfig] = None):
    if client_manager.start(config):
        return {"success": True, "message": "Client started"}
    return JSONResponse(
        status_code=400,
        content={"success": False, "message": "Client already running"}
    )


@app.post("/api/stop")
async def stop_client():
    if client_manager.stop():
        return {"success": True, "message": "Client stopped"}
    return JSONResponse(
        status_code=400,
        content={"success": False, "message": "Client not running"}
    )


@app.post("/api/restart")
async def restart_client(config: Optional[ClientConfig] = None):
    if client_manager.restart(config):
        return {"success": True, "message": "Client restarted"}
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Failed to restart client"}
    )


@app.get("/api/config")
async def get_config():
    return client_manager.config.model_dump()


@app.post("/api/config")
async def update_config(config: ClientConfig):
    client_manager.config = config
    ws_handler.send_config(config.model_dump())
    return {"success": True, "config": config.model_dump()}


@app.get("/api/manual_prompts")
async def get_manual_prompts():
    """Get list of available manual prompt files from prompts/manual/ directory."""
    try:
        manual_dir = os.path.join(os.path.dirname(__file__), "..", "prompts", "manual")
        
        if not os.path.exists(manual_dir):
            return {
                "prompts": [],
                "success": True,
                "message": f"Manual prompts directory not found: {manual_dir}"
            }
        
        # Find all .mid files in the manual directory
        prompt_files = []
        for filename in os.listdir(manual_dir):
            if filename.endswith('.mid'):
                file_path = os.path.join(manual_dir, filename)
                prompt_files.append({
                    "name": filename,
                    "path": file_path
                })
        
        # Sort by name
        prompt_files.sort(key=lambda x: x['name'])
        
        print(f"[DEBUG] Found {len(prompt_files)} manual prompts in {manual_dir}")
        
        return {
            "prompts": prompt_files,
            "success": True
        }
    except Exception as e:
        print(f"Error getting manual prompts: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=200,
            content={
                "prompts": [],
                "success": False,
                "error": str(e)
            }
        )


@app.get("/api/midi_devices")
async def get_midi_devices():
    """Get list of available MIDI input and output devices."""
    try:
        import mido
        
        # Don't reset backend if already set - it can cause issues
        print(f"[DEBUG] MIDI backend: {mido.backend.name}")
        
        raw_input = mido.get_input_names()
        raw_output = mido.get_output_names()
        print(f"[DEBUG] Raw devices - Input: {raw_input}, Output: {raw_output}")
        
        input_devices = list(raw_input)
        output_devices = list(raw_output)
        print(f"[DEBUG] As lists - Input: {input_devices}, Output: {output_devices}")
        
        # Filter out any duplicate or system devices if needed
        # But keep FluidSynth and other virtual ports
        input_devices = [d for d in input_devices if not d.startswith('RtMidi')]
        output_devices = [d for d in output_devices if not d.startswith('RtMidi')]
        
        print(f"[DEBUG] After filter - Input: {input_devices}, Output: {output_devices}")
        
        return {
            "input_devices": input_devices,
            "output_devices": output_devices,
            "success": True
        }
    except Exception as e:
        print(f"Error getting MIDI devices: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=200,  # Return 200 to avoid breaking UI
            content={
                "input_devices": [],
                "output_devices": [],
                "success": False,
                "error": str(e)
            }
        )


@app.get("/api/status")
async def get_status():
    return {
        "is_running": client_manager.is_running,
        "config": client_manager.config.model_dump()
    }


if os.path.exists(os.path.join(web_ui_dir, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(web_ui_dir, "css")), name="css")
if os.path.exists(os.path.join(web_ui_dir, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(web_ui_dir, "js")), name="js")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamMUSE Web Client Server")
    parser.add_argument("--server_url", type=str, default="http://localhost:8988/generate_accompaniment",
                        help="URL of the StreamMUSE inference server")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--ticks_per_beat", type=int, default=4, help="Ticks per beat")
    parser.add_argument("--beats_per_bar", type=int, default=4, help="Beats per bar")
    parser.add_argument("--generation_interval_ticks", type=int, default=4,
                        help="Ticks between generation requests (should equal ticks_per_beat)")
    parser.add_argument("--generation_length_per_request", type=int, default=5,
                        help="Frames to generate per request")
    parser.add_argument("--accompaniment_velocity", type=int, default=50,
                        help="Velocity for accompaniment notes (0-127)")
    parser.add_argument("--listening_duration_ticks", type=int, default=0,
                        help="Duration for listening mode (0 = disabled)")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Directory containing prompt MIDI files")
    parser.add_argument("--key_detection_method", type=str, default="lightweight",
                        choices=["lightweight", "music21"],
                        help="Method for key detection")
    parser.add_argument("--port", type=int, default=8080, help="Port for web UI server")

    args = parser.parse_args()

    client_manager.config = ClientConfig(
        server_url=args.server_url,
        tempo=args.tempo,
        ticks_per_beat=args.ticks_per_beat,
        beats_per_bar=args.beats_per_bar,
        generation_interval_ticks=args.generation_interval_ticks,
        generation_length_per_request=args.generation_length_per_request,
        accompaniment_velocity=args.accompaniment_velocity,
        listening_duration_ticks=args.listening_duration_ticks,
        prompt_dir=args.prompt_dir,
        key_detection_method=args.key_detection_method,
    )

    print("Starting StreamMUSE Web Client Server...")
    print(f"Server URL: {args.server_url}")
    print(f"Open http://localhost:{args.port} in your browser")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

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


class ClientConfig(BaseModel):
    server_url: str = "http://localhost:8000/generate_accompaniment"
    
    tempo: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
    generation_interval_ticks: int = 1
    generation_length_per_request: int = 5
    
    note_duration_ticks: int = 4
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
    
    key_detection_method: str = "lightweight"
    prompt_dir: Optional[str] = None


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
    injection_length_ticks: int
) -> bool:
    import requests
    injection_url = server_url.replace("/generate_accompaniment", "/inject_notes")
    try:
        request_data = {
            "melody_notes": melody_notes,
            "accompaniment_notes": accompaniment_notes,
            "injection_length_ticks": injection_length_ticks
        }
        print(f"Injecting {len(melody_notes)} melody notes and {len(accompaniment_notes)} accompaniment notes...")
        response = requests.post(injection_url, json=request_data, timeout=5.0)
        response.raise_for_status()
        result = response.json()
        if result["success"]:
            print(f"Injection successful: {result['message']}")
            return True
        else:
            print(f"Injection failed: {result['message']}")
            return False
    except Exception as e:
        print(f"Injection request failed: {e}")
        return False


def listening_mode_worker(
    collected_melody_notes: list,
    listening_duration_ticks: int,
    prompt_library: PromptLibrary,
    server_url: str,
    key_detection_method: str,
    result_queue: Queue
):
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
            listening_duration_ticks
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
    
    DEFAULT_NOTE_DURATION_TICKS = 4
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
        self.all_timing_data = []
        self.tick_history = []
    
    def start(self, config: Optional[ClientConfig] = None):
        """Start the client with given config."""
        if self.is_running:
            return False
        
        if config:
            self.config = config
        
        print(f"Starting with config: server_url={self.config.server_url}, generation_length_per_request={self.config.generation_length_per_request}, listening_duration_ticks={self.config.listening_duration_ticks}, prompt_dir={self.config.prompt_dir}")
        
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
        
        current_tick_ref = {"current_tick": 0}
        
        if self.config.input_mode == "file" and self.config.midi_file_path:
            self.input_thread = threading.Thread(
                target=read_midi_file_input,
                args=(
                    self.event_queue,
                    self.config.midi_file_path,
                    current_tick_ref,
                    self.config.tempo,
                    self.config.ticks_per_beat,
                    0,
                    0,
                    True,
                    self.DEFAULT_NOTE_DURATION_TICKS,
                    self.audio_output_handler,
                    self.config.melody_channel,
                ),
                daemon=True,
            )
        elif self.config.input_mode == "midi":
            self.input_thread = threading.Thread(
                target=read_midi_input,
                args=(self.event_queue, self.config.midi_input_name, self.audio_output_handler, self.config.melody_channel),
                daemon=True,
            )
        else:
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
        
        self.stop_event.set()
        
        if self.event_queue:
            self.event_queue.put(None)
        if self.inference_request_queue:
            self.inference_request_queue.put(None)
        
        if self.tick_thread and self.tick_thread.is_alive():
            self.tick_thread.join(timeout=2.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        if self.audio_output_handler:
            self.audio_output_handler.close()
            self.audio_output_handler = None
        
        self.is_running = False
        self.ws_handler.send_status("stopped", "Client stopped")
        
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
            
            user_notes_this_tick = []
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                except Exception:
                    break
                
                if event is None:
                    self.stop_event.set()
                    break
                
                if event["type"] == "note_on":
                    quantized_note = {
                        "pitch": event["pitch"],
                        "tick": tick_count - 1,
                        "duration": self.DEFAULT_NOTE_DURATION_TICKS,
                    }
                    notes_for_next_request.append(quantized_note)
                    user_notes_this_tick.append(quantized_note)
                    
                    self.ws_handler.send_note_on(
                        pitch=event["pitch"],
                        velocity=event["velocity"],
                        tick=tick_count - 1,
                        duration=self.DEFAULT_NOTE_DURATION_TICKS,
                        source="user"
                    )
                    # Audio playback moved to input thread for lower latency
                
                elif event["type"] == "note_off":
                    self.ws_handler.send_note_off(
                        pitch=event["pitch"],
                        tick=tick_count,
                        source="user"
                    )
                    # Audio note_off handled in input thread
            
            while not self.inference_response_queue.empty():
                try:
                    response_data, round_trip_time, request_data = self.inference_response_queue.get_nowait()
                except Exception:
                    break
                
                if response_data:
                    timings = response_data.get("timings", {})
                    timings["round_trip_time"] = round_trip_time
                    
                    if "request_arrival_time" in timings and "response_output_time" in timings:
                        server_processing_duration = timings["response_output_time"] - timings["request_arrival_time"]
                        timings["server_processing_duration"] = server_processing_duration
                        timings["total_network_latency"] = round_trip_time - server_processing_duration
                    
                    self.all_timing_data.append(timings)
                    
                    self.ws_handler.send_stats(
                        round_trip_ms=round_trip_time * 1000,
                        server_process_ms=timings.get("server_processing_duration", 0) * 1000,
                        network_latency_ms=timings.get("total_network_latency", 0) * 1000
                    )
                    
                    newly_generated_notes = response_data.get("accompaniment", [])
                    gen_start = request_data.get("generation_start_tick") if isinstance(request_data, dict) else None
                    
                    print(f"[DEBUG] Received {len(newly_generated_notes)} accompaniment notes at tick {tick_count}, gen_start={gen_start}")
                    
                    for note in newly_generated_notes:
                        note["backup_level"] = int(note["tick"] - gen_start) if gen_start else 0
                    
                    if newly_generated_notes:
                        first_new_note_tick = min(note["tick"] for note in newly_generated_notes)
                        ticks_to_clean = [t for t in playback_schedule if t >= first_new_note_tick]
                        for tick in ticks_to_clean:
                            playback_schedule[tick] = [
                                event for event in playback_schedule[tick]
                                if event.get("source") != "model"
                            ]
                            if not playback_schedule[tick]:
                                del playback_schedule[tick]
                    
                    for note in newly_generated_notes:
                        if note["tick"] >= tick_count:
                            if note["tick"] not in playback_schedule:
                                playback_schedule[note["tick"]] = []
                            playback_schedule[note["tick"]].append({**note, "source": "model"})
                            
                            self.ws_handler.send_note_on(
                                pitch=note["pitch"],
                                velocity=self.config.accompaniment_velocity,
                                tick=note["tick"],
                                duration=note.get("duration", 4),
                                source="model",
                                backup_level=note.get("backup_level", 0)
                            )
                    
                    last_inference_timings = timings
            
            if listening_mode_active and not listening_mode_completed:
                if tick_count == self.config.listening_duration_ticks:
                    print(f"\n{'='*60}")
                    print(f"LISTENING PERIOD COMPLETE at tick {tick_count}")
                    print(f"Collected {len(notes_for_next_request)} melody notes")
                    print(f"Spawning background worker for key detection...")
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
                                listening_worker_result_queue
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
            
            is_trigger_tick = (tick_count % self.config.generation_interval_ticks) == 0
            
            if is_trigger_tick:
                next_interval_start_tick = tick_count + 1
                request_data = {
                    "melody_notes": notes_for_next_request,
                    "generation_start_tick": next_interval_start_tick,
                    "generation_length_frames": self.config.generation_length_per_request,
                }
                print(f"[DEBUG] Tick {tick_count}: Sending inference request, gen_start={next_interval_start_tick}")
                self.inference_request_queue.put((request_data, request_data.copy()))
                notes_for_next_request = []
            
            notes_to_play_this_tick = []
            notes_to_stop_this_tick = []
            is_hit = False
            this_backup_level = 0
            
            scheduled_events = playback_schedule.pop(tick_count, [])
            if scheduled_events:
                print(f"[DEBUG] Tick {tick_count}: Playing {len(scheduled_events)} scheduled events")
            for event in scheduled_events:
                if event.get("source") == "model":
                    is_hit = True
                    this_backup_level = event.get("backup_level", 0)
                
                if event.get("type") == "note_off":
                    notes_to_stop_this_tick.append(event)
                else:
                    notes_to_play_this_tick.append(event)
            
            if is_hit:
                number_of_hit += 1
                total_backup_level += this_backup_level
            
            self.tick_history.append({
                "tick": tick_count,
                "is_hit": is_hit,
                "backup_level": this_backup_level,
                "num_model_notes": len(notes_to_play_this_tick),
                "num_user_notes": len(user_notes_this_tick),
            })
            
            for event in notes_to_stop_this_tick:
                if self.audio_output_handler:
                    self.audio_output_handler.off(event["pitch"])
                self.ws_handler.send_note_off(event["pitch"], tick_count, "model")
            
            for event in notes_to_play_this_tick:
                if self.audio_output_handler:
                    self.audio_output_handler.on(event["pitch"], self.config.accompaniment_velocity)
                
                note_off_tick = tick_count + event.get("duration", 4)
                if note_off_tick not in playback_schedule:
                    playback_schedule[note_off_tick] = []
                playback_schedule[note_off_tick].append({**event, "type": "note_off", "source": "model"})
            
            if self.config.metronome and self.audio_output_handler:
                is_beat_tick = (tick_count % self.config.ticks_per_beat) == 0
                if is_beat_tick:
                    beat_in_bar_metro = (tick_count % ticks_per_bar) // self.config.ticks_per_beat
                    if beat_in_bar_metro == 0:
                        self.audio_output_handler.metro_first()
                    else:
                        self.audio_output_handler.metro_other()
            
            if tick_count > 0 and tick_count % 16 == 0:
                hit_rate = number_of_hit / tick_count if tick_count > 0 else 0
                avg_backup = total_backup_level / number_of_hit if number_of_hit > 0 else 0
                self.ws_handler.send_stats(
                    hit_rate=hit_rate,
                    avg_backup_level=avg_backup,
                    total_hits=number_of_hit,
                    total_ticks=tick_count
                )
            
            time.sleep(seconds_per_tick)
        
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
    parser.add_argument("--server_url", type=str, default="http://localhost:8000/generate_accompaniment",
                        help="URL of the StreamMUSE inference server")
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--ticks_per_beat", type=int, default=4, help="Ticks per beat")
    parser.add_argument("--beats_per_bar", type=int, default=4, help="Beats per bar")
    parser.add_argument("--generation_interval_ticks", type=int, default=1,
                        help="Ticks between generation requests")
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

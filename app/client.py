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
import json

from output_handlers.cli_output import CLIOutputHandler
from output_handlers.audio_output import AudioOutputHandler
from output_handlers.midi_file_handler import MidiFileHandler
from output_handlers.json_log_handler import JsonLogHandler
from input_handlers.input_handler import (
    read_midi_input,
    read_keyboard_input,
    read_midi_file_input,
)
from key_detection import detect_key_lightweight, detect_key_music21
from prompt_library import PromptLibrary
from midi_utils import midi_to_note


# --- Configuration ---
class StreamMUSEConfig:
    """Configuration settings for StreamMUSE client"""

    # Network
    DEFAULT_SERVER_URL = "http://localhost:8000/generate_accompaniment"
    DEFAULT_INJECTION_URL = "http://localhost:8000/inject_music"
    DEFAULT_INJECTION_STATUS_URL = "http://localhost:8000/injection_status"

    # Musical timing
    DEFAULT_TEMPO = 120.0
    DEFAULT_TICKS_PER_BEAT = 4
    DEFAULT_BEATS_PER_BAR = 4
    DEFAULT_GENERATION_INTERVAL_TICKS = 1
    DEFAULT_GENERATION_LENGTH = None  # For experiments only

    # Note handling
    DEFAULT_NOTE_DURATION_TICKS = 4
    LATENCY_OFFSET_TICKS = 2
    DEFAULT_ACCOMPANIMENT_VELOCITY = 50

    # Display
    DEFAULT_LOG_LINES = 10

    # MIDI File Input
    DEFAULT_MIDI_FILE_DELAY_TICKS = 0

    # Listening mode for adaptive prompting
    DEFAULT_LISTENING_DURATION_TICKS = 0  # 0 = disabled
    DEFAULT_PROMPT_LIBRARY_PATH = "prompts"
    DEFAULT_KEY_DETECTION_METHOD = "lightweight"  # or "music21"

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


def save_prompt_midi(
    original_file_path: str,
    injection_length_ticks: int,
    session_log_dir: str,
    client_ticks_per_beat: int = 4,
    injected_notes: dict = None,
):
    """
    保存注入的 prompt 为单独的 MIDI 文件
    injection_length_ticks: 以客户端 tick 为单位的长度 (1 tick = 1/4 beat)
    client_ticks_per_beat: 客户端使用的 ticks_per_beat (默认4)
    """

    try:
        # 确定 mel 和 acc 文件路径
        mel_file_path = original_file_path
        acc_file_path = original_file_path.replace("mel", "acc")

        print(f"处理 Melody 文件: {mel_file_path}")
        print(f"处理 Accompaniment 文件: {acc_file_path}")

        # 创建合并的 MIDI 文件
        prompt_midi = mido.MidiFile()

        # 使用 mel 文件的 ticks_per_beat 作为基准
        mel_midi = mido.MidiFile(mel_file_path)
        prompt_midi.ticks_per_beat = mel_midi.ticks_per_beat

        # 转换客户端 ticks 到 MIDI ticks
        midi_ticks_per_client_tick = prompt_midi.ticks_per_beat / client_ticks_per_beat
        injection_length_midi_ticks = int(
            injection_length_ticks * midi_ticks_per_client_tick
        )

        print(
            f"客户端 ticks: {injection_length_ticks}, MIDI ticks: {injection_length_midi_ticks}"
        )
        print(
            f"MIDI文件 ticks_per_beat: {prompt_midi.ticks_per_beat}, 客户端 ticks_per_beat: {client_ticks_per_beat}"
        )

        # 处理文件列表
        files_to_process = [("melody", mel_file_path), ("accompaniment", acc_file_path)]

        # 处理每个文件（mel 和 acc）
        for file_type, file_path in files_to_process:
            print(f"处理 {file_type} 文件: {file_path}")

            try:
                if not os.path.exists(file_path):
                    print(f"警告: {file_type} 文件不存在: {file_path}")
                    continue

                original_midi = mido.MidiFile(file_path)

                # 处理每个轨道
                for track_idx, track in enumerate(original_midi.tracks):
                    new_track = mido.MidiTrack()
                    # 设置轨道名称
                    track_name = f"{file_type}_track_{track_idx}"
                    new_track.append(
                        mido.MetaMessage("track_name", name=track_name, time=0)
                    )

                    current_time = 0

                    for msg in track:
                        # 计算这个消息的绝对时间位置（MIDI ticks）
                        msg_absolute_time = current_time + msg.time

                        if msg_absolute_time <= injection_length_midi_ticks:
                            # 完全在范围内，直接添加
                            new_track.append(msg.copy())
                            current_time = msg_absolute_time

                        elif current_time < injection_length_midi_ticks:
                            # 跨越边界的情况
                            if msg.type in [
                                "note_off",
                                "control_change",
                                "program_change",
                                "end_of_track",
                            ]:
                                # 重要的结束消息，调整时间后添加
                                adjusted_time = (
                                    injection_length_midi_ticks - current_time
                                )
                                adjusted_msg = msg.copy(time=adjusted_time)
                                new_track.append(adjusted_msg)
                            break
                        else:
                            # 完全超出范围
                            break

                    # 确保轨道以 end_of_track 结束
                    if new_track and new_track[-1].type != "end_of_track":
                        new_track.append(mido.MetaMessage("end_of_track", time=0))

                    if new_track:
                        prompt_midi.tracks.append(new_track)

            except Exception as e:
                print(f"处理 {file_type} 文件时出错: {e}")
                continue

        # 保存 prompt MIDI 文件
        prompt_file_path = os.path.join(session_log_dir, "prompt.mid")
        prompt_midi.save(prompt_file_path)
        print(f"✓ Prompt 已保存到: {prompt_file_path}")
        print(f"  包含 {len(prompt_midi.tracks)} 个轨道")

    except Exception as e:
        print(f"✗ 保存 prompt MIDI 文件失败: {e}")


def inject_notes_to_server(
    server_url: str,
    melody_notes: list,
    accompaniment_notes: list,
    injection_length_ticks: int
) -> bool:
    """
    Inject notes directly to server (new client-side prompting approach).

    Args:
        server_url: Base server URL
        melody_notes: List of melody note dicts with 'pitch', 'tick', 'duration'
        accompaniment_notes: List of accompaniment note dicts with 'pitch', 'tick', 'duration', 'program'
        injection_length_ticks: Length in ticks

    Returns:
        True if successful, False otherwise
    """
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
            print(f"✓ Injection successful: {result['message']}")
            return True
        else:
            print(f"✗ Injection failed: {result['message']}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ Injection request failed: {e}")
        return False


def listening_mode_worker(
    collected_melody_notes: list,
    listening_duration_ticks: int,
    prompt_library: PromptLibrary,
    server_url: str,
    key_detection_method: str,
    result_queue: Queue
):
    """
    Background worker thread for key detection and prompt injection.
    Runs asynchronously to avoid blocking the tick loop.

    Args:
        collected_melody_notes: Notes collected during listening period
        listening_duration_ticks: Duration of listening period
        prompt_library: PromptLibrary instance
        server_url: Server URL
        key_detection_method: "lightweight" or "music21"
        result_queue: Queue to put result (True/False) when done

    Returns:
        None (puts result in queue)
    """
    try:
        print(f"\n{'='*60}")
        print(f"LISTENING MODE WORKER - Analyzing {len(collected_melody_notes)} notes")
        print(f"{'='*60}")

        # 1. Detect key (might take 50-200ms)
        start_time = time.perf_counter()
        if key_detection_method == "music21":
            detected_key = detect_key_music21(collected_melody_notes)
        else:
            detected_key = detect_key_lightweight(collected_melody_notes)
        key_detection_time = time.perf_counter() - start_time

        print(f"✓ Detected key: {detected_key} (took {key_detection_time*1000:.1f}ms)")

        # 2. Select prompt from library (fast, <1ms)
        selected_prompt = prompt_library.select_prompt(detected_key, strategy="random")

        if not selected_prompt:
            print("✗ No prompt available, continuing without injection")
            result_queue.put(False)
            return

        print(f"✓ Selected prompt: {selected_prompt.get('name', 'unknown')}")

        # 3. Load accompaniment from selected prompt (might take 100-300ms for file I/O)
        load_start_time = time.perf_counter()
        _, accompaniment_notes = prompt_library.load_prompt_notes(
            selected_prompt,
            max_ticks=listening_duration_ticks,
            load_melody=False,  # Don't load prompt melody, use user's actual input
            load_accompaniment=True
        )
        load_time = time.perf_counter() - load_start_time

        if not accompaniment_notes:
            print("✗ No accompaniment notes loaded from prompt")
            result_queue.put(False)
            return

        print(f"✓ Loaded {len(accompaniment_notes)} accompaniment notes (took {load_time*1000:.1f}ms)")

        # 4. Inject to server (network call, 10-50ms)
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
            print(f"✓ Injection complete (took {inject_time*1000:.1f}ms)")
            print(f"{'='*60}")
            print(f"TOTAL PROCESSING TIME: {total_time*1000:.1f}ms")
            print(f"READY FOR REAL-TIME GENERATION")
            print(f"{'='*60}\n")
            result_queue.put(True)
        else:
            print(f"✗ Injection failed")
            result_queue.put(False)

    except Exception as e:
        print(f"✗ Listening mode worker error: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(False)


def perform_direct_injection(
    injection_file_path: str,
    injection_length_ticks: int,
    server_url: str,
    ticks_per_beat: int = 4
) -> int:
    """
    Perform direct injection from user-specified MIDI file (MANUAL mode).
    Client-side implementation of the old inject_music_to_server.

    Args:
        injection_file_path: Path to melody MIDI file (client-side)
        injection_length_ticks: Number of ticks to inject
        server_url: Server URL
        ticks_per_beat: Ticks per beat for MIDI conversion

    Returns:
        injection_length_ticks if successful, 0 otherwise
    """
    try:
        # Determine mel and acc file paths (same convention as before)
        mel_file_path = injection_file_path
        acc_file_path = injection_file_path.replace("mel", "acc")

        if not os.path.exists(mel_file_path):
            print(f"✗ Melody file not found: {mel_file_path}")
            return 0

        if not os.path.exists(acc_file_path):
            print(f"✗ Accompaniment file not found: {acc_file_path}")
            return 0

        print(f"Reading melody from: {mel_file_path}")
        print(f"Reading accompaniment from: {acc_file_path}")

        # Read both files
        melody_notes, _, _ = midi_to_note(mel_file_path, max_tick=injection_length_ticks)
        accompaniment_notes, _, _ = midi_to_note(acc_file_path, max_tick=injection_length_ticks)

        # Filter to injection length
        melody_notes = [n for n in melody_notes if n['tick'] < injection_length_ticks]
        accompaniment_notes = [n for n in accompaniment_notes if n['tick'] < injection_length_ticks]

        # Add program field to accompaniment notes if missing
        for note in accompaniment_notes:
            if 'program' not in note:
                note['program'] = 1

        print(f"Loaded {len(melody_notes)} melody notes and {len(accompaniment_notes)} accompaniment notes")

        # Inject to server using new endpoint
        success = inject_notes_to_server(
            server_url,
            melody_notes,
            accompaniment_notes,
            injection_length_ticks
        )

        if success:
            return injection_length_ticks
        else:
            return 0

    except Exception as e:
        print(f"✗ Direct injection failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


# 添加注入功能函数 (DEPRECATED - use perform_direct_injection instead)
def inject_music_to_server(
    server_base_url: str, injection_file_path: str, injection_length_ticks: int
):
    """
    向服务器注入音乐
    """
    injection_url = server_base_url.replace("/generate_accompaniment", "/inject_music")

    try:
        request_data = {
            "injection_file_path": injection_file_path,
            "injection_length_ticks": injection_length_ticks,
        }

        print(f"注入音乐: {injection_file_path} (前 {injection_length_ticks} ticks)")
        response = requests.post(injection_url, json=request_data)
        response.raise_for_status()

        result = response.json()
        if result["success"]:
            print(
                f"✓ 注入成功: {result['melody_notes_injected']} 旋律音符, {result['accompaniment_notes_injected']} 伴奏音符"
            )
            return result["injection_length_ticks"]
        else:
            print(f"✗ 注入失败: {result['message']}")
            return 0

    except requests.exceptions.RequestException as e:
        print(f"✗ 注入请求失败: {e}")
        return 0


def get_injection_status(server_base_url: str):
    """
    获取服务器注入状态
    """
    status_url = server_base_url.replace("/generate_accompaniment", "/injection_status")

    try:
        response = requests.get(status_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"获取注入状态失败: {e}")
        return {"is_injected": False, "injection_length_ticks": 0}


def clear_server_history(server_base_url: str) -> bool:
    """
    请求服务器清除历史和注入状态，返回是否成功
    """
    clear_url = server_base_url.replace("/generate_accompaniment", "/clear_history")
    try:
        resp = requests.post(clear_url)
        resp.raise_for_status()
        print("✓ 已请求服务器清除历史")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求服务器清除历史失败: {e}")
        return False


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
        request_data["client_request_send_time"] = client_send_time
        full_request_dict["client_request_send_time"] = client_send_time  # For logging

        start_time = client_send_time
        try:
            response = requests.post(server_url, json=request_data)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error contacting server: {e}")
            response_json = None  # Indicate failure

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
    all_timing_data: list,  # Pass list in to be mutated
    tick_history: list,  # 每个 tick 的 hit/miss/backup 记录
    metronome_enabled: bool,
    generation_interval_ticks: int,
    generation_length_ticks: int = None,  # total generation length, for experiments
    generation_length_per_request: int = None,  # for each request, we want the server to generate how many frames
    current_tick_ref: dict = None,  # Optional shared tick reference for MIDI file input
    listening_duration_ticks: int = 0,  # NEW: listening mode duration (0 = disabled)
    prompt_library: PromptLibrary = None,  # NEW: prompt library for listening mode
    server_url: str = None,  # NEW: server URL for listening mode
    key_detection_method: str = "lightweight",  # NEW: key detection method
):
    """
    Main tick loop for the client (main thread).
    Supports both direct generation and listening mode for adaptive prompting.
    """
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    number_of_hit = 0
    total_backup_level = 0

    # New state variables
    notes_for_next_request = []
    last_inference_timings = {}  # To persist timing info for display
    ticks_per_bar = ticks_per_beat * beats_per_bar

    # Listening mode state
    listening_mode_active = listening_duration_ticks > 0
    listening_mode_completed = False
    listening_worker_thread = None
    listening_worker_result_queue = Queue()
    GRACE_PERIOD_TICKS = 8  # Half a bar (8 ticks) for key detection processing

    if listening_mode_active:
        print(f"\n{'='*60}")
        print(f"LISTENING MODE ACTIVE")
        print(f"Will collect user input for {listening_duration_ticks} ticks")
        print(f"Then process key detection for {GRACE_PERIOD_TICKS} ticks (grace period)")
        print(f"Real-time generation starts at tick {listening_duration_ticks + GRACE_PERIOD_TICKS}")
        print(f"{'='*60}\n")

    # --- Main Loop ---
    while True:
        tick_count += 1

        if (
            generation_length_ticks is not None
            and tick_count >= generation_length_ticks
        ):
            print(
                f"Reached generation_length {generation_length_ticks} ticks — stopping tick loop."
            )
            print(
                f"hit rate: {number_of_hit}/{generation_length_ticks} = {number_of_hit / generation_length_ticks:.2%}"
            )
            print(
                f"average backup level: {total_backup_level / number_of_hit if number_of_hit > 0 else 0:.2f}"
            )
            # 可在此放置清理/通知逻辑，例如向其他队列放置终止事件
            return

        # Update shared tick reference for MIDI file input
        if current_tick_ref is not None:
            current_tick_ref["current_tick"] = tick_count

        # --- 1. Process User Input ---
        user_notes_this_tick = []

        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                inference_request_queue.put(None)
                return

            # --- Note Quantization & Audio Playback ---
            if event["type"] == "note_on":
                # 1. Quantize the note for the inference engine request.
                # All user notes are given a fixed duration for the model.
                quantized_note = {
                    "pitch": event["pitch"],
                    "tick": tick_count - 1,
                    "duration": DEFAULT_NOTE_DURATION_TICKS,
                }
                notes_for_next_request.append(quantized_note)
                user_notes_this_tick.append(quantized_note)
                midi_file_handler.add_user_note(quantized_note)

                # 2. Play the note immediately for audio feedback.
                audio_output_handler.on(event["pitch"], event["velocity"])

            elif event["type"] == "note_off":
                # Pass the note_off event directly to the audio handler
                audio_output_handler.off(event["pitch"])

        # --- 2. Handle Inference Responses ---
        while not inference_response_queue.empty():
            response_data, round_trip_time, request_data = (
                inference_response_queue.get()
            )

            if response_data:
                # --- Log the complete inference event ---
                json_log_handler.log_inference_event(request_data, response_data)

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
                timings["total_network_latency"] = (
                    round_trip_time - server_processing_duration
                )

                all_timing_data.append(timings)

                # --- Tick Consistency Filter ---
                newly_generated_notes = response_data["accompaniment"]

                gen_start = None
                if isinstance(request_data, dict):
                    gen_start = request_data.get("generation_start_tick")

                # 如果知道期望帧数和 generation 起始 tick，补全缺失 ticks（占位 pitch=-1）
                if generation_length_per_request is not None and gen_start is not None:
                    # map existing ticks
                    existing_ticks = {n["tick"] for n in newly_generated_notes}
                    for t in range(
                        gen_start,
                        gen_start
                        + (generation_length_per_request + 1) // 2,  # 向上取整
                    ):
                        if t not in existing_ticks:
                            newly_generated_notes.append(
                                {
                                    "pitch": -1,
                                    "tick": t,
                                    "duration": DEFAULT_NOTE_DURATION_TICKS,
                                    "is_placeholder": True,
                                }
                            )
                    # 现在为所有生成的 note 按相对 tick 顺序简单赋 backup_level = tick - gen_start
                    for n in newly_generated_notes:
                        n["backup_level"] = int(n["tick"] - gen_start)
                else:
                    # 无法补全时给默认 backup_level=0
                    for n in newly_generated_notes:
                        n["backup_level"] = 0

                # --- Clear stale notes from the previous generation ---
                # This ensures that if a new response arrives before the old one is
                # fully played out, we only replace future model-generated notes.
                # User-played note_offs are preserved.
                if newly_generated_notes:
                    # Find the first tick where the new generation actually places a note.
                    # This prevents clearing old notes if there's a gap before the new music starts.
                    first_new_note_tick = min(
                        note["tick"] for note in newly_generated_notes
                    )

                    ticks_to_clean = [
                        t for t in playback_schedule if t >= first_new_note_tick
                    ]
                    for tick in ticks_to_clean:
                        # Filter out events sourced from the model, keep user events
                        playback_schedule[tick] = [
                            event
                            for event in playback_schedule[tick]
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
                        # Tag as a model-originated event
                        playback_schedule[note["tick"]].append(
                            {**note, "source": "model"}
                        )

                # Store timings for display, making them persistent
                last_inference_timings = timings

        # --- 3. LISTENING MODE CHECK (NEW) ---
        if listening_mode_active and not listening_mode_completed:
            # PHASE 1: At end of listening period, spawn worker thread
            if tick_count == listening_duration_ticks:
                print(f"\n{'='*60}")
                print(f"LISTENING PERIOD COMPLETE at tick {tick_count}")
                print(f"Collected {len(notes_for_next_request)} melody notes")
                print(f"Spawning background worker for key detection...")
                print(f"{'='*60}")

                # Spawn worker thread (non-blocking)
                if prompt_library and server_url:
                    listening_worker_thread = threading.Thread(
                        target=listening_mode_worker,
                        args=(
                            notes_for_next_request.copy(),  # Copy to avoid race conditions
                            listening_duration_ticks,
                            prompt_library,
                            server_url,
                            key_detection_method,
                            listening_worker_result_queue
                        ),
                        daemon=True
                    )
                    listening_worker_thread.start()
                else:
                    print("✗ Prompt library not initialized, skipping injection")
                    listening_mode_completed = True
                    notes_for_next_request = []

            # PHASE 2: Check if worker completed at end of grace period
            elif tick_count == listening_duration_ticks + GRACE_PERIOD_TICKS:
                print(f"\n{'='*60}")
                print(f"GRACE PERIOD COMPLETE at tick {tick_count}")
                print(f"Checking worker status...")
                print(f"{'='*60}")

                # Check if worker has finished
                if not listening_worker_result_queue.empty():
                    success = listening_worker_result_queue.get()
                    if success:
                        print("✓ Listening mode injection successful!")
                    else:
                        print("✗ Listening mode injection failed, continuing without prompt")
                else:
                    # Worker still running (took longer than grace period)
                    print("⚠ Warning: Key detection took longer than grace period")
                    print("⚠ Will wait for completion before starting generation...")
                    # Wait for worker to finish (blocking, but should be quick)
                    if listening_worker_thread and listening_worker_thread.is_alive():
                        listening_worker_thread.join(timeout=2.0)  # Max 2s wait
                    if not listening_worker_result_queue.empty():
                        success = listening_worker_result_queue.get()
                        print(f"✓ Worker completed with result: {success}")
                    else:
                        print("✗ Worker timed out, continuing without prompt")

                # Mark listening mode as complete
                listening_mode_completed = True

            # PHASE 3: During listening/grace period, display status and skip inference
            if not listening_mode_completed:
                # Calculate display info
                bar_count = tick_count // ticks_per_bar
                beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
                user_notes_this_tick_display = [n["pitch"] for n in user_notes_this_tick]
                pending_user_notes_display = [n["pitch"] for n in notes_for_next_request]

                # Determine status message
                if tick_count < listening_duration_ticks:
                    status_msg = f"Listening... ({listening_duration_ticks - tick_count} ticks remaining)"
                else:
                    status_msg = f"Processing key detection... ({listening_duration_ticks + GRACE_PERIOD_TICKS - tick_count} ticks remaining)"

                music_info = {
                    "bar": bar_count,
                    "beat": beat_in_bar,
                    "ticks_per_beat": ticks_per_beat,
                    "inference_triggered": False,
                    "all_timing_data": all_timing_data,
                }

                # Update display with status
                output_handler.update_and_display(
                    tick_count,
                    music_info,
                    user_notes_this_tick_display,
                    [],  # No model notes during listening/grace period
                    pending_user_notes_display,
                )

                # Play metronome during listening mode (IMPORTANT: before continue)
                if metronome_enabled:
                    is_beat_tick = (tick_count % ticks_per_beat) == 0
                    if is_beat_tick:
                        beat_in_bar_metro = (tick_count % ticks_per_bar) // ticks_per_beat
                        if beat_in_bar_metro == 0:
                            audio_output_handler.metro_first()
                        else:
                            audio_output_handler.metro_other()

                time.sleep(seconds_per_tick)
                continue  # Skip rest of loop (note playback, inference)

        # --- 4. Trigger New Inference (Latency-Aware) ---
        # Only trigger if listening mode is complete (or not active)
        # 感觉这个所谓的 lantency aware 没意义，而且我们的建模不是这样的，改成固定间隔
        # is_trigger_tick = (tick_count % generation_interval_ticks) == (generation_interval_ticks - LATENCY_OFFSET_TICKS)
        is_trigger_tick = (tick_count % generation_interval_ticks) == 0

        if is_trigger_tick:  # and notes_for_next_request:
            # The model should start generating from the beginning of the *next* generation interval.
            # current_interval_start_tick = (tick_count // generation_interval_ticks) * generation_interval_ticks
            # next_interval_start_tick = current_interval_start_tick + generation_interval_ticks
            next_interval_start_tick = tick_count + 1

            request_data = {
                "melody_notes": notes_for_next_request,
                "generation_start_tick": next_interval_start_tick,
            }
            # 如果在 CLI/外部设置了每次请求的生成长度，传给服务器（frames）
            if generation_length_per_request is not None:
                request_data["generation_length_frames"] = generation_length_per_request
            else:
                print(
                    "Warning: generation_length_per_request is not set, server may use default length 5."
                )
                request_data["generation_length_frames"] = 5  # 默认值
            inference_request_queue.put(
                (request_data, request_data.copy())
            )  # Pass a copy for logging
            notes_for_next_request = []  # Clear the buffer

        # --- 5. Play Scheduled Notes ---
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
            else:  # It's a note_on
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
            tick_history.append(tick_record)
        except Exception:
            # 不让记录失败影响主循环
            pass
        # Process note-offs first
        for event in notes_to_stop_this_tick:
            audio_output_handler.off(event["pitch"])

        # Process note-ons and schedule their corresponding note-offs
        for event in notes_to_play_this_tick:
            # This loop only processes model-generated notes.
            audio_output_handler.on(
                event["pitch"], audio_output_handler.accompaniment_velocity
            )
            midi_file_handler.add_model_note(event)

            note_off_tick = tick_count + event["duration"]
            if note_off_tick not in playback_schedule:
                playback_schedule[note_off_tick] = []

            # The source tag is preserved from the original event
            playback_schedule[note_off_tick].append(
                {**event, "type": "note_off", "source": "model"}
            )

        # --- 6. Metronome ---
        if metronome_enabled:
            is_beat_tick = (tick_count % ticks_per_beat) == 0
            if is_beat_tick:
                beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
                if beat_in_bar == 0:
                    audio_output_handler.metro_first()
                else:
                    audio_output_handler.metro_other()

        # --- 7. Update Display ---
        bar_count = tick_count // ticks_per_bar
        beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat

        pending_user_notes_display = [n["pitch"] for n in notes_for_next_request]
        user_notes_this_tick_display = [n["pitch"] for n in user_notes_this_tick]
        model_notes_for_display = [n["pitch"] for n in notes_to_play_this_tick]

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
            pending_user_notes_display,
        )

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
            Music Injection:
            --injection-file path/to/music.mid --injection-length 100

            Examples:
            %(prog)s
            %(prog)s --use-keyboard-input --tempo 120
            %(prog)s --midi-file-input song.mid --midi-file-delay-ticks 8
             %(prog)s --injection-file prelude.mid --injection-length 50 --use-keyboard-input
        """,
    )

    # Network arguments
    parser.add_argument(
        "--server_url",
        type=str,
        default=config.DEFAULT_SERVER_URL,
        help="URL of the StreamMUSE server",
    )

    # Musical timing arguments
    parser.add_argument(
        "--tempo", type=float, default=config.DEFAULT_TEMPO, help="Tempo in BPM"
    )
    parser.add_argument(
        "--ticks_per_beat",
        type=int,
        default=config.DEFAULT_TICKS_PER_BEAT,
        help="Number of ticks per beat",
    )
    parser.add_argument(
        "--beats_per_bar",
        type=int,
        default=config.DEFAULT_BEATS_PER_BAR,
        help="Number of beats per bar",
    )
    parser.add_argument(
        "--generation_interval_ticks",
        type=int,
        default=config.DEFAULT_GENERATION_INTERVAL_TICKS,
        help="Number of ticks between generation requests",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=config.DEFAULT_GENERATION_LENGTH,
        help="Number of frames to generate in total (for experiments only)",
    )
    parser.add_argument(
        "--generation_length_per_request",
        type=int,
        default=None,
        help="Number of frames to generate per request (for experiments only)",
    )

    # Display arguments
    parser.add_argument(
        "--log_lines",
        type=int,
        default=config.DEFAULT_LOG_LINES,
        help="Number of log lines to display",
    )
    parser.add_argument(
        "--metronome", action="store_true", help="Enable audible MIDI metronome click"
    )

    # MIDI I/O arguments
    parser.add_argument(
        "--midi_output_name",
        type=str,
        default=None,
        help="Specify MIDI output port name",
    )
    parser.add_argument(
        "--midi_input_name", type=str, default=None, help="Specify MIDI input port name"
    )
    parser.add_argument(
        "--accompaniment-velocity",
        type=int,
        default=config.DEFAULT_ACCOMPANIMENT_VELOCITY,
        help="MIDI velocity for generated accompaniment notes (0-127)",
    )

    # Input mode arguments
    parser.add_argument(
        "--use-keyboard-input",
        action="store_true",
        help="Use computer keyboard as MIDI input",
    )
    parser.add_argument(
        "--midi-file-input",
        type=str,
        default=None,
        help="Path to MIDI file to simulate user input",
    )
    parser.add_argument(
        "--midi-file-delay-ticks",
        type=int,
        default=config.DEFAULT_MIDI_FILE_DELAY_TICKS,
        help="Number of ticks to delay before MIDI file starts playing",
    )
    parser.add_argument(
        "--midi-file-use-original-duration",
        action="store_true",
        help="Use original MIDI note durations instead of fixed duration",
    )

    # 添加音乐注入参数 (Direct injection mode)
    parser.add_argument(
        "--injection-file",
        type=str,
        default=None,
        help="Path to MIDI file to inject into inference engine history (client-side)",
    )
    parser.add_argument(
        "--injection-length",
        type=int,
        default=0,
        help="Number of ticks to inject from the injection file",
    )

    # Listening mode arguments (Auto key detection and prompt selection)
    parser.add_argument(
        "--listening-duration-ticks",
        type=int,
        default=config.DEFAULT_LISTENING_DURATION_TICKS,
        help="Number of ticks to listen before auto-injecting prompt based on detected key (0=disabled)",
    )
    parser.add_argument(
        "--prompt-library-path",
        type=str,
        default=config.DEFAULT_PROMPT_LIBRARY_PATH,
        help="Path to local prompt library directory (for listening mode)",
    )
    parser.add_argument(
        "--key-detection-method",
        type=str,
        choices=["lightweight", "music21"],
        default=config.DEFAULT_KEY_DETECTION_METHOD,
        help="Key detection algorithm: 'lightweight' (fast) or 'music21' (requires music21 library)",
    )

    args = parser.parse_args()

    # --- Validate Arguments ---
    if not StreamMUSEConfig.validate_args(args):
        return

    # Validate injection modes (mutually exclusive)
    has_direct_injection = args.injection_file is not None
    has_listening_mode = args.listening_duration_ticks > 0

    if has_direct_injection and has_listening_mode:
        print("Error: Cannot use both --injection-file and --listening-duration-ticks")
        print("Choose one injection mode:")
        print("  Direct mode: --injection-file <path> --injection-length <ticks>")
        print("  Listening mode: --listening-duration-ticks <ticks>")
        return

    # Validate direct injection parameters
    if has_direct_injection:
        if args.injection_length <= 0:
            print("Error: --injection-length must be positive when --injection-file is specified")
            return
        if not os.path.exists(args.injection_file):
            print(f"Error: injection file not found: {args.injection_file}")
            return

    # --- Initialize Prompt Library (for listening mode) ---
    prompt_library = None
    if has_listening_mode:
        print(f"Initializing prompt library from: {args.prompt_library_path}")
        prompt_library = PromptLibrary(args.prompt_library_path)
        if not prompt_library.metadata:
            print("Warning: Prompt library is empty")
            print(f"Please create prompts in {args.prompt_library_path} with metadata.json")

    # --- Create Session Log Directory ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    session_log_dir = os.path.join("app", "logs", f"session_{timestamp}")
    os.makedirs(session_log_dir, exist_ok=True)

    # 清除服务器历史
    if not clear_server_history(args.server_url):
        return

    # --- 处理音乐注入 (Direct Injection Mode) ---
    injection_offset_ticks = 0
    if has_direct_injection:
        print("\n=== DIRECT INJECTION MODE ===")
        injection_offset_ticks = perform_direct_injection(
            args.injection_file,
            args.injection_length,
            args.server_url,
            args.ticks_per_beat
        )

        if injection_offset_ticks == 0:
            print("✗ Direct injection failed, exiting")
            return
        else:
            print(f"✓ Direct injection successful: {injection_offset_ticks} ticks")
            # 保存 prompt MIDI 文件
            save_prompt_midi(
                args.injection_file,
                args.injection_length,
                session_log_dir,
                args.ticks_per_beat,
            )
    elif has_listening_mode:
        print("\n=== LISTENING MODE ===")
        print(f"Will listen for {args.listening_duration_ticks} ticks before auto-injecting prompt")
        print(f"Key detection method: {args.key_detection_method}")

    event_queue = Queue()
    inference_request_queue = Queue()
    inference_response_queue = Queue()
    audio_output_handler = AudioOutputHandler(
        port_name=args.midi_output_name,
        accompaniment_velocity=args.accompaniment_velocity,
    )
    output_handler = CLIOutputHandler(args.log_lines)
    midi_file_handler = MidiFileHandler(args.tempo, args.ticks_per_beat)
    json_log_handler = JsonLogHandler()
    all_timing_data = []  # Initialize list in main scope
    tick_history = []  # 用于记录每个 tick system metrics

    # Create shared reference for current tick count (for MIDI file input)
    current_tick_ref = {"current_tick": 0}

    if args.midi_file_input:
        # MIDI file input mode - 考虑偏移，以及跳过注入的部分
        print(f"Using MIDI file input: {args.midi_file_input}")

        input_thread = threading.Thread(
            target=read_midi_file_input,
            args=(
                event_queue,
                args.midi_file_input,
                current_tick_ref,
                args.tempo,
                args.ticks_per_beat,
                args.midi_file_delay_ticks,
                injection_offset_ticks,
                args.midi_file_use_original_duration,
                DEFAULT_NOTE_DURATION_TICKS,
            ),
            daemon=True,
        )
    elif args.use_keyboard_input:
        input_thread = threading.Thread(
            target=read_keyboard_input, args=(event_queue,), daemon=True
        )
    else:
        # A check to see if MIDI input is available.
        try:
            if not mido.get_input_names():
                print(
                    "No MIDI input devices found. Please connect a MIDI device or use --use-keyboard-input."
                )
                return
            midi_input_name = args.midi_input_name or mido.get_input_names()[0]
        except Exception as e:
            print(f"Could not list MIDI devices: {e}")
            return

        input_thread = threading.Thread(
            target=read_midi_input, args=(event_queue, midi_input_name), daemon=True
        )

    inference_thread = threading.Thread(
        target=inference_worker,
        args=(inference_request_queue, inference_response_queue, args.server_url),
        daemon=True,
    )

    if args.generation_length is not None:
        generation_length_ticks = args.generation_length // 2
    else:
        generation_length_ticks = None
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
            tick_history,
            args.metronome,
            args.generation_interval_ticks,
            generation_length_ticks,
            args.generation_length_per_request,
            current_tick_ref,
            args.listening_duration_ticks,  # NEW: listening mode duration
            prompt_library,  # NEW: prompt library instance
            args.server_url,  # NEW: server URL for listening mode
            args.key_detection_method,  # NEW: key detection method
        ),
        daemon=True,
    )

    print("Starting StreamMUSE Client")
    print(f"Connecting to server at {args.server_url}")

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
        if args.generation_length is None:
            print("\n--- Saving all session logs ---")
            # Pass the benchmark data to be saved
            output_handler.save_log_on_exit(session_log_dir, all_timing_data)
            midi_file_handler.save_to_midi(session_log_dir)
            json_log_handler.save_logs(session_log_dir)
            try:
                with open(
                    os.path.join(session_log_dir, "tick_history.json"), "w"
                ) as fh:
                    json.dump(tick_history, fh, indent=2)
                print(
                    f"✓ tick_history 已保存到: {os.path.join(session_log_dir, 'tick_history.json')}"
                )
            except Exception as e:
                print(f"✗ 保存 tick_history 失败: {e}")
            audio_output_handler.close()
        else:
            print(
                "\nExperiment Mode: Generation length reached, exiting without saving logs."
            )
            test_midi_file_name = os.path.splitext(
                os.path.basename(args.midi_file_input)
            )[0]
            base_log_dir = f"experiments_local_server/realtime/baseline/interval_{args.generation_interval_ticks}_gen_frame_{args.generation_length_per_request}/prompt_{args.injection_length}_gen_{args.generation_length}"
            session_log_dir = os.path.join(
                base_log_dir, "batch_run", test_midi_file_name
            )
            os.makedirs(session_log_dir, exist_ok=True)
            output_handler.save_log_on_exit(session_log_dir, all_timing_data)
            experiment_dir = f"{base_log_dir}/generated"
            os.makedirs(experiment_dir, exist_ok=True)
            midi_file_handler.save_to_midi(
                experiment_dir, midi_file_name=test_midi_file_name
            )
            json_log_handler.save_logs(session_log_dir)
            audio_output_handler.close()
            try:
                with open(
                    os.path.join(session_log_dir, "tick_history.json"), "w"
                ) as fh:
                    json.dump(tick_history, fh, indent=2)
                print(
                    f"✓ tick_history 已保存到: {os.path.join(session_log_dir, 'tick_history.json')}"
                )
            except Exception as e:
                print(f"✗ 保存 tick_history 失败: {e}")


if __name__ == "__main__":
    main()

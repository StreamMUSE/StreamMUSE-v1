"""
This is the CLI output handler for the StreamMUSE end to end system.
"""

from collections import deque
import shutil
import time
import os
import csv

class CLIOutputHandler:
    """
    Handles displaying a persistent log with updating status lines at the bottom.
    """

    def __init__(self, log_display_count: int = 10):
        """
        Initialize the CLI output handler.
        """
        self.log_display_count = log_display_count
        self.status_line_count = 4 # Now 4 lines: Pending, Last Played, Time, Timings
        self.total_managed_lines = self.status_line_count + self.log_display_count

        # Use deque for maintaining fixed size history
        self.log_history = deque(maxlen=100)
        self.last_model_output_str = "None"
        self.is_first_display = True
        self.all_round_trip_times = []

    def _midi_to_note_name(self, pitch: int) -> str:
        """
        Converts a MIDI pitch number (0-127) to its scientific pitch notation.
        e.g., 60 -> "C4", 69 -> "A4", 38 -> "D2"
        """
        if not 0 <= pitch <= 127:
            return "N/A"
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        return f"{note_names[note_index]}{octave}"
    
    def update_and_display(self, tick_count, music_info, user_notes_this_tick, model_notes_played, pending_user_notes):
        """
        Update the display with the latest log and status lines.
        """
        # --- Prepare strings for display ---
        user_this_tick_str = ', '.join([self._midi_to_note_name(pitch) for pitch in user_notes_this_tick])
        model_notes_str = ', '.join([self._midi_to_note_name(pitch) for pitch in model_notes_played])
        pending_notes_str = ', '.join([self._midi_to_note_name(pitch) for pitch in pending_user_notes]) or "None"

        # --- Build Log Entry ---
        ticks_per_beat = music_info.get("ticks_per_beat", 4)
        bar = music_info.get("bar", 0)
        beat = music_info.get("beat", 0)
        tick_in_beat = (tick_count % ticks_per_beat)

        log_entry = f"[{bar:03d}.{beat}.{tick_in_beat}]"
        
        # Define a fixed width for the user notes part to align the '|' separator.
        # This is calculated to fit up to 4 notes with long names (e.g., 'G#-1').
        USER_PART_WIDTH = 32

        # Prepare the user and model parts of the log line
        user_display_part = ""
        if user_notes_this_tick:
            user_display_part = f" USER: [{user_this_tick_str}]"
        
        model_display_part = ""
        if model_notes_played:
            model_display_part = f" MODEL: [{model_notes_str}]"

        # Combine them for the final log entry, ensuring alignment
        if user_notes_this_tick or model_notes_played:
            log_entry += user_display_part.ljust(USER_PART_WIDTH) + " |" + model_display_part
        else:
            log_entry += " ..."

        # Update log history
        self.log_history.append(log_entry)

        # --- Build Status Lines ---
        # Line 1: Shows the buffer of notes waiting to be sent for inference
        inputs_line = f'PENDING USER NOTES: {pending_notes_str}'

        # Line 2: Shows the last notes played by the model
        if music_info.get('inference_triggered'):
            self.last_model_output_str = "Waiting for server..."
        elif model_notes_played:
            self.last_model_output_str = model_notes_str or "None"
        else:
            # Clear the status if no notes were played on this tick
            self.last_model_output_str = "None"

        output_line = f"LAST MODEL NOTES PLAYED: {self.last_model_output_str}"

        # Line 3: Real-time music time
        time_status_line = f"TIME: Bar {bar:03d}, Beat {beat}"
        
        # Line 4: Last inference timings (persistent)
        if "round_trip_time" in music_info:
            all_times = music_info.get("all_round_trip_times", [])
            last_time = music_info['round_trip_time']
            avg_time = sum(all_times) / len(all_times) if all_times else 0
            count = len(all_times)
            
            # Detailed breakdown
            server_total = music_info['response_output_time'] - music_info['request_arrival_time']
            inference_total = music_info['inference_end_time'] - music_info['inference_start_time']
            
            timing_status_line = f"TIMING: Round-trip: {last_time:.3f}s (Avg: {avg_time:.3f}s) | Server: {server_total:.3f}s | Inference: {inference_total:.3f}s"
        else:
            timing_status_line = "TIMING: Waiting for first inference..."
        
        # 2. --- Render the display ---

        if self.is_first_display:
            print("\n" * self.total_managed_lines, end="")
            self.is_first_display = False

        # Move cursor up to the top of the managed area
        print(f"\r\x1b[{self.total_managed_lines}A", end="")

        # Get the last N log lines to display
        display_logs = list(self.log_history)[-self.log_display_count:]
        
        # Print logs
        for i in range(self.log_display_count):
            line_content = display_logs[i] if i < len(display_logs) else ""
            print(f"\x1b[2K{line_content}", flush=True)

        # Print separator and status lines
        try:
            terminal_width = shutil.get_terminal_size().columns
        except OSError:
            terminal_width = 80 # Default width
        print(f"\x1b[2K" + "═"*terminal_width, flush=True) # Changed to a different separator
        print(f"\x1b[2K{inputs_line}", flush=True)
        print(f"\x1b[2K{output_line}", flush=True)
        print(f"\x1b[2K{time_status_line}", flush=True)
        print(f"\x1b[2K{timing_status_line}", flush=True)

    def save_log_on_exit(self, session_log_dir: str, all_timing_data: list, log_file_prefix="session"):
        """Saves collected timings to a .txt summary and a detailed .csv file."""
        if not all_timing_data:
            print("\nNo inference times to log.")
            return

        # --- Save Detailed CSV Log ---
        csv_filename = f"{log_file_prefix}.csv"
        csv_filepath = os.path.join(session_log_dir, csv_filename)
        
        # Define header based on the keys in the timing data dictionary
        header = list(all_timing_data[0].keys())

        try:
            with open(csv_filepath, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(all_timing_data)
        except IOError as e:
            print(f"\nError saving CSV log file: {e}")
            return # Don't proceed if CSV fails

        # --- Calculate Averages for Text Log ---
        num_inferences = len(all_timing_data)
        avg_round_trip = sum(d['round_trip_time'] for d in all_timing_data) / num_inferences
        min_round_trip = min(d['round_trip_time'] for d in all_timing_data)
        max_round_trip = max(d['round_trip_time'] for d in all_timing_data)

        # --- Save Summary .txt Log ---
        txt_filename = f"{log_file_prefix}.txt"
        txt_filepath = os.path.join(session_log_dir, txt_filename)

        try:
            with open(txt_filepath, "w") as f:
                f.write("--- StreamMUSE Client Session Summary ---\n")
                f.write(f"Timestamp: {os.path.basename(session_log_dir)}\n")
                f.write(f"Total Inferences Logged: {num_inferences}\n")
                f.write(f"Average Round-Trip Time: {avg_round_trip:.4f}s\n")
                f.write(f"Min Round-Trip Time: {min_round_trip:.4f}s\n")
                f.write(f"Max Round-Trip Time: {max_round_trip:.4f}s\n")
            
            # --- Create Visual Confirmation ---
            try:
                width = min(80, shutil.get_terminal_size().columns - 4)
            except OSError:
                width = 76

            title = "LOGS SAVED"
            path_str_1 = f"Summary: {txt_filepath}"
            path_str_2 = f"Detailed CSV: {csv_filepath}"

            print("\n")
            print("+" + "-" * width + "+")
            print("|" + " " * width + "|")
            print(f"|{title:^{width}}|")
            print("|" + " " * width + "|")
            print(f"|{path_str_1:<{width}}|")
            print(f"|{path_str_2:<{width}}|")
            print("|" + " " * width + "|")
            print("+" + "-" * width + "+\n")

        except IOError as e:
            print(f"\nError saving summary log file: {e}")

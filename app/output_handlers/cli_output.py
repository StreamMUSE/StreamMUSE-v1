"""
This is the CLI output handler for the StreamMUSE end to end system.
"""

from collections import deque
import shutil
import time

class CLIOutputHandler:
    """
    Handles displaying a persistent log with updating status lines at the bottom.
    """

    def __init__(self, log_display_count: int = 10):
        """
        Initialize the CLI output handler.
        """
        self.log_display_count = log_display_count
        self.status_line_count = 3 # Changed from 4 to 3
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
    
    def update_and_display(self, tick_count, music_info, user_notes_played, model_notes_played):
        """
        Update the display with the latest log and status lines.
        """
        user_notes_str = ', '.join([self._midi_to_note_name(pitch) for pitch in user_notes_played]) or "None"
        model_notes_str = ', '.join([self._midi_to_note_name(pitch) for pitch in model_notes_played]) or "None"

        ticks_per_beat = music_info.get("ticks_per_beat", 4)
        bar = music_info.get("bar", 0)
        beat = music_info.get("beat", 0)
        tick_in_beat = (tick_count % ticks_per_beat)

        log_entry = f"[{bar:03d}.{beat}.{tick_in_beat}]"
        
        if user_notes_played:
            log_entry += f" USER: [{user_notes_str}]"
        if model_notes_played:
            log_entry += f" | MODEL: [{model_notes_str}]"
        
        # Placeholder for no note events
        if not user_notes_played and not model_notes_played:
            log_entry += " ..."

        # Update log history
        self.log_history.append(log_entry)

        # Line 1: Inputs during this tick
        inputs_line = f'PENDING USER NOTES: {user_notes_str}'

        # Line 2: Model output
        if music_info.get('inference_triggered'):
            self.last_model_output_str = "Waiting for server..."
        elif model_notes_played:
            self.last_model_output_str = model_notes_str

        output_line = f"LAST MODEL NOTES PLAYED: {self.last_model_output_str}"

        # Line 3: General status and benchmarking
        status_line = f"TIME: Bar {bar:03d}, Beat {beat} |"
        
        if "round_trip_time" in music_info:
            last_time = music_info['round_trip_time']
            self.all_round_trip_times.append(last_time)
            avg_time = sum(self.all_round_trip_times) / len(self.all_round_trip_times)
            count = len(self.all_round_trip_times)
            status_line += f" Round-trip: {last_time:.3f}s (Avg: {avg_time:.3f}s over {count} calls)"
        else:
            status_line += " Waiting for first inference..."
        
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
        print(f"\x1b[2K{status_line}", flush=True)

    def save_log_on_exit(self, log_file_prefix="inference_log"):
        """Saves the collected inference times to a file on exit."""
        if not self.all_round_trip_times:
            print("\nNo inference times to log.")
            return

        avg_time = sum(self.all_round_trip_times) / len(self.all_round_trip_times)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"{log_file_prefix}_{timestamp}.txt"

        try:
            with open(log_file, "w") as f:
                f.write("--- StreamMUSE Client Round-Trip Benchmark Log ---\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Total Inferences Logged: {len(self.all_round_trip_times)}\n")
                f.write(f"Average Round-Trip Time: {avg_time:.4f}s\n")
                f.write(f"Min Round-Trip Time: {min(self.all_round_trip_times):.4f}s\n")
                f.write(f"Max Round-Trip Time: {max(self.all_round_trip_times):.4f}s\n")
                f.write("\n--- Raw Data (seconds) ---\n")
                for i, t in enumerate(self.all_round_trip_times):
                    f.write(f"Inference {i+1}: {t:.6f}\n")
            
            print(f"\nRound-trip log saved to {log_file}")
        except IOError as e:
            print(f"\nError saving log file: {e}")

"""
This is the log output handler for the StreamMUSE end to end system.
"""

import time
import os

class LogOutputHandler:
    """
    Handles logging messages to a file.
    """
    def __init__(self, log_file_path: str = "~/app/logs", log_file_prefix: str = "log"):
        """
        Initialize the log output handler.
        """
        self.log_file_path = log_file_path
        self.log_file_prefix = log_file_prefix

    def save_benchmark_log(self, benchmark_log: list[dict], path_to_benchmark_log: str):
        """
        Save the benchmark log to a file.
        """
        time = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{self.log_file_prefix}_{time}.log"
        
        try:
            with open(path_to_benchmark_log, "w") as f:
                f.write(benchmark_log)
        except Exception as e:
            print(f"Error saving log: {e}")
        ...

    def save_midi(self, midi_log: str, path_to_midi_file: str):
        """
        Save the MIDI file to a file.
        """
        ...

    def save_summary_log(self, summary_log: list[dict], path_to_summary_log: str):
        """
        Save the summary log to a file.
        """
        ...

    def log_on_exit(self, benchmark_log: list[dict], midi_log: str, summary_log: list[dict]):
        """
        Log a message to the log file.
        """
        benchmark_path = os.path.join(self.log_file_path, "benchmark.csv")
        midi_path = os.path.join(self.log_file_path, "midi.mid")
        summary_path = os.path.join(self.log_file_path, "summary.txt")

        self.save_benchmark_log(benchmark_log, benchmark_path)
        self.save_midi(midi_log, midi_path)
        self.save_summary_log(summary_log, summary_path)
        

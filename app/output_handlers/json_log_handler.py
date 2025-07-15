"""
Handles saving detailed JSON logs of the session.
"""

import json
import os

class JsonLogHandler:
    """
    Collects detailed information about the session and saves it to JSON files.
    """

    def __init__(self):
        self.inference_log = []

    def log_inference_event(self, request_data: dict, response_data: dict):
        """
        Logs a single, complete inference event (request and response).

        Args:
            request_data (dict): The data sent to the server.
            response_data (dict): The full JSON response from the server.
        """
        self.inference_log.append({
            "request": request_data,
            "response": response_data
        })

    def save_logs(self, session_log_dir: str, log_filename: str = "inferences.json"):
        """
        Saves all collected logs to their respective files in the session directory.
        """
        if not self.inference_log:
            return

        log_filepath = os.path.join(session_log_dir, log_filename)
        try:
            with open(log_filepath, "w") as f:
                json.dump(self.inference_log, f, indent=2)
            print(f"Raw inference JSON log saved to: {log_filepath}")
        except IOError as e:
            print(f"\nError saving JSON log file: {e}") 
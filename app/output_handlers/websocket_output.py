"""
WebSocket Output Handler for StreamMUSE Web UI

This handler sends real-time events to the browser via WebSocket.
It uses a thread-safe queue to bridge sync client code with async WebSocket.
"""

import json
import queue
import time
from typing import Optional


class WebSocketOutputHandler:
    """
    Output handler that queues messages for WebSocket broadcast.
    
    Thread-safe: can be called from sync threads (tick_loop, etc.)
    Messages are consumed by async WebSocket handler.
    """
    
    def __init__(self):
        self.message_queue = queue.Queue()
        self.is_connected = False
    
    def _send(self, message: dict):
        """Queue a message for sending."""
        self.message_queue.put(json.dumps(message))
    
    def send_tick(self, tick: int, bar: int, beat: int):
        """Send tick update to browser."""
        self._send({
            "type": "tick",
            "tick": tick,
            "bar": bar,
            "beat": beat,
            "timestamp": time.time()
        })
    
    def send_note_on(
        self,
        pitch: int,
        velocity: int,
        tick: int,
        duration: int,
        source: str,
        backup_level: int = 0,
        is_late: bool = False,
        was_swapped: bool = False
    ):
        """
        Send note_on event to browser.
        
        Args:
            pitch: MIDI pitch (0-127)
            velocity: MIDI velocity (0-127)
            tick: Tick when note starts
            duration: Note duration in ticks
            source: 'user' or 'model'
            backup_level: For model notes, which backup was used (0=on-time)
            is_late: Whether this note arrived late
            was_swapped: Whether this note replaced another
        """
        self._send({
            "type": "note",
            "event": "on",
            "pitch": pitch,
            "velocity": velocity,
            "tick": tick,
            "duration": duration,
            "source": source,
            "backup_level": backup_level,
            "is_late": is_late,
            "was_swapped": was_swapped
        })
    
    def send_note_off(self, pitch: int, tick: int, source: str):
        """Send note_off event to browser."""
        self._send({
            "type": "note",
            "event": "off",
            "pitch": pitch,
            "tick": tick,
            "source": source
        })
    
    def send_stats(
        self,
        hit_rate: Optional[float] = None,
        avg_backup_level: Optional[float] = None,
        round_trip_ms: Optional[float] = None,
        server_process_ms: Optional[float] = None,
        network_latency_ms: Optional[float] = None,
        total_hits: Optional[int] = None,
        total_ticks: Optional[int] = None
    ):
        """Send stats update to browser."""
        stats = {"type": "stats"}
        
        if hit_rate is not None:
            stats["hit_rate"] = hit_rate
        if avg_backup_level is not None:
            stats["avg_backup_level"] = avg_backup_level
        if round_trip_ms is not None:
            stats["round_trip_ms"] = round_trip_ms
        if server_process_ms is not None:
            stats["server_process_ms"] = server_process_ms
        if network_latency_ms is not None:
            stats["network_latency_ms"] = network_latency_ms
        if total_hits is not None:
            stats["total_hits"] = total_hits
        if total_ticks is not None:
            stats["total_ticks"] = total_ticks
        
        self._send(stats)
    
    def send_config(self, config: dict):
        """Send config update to browser."""
        self._send({
            "type": "config",
            **config
        })
    
    def send_status(self, state: str, message: str = ""):
        """
        Send status update to browser.
        
        Args:
            state: 'stopped', 'running', 'listening', 'error'
            message: Optional status message
        """
        self._send({
            "type": "status",
            "state": state,
            "message": message
        })
    
    def send_inference_result(
        self,
        request_tick: int,
        generation_start_tick: int,
        num_notes_generated: int,
        round_trip_ms: float,
        notes: list
    ):
        """
        Send inference result details to browser.
        Useful for visualizing what the model generated.
        """
        self._send({
            "type": "inference_result",
            "request_tick": request_tick,
            "generation_start_tick": generation_start_tick,
            "num_notes_generated": num_notes_generated,
            "round_trip_ms": round_trip_ms,
            "notes": notes
        })
    
    def get_pending_messages(self) -> list[str]:
        """
        Get all pending messages from queue.
        Called by async WebSocket handler.
        
        Returns:
            List of JSON-encoded messages
        """
        messages = []
        while True:
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return messages
    
    def clear_queue(self):
        """Clear any pending messages."""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

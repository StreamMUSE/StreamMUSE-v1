"""
Timing information returned from inference.

Holds server-side timings; optional client-side fields can be set when
the response is received (e.g. round_trip_time, server_processing_duration).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimingInfo:
    """
    Timing data for an inference call.

    Server timings are always present; client timings are optional
    (e.g. in debug mode).
    """

    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float
    # Optional client-side (set when response is received)
    round_trip_time: Optional[float] = None
    server_processing_duration: Optional[float] = None
    total_network_latency: Optional[float] = None

"""Domain protocols: input, output, inference."""

from streammuse.domain.interfaces.inference import InferenceEngine
from streammuse.domain.interfaces.input import InputSource
from streammuse.domain.interfaces.output import OutputSink
from streammuse.domain.interfaces.timing_info import TimingInfo

__all__ = ["InferenceEngine", "InputSource", "OutputSink", "TimingInfo"]

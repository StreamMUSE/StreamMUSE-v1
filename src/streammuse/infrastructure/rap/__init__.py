"""Infrastructure adapters for rap-alignment candidate generation."""

from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog, PrevalidatedFallbackLine
from streammuse.infrastructure.rap.generators import (
    IndependentChoiceCandidateGenerator,
    LocalChatCandidateGenerator,
    PhraseBankGenerator,
)
from streammuse.infrastructure.rap.recorder import (
    RapSessionManifest,
    RapSessionRecorder,
    build_session_manifest,
    derive_bar_rows,
    derive_summary,
    validate_session_manifest,
)
from streammuse.infrastructure.rap.speech import CommandRunner, EspeakPhonemeSynthesizer, arpabet_syllable_to_espeak
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.audio_output import (
    CompositeAudioSink,
    Float32WavAudioSink,
    NullAudioSink,
    SoundDeviceAudioSink,
    TimedAudioSink,
)

__all__ = [
    "LocalChatCandidateGenerator",
    "IndependentChoiceCandidateGenerator",
    "PhraseBankGenerator",
    "PrevalidatedFallbackCatalog",
    "PrevalidatedFallbackLine",
    "RapSessionManifest",
    "RapSessionRecorder",
    "build_session_manifest",
    "derive_bar_rows",
    "derive_summary",
    "validate_session_manifest",
    "CommandRunner",
    "EspeakPhonemeSynthesizer",
    "arpabet_syllable_to_espeak",
    "ProceduralBoomBapRenderer",
    "CompositeAudioSink",
    "Float32WavAudioSink",
    "NullAudioSink",
    "SoundDeviceAudioSink",
    "TimedAudioSink",
]

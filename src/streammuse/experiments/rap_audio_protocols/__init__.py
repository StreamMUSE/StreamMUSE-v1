"""Backend-neutral corpus contracts for rap audio protocol comparisons."""

from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    SongCorpus,
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_bytes,
    canonical_json_dumps,
    sha256_hex,
)
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus

__all__ = [
    "ChunkRenderRecord",
    "ProtocolId",
    "SongCorpus",
    "SyllableTarget",
    "TwoBarRenderRequest",
    "canonical_json_bytes",
    "canonical_json_dumps",
    "load_song_corpus",
    "sha256_hex",
]

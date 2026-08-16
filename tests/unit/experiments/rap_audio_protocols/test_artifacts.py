from __future__ import annotations

import json
from pathlib import Path

import pytest

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    build_protocol_artifact_manifest,
    chunk_record_is_complete,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.audio import render_common_drums, write_listening_wav
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, SyllableTarget, TwoBarRenderRequest


def _request(chunk_index: int) -> TwoBarRenderRequest:
    start_bar = chunk_index * 2
    syllables = tuple(
        SyllableTarget(
            word=f"word{index // 2}",
            index_in_word=index % 2,
            phonemes=("W", "ER1", "D"),
            lexical_stress=1 if index % 3 == 0 else 0,
            target_stress=1.0 if index % 4 == 0 else 0.4,
            boundary_strength=3 if index == 17 else 0,
            absolute_tick=start_bar * 16 + index,
            tick_in_chunk=index,
            target_seconds=index / 6,
        )
        for index in range(18)
    )
    return TwoBarRenderRequest(
        song_id="01_space_exploration",
        chunk_index=chunk_index,
        start_bar=start_bar,
        end_bar=start_bar + 2,
        text=f"chunk {chunk_index}",
        syllables=syllables,
    )


def _record(*, protocol_id: ProtocolId = ProtocolId.MOSS_GLOBAL, output_sha256: str = "a" * 64) -> ChunkRenderRecord:
    request = _request(0)
    return ChunkRenderRecord(
        protocol_id=protocol_id,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path="chunks/chunk-000.wav",
        output_sha256=output_sha256,
        source_chunk_sha256=request.sha256,
        sample_rate_hz=48_000,
        attempts=1,
    )


def test_protocol_manifests_share_the_same_common_drum_hash(tmp_path: Path) -> None:
    request = _request(0)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    record = _record()

    write_listening_wav(drums_path, render_common_drums((request,), song_index=0))
    write_listening_wav(vocal_path, render_common_drums((request,), song_index=1))
    write_listening_wav(mix_path, render_common_drums((request,), song_index=2))

    first = build_protocol_artifact_manifest(
        ProtocolId.MOSS_GLOBAL,
        requests=(request,),
        chunk_records=(record,),
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
    )
    second = build_protocol_artifact_manifest(
        ProtocolId.TED_LOCAL,
        requests=(request,),
        chunk_records=(record,),
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
    )

    assert first["request_sha256"] == [request.sha256]
    assert first["drums"]["sha256"] == second["drums"]["sha256"]


def test_chunk_record_is_complete_only_for_matching_successful_record_and_wav(tmp_path: Path) -> None:
    ledger_path = tmp_path / "records.jsonl"
    wav_path = tmp_path / "chunk.wav"
    request = _request(0)
    audio = render_common_drums((request,), song_index=0)
    write_listening_wav(wav_path, audio)

    record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path=str(wav_path),
        output_sha256="",
        source_chunk_sha256=request.sha256,
        sample_rate_hz=48_000,
        attempts=1,
    )
    stored = append_chunk_record(ledger_path, record)
    stored_payload = json.loads(stored)

    assert chunk_record_is_complete(ledger_path, wav_path, request=request, protocol_id=ProtocolId.MOSS_GLOBAL)
    assert stored_payload["output_sha256"] == read_chunk_record_index(ledger_path)[(ProtocolId.MOSS_GLOBAL, request.song_id, 0)].output_sha256


def test_append_chunk_record_rejects_duplicate_and_conflicting_rows(tmp_path: Path) -> None:
    duplicate_path = tmp_path / "duplicate.jsonl"
    conflicting_path = tmp_path / "conflicting.jsonl"
    record = _record()

    append_chunk_record(duplicate_path, record)
    with pytest.raises(ValueError, match="duplicate"):
        append_chunk_record(duplicate_path, record)

    append_chunk_record(conflicting_path, record)
    with pytest.raises(ValueError, match="conflict"):
        append_chunk_record(conflicting_path, _record(output_sha256="b" * 64))

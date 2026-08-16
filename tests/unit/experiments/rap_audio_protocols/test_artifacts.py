from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    build_protocol_artifact_manifest,
    chunk_record_is_complete,
    file_sha256,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.audio import CHUNK_FRAME_COUNT, render_common_drums, write_listening_wav
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, SyllableTarget, TwoBarRenderRequest


def _requests(total_chunks: int = 25) -> tuple[TwoBarRenderRequest, ...]:
    return tuple(_request(index) for index in range(total_chunks))


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


def _record(
    *,
    request: TwoBarRenderRequest | None = None,
    protocol_id: ProtocolId = ProtocolId.MOSS_GLOBAL,
    output_sha256: str = "a" * 64,
    source_chunk_sha256: str | None = None,
) -> ChunkRenderRecord:
    request = request or _request(0)
    return ChunkRenderRecord(
        protocol_id=protocol_id,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path="chunks/chunk-000.wav",
        output_sha256=output_sha256,
        source_chunk_sha256=source_chunk_sha256,
        sample_rate_hz=48_000,
        attempts=1,
    )


def _write_listening_wav(path: Path, *, frames: int, channels: int) -> None:
    shape = (frames,) if channels == 1 else (frames, channels)
    wavfile.write(path, 48_000, np.zeros(shape, dtype=np.int16))


def test_protocol_manifests_share_the_same_common_drum_hash(tmp_path: Path) -> None:
    request = _request(0)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    first_record = _record(request=request, protocol_id=ProtocolId.MOSS_GLOBAL)
    second_record = _record(request=request, protocol_id=ProtocolId.TED_LOCAL)

    write_listening_wav(drums_path, render_common_drums((request,), song_index=0, allow_smoke_test=True))
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    first = build_protocol_artifact_manifest(
        ProtocolId.MOSS_GLOBAL,
        requests=(request,),
        chunk_records=(first_record,),
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
        allow_smoke_test=True,
    )
    second = build_protocol_artifact_manifest(
        ProtocolId.TED_LOCAL,
        requests=(request,),
        chunk_records=(second_record,),
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
        allow_smoke_test=True,
    )

    assert first["request_sha256"] == [request.sha256]
    assert first["drums"]["sha256"] == second["drums"]["sha256"]


def test_chunk_record_is_complete_accepts_actual_source_chunk_hash(tmp_path: Path) -> None:
    ledger_path = tmp_path / "records.jsonl"
    wav_path = tmp_path / "chunk.wav"
    request = _request(0)
    audio = render_common_drums((request,), song_index=0, allow_smoke_test=True)
    write_listening_wav(wav_path, audio)
    source_hash = file_sha256(wav_path)

    record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path=str(wav_path),
        output_sha256="",
        source_chunk_sha256=source_hash,
        sample_rate_hz=48_000,
        attempts=1,
    )
    stored = append_chunk_record(ledger_path, record)
    stored_payload = json.loads(stored)

    assert chunk_record_is_complete(ledger_path, wav_path, request=request, protocol_id=ProtocolId.MOSS_GLOBAL)
    assert stored_payload["output_sha256"] == read_chunk_record_index(ledger_path)[(ProtocolId.MOSS_GLOBAL, request.song_id, 0)].output_sha256
    assert stored_payload["source_chunk_sha256"] == source_hash


def test_chunk_record_is_complete_allows_missing_source_hash_for_native_protocols(tmp_path: Path) -> None:
    ledger_path = tmp_path / "records.jsonl"
    wav_path = tmp_path / "chunk.wav"
    request = _request(0)
    audio = render_common_drums((request,), song_index=1, allow_smoke_test=True)
    write_listening_wav(wav_path, audio)

    record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path=str(wav_path),
        output_sha256="",
        source_chunk_sha256=None,
        sample_rate_hz=48_000,
        attempts=1,
    )
    append_chunk_record(ledger_path, record)

    assert chunk_record_is_complete(ledger_path, wav_path, request=request, protocol_id=ProtocolId.MOSS_GLOBAL)


def test_chunk_record_is_complete_rejects_missing_source_hash_for_moss_aligned(tmp_path: Path) -> None:
    ledger_path = tmp_path / "records.jsonl"
    wav_path = tmp_path / "chunk.wav"
    request = _request(0)
    audio = render_common_drums((request,), song_index=2, allow_smoke_test=True)
    write_listening_wav(wav_path, audio)
    append_chunk_record(
        ledger_path,
        ChunkRenderRecord(
            protocol_id=ProtocolId.MOSS_ALIGNED,
            song_id=request.song_id,
            chunk_index=request.chunk_index,
            request_sha256=request.sha256,
            success=True,
            output_path=str(wav_path),
            output_sha256="",
            source_chunk_sha256=None,
            sample_rate_hz=48_000,
            attempts=1,
        ),
    )

    assert not chunk_record_is_complete(
        ledger_path,
        wav_path,
        request=request,
        protocol_id=ProtocolId.MOSS_ALIGNED,
    )


def test_chunk_record_is_complete_accepts_valid_source_hash_for_moss_aligned(tmp_path: Path) -> None:
    ledger_path = tmp_path / "records.jsonl"
    wav_path = tmp_path / "chunk.wav"
    request = _request(0)
    audio = render_common_drums((request,), song_index=3, allow_smoke_test=True)
    write_listening_wav(wav_path, audio)
    append_chunk_record(
        ledger_path,
        ChunkRenderRecord(
            protocol_id=ProtocolId.MOSS_ALIGNED,
            song_id=request.song_id,
            chunk_index=request.chunk_index,
            request_sha256=request.sha256,
            success=True,
            output_path=str(wav_path),
            output_sha256="",
            source_chunk_sha256="b" * 64,
            sample_rate_hz=48_000,
            attempts=1,
        ),
    )

    assert chunk_record_is_complete(
        ledger_path,
        wav_path,
        request=request,
        protocol_id=ProtocolId.MOSS_ALIGNED,
    )


@pytest.mark.parametrize("source_chunk_sha256", [None, "", "not-a-sha256"])
def test_manifest_rejects_missing_or_invalid_source_hash_for_moss_aligned(
    tmp_path: Path,
    source_chunk_sha256: str | None,
) -> None:
    request = _request(0)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    _write_listening_wav(drums_path, frames=CHUNK_FRAME_COUNT, channels=2)
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    with pytest.raises(ValueError, match="source_chunk_sha256"):
        build_protocol_artifact_manifest(
            ProtocolId.MOSS_ALIGNED,
            requests=(request,),
            chunk_records=(
                _record(
                    request=request,
                    protocol_id=ProtocolId.MOSS_ALIGNED,
                    source_chunk_sha256=source_chunk_sha256,
                ),
            ),
            vocal_stem_path=vocal_path,
            drums_path=drums_path,
            mix_path=mix_path,
            allow_smoke_test=True,
        )


def test_manifest_accepts_valid_source_hash_for_moss_aligned(tmp_path: Path) -> None:
    request = _request(0)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    source_hash = "c" * 64
    _write_listening_wav(drums_path, frames=CHUNK_FRAME_COUNT, channels=2)
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    manifest = build_protocol_artifact_manifest(
        ProtocolId.MOSS_ALIGNED,
        requests=(request,),
        chunk_records=(
            _record(
                request=request,
                protocol_id=ProtocolId.MOSS_ALIGNED,
                source_chunk_sha256=source_hash,
            ),
        ),
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
        allow_smoke_test=True,
    )

    assert manifest["source_chunks"][0]["source_chunk_sha256"] == source_hash


def test_build_protocol_artifact_manifest_rejects_chunk_records_from_other_protocols(tmp_path: Path) -> None:
    request = _request(0)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    write_listening_wav(drums_path, render_common_drums((request,), song_index=0, allow_smoke_test=True))
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    with pytest.raises(ValueError, match="protocol"):
        build_protocol_artifact_manifest(
            ProtocolId.TED_LOCAL,
            requests=(request,),
            chunk_records=(_record(request=request, protocol_id=ProtocolId.MOSS_GLOBAL),),
            vocal_stem_path=vocal_path,
            drums_path=drums_path,
            mix_path=mix_path,
            allow_smoke_test=True,
        )


def test_build_protocol_artifact_manifest_rejects_chunk_records_for_other_requests(tmp_path: Path) -> None:
    request = _request(0)
    other_request = _request(1)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    write_listening_wav(drums_path, render_common_drums((request,), song_index=0, allow_smoke_test=True))
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    with pytest.raises(ValueError, match="request set"):
        build_protocol_artifact_manifest(
            ProtocolId.MOSS_GLOBAL,
            requests=(request,),
            chunk_records=(_record(request=other_request, protocol_id=ProtocolId.MOSS_GLOBAL),),
            vocal_stem_path=vocal_path,
            drums_path=drums_path,
            mix_path=mix_path,
            allow_smoke_test=True,
        )


def test_build_protocol_artifact_manifest_rejects_wrong_wav_shape(tmp_path: Path) -> None:
    request = _request(0)
    record = _record(request=request, protocol_id=ProtocolId.MOSS_GLOBAL)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    _write_listening_wav(drums_path, frames=CHUNK_FRAME_COUNT - 1, channels=2)
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    with pytest.raises(ValueError, match="frame count"):
        build_protocol_artifact_manifest(
            ProtocolId.MOSS_GLOBAL,
            requests=(request,),
            chunk_records=(record,),
            vocal_stem_path=vocal_path,
            drums_path=drums_path,
            mix_path=mix_path,
            allow_smoke_test=True,
        )


def test_build_protocol_artifact_manifest_rejects_non_campaign_request_sets_without_override(tmp_path: Path) -> None:
    requests = _requests(1)
    request = requests[0]
    record = _record(request=request, protocol_id=ProtocolId.MOSS_GLOBAL)
    drums_path = tmp_path / "drums.wav"
    vocal_path = tmp_path / "vocals.wav"
    mix_path = tmp_path / "mix.wav"
    _write_listening_wav(drums_path, frames=CHUNK_FRAME_COUNT, channels=2)
    _write_listening_wav(vocal_path, frames=CHUNK_FRAME_COUNT, channels=1)
    _write_listening_wav(mix_path, frames=CHUNK_FRAME_COUNT, channels=2)

    with pytest.raises(ValueError, match="25 requests"):
        build_protocol_artifact_manifest(
            ProtocolId.MOSS_GLOBAL,
            requests=requests,
            chunk_records=(record,),
            vocal_stem_path=vocal_path,
            drums_path=drums_path,
            mix_path=mix_path,
        )


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

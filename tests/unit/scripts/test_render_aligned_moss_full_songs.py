from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)


MODE = "continuous_onset_gentle_sparse_r3"
SONG_ID = "01_space_exploration"


def _request(chunk_index: int, *, tempo_bpm: float = 90.0) -> TwoBarRenderRequest:
    syllables = tuple(
        SyllableTarget(
            word=f"word{index}",
            index_in_word=0,
            phonemes=("AA1",),
            lexical_stress=1,
            target_stress=float(index % 3) / 2.0,
            boundary_strength=0,
            absolute_tick=(chunk_index * 32) + index,
            tick_in_chunk=index,
            target_seconds=0.1 + (index * 0.25),
        )
        for index in range(18)
    )
    return TwoBarRenderRequest(
        song_id=SONG_ID,
        chunk_index=chunk_index,
        start_bar=chunk_index * 2,
        end_bar=(chunk_index * 2) + 2,
        text=" ".join(item.word for item in syllables),
        syllables=syllables,
        tempo_bpm=tempo_bpm,
    )


def _write_campaign(
    root: Path,
    *,
    chunk_count: int = 2,
    tempo_bpm: float = 90.0,
) -> tuple[TwoBarRenderRequest, ...]:
    requests = tuple(_request(index, tempo_bpm=tempo_bpm) for index in range(chunk_count))
    common = root / "common" / SONG_ID
    common.mkdir(parents=True)
    (common / "requests.jsonl").write_text(
        "\n".join(
            canonical_json_dumps({**request.to_payload(), "request_sha256": request.sha256})
            for request in requests
        )
        + "\n",
        encoding="utf-8",
    )
    drum_frames = round(chunk_count * 2 * 4 * 60 / tempo_bpm * 48_000)
    wavfile.write(
        common / "drums.wav",
        48_000,
        np.full((drum_frames, 2), round(0.20 * 32767), dtype=np.int16),
    )

    moss_root = root / "moss_global" / SONG_ID
    moss_root.mkdir(parents=True)
    records = []
    for request in requests:
        source_path = moss_root / f"chunk-{request.chunk_index:03d}.wav"
        wavfile.write(
            source_path,
            24_000,
            np.full(round(request.duration_seconds * 24_000), 0.10, dtype=np.float32),
        )
        records.append(
            ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_GLOBAL,
                song_id=SONG_ID,
                chunk_index=request.chunk_index,
                request_sha256=request.sha256,
                success=True,
                output_path=str(source_path.resolve()),
                output_sha256=file_sha256(source_path),
                sample_rate_hz=24_000,
                attempts=1,
            )
        )
        grid = (
            root
            / "moss_aligned"
            / SONG_ID
            / "mfa-output"
            / f"{SONG_ID}__chunk_{request.chunk_index:02d}.TextGrid"
        )
        grid.parent.mkdir(parents=True, exist_ok=True)
        grid.write_text("fixture", encoding="utf-8")
    (moss_root / "render_chunks.jsonl").write_text(
        "\n".join(canonical_json_dumps(record.to_payload()) for record in records) + "\n",
        encoding="utf-8",
    )
    return requests


@pytest.mark.parametrize("tempo_bpm", (90.0, 133.0))
def test_full_song_runner_renders_resumably_and_assembles_without_separator_gaps(
    load_script,
    tmp_path: Path,
    tempo_bpm: float,
) -> None:
    module = load_script("render_aligned_moss_full_songs")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    requests = _write_campaign(input_dir, tempo_bpm=tempo_bpm)
    rendered_chunks: list[int] = []

    def render_aligned_chunk(**kwargs):
        request = kwargs["request"]
        output_path = Path(kwargs["output_wav_path"])
        assert kwargs["mode"] == MODE
        rendered_chunks.append(request.chunk_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            output_path,
            24_000,
            np.full(
                round(request.duration_seconds * 24_000),
                0.25 + (request.chunk_index * 0.05),
                dtype=np.float32,
            ),
        )
        diagnostics_path = output_path.with_suffix(".wav.alignment.json")
        diagnostics = {
            "schema_version": "streammuse.rap_audio_protocols.alignment_diagnostics.v2",
            "success": True,
            "mode": MODE,
            "request_sha256": request.sha256,
            "source_sha256": kwargs["expected_source_sha256"],
            "output_sha256": file_sha256(output_path),
            "stretch_ratios": [0.75, 1.25],
            "fallback_count": 0,
            "boundary_adjustment_count": 0,
            "source_boundary_adjustment_count": 0,
            "timing_regularization": {
                "applied": True,
                "target_drift_seconds": [0.01, -0.02],
            },
            "stress": {"applied": False, "peak_limited": False},
            "error": None,
        }
        diagnostics_path.write_text(canonical_json_dumps(diagnostics) + "\n", encoding="utf-8")
        return SimpleNamespace(
            record=ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_ALIGNED,
                song_id=request.song_id,
                chunk_index=request.chunk_index,
                request_sha256=request.sha256,
                success=True,
                output_path=str(output_path.resolve()),
                output_sha256=file_sha256(output_path),
                source_chunk_sha256=kwargs["expected_source_sha256"],
                sample_rate_hz=24_000,
                attempts=1,
            ),
            diagnostics_path=diagnostics_path,
        )

    backend = SimpleNamespace(render_aligned_chunk=render_aligned_chunk)
    assert module.run_campaign(
        input_dir=input_dir,
        output_dir=output_dir,
        song_ids=(SONG_ID,),
        backend=backend,
        expected_chunk_count=2,
    ) == 0
    assert rendered_chunks == [0, 1]

    song_root = output_dir / SONG_ID
    vocal_rate, vocals = wavfile.read(song_root / "vocals.wav")
    mix_rate, mix = wavfile.read(song_root / "mix.wav")
    assert vocal_rate == mix_rate == 48_000
    expected_frames = round(4 * 4 * 60 / tempo_bpm * 48_000)
    first_boundary = round(2 * 4 * 60 / tempo_bpm * 48_000)
    assert vocals.shape == (expected_frames,)
    assert mix.shape == (expected_frames, 2)
    assert np.all(np.abs(mix[first_boundary - 5 : first_boundary + 5]) > 0)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == MODE
    assert manifest["tempo_bpm"] == tempo_bpm
    assert manifest["bars_per_song"] == 4
    assert manifest["total_chunk_count"] == len(requests)
    assert manifest["songs"][0]["frame_count"] == expected_frames
    assert manifest["songs"][0]["separator_silence_seconds"] == 0.0

    assert module.run_campaign(
        input_dir=input_dir,
        output_dir=output_dir,
        song_ids=(SONG_ID,),
        backend=backend,
        expected_chunk_count=2,
    ) == 0
    assert rendered_chunks == [0, 1]


def test_full_song_runner_refuses_a_missing_textgrid(load_script, tmp_path: Path) -> None:
    module = load_script("render_aligned_moss_full_songs")
    input_dir = tmp_path / "input"
    requests = _write_campaign(input_dir, chunk_count=1)
    grid = (
        input_dir
        / "moss_aligned"
        / SONG_ID
        / "mfa-output"
        / f"{SONG_ID}__chunk_{requests[0].chunk_index:02d}.TextGrid"
    )
    grid.unlink()

    try:
        module.run_campaign(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            song_ids=(SONG_ID,),
            backend=SimpleNamespace(render_aligned_chunk=lambda **_: None),
            allow_smoke_test=True,
        )
    except ValueError as error:
        assert "missing MFA TextGrid" in str(error)
    else:
        raise AssertionError("missing TextGrid must fail before rendering")


def test_full_song_runner_supports_the_varied_flow_song_set(load_script) -> None:
    module = load_script("render_aligned_moss_full_songs")

    assert {
        "04_city_nights",
        "08_street_basketball",
        "10_future_music",
    }.issubset(module.SONGS)
    assert module.SONG_TITLES["04_city_nights"] == "Lights After Midnight"
    assert module.SONG_TITLES["08_street_basketball"] == "Concrete Court"
    assert module.SONG_TITLES["10_future_music"] == "The Next Sound"


def test_plain_r3_diagnostics_require_stress_and_regularization_to_be_disabled(
    load_script,
    tmp_path: Path,
) -> None:
    module = load_script("render_aligned_moss_full_songs")
    module.MODE = "continuous_onset_r3"
    request = _request(0)
    path = tmp_path / "chunk.wav.alignment.json"
    payload = {
        "success": True,
        "mode": module.MODE,
        "request_sha256": request.sha256,
        "source_sha256": "source-sha",
        "output_sha256": "output-sha",
        "stress": {"applied": False},
        "timing_regularization": {"applied": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert module._load_verified_diagnostics(
        path,
        request=request,
        expected_source_sha256="source-sha",
        expected_output_sha256="output-sha",
    ) == payload

    payload["stress"]["applied"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        module._load_verified_diagnostics(
            path,
            request=request,
            expected_source_sha256="source-sha",
            expected_output_sha256="output-sha",
        )
    except ValueError as error:
        assert "stress application mismatch" in str(error)
    else:
        raise AssertionError("plain R3 diagnostics must reject stress augmentation")


def test_full_song_request_loader_preserves_tempo(load_script, tmp_path: Path) -> None:
    module = load_script("render_aligned_moss_full_songs")
    payload = _request(0).to_payload()
    payload["tempo_bpm"] = 133.0
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    request = module._load_requests(path)[0]

    assert request.tempo_bpm == 133.0
    assert request.duration_seconds == 480 / 133


def test_full_song_runner_accepts_legacy_ninety_bpm_request_hashes(
    load_script,
    tmp_path: Path,
) -> None:
    module = load_script("render_aligned_moss_full_songs")
    payload = _request(0).to_payload()
    payload.pop("tempo_bpm")
    legacy_sha256 = hashlib.sha256(
        canonical_json_dumps(payload).encode("utf-8")
    ).hexdigest()
    payload["request_sha256"] = legacy_sha256
    path = tmp_path / "requests.jsonl"
    path.write_text(canonical_json_dumps(payload) + "\n", encoding="utf-8")

    request = module._load_requests(path)[0]

    assert request.tempo_bpm == 90.0
    assert request.sha256 != legacy_sha256
    assert module._request_sha256_matches(request, legacy_sha256)

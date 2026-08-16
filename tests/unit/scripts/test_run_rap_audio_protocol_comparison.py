from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.audio import CHUNK_FRAME_COUNT, SONG_FRAME_COUNT
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, canonical_json_dumps
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "rap_audio_protocols" / "two_bar_records.jsonl"


def _fixture_records() -> list[dict]:
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]


def _song_records(*, song_index: int, topic: str, title: str) -> list[dict]:
    base_records = _fixture_records()
    records = []
    for bar in range(50):
        base = copy.deepcopy(base_records[bar % 2])
        base_bar = int(base["bar"])
        bar_delta = bar - base_bar
        base["bar"] = bar
        base["song_index"] = song_index
        base["topic"] = topic
        base["title"] = title
        for item in base["schedule"]:
            item["absolute_tick"] = int(item["absolute_tick"]) + (bar_delta * 16)
            item["seconds_from_song_start"] = float(item["seconds_from_song_start"]) + (bar_delta * (8 / 3))
        records.append(base)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")


def _build_source_album(root: Path) -> Path:
    album = root / "source_album"
    songs = (
        ("01_space_exploration", 1, "space exploration", "Signals Beyond Earth"),
        ("02_deep_ocean", 2, "deep ocean", "Pressure Below"),
        ("03_artificial_intelligence", 3, "artificial intelligence", "Learning Machines"),
        ("04_city_nights", 4, "city nights", "Lights After Midnight"),
    )
    for song_id, song_index, topic, title in songs:
        _write_jsonl(album / song_id / "chosen_lyrics.jsonl", _song_records(song_index=song_index, topic=topic, title=title))
    return album


def _prepare_campaign(module, source_album: Path, output_dir: Path) -> int:
    return module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(output_dir),
            "--stage",
            "prepare",
        ]
    )


def _write_chunk_wav(path: Path, *, frame_count: int = CHUNK_FRAME_COUNT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 48_000, np.full(frame_count, 0.25, dtype=np.float32))


def _protocol_records(song_id: str, request_path: Path, output_dir: Path) -> list[ChunkRenderRecord]:
    requests = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    del requests  # silence lint in the test file
    payloads = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines()]
    records = []
    for payload in payloads:
        chunk_path = output_dir / song_id / f"chunk-{int(payload['chunk_index']):03d}.wav"
        _write_chunk_wav(chunk_path)
        records.append(
            ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_GLOBAL,
                song_id=song_id,
                chunk_index=int(payload["chunk_index"]),
                request_sha256=payload["request_sha256"],
                success=True,
                output_path=str(chunk_path),
                output_sha256="a" * 64,
                sample_rate_hz=48_000,
                attempts=1,
            )
        )
    return records


def test_prepare_selects_only_songs_01_to_03_and_writes_exactly_75_canonical_requests(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    output_dir = tmp_path / "campaign"

    exit_code = _prepare_campaign(module, source_album, output_dir)

    assert exit_code == 0
    common_root = output_dir / "common"
    requests_paths = sorted(common_root.glob("*/requests.jsonl"))
    assert [path.parent.name for path in requests_paths] == [
        "01_space_exploration",
        "02_deep_ocean",
        "03_artificial_intelligence",
    ]
    assert not (common_root / "04_city_nights").exists()
    assert sum(len(path.read_text(encoding="utf-8").splitlines()) for path in requests_paths) == 75
    for path in requests_paths:
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == canonical_json_dumps(payload)
        assert (path.parent / "drums.wav").is_file()


def test_prepare_refuses_mismatched_existing_manifest(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    output_dir = tmp_path / "campaign"
    manifest_path = output_dir / "common" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"schema_version":"wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched existing manifest"):
        _prepare_campaign(module, source_album, output_dir)


def test_assemble_requires_25_records_and_writes_exact_length_artifacts(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    output_dir = tmp_path / "campaign"
    assert _prepare_campaign(module, source_album, output_dir) == 0

    song_id = "01_space_exploration"
    request_path = output_dir / "common" / song_id / "requests.jsonl"
    record_dir = output_dir / ProtocolId.MOSS_GLOBAL.value / song_id
    records = []
    for payload in [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines()]:
        chunk_path = record_dir / f"chunk-{int(payload['chunk_index']):03d}.wav"
        _write_chunk_wav(chunk_path)
        records.append(
            ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_GLOBAL,
                song_id=song_id,
                chunk_index=int(payload["chunk_index"]),
                request_sha256=payload["request_sha256"],
                success=True,
                output_path=str(chunk_path),
                output_sha256="a" * 64,
                sample_rate_hz=48_000,
                attempts=1,
            )
        )
    (record_dir / "render_chunks.jsonl").write_text(
        "\n".join(canonical_json_dumps(record.to_payload()) for record in records) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(output_dir),
            "--stage",
            "assemble",
            "--song",
            song_id,
            "--protocol",
            ProtocolId.MOSS_GLOBAL.value,
        ]
    )

    assert exit_code == 0
    vocals_path = record_dir / "vocals.wav"
    mix_path = record_dir / "mix.wav"
    assert wavfile.read(vocals_path)[1].shape[0] == SONG_FRAME_COUNT
    assert wavfile.read(mix_path)[1].shape[0] == SONG_FRAME_COUNT


def test_evaluate_is_lazy_and_package_writes_blinded_outputs(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    output_dir = tmp_path / "campaign"
    assert _prepare_campaign(module, source_album, output_dir) == 0

    song_id = "01_space_exploration"
    protocol_id = ProtocolId.MOSS_GLOBAL.value
    protocol_dir = output_dir / protocol_id / song_id
    protocol_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "common" / song_id / "requests.jsonl"
    records = []
    for payload in [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines()]:
        chunk_path = protocol_dir / f"chunk-{int(payload['chunk_index']):03d}.wav"
        _write_chunk_wav(chunk_path)
        records.append(
            ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_GLOBAL,
                song_id=song_id,
                chunk_index=int(payload["chunk_index"]),
                request_sha256=payload["request_sha256"],
                success=True,
                output_path=str(chunk_path),
                output_sha256="a" * 64,
                sample_rate_hz=48_000,
                attempts=1,
            )
        )
    (protocol_dir / "render_chunks.jsonl").write_text(
        "\n".join(canonical_json_dumps(record.to_payload()) for record in records) + "\n",
        encoding="utf-8",
    )
    wavfile.write(protocol_dir / "vocals.wav", 48_000, np.zeros(SONG_FRAME_COUNT, dtype=np.int16))
    wavfile.write(protocol_dir / "mix.wav", 48_000, np.zeros((SONG_FRAME_COUNT, 2), dtype=np.int16))

    created = []

    def fake_transcriber_factory(**kwargs):
        created.append(kwargs)

        def _transcribe(_path: Path):
            return ()

        return _transcribe

    assert module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(output_dir),
            "--stage",
            "prepare",
        ],
        transcriber_factory=fake_transcriber_factory,
    ) == 0
    assert created == []

    metrics_path = protocol_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"protocol_id": protocol_id, "song_id": song_id, "failed_chunk_count": 0}),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(output_dir),
            "--stage",
            "package",
            "--song",
            song_id,
            "--protocol",
            protocol_id,
        ],
        transcriber_factory=fake_transcriber_factory,
    )

    assert exit_code == 0
    assert (output_dir / "blind_map.json").is_file()
    assert (output_dir / "listening.html").is_file()
    assert (output_dir / "COMPARISON.md").is_file()
    assert (output_dir / "package_audit.json").is_file()

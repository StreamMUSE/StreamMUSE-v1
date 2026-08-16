from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.audio import CHUNK_FRAME_COUNT, SONG_FRAME_COUNT
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, canonical_json_dumps, sha256_hex
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


def _write_package_mix(output_dir: Path, *, song_id: str, protocol_id: str) -> None:
    mix_path = output_dir / protocol_id / song_id / "mix.wav"
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    mix_path.write_bytes(b"RIFF-test-package-wave")
    manifest = {
        "protocol_id": protocol_id,
        "mix": {
            "path": str(mix_path),
            "sha256": hashlib.sha256(mix_path.read_bytes()).hexdigest(),
        },
    }
    (mix_path.parent / "artifact_manifest.json").write_text(
        json.dumps(
            {**manifest, "artifact_manifest_sha256": sha256_hex(manifest)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


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
                output_sha256=hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
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
    lyrics = (common_root / "lyrics.md").read_text(encoding="utf-8")
    assert lyrics.startswith("# Rap Audio Protocol Comparison Lyrics\n")
    assert "## 01 space exploration" in lyrics
    assert "## 02 deep ocean" in lyrics
    assert "## 03 artificial intelligence" in lyrics
    assert "04 city nights" not in lyrics
    assert lyrics.count("### Bars ") == 75
    assert "Rocket blasts liftoff, silence breaks free," in lyrics
    for path in requests_paths:
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == canonical_json_dumps(payload)
        assert (path.parent / "drums.wav").is_file()


def test_prepare_prints_one_dense_progress_line_per_chunk(load_script, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    song_id = "01_space_exploration"

    assert module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(tmp_path / "campaign"),
            "--stage",
            "prepare",
            "--song",
            song_id,
        ]
    ) == 0

    progress_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(f"stage=prepare song={song_id} ")
    ]
    assert len(progress_lines) == 25
    assert "protocol=common chunk=000 count=1/25 status=ready request_sha256=" in progress_lines[0]
    assert "protocol=common chunk=024 count=25/25 status=ready request_sha256=" in progress_lines[-1]


def test_prepare_refuses_mismatched_existing_manifest(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    source_album = _build_source_album(tmp_path)
    output_dir = tmp_path / "campaign"
    manifest_path = output_dir / "common" / "corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"schema_version":"wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched existing manifest"):
        _prepare_campaign(module, source_album, output_dir)

    error_records = [
        json.loads(line)
        for line in (output_dir / "campaign_errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert error_records == [
        {
            "chunk_index": None,
            "error_type": "ValueError",
            "message": "mismatched existing manifest",
            "protocol_id": None,
            "schema_version": "streammuse.rap_audio_protocols.campaign_error.v1",
            "song_id": None,
            "stage": "prepare",
        }
    ]


def test_load_chunk_records_rejects_a_modified_rendered_wav(load_script, tmp_path: Path) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    chunk_path = tmp_path / "chunk-000.wav"
    chunk_path.write_bytes(b"modified-rendered-audio")
    ledger_path = tmp_path / "render_chunks.jsonl"
    ledger_path.write_text(
        canonical_json_dumps(
            {
                "attempts": 1,
                "chunk_index": 0,
                "error": None,
                "output_path": str(chunk_path),
                "output_sha256": "a" * 64,
                "protocol_id": ProtocolId.MOSS_GLOBAL.value,
                "request_sha256": "b" * 64,
                "sample_rate_hz": 48_000,
                "song_id": "01_space_exploration",
                "source_chunk_sha256": None,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_sha256 mismatch"):
        module._load_chunk_records(ledger_path)


def test_validate_aligned_sources_rejects_a_stale_moss_hash(load_script) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    moss_record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id="01_space_exploration",
        chunk_index=0,
        request_sha256="a" * 64,
        success=True,
        output_path="moss.wav",
        output_sha256="b" * 64,
        sample_rate_hz=24_000,
        attempts=1,
    )
    aligned_record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_ALIGNED,
        song_id="01_space_exploration",
        chunk_index=0,
        request_sha256="a" * 64,
        success=True,
        output_path="aligned.wav",
        output_sha256="c" * 64,
        source_chunk_sha256="d" * 64,
        sample_rate_hz=24_000,
        attempts=1,
    )

    with pytest.raises(ValueError, match="source_chunk_sha256 mismatch"):
        module._validate_aligned_source_records((aligned_record,), (moss_record,))


def test_assemble_requires_25_records_and_writes_exact_length_artifacts(
    load_script,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
                output_sha256=hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
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
    progress_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(f"stage=assemble song={song_id} protocol={ProtocolId.MOSS_GLOBAL.value} ")
    ]
    assert len(progress_lines) == 25
    assert "chunk=000 count=1/25 status=verified" in progress_lines[0]
    assert "chunk=024 count=25/25 status=verified" in progress_lines[-1]


def test_evaluate_is_lazy_and_package_writes_blinded_outputs(
    load_script,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
                output_sha256=hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
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

    assert module.main(
        [
            "--source-album",
            str(source_album),
            "--output-dir",
            str(output_dir),
            "--stage",
            "evaluate",
            "--song",
            song_id,
            "--protocol",
            protocol_id,
        ],
        transcriber_factory=fake_transcriber_factory,
    ) == 0
    assert created == [{"model_size": "small", "device": "cpu", "compute_type": "int8"}]
    evaluate_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(f"stage=evaluate song={song_id} protocol={protocol_id} ")
    ]
    assert len(evaluate_lines) == 25
    assert "chunk=000 count=1/25 status=success" in evaluate_lines[0]
    assert "chunk=024 count=25/25 status=success" in evaluate_lines[-1]

    metrics_path = protocol_dir / "metrics.json"
    assert metrics_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["failed_chunk_count"] == 0

    requests = module._load_requests(request_path)
    artifact_manifest = module.build_protocol_artifact_manifest(
        ProtocolId.MOSS_GLOBAL,
        requests=requests,
        chunk_records=records,
        vocal_stem_path=protocol_dir / "vocals.wav",
        drums_path=output_dir / "common" / song_id / "drums.wav",
        mix_path=protocol_dir / "mix.wav",
    )
    (protocol_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True),
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

    (protocol_dir / "mix.wav").write_bytes(b"tampered-after-assembly")
    with pytest.raises(ValueError, match="artifact manifest mix SHA-256 mismatch"):
        module.main(
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
            ]
        )


@pytest.mark.parametrize(
    ("filters", "missing_count", "selected_song", "selected_protocol"),
    (
        ((), 11, None, None),
        (("--song", "01_space_exploration"), 3, "01_space_exploration", None),
        (("--protocol", ProtocolId.MOSS_GLOBAL.value), 2, None, ProtocolId.MOSS_GLOBAL.value),
    ),
)
def test_package_rejects_any_partial_selected_matrix(
    load_script,
    tmp_path: Path,
    filters: tuple[str, ...],
    missing_count: int,
    selected_song: str | None,
    selected_protocol: str | None,
) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    output_dir = tmp_path / "campaign"
    _write_package_mix(
        output_dir,
        song_id="01_space_exploration",
        protocol_id=ProtocolId.MOSS_GLOBAL.value,
    )

    with pytest.raises(
        ValueError,
        match=rf"package stage requires complete selected matrix: missing {missing_count} mix\.wav files",
    ):
        module.main(
            [
                "--source-album",
                str(tmp_path / "source-album"),
                "--output-dir",
                str(output_dir),
                "--stage",
                "package",
                *filters,
            ]
        )

    assert not (output_dir / "blind_map.json").exists()
    assert not (output_dir / "listening.html").exists()
    error = json.loads((output_dir / "campaign_errors.jsonl").read_text(encoding="utf-8"))
    assert error["stage"] == "package"
    assert error["song_id"] == selected_song
    assert error["protocol_id"] == selected_protocol


@pytest.mark.parametrize(
    ("filters", "song_ids", "protocol_ids", "expected_audio_count"),
    (
        (
            ("--song", "01_space_exploration"),
            ("01_space_exploration",),
            tuple(protocol.value for protocol in ProtocolId),
            4,
        ),
        (
            ("--protocol", ProtocolId.MOSS_GLOBAL.value),
            (
                "01_space_exploration",
                "02_deep_ocean",
                "03_artificial_intelligence",
            ),
            (ProtocolId.MOSS_GLOBAL.value,),
            3,
        ),
        (
            (
                "--song",
                "01_space_exploration",
                "--protocol",
                ProtocolId.MOSS_GLOBAL.value,
            ),
            ("01_space_exploration",),
            (ProtocolId.MOSS_GLOBAL.value,),
            1,
        ),
    ),
)
def test_package_preserves_complete_explicit_selection_matrix(
    load_script,
    tmp_path: Path,
    filters: tuple[str, ...],
    song_ids: tuple[str, ...],
    protocol_ids: tuple[str, ...],
    expected_audio_count: int,
) -> None:
    module = load_script("run_rap_audio_protocol_comparison")
    output_dir = tmp_path / "campaign"
    for song_id in song_ids:
        for protocol_id in protocol_ids:
            _write_package_mix(output_dir, song_id=song_id, protocol_id=protocol_id)

    assert module.main(
        [
            "--source-album",
            str(tmp_path / "source-album"),
            "--output-dir",
            str(output_dir),
            "--stage",
            "package",
            *filters,
        ]
    ) == 0

    audit = json.loads((output_dir / "package_audit.json").read_text(encoding="utf-8"))
    blind_map = json.loads((output_dir / "blind_map.json").read_text(encoding="utf-8"))
    assert audit["audio_file_count"] == expected_audio_count
    assert set(blind_map) == set(song_ids)
    assert all(len(mapping) == len(protocol_ids) for mapping in blind_map.values())


def test_h200_setup_is_idempotent_and_records_pinned_environments_models_and_reference(tmp_path: Path) -> None:
    setup_script = ROOT / "scripts" / "setup_rap_audio_protocols_h200.sh"
    root_prefix = tmp_path / "root"
    env_root = root_prefix / "envs" / "rap-audio-protocols"
    checkout_root = root_prefix / "checkouts" / "rap-audio-protocols"
    asset_root = root_prefix / "assets" / "rap-audio-protocols"
    hf_home = tmp_path / "shared-hf"
    command_log = tmp_path / "commands.log"
    fake_bin = tmp_path / "fake-bin"

    moss_commit = "58b20a0d5fcc6766658d50967a90a9d890009a46"
    ted_commit = "36ffc3e2de346156baa7d60a1749ca4a9365625b"
    moss_snapshot = "cdd3b911b1585e3f2dbc7775ef10f9926f58850a"
    indextts_snapshot = "740dcaff396282ffb241903d150ac011cd4b1ede"

    _write_executable(
        fake_bin / "git",
        r'''
printf 'git' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
if [ "${1:-}" = "clone" ]; then
  target="${@: -1}"
  mkdir -p "${target}/.git" "${target}/datasets/Ref"
  printf 'ted-reference-audio\n' > "${target}/datasets/Ref/0011_000001.wav"
  exit 0
fi
if [ "${1:-}" = "-C" ]; then
  target="$2"
  shift 2
  case "${1:-}" in
    checkout)
      printf '%s\n' "${@: -1}" > "${target}/.fake-head"
      ;;
    rev-parse)
      cat "${target}/.fake-head"
      ;;
  esac
fi
''',
    )
    _write_executable(
        fake_bin / "uv",
        r'''
printf 'cwd=%s UV_PROJECT_ENVIRONMENT=%s uv' "$PWD" "${UV_PROJECT_ENVIRONMENT:-}" >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
if [ "${1:-}" = "--version" ]; then
  printf 'uv 0.test\n'
fi
''',
    )
    _write_executable(
        fake_bin / "conda",
        r'''
printf 'conda' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
if [ "${1:-}" = "--version" ]; then
  printf 'conda 25.test\n'
  exit 0
fi
prefix=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--prefix" ] || [ "$1" = "-p" ]; then
    prefix="$2"
    break
  fi
  shift
done
mkdir -p "${prefix}/conda-meta"
''',
    )
    _write_executable(fake_bin / "sudo", "exit 99\n")
    _write_executable(
        fake_bin / "apt",
        r'''
printf 'apt' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
if [ "${1:-}" = "download" ]; then
  shift
  for spec in "$@"; do
    package="${spec%%=*}"
    printf 'fake deb\n' > "${package}_fake_amd64.deb"
  done
fi
''',
    )
    _write_executable(
        fake_bin / "dpkg-deb",
        r'''
printf 'dpkg-deb' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
deb="$2"
target="$3"
mkdir -p "${target}/usr/bin" "${target}/usr/lib/x86_64-linux-gnu"
if [[ "${deb}" == *rubberband-cli* ]]; then
  cat > "${target}/usr/bin/rubberband" <<'EOF'
#!/usr/bin/env bash
if [[ "${LD_LIBRARY_PATH}" != "${EXPECTED_ALIGN_LIB}:"* ]]; then
  printf 'expected ALIGN_ENV/lib first, got %s\\n' "${LD_LIBRARY_PATH}" >&2
  exit 9
fi
printf 'rubberband 3.3.0\n'
EOF
  chmod +x "${target}/usr/bin/rubberband"
fi
''',
    )

    for env_name in ("moss", "ted"):
        env_dir = env_root / env_name
        _write_executable(env_dir / "bin" / "python", "exit 0\n")
        _write_executable(
            env_dir / "bin" / "hf",
            r'''
printf 'hf' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
repo="$2"
shift 2
revision=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--revision" ]; then
    revision="$2"
    break
  fi
  shift
done
repo_dir="${repo//\//--}"
snapshot_path="${HF_HOME}/hub/models--${repo_dir}/snapshots/${revision}"
mkdir -p "${snapshot_path}"
printf 'model-config\n' > "${snapshot_path}/config.yaml"
printf '%s\n' "${snapshot_path}"
''',
        )

    _write_executable(
        env_root / "nemo" / "bin" / "python",
        r'''
printf 'nemo-python' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
cat >/dev/null
printf '%s\n' '{"cuda_version":"12.8","models":["tts_en_fastpitch","tts_en_hifigan"],"nemo_toolkit_version":"2.7.3","torch_version":"2.9.1+cu128","torchaudio_version":"2.9.1+cu128"}'
''',
    )
    _write_executable(env_root / "align" / "bin" / "python", "exit 0\n")
    _write_executable(
        env_root / "align" / "bin" / "mfa",
        r'''
printf 'mfa' >> "${COMMAND_LOG}"
printf ' %s' "$@" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
if [ "${1:-}" = "version" ]; then
  printf '3.4.1\n'
fi
''',
    )
    _write_executable(
        env_root / "ffmpeg7" / "bin" / "ffmpeg",
        "printf 'ffmpeg version 7.1.1 test-build\\n'\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "COMMAND_LOG": str(command_log),
            "ROOT_PREFIX": str(root_prefix),
            "ENV_ROOT": str(env_root),
            "CHECKOUT_ROOT": str(checkout_root),
            "ASSET_ROOT": str(asset_root),
            "HF_HOME": str(hf_home),
            "MANIFEST_PATH": str(env_root / "environment_manifest.json"),
            "UV_BIN": str(fake_bin / "uv"),
            "PYTHON_BIN": sys.executable,
            "MOSS_PYTHON_BIN": "/usr/bin/python3.12",
            "PYTHON310_BIN": "/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/python",
            "CONDA_BIN": str(fake_bin / "conda"),
            "EXPECTED_ALIGN_LIB": str(env_root / "align" / "lib"),
        }
    )

    first = subprocess.run(["bash", str(setup_script)], cwd=ROOT, env=environment, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    manifest_path = env_root / "environment_manifest.json"
    first_manifest = manifest_path.read_bytes()
    second = subprocess.run(["bash", str(setup_script)], cwd=ROOT, env=environment, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    assert manifest_path.read_bytes() == first_manifest

    manifest = json.loads(first_manifest)
    assert manifest["repositories"]["moss"]["requested_ref"] == moss_commit
    assert manifest["repositories"]["moss"]["resolved_commit"] == moss_commit
    assert manifest["repositories"]["ted"]["requested_ref"] == ted_commit
    assert manifest["repositories"]["ted"]["resolved_commit"] == ted_commit
    assert manifest["packages"]["nemo_toolkit"] == {"package_version": "2.7.3", "source_tag": "v2.7.3"}
    assert manifest["packages"]["montreal_forced_aligner"] == {"package_version": "3.4.1", "source_tag": "v3.4.1"}
    assert manifest["packages"]["faster_whisper"] == {"package_version": "1.2.1"}
    assert manifest["packages"]["pronouncing"] == {"package_version": "0.3.0"}
    assert manifest["packages"]["ffmpeg"] == {
        "package_version": "7.1.1",
        "resolved_version": "ffmpeg version 7.1.1 test-build",
    }
    assert manifest["packages"]["rubberband"] == {
        "apt_package_version": "3.3.0+dfsg-2build1",
        "binary_path": str(asset_root / "rubberband" / "rootfs" / "usr" / "bin" / "rubberband"),
        "resolved_version": "rubberband 3.3.0",
    }
    assert manifest["models"]["moss_tts"]["revision"] == moss_snapshot
    assert manifest["models"]["index_tts_2"]["revision"] == indextts_snapshot
    assert manifest["models"]["nemo"] == ["tts_en_fastpitch", "tts_en_hifigan"]
    assert manifest["hf_home"] == str(hf_home)
    assert set(manifest["environments"]) == {"align", "ffmpeg7", "moss", "nemo", "ted"}

    expected_reference_hash = hashlib.sha256(b"ted-reference-audio\n").hexdigest()
    assert manifest["ted_reference"]["relative_source_path"] == "datasets/Ref/0011_000001.wav"
    assert manifest["ted_reference"]["source_sha256"] == expected_reference_hash
    assert manifest["ted_reference"]["copied_sha256"] == expected_reference_hash
    assert Path(manifest["ted_reference"]["copied_path"]).read_bytes() == b"ted-reference-audio\n"

    commands = command_log.read_text(encoding="utf-8")
    assert f"git -C {checkout_root / 'moss'} checkout --detach {moss_commit}" in commands
    assert f"git -C {checkout_root / 'ted'} checkout --detach {ted_commit}" in commands
    assert (
        f"uv venv --python /usr/bin/python3.12 {env_root / 'moss'}" in commands
    )
    assert (
        f"uv pip install --python {env_root / 'moss' / 'bin' / 'python'} "
        "--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 "
        f"--index-strategy unsafe-best-match -e {checkout_root / 'moss'}[torch-runtime]"
    ) in commands
    assert f"uv pip install --python {env_root / 'moss' / 'bin' / 'python'} faster-whisper==1.2.1" in commands
    assert f"UV_PROJECT_ENVIRONMENT={env_root / 'ted'} uv sync --all-extras" in commands
    assert "torch==2.9.1+cu128 torchaudio==2.9.1+cu128 nemo_toolkit[tts]==2.7.3" in commands
    for env_name in ("moss", "ted", "nemo", "align"):
        assert f"uv pip install --python {env_root / env_name / 'bin' / 'python'} pronouncing==0.3.0" in commands
    assert "conda create --yes --prefix" in commands
    assert "python=3.10 montreal-forced-aligner=3.4.1" in commands
    assert "conda create --yes --prefix" in commands
    assert (
        f"conda create --yes --prefix {env_root / 'ffmpeg7'} --channel conda-forge ffmpeg=7.1.1"
        in commands
    )
    assert "fftw" not in commands
    assert "libsamplerate" not in commands
    assert "libsndfile" not in commands
    assert "apt download rubberband-cli=3.3.0+dfsg-2build1 librubberband2=3.3.0+dfsg-2build1" in commands
    assert "dpkg-deb -x" in commands
    assert f"hf download OpenMOSS-Team/MOSS-TTS-v1.5 --revision {moss_snapshot}" in commands
    assert f"hf download IndexTeam/IndexTTS-2 --revision {indextts_snapshot}" in commands
    assert "mfa model download acoustic english_us_arpa" in commands
    assert "mfa model download dictionary english_us_arpa" in commands

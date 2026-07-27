from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from streammuse.infrastructure.voice import SpeechRecognitionError, TranscriptionResult


_FAKE_COMMIT = "a" * 40


def _write_wav(path: Path, *, sample_rate: int = 16_000, value: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.full(sample_rate // 10, value, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames.tobytes())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeRecognizer:
    def __init__(
        self,
        config,
        outputs: list[tuple[str, float]],
        *,
        provenance_overrides: dict | None = None,
    ) -> None:
        self.config = config
        self.outputs = iter(outputs)
        self.provenance_overrides = provenance_overrides or {}
        self.start_count = 0
        self.close_count = 0
        self.contexts = []
        self.audio_shapes: list[tuple[int, ...]] = []

    @property
    def provenance(self):
        configured_path = Path(self.config.model).expanduser()
        if configured_path.is_dir():
            resolved_path = configured_path
        else:
            resolved_path = (
                Path(self.config.model_cache)
                / "models--Systran--faster-whisper-tiny.en"
                / "snapshots"
                / _FAKE_COMMIT
            )
        return {
            "model": self.config.model,
            "local_files_only": self.config.local_files_only,
            "model_revision_requested": self.config.model_revision,
            "model_revision_resolved": _FAKE_COMMIT,
            "model_path_resolved": str(resolved_path),
            **self.provenance_overrides,
        }

    def start(self) -> None:
        self.start_count += 1

    def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
        self.audio_shapes.append(audio.shape)
        self.contexts.append(speech_context)
        text, latency_ms = next(self.outputs)
        return TranscriptionResult(
            text=text,
            latency_ms=latency_ms,
            diagnostics={"fake": True},
        )

    def close(self) -> None:
        self.close_count += 1


def _acceptance_rows(root: Path, *, sample_rate: int = 16_000) -> list[dict]:
    rows: list[dict] = []
    definitions = [
        ("Zip", "zip"),
        ("Zap", "zap"),
        ("Zop", "zop"),
        ("ZipZap", "combination"),
        ("20", "number"),
        (None, "silence"),
        (None, "noise"),
        (None, "playback"),
        (None, "non_command"),
    ]
    for index, (expected, category) in enumerate(definitions):
        audio = root / f"acceptance-{index + 1:02d}.wav"
        _write_wav(audio, sample_rate=sample_rate, value=1000 + index)
        rows.append(
            {
                "audio_path": audio.name,
                "expected": expected,
                "split": "acceptance",
                "category": category,
                "speaker": f"speaker-{(index % 5) + 1}",
                "session": f"session-{index + 1}",
                "distance": ("near", "mid", "far")[index % 3],
                "environment": ("quiet", "office")[index % 2],
            }
        )
    return rows


def _development_row(root: Path, *, sample_rate: int = 16_000) -> dict:
    audio = root / "dev-01.wav"
    _write_wav(audio, sample_rate=sample_rate, value=3000)
    return {
        "audio_path": audio.name,
        "expected": "Zip",
        "split": "dev",
        "speaker": "dev-speaker-1",
        "session": "dev-session-1",
        "distance": "near",
        "environment": "quiet",
    }


def _args(qualify, manifest: Path, output_dir: Path, *extra: str):
    model_cache = manifest.parent / "model-cache"
    model_snapshot = (
        model_cache
        / "models--Systran--faster-whisper-tiny.en"
        / "snapshots"
        / _FAKE_COMMIT
    )
    model_snapshot.mkdir(parents=True, exist_ok=True)
    (model_snapshot / "model.bin").write_bytes(b"fake-model")
    (model_snapshot / "config.json").write_text("{}", encoding="utf-8")
    return qualify.build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--model-cache",
            str(model_cache),
            "--model-revision",
            _FAKE_COMMIT,
            "--min-speakers",
            "5",
            "--min-base-word-samples",
            "1",
            "--min-combination-samples",
            "1",
            "--min-number-samples",
            "1",
            "--min-negative-samples",
            "4",
            "--min-negative-category-samples",
            "1",
            *extra,
        ]
    )


def test_replay_uses_production_contract_and_writes_private_aggregate_evidence(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    rows = _acceptance_rows(tmp_path, sample_rate=8_000)
    rows.append(_development_row(tmp_path, sample_rate=8_000))
    _write_json(manifest, {"entries": rows})
    output_dir = tmp_path / "output"
    fake = FakeRecognizer(
        None,
        [
            ("Zip.", 10.0),
            ("Zap", 20.0),
            ("Zop", 30.0),
            ("zip zap", 40.0),
            ("twenty", 50.0),
            ("", 60.0),
            ("", 70.0),
            ("", 80.0),
            ("", 90.0),
            ("Zip", 100.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(qualify, manifest, output_dir), recognizer_factory=factory
    )

    assert summary["passed"] is False
    assert summary["qualification_profile"] == "exploratory"
    assert summary["qualification_scope"] == "offline_asr_corpus"
    assert summary["full_feature_qualification"] is False
    assert summary["configuration"] == {
        "context_profile": "baseline",
        "initial_prompt": None,
        "hotwords": [],
        "model": "tiny.en",
        "device": "cpu",
        "compute_type": "int8",
        "model_cache": str(tmp_path / "model-cache"),
        "model_revision": _FAKE_COMMIT,
        "local_files_only": True,
    }
    assert fake.start_count == 1
    assert fake.close_count == 1
    assert fake.contexts == [None] * 10
    assert fake.audio_shapes == [(1600,)] * 10

    metrics = summary["metrics"]["overall"]
    assert metrics["raw_exact"]["accuracy"] == pytest.approx(7 / 10)
    assert metrics["canonical_exact"]["accuracy"] == 1.0
    assert metrics["canonical_exact"]["wilson_95"]["lower"] < 1.0
    assert metrics["asr_latency_ms"]["p50"] == 55.0
    assert metrics["asr_latency_ms"]["p95"] == pytest.approx(95.5)
    assert metrics["positive_empty_transcript"]["rate"] == 0.0
    assert "positive_false_no_speech" not in metrics
    assert summary["metrics"]["by_split"]["acceptance"]["count"] == 9
    assert summary["metrics"]["by_category"]["combination"]["count"] == 1
    assert summary["metrics"]["by_split_category"]["acceptance"]["number"]["count"] == 1
    assert summary["acceptance"]["protocol"]["negative_category_counts"] == {
        "silence": 1,
        "noise": 1,
        "playback": 1,
        "non_command": 1,
    }
    assert summary["acceptance"]["protocol"]["development_sample_count"] == 1
    assert summary["acceptance"]["protocol"]["development_speaker_count"] == 1
    assert summary["acceptance"]["protocol"]["samples_missing_session"] == 0
    assert summary["acceptance"]["protocol"]["samples_missing_distance"] == 0
    assert summary["acceptance"]["protocol"]["samples_missing_environment"] == 0
    assert summary["acceptance"]["protocol"]["session_split_overlap_count"] == 0
    assert summary["acceptance"]["protocol"]["distance_category_counts"] == {
        "far": 3,
        "mid": 3,
        "near": 3,
    }
    gate_names = {check["name"] for check in summary["acceptance"]["checks"]}
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}
    assert checks["formal_threshold_profile"]["passed"] is False
    assert checks["production_decode_context_profile"]["passed"] is False
    assert checks["configured_model_is_tiny_en"]["passed"] is True
    assert checks["model_huggingface_repository"]["passed"] is True
    assert "positive_empty_transcript_rate" in gate_names
    assert "positive_false_no_speech_rate" not in gate_names
    assert (
        "positive false-no-speech rate from microphone/VAD endpointing"
        in summary["acceptance"]["not_measured"]
    )
    assert len(summary["reproducibility"]["dependency_lock"]["sha256"]) == 64
    dependency_evidence = summary["reproducibility"]["dependency_lock"]["packages"]
    assert set(dependency_evidence) == {
        "faster-whisper",
        "ctranslate2",
        "av",
        "onnxruntime",
        "tokenizers",
        "huggingface-hub",
        "sounddevice",
        "webrtcvad-wheels",
        "numpy",
        "scipy",
    }
    assert all(value["matches_lock"] for value in dependency_evidence.values())
    assert summary["environment"]["voice_dependencies"] == {
        name: value["installed"] for name, value in dependency_evidence.items()
    }
    assert summary["reproducibility"]["model"]["evidence_mode"] == "huggingface_snapshot"
    assert summary["reproducibility"]["model"]["snapshot_commit"] == _FAKE_COMMIT
    assert summary["reproducibility"]["model"]["requested_revision_pinned"] is True
    assert summary["reproducibility"]["model"]["snapshot_repository"] == (
        "Systran/faster-whisper-tiny.en"
    )
    assert len(summary["reproducibility"]["model"]["critical_file_tree"]["sha256"]) == 64
    assert set(summary["source_code"]) == {
        "qualification_script",
        "zip_zap_zop_parser",
        "recognizer",
    }
    assert all(len(value["sha256"]) == 64 for value in summary["source_code"].values())

    result_lines = (output_dir / "samples.jsonl").read_text(encoding="utf-8")
    assert "speaker-" not in result_lines
    assert "session-" not in result_lines
    assert '"distance"' not in result_lines
    assert '"environment"' not in result_lines
    results = [json.loads(line) for line in result_lines.splitlines()]
    assert len({result["audio_sha256"] for result in results}) == 10
    assert results[0]["raw_exact"] is False
    assert results[0]["predicted_canonical"] == "Zip"
    negative_results = [result for result in results if result["expected_canonical"] is None]
    assert len(negative_results) == 4
    assert all(result["canonical_exact"] for result in negative_results)
    assert results[0]["positive_empty_transcript"] is False
    assert "false_no_speech" not in results[0]
    assert "audio_path" not in results[0]
    assert len(results[0]["audio_sha256"]) == 64
    assert summary["artifact_set"]["id"] == summary["artifact_set_id"]
    assert summary["artifact_set"]["samples"]["sha256"] == qualify._sha256(
        output_dir / "samples.jsonl"
    )
    assert json.loads((output_dir / "summary.json").read_text()) == summary


def test_quality_rejected_transcript_cannot_become_a_canonical_answer(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"entries": _acceptance_rows(tmp_path)})
    output_dir = tmp_path / "output"

    class RejectingRecognizer(FakeRecognizer):
        def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
            result = super().transcribe(audio, speech_context=speech_context)
            if len(self.contexts) == 1:
                raw_text = ", ".join(["Zap"] * 112)
                return TranscriptionResult(
                    text=raw_text,
                    latency_ms=result.latency_ms,
                    diagnostics={
                        "quality_gate": {
                            "accepted": False,
                            "reasons": [
                                "excessive_token_repetition",
                                "excessive_compression_ratio",
                            ],
                        }
                    },
                    rejection_reasons=(
                        "excessive_token_repetition",
                        "excessive_compression_ratio",
                    ),
                )
            return result

    fake = RejectingRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(qualify, manifest, output_dir),
        recognizer_factory=factory,
    )

    first = json.loads(
        (output_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first["raw_transcript"].startswith("Zap, Zap")
    assert first["predicted_canonical"] is None
    assert first["parse_status"] == "rejected_transcript"
    assert first["transcript_quality_rejected"] is True
    assert first["transcript_rejection_reasons"] == [
        "excessive_token_repetition",
        "excessive_compression_ratio",
    ]
    rejection_metrics = summary["metrics"]["overall"][
        "transcript_quality_rejected"
    ]
    assert rejection_metrics == {
        "count": 1,
        "rate": pytest.approx(1 / 9),
        "positive_count": 1,
        "positive_rate": pytest.approx(1 / 5),
        "negative_count": 0,
        "negative_rate": 0.0,
        "reason_counts": {
            "excessive_compression_ratio": 1,
            "excessive_token_repetition": 1,
        },
    }


def test_custom_prompt_hotwords_and_revision_are_passed_without_download(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": audio.name,
                "expected": "Zip",
                "split": "dev",
                "speaker": "private-speaker",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fake = FakeRecognizer(None, [("Zip", 5.0)])

    def factory(config):
        fake.config = config
        return fake

    args = _args(
        qualify,
        manifest,
        tmp_path / "out",
        "--acceptance-split",
        "dev",
        "--min-speakers",
        "1",
        "--min-combination-samples",
        "0",
        "--min-number-samples",
        "0",
        "--min-negative-samples",
        "0",
        "--context-profile",
        "custom",
        "--initial-prompt",
        "Valid words.",
        "--hotword",
        "Zip",
        "--hotword",
        "Zap",
        "--model-cache",
        str(tmp_path / "cache"),
        "--model-revision",
        "revision-123",
    )
    summary = qualify.run_qualification(args, recognizer_factory=factory)

    assert fake.config.local_files_only is True
    assert fake.config.model_cache == str(tmp_path / "cache")
    assert fake.config.model_revision == "revision-123"
    assert fake.contexts[0].initial_prompt == "Valid words."
    assert fake.contexts[0].hotwords == ("Zip", "Zap")
    assert summary["configuration"]["context_profile"] == "custom"
    assert summary["recognizer_provenance"]["model_revision_requested"] == "revision-123"


def test_per_category_false_command_gate_cannot_be_hidden_by_aggregate_rate(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    rows = _acceptance_rows(tmp_path)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, rows)
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 10.0),
            ("Zap", 10.0),
            ("Zop", 10.0),
            ("Zip Zap", 10.0),
            ("20", 10.0),
            ("Zip", 10.0),
            ("", 10.0),
            ("", 10.0),
            ("", 10.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(
            qualify,
            manifest,
            tmp_path / "out",
            "--max-negative-false-command-rate",
            "0.3",
        ),
        recognizer_factory=factory,
    )

    assert summary["passed"] is False
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}
    assert checks["negative_false_game_command_rate"] == {
        "name": "negative_false_game_command_rate",
        "passed": True,
        "actual": 0.25,
        "operator": "<=",
        "threshold": 0.3,
    }
    assert checks["silence_negative_false_game_command_rate"] == {
        "name": "silence_negative_false_game_command_rate",
        "passed": False,
        "actual": 1.0,
        "operator": "<=",
        "threshold": 0.3,
    }
    assert {
        "expected": "<negative>",
        "predicted": "Zip",
        "count": 1,
    } in summary["metrics"]["overall"]["confusion_counts"]


def test_manifest_rejects_remote_audio_and_cross_split_speaker_leakage(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    remote_manifest = tmp_path / "remote.json"
    _write_json(
        remote_manifest,
        [{"audio_path": "https://example.test/voice.wav", "expected": "Zip", "split": "dev"}],
    )
    with pytest.raises(qualify.QualificationError, match="local filesystem path"):
        qualify.load_manifest(remote_manifest)

    rows = _acceptance_rows(tmp_path)
    dev_audio = tmp_path / "dev.wav"
    _write_wav(dev_audio, value=2000)
    rows.append(
        {
            "audio_path": dev_audio.name,
            "expected": "Zip",
            "split": "dev",
            "speaker": rows[0]["speaker"],
            "session": rows[0]["session"],
            "distance": "near",
            "environment": "quiet",
        }
    )
    manifest = tmp_path / "overlap.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("Zip", 1.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
    )
    overlap = next(
        check
        for check in summary["acceptance"]["checks"]
        if check["name"] == "speaker_split_overlap_count"
    )
    assert overlap["passed"] is False
    assert overlap["actual"] == 1
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}
    assert checks["acceptance_development_session_overlap_count"]["actual"] == 1
    assert checks["acceptance_development_session_overlap_count"]["passed"] is False


def test_manifest_rejects_duplicate_paths_and_cross_split_audio_content(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_wav(first, value=1234)
    _write_wav(second, value=1234)
    manifest = tmp_path / "manifest.json"

    _write_json(
        manifest,
        [
            {"audio_path": first.name, "expected": "Zip", "split": "acceptance"},
            {"audio_path": first.name, "expected": "Zap", "split": "acceptance"},
        ],
    )
    with pytest.raises(qualify.QualificationError, match="audio_path duplicates sample 1"):
        qualify.load_manifest(manifest)

    _write_json(
        manifest,
        [
            {"audio_path": first.name, "expected": "Zip", "split": "acceptance"},
            {"audio_path": second.name, "expected": "Zap", "split": "dev"},
        ],
    )
    with pytest.raises(qualify.QualificationError, match="audio content duplicates sample 1"):
        qualify.load_manifest(manifest)


def test_output_directory_cannot_contain_manifest_but_unrelated_output_is_allowed(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    corpus_dir = tmp_path / "corpus"
    audio = corpus_dir / "clip.wav"
    _write_wav(audio)
    manifest = corpus_dir / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "acceptance"}],
    )

    with pytest.raises(qualify.QualificationError, match="output directory must be new"):
        qualify.run_qualification(
            _args(qualify, manifest, corpus_dir),
            recognizer_factory=lambda config: None,
        )

    fake = FakeRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    output_dir = tmp_path / "qualification-output"
    summary = qualify.run_qualification(
        _args(
            qualify,
            manifest,
            output_dir,
            "--min-speakers",
            "1",
            "--min-combination-samples",
            "0",
            "--min-number-samples",
            "0",
            "--min-negative-samples",
            "0",
            "--min-negative-category-samples",
            "0",
        ),
        recognizer_factory=factory,
    )

    assert summary["manifest"]["path"] == str(manifest.resolve())
    assert (output_dir / "summary.json").is_file()
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}
    assert checks["development_sample_count"]["passed"] is False


def test_existing_output_is_rejected_before_recognizer_construction(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    (output_dir / "samples.jsonl").write_text("old-samples\n", encoding="utf-8")
    (output_dir / "summary.json").write_text("old-summary\n", encoding="utf-8")
    factory_called = False

    def factory(config):
        nonlocal factory_called
        factory_called = True
        return FakeRecognizer(config, [("Zip", 1.0)])

    with pytest.raises(qualify.QualificationError, match="output directory must be new"):
        qualify.run_qualification(
            _args(qualify, manifest, output_dir), recognizer_factory=factory
        )

    assert factory_called is False
    assert (output_dir / "samples.jsonl").read_text() == "old-samples\n"
    assert (output_dir / "summary.json").read_text() == "old-summary\n"


def test_resolved_model_directory_cannot_overlap_output_after_start(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    output_dir = tmp_path / "out"
    fake = FakeRecognizer(
        None,
        [("Zip", 1.0)],
        provenance_overrides={
            "model_path_resolved": str(output_dir / "resolved-model"),
            "model_revision_resolved": None,
        },
    )

    def factory(config):
        fake.config = config
        return fake

    with pytest.raises(qualify.QualificationError, match="collides with or contains"):
        qualify.run_qualification(
            _args(qualify, manifest, output_dir), recognizer_factory=factory
        )

    assert fake.start_count == 1
    assert fake.close_count == 1
    assert not output_dir.exists()


def test_default_thresholds_and_task_context_are_the_only_formal_profile(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    acceptance = tmp_path / "acceptance.wav"
    development = tmp_path / "development.wav"
    _write_wav(acceptance, value=1100)
    _write_wav(development, value=1200)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [
            {
                "audio_path": acceptance.name,
                "expected": "Zip",
                "split": "acceptance",
                "speaker": "speaker-a",
                "session": "session-a",
                "distance": "near",
                "environment": "quiet",
            },
            {
                "audio_path": development.name,
                "expected": "Zip",
                "split": "dev",
                "speaker": "speaker-b",
                "session": "session-b",
                "distance": "far",
                "environment": "office",
            },
        ],
    )
    prepared = _args(qualify, manifest, tmp_path / "discarded")
    args = qualify.build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "formal-output"),
            "--model-cache",
            prepared.model_cache,
            "--model-revision",
            _FAKE_COMMIT,
            "--context-profile",
            "task",
        ]
    )
    fake = FakeRecognizer(None, [("Zip", 1.0), ("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(args, recognizer_factory=factory)
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}

    assert summary["qualification_profile"] == "formal"
    assert summary["passed"] is False
    assert checks["formal_threshold_profile"]["passed"] is True
    assert checks["production_decode_context_profile"]["passed"] is True
    assert checks["configured_model_is_tiny_en"]["passed"] is True
    assert fake.contexts[0].hotwords == ("Zip", "Zap", "Zop")


def test_parser_defaults_to_offline_and_validates_context_profile(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    parser = qualify.build_parser()
    args = parser.parse_args(["--manifest", "input.json", "--output-dir", "out"])
    assert args.model == "tiny.en"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.local_files_only is True
    assert args.min_negative_category_samples == 50

    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}])
    invalid = parser.parse_args(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--initial-prompt",
            "not baseline",
        ]
    )
    with pytest.raises(qualify.QualificationError, match="context-profile custom"):
        qualify.run_qualification(invalid, recognizer_factory=lambda config: None)


def test_recognizer_is_closed_when_startup_fails(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}])

    class BrokenRecognizer(FakeRecognizer):
        def start(self) -> None:
            super().start()
            raise RuntimeError("model cache miss")

    broken = BrokenRecognizer(None, [])

    def factory(config):
        broken.config = config
        return broken

    with pytest.raises(RuntimeError, match="model cache miss"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
        )
    assert broken.start_count == 1
    assert broken.close_count == 1
    assert not (tmp_path / "out" / "summary.json").exists()


def test_recognizer_close_error_does_not_replace_transcription_error(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )

    class BrokenRecognizer(FakeRecognizer):
        def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
            raise SpeechRecognitionError("decode failed")

        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("close failed")

    broken = BrokenRecognizer(None, [])

    def factory(config):
        broken.config = config
        return broken

    with pytest.raises(SpeechRecognitionError, match="decode failed"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
        )

    assert broken.close_count == 1


def test_manifest_categories_are_strict_and_cannot_persist_identity(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"

    _write_json(
        manifest,
        [
            {
                "audio_path": audio.name,
                "expected": None,
                "split": "acceptance",
                "category": "speaker-alice",
            }
        ],
    )
    with pytest.raises(qualify.QualificationError, match="category must be one of"):
        qualify.load_manifest(manifest)

    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": None, "split": "acceptance"}],
    )
    with pytest.raises(qualify.QualificationError, match="negative sample requires category"):
        qualify.load_manifest(manifest)

    _write_json(
        manifest,
        [
            {
                "audio_path": audio.name,
                "expected": "Zip",
                "split": "acceptance",
                "category": "noise",
            }
        ],
    )
    with pytest.raises(qualify.QualificationError, match="does not match the derived category"):
        qualify.load_manifest(manifest)

    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "acceptance"}],
    )
    assert qualify.load_manifest(manifest)[0].category == "zip"

    _write_json(
        manifest,
        [
            {
                "audio_path": audio.name,
                "expected": "Zip",
                "split": "acceptance",
                "distance": "near alice",
                "environment": "../../private",
            }
        ],
    )
    with pytest.raises(qualify.QualificationError, match="distance must match"):
        qualify.load_manifest(manifest)


def test_each_negative_semantic_category_has_an_independent_gate(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    rows = _acceptance_rows(tmp_path)
    for row in rows:
        if row["expected"] is None:
            row["category"] = "noise"
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, rows)
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
    )
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}

    assert checks["negative_sample_count"]["passed"] is True
    assert checks["noise_negative_sample_count"]["actual"] == 4
    assert checks["noise_negative_sample_count"]["passed"] is True
    for category in ("silence", "playback", "non_command"):
        assert checks[f"{category}_negative_sample_count"]["actual"] == 0
        assert checks[f"{category}_negative_sample_count"]["passed"] is False
    assert summary["passed"] is False


def test_missing_resolved_model_evidence_fails_reproducibility_gates(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _acceptance_rows(tmp_path))
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
        provenance_overrides={
            "model_path_resolved": None,
            "model_revision_resolved": None,
        },
    )

    def factory(config):
        fake.config = config
        return fake

    with pytest.raises(qualify.QualificationError, match="resolved model path is unavailable"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
        )
    assert fake.close_count == 1
    assert not (tmp_path / "out").exists()


def test_installed_voice_dependency_version_must_match_uv_lock(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _acceptance_rows(tmp_path))
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    def versions(name: str) -> str | None:
        if name == "sounddevice":
            return "0.0.0-drifted"
        return qualify._package_version(name)

    summary = qualify.run_qualification(
        _args(qualify, manifest, tmp_path / "out"),
        recognizer_factory=factory,
        package_version_getter=versions,
    )
    evidence = summary["reproducibility"]["dependency_lock"]["packages"][
        "sounddevice"
    ]
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}

    assert evidence["installed"] == "0.0.0-drifted"
    assert evidence["locked"] == ["0.5.5"]
    assert evidence["matches_lock"] is False
    assert checks["dependency_sounddevice_matches_lock"]["passed"] is False
    assert summary["passed"] is False


def test_hugging_face_snapshot_requires_hex_commit_and_matching_requested_pin(
    load_script, tmp_path
):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _acceptance_rows(tmp_path))
    invalid_snapshot = tmp_path / "models" / "snapshots" / "main"
    invalid_snapshot.mkdir(parents=True)
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
        provenance_overrides={
            "model_path_resolved": str(invalid_snapshot),
            "model_revision_resolved": "main",
        },
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
    )
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}

    assert summary["reproducibility"]["model"]["snapshot_commit"] == "main"
    assert checks["hf_snapshot_commit_valid"]["passed"] is False
    assert checks["hf_requested_revision_pinned"]["passed"] is False
    assert checks["model_reproducibility_evidence_valid"]["passed"] is False

    pinned_output = tmp_path / "unpinned-output"
    pinned_fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
    )

    def pinned_factory(config):
        pinned_fake.config = config
        return pinned_fake

    unpinned = qualify.run_qualification(
        _args(
            qualify,
            manifest,
            pinned_output,
            "--model-revision",
            "main",
        ),
        recognizer_factory=pinned_factory,
    )
    unpinned_checks = {
        check["name"]: check for check in unpinned["acceptance"]["checks"]
    }

    assert unpinned_checks["hf_snapshot_commit_valid"]["passed"] is True
    assert unpinned_checks["hf_requested_revision_pinned"]["passed"] is False


def test_local_model_directory_uses_critical_file_tree_hash(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, _acceptance_rows(tmp_path))
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    (local_model / "model.bin").write_bytes(b"model-weights")
    (local_model / "config.json").write_text('{"model": "tiny.en"}', encoding="utf-8")
    fake = FakeRecognizer(
        None,
        [
            ("Zip", 1.0),
            ("Zap", 1.0),
            ("Zop", 1.0),
            ("ZipZap", 1.0),
            ("20", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
            ("", 1.0),
        ],
    )

    def factory(config):
        fake.config = config
        return fake

    summary = qualify.run_qualification(
        _args(
            qualify,
            manifest,
            tmp_path / "out",
            "--model",
            str(local_model),
        ),
        recognizer_factory=factory,
    )
    model = summary["reproducibility"]["model"]
    checks = {check["name"]: check for check in summary["acceptance"]["checks"]}

    assert model["evidence_mode"] == "local_directory"
    assert len(model["critical_file_tree"]["sha256"]) == 64
    assert model["critical_file_tree"]["missing_required_files"] == []
    assert {item["path"] for item in model["critical_file_tree"]["files"]} == {
        "config.json",
        "model.bin",
    }
    assert checks["model_critical_file_tree_recorded"]["passed"] is True
    assert checks["model_reproducibility_evidence_valid"]["passed"] is True


def test_audio_hash_is_rechecked_around_transcription(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    fake = FakeRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    def mutating_loader(path: Path) -> np.ndarray:
        decoded = qualify.load_audio(path)
        _write_wav(path, value=9999)
        return decoded

    with pytest.raises(qualify.QualificationError, match="sample 1 audio changed"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"),
            recognizer_factory=factory,
            audio_loader=mutating_loader,
        )
    assert fake.close_count == 1


def test_lock_hash_is_rechecked_after_transcription(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    lock_copy = tmp_path / "uv.lock"
    lock_copy.write_bytes((Path(__file__).parents[3] / "uv.lock").read_bytes())

    class MutatingRecognizer(FakeRecognizer):
        def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
            result = super().transcribe(audio, speech_context=speech_context)
            lock_copy.write_text(lock_copy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return result

    fake = MutatingRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    with pytest.raises(qualify.QualificationError, match="dependency lock changed"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"),
            recognizer_factory=factory,
            dependency_lock_path=lock_copy,
        )


def test_resolved_model_tree_change_prevents_publication(load_script, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )

    class MutatingRecognizer(FakeRecognizer):
        def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
            result = super().transcribe(audio, speech_context=speech_context)
            model_path = Path(self.provenance["model_path_resolved"])
            (model_path / "model.bin").write_bytes(b"changed-model")
            return result

    fake = MutatingRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    output_dir = tmp_path / "out"
    with pytest.raises(qualify.QualificationError, match="model critical file tree changed"):
        qualify.run_qualification(
            _args(qualify, manifest, output_dir), recognizer_factory=factory
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_source_change_prevents_publication(load_script, monkeypatch, tmp_path):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    sources = {}
    for name in ("qualification_script", "zip_zap_zop_parser", "recognizer"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        sources[name] = path
    static_baseline = {
        name: {"path": str(sources[name]), "sha256": qualify._sha256(sources[name])}
        for name in ("qualification_script", "zip_zap_zop_parser")
    }
    monkeypatch.setattr(qualify, "_static_source_baseline", lambda: static_baseline)
    monkeypatch.setattr(
        qualify.inspect,
        "getsourcefile",
        lambda value: str(sources["recognizer"]),
    )

    class MutatingRecognizer(FakeRecognizer):
        def transcribe(self, audio, *, speech_context=None) -> TranscriptionResult:
            result = super().transcribe(audio, speech_context=speech_context)
            sources["recognizer"].write_text("# changed\n", encoding="utf-8")
            return result

    fake = MutatingRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    output_dir = tmp_path / "out"
    with pytest.raises(qualify.QualificationError, match="recognizer source changed"):
        qualify.run_qualification(
            _args(qualify, manifest, output_dir), recognizer_factory=factory
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_manifest_hash_is_rechecked_between_evidence_writes(
    load_script, monkeypatch, tmp_path
):
    qualify = load_script("qualify_voice_input")
    audio = tmp_path / "clip.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        [{"audio_path": audio.name, "expected": "Zip", "split": "dev"}],
    )
    fake = FakeRecognizer(None, [("Zip", 1.0)])

    def factory(config):
        fake.config = config
        return fake

    original_write_jsonl = qualify._write_jsonl

    def write_then_mutate(path, rows):
        original_write_jsonl(path, rows)
        manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    monkeypatch.setattr(qualify, "_write_jsonl", write_then_mutate)
    with pytest.raises(qualify.QualificationError, match="manifest changed"):
        qualify.run_qualification(
            _args(qualify, manifest, tmp_path / "out"), recognizer_factory=factory
        )
    assert not (tmp_path / "out" / "summary.json").exists()
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_main_reports_speech_infrastructure_errors_without_traceback(
    load_script, monkeypatch, capsys, tmp_path
):
    qualify = load_script("qualify_voice_input")

    def fail(_args):
        raise SpeechRecognitionError("offline model snapshot is unavailable")

    monkeypatch.setattr(qualify, "run_qualification", fail)
    exit_code = qualify.main(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == (
        "voice qualification failed: offline model snapshot is unavailable\n"
    )
    assert "Traceback" not in captured.err

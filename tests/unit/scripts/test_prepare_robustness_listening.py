from __future__ import annotations

import argparse
import json
import wave

import numpy as np
import pytest

from streammuse.experiments.melody_robustness import (
    SEEDS,
    build_run_schedule,
    file_sha256,
    write_canonical_json,
    write_jsonl,
)


def test_fluidsynth_root_is_forwarded_to_local_wrapper_environment(
    load_script, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    root = tmp_path / "rootfs"
    binary = tmp_path / "local_fluidsynth.sh"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    (root / "usr" / "lib" / "x86_64-linux-gnu" / "pulseaudio").mkdir(
        parents=True
    )

    environment, _provenance = listening._fluidsynth_environment(
        argparse.Namespace(
            fluidsynth=str(binary),
            fluidsynth_root=str(root),
            fluidsynth_lib_dir=None,
        )
    )

    assert environment["STREAMMUSE_FLUIDSYNTH_ROOT"] == str(root.resolve())


def _freeze(listening, fixture, output, **overrides):
    values = {
        "input_manifest": str(fixture.input_manifest),
        "output": str(output),
        "perturb_seed": SEEDS["perturb"][0],
        "sample_seed": SEEDS["sample"][0],
        "blind_order_seed": SEEDS["blind_order"],
        "excerpt_start_beat": 0,
        "excerpt_starts_json": None,
    }
    values.update(overrides)
    listening.freeze_selection(argparse.Namespace(**values))
    return json.loads(output.read_text(encoding="utf-8"))


def test_freeze_selection_is_exactly_24_with_two_literal_repeats_and_fixed_blind_order(
    load_script, robustness_fixture, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    first_path = tmp_path / "selection-a.json"
    second_path = tmp_path / "selection-b.json"

    first = _freeze(listening, robustness_fixture, first_path)
    second = _freeze(listening, robustness_fixture, second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["clip_count"] == 24
    assert first["analysis_horizons_ticks"] == {
        str(index): 240 for index in range(1, 6)
    }
    assert len(first["clips"]) == 24
    assert len({clip["sample_id"] for clip in first["clips"]}) == 24
    assert first["repeat_source_semantic_indices"] == [9, 5]
    assert [clip["semantic_index"] for clip in first["clips"]] == [
        1, 10, 21, 20, 11, 9, 16, 14, 22, 3, 5, 23,
        15, 4, 2, 13, 18, 6, 0, 7, 17, 12, 19, 8,
    ]
    assert sum(clip["block"] == "anchor" for clip in first["clips"]) == 2
    repeats = [clip for clip in first["clips"] if clip["block"] == "repeat"]
    assert len(repeats) == 2
    originals = {clip["semantic_index"]: clip for clip in first["clips"]}
    for repeat in repeats:
        source = originals[repeat["duplicate_semantic_index"]]
        assert {
            key: repeat[key]
            for key in ("song", "condition", "pipeline", "perturb_seed", "sample_seed")
        } == {
            key: source[key]
            for key in ("song", "condition", "pipeline", "perturb_seed", "sample_seed")
        }


@pytest.mark.parametrize(
    "override",
    [
        {"perturb_seed": 999},
        {"sample_seed": 999},
        {"blind_order_seed": 999},
    ],
)
def test_formal_selection_rejects_non_frozen_seeds(
    load_script, robustness_fixture, tmp_path, override
):
    listening = load_script("prepare_robustness_listening")

    with pytest.raises(ValueError, match="frozen"):
        _freeze(listening, robustness_fixture, tmp_path / "invalid.json", **override)


@pytest.mark.parametrize("start", [-1, 11])
def test_freeze_selection_rejects_negative_or_out_of_horizon_excerpt(
    load_script, robustness_fixture, tmp_path, start
):
    listening = load_script("prepare_robustness_listening")

    with pytest.raises(ValueError, match="non-negative|exceeds analysis_end_tick"):
        _freeze(
            listening,
            robustness_fixture,
            tmp_path / "invalid-excerpt.json",
            excerpt_start_beat=start,
        )


def test_freeze_selection_requires_reliable_manifest_horizon(
    load_script, robustness_fixture, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    payload = json.loads(robustness_fixture.input_manifest.read_text(encoding="utf-8"))
    payload["entries"][0].pop("analysis_end_tick")
    broken = tmp_path / "missing-horizon.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="horizons must be integer ticks|analysis_end_tick"):
        listening.freeze_selection(
            argparse.Namespace(
                input_manifest=str(broken),
                output=str(tmp_path / "selection.json"),
                perturb_seed=SEEDS["perturb"][0],
                sample_seed=SEEDS["sample"][0],
                blind_order_seed=SEEDS["blind_order"],
                excerpt_start_beat=0,
                excerpt_starts_json=None,
            )
        )


def test_build_rejects_selection_manifest_or_excerpt_drift_before_creating_package(
    load_script, robustness_fixture, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    selection_path = tmp_path / "selection.json"
    selection = _freeze(listening, robustness_fixture, selection_path)
    selection["clips"][0]["excerpt_end_model_beat"] -= 1
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    input_manifest = json.loads(
        robustness_fixture.input_manifest.read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="listening selection manifest mismatch"):
        listening.validate_listening_selection_manifest(
            selection,
            input_manifest,
            manifest_path=robustness_fixture.input_manifest,
            verify_files=True,
        )


def test_listening_source_verdict_and_artifact_index_fail_closed_after_tampering(
    load_script, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    row = {"run_id": "mr-source", "pipeline": "rt"}
    run_dir = tmp_path / "runs" / "mr-source"
    attempt = run_dir / "attempt-001"
    attempt.mkdir(parents=True)
    artifact = attempt / "theoretical_model.mid"
    artifact.write_bytes(b"immutable-midi")
    binding = {
        "campaign_config_sha256": "a" * 64,
        "run_schedule_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "code_identity": "e" * 40,
        "campaign_binding_sha256": "f" * 64,
    }
    verdict = {
        "run_id": "mr-source",
        "pipeline": "rt",
        "attempt_id": "attempt-001",
        "content_valid": True,
        "operational_valid": True,
        **binding,
        "artifact_index": [
            {
                "path": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": file_sha256(artifact),
            }
        ],
    }
    write_canonical_json(attempt / "verdict.json", verdict)
    write_canonical_json(run_dir / "latest_verdict.json", verdict)

    verified_attempt, _, indexed = listening._verified_attempt(tmp_path, row, binding)
    assert verified_attempt == attempt.resolve()
    assert indexed == {artifact.resolve()}

    with pytest.raises(RuntimeError, match="campaign binding mismatch"):
        listening._verified_attempt(
            tmp_path,
            row,
            {**binding, "campaign_config_sha256": "0" * 64},
        )

    artifact.write_bytes(b"tampered-midi")
    with pytest.raises(RuntimeError, match="missing/corrupt indexed artifact"):
        listening._verified_attempt(tmp_path, row, binding)


def test_listening_audit_rejects_silence_and_unverified_sample_peak_as_true_peak(
    load_script, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    package = tmp_path / "package"
    midi_dir = package / "blind" / "midi"
    wav_dir = package / "blind" / "wav"
    midi_dir.mkdir(parents=True)
    wav_dir.mkdir(parents=True)
    midi = midi_dir / "S001.mid"
    midi.write_bytes(b"test-midi")
    wav = wav_dir / "S001.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\0\0" * 16)
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    key = {
        "campaign_config_sha256": "a" * 64,
        "run_schedule_sha256": "b" * 64,
        "selection_sha256": file_sha256(selection),
        "clips": [
            {
                "sample_id": "S001",
                "semantic_index": 0,
                "duplicate_semantic_index": None,
            }
        ],
    }
    write_canonical_json(package / "private_key.json", key)
    render = {
        "campaign_config_sha256": "a" * 64,
        "run_schedule_sha256": "b" * 64,
        "selection_path": str(selection),
        "selection_sha256": file_sha256(selection),
        "private_key_sha256": file_sha256(package / "private_key.json"),
        "render_bpm": 120,
        "sample_rate": 44100,
        "bit_depth": 16,
        "gain": 0.5,
        "gain_policy": listening.GAIN_POLICY,
        "clips": [
            {
                "sample_id": "S001",
                "midi": "blind/midi/S001.mid",
                "midi_sha256": file_sha256(midi),
                "wav": "blind/wav/S001.wav",
                "wav_sha256": file_sha256(wav),
                "audio": {
                    "silent": True,
                    "true_peak_verified": False,
                    "peak_measurement": "integer_sample_peak_only_not_true_peak",
                },
            }
        ],
    }
    write_canonical_json(package / "render_manifest.json", render)

    result = listening.audit_package_dir(package, require_wav=True)

    assert not result["valid"]
    assert not result["accepted_final"]
    assert result["true_peak_limitation"]
    assert any("silent listening clip" in error for error in result["errors"])
    assert any("WAV contains no nonzero samples" in error for error in result["errors"])
    assert any("true peak is unverified" in error for error in result["errors"])


def test_fix_wav_measures_and_protects_post_quantization_true_peak(
    load_script, tmp_path
):
    listening = load_script("prepare_robustness_listening")
    sample_rate = 44100
    seconds = 1
    time = np.arange(sample_rate * seconds, dtype=np.float64) / sample_rate
    samples = np.rint(np.sin(2.0 * np.pi * 997.0 * time) * 32767.0).astype("<i2")
    wav = tmp_path / "near-full-scale.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())

    result = listening._fix_wav(wav, sample_rate=sample_rate, seconds=seconds)

    assert result["true_peak_verified"] is True
    assert result["peak_measurement"] == listening.TRUE_PEAK_IMPLEMENTATION
    assert result["true_peak_oversample_factor"] == 4
    assert result["true_peak_protection_applied"] is True
    assert result["true_peak_dbtp_after"] <= listening.TRUE_PEAK_LIMIT_DBTP
    assert result["silent"] is False
    with wave.open(str(wav), "rb") as handle:
        assert handle.getnframes() == sample_rate * seconds
        assert handle.getsampwidth() == 2


def test_listening_audit_fails_closed_for_incomplete_package(load_script, tmp_path):
    listening = load_script("prepare_robustness_listening")
    package = tmp_path / "package"
    package.mkdir()
    (package / "render_manifest.json").write_text(
        json.dumps({"clips": [], "sample_rate": 44100}), encoding="utf-8"
    )
    (package / "private_key.json").write_text(
        json.dumps({"clips": []}), encoding="utf-8"
    )

    result = listening.audit_package_dir(package, require_wav=False)

    assert not result["valid"]
    assert result["clip_count"] == 0
    assert result["repeat_trial_count"] == 0
    assert any("expected 24" in error for error in result["errors"])
    assert any("two repeated trials" in error for error in result["errors"])

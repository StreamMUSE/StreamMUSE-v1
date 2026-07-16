from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_lekai_offline.py"
_SPEC = importlib.util.spec_from_file_location("run_lekai_offline_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
offline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(offline)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_writes_seeded_auditable_run_artifacts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint-fixture")
    npz_dir = tmp_path / "staging" / "npz"
    mel_dir = tmp_path / "staging" / "mel"
    output_dir = tmp_path / "output"
    npz_dir.mkdir(parents=True)
    mel_dir.mkdir(parents=True)
    npz_path = npz_dir / "song_variant.npz"
    source_midi = mel_dir / "song_variant.mid"
    npz_path.write_bytes(b"npz-fixture")
    source_midi.write_bytes(b"midi-source-fixture")

    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        npz_dir=str(npz_dir),
        output_dir=str(output_dir),
        device="cpu",
        dtype="float32",
        condition_idx=None,
        condition_stem="song_variant",
        condition_path=None,
        gt_prefix_beats=0,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        delay_beats=-1,
        seed=20260712,
        run_id="offline/song-variant seed-1",
        bpm=120,
        cache_lengths=False,
        expected_dataset_size=1,
        source_midi=None,
        source_midi_dir=None,
        require_source_midi=True,
    )
    monkeypatch.setattr(offline, "parse_args", lambda: args)

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.received_generator = None

        def generate_accompaniment(self, dataset, condition_idx, **kwargs):
            self.received_generator = kwargs["generator"]
            assert dataset.data_files == ["song_variant.npz"]
            assert condition_idx == 0
            return {
                "generated_sequence": torch.tensor([[257, 263, 265, 169]]),
                "sampled_token_trace": [169],
                "part0_beats": [],
                "part1_beats": [[169]],
                "GT_path": str(npz_path),
                "metadata": {"delay_beats": -1, "bpm": 120},
            }

    model = _FakeModel()
    monkeypatch.setattr(offline, "load_model", lambda **kwargs: model)
    monkeypatch.setattr(
        offline,
        "verify_part0_roundtrip",
        lambda **kwargs: {
            "valid": True,
            "expected_shape": [2, 88, 4],
            "decoded_shape": [2, 88, 4],
            "differing_cells": 0,
            "expected_roll_sha256": "a" * 64,
            "decoded_roll_sha256": "a" * 64,
            "part0_beat_tokens_sha256": "b" * 64,
            "bar_token": 255,
            "pad_marker": 173,
            "part0_end_marker": 170,
        },
    )
    monkeypatch.setattr(
        offline,
        "tokens_to_midi",
        lambda result_dict, save_path, velocity: Path(save_path).write_bytes(b"generated-midi"),
    )
    monkeypatch.setattr(
        offline,
        "save_gt_midi",
        lambda save_path, gt_path, velocity: Path(save_path).write_bytes(b"ground-truth-midi"),
    )

    offline.main()

    artifact_id = "offline_song-variant_seed-1"
    config_path = output_dir / f"{artifact_id}_run_config.json"
    trace_path = output_dir / f"{artifact_id}_tokens.json"
    generated_path = output_dir / f"{artifact_id}_generated.mid"
    assert config_path.is_file()
    assert trace_path.is_file()
    assert generated_path.is_file()

    config = json.loads(config_path.read_text())
    trace = json.loads(trace_path.read_text())
    assert config["run_id"] == args.run_id
    assert config["input"]["npz_stem"] == "song_variant"
    assert config["input"]["npz_sha256"] == _sha256(npz_path)
    assert config["input"]["source_midi"]["sha256"] == _sha256(source_midi)
    assert config["input"]["part0_roundtrip"]["valid"] is True
    assert config["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert config["sampling"]["seed"] == args.seed
    assert config["outputs"]["generated_midi_sha256"] == _sha256(generated_path)
    assert config["outputs"]["token_trace_sha256"] == _sha256(trace_path)
    assert trace["sampled_tokens"] == [169]
    assert trace["full_interleaved_sequence"] == [257, 263, 265, 169]
    assert trace["part0_beat_tokens_sha256"] == "b" * 64
    assert model.received_generator.device.type == "cpu"
    assert model.received_generator.initial_seed() == args.seed

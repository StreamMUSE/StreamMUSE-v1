from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import mido
import pretty_midi
import pytest
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.config import (
    ModelConfig,
)
from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (
    PianoMusicTokenizer,
)


@pytest.fixture
def runner() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    module_name = "_test_run_matched_beat_gt_offline_eval"
    spec = importlib.util.spec_from_file_location(
        module_name, root / "scripts" / "run_matched_beat_gt_offline_eval.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_matched_beat_gt_offline_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


def _write_npz(
    path: Path,
    *,
    measures: int = 8,
    time_signature="4/4",
    time_signature_idx=0,
    bad_shape: bool = False,
) -> None:
    payload: dict[str, object] = {
        "metadata": {
            "num_channels": 4,
            "num_measures": measures,
            "time_signature": time_signature,
            "time_signature_idx": time_signature_idx,
            "bpm": 87,
        }
    }
    for measure_index in range(measures):
        shape = (4, 87, 16) if bad_shape and measure_index == 0 else (4, 88, 16)
        roll = np.zeros(shape, dtype=np.uint8)
        if shape[1] == 88:
            for beat in range(4):
                step = beat * 4
                mel_pitch = 39 + (measure_index + beat) % 5
                acc_pitch = 27 + (measure_index + beat) % 4
                roll[1, mel_pitch, step] = 1
                roll[0, mel_pitch, step : step + 2] = 1
                roll[3, acc_pitch, step] = 1
                roll[2, acc_pitch, step : step + 3] = 1
        payload[f"measure_{measure_index}"] = roll
    np.savez_compressed(path, **payload)


def _write_manifest(runner, root: Path, rows: list[dict]) -> Path:
    manifest = root / "cohort_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_contract": "checkpoint-aligned legacy BEAT 192788 NPZ",
                "samples": rows,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _sample_row(runner, npz: Path, piece_id: str, order: int = 1) -> dict:
    return {
        "order": order,
        "piece_id": piece_id,
        "source_npz": npz.name,
        "source_npz_sha256": runner.file_sha256(npz),
    }


class _FakeWrapper:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.generators: list[torch.Generator] = []

    def generate_accompaniment(self, **kwargs):
        generator = kwargs["generator"]
        self.generators.append(generator)
        self.calls.append(kwargs)
        acc_beats = []
        generated_template = None
        for step in kwargs["schedule"]:
            if step.action == "inject_gt":
                generated_template = step.data.cpu().tolist()
                acc_beats.append(generated_template)
            elif step.action == "generate":
                acc_beats.append(generated_template or [169, 170])
        return acc_beats, torch.tensor([[257, 259, 265]], dtype=torch.long)


class _FakeAdapter:
    def __init__(self) -> None:
        self.tokenizer = PianoMusicTokenizer(config=ModelConfig())
        self.wrapper = _FakeWrapper()


class _EmptyPostjoinWrapper(_FakeWrapper):
    def generate_accompaniment(self, **kwargs):
        generator = kwargs["generator"]
        self.generators.append(generator)
        self.calls.append(kwargs)
        acc_beats = []
        for step in kwargs["schedule"]:
            if step.action == "inject_gt":
                acc_beats.append(step.data.cpu().tolist())
            elif step.action == "generate":
                acc_beats.append([169, 170])
        return acc_beats, torch.tensor([[257, 259, 265]], dtype=torch.long)


class _EmptyPostjoinAdapter(_FakeAdapter):
    def __init__(self) -> None:
        self.tokenizer = PianoMusicTokenizer(config=ModelConfig())
        self.wrapper = _EmptyPostjoinWrapper()


class _Loader:
    def __init__(self, adapter=None) -> None:
        self.calls: list[dict] = []
        self.adapter = adapter or _FakeAdapter()

    def __call__(self, checkpoint_path: str, **kwargs):
        self.calls.append({"checkpoint_path": checkpoint_path, **kwargs})
        return self.adapter


def _args(runner, manifest: Path, checkpoint: Path, output: Path, *, seeds="0,1"):
    return runner.parse_args(
        [
            "--cohort-manifest",
            str(manifest),
            "--output-root",
            str(output),
            "--continuation-checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--no-fp16",
            "--seeds",
            seeds,
        ]
    )


def test_default_contract_and_samples_manifest_schema(runner, tmp_path: Path) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece-01")])

    args = runner.parse_args(
        [
            "--cohort-manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "out"),
            "--continuation-checkpoint",
            str(tmp_path / "model.safetensors"),
        ]
    )
    samples = runner.load_cohort_manifest(manifest)

    assert args.seeds == "0,1,2"
    assert runner.TEMPERATURE == pytest.approx(1.05)
    assert runner.TOP_P == pytest.approx(0.98)
    assert runner.TOP_K == 0
    assert runner.REPETITION_PENALTY == pytest.approx(1.0)
    assert samples[0].source_npz == npz.resolve()
    assert samples[0].source_npz_sha256 == runner.file_sha256(npz)


def test_checkpoint_aligned_schedule_is_exactly_8_gt_plus_24_generated(
    runner, tmp_path: Path
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    sample = runner.CohortSample(1, "piece", npz, runner.file_sha256(npz))
    window, _ = runner.load_prepared_window(sample)
    tokenizer = PianoMusicTokenizer(config=ModelConfig())

    prepared = tokenizer.build_generation_schedule(
        window.measures,
        window.metadata,
        gt_prefix_beats=runner.PREFIX_BEATS,
        timesteps_per_beat=runner.STEPS_PER_BEAT,
    )
    counts = runner.validate_schedule(prepared)

    assert counts == {
        "window_beats": 32,
        "gt_inject_beats": 8,
        "generated_beats": 24,
        "first_beat": 0,
        "last_beat_inclusive": 31,
    }
    assert prepared["initial_tokens"][1].item() == 259
    assert window.metadata["bpm"] == 120.0
    assert window.metadata["time_signature_idx"] == 0


def test_existing_nonempty_output_root_is_rejected_without_changes(
    runner, tmp_path: Path
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output root is not empty"):
        runner.run_batch(
            _args(runner, manifest, checkpoint, output, seeds="0"),
            adapter_loader=_Loader(),
        )

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert [path.name for path in output.iterdir()] == ["existing.txt"]


def test_existing_empty_output_root_is_allowed(runner, tmp_path: Path) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output"
    output.mkdir()

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output, seeds="0"),
        adapter_loader=_Loader(),
    )

    assert result.exit_code == 0
    assert (output / "frozen_settings.json").is_file()
    assert (output / "run_manifest.csv").is_file()


def test_output_root_file_is_rejected_without_modification(
    runner, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="is not a directory"):
        runner.prepare_output_root(output)

    assert output.read_text(encoding="utf-8") == "not a directory"


def test_batch_loads_model_once_and_resets_generator_per_trial(
    runner, tmp_path: Path
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece-01")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fake-checkpoint")
    output = tmp_path / "output"
    loader = _Loader()

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output), adapter_loader=loader
    )

    assert result.exit_code == 0
    assert result.complete == 2
    assert result.failed == 0
    assert len(loader.calls) == 1
    assert len(loader.adapter.wrapper.calls) == 2
    generators = loader.adapter.wrapper.generators
    assert generators[0] is not generators[1]
    assert [generator.initial_seed() for generator in generators] == [0, 1]
    assert all(
        [step.action for step in call["schedule"] if step.action in {"inject_gt", "generate"}]
        == ["inject_gt"] * 8 + ["generate"] * 24
        for call in loader.adapter.wrapper.calls
    )

    with (output / "run_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["complete", "complete"]
    assert all(row["failure_reason"] == "" for row in rows)
    assert all(row["source_npz_sha256"] == runner.file_sha256(npz) for row in rows)

    trial = json.loads(
        (output / "001_piece-01" / "seed0" / "trial_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert trial["schedule"]["gt_inject_beats"] == 8
    assert trial["schedule"]["generated_beats"] == 24
    assert trial["rng"]["seed"] == 0
    assert trial["effective_metadata"]["time_signature_idx"] == 0
    assert trial["effective_metadata"]["bpm"] == 120.0
    assert trial["frozen_settings"]["continuation_checkpoint"]["sha256"] == runner.file_sha256(
        checkpoint
    )
    assert trial["frozen_settings"]["code"]["runner_sha256"] == runner.file_sha256(
        Path(runner.__file__)
    )


def test_outputs_have_two_tracks_and_postjoin_is_shifted_to_zero(
    runner, tmp_path: Path
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "out"

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output, seeds="0"),
        adapter_loader=_Loader(),
    )

    assert result.exit_code == 0
    trial_dir = output / "001_piece" / "seed0"
    for filename in (
        "full_generated.mid",
        "full_gt.mid",
        "postjoin_generated.mid",
        "postjoin_gt.mid",
    ):
        midi = pretty_midi.PrettyMIDI(str(trial_dir / filename))
        assert [instrument.name for instrument in midi.instruments] == [
            "Melody",
            "Accompaniment",
        ]
    postjoin_gt = pretty_midi.PrettyMIDI(str(trial_dir / "postjoin_gt.mid"))
    assert min(note.start for note in postjoin_gt.instruments[0].notes) == pytest.approx(0.0)
    assert min(note.start for note in postjoin_gt.instruments[1].notes) == pytest.approx(0.0)
    assert postjoin_gt.get_end_time() <= 12.0


def test_postjoin_crop_clips_note_crossing_join_and_shifts_it_to_zero(
    runner, tmp_path: Path
) -> None:
    source_path = tmp_path / "full.mid"
    cropped_path = tmp_path / "cropped.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    seconds_per_step = 0.125
    for name, pitch in (("Melody", 60), ("Accompaniment", 48)):
        instrument = pretty_midi.Instrument(program=0, name=name)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=30 * seconds_per_step,
                end=34 * seconds_per_step,
            )
        )
        midi.instruments.append(instrument)
    midi.write(str(source_path))

    runner.crop_midi_window(
        source_path, cropped_path, start_step=32, end_step=128
    )

    cropped = pretty_midi.PrettyMIDI(str(cropped_path))
    assert [instrument.name for instrument in cropped.instruments] == [
        "Melody",
        "Accompaniment",
    ]
    for instrument in cropped.instruments:
        assert len(instrument.notes) == 1
        assert instrument.notes[0].start == pytest.approx(0.0)
        assert instrument.notes[0].end == pytest.approx(2 * seconds_per_step)


def test_empty_generated_postjoin_is_complete_and_preserves_empty_acc_track(
    runner, tmp_path: Path
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "out"
    loader = _Loader(_EmptyPostjoinAdapter())

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output, seeds="0"),
        adapter_loader=loader,
    )

    assert result.exit_code == 0
    trial_dir = output / "001_piece" / "seed0"
    assert all(
        (trial_dir / filename).is_file()
        for filename in (
            "full_generated.mid",
            "full_gt.mid",
            "postjoin_generated.mid",
            "postjoin_gt.mid",
        )
    )
    postjoin_path = trial_dir / "postjoin_generated.mid"
    physical_names = [
        message.name
        for track in mido.MidiFile(postjoin_path).tracks
        for message in track
        if message.is_meta
        and message.type == "track_name"
        and message.name in {"Melody", "Accompaniment"}
    ]
    assert physical_names == ["Melody", "Accompaniment"]
    postjoin = pretty_midi.PrettyMIDI(str(postjoin_path))
    assert all(instrument.name != "Accompaniment" for instrument in postjoin.instruments)
    trial = json.loads((trial_dir / "trial_manifest.json").read_text(encoding="utf-8"))
    assert trial["status"] == "complete"
    assert trial["outputs"]["postjoin_generated_midi"]["track_note_counts"] == {
        "Melody": 24,
        "Accompaniment": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "hash mismatch"),
        ("time_signature_text", "time_signature must be exactly"),
        ("time_signature", "time_signature_idx"),
        ("shape", "shape must be"),
        ("length", "too short"),
    ],
)
def test_source_npz_validation_rejects_invalid_inputs(
    runner, tmp_path: Path, mutation: str, message: str
) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(
        npz,
        measures=7 if mutation == "length" else 8,
        time_signature="3/4" if mutation == "time_signature_text" else "4/4",
        time_signature_idx=1 if mutation == "time_signature" else 0,
        bad_shape=mutation == "shape",
    )
    expected_hash = "f" * 64 if mutation == "hash" else runner.file_sha256(npz)
    sample = runner.CohortSample(1, "piece", npz, expected_hash)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        runner.load_prepared_window(sample)


def test_batch_continues_after_failure_and_returns_nonzero(
    runner, tmp_path: Path
) -> None:
    invalid = tmp_path / "invalid.npz"
    valid = tmp_path / "valid.npz"
    _write_npz(invalid, time_signature_idx=1)
    _write_npz(valid)
    manifest = _write_manifest(
        runner,
        tmp_path,
        [
            _sample_row(runner, invalid, "bad", order=1),
            _sample_row(runner, valid, "good", order=2),
        ],
    )
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "out"
    loader = _Loader()

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output, seeds="0"), adapter_loader=loader
    )

    assert result.exit_code == 1
    assert result.complete == 1
    assert result.failed == 1
    assert len(loader.calls) == 1
    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["piece_id"], row["status"]) for row in rows] == [
        ("bad", "failed"),
        ("good", "complete"),
    ]
    assert "time_signature_idx" in rows[0]["failure_reason"]
    assert rows[1]["failure_reason"] == ""
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed_with_failures"
    assert summary["model_loads"] == 1


def test_model_load_failure_writes_every_planned_trial(runner, tmp_path: Path) -> None:
    npz = tmp_path / "source.npz"
    _write_npz(npz)
    manifest = _write_manifest(runner, tmp_path, [_sample_row(runner, npz, "piece")])
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "out"

    def failing_loader(*_args, **_kwargs):
        raise RuntimeError("checkpoint cannot be loaded")

    result = runner.run_batch(
        _args(runner, manifest, checkpoint, output, seeds="0,1"),
        adapter_loader=failing_loader,
    )

    assert result.exit_code == 1
    assert result.complete == 0
    assert result.failed == 2
    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["status"] == "failed" for row in rows)
    assert all("checkpoint cannot be loaded" in row["failure_reason"] for row in rows)
    for seed in (0, 1):
        trial = json.loads(
            (output / "001_piece" / f"seed{seed}" / "trial_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert trial["status"] == "failed"
        assert "checkpoint cannot be loaded" in trial["failure_reason"]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["model_load_attempts"] == 1
    assert summary["model_loads"] == 0

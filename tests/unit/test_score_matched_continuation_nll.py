from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (
    PianoMusicTokenizer,
)


@pytest.fixture
def scorer() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    module_name = "_test_score_matched_continuation_nll"
    spec = importlib.util.spec_from_file_location(
        module_name, root / "scripts" / "score_matched_continuation_nll.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load score_matched_continuation_nll.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(module_name, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_midi(
    path: Path,
    *,
    names: tuple[str, str] = ("Melody", "Accompaniment"),
    tempo: float = 120.0,
    time_signature: tuple[int, int] | None = (4, 4),
    melody_note: tuple[float, float, int] = (0.0, 0.25, 60),
    accompaniment_note: tuple[float, float, int] = (0.5, 1.0, 48),
    extra_track: bool = False,
) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    if time_signature is not None:
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(*time_signature, time=0.0)
        )
    for name, note_data in zip(names, (melody_note, accompaniment_note)):
        instrument = pretty_midi.Instrument(program=0, name=name)
        start, end, pitch = note_data
        instrument.notes.append(
            pretty_midi.Note(
                velocity=80, pitch=pitch, start=float(start), end=float(end)
            )
        )
        midi.instruments.append(instrument)
    if extra_track:
        extra = pretty_midi.Instrument(program=0, name="Extra")
        extra.notes.append(
            pretty_midi.Note(velocity=80, pitch=55, start=0.0, end=0.25)
        )
        midi.instruments.append(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def _write_manifest(
    root: Path,
    midi_by_system: dict[str, Path],
    *,
    piece_id: str = "piece-01",
    seed: str = "0",
) -> Path:
    system_ids = list(midi_by_system)
    basename = f"piece-{piece_id}__seed-{seed}.mid"
    trials = []
    for system_id, source in midi_by_system.items():
        target = root / system_id / "generated" / basename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        trials.append(
            {
                "system_id": system_id,
                "piece_id": piece_id,
                "seed": seed,
                "basename": basename,
                "common_generated_midi": target.relative_to(root).as_posix(),
                "generated_sha256": _sha256(target),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key_fields": ["piece_id", "seed"],
                "system_ids": system_ids,
                "common_valid_key_count": 1,
                "common_valid_keys": [{"piece_id": piece_id, "seed": seed}],
                "trials": trials,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


class _FakeTokenizer:
    def __init__(self, *, sequence_length: int = 9) -> None:
        self.vocab = SimpleNamespace(pad_token_id=258)
        self.sequence_length = sequence_length
        self.calls: list[tuple[list[np.ndarray], dict, dict]] = []

    def build_training_sequence(self, measures, metadata, **kwargs):
        self.calls.append((measures, dict(metadata), dict(kwargs)))
        if self.sequence_length == 9:
            input_ids = torch.tensor([257, 259, 265, 255, 172, 10, 170, 11, 171])
            labels = torch.tensor([258, 258, 258, 258, 258, 10, 170, 258, 258])
            return input_ids, labels
        input_ids = torch.arange(self.sequence_length, dtype=torch.long) % 20
        labels = torch.full((self.sequence_length,), 258, dtype=torch.long)
        labels[-1] = 11
        return input_ids, labels


class _UniformModel:
    def __init__(self, vocab_size: int = 300) -> None:
        self.vocab_size = vocab_size
        self.calls = 0
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, *, input_ids, attention_mask):
        self.calls += 1
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        return SimpleNamespace(
            logits=torch.zeros(
                input_ids.shape[0], input_ids.shape[1], self.vocab_size,
                dtype=torch.float16,
                device=input_ids.device,
            )
        )


class _Loader:
    def __init__(self, tokenizer: _FakeTokenizer, model: _UniformModel) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, checkpoint_path: str, **kwargs):
        self.calls.append((checkpoint_path, dict(kwargs)))
        return SimpleNamespace(tokenizer=self.tokenizer, model=self.model)


def test_manifest_rejects_grid_basename_and_hash_contracts(
    scorer: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    _write_midi(source)
    manifest = _write_manifest(tmp_path / "valid", {"a": source, "b": source})
    _payload, selected, rows = scorer.validate_manifest(manifest)
    assert selected == ["a", "b"]
    assert [row["_key"] for row in rows["a"]] == [("piece-01", "0")]

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["schema_version"] = 2
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(scorer.ContractError, match="schema_version"):
        scorer.validate_manifest(manifest)

    manifest = _write_manifest(tmp_path / "bad-grid", {"a": source, "b": source})

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["trials"] = data["trials"][:-1]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(scorer.ContractError, match="key grid"):
        scorer.validate_manifest(manifest)

    manifest = _write_manifest(tmp_path / "bad-name", {"a": source})
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["trials"][0]["basename"] = "wrong.mid"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(scorer.ContractError, match="does not match path"):
        scorer.validate_manifest(manifest)

    manifest = _write_manifest(tmp_path / "bad-hash", {"a": source})
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["trials"][0]["generated_sha256"] = "A" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(scorer.ContractError, match="lowercase SHA256"):
        scorer.validate_manifest(manifest)


def test_manifest_path_and_content_hash_are_fail_closed(
    scorer: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    _write_midi(source)
    manifest = _write_manifest(tmp_path / "case", {"a": source})
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    row = json.loads(manifest.read_text(encoding="utf-8"))["trials"][0]
    assert scorer._safe_manifest_path(manifest, row["common_generated_midi"]).is_file()
    with pytest.raises(scorer.ContractError, match="must be relative"):
        scorer._safe_manifest_path(manifest, str(source.resolve()))
    with pytest.raises(scorer.ContractError, match="escapes"):
        scorer._safe_manifest_path(manifest, "../source.mid")

    midi_path = manifest.parent / row["common_generated_midi"]
    midi_path.write_bytes(midi_path.read_bytes() + b"tampered")
    loader = _Loader(_FakeTokenizer(), _UniformModel())
    with pytest.raises(scorer.ContractError, match="SHA256 mismatch"):
        scorer.score_manifest(
            common_valid_manifest=manifest,
            continuation_checkpoint=checkpoint,
            system_ids=None,
            device="cpu",
            fp16=False,
            adapter_loader=loader,
            code_identity_loader=lambda _root: {"git_commit": "test"},
        )
    assert loader.calls == []

    malformed_source = tmp_path / "malformed.mid"
    malformed_source.write_bytes(b"this-is-not-a-midi-file")
    malformed_manifest = _write_manifest(
        tmp_path / "malformed-case", {"a": malformed_source}
    )
    with pytest.raises(scorer.ContractError, match="cannot parse MIDI"):
        scorer.score_manifest(
            common_valid_manifest=malformed_manifest,
            continuation_checkpoint=checkpoint,
            system_ids=None,
            device="cpu",
            fp16=False,
            adapter_loader=loader,
            code_identity_loader=lambda _root: {"git_commit": "test"},
        )
    assert loader.calls == []


def test_track_split_quantization_and_fixed_96_tick_horizon(
    scorer: ModuleType, tmp_path: Path
) -> None:
    midi = tmp_path / "valid.mid"
    _write_midi(midi)
    measures = scorer.midi_to_measures(midi)
    assert len(measures) == 6
    assert all(measure.shape == (4, 88, 16) for measure in measures)
    full_roll = np.concatenate(measures, axis=2)
    assert full_roll.shape == (4, 88, 96)
    assert full_roll[1, 60 - 21, 0] == 1
    assert full_roll[3, 48 - 21, 4] == 1
    assert full_roll[3, 60 - 21, 0] == 0

    reversed_tracks = tmp_path / "reversed.mid"
    _write_midi(reversed_tracks, names=("Accompaniment", "Melody"))
    with pytest.raises(scorer.ContractError, match="named and ordered exactly"):
        scorer.midi_to_measures(reversed_tracks)

    extra = tmp_path / "extra.mid"
    _write_midi(extra, extra_track=True)
    with pytest.raises(scorer.ContractError, match="exactly two"):
        scorer.midi_to_measures(extra)

    too_long = tmp_path / "too-long.mid"
    _write_midi(too_long, accompaniment_note=(11.9, 12.1, 48))
    with pytest.raises(scorer.ContractError, match="12.0-second horizon"):
        scorer.midi_to_measures(too_long)

    wrong_tempo = tmp_path / "tempo.mid"
    _write_midi(wrong_tempo, tempo=121.0)
    with pytest.raises(scorer.ContractError, match="constant 120 BPM"):
        scorer.midi_to_measures(wrong_tempo)

    wrong_signature = tmp_path / "signature.mid"
    _write_midi(wrong_signature, time_signature=(3, 4))
    with pytest.raises(scorer.ContractError, match="4/4"):
        scorer.midi_to_measures(wrong_signature)

    missing_signature = tmp_path / "missing-signature.mid"
    _write_midi(missing_signature, time_signature=None)
    missing_signature_midi = pretty_midi.PrettyMIDI(str(missing_signature))
    missing_signature_midi.time_signature_changes = []
    with pytest.raises(scorer.ContractError, match="explicitly declare 4/4"):
        scorer.midi_to_measures(
            missing_signature, midi_loader=lambda _path: missing_signature_midi
        )


def test_mask_causal_token_count_and_float32_cross_entropy(
    scorer: ModuleType,
) -> None:
    tokenizer = _FakeTokenizer()
    measures = [np.zeros((4, 88, 16), dtype=np.uint8) for _ in range(6)]
    input_ids, labels, token_count = scorer.build_scoring_sequence(tokenizer, measures)
    assert input_ids.tolist() == [257, 259, 265, 255, 172, 10, 170, 11, 171]
    assert labels.tolist() == [-100, -100, -100, -100, -100, 10, 170, -100, -100]
    assert token_count == 2
    call_measures, metadata, kwargs = tokenizer.calls[0]
    assert len(call_measures) == 6
    assert metadata == {
        "time_signature_idx": 0,
        "bpm": 120,
        "num_measures": 6,
        "is_continuation": True,
    }
    assert kwargs["timesteps_per_beat"] == 4
    assert kwargs["include_melody_loss"] is False

    model = _UniformModel(vocab_size=300)
    total_nll, avg_nll, scored_tokens = scorer.score_sequence(
        model, input_ids, labels, device="cpu"
    )
    assert scored_tokens == 2
    assert total_nll == pytest.approx(2 * math.log(300), rel=1e-6)
    assert avg_nll == pytest.approx(math.log(300), rel=1e-6)


def test_real_tokenizer_labels_only_accompaniment_segments(scorer: ModuleType) -> None:
    tokenizer = PianoMusicTokenizer()
    measures = [np.zeros((4, 88, 16), dtype=np.uint8) for _ in range(6)]
    _input_ids, labels, token_count = scorer.build_scoring_sequence(tokenizer, measures)
    metadata = {
        "time_signature_idx": 0,
        "bpm": 120,
        "num_measures": 6,
        "is_continuation": True,
    }
    _ts, _bpm, _continuation, measure_beats = tokenizer._encode_measures(
        measures, metadata, 4, 0
    )
    expected_acc = torch.cat(
        [acc for beats in measure_beats for acc, _mel in beats]
    )
    actual_targets = labels[labels != -100]
    assert torch.equal(actual_targets, expected_acc)
    assert token_count == expected_acc.numel()


def test_sequence_over_2048_is_rejected_without_truncation(
    scorer: ModuleType,
) -> None:
    tokenizer = _FakeTokenizer(sequence_length=2049)
    measures = [np.zeros((4, 88, 16), dtype=np.uint8) for _ in range(6)]
    with pytest.raises(scorer.ContractError, match="2049 exceeds 2048"):
        scorer.build_scoring_sequence(tokenizer, measures)


def test_batch_loads_model_once_and_emits_deterministic_trial_records(
    scorer: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source.mid"
    _write_midi(source)
    manifest = _write_manifest(tmp_path / "common", {"a": source, "b": source})
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    tokenizer = _FakeTokenizer()
    model = _UniformModel()
    loader = _Loader(tokenizer, model)

    payload = scorer.score_manifest(
        common_valid_manifest=manifest,
        continuation_checkpoint=checkpoint,
        system_ids=["b", "a"],
        device="cpu",
        fp16=False,
        adapter_loader=loader,
        code_identity_loader=lambda root: {
            "repository_root": str(root),
            "git_commit": "test-commit",
        },
    )
    assert len(loader.calls) == 1
    assert loader.calls[0][1] == {
        "device": "cpu",
        "dtype": torch.float32,
        "use_cache": False,
    }
    assert payload["schema_version"] == "streammuse.matched_continuation_nll.v1"
    assert payload["metric_contract"]["scope"] == "source beats [8, 32), shifted to MIDI beat 0"
    assert payload["metric_contract"]["midi_window_shifted_to_zero"] is True
    assert "not online latency" in payload["metric_contract"]["evaluator_scope"]
    assert payload["system_ids"] == ["b", "a"]
    assert payload["counts"] == {"systems": 2, "trials": 2, "scored": 2, "errors": 0}
    assert payload["errors"] == []
    assert model.calls == 2
    for system_id in payload["system_ids"]:
        trial = payload["systems"][system_id]["trials"][0]
        assert set(trial) == {
            "system_id",
            "piece_id",
            "seed",
            "basename",
            "generated_midi",
            "generated_sha256",
            "sequence_tokens",
            "total_tokens",
            "total_nll",
            "avg_nll",
        }
        assert trial["system_id"] == system_id
        assert trial["total_tokens"] == 2
        assert trial["avg_nll"] == pytest.approx(math.log(300), rel=1e-6)


def test_atomic_success_and_main_failure_does_not_publish(
    scorer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "nested" / "scores.json"
    payload = {"schema_version": scorer.SCHEMA_VERSION, "value": 1}
    monkeypatch.setattr(scorer, "score_manifest", lambda **_kwargs: payload)
    argv = [
        "--common-valid-manifest",
        str(manifest),
        "--continuation-checkpoint",
        str(checkpoint),
        "--output-json",
        str(output),
        "--device",
        "cpu",
        "--no-fp16",
    ]
    assert scorer.main(argv) == 0
    first_bytes = output.read_bytes()
    assert json.loads(first_bytes) == payload
    assert not list(output.parent.glob(".*.tmp"))
    assert scorer.main(argv) == 0
    assert output.read_bytes() == first_bytes

    output.unlink()

    def fail(**_kwargs):
        raise scorer.ContractError("fixture failure")

    monkeypatch.setattr(scorer, "score_manifest", fail)
    assert scorer.main(argv) == 1
    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp"))

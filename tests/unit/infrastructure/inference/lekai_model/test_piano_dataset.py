from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from streammuse.infrastructure.inference.lekai_model.PianoDataset import PianoDataset


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        patch_h=1,
        patch_w=4,
        train_cutoff_len=512,
        pad_token_id=256,
        bos_token_id=257,
        eos_token_id=258,
        bar_token_id=255,
        time_sig_offset_id=259,
        bpm_offset_id=264,
    )


def _touch_npz_files(directory, names: list[str]) -> None:
    for name in names:
        (directory / name).write_bytes(b"fixture")


def test_all_mode_keeps_all_40_npzs_in_stable_sorted_order(tmp_path):
    names = [f"variant_{idx:02d}.npz" for idx in reversed(range(40))]
    _touch_npz_files(tmp_path, names)
    (tmp_path / "ignore.txt").write_text("not an npz")

    dataset = PianoDataset(tmp_path, _config(), cache_lengths=False, mode="all")

    assert len(dataset) == 40
    assert dataset.data_files == sorted(names)
    assert dataset.data_files == sorted(dataset.data_files)
    assert [dataset.index_for_stem(name[:-4]) for name in sorted(names)] == list(range(40))


def test_all_mode_resolves_exact_stem_and_path(tmp_path):
    _touch_npz_files(tmp_path, ["b.npz", "a.npz"])
    dataset = PianoDataset(tmp_path, _config(), cache_lengths=False, mode="all")

    assert dataset.index_for_stem("b") == 1
    assert dataset.index_for_stem("b.npz") == 1
    assert dataset.index_for_path("a.npz") == 0
    assert dataset.index_for_path(tmp_path / "b.npz") == 1

    with pytest.raises(ValueError, match="No NPZ"):
        dataset.index_for_stem("missing")
    with pytest.raises(ValueError, match="not an item"):
        dataset.index_for_path(tmp_path / "missing.npz")


def test_length_cache_is_metadata_for_exact_current_sorted_list(tmp_path):
    names = ["a.npz", "b.npz", "c.npz"]
    _touch_npz_files(tmp_path, names)
    cache = {
        "patch_h": 1,
        "patch_w": 4,
        "data_files": names,
        "lengths": [30, 10, 20],
        "sorted_indices": [1, 2, 0],
    }
    with (tmp_path / ".lengths_cache.pkl").open("wb") as handle:
        pickle.dump(cache, handle)

    dataset = PianoDataset(tmp_path, _config(), cache_lengths=True, mode="all")

    assert dataset.data_files == names
    assert dataset.file_lengths == [30, 10, 20]
    assert dataset.sorted_indices == [1, 2, 0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda cache: cache.update(data_files=["b.npz", "a.npz"]), "file list/order"),
        (lambda cache: cache.update(lengths=[10]), "lengths for"),
        (lambda cache: cache.update(sorted_indices=[0, 1]), "sorted_indices"),
    ],
)
def test_stale_or_incomplete_length_cache_fails_closed(tmp_path, change, message):
    names = ["a.npz", "b.npz"]
    _touch_npz_files(tmp_path, names)
    cache = {
        "patch_h": 1,
        "patch_w": 4,
        "data_files": names,
        "lengths": [20, 10],
        "sorted_indices": [1, 0],
    }
    change(cache)
    with (tmp_path / ".lengths_cache.pkl").open("wb") as handle:
        pickle.dump(cache, handle)

    with pytest.raises(ValueError, match=message):
        PianoDataset(tmp_path, _config(), cache_lengths=True, mode="all")


def test_invalid_mode_fails_before_dataset_split(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        PianoDataset(tmp_path, _config(), cache_lengths=False, mode="validation")

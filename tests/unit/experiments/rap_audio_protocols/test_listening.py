from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF-test-wave")


def test_build_blind_map_is_deterministic_and_complete() -> None:
    listening = importlib.import_module("streammuse.experiments.rap_audio_protocols.listening")

    first = listening.build_blind_map(
        ("01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"),
        (
            ProtocolId.MOSS_GLOBAL,
            ProtocolId.TED_LOCAL,
            ProtocolId.FASTPITCH_PHONEME,
            ProtocolId.MOSS_ALIGNED,
        ),
        blind_order_seed=20260816,
    )
    second = listening.build_blind_map(
        ("01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"),
        (
            ProtocolId.MOSS_GLOBAL,
            ProtocolId.TED_LOCAL,
            ProtocolId.FASTPITCH_PHONEME,
            ProtocolId.MOSS_ALIGNED,
        ),
        blind_order_seed=20260816,
    )

    assert first == second
    assert set(first) == {"01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"}
    for mapping in first.values():
        assert set(mapping) == {"A", "B", "C", "D"}
        assert set(mapping.values()) == {
            "moss_global",
            "ted_local",
            "fastpitch_phoneme",
            "moss_aligned",
        }


def test_write_listening_package_creates_neutral_relative_a_to_d_page(tmp_path: Path) -> None:
    listening = importlib.import_module("streammuse.experiments.rap_audio_protocols.listening")
    assets = []
    for song_id in ("01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"):
        for protocol_id in (
            ProtocolId.MOSS_GLOBAL,
            ProtocolId.TED_LOCAL,
            ProtocolId.FASTPITCH_PHONEME,
            ProtocolId.MOSS_ALIGNED,
        ):
            source = tmp_path / "source" / protocol_id.value / song_id / "mix.wav"
            _write_wav(source)
            assets.append(
                listening.ListeningAsset(
                    song_id=song_id,
                    protocol_id=protocol_id,
                    title=song_id.replace("_", " "),
                    audio_path=source,
                )
            )

    outputs = listening.write_listening_package(
        output_dir=tmp_path / "package",
        assets=tuple(assets),
        blind_order_seed=20260816,
    )

    html = outputs["listening_html"].read_text(encoding="utf-8")
    blind_map = json.loads(outputs["blind_map_json"].read_text(encoding="utf-8"))
    audit = json.loads(outputs["package_audit_json"].read_text(encoding="utf-8"))

    assert html.count("<section") == 3
    assert html.count("<audio controls") == 12
    assert "moss_global" not in html
    assert "ted_local" not in html
    assert "fastpitch_phoneme" not in html
    assert "moss_aligned" not in html
    assert 'src="blind/' in html
    assert "../" not in html
    assert "Method A" in html and "Method D" in html
    assert set(blind_map) == {"01_space_exploration", "02_deep_ocean", "03_artificial_intelligence"}
    assert audit["audio_file_count"] == 12
    assert len(audit["audio_files"]) == 12

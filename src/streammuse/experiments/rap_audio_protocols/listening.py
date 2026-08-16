"""Blinded listening-page packaging for offline rap protocol comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ListeningAsset:
    song_id: str
    protocol_id: Any
    title: str
    audio_path: Path


def build_blind_map(
    song_ids: Sequence[str],
    protocol_ids: Sequence[Any],
    *,
    blind_order_seed: int = 20260816,
) -> dict[str, dict[str, str]]:
    letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    protocols = tuple(_protocol_name(protocol_id) for protocol_id in protocol_ids)
    blind_map: dict[str, dict[str, str]] = {}
    for offset, song_id in enumerate(song_ids):
        rng = random.Random(blind_order_seed + offset)
        shuffled = list(protocols)
        rng.shuffle(shuffled)
        blind_map[song_id] = {
            letters[index]: protocol_name
            for index, protocol_name in enumerate(shuffled)
        }
    return blind_map


def write_listening_package(
    *,
    output_dir: Path | str,
    assets: Sequence[ListeningAsset],
    blind_order_seed: int = 20260816,
) -> dict[str, Path]:
    if not assets:
        raise ValueError("listening package requires at least one asset")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    grouped = _group_assets(assets)
    protocol_order = tuple(_protocol_name(asset.protocol_id) for asset in grouped[next(iter(grouped))])
    blind_map = build_blind_map(tuple(grouped), protocol_order, blind_order_seed=blind_order_seed)

    audio_files = []
    for song_id, song_assets in grouped.items():
        asset_by_protocol = {_protocol_name(asset.protocol_id): asset for asset in song_assets}
        for label, protocol_name in blind_map[song_id].items():
            source = asset_by_protocol[protocol_name].audio_path
            destination = output_root / "blind" / song_id / f"{label}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            audio_files.append(
                {
                    "song_id": song_id,
                    "label": label,
                    "relative_path": str(destination.relative_to(output_root)),
                    "source_sha256": _file_sha256(source),
                    "blind_sha256": _file_sha256(destination),
                }
            )

    listening_html = output_root / "listening.html"
    blind_map_json = output_root / "blind_map.json"
    package_audit_json = output_root / "package_audit.json"
    blind_map_json.write_text(json.dumps(blind_map, indent=2, sort_keys=True), encoding="utf-8")
    package_audit_json.write_text(
        json.dumps(
            {
                "audio_file_count": len(audio_files),
                "audio_files": audio_files,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    listening_html.write_text(_render_html(grouped, blind_map, output_root), encoding="utf-8")
    return {
        "listening_html": listening_html,
        "blind_map_json": blind_map_json,
        "package_audit_json": package_audit_json,
    }


def _group_assets(assets: Sequence[ListeningAsset]) -> dict[str, tuple[ListeningAsset, ...]]:
    grouped: dict[str, list[ListeningAsset]] = {}
    for asset in assets:
        grouped.setdefault(asset.song_id, []).append(asset)
    normalized = {song_id: tuple(sorted(song_assets, key=lambda item: _protocol_name(item.protocol_id))) for song_id, song_assets in sorted(grouped.items())}
    expected_count = len(next(iter(normalized.values())))
    for song_id, song_assets in normalized.items():
        if len(song_assets) != expected_count:
            raise ValueError("every song must expose the same number of listening assets")
        if len({_protocol_name(asset.protocol_id) for asset in song_assets}) != len(song_assets):
            raise ValueError(f"duplicate protocol entries for {song_id}")
    return normalized


def _protocol_name(protocol_id: Any) -> str:
    return getattr(protocol_id, "value", str(protocol_id))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_html(
    grouped: dict[str, tuple[ListeningAsset, ...]],
    blind_map: dict[str, dict[str, str]],
    output_root: Path,
) -> str:
    sections = []
    for song_id, song_assets in grouped.items():
        title = song_assets[0].title
        controls = []
        for label in blind_map[song_id]:
            relative_path = Path("blind") / song_id / f"{label}.wav"
            controls.append(
                "\n".join(
                    [
                        '      <div class="method">',
                        f"        <h3>Method {label}</h3>",
                        f'        <audio controls preload="none" src="{relative_path.as_posix()}"></audio>',
                        "      </div>",
                    ]
                )
            )
        sections.append(
            "\n".join(
                [
                    f'  <section data-song-id="{song_id}">',
                    f"    <h2>{title}</h2>",
                    '    <div class="methods">',
                    *controls,
                    "    </div>",
                    "  </section>",
                ]
            )
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "  <head>",
            '    <meta charset="utf-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1">',
            "    <title>Rap Audio Protocol Comparison</title>",
            "    <style>",
            "      body { font-family: system-ui, sans-serif; margin: 24px; }",
            "      section { margin-bottom: 32px; }",
            "      .methods { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }",
            "      .method { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }",
            "      audio { width: 100%; }",
            "    </style>",
            "  </head>",
            "  <body>",
            "    <h1>Rap Audio Protocol Comparison</h1>",
            *sections,
            "  </body>",
            "</html>",
        ]
    )

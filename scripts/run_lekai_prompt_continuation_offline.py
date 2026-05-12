#!/usr/bin/env python
"""Run Lekai two-stage prompt+continuation offline inference.

The runner intentionally mirrors the two-stage path used by
scripts/compare_lekai_offline_realtime_raw.py:

1. Run the prompt model on the first N melody beats.
2. Convert the prompt accompaniment beats into the continuation tokenizer.
3. Inject those converted beats into the continuation schedule.
4. Run the continuation model and export final MIDI plus detailed logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODEL_ROOT = Path(os.environ.get("STREAMMUSE_MODEL_ROOT", Path.home() / "mbzuai-projects" / "models"))
DATA_DIR = Path(
    os.environ.get(
        "LEKAI_DATA_DIR",
        "/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore",
    )
)
PROMPT_CKPT = Path(
    os.environ.get(
        "LEKAI_PROMPT_CHECKPOINT_PATH",
        MODEL_ROOT / "lekai_prompt_model" / "model.safetensors",
    )
)
CONTINUATION_CKPT = Path(
    os.environ.get(
        "LEKAI_CONTINUATION_CHECKPOINT_PATH",
        MODEL_ROOT / "lekai_continuation_model" / "model.safetensors",
    )
)

TICKS_PER_BEAT = 4


def load_prompt_components():
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.config import (
        ModelConfig,
    )
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.inference import (
        load_model,
        prepare_condition,
    )
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.my_tokenizer import (
        PianoMusicTokenizer,
    )

    return SimpleNamespace(
        ModelConfig=ModelConfig,
        load_model=load_model,
        prepare_condition=prepare_condition,
        PianoMusicTokenizer=PianoMusicTokenizer,
    )


def load_continuation_components():
    from streammuse.infrastructure.inference.lekai_continuation_model.PianoDataset import (
        PianoDataset,
    )
    from streammuse.infrastructure.inference.lekai_continuation_model.config import (
        ModelConfig,
    )
    from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
        PianoContinuationAdapter,
    )
    from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import (
        PianoMusicTokenizer,
    )

    return SimpleNamespace(
        ModelConfig=ModelConfig,
        PianoContinuationAdapter=PianoContinuationAdapter,
        PianoDataset=PianoDataset,
        PianoMusicTokenizer=PianoMusicTokenizer,
    )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int]:
    event_type_order = 0 if event["type"] == "note_off" else 1
    return int(event["time"]), event_type_order, int(event["note"])


def normalize_events(
    events: Iterable[dict[str, Any]],
    min_tick_inclusive: int = 0,
    max_tick_exclusive: int | None = None,
) -> list[dict[str, int | str]]:
    out: list[dict[str, int | str]] = []
    for event in events:
        tick = int(event["time"])
        if tick < min_tick_inclusive:
            continue
        if max_tick_exclusive is not None and tick >= max_tick_exclusive:
            continue
        out.append(
            {
                "time": tick,
                "type": str(event["type"]),
                "note": int(event["note"]),
                "velocity": int(event.get("velocity", 0)),
            }
        )
    return sorted(out, key=event_sort_key)


def sha_events(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def beats_to_events(
    tokenizer: Any,
    acc_beats: Iterable[Any],
    min_tick_inclusive: int = 0,
    max_tick_exclusive: int | None = None,
) -> list[dict[str, int | str]]:
    pr = tokenizer.decode_beats_to_pianoroll(
        list(acc_beats),
        track_marker_id=tokenizer.vocab.track_marker_acc,
    )
    sustain = pr[0]
    onset = pr[1]
    events: list[dict[str, int | str]] = []

    for pitch_idx in range(88):
        note = pitch_idx + 21
        for onset_pos in np.where(onset[pitch_idx] > 0)[0]:
            start = int(onset_pos)
            end = start + 1
            while end < sustain.shape[1] and sustain[pitch_idx, end] > 0:
                end += 1
            events.append({"time": start, "type": "note_on", "note": note, "velocity": 80})
            events.append({"time": end, "type": "note_off", "note": note, "velocity": 0})

    return normalize_events(events, min_tick_inclusive, max_tick_exclusive)


def convert_prompt_acc_to_continuation(
    prompt_tokenizer: Any,
    continuation_tokenizer: Any,
    prompt_acc_beat: Any,
) -> torch.Tensor:
    prompt_vocab = prompt_tokenizer.vocab
    continuation_vocab = continuation_tokenizer.vocab
    pr = prompt_tokenizer.decode_beats_to_pianoroll(
        [prompt_acc_beat],
        track_marker_id=prompt_vocab.track_marker_acc,
    )

    beat_width = getattr(continuation_vocab, "default_patch_w", TICKS_PER_BEAT)
    if pr.shape[2] < beat_width:
        pr = np.pad(pr, ((0, 0), (0, 0), (0, beat_width - pr.shape[2])), mode="constant")
    elif pr.shape[2] > beat_width:
        pr = pr[:, :, :beat_width]

    tokens = continuation_tokenizer._codec.image_to_patch_tokens(pr, strict_mode=True)
    compressed = continuation_tokenizer.compress_tokens(
        tokens,
        track_marker=continuation_vocab.track_marker_acc,
    )
    return torch.tensor(compressed, dtype=torch.long)


def inject_prompt_prefix(schedule: list[Any], prompt_prefix_beats: list[torch.Tensor]) -> tuple[list[Any], int]:
    patched_schedule = []
    acc_action_idx = 0
    injected = 0

    for step in schedule:
        if step.action in {"generate", "inject_gt"}:
            if acc_action_idx < len(prompt_prefix_beats):
                patched_schedule.append(type(step)("inject_gt", prompt_prefix_beats[acc_action_idx]))
                injected += 1
            else:
                patched_schedule.append(type(step)("generate", None))
            acc_action_idx += 1
        else:
            patched_schedule.append(step)

    return patched_schedule, injected


def truncate_schedule_to_acc_beats(schedule: list[Any], acc_beat_count: int) -> list[Any]:
    kept = []
    acc_seen = 0

    for step in schedule:
        kept.append(step)
        if step.action in {"generate", "inject_gt"}:
            acc_seen += 1
            if acc_seen >= acc_beat_count:
                break

    return kept


def tokens_to_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [tokens_to_list(item) for item in value]
    if isinstance(value, list):
        return [tokens_to_list(item) for item in value]
    if isinstance(value, dict):
        return {str(key): tokens_to_list(item) for key, item in value.items()}
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return tokens_to_list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def save_tensor(path: Path, tensor: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def schedule_to_log(schedule: list[Any], preview_tokens: int, full: bool) -> list[dict[str, Any]]:
    log = []

    for index, step in enumerate(schedule):
        data = tokens_to_list(step.data)
        entry: dict[str, Any] = {
            "index": index,
            "action": step.action,
            "data_len": len(data) if isinstance(data, list) else None,
        }

        if full:
            entry["data"] = data
        elif isinstance(data, list):
            entry["data_head"] = data[:preview_tokens]
            entry["data_tail"] = data[-preview_tokens:] if len(data) > preview_tokens else []
        elif data is not None:
            entry["data"] = data

        log.append(entry)

    return log


def get_npz_path(data_dir: Path, npz_id: str) -> Path:
    candidate = Path(npz_id)
    if candidate.exists():
        return candidate.resolve()

    stem = candidate.stem
    npz_path = data_dir / f"{stem}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find npz file: {candidate} or {npz_path}")

    return npz_path


def require_existing_path(path: Path, label: str) -> None:
    option = f"--{label.replace('_', '-')}"
    env_var = {
        "prompt_checkpoint": "LEKAI_PROMPT_CHECKPOINT_PATH",
        "continuation_checkpoint": "LEKAI_CONTINUATION_CHECKPOINT_PATH",
    }.get(label, "the matching environment variable")

    if path.exists():
        return
    if path.is_symlink():
        raise FileNotFoundError(
            f"{label} symlink is broken: {path} -> {os.readlink(path)}. "
            f"Pass {option} or set {env_var}."
        )
    raise FileNotFoundError(
        f"{label} not found: {path}. "
        f"Pass {option} or set {env_var}."
    )


def prepare_continuation_generation(dataset: Any, index: int, gt_prefix_beats: int = 0) -> dict[str, Any]:
    file_path = Path(dataset.root_dir) / dataset.data_files[index]
    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    num_measures = int(metadata["num_measures"])
    measures = [save_dict[f"measure_{measure_idx}"] for measure_idx in range(num_measures)]
    prep = dataset.tokenizer.build_generation_schedule(
        measures=measures,
        metadata=metadata,
        gt_prefix_beats=gt_prefix_beats,
    )
    prep["vocab"] = dataset.tokenizer.vocab
    return prep


def load_npz_metadata(npz_path: Path) -> tuple[dict[str, Any], int]:
    data = np.load(npz_path, allow_pickle=True)
    metadata = data["metadata"].item() if "metadata" in data else {}
    beats_per_bar = int(data["measure_0"].shape[2] // TICKS_PER_BEAT)
    return metadata, beats_per_bar


def load_full_roll(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    metadata = data["metadata"].item() if "metadata" in data else {}
    num_measures = int(metadata.get("num_measures", 0))
    if num_measures <= 0:
        measure_keys = sorted(
            (key for key in data.files if key.startswith("measure_")),
            key=lambda item: int(item.split("_")[1]),
        )
    else:
        measure_keys = [f"measure_{idx}" for idx in range(num_measures)]
    return np.concatenate([data[key] for key in measure_keys], axis=2)


def piano_roll_to_note_events(track_roll: np.ndarray) -> list[dict[str, int | str]]:
    sustain = track_roll[0] > 0
    onset = track_roll[1] > 0
    events: list[dict[str, int | str]] = []

    for pitch_idx in range(sustain.shape[0]):
        tick = 0
        while tick < sustain.shape[1]:
            if onset[pitch_idx, tick] and sustain[pitch_idx, tick]:
                end_tick = tick + 1
                while end_tick < sustain.shape[1] and sustain[pitch_idx, end_tick]:
                    end_tick += 1

                note = pitch_idx + 21
                events.append({"time": tick, "type": "note_on", "note": note, "velocity": 80})
                events.append({"time": end_tick, "type": "note_off", "note": note, "velocity": 0})
                tick = end_tick
            else:
                tick += 1

    return normalize_events(events)


def midi_varlen(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"MIDI delta time cannot be negative: {value}")

    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7

    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def midi_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return chunk_type + len(data).to_bytes(4, "big") + data


def midi_track(messages: list[tuple[int, bytes]], end_tick: int | None = None) -> bytes:
    data = bytearray()
    previous_tick = 0

    for tick, payload in sorted(messages, key=lambda item: item[0]):
        tick = int(tick)
        data.extend(midi_varlen(tick - previous_tick))
        data.extend(payload)
        previous_tick = tick

    if end_tick is None:
        end_tick = previous_tick
    data.extend(midi_varlen(max(0, int(end_tick) - previous_tick)))
    data.extend(b"\xff\x2f\x00")
    return midi_chunk(b"MTrk", bytes(data))


def midi_track_name(name: str) -> bytes:
    encoded = name.encode("ascii", errors="replace")
    return b"\xff\x03" + midi_varlen(len(encoded)) + encoded


def events_to_midi_messages(
    events: list[dict[str, Any]],
    channel: int,
    velocity: int,
) -> tuple[list[tuple[int, bytes]], int]:
    messages: list[tuple[int, bytes]] = [(0, midi_track_name("Melody" if channel == 0 else "Accompaniment"))]
    messages.append((0, bytes([0xC0 | channel, 0])))
    max_tick = 0

    for event in sorted(events, key=event_sort_key):
        tick = int(event["time"])
        note = max(0, min(127, int(event["note"])))
        max_tick = max(max_tick, tick)

        if event["type"] == "note_on":
            event_velocity = int(event.get("velocity", velocity)) or velocity
            event_velocity = max(1, min(127, event_velocity))
            messages.append((tick, bytes([0x90 | channel, note, event_velocity])))
        elif event["type"] == "note_off":
            messages.append((tick, bytes([0x80 | channel, note, 0])))

    return messages, max_tick


def write_midi(
    path: Path,
    melody_events: list[dict[str, Any]],
    accompaniment_events: list[dict[str, Any]],
    tempo: float,
    beats_per_bar: int,
    velocity: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_quarter = TICKS_PER_BEAT
    microseconds_per_quarter = int(round(60_000_000 / float(tempo)))

    meta_messages = [
        (0, midi_track_name("Tempo")),
        (0, b"\xff\x51\x03" + microseconds_per_quarter.to_bytes(3, "big")),
        (0, bytes([0xFF, 0x58, 0x04, int(beats_per_bar), 2, 24, 8])),
    ]

    melody_messages, melody_max_tick = events_to_midi_messages(melody_events, channel=0, velocity=velocity)
    acc_messages, acc_max_tick = events_to_midi_messages(accompaniment_events, channel=1, velocity=velocity)
    end_tick = max(melody_max_tick, acc_max_tick) + TICKS_PER_BEAT

    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big")
    header += (3).to_bytes(2, "big") + ticks_per_quarter.to_bytes(2, "big")
    payload = b"".join(
        [
            header,
            midi_track(meta_messages, end_tick=0),
            midi_track(melody_messages, end_tick=end_tick),
            midi_track(acc_messages, end_tick=end_tick),
        ]
    )
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-stage Lekai prompt+continuation offline inference and export MIDI/logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--npz-id", help="Single NPZ id, NPZ filename, or full NPZ path.")
    input_group.add_argument("--npz-dir", type=Path, help="Run every .npz file in this directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing NPZ files. Used only when --npz-id is not a full path.",
    )
    parser.add_argument(
        "--prompt-checkpoint",
        type=Path,
        default=PROMPT_CKPT,
        help="Prompt model checkpoint path. Defaults to LEKAI_PROMPT_CHECKPOINT_PATH or ~/mbzuai-projects/models.",
    )
    parser.add_argument(
        "--continuation-checkpoint",
        type=Path,
        default=CONTINUATION_CKPT,
        help=(
            "Continuation model checkpoint path. Defaults to "
            "LEKAI_CONTINUATION_CHECKPOINT_PATH or ~/mbzuai-projects/models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. For a single NPZ this is the exact output dir; "
            "for --npz-dir this is the batch root and each NPZ gets a subdir."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="With --npz-dir, scan recursively.")
    parser.add_argument("--limit", type=int, default=0, help="With --npz-dir, cap the number of NPZ files. 0 means all.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an NPZ if manifest.json and two_stage_final.mid already exist in its output dir.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="With --npz-dir, keep running remaining files after a failure.",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cpu, or auto.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--prompt-beats",
        type=int,
        default=0,
        help="Stage-1 prompt length in beats. 0 means 8 beats for 4/4 and 6 beats for 3/4.",
    )
    parser.add_argument(
        "--max-acc-beats",
        type=int,
        default=0,
        help="Optional cap for continuation accompaniment beats after prompt injection.",
    )

    parser.add_argument("--prompt-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-temperature", type=float, default=1.1)
    parser.add_argument("--prompt-top-k", type=int, default=0)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--prompt-fp16", action="store_true")

    parser.add_argument("--rt-temperature", type=float, default=0.8)
    parser.add_argument("--rt-top-k", type=int, default=1)
    parser.add_argument("--rt-top-p", type=float, default=0.95)
    parser.add_argument("--rt-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--rt-fp16", action="store_true")

    parser.add_argument("--schedule-token-preview", type=int, default=32)
    parser.add_argument("--log-full-schedule", action="store_true")
    parser.add_argument("--midi-tempo", type=float, default=120.0)
    parser.add_argument("--midi-velocity", type=int, default=80)

    return parser.parse_args()


def collect_npz_paths(args: argparse.Namespace) -> list[Path]:
    if args.npz_dir is None:
        return [get_npz_path(args.data_dir, args.npz_id)]

    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.exists():
        raise FileNotFoundError(f"--npz-dir not found: {npz_dir}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"--npz-dir is not a directory: {npz_dir}")

    pattern = "**/*.npz" if args.recursive else "*.npz"
    paths = sorted(path.resolve() for path in npz_dir.glob(pattern) if path.is_file())
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .npz files found under: {npz_dir}")
    return paths


def default_output_root() -> Path:
    return Path("outputs/lekai_prompt_continuation_offline")


def output_dir_for_npz(args: argparse.Namespace, npz_path: Path, batch_mode: bool) -> Path:
    if not batch_mode:
        return args.output_dir or default_output_root() / npz_path.stem

    root = args.output_dir or default_output_root()
    if args.npz_dir is not None:
        try:
            rel = npz_path.relative_to(args.npz_dir.expanduser().resolve()).with_suffix("")
            return root / rel
        except ValueError:
            pass
    return root / npz_path.stem


def is_complete_output(out_dir: Path) -> bool:
    return (out_dir / "manifest.json").exists() and (out_dir / "two_stage_final.mid").exists()


def build_runtime(args: argparse.Namespace, device: str) -> SimpleNamespace:
    load_started_at = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    generator.manual_seed(args.seed)

    timings: dict[str, float] = {}
    t0 = time.time()
    continuation = load_continuation_components()
    continuation_config = continuation.ModelConfig()
    continuation_dtype = torch.float16 if args.rt_fp16 and device.startswith("cuda") else None
    continuation_adapter = continuation.PianoContinuationAdapter.from_checkpoint(
        str(args.continuation_checkpoint),
        device=device,
        dtype=continuation_dtype,
        use_cache=True,
    )
    timings["stage2_model_load_seconds"] = time.time() - t0

    t0 = time.time()
    prompt = load_prompt_components()
    prompt_config = prompt.ModelConfig()
    prompt_tokenizer = prompt.PianoMusicTokenizer(config=prompt_config)
    prompt_model = prompt.load_model(
        str(args.prompt_checkpoint),
        prompt_config,
        device=device,
        use_fp16=args.prompt_fp16,
    )
    timings["stage1_model_load_seconds"] = time.time() - t0
    timings["runtime_load_seconds"] = time.time() - load_started_at

    return SimpleNamespace(
        device=device,
        generator=generator,
        continuation=continuation,
        continuation_config=continuation_config,
        continuation_model=continuation_adapter.wrapper,
        prompt=prompt,
        prompt_config=prompt_config,
        prompt_tokenizer=prompt_tokenizer,
        prompt_model=prompt_model,
        load_timings=timings,
    )


def run_one_npz(
    args: argparse.Namespace,
    runtime: SimpleNamespace,
    npz_path: Path,
    out_dir: Path,
    *,
    batch_index: int | None = None,
    batch_total: int | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    effective_data_dir = npz_path.parent
    npz_stem = npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = runtime.device
    generator = runtime.generator

    metadata, beats_per_bar = load_npz_metadata(npz_path)
    prompt_beats = args.prompt_beats or (6 if beats_per_bar == 3 else 8)
    if prompt_beats % beats_per_bar != 0:
        raise ValueError(
            f"--prompt-beats must be divisible by beats_per_bar={beats_per_bar}; got {prompt_beats}",
        )
    prompt_bars = prompt_beats // beats_per_bar

    full_roll = load_full_roll(npz_path)
    melody_events = piano_roll_to_note_events(full_roll[:2])
    write_json(out_dir / "melody_events.json", melody_events)

    timings: dict[str, float] = {}

    t0 = time.time()
    dataset = runtime.continuation.PianoDataset(
        str(effective_data_dir),
        config=runtime.continuation_config,
        cache_lengths=False,
        mode="test",
    )
    dataset.data_files = [npz_path.name]
    continuation_prep = prepare_continuation_generation(dataset, 0, gt_prefix_beats=0)
    timings["stage2_prepare_seconds"] = time.time() - t0

    t0 = time.time()
    prompt_dataset = SimpleNamespace(
        root_dir=str(effective_data_dir),
        data_files=[npz_path.name],
        tokenizer=runtime.prompt_tokenizer,
    )
    prompt_prep = runtime.prompt.prepare_condition(prompt_dataset, 0, num_bars=prompt_bars)
    if prompt_prep is None:
        raise RuntimeError(f"Prompt preparation returned None for {npz_path}")
    timings["stage1_prepare_seconds"] = time.time() - t0

    t0 = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    prompt_generated = runtime.prompt_model.generate_music(
        initial_tokens=prompt_prep["prompt_tokens"],
        device=device,
        max_length=len(prompt_prep["prompt_tokens"]) + args.prompt_max_new_tokens,
        temperature=args.prompt_temperature,
        top_k=args.prompt_top_k,
        top_p=args.prompt_top_p,
        repetition_penalty=args.prompt_repetition_penalty,
    )
    _, prompt_acc_beats = runtime.prompt_tokenizer.parse_generated_sequence(prompt_generated.squeeze(0))
    prompt_acc_beats = prompt_acc_beats[:prompt_beats]
    timings["stage1_generate_seconds"] = time.time() - t0

    stage1_events = beats_to_events(
        runtime.prompt_tokenizer,
        prompt_acc_beats,
        max_tick_exclusive=prompt_beats * TICKS_PER_BEAT,
    )
    stage1_midi_path = out_dir / "stage1_prompt_accompaniment.mid"
    write_midi(
        stage1_midi_path,
        melody_events,
        stage1_events,
        tempo=args.midi_tempo,
        beats_per_bar=beats_per_bar,
        velocity=args.midi_velocity,
    )

    stage1_token_log = {
        "npz_path": str(npz_path),
        "prompt_beats": prompt_beats,
        "prompt_bars": prompt_bars,
        "params": {
            "max_new_tokens": args.prompt_max_new_tokens,
            "temperature": args.prompt_temperature,
            "top_k": args.prompt_top_k,
            "top_p": args.prompt_top_p,
            "repetition_penalty": args.prompt_repetition_penalty,
            "fp16": args.prompt_fp16,
        },
        "prompt_tokens": tokens_to_list(prompt_prep["prompt_tokens"]),
        "generated_tokens": tokens_to_list(prompt_generated.squeeze(0)),
        "prompt_acc_beats": tokens_to_list(prompt_acc_beats),
        "event_count": len(stage1_events),
        "event_sha256": sha_events(stage1_events),
    }
    save_tensor(out_dir / "stage1_prompt_tokens.pt", prompt_prep["prompt_tokens"])
    save_tensor(out_dir / "stage1_generated_tokens.pt", prompt_generated)
    write_json(out_dir / "stage1_token_log.json", stage1_token_log)
    write_json(out_dir / "stage1_prompt_accompaniment_events.json", stage1_events)

    converted_prompt_beats = [
        convert_prompt_acc_to_continuation(runtime.prompt_tokenizer, dataset.tokenizer, beat)
        for beat in prompt_acc_beats
    ]
    continuation_prep["schedule"], injected = inject_prompt_prefix(
        continuation_prep["schedule"],
        converted_prompt_beats,
    )
    if injected != len(converted_prompt_beats):
        raise RuntimeError(
            f"Only injected {injected}/{len(converted_prompt_beats)} prompt beats into continuation schedule",
        )

    if args.max_acc_beats > 0:
        continuation_prep["schedule"] = truncate_schedule_to_acc_beats(
            continuation_prep["schedule"],
            args.max_acc_beats,
        )

    schedule_log = schedule_to_log(
        continuation_prep["schedule"],
        preview_tokens=args.schedule_token_preview,
        full=args.log_full_schedule,
    )
    write_json(out_dir / "stage2_schedule_log.json", schedule_log)

    t0 = time.time()
    generator.manual_seed(args.seed)
    acc_beats, continuation_generated = runtime.continuation_model.generate_accompaniment(
        initial_tokens=continuation_prep["initial_tokens"],
        schedule=continuation_prep["schedule"],
        vocab=continuation_prep["vocab"],
        device=device,
        temperature=args.rt_temperature,
        top_k=args.rt_top_k,
        top_p=args.rt_top_p,
        repetition_penalty=args.rt_repetition_penalty,
        generator=generator,
        verbose=False,
    )
    timings["stage2_generate_seconds"] = time.time() - t0

    final_events = beats_to_events(dataset.tokenizer, acc_beats)
    final_midi_path = out_dir / "two_stage_final.mid"
    write_midi(
        final_midi_path,
        melody_events,
        final_events,
        tempo=args.midi_tempo,
        beats_per_bar=beats_per_bar,
        velocity=args.midi_velocity,
    )

    stage2_token_log = {
        "npz_path": str(npz_path),
        "prompt_beats": prompt_beats,
        "injected_prompt_beats": injected,
        "max_acc_beats": args.max_acc_beats or None,
        "params": {
            "temperature": args.rt_temperature,
            "top_k": args.rt_top_k,
            "top_p": args.rt_top_p,
            "repetition_penalty": args.rt_repetition_penalty,
            "fp16": args.rt_fp16,
        },
        "initial_tokens": tokens_to_list(continuation_prep["initial_tokens"]),
        "generated_tokens": tokens_to_list(continuation_generated),
        "acc_beats": tokens_to_list(acc_beats),
        "converted_prompt_prefix_beats": tokens_to_list(converted_prompt_beats),
        "schedule_log_path": str(out_dir / "stage2_schedule_log.json"),
        "event_count": len(final_events),
        "event_sha256": sha_events(final_events),
    }
    save_tensor(out_dir / "stage2_initial_tokens.pt", continuation_prep["initial_tokens"])
    save_tensor(out_dir / "stage2_generated_tokens.pt", continuation_generated)
    write_json(out_dir / "stage2_token_log.json", stage2_token_log)
    write_json(out_dir / "two_stage_accompaniment_events.json", final_events)

    timings["total_seconds"] = time.time() - started_at
    timings.update(runtime.load_timings)
    manifest = {
        "npz_id": npz_stem,
        "npz_path": str(npz_path),
        "data_dir": str(effective_data_dir),
        "output_dir": str(out_dir),
        "device": device,
        "seed": args.seed,
        "metadata": metadata,
        "beats_per_bar": beats_per_bar,
        "prompt_beats": prompt_beats,
        "prompt_bars": prompt_bars,
        "batch": {
            "index": batch_index,
            "total": batch_total,
        },
        "paths": {
            "prompt_checkpoint": str(args.prompt_checkpoint),
            "continuation_checkpoint": str(args.continuation_checkpoint),
            "final_midi": str(final_midi_path),
            "stage1_midi": str(stage1_midi_path),
            "stage1_token_log": str(out_dir / "stage1_token_log.json"),
            "stage2_token_log": str(out_dir / "stage2_token_log.json"),
            "stage2_schedule_log": str(out_dir / "stage2_schedule_log.json"),
            "final_events": str(out_dir / "two_stage_accompaniment_events.json"),
        },
        "counts": {
            "stage1_prompt_acc_beats": len(prompt_acc_beats),
            "stage1_events": len(stage1_events),
            "stage2_acc_beats": len(acc_beats),
            "final_events": len(final_events),
        },
        "sha256": {
            "stage1_events": sha_events(stage1_events),
            "final_events": sha_events(final_events),
        },
        "timings": timings,
        "env": {
            "STREAMMUSE_MODEL_ROOT": os.environ.get("STREAMMUSE_MODEL_ROOT", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    if batch_index is not None and batch_total is not None:
        print(f"[{batch_index}/{batch_total}] wrote {final_midi_path}")
    else:
        print(json.dumps(manifest["paths"], ensure_ascii=False, indent=2))

    return manifest


def main() -> None:
    args = parse_args()
    batch_started_at = time.time()
    npz_paths = collect_npz_paths(args)
    batch_mode = args.npz_dir is not None

    jobs: list[tuple[Path, Path]] = [
        (npz_path, output_dir_for_npz(args, npz_path, batch_mode))
        for npz_path in npz_paths
    ]
    results: list[dict[str, Any]] = []
    jobs_to_run: list[tuple[Path, Path]] = []

    for npz_path, out_dir in jobs:
        if args.skip_existing and is_complete_output(out_dir):
            results.append(
                {
                    "status": "skipped",
                    "reason": "existing_output",
                    "npz_path": str(npz_path),
                    "output_dir": str(out_dir),
                    "final_midi": str(out_dir / "two_stage_final.mid"),
                }
            )
        else:
            jobs_to_run.append((npz_path, out_dir))

    if jobs_to_run:
        require_existing_path(args.prompt_checkpoint, "prompt_checkpoint")
        require_existing_path(args.continuation_checkpoint, "continuation_checkpoint")
        device = resolve_device(args.device)
        runtime = build_runtime(args, device)
    else:
        runtime = None

    for offset, (npz_path, out_dir) in enumerate(jobs_to_run, start=1):
        batch_index = offset if batch_mode else None
        batch_total = len(jobs_to_run) if batch_mode else None
        try:
            manifest = run_one_npz(
                args,
                runtime,
                npz_path,
                out_dir,
                batch_index=batch_index,
                batch_total=batch_total,
            )
            results.append(
                {
                    "status": "ok",
                    "npz_path": str(npz_path),
                    "output_dir": str(out_dir),
                    "final_midi": manifest["paths"]["final_midi"],
                    "stage1_token_log": manifest["paths"]["stage1_token_log"],
                    "stage2_token_log": manifest["paths"]["stage2_token_log"],
                    "manifest": str(out_dir / "manifest.json"),
                }
            )
        except Exception as exc:
            error_payload = {
                "status": "failed",
                "npz_path": str(npz_path),
                "output_dir": str(out_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / "error.json", error_payload)
            results.append(error_payload)
            if not args.continue_on_error or not batch_mode:
                raise

    if batch_mode:
        summary_root = args.output_dir or default_output_root()
        summary = {
            "status": "completed",
            "npz_dir": str(args.npz_dir.expanduser().resolve()),
            "recursive": args.recursive,
            "limit": args.limit or None,
            "output_root": str(summary_root),
            "counts": {
                "total": len(jobs),
                "ok": sum(1 for item in results if item["status"] == "ok"),
                "skipped": sum(1 for item in results if item["status"] == "skipped"),
                "failed": sum(1 for item in results if item["status"] == "failed"),
            },
            "timings": {
                "total_seconds": time.time() - batch_started_at,
            },
            "results": results,
        }
        summary_path = summary_root / "batch_summary.json"
        write_json(summary_path, summary)
        print(json.dumps({"batch_summary": str(summary_path), **summary["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

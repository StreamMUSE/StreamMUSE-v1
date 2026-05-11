#!/usr/bin/env python
"""Compare RT offline two-stage output with StreamMUSE realtime raw history.

This helper is intentionally separate from audible MIDI comparison. It compares
raw accompaniment events only:

- offline: RT two-stage prompt + continuation decoded from acc beats;
- realtime: `prompt_continuation_raw_history.json` saved by streammuse-cli.

The comparison normalizes to {type, pitch, tick}, ignores velocity/channel, and
optionally truncates both sides to a common tick window.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


RT_ROOT = Path("/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time")
DATA_DIR = Path("/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore")
PROMPT_CKPT = RT_ROOT / "prompt_model/checkpoints/best_model/model.safetensors"
CONT_CKPT = Path("/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors")
LOCAL_MODULE_NAMES = ("config", "inference", "model", "PianoDataset", "Token2Midi", "my_tokenizer")


@contextmanager
def import_codebase(module_dir: Path):
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(module_dir))
    try:
        yield importlib.import_module
    finally:
        try:
            sys.path.remove(str(module_dir))
        except ValueError:
            pass
        for name in LOCAL_MODULE_NAMES:
            sys.modules.pop(name, None)


def load_prompt_components(rt_root: Path) -> SimpleNamespace:
    with import_codebase(rt_root / "prompt_model") as importer:
        return SimpleNamespace(
            ModelConfig=importer("config").ModelConfig,
            load_model=importer("inference").load_model,
            prepare_condition=importer("inference").prepare_condition,
            PianoMusicTokenizer=importer("my_tokenizer").PianoMusicTokenizer,
        )


def load_offline_components(rt_root: Path) -> SimpleNamespace:
    with import_codebase(rt_root / "offline_model") as importer:
        return SimpleNamespace(
            ModelConfig=importer("config").ModelConfig,
            load_model=importer("inference").load_model,
            prepare_generation=importer("inference").prepare_generation,
            PianoDataset=importer("PianoDataset").PianoDataset,
            PianoMusicTokenizer=importer("my_tokenizer").PianoMusicTokenizer,
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(event.get("tick", 0)),
        0 if str(event.get("type", "")) == "note_off" else 1,
        int(event.get("pitch", 0)),
    )


def normalize_events(
    events: list[dict[str, Any]],
    *,
    min_tick_inclusive: int = 0,
    max_tick_exclusive: int | None = None,
) -> list[dict[str, int | str]]:
    normalized: list[dict[str, int | str]] = []
    for event in events:
        if "pitch" not in event or "tick" not in event or "type" not in event:
            continue
        tick = int(event["tick"])
        if tick < int(min_tick_inclusive):
            continue
        if max_tick_exclusive is not None and tick >= int(max_tick_exclusive):
            continue
        normalized.append({
            "type": str(event["type"]),
            "pitch": int(event["pitch"]),
            "tick": tick,
        })
    normalized.sort(key=event_sort_key)
    return normalized


def sha_events(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def beats_to_events(
    tokenizer: Any,
    acc_beats: list[Any],
    *,
    min_tick_inclusive: int = 0,
    max_tick_exclusive: int | None = None,
) -> list[dict[str, int | str]]:
    pr = tokenizer.decode_beats_to_pianoroll(acc_beats, track_marker_id=tokenizer.vocab.track_marker_acc)
    sustain, onset = pr[0], pr[1]
    events: list[dict[str, int | str]] = []
    for pitch_idx in range(88):
        pitch = pitch_idx + 21
        for onset_pos in np.where(onset[pitch_idx] > 0)[0]:
            start = int(onset_pos)
            end = start + 1
            while end < sustain.shape[1] and sustain[pitch_idx, end] > 0:
                end += 1
            events.append({"type": "note_on", "pitch": int(pitch), "tick": int(start)})
            events.append({"type": "note_off", "pitch": int(pitch), "tick": int(end)})
    return normalize_events(
        events,
        min_tick_inclusive=min_tick_inclusive,
        max_tick_exclusive=max_tick_exclusive,
    )


def convert_prompt_acc_to_offline(prompt_tokenizer: Any, offline_tokenizer: Any, prompt_acc_beat: Any) -> torch.Tensor:
    prompt_v = prompt_tokenizer.vocab
    offline_v = offline_tokenizer.vocab
    pr = prompt_tokenizer.decode_beats_to_pianoroll([prompt_acc_beat], track_marker_id=prompt_v.track_marker_acc)

    beat_width = getattr(offline_v, "default_patch_w", 4)
    if pr.shape[2] < beat_width:
        pr = np.pad(pr, ((0, 0), (0, 0), (0, beat_width - pr.shape[2])), mode="constant")
    elif pr.shape[2] > beat_width:
        pr = pr[:, :, :beat_width]

    tokens = offline_tokenizer._codec.image_to_patch_tokens(pr, strict_mode=True)
    compressed = offline_tokenizer.compress_tokens(tokens, track_marker=offline_v.track_marker_acc)
    return torch.tensor(compressed, dtype=torch.long)


def inject_prompt_prefix(schedule: list[Any], prompt_prefix_beats: list[torch.Tensor]) -> tuple[list[Any], int]:
    new_schedule = []
    acc_action_idx = 0
    injected = 0
    for step in schedule:
        if step.action in ("generate", "inject_gt"):
            if acc_action_idx < len(prompt_prefix_beats):
                new_schedule.append(type(step)("inject_gt", prompt_prefix_beats[acc_action_idx]))
                injected += 1
            else:
                new_schedule.append(type(step)("generate", None))
            acc_action_idx += 1
        else:
            new_schedule.append(step)
    return new_schedule, injected


def truncate_schedule_to_acc_beats(schedule: list[Any], acc_beat_count: int) -> list[Any]:
    if acc_beat_count <= 0:
        return []
    out = []
    seen = 0
    for step in schedule:
        out.append(step)
        if step.action in ("generate", "inject_gt"):
            seen += 1
            if seen >= acc_beat_count:
                break
    return out


def first_diff(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any] | None:
    for idx, (ea, eb) in enumerate(zip(a, b)):
        if ea != eb:
            return {"index": idx, "offline": ea, "realtime": eb}
    if len(a) != len(b):
        idx = min(len(a), len(b))
        return {"index": idx, "offline_remaining": a[idx:idx + 5], "realtime_remaining": b[idx:idx + 5]}
    return None


def generate_offline_events(
    args: argparse.Namespace,
    *,
    compare_start_tick: int,
    compare_tick_limit: int | None,
) -> dict[str, Any]:
    device = resolve_device(args.device)
    npz_path = args.data_dir / f"{args.npz_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    beats_per_bar = int(data["measure_0"].shape[2] // 4)
    prompt_beats = args.prompt_beats
    if prompt_beats == 0:
        prompt_beats = 6 if beats_per_bar == 3 else 8
    prompt_bars = int(prompt_beats / beats_per_bar)
    if prompt_bars * beats_per_bar != prompt_beats:
        raise ValueError("prompt_beats must be divisible by beats_per_bar")

    offline = load_offline_components(args.rt_root)
    prompt = load_prompt_components(args.rt_root)

    offline_config = offline.ModelConfig()
    dataset = offline.PianoDataset(str(args.data_dir), config=offline_config, cache_lengths=False, mode="test")
    dataset.data_files = [npz_path.name]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
    generator.manual_seed(args.seed)

    offline_model = offline.load_model(
        model_path=str(args.continuation_checkpoint),
        model_config=offline_config,
        device=device,
        use_fp16=args.rt_fp16,
    )
    prep = offline.prepare_generation(dataset, 0, gt_prefix_beats=0)

    prompt_config = prompt.ModelConfig()
    prompt_tokenizer = prompt.PianoMusicTokenizer(config=prompt_config)
    prompt_dataset = SimpleNamespace(root_dir=str(args.data_dir), data_files=[npz_path.name], tokenizer=prompt_tokenizer)
    prompt_prep = prompt.prepare_condition(prompt_dataset, 0, num_bars=prompt_bars)
    if prompt_prep is None:
        raise RuntimeError("prompt prepare_condition returned None")
    prompt_model = prompt.load_model(
        model_path=str(args.prompt_checkpoint),
        model_config=prompt_config,
        device=device,
        use_fp16=args.prompt_fp16,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    prompt_generated = prompt_model.generate_music(
        initial_tokens=prompt_prep["prompt_tokens"],
        device=device,
        max_length=len(prompt_prep["prompt_tokens"]) + args.prompt_max_new_tokens,
        temperature=args.prompt_temperature,
        top_k=args.prompt_top_k,
        top_p=args.prompt_top_p,
        repetition_penalty=args.prompt_repetition_penalty,
    )
    _, prompt_acc_beats = prompt_tokenizer.parse_generated_sequence(prompt_generated.squeeze(0))
    prompt_acc_beats = prompt_acc_beats[:prompt_beats]
    offline_prefix_beats = [
        convert_prompt_acc_to_offline(prompt_tokenizer, dataset.tokenizer, beat)
        for beat in prompt_acc_beats
    ]
    prep["schedule"], injected = inject_prompt_prefix(prep["schedule"], offline_prefix_beats)

    acc_beat_limit = None
    if compare_tick_limit is not None:
        acc_beat_limit = max(1, int(np.ceil(compare_tick_limit / 4)))
        prep["schedule"] = truncate_schedule_to_acc_beats(prep["schedule"], acc_beat_limit)
    if args.max_acc_beats > 0:
        acc_beat_limit = args.max_acc_beats if acc_beat_limit is None else min(acc_beat_limit, args.max_acc_beats)
        prep["schedule"] = truncate_schedule_to_acc_beats(prep["schedule"], acc_beat_limit)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator.manual_seed(args.seed)
    acc_beats, generated = offline_model.generate_accompaniment(
        initial_tokens=prep["initial_tokens"],
        schedule=prep["schedule"],
        vocab=prep["vocab"],
        device=device,
        temperature=args.rt_temperature,
        top_k=args.rt_top_k,
        top_p=args.rt_top_p,
        repetition_penalty=args.rt_repetition_penalty,
        generator=generator,
        verbose=False,
    )
    events = beats_to_events(
        dataset.tokenizer,
        acc_beats,
        min_tick_inclusive=compare_start_tick,
        max_tick_exclusive=compare_tick_limit,
    )
    return {
        "npz_id": args.npz_id,
        "beats_per_bar": beats_per_bar,
        "prompt_beats": prompt_beats,
        "prompt_bars": prompt_bars,
        "prompt_acc_beats": len(prompt_acc_beats),
        "prompt_injected_beats": injected,
        "offline_acc_beats": len(acc_beats),
        "offline_generated_token_count": int(generated.numel()),
        "compare_tick_limit": compare_tick_limit,
        "compare_start_tick": compare_start_tick,
        "events": events,
        "events_count": len(events),
        "events_sha256": sha_events(events),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-id", required=True)
    parser.add_argument("--realtime-raw-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rt-root", type=Path, default=RT_ROOT)
    parser.add_argument("--prompt-checkpoint", type=Path, default=PROMPT_CKPT)
    parser.add_argument("--continuation-checkpoint", type=Path, default=CONT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-beats", type=int, default=0, help="0 means auto: 8 except 3/4 uses 6")
    parser.add_argument("--prompt-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-temperature", type=float, default=1.1)
    parser.add_argument("--prompt-top-k", type=int, default=0)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--rt-temperature", type=float, default=0.8)
    parser.add_argument("--rt-top-k", type=int, default=1)
    parser.add_argument("--rt-top-p", type=float, default=0.95)
    parser.add_argument("--rt-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--prompt-fp16", action="store_true")
    parser.add_argument("--rt-fp16", action="store_true")
    parser.add_argument("--compare-tick-limit", type=int, default=-1, help="-1 means infer from realtime raw max tick + 1")
    parser.add_argument("--compare-start-tick", type=int, default=0)
    parser.add_argument("--max-acc-beats", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_events = json.loads(args.realtime_raw_json.read_text(encoding="utf-8"))
    inferred_limit = None
    if raw_events:
        inferred_limit = max(int(e.get("tick", 0)) for e in raw_events if "tick" in e) + 1
    compare_tick_limit = args.compare_tick_limit
    if compare_tick_limit < 0:
        compare_tick_limit = inferred_limit
    if compare_tick_limit is not None and compare_tick_limit <= 0:
        compare_tick_limit = None

    compare_start_tick = max(0, int(args.compare_start_tick))
    realtime_events = normalize_events(
        raw_events,
        min_tick_inclusive=compare_start_tick,
        max_tick_exclusive=compare_tick_limit,
    )
    offline = generate_offline_events(
        args,
        compare_start_tick=compare_start_tick,
        compare_tick_limit=compare_tick_limit,
    )
    offline_events = offline["events"]

    result = {
        "npz_id": args.npz_id,
        "realtime_raw_json": str(args.realtime_raw_json),
        "compare_start_tick": compare_start_tick,
        "compare_tick_limit": compare_tick_limit,
        "params": {
            "seed": args.seed,
            "prompt_temperature": args.prompt_temperature,
            "prompt_top_k": args.prompt_top_k,
            "prompt_top_p": args.prompt_top_p,
            "rt_temperature": args.rt_temperature,
            "rt_top_k": args.rt_top_k,
            "rt_top_p": args.rt_top_p,
            "rt_repetition_penalty": args.rt_repetition_penalty,
            "prompt_fp16": args.prompt_fp16,
            "rt_fp16": args.rt_fp16,
        },
        "offline": {k: v for k, v in offline.items() if k != "events"},
        "realtime_event_count": len(realtime_events),
        "realtime_sha256": sha_events(realtime_events),
        "events_equal": offline_events == realtime_events,
        "first_diff": first_diff(offline_events, realtime_events),
        "offline_head": offline_events[:20],
        "realtime_head": realtime_events[:20],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["events_equal"] else 2)


if __name__ == "__main__":
    main()

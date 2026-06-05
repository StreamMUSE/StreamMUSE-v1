#!/usr/bin/env python
"""Prepare and compare StreamMUSE CLI prompt output against RT prompt reference.

This is not a StreamMUSE inference runner. The StreamMUSE candidate output must
come from `streammuse-cli` session files. This helper only:
1. exports a melody MIDI from the same NPZ used by RT offline, so CLI receives a
   deterministic input;
2. generates the RT prompt-model reference with the original RT prompt code;
3. compares that reference to `prompt_continuation_prompt_history.json` saved by
   a StreamMUSE CLI session.
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
import pretty_midi
import torch


RT_ROOT = Path("/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time")
DATA_DIR = Path("/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore")
OUT_ROOT = Path("realtime_runs/0509_prompt_alignment")
PROMPT_CKPT = RT_ROOT / "prompt_model/checkpoints/best_model/model.safetensors"
LOCAL_MODULE_NAMES = ("config", "inference", "model", "PianoDataset", "Token2Midi", "my_tokenizer")


@contextmanager
def import_rt_prompt_codebase(prompt_model_dir: Path):
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(prompt_model_dir))
    try:
        yield importlib.import_module
    finally:
        try:
            sys.path.remove(str(prompt_model_dir))
        except ValueError:
            pass
        for name in LOCAL_MODULE_NAMES:
            sys.modules.pop(name, None)


def load_rt_prompt_components(rt_root: Path) -> SimpleNamespace:
    with import_rt_prompt_codebase(rt_root / "prompt_model") as importer:
        return SimpleNamespace(
            ModelConfig=importer("config").ModelConfig,
            load_model=importer("inference").load_model,
            prepare_condition=importer("inference").prepare_condition,
            PianoMusicTokenizer=importer("my_tokenizer").PianoMusicTokenizer,
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def export_npz_melody_midi(npz_path: Path, out_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    meta = data["metadata"].item()
    tempo = float(meta.get("bpm") or 120)
    measures = [data[f"measure_{i}"] for i in range(int(meta["num_measures"]))]
    full_pr = np.concatenate(measures, axis=2)
    melody_pr = full_pr[:2]

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="Melody")
    sustain, onset = melody_pr[0], melody_pr[1]
    valid_onset = (onset > 0) & (sustain > 0)
    sec_per_tick = 60.0 / tempo / 4.0
    for pitch_idx in range(88):
        pitch = pitch_idx + 21
        for onset_pos in np.where(valid_onset[pitch_idx])[0]:
            end_pos = int(onset_pos) + 1
            while end_pos < sustain.shape[1] and sustain[pitch_idx, end_pos] > 0:
                end_pos += 1
            inst.notes.append(
                pretty_midi.Note(
                    velocity=80,
                    pitch=int(pitch),
                    start=float(onset_pos) * sec_per_tick,
                    end=float(end_pos) * sec_per_tick,
                )
            )
    midi.instruments.append(inst)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))
    return {
        "bpm": tempo,
        "time_signature": meta.get("time_signature"),
        "time_signature_idx": int(meta.get("time_signature_idx", -1)),
        "num_measures": int(meta["num_measures"]),
        "melody_notes": len(inst.notes),
        "total_ticks": int(melody_pr.shape[2]),
    }


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(event.get("tick", 0)),
        0 if str(event.get("type", "")) == "note_off" else 1,
        int(event.get("pitch", 0)),
    )


def normalize_events(events: list[dict[str, Any]], *, include_velocity: bool) -> list[dict[str, int | str]]:
    normalized = []
    for event in events:
        item: dict[str, int | str] = {
            "type": str(event["type"]),
            "pitch": int(event["pitch"]),
            "tick": int(event["tick"]),
        }
        if include_velocity and "velocity" in event and event["velocity"] is not None:
            item["velocity"] = int(event["velocity"])
        normalized.append(item)
    normalized.sort(key=event_sort_key)
    return normalized


def beats_to_events(tokenizer: Any, acc_beats: list[Any], prompt_ticks: int) -> list[dict[str, int | str]]:
    pr = tokenizer.decode_beats_to_pianoroll(acc_beats, track_marker_id=tokenizer.vocab.track_marker_acc)
    if pr.shape[2] > prompt_ticks:
        pr = pr[:, :, :prompt_ticks]
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
    return normalize_events(events, include_velocity=False)


def sha_events(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_rt_prompt_reference(args: argparse.Namespace, npz_path: Path, out_dir: Path) -> dict[str, Any]:
    rt = load_rt_prompt_components(args.rt_root)
    config = rt.ModelConfig()
    tokenizer = rt.PianoMusicTokenizer(config=config)
    dataset = SimpleNamespace(root_dir=str(args.data_dir), data_files=[npz_path.name], tokenizer=tokenizer)
    data = np.load(npz_path, allow_pickle=True)
    inferred_beats_per_bar = int(data["measure_0"].shape[2] // 4)
    beats_per_bar = int(args.beats_per_bar or inferred_beats_per_bar)
    prompt_bars = int(args.prompt_beats / beats_per_bar)
    if prompt_bars * beats_per_bar != args.prompt_beats:
        raise ValueError("prompt_beats must be divisible by beats_per_bar for RT prepare_condition")
    prep = rt.prepare_condition(dataset, 0, num_bars=prompt_bars)
    if prep is None:
        raise RuntimeError(f"RT prepare_condition returned None for {npz_path}")

    device = resolve_device(args.device)
    model = rt.load_model(
        model_path=str(args.prompt_checkpoint),
        model_config=config,
        device=device,
        use_fp16=bool(args.fp16),
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generated = model.generate_music(
        initial_tokens=prep["prompt_tokens"],
        device=device,
        max_length=len(prep["prompt_tokens"]) + args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    _mel, acc_beats = tokenizer.parse_generated_sequence(generated.squeeze(0))
    acc_beats = acc_beats[: args.prompt_beats]
    events = beats_to_events(tokenizer, acc_beats, prompt_ticks=args.prompt_beats * 4)

    ref = {
        "npz_id": npz_path.stem,
        "prompt_beats": args.prompt_beats,
        "beats_per_bar": beats_per_bar,
        "prompt_bars": prompt_bars,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "prompt_token_count": int(len(prep["prompt_tokens"])),
        "generated_token_count": int(generated.numel()),
        "acc_beats_count": int(len(acc_beats)),
        "events_count": int(len(events)),
        "events_sha256": sha_events(events),
        "events": events,
    }
    (out_dir / "rt_prompt_reference.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
    torch.save(prep["prompt_tokens"].detach().cpu(), out_dir / "rt_prompt_tokens.pt")
    torch.save(generated.detach().cpu(), out_dir / "rt_prompt_generated_tokens.pt")
    return ref


def compare_cli_to_reference(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    ref_path = out_dir / "rt_prompt_reference.json"
    cli_path = Path(args.cli_prompt_json)
    ref = json.loads(ref_path.read_text())
    cli_raw = json.loads(cli_path.read_text())
    cli_events = normalize_events(cli_raw, include_velocity=False)
    ref_events = normalize_events(ref["events"], include_velocity=False)
    first_diff = None
    for idx, (a, b) in enumerate(zip(ref_events, cli_events)):
        if a != b:
            first_diff = {"index": idx, "reference": a, "cli": b}
            break
    if first_diff is None and len(ref_events) != len(cli_events):
        first_diff = {
            "index": min(len(ref_events), len(cli_events)),
            "reference_remaining": ref_events[min(len(ref_events), len(cli_events)):][:5],
            "cli_remaining": cli_events[min(len(ref_events), len(cli_events)):][:5],
        }
    result = {
        "reference_path": str(ref_path),
        "cli_prompt_json": str(cli_path),
        "reference_event_count": len(ref_events),
        "cli_event_count": len(cli_events),
        "reference_sha256": sha_events(ref_events),
        "cli_sha256": sha_events(cli_events),
        "events_equal_ignoring_velocity": ref_events == cli_events,
        "first_diff": first_diff,
    }
    (out_dir / "prompt_alignment_compare.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-id", default="6217163")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--rt-root", type=Path, default=RT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--prompt-checkpoint", type=Path, default=PROMPT_CKPT)
    parser.add_argument("--prompt-beats", type=int, default=8)
    parser.add_argument("--beats-per-bar", type=int, default=0, help="0 means infer from NPZ measure width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--cli-prompt-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    npz_path = args.data_dir / f"{Path(args.npz_id).stem}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    out_dir = args.output_root / npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        meta = export_npz_melody_midi(npz_path, out_dir / f"{npz_path.stem}_npz_melody_input.mid")
        (out_dir / "input_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps({"melody_midi": str(out_dir / f"{npz_path.stem}_npz_melody_input.mid"), **meta}, indent=2))
    if args.reference:
        ref = generate_rt_prompt_reference(args, npz_path, out_dir)
        print(json.dumps({k: v for k, v in ref.items() if k != "events"}, indent=2))
    if args.compare:
        if not args.cli_prompt_json:
            raise ValueError("--cli-prompt-json is required with --compare")
        result = compare_cli_to_reference(args, out_dir)
        return 0 if result["events_equal_ignoring_velocity"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

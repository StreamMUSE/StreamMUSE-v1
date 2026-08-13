from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.inference.lekai_prompt_continuation import (
    LekaiPromptContinuationBackend,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    event_representation_summary,
)
from streammuse.infrastructure.inference.serialization import event_from_dict
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink

TIMESTEPS_PER_BEAT = 4
DEFAULT_MODELS_DIR = Path.home() / "mbzuai-projects" / "models"
DEFAULT_PROMPT_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_prompt_model" / "model.safetensors"
DEFAULT_CONTINUATION_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_continuation_model" / "model.safetensors"

EventPayload = dict[str, int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline two-stage Lekai prompt+continuation inference and save MIDI/logs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--midi-file", type=str, help="Single melody MIDI file")
    input_group.add_argument("--midi-dir", type=str, help="Directory of melody MIDI files")
    input_group.add_argument("--npz-file", type=str, help="Single NPZ file; corresponding MIDI is resolved from mel/")
    input_group.add_argument("--npz-dir", type=str, help="Directory of NPZ files; corresponding MIDI files are resolved from mel/")

    parser.add_argument("--output-dir", type=str, required=True, help="Directory for generated MIDI/log artifacts")
    parser.add_argument("--prompt-checkpoint", type=str, default=str(DEFAULT_PROMPT_CHECKPOINT))
    parser.add_argument("--continuation-checkpoint", type=str, default=str(DEFAULT_CONTINUATION_CHECKPOINT))
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda or cuda:<index>")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16"])
    parser.add_argument("--prompt-dtype", type=str, default=None, choices=["auto", "float32", "float16"])
    parser.add_argument("--rt-dtype", type=str, default=None, choices=["auto", "float32", "float16"])
    parser.add_argument("--prompt-fp16", action="store_true", help="Alias for --prompt-dtype float16")
    parser.add_argument("--rt-fp16", action="store_true", help="Alias for --rt-dtype float16")
    parser.add_argument("--require-real-models", action="store_true", default=True)
    parser.add_argument("--allow-fallback", action="store_true", help="Allow rule/stub fallback instead of failing")

    parser.add_argument("--prompt-length-ticks", type=int, default=32)
    parser.add_argument("--generation-interval-ticks", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--ticks-per-beat", type=int, default=4)
    parser.add_argument("--max-tick", type=int, default=None)
    parser.add_argument("--trim-leading-rest", action="store_true")

    parser.add_argument("--prompt-seed", type=int, default=12345)
    parser.add_argument(
        "--prompt-selection-mode",
        choices=("single", "batch_first", "rule_s"),
        default="single",
    )
    parser.add_argument("--prompt-batch-candidates", type=int, default=5)
    parser.add_argument("--prompt-temperature", type=float, default=1.1)
    parser.add_argument("--prompt-top-k", type=int, default=0)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--rt-temperature", type=float, default=0.0)
    parser.add_argument("--rt-top-k", type=int, default=1)
    parser.add_argument("--rt-top-p", type=float, default=0.0)
    parser.add_argument("--rt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--rt-seed", type=int, default=0)

    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def require_path(path: Path, label: str) -> Path:
    expanded = path.expanduser().resolve()
    if not expanded.exists():
        raise FileNotFoundError(f"{label} not found: {expanded}")
    return expanded


def resolve_npz_to_midi(npz_path: Path) -> Path:
    npz_path = npz_path.expanduser().resolve()
    stem = npz_path.stem
    candidates: list[Path] = []
    if npz_path.parent.name == "npz":
        candidates.append(npz_path.parent.parent / "mel" / f"{stem}.mid")
    candidates.append(npz_path.with_suffix(".mid"))
    candidates.append(Path.cwd() / "prompts" / "inputs_lekai" / "mel" / f"{stem}.mid")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve melody MIDI for NPZ {npz_path}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    if args.midi_file:
        return [require_path(Path(args.midi_file), "midi_file")]
    if args.midi_dir:
        midi_dir = require_path(Path(args.midi_dir), "midi_dir")
        files = sorted(midi_dir.glob("*.mid"))
        if not files:
            raise FileNotFoundError(f"No .mid files found under {midi_dir}")
        return [p.resolve() for p in files]
    if args.npz_file:
        return [resolve_npz_to_midi(Path(args.npz_file))]
    npz_dir = require_path(Path(args.npz_dir), "npz_dir")
    npz_files = sorted(npz_dir.glob("*.npz"), key=lambda p: (len(p.stem), p.stem))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir}")
    return [resolve_npz_to_midi(p) for p in npz_files]


def apply_runtime_env(args: argparse.Namespace, *, stage2_log_dir: Path) -> dict[str, str | None]:
    updates: dict[str, str] = {}
    device = str(args.device)
    if device.startswith("cuda:"):
        updates["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        device = "cuda"
    updates["LEKAI_DEVICE"] = device
    updates["LEKAI_PROMPT_DEVICE"] = device

    prompt_dtype = "float16" if args.prompt_fp16 else (args.prompt_dtype or args.dtype)
    rt_dtype = "float16" if args.rt_fp16 else (args.rt_dtype or args.dtype)
    updates["LEKAI_PROMPT_DTYPE"] = str(prompt_dtype)
    updates["LEKAI_DTYPE"] = str(rt_dtype)

    updates["LEKAI_PROMPT_SEED"] = str(int(args.prompt_seed))
    updates["LEKAI_PROMPT_SELECTION_MODE"] = str(args.prompt_selection_mode)
    updates["LEKAI_PROMPT_BATCH_CANDIDATES"] = str(
        int(args.prompt_batch_candidates)
    )
    updates["LEKAI_PROMPT_BPM"] = str(int(args.bpm))
    updates["LEKAI_DEFAULT_BPM"] = str(int(args.bpm))
    updates["LEKAI_PROMPT_TEMPERATURE"] = str(float(args.prompt_temperature))
    updates["LEKAI_PROMPT_TOP_K"] = str(int(args.prompt_top_k))
    updates["LEKAI_PROMPT_TOP_P"] = str(float(args.prompt_top_p))
    updates["LEKAI_PROMPT_REPETITION_PENALTY"] = str(float(args.prompt_repetition_penalty))
    updates["LEKAI_RT_TEMPERATURE"] = str(float(args.rt_temperature))
    updates["LEKAI_RT_TOP_K"] = str(int(args.rt_top_k))
    updates["LEKAI_RT_TOP_P"] = str(float(args.rt_top_p))
    updates["LEKAI_RT_REPETITION_PENALTY"] = str(float(args.rt_repetition_penalty))
    updates["LEKAI_RT_SEED"] = str(int(args.rt_seed))
    updates["LEKAI_PROMPT_CONTINUATION_ENGINE"] = "standard"
    updates["LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS"] = "0"
    updates["LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES"] = "0"
    updates["LEKAI_RT_LOG_DIR"] = str(stage2_log_dir)
    if args.require_real_models and not args.allow_fallback:
        updates["LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS"] = "1"
    elif args.allow_fallback:
        updates["LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS"] = "0"

    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def load_midi_events(
    midi_path: Path,
    *,
    ticks_per_beat: int,
    max_tick: int | None,
    trim_leading_rest: bool,
    velocity: int = 80,
) -> tuple[list[EventPayload], dict[str, Any]]:
    notes, resolution, original_max_tick = MidiFileInput._midi_to_notes(
        str(midi_path),
        beat_div=int(ticks_per_beat),
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=None,
    )
    first_tick = min((int(note["tick"]) for note in notes), default=0)
    offset = first_tick if trim_leading_rest else 0
    relative_max_tick = max(0, int(original_max_tick) - offset)
    if max_tick is not None:
        notes = [
            note
            for note in notes
            if int(note["tick"]) - offset < int(max_tick)
        ]
        relative_max_tick = min(relative_max_tick, int(max_tick))
    events: list[EventPayload] = []
    for note in notes:
        tick = int(note["tick"]) - offset
        if tick < 0:
            continue
        pitch = int(note["pitch"])
        duration = max(1, int(note["duration"]))
        events.append({"type": "note_on", "pitch": pitch, "tick": tick, "velocity": int(velocity)})
        note_off_tick = tick + duration
        if max_tick is None or note_off_tick < int(max_tick):
            events.append({"type": "note_off", "pitch": pitch, "tick": note_off_tick, "velocity": 0})
    events.sort(key=event_sort_key)
    return events, {
        "midi_path": str(midi_path),
        "note_count": len(notes),
        "event_count": len(events),
        "resolution": int(resolution),
        "original_max_tick": int(original_max_tick),
        "actual_max_tick": int(relative_max_tick),
        "first_note_tick_original": int(first_tick),
        "trim_leading_rest": bool(trim_leading_rest),
        "offset_ticks": int(offset),
    }


def event_sort_key(event: EventPayload) -> tuple[int, int, int]:
    return (
        int(event.get("tick", 0)),
        0 if str(event.get("type", "")) == "note_off" else 1,
        int(event.get("pitch", -1)),
    )


def events_until(events: list[EventPayload], start_tick: int, end_tick: int) -> list[EventPayload]:
    return [
        dict(event)
        for event in events
        if int(start_tick) <= int(event.get("tick", 0)) < int(end_tick)
    ]


def payload_to_musical_event(event: EventPayload) -> MusicalEvent:
    return event_from_dict(dict(event))


def write_events_midi(
    path: Path,
    *,
    melody_events: list[EventPayload] | None,
    accompaniment_events: list[EventPayload],
    bpm: int,
    ticks_per_beat: int,
) -> None:
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            output_path=str(path),
        )
    )
    combined: list[tuple[str, EventPayload]] = []
    for event in melody_events or []:
        combined.append(("user", event))
    for event in accompaniment_events:
        combined.append(("model", event))
    combined.sort(key=lambda item: event_sort_key(item[1]))
    for source, event in combined:
        sink.output_event(payload_to_musical_event(event), source=source)
    sink.close()


def wait_for_scheduler(
    backend: LekaiPromptContinuationBackend,
    *,
    timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    deadline = start + float(timeout_s)
    samples: list[dict[str, Any]] = []
    last_marker: tuple[Any, Any, Any] | None = None
    while True:
        status = dict(backend.scheduler_status())
        marker = (status.get("phase"), status.get("continuation_calls"), status.get("accompaniment_event_count"))
        if marker != last_marker:
            samples.append({"elapsed_s": round(time.perf_counter() - start, 3), **status})
            last_marker = marker
        if status.get("is_failed"):
            raise RuntimeError(f"offline scheduler failed: {status.get('error')}")
        if status.get("is_playback_ready"):
            if int(status.get("accompaniment_history_beats", 0) or 0) >= int(
                status.get("target_playable_accompaniment_beats", 0) or 0
            ):
                return status, samples
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"offline scheduler not ready in {timeout_s}s: {status}")
        time.sleep(0.05)


def backend_prompt_token_log(backend: LekaiPromptContinuationBackend) -> dict[str, Any]:
    engine = getattr(backend, "_engine", None)
    prompt_engine = getattr(engine, "_prompt_engine", None)
    if prompt_engine is None or not hasattr(prompt_engine, "last_generation_log"):
        return {"available": False, "reason": "prompt_engine_log_not_available"}
    log = dict(prompt_engine.last_generation_log())
    log["available"] = True
    return log


def run_one(
    midi_path: Path,
    *,
    index: int,
    args: argparse.Namespace,
    prompt_checkpoint: Path,
    continuation_checkpoint: Path,
    output_root: Path,
) -> dict[str, Any]:
    song_dir = output_root / f"{index:03d}_{midi_path.stem}"
    song_dir.mkdir(parents=True, exist_ok=True)
    stage2_log_dir = song_dir / "stage2_token_logs"
    stage2_log_dir.mkdir(parents=True, exist_ok=True)

    previous_env = apply_runtime_env(args, stage2_log_dir=stage2_log_dir)
    start_time = time.perf_counter()
    backend: LekaiPromptContinuationBackend | None = None
    try:
        melody_events, midi_info = load_midi_events(
            midi_path,
            ticks_per_beat=int(args.ticks_per_beat),
            max_tick=args.max_tick,
            trim_leading_rest=bool(args.trim_leading_rest),
        )
        melody_end_tick = max([int(event.get("tick", 0)) for event in melody_events] or [0])
        if args.max_tick is not None:
            final_observed_until_tick = max(
                int(args.prompt_length_ticks),
                int(args.max_tick),
            )
        else:
            final_observed_until_tick = max(
                int(args.prompt_length_ticks),
                (
                    (int(melody_end_tick) + TIMESTEPS_PER_BEAT - 1)
                    // TIMESTEPS_PER_BEAT
                )
                * TIMESTEPS_PER_BEAT,
            )
        prompt_events = events_until(melody_events, 0, int(args.prompt_length_ticks))
        append_events = events_until(melody_events, int(args.prompt_length_ticks), final_observed_until_tick)

        backend = LekaiPromptContinuationBackend(
            prompt_checkpoint_path=str(prompt_checkpoint),
            continuation_checkpoint_path=str(continuation_checkpoint),
        )
        runtime_info = dict(backend.runtime_info())
        start_status = backend.start_prompt_catchup(
            melody_events=prompt_events,
            prompt_length_ticks=int(args.prompt_length_ticks),
            generation_interval_ticks=int(args.generation_interval_ticks),
            inference_mode="sliding_window",
            model_name="lekai_prompt_continuation",
            checkpoint_path=None,
            observed_until_tick=int(args.prompt_length_ticks),
        )
        append_status = backend.append_melody_events(
            append_events,
            observed_until_tick=final_observed_until_tick,
        )
        final_status, scheduler_samples = wait_for_scheduler(backend, timeout_s=float(args.timeout_s))

        prompt_history = backend.prompt_accompaniment_history()
        raw_history = backend.raw_accompaniment_history()
        playable = backend.playable_accompaniment()
        prompt_token_log = backend_prompt_token_log(backend)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        final_midi = song_dir / "final.mid"
        prompt_midi = song_dir / "stage1_prompt.mid"
        raw_midi = song_dir / "stage2_raw_history.mid"
        write_events_midi(
            final_midi,
            melody_events=melody_events,
            accompaniment_events=raw_history,
            bpm=int(args.bpm),
            ticks_per_beat=int(args.ticks_per_beat),
        )
        write_events_midi(
            prompt_midi,
            melody_events=prompt_events,
            accompaniment_events=prompt_history,
            bpm=int(args.bpm),
            ticks_per_beat=int(args.ticks_per_beat),
        )
        write_events_midi(
            raw_midi,
            melody_events=None,
            accompaniment_events=raw_history,
            bpm=int(args.bpm),
            ticks_per_beat=int(args.ticks_per_beat),
        )

        write_json(song_dir / "melody_events.json", melody_events)
        write_json(song_dir / "stage1_prompt_history.json", prompt_history)
        write_json(song_dir / "stage1_token_log.json", prompt_token_log)
        write_json(song_dir / "stage2_raw_history.json", raw_history)
        write_json(song_dir / "stage2_playable_history.json", playable)
        write_json(song_dir / "scheduler_status_log.json", scheduler_samples)
        stage2_token_logs = sorted(str(path.relative_to(song_dir)) for path in stage2_log_dir.glob("*.json"))

        summary = {
            "input_midi": str(midi_path),
            "output_dir": str(song_dir),
            "final_midi": str(final_midi),
            "stage1_prompt_midi": str(prompt_midi),
            "stage2_raw_midi": str(raw_midi),
            "prompt_checkpoint": str(prompt_checkpoint),
            "continuation_checkpoint": str(continuation_checkpoint),
            "config": {
                "prompt_length_ticks": int(args.prompt_length_ticks),
                "generation_interval_ticks": int(args.generation_interval_ticks),
                "bpm": int(args.bpm),
                "ticks_per_beat": int(args.ticks_per_beat),
                "device": str(args.device),
                "dtype": str(args.dtype),
                "prompt_dtype": str(args.prompt_dtype or ("float16" if args.prompt_fp16 else args.dtype)),
                "rt_dtype": str(args.rt_dtype or ("float16" if args.rt_fp16 else args.dtype)),
                "prompt_seed": int(args.prompt_seed),
                "prompt_selection_mode": str(args.prompt_selection_mode),
                "prompt_batch_candidates": int(args.prompt_batch_candidates),
                "prompt_temperature": float(args.prompt_temperature),
                "prompt_top_k": int(args.prompt_top_k),
                "prompt_top_p": float(args.prompt_top_p),
                "prompt_repetition_penalty": float(args.prompt_repetition_penalty),
                "rt_temperature": float(args.rt_temperature),
                "rt_top_k": int(args.rt_top_k),
                "rt_top_p": float(args.rt_top_p),
                "rt_repetition_penalty": float(args.rt_repetition_penalty),
                "rt_seed": int(args.rt_seed),
            },
            "midi_info": midi_info,
            "observed_until_tick": int(final_observed_until_tick),
            "melody_event_count": len(melody_events),
            "prompt_melody_event_count": len(prompt_events),
            "append_melody_event_count": len(append_events),
            "prompt_history_event_count": len(prompt_history),
            "raw_history_event_count": len(raw_history),
            "playable_event_count": len(playable),
            "runtime_info": runtime_info,
            "start_status": start_status,
            "append_status": append_status,
            "final_status": final_status,
            "elapsed_ms": elapsed_ms,
            "stage1_representation": event_representation_summary(prompt_history),
            "stage2_representation": event_representation_summary(raw_history),
            "stage2_token_log_files": stage2_token_logs,
        }
        write_json(song_dir / "summary.json", summary)
        return summary
    finally:
        if backend is not None:
            try:
                backend.clear_history()
            except Exception:
                pass
        restore_env(previous_env)


def main() -> None:
    args = parse_args()
    if args.prompt_selection_mode != "single" and args.prompt_batch_candidates < 2:
        raise ValueError("--prompt-batch-candidates must be at least 2")
    prompt_checkpoint = require_path(Path(args.prompt_checkpoint), "prompt_checkpoint")
    continuation_checkpoint = require_path(Path(args.continuation_checkpoint), "continuation_checkpoint")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    midi_files = collect_inputs(args)

    summaries: list[dict[str, Any]] = []
    for index, midi_path in enumerate(midi_files, start=1):
        print(f"[two-stage-offline] ({index}/{len(midi_files)}) {midi_path}")
        summary = run_one(
            midi_path,
            index=index,
            args=args,
            prompt_checkpoint=prompt_checkpoint,
            continuation_checkpoint=continuation_checkpoint,
            output_root=output_root,
        )
        summaries.append(summary)
        print(
            "[two-stage-offline] done "
            f"events={summary['raw_history_event_count']} final={summary['final_midi']} "
            f"elapsed_ms={summary['elapsed_ms']:.1f}"
        )

    batch_summary = {
        "input_count": len(midi_files),
        "output_dir": str(output_root),
        "summaries": summaries,
    }
    write_json(output_root / "batch_summary.json", batch_summary)
    print(f"[two-stage-offline] batch summary: {output_root / 'batch_summary.json'}")


if __name__ == "__main__":
    main()

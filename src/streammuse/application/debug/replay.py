"""Replay runner for debugger traces."""

from __future__ import annotations

import json
import os
import hashlib
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from streammuse.application.debug.compare import compare_trace_events, load_trace
from streammuse.domain.debug.canonical import (
    canonical_event_payloads,
    hash_jsonable,
    summarize_events,
)
from streammuse.domain.debug.trace import ArtifactRef, DebugTraceEvent
from streammuse.infrastructure.debug.trace_recorder import JsonlDebugTraceRecorder
from streammuse.infrastructure.inference.lekai_prompt_continuation import (
    LekaiPromptContinuationBackend,
)
from streammuse.infrastructure.inference.serialization import event_from_dict
from streammuse.infrastructure.input.midi_file import MidiFileInput
from streammuse.infrastructure.output.midi_file import MidiFileOutputConfig, MidiFileOutputSink


EventPayload = dict[str, Any]
SCENARIO_LEKAI_PROMPT_CONTINUATION = "lekai-prompt-continuation"


@dataclass(frozen=True)
class ReplayConfig:
    scenario: str
    midi_file: str
    output_dir: str
    compare: tuple[str, ...] = ("offline", "sim")
    prompt_checkpoint: str | None = None
    continuation_checkpoint: str | None = None
    device: str = "auto"
    prompt_length_ticks: int = 32
    generation_interval_ticks: int = 4
    timeout_s: float = 120.0
    bpm: int = 120
    ticks_per_beat: int = 4
    max_tick: int | None = None
    trim_leading_rest: bool = False


@contextmanager
def _runtime_env(config: ReplayConfig, *, stage2_log_dir: Path) -> Iterator[None]:
    updates = {
        "LEKAI_DEVICE": config.device,
        "LEKAI_PROMPT_DEVICE": config.device,
        "LEKAI_PROMPT_CONTINUATION_ENGINE": "standard",
        "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
        "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "0",
        "LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS": "0",
        "LEKAI_RT_LOG_DIR": str(stage2_log_dir),
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_replay(config: ReplayConfig) -> dict[str, Any]:
    if config.scenario != SCENARIO_LEKAI_PROMPT_CONTINUATION:
        raise ValueError(f"unsupported debug scenario: {config.scenario}")

    root = Path(config.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / f"replay_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scenario": config.scenario,
        "midi_file": str(Path(config.midi_file).expanduser()),
        "runs": {},
    }
    requested = _normalize_compare(config.compare)
    if "offline" in requested:
        manifest["runs"]["offline_direct"] = _run_prompt_continuation(
            config,
            runner_kind="offline_direct",
            run_dir=run_root / "offline_direct",
        )
    if "sim" in requested:
        manifest["runs"]["realtime_sim"] = _run_prompt_continuation(
            config,
            runner_kind="realtime_sim",
            run_dir=run_root / "realtime_sim",
        )

    comparison_path = None
    if {"offline_direct", "realtime_sim"}.issubset(manifest["runs"]):
        left_trace = load_trace(run_root / "offline_direct" / "trace.jsonl")
        right_trace = load_trace(run_root / "realtime_sim" / "trace.jsonl")
        comparison = compare_trace_events(left_trace, right_trace)
        comparison_path = run_root / "comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        manifest["comparison"] = "comparison.json"

    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": str(run_root),
        "manifest_path": str(run_root / "manifest.json"),
        "comparison_path": str(comparison_path) if comparison_path else None,
    }


def _normalize_compare(compare: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for item in compare:
        for part in str(item).split(","):
            normalized = part.strip().lower()
            if normalized in {"offline_direct", "offline"}:
                values.add("offline")
            elif normalized in {"realtime_sim", "sim"}:
                values.add("sim")
            elif normalized:
                raise ValueError(f"unsupported compare runner: {part}")
    return values or {"offline", "sim"}


def _run_prompt_continuation(
    config: ReplayConfig,
    *,
    runner_kind: str,
    run_dir: Path,
) -> dict[str, Any]:
    stage2_log_dir = run_dir / "artifacts" / "stage2_token_logs"
    stage2_log_dir.mkdir(parents=True, exist_ok=True)
    recorder = JsonlDebugTraceRecorder(
        root_dir=run_dir,
        run_id=f"{runner_kind}-{uuid.uuid4().hex[:8]}",
        runner_kind=runner_kind,
        scenario=config.scenario,
    )
    backend: LekaiPromptContinuationBackend | None = None
    try:
        with _runtime_env(config, stage2_log_dir=stage2_log_dir):
            melody_events, midi_info = _load_midi_events(config)
            _record_payload(recorder, "input_midi_loaded", midi_info)
            _record_events(recorder, "normalized_melody_events", melody_events)

            melody_end_tick = max([int(event.get("tick", 0)) for event in melody_events] or [0])
            final_observed_until_tick = max(
                int(config.prompt_length_ticks),
                _ceil_to_interval(melody_end_tick, int(config.generation_interval_ticks)),
            )
            prompt_events = _events_until(melody_events, 0, int(config.prompt_length_ticks))
            append_events = _events_until(
                melody_events,
                int(config.prompt_length_ticks),
                final_observed_until_tick,
            )
            _record_events(recorder, "prompt_window_selected", prompt_events)
            _record_events(recorder, "append_chunks_selected", append_events)

            backend = LekaiPromptContinuationBackend(
                prompt_checkpoint_path=config.prompt_checkpoint,
                continuation_checkpoint_path=config.continuation_checkpoint,
            )
            runtime_info = dict(backend.runtime_info())
            _record_payload(recorder, "continuation_context_built", runtime_info)

            start_status = backend.start_prompt_catchup(
                melody_events=prompt_events,
                prompt_length_ticks=int(config.prompt_length_ticks),
                generation_interval_ticks=int(config.generation_interval_ticks),
                inference_mode="sliding_window",
                model_name="lekai_prompt_continuation",
                checkpoint_path=None,
                observed_until_tick=int(config.prompt_length_ticks),
            )
            _record_payload(
                recorder,
                "prompt_start_submitted",
                start_status,
                logical_tick=int(config.prompt_length_ticks),
            )

            if runner_kind == "realtime_sim":
                _append_in_chunks(
                    backend,
                    recorder,
                    append_events,
                    start_tick=int(config.prompt_length_ticks),
                    final_tick=final_observed_until_tick,
                    interval_ticks=int(config.generation_interval_ticks),
                )
            else:
                append_status = backend.append_melody_events(
                    append_events,
                    observed_until_tick=final_observed_until_tick,
                )
                _record_payload(
                    recorder,
                    "scheduler_status",
                    append_status,
                    logical_tick=final_observed_until_tick,
                )

            final_status = _wait_for_ready(backend, timeout_s=float(config.timeout_s))
            prompt_history = backend.prompt_accompaniment_history()
            raw_history = backend.raw_accompaniment_history()
            playable = backend.playable_accompaniment()

            prompt_log = _backend_prompt_token_log(backend)
            _record_payload(recorder, "prompt_tokenization", prompt_log.get("prompt_token_ids", []))
            _record_payload(recorder, "prompt_model_generate", prompt_log)
            _record_events(recorder, "prompt_decode_events", prompt_history)

            stage2_logs = _load_stage2_token_logs(stage2_log_dir)
            _record_payload(recorder, "continuation_tokenization", stage2_logs)
            _record_payload(recorder, "continuation_model_generate", stage2_logs)
            _record_events(recorder, "continuation_decode_events", raw_history)
            _record_payload(recorder, "scheduler_status", final_status)
            _record_events(recorder, "playable_history", playable)

            audible = _audible_events_from_playable(playable)
            _record_events(recorder, "audible_scheduling", audible)
            midi_ref = _write_events_midi(
                run_dir / "artifacts" / "midi" / "output_midi_render.mid",
                melody_events=melody_events,
                accompaniment_events=raw_history,
                bpm=int(config.bpm),
                ticks_per_beat=int(config.ticks_per_beat),
            )
            recorder.record(
                DebugTraceEvent(
                    run_id=recorder.run_id,
                    runner_kind=runner_kind,
                    scenario=config.scenario,
                    stage="output_midi_render",
                    output_refs=[midi_ref],
                    output_hash=midi_ref.hash,
                    summary={"path": midi_ref.path},
                )
            )

            run_summary = {
                "runner_kind": runner_kind,
                "trace_path": "trace.jsonl",
                "manifest_path": "manifest.json",
                "prompt_history_event_count": len(prompt_history),
                "raw_history_event_count": len(raw_history),
                "playable_event_count": len(playable),
                "final_status": final_status,
            }
            (run_dir / "run_summary.json").write_text(
                json.dumps(run_summary, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            return run_summary
    except Exception as exc:
        recorder.record(
            DebugTraceEvent(
                run_id=recorder.run_id,
                runner_kind=runner_kind,
                scenario=config.scenario,
                stage="run_error",
                status="error",
                summary={"error": str(exc)},
            )
        )
        raise
    finally:
        if backend is not None:
            try:
                backend.clear_history()
            except Exception:
                pass
        recorder.close()


def _record_payload(
    recorder: JsonlDebugTraceRecorder,
    stage: str,
    payload: Any,
    *,
    logical_tick: int | None = None,
) -> ArtifactRef:
    ref = recorder.artifact("summary", payload, name_hint=stage)
    recorder.record(
        DebugTraceEvent(
            run_id=recorder.run_id,
            runner_kind=recorder.runner_kind,
            scenario=recorder.scenario,
            stage=stage,
            logical_tick=logical_tick,
            output_refs=[ref],
            output_hash=ref.hash,
            summary={"kind": type(payload).__name__},
        )
    )
    return ref


def _record_events(
    recorder: JsonlDebugTraceRecorder,
    stage: str,
    events: list[EventPayload],
) -> ArtifactRef:
    canonical = canonical_event_payloads(events)
    summary = summarize_events(canonical)
    ref = recorder.artifact("events", canonical, name_hint=stage)
    recorder.record(
        DebugTraceEvent(
            run_id=recorder.run_id,
            runner_kind=recorder.runner_kind,
            scenario=recorder.scenario,
            stage=stage,
            output_refs=[ref],
            output_hash=hash_jsonable(canonical),
            summary=summary,
        )
    )
    return ref


def _load_midi_events(config: ReplayConfig) -> tuple[list[EventPayload], dict[str, Any]]:
    notes, resolution, actual_max_tick = MidiFileInput._midi_to_notes(
        str(Path(config.midi_file).expanduser()),
        beat_div=int(config.ticks_per_beat),
        min_pitch=0,
        max_pitch=127,
        program=None,
        max_tick=config.max_tick,
    )
    first_tick = min((int(note["tick"]) for note in notes), default=0)
    offset = first_tick if config.trim_leading_rest else 0
    events: list[EventPayload] = []
    for note in notes:
        tick = int(note["tick"]) - offset
        if tick < 0:
            continue
        pitch = int(note["pitch"])
        duration = max(1, int(note["duration"]))
        events.append({"type": "note_on", "pitch": pitch, "tick": tick, "velocity": 80})
        events.append({"type": "note_off", "pitch": pitch, "tick": tick + duration, "velocity": 0})
    events.sort(key=_event_sort_key)
    return events, {
        "midi_path": str(Path(config.midi_file).expanduser()),
        "note_count": len(notes),
        "event_count": len(events),
        "resolution": int(resolution),
        "actual_max_tick": int(actual_max_tick),
        "first_note_tick_original": int(first_tick),
        "trim_leading_rest": bool(config.trim_leading_rest),
        "offset_ticks": int(offset),
    }


def _event_sort_key(event: EventPayload) -> tuple[int, int, int]:
    return (
        int(event.get("tick", 0)),
        0 if str(event.get("type", "")) == "note_off" else 1,
        int(event.get("pitch", -1)),
    )


def _events_until(events: list[EventPayload], start_tick: int, end_tick: int) -> list[EventPayload]:
    return [dict(event) for event in events if int(start_tick) <= int(event.get("tick", 0)) < int(end_tick)]


def _ceil_to_interval(tick: int, interval: int) -> int:
    interval = max(1, int(interval))
    return ((int(tick) + interval - 1) // interval) * interval


def _append_in_chunks(
    backend: LekaiPromptContinuationBackend,
    recorder: JsonlDebugTraceRecorder,
    append_events: list[EventPayload],
    *,
    start_tick: int,
    final_tick: int,
    interval_ticks: int,
) -> None:
    observed = int(start_tick)
    while observed < int(final_tick):
        next_tick = min(int(final_tick), observed + int(interval_ticks))
        chunk = _events_until(append_events, observed, next_tick)
        status = backend.append_melody_events(chunk, observed_until_tick=next_tick)
        _record_payload(recorder, "scheduler_status", status, logical_tick=next_tick)
        observed = next_tick


def _wait_for_ready(
    backend: LekaiPromptContinuationBackend,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + float(timeout_s)
    while True:
        status = dict(backend.wait_for_scheduler(timeout=timeout_s))
        if status.get("is_failed"):
            raise RuntimeError(f"prompt-continuation scheduler failed: {status.get('error')}")
        if status.get("is_playback_ready") or not status.get("is_running"):
            return status
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"prompt-continuation scheduler not ready in {timeout_s}s: {status}")
        time.sleep(0.05)


def _backend_prompt_token_log(backend: LekaiPromptContinuationBackend) -> dict[str, Any]:
    engine = getattr(backend, "_engine", None)
    prompt_engine = getattr(engine, "_prompt_engine", None)
    if prompt_engine is None or not hasattr(prompt_engine, "last_generation_log"):
        return {"available": False, "reason": "prompt_engine_log_not_available"}
    log = dict(prompt_engine.last_generation_log())
    log["available"] = True
    return log


def _load_stage2_token_logs(stage2_log_dir: Path) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for path in sorted(stage2_log_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["_path"] = str(path.relative_to(stage2_log_dir.parent.parent))
        logs.append(payload)
    return logs


def _audible_events_from_playable(playable: list[EventPayload]) -> list[EventPayload]:
    return [dict(event) for event in sorted(playable, key=_event_sort_key)]


def _write_events_midi(
    path: Path,
    *,
    melody_events: list[EventPayload],
    accompaniment_events: list[EventPayload],
    bpm: int,
    ticks_per_beat: int,
) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = MidiFileOutputSink(
        MidiFileOutputConfig(
            bpm=float(bpm),
            ticks_per_beat=int(ticks_per_beat),
            output_path=str(path),
        )
    )
    combined: list[tuple[str, EventPayload]] = []
    for event in melody_events:
        combined.append(("user", event))
    for event in accompaniment_events:
        combined.append(("model", event))
    for source, event in sorted(combined, key=lambda item: _event_sort_key(item[1])):
        sink.output_event(event_from_dict(dict(event)), source=source)
    sink.close()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(
        kind="midi",
        path=str(path.relative_to(path.parents[2])),
        hash=digest,
        summary={"size": path.stat().st_size},
    )

#!/usr/bin/env python3
"""Produce auditable offline rap lyrics and drum-backed vocal WAV songs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from typing import Any

import httpx
import numpy as np

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.audio_rendering import tick_frame_in_bar
from streammuse.application.rap.bar_renderer import DeterministicRapBarRenderer
from streammuse.application.rap.offline_song import (
    narrative_focus_for_bar,
    overused_closing_words,
    overused_opening_words,
    select_flow_with_fallback,
    template_id_for_bar,
    word_trigrams,
)
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.application.rap.scoring import rank_candidates
from streammuse.domain.rap import (
    AudioFormat,
    CandidateEvaluation,
    CandidateRequest,
    ScoreWeights,
    ScenarioSegment,
    materialize_flow,
)
from streammuse.domain.rap.prosody import normalize_text
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.audio_output import Float32WavAudioSink
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.generators import _build_messages, _parse_candidate_lines
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.speech import EspeakPhonemeSynthesizer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES
from streammuse.infrastructure.rap.time_stretch import RubberBandTimeStretcher


@dataclass(frozen=True)
class SongDefinition:
    index: int
    slug: str
    title: str
    topic: str


@dataclass(frozen=True)
class ParsedChoices:
    lines: tuple[str, ...]
    raw_choice_count: int
    finish_reasons: dict[str, int]
    prompt_tokens: int
    completion_tokens: int


DEFAULT_SONGS = (
    SongDefinition(1, "space_exploration", "Signals Beyond Earth", "space exploration"),
    SongDefinition(2, "deep_ocean", "Pressure Below", "deep ocean"),
    SongDefinition(3, "artificial_intelligence", "Learning Machines", "artificial intelligence"),
    SongDefinition(4, "city_nights", "Lights After Midnight", "city nights"),
    SongDefinition(5, "climate_resilience", "Stronger Than The Storm", "climate resilience"),
    SongDefinition(6, "human_memory", "Rooms Inside The Mind", "human memory"),
    SongDefinition(7, "quantum_physics", "Probability Steps", "quantum physics"),
    SongDefinition(8, "street_basketball", "Concrete Court", "street basketball"),
    SongDefinition(9, "renewable_energy", "Power From Tomorrow", "renewable energy"),
    SongDefinition(10, "future_music", "The Next Sound", "future music"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a resumable ten-song beat-aligned rap research album.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("lyrics", "audio", "all"), default="all")
    parser.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--model", default="qwen-rap")
    parser.add_argument("--bars", type=int, default=50)
    parser.add_argument("--tempo", type=float, default=90.0)
    parser.add_argument("--choices", type=int, default=64)
    parser.add_argument("--minimum-score", type=float, default=0.30)
    parser.add_argument("--minimum-stress", type=float, default=0.60)
    parser.add_argument("--target-pool", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--voice-speed", type=int, default=175)
    parser.add_argument("--voice-pitch", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.stage in ("audio", "all"):
        _require_audio_executables()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(args)

    if args.stage in ("lyrics", "all"):
        generate_lyrics(args)
    if args.stage in ("audio", "all"):
        render_audio(args)
    write_album_summary(args)
    write_artifact_index(args)
    return 0


def _require_audio_executables() -> None:
    for executable in ("espeak-ng", "rubberband"):
        if shutil.which(executable) is None:
            raise OSError(f"audio rendering requires the {executable} executable")


def parse_choice_payload(body: Mapping[str, Any]) -> ParsedChoices:
    raw_choices = body.get("choices", ())
    if not isinstance(raw_choices, Sequence):
        raise ValueError("chat response choices must be a sequence")
    ordered = sorted(raw_choices, key=lambda choice: choice.get("index", 0))
    lines: list[str] = []
    seen: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    for choice in ordered:
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reasons[str(choice.get("finish_reason") or "unknown")] += 1
        for line in _parse_candidate_lines(content, 1):
            key = normalize_text(line)
            if key and key not in seen:
                seen.add(key)
                lines.append(line)
    usage = body.get("usage") or {}
    return ParsedChoices(
        lines=tuple(lines),
        raw_choice_count=len(ordered),
        finish_reasons=dict(sorted(finish_reasons.items())),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def validate_chosen_records(records: Sequence[Mapping[str, Any]], *, expected_bars: int) -> None:
    if len(records) > expected_bars or [record.get("bar") for record in records] != list(range(len(records))):
        raise ValueError("chosen lyric records must contain contiguous bars starting at zero")


def expected_song_frames(*, bars: int, tempo_bpm: float, sample_rate_hz: int) -> int:
    tempo = Tempo(tempo_bpm, 4, 4)
    return round(tempo.tick_to_seconds(bars * tempo.ticks_per_bar) * sample_rate_hz)


def offline_messages(
    request: CandidateRequest,
    *,
    total_bars: int,
    blocked_openings: frozenset[str] = frozenset(),
    blocked_closings: frozenset[str] = frozenset(),
) -> tuple[dict[str, str], ...]:
    """Add lightweight song progression guidance to the validated flow prompt."""
    messages = [dict(message) for message in _build_messages(request)]
    focus = narrative_focus_for_bar(request.target_bar, total_bars)
    messages[-1]["content"] += (
        "\nContinue the song with a fresh image and action. Vary the opening words and sentence "
        "structure instead of echoing recent lines. "
        f"The current narrative focus is {focus}."
    )
    if blocked_openings:
        messages[-1]["content"] += (
            " Do not start with these overused words: " + ", ".join(sorted(blocked_openings)) + "."
        )
    if blocked_closings:
        messages[-1]["content"] += (
            " Do not end with these overused words: " + ", ".join(sorted(blocked_closings)) + "."
        )
    return tuple(messages)


def generate_lyrics(args: argparse.Namespace) -> None:
    analyzer = CmuProsodyAnalyzer()
    weights = ScoreWeights()
    timeout = httpx.Timeout(args.timeout_s)
    with httpx.Client(timeout=timeout) as client:
        for song in DEFAULT_SONGS:
            _generate_song_lyrics(args, song, client, analyzer, weights)


def _generate_song_lyrics(
    args: argparse.Namespace,
    song: SongDefinition,
    client: httpx.Client,
    analyzer: CmuProsodyAnalyzer,
    weights: ScoreWeights,
) -> None:
    directory = _song_directory(args.output_dir, song)
    directory.mkdir(parents=True, exist_ok=True)
    chosen_path = directory / "chosen_lyrics.jsonl"
    attempts_path = directory / "generation_attempts.jsonl"
    chosen = _read_jsonl(chosen_path)
    attempts = _read_jsonl(attempts_path)
    validate_chosen_records(chosen, expected_bars=args.bars)
    attempts_by_bar: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_bar[int(attempt["bar"])].append(attempt)

    history = [analyzer.analyze(str(record["text"])) for record in chosen]
    rhyme_anchors: dict[tuple[int, str], tuple[str, ...]] = {}
    for record, analysis in zip(chosen, history, strict=True):
        if analysis.end_rhyme_tail:
            rhyme_anchors.setdefault((int(record["rhyme_section_start"]), "A"), analysis.end_rhyme_tail)

    for bar in range(len(chosen), args.bars):
        selected, qualified, evaluations, bar_attempts = _generate_bar(
            args=args,
            song=song,
            bar=bar,
            prior_attempts=attempts_by_bar.get(bar, []),
            client=client,
            analyzer=analyzer,
            weights=weights,
            history=history,
            rhyme_anchors=rhyme_anchors,
            attempts_path=attempts_path,
        )
        record = _chosen_record(
            args=args,
            song=song,
            bar=bar,
            selected=selected,
            qualified=qualified,
            evaluations=evaluations,
            attempts=bar_attempts,
        )
        _append_jsonl(chosen_path, record)
        chosen.append(record)
        history.append(selected.analysis)
        if selected.analysis.end_rhyme_tail:
            rhyme_anchors.setdefault((bar // 4 * 4, "A"), selected.analysis.end_rhyme_tail)
        _write_song_lyric_artifacts(args, song, chosen, _read_jsonl(attempts_path))
        print(
            f"[lyrics][{song.index:02d}/10][bar {bar + 1:02d}/{args.bars}] "
            f"qualified={len(qualified):2d} score={selected.total_score:.3f} "
            f"stress={selected.component('stress_alignment').value:.3f} {selected.text}",
            flush=True,
        )

    validate_chosen_records(chosen, expected_bars=args.bars)
    if len(chosen) != args.bars:
        raise RuntimeError(f"song {song.index} contains {len(chosen)} chosen bars, expected {args.bars}")
    _write_song_lyric_artifacts(args, song, chosen, _read_jsonl(attempts_path))


def _generate_bar(
    *,
    args: argparse.Namespace,
    song: SongDefinition,
    bar: int,
    prior_attempts: Sequence[Mapping[str, Any]],
    client: httpx.Client,
    analyzer: CmuProsodyAnalyzer,
    weights: ScoreWeights,
    history: Sequence[Any],
    rhyme_anchors: Mapping[tuple[int, str], tuple[str, ...]],
    attempts_path: Path,
) -> tuple[CandidateEvaluation, tuple[CandidateEvaluation, ...], tuple[CandidateEvaluation, ...], list[dict[str, Any]]]:
    template = BUILTIN_TEMPLATES.get(template_id_for_bar(bar))
    request = CandidateRequest(
        request_id=f"album-song-{song.index:02d}-bar-{bar:03d}",
        target_bar=bar,
        topic=song.topic,
        flow_template=template,
        count=1,
        context_lines=tuple(item.text for item in history[-4:]),
        seed=args.seed + song.index * 100_000 + bar,
    )
    blocked_openings = overused_opening_words(
        (item.text for item in history),
        maximum_uses=2,
    )
    blocked_closings = overused_closing_words(
        (item.text for item in history),
        maximum_uses=2,
    )
    prompt = offline_messages(
        request,
        total_bars=args.bars,
        blocked_openings=blocked_openings,
        blocked_closings=blocked_closings,
    )
    bar_attempts = [dict(item) for item in prior_attempts]
    pooled_lines = _deduplicate(
        line
        for attempt in bar_attempts
        if not attempt.get("error")
        for line in attempt.get("candidates", ())
    )
    evaluations, qualified, selected = _evaluate_pool(
        args, song, bar, pooled_lines, analyzer, weights, history, rhyme_anchors
    )

    for attempt_number in range(len(bar_attempts) + 1, args.max_attempts + 1):
        if selected is not None and len(qualified) >= args.target_pool:
            break
        started = time.perf_counter()
        error: str | None = None
        parsed = ParsedChoices((), 0, {}, 0, 0)
        try:
            response = client.post(
                f"{args.base_url.rstrip('/')}/chat/completions",
                json={
                    "model": args.model,
                    "messages": [dict(message) for message in prompt],
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": 32,
                    "n": args.choices,
                },
            )
            response.raise_for_status()
            parsed = parse_choice_payload(response.json())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0
        pooled_lines = _deduplicate((*pooled_lines, *parsed.lines))
        evaluations, qualified, selected = _evaluate_pool(
            args, song, bar, pooled_lines, analyzer, weights, history, rhyme_anchors
        )
        attempt = {
            "song_index": song.index,
            "topic": song.topic,
            "bar": bar,
            "attempt": attempt_number,
            "request_id": request.request_id,
            "template_id": template.template_id,
            "context_lines": list(request.context_lines),
            "flow": _template_payload(template),
            "prompt": list(prompt),
            "requested_choices": args.choices,
            "raw_choice_count": parsed.raw_choice_count,
            "candidates": list(parsed.lines),
            "unique_pool_count": len(pooled_lines),
            "hard_valid_pool_count": sum(item.valid for item in evaluations),
            "flow_qualified_pool_count": len(qualified),
            "best_qualified_score": selected.total_score if selected else None,
            "latency_ms": latency_ms,
            "prompt_tokens": parsed.prompt_tokens,
            "completion_tokens": parsed.completion_tokens,
            "finish_reasons": parsed.finish_reasons,
            "error": error,
        }
        _append_jsonl(attempts_path, attempt)
        bar_attempts.append(attempt)
        if error:
            print(
                f"[lyrics][{song.index:02d}/10][bar {bar + 1:02d}] attempt={attempt_number} error={error}",
                flush=True,
            )
            time.sleep(min(5.0, attempt_number))

    if selected is None:
        raise RuntimeError(
            f"song {song.index} bar {bar} has no exact, score-qualified, stress-qualified candidate "
            f"after {len(bar_attempts)} attempts"
        )
    return selected, qualified, evaluations, bar_attempts


def _evaluate_pool(
    args: argparse.Namespace,
    song: SongDefinition,
    bar: int,
    lines: Sequence[str],
    analyzer: CmuProsodyAnalyzer,
    weights: ScoreWeights,
    history: Sequence[Any],
    rhyme_anchors: Mapping[tuple[int, str], tuple[str, ...]],
) -> tuple[tuple[CandidateEvaluation, ...], tuple[CandidateEvaluation, ...], CandidateEvaluation | None]:
    template = BUILTIN_TEMPLATES.get(template_id_for_bar(bar))
    candidates = tuple(
        (f"song-{song.index:02d}-bar-{bar:03d}-candidate-{index + 1}", text, analyzer.analyze(text))
        for index, text in enumerate(lines)
    )
    selection = rank_candidates(
        candidates,
        template=template,
        topic=song.topic,
        history=history,
        rhyme_anchors=rhyme_anchors,
        weights=weights,
        minimum_score=args.minimum_score,
        segment_start_bar=bar // 4 * 4,
        target_bar=bar,
    )
    qualified, selected, _selection_mode = select_flow_with_fallback(
        selection.evaluations,
        minimum_score=args.minimum_score,
        minimum_stress=args.minimum_stress,
        blocked_trigrams=frozenset().union(*(word_trigrams(item.text) for item in history)),
        blocked_opening_words=overused_opening_words(
            (item.text for item in history),
            maximum_uses=2,
        ),
        blocked_closing_words=overused_closing_words(
            (item.text for item in history),
            maximum_uses=2,
        ),
    )
    return selection.evaluations, qualified, selected


def _chosen_record(
    *,
    args: argparse.Namespace,
    song: SongDefinition,
    bar: int,
    selected: CandidateEvaluation,
    qualified: Sequence[CandidateEvaluation],
    evaluations: Sequence[CandidateEvaluation],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tempo = Tempo(args.tempo, 4, 4)
    audio_format = AudioFormat()
    template = BUILTIN_TEMPLATES.get(template_id_for_bar(bar))
    schedule = []
    for item in selected.scheduled:
        tick_in_bar = item.slot.tick - bar * tempo.ticks_per_bar
        schedule.append(
            {
                "slot_index": item.slot.slot_index,
                "absolute_tick": item.slot.tick,
                "tick_in_bar": tick_in_bar,
                "seconds_from_song_start": tempo.tick_to_seconds(item.slot.tick),
                "target_sample_in_bar": tick_frame_in_bar(bar, tick_in_bar, tempo, audio_format),
                "target_stress": item.slot.accent,
                "word": item.syllable.word,
                "syllable_label": item.syllable.label,
                "lexical_stress": item.syllable.stress,
                "phonemes": list(item.syllable.phonemes),
                "analysis_source": item.syllable.analysis_source,
            }
        )
    successful_attempts = [attempt for attempt in attempts if not attempt.get("error")]
    return {
        "song_index": song.index,
        "title": song.title,
        "topic": song.topic,
        "bar": bar,
        "rhyme_section_start": bar // 4 * 4,
        "template_id": template.template_id,
        "source": "h200_api_choices",
        "candidate_id": selected.candidate_id,
        "text": selected.text,
        "normalized_text": selected.analysis.normalized_text,
        "syllable_count": len(selected.analysis.syllables),
        "oov_words": list(selected.analysis.oov_words),
        "rhyme_tail": list(selected.analysis.end_rhyme_tail),
        "total_score": selected.total_score,
        "score_components": [asdict(component) for component in selected.components],
        "schedule": schedule,
        "generation": {
            "selection_mode": "strict" if selected in qualified else "relaxed_opening_word_cap",
            "attempts": len(attempts),
            "successful_attempts": len(successful_attempts),
            "requested_choices": sum(int(attempt["requested_choices"]) for attempt in attempts),
            "unique_pool_count": len(evaluations),
            "hard_valid_pool_count": sum(item.valid for item in evaluations),
            "flow_qualified_pool_count": len(qualified),
            "target_pool_met": len(qualified) >= args.target_pool,
            "total_latency_ms": sum(float(attempt["latency_ms"]) for attempt in attempts),
            "prompt_tokens": sum(int(attempt["prompt_tokens"]) for attempt in successful_attempts),
            "completion_tokens": sum(int(attempt["completion_tokens"]) for attempt in successful_attempts),
        },
    }


def render_audio(args: argparse.Namespace) -> None:
    analyzer = CmuProsodyAnalyzer()
    synthesizer = EspeakPhonemeSynthesizer(cache_size=4096)
    audio_format = AudioFormat()
    tempo = Tempo(args.tempo, 4, 4)
    for song in DEFAULT_SONGS:
        directory = _song_directory(args.output_dir, song)
        records = _read_jsonl(directory / "chosen_lyrics.jsonl")
        validate_chosen_records(records, expected_bars=args.bars)
        if len(records) != args.bars:
            raise ValueError(f"song {song.index} needs {args.bars} chosen bars before audio rendering")
        renderer = DeterministicRapBarRenderer(
            tempo=tempo,
            audio_format=audio_format,
            synthesizer=synthesizer,
            drums=ProceduralBoomBapRenderer(seed=args.seed + song.index * 10_000),
            time_stretcher=RubberBandTimeStretcher(),
            voice=args.voice,
            speed_wpm=args.voice_speed,
            pitch=args.voice_pitch,
        )
        _render_song_audio(args, song, records, analyzer, renderer, tempo, audio_format)


def _render_song_audio(
    args: argparse.Namespace,
    song: SongDefinition,
    records: Sequence[Mapping[str, Any]],
    analyzer: CmuProsodyAnalyzer,
    renderer: DeterministicRapBarRenderer,
    tempo: Tempo,
    audio_format: AudioFormat,
) -> None:
    directory = _song_directory(args.output_dir, song)
    wav_path = directory / "song.wav"
    log_path = directory / "audio_render_log.jsonl"
    warning_counts: Counter[str] = Counter()
    pronunciation_counts: Counter[str] = Counter()
    render_latencies: list[float] = []
    overlap_slots = 0
    max_software_error = 0
    writer = Float32WavAudioSink(wav_path, audio_format)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for record in records:
                bar = int(record["bar"])
                template = BUILTIN_TEMPLATES.get(str(record["template_id"]))
                analysis = analyzer.analyze(str(record["text"]))
                scheduled = align_exact(analysis, materialize_flow(template, bar))
                plan = PlannedRapBar(
                    bar=bar,
                    segment=ScenarioSegment(0, args.bars, song.topic, template.template_id, (str(record["text"]),)),
                    template=template,
                    analysis=analysis,
                    scheduled=scheduled,
                    text=str(record["text"]),
                    source=str(record["source"]),
                    fallback_reason=None,
                    frozen=True,
                )
                prepared = renderer.render(plan)
                _validate_prepared_timing(prepared, plan, tempo, audio_format)
                start_frame = writer.frame_count
                writer.enqueue(prepared)
                writer.mark_started(prepared, start_frame)
                writer.mark_completed(prepared)
                warning_counts.update(warning.code.value for warning in prepared.warnings)
                pronunciation_counts.update(item.pronunciation_source for item in prepared.diagnostics)
                overlap_slots += sum(item.overlap_frames > 0 for item in prepared.diagnostics)
                max_software_error = max(
                    max_software_error,
                    max((abs(item.software_error_samples) for item in prepared.diagnostics), default=0),
                )
                render_latencies.append(prepared.render_latency_ms)
                payload = {
                    "song_index": song.index,
                    "bar": bar,
                    "text": prepared.text,
                    "frame_count": prepared.audio.frame_count,
                    "start_frame": start_frame,
                    "render_latency_ms": prepared.render_latency_ms,
                    "warnings": [asdict(item) for item in prepared.warnings],
                    "diagnostics": [asdict(item) for item in prepared.diagnostics],
                }
                log.write(json.dumps(payload, sort_keys=True) + "\n")
                log.flush()
                print(
                    f"[audio][{song.index:02d}/10][bar {bar + 1:02d}/{args.bars}] "
                    f"render_ms={prepared.render_latency_ms:.1f} warnings={len(prepared.warnings)}",
                    flush=True,
                )
    finally:
        writer.close()

    expected_frames = expected_song_frames(
        bars=args.bars,
        tempo_bpm=args.tempo,
        sample_rate_hz=audio_format.sample_rate_hz,
    )
    if writer.frame_count != expected_frames:
        raise RuntimeError(f"song {song.index} WAV has {writer.frame_count} frames, expected {expected_frames}")
    expected_bytes = 44 + expected_frames * audio_format.channels * audio_format.sample_width_bytes
    if wav_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"song {song.index} WAV byte size does not match its frame count")
    samples = np.memmap(wav_path, dtype="<f4", mode="r", offset=44)
    peak = float(np.max(np.abs(samples), initial=0.0))
    nonzero_samples = int(np.count_nonzero(samples))
    del samples
    stats = {
        "song_index": song.index,
        "title": song.title,
        "topic": song.topic,
        "bars": args.bars,
        "tempo_bpm": args.tempo,
        "sample_rate_hz": audio_format.sample_rate_hz,
        "channels": audio_format.channels,
        "sample_format": "IEEE float32",
        "frame_count": writer.frame_count,
        "duration_seconds": writer.frame_count / audio_format.sample_rate_hz,
        "file_bytes": wav_path.stat().st_size,
        "peak": peak,
        "nonzero_samples": nonzero_samples,
        "render_latency_ms": _distribution(render_latencies),
        "warning_counts": dict(sorted(warning_counts.items())),
        "pronunciation_source_counts": dict(sorted(pronunciation_counts.items())),
        "overlap_slot_count": overlap_slots,
        "max_software_timing_error_samples": max_software_error,
        "wav": str(wav_path),
    }
    _write_json(directory / "audio_stats.json", stats)


def _validate_prepared_timing(prepared, plan, tempo: Tempo, audio_format: AudioFormat) -> None:
    if len(prepared.diagnostics) != len(plan.scheduled):
        raise RuntimeError("audio diagnostics do not cover every scheduled syllable")
    for diagnostic, scheduled in zip(prepared.diagnostics, plan.scheduled, strict=True):
        tick_in_bar = scheduled.slot.tick - plan.bar * tempo.ticks_per_bar
        expected = tick_frame_in_bar(plan.bar, tick_in_bar, tempo, audio_format)
        if diagnostic.target_sample != expected or diagnostic.software_error_samples != 0:
            raise RuntimeError(
                f"bar {plan.bar} slot {diagnostic.slot_index} missed target sample {expected}"
            )


def _write_song_lyric_artifacts(
    args: argparse.Namespace,
    song: SongDefinition,
    chosen: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    directory = _song_directory(args.output_dir, song)
    lines = [
        f"{song.title}",
        f"Topic: {song.topic}",
        f"Tempo: {args.tempo:g} BPM | Bars: {args.bars} | Syllables per bar: 9",
        "",
    ]
    lines.extend(
        f"{int(record['bar']) + 1:02d}. [{record['template_id']}] {record['text']}"
        for record in chosen
    )
    (directory / "lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected_scores = [float(record["total_score"]) for record in chosen]
    stress_scores = [
        float(next(item["value"] for item in record["score_components"] if item["name"] == "stress_alignment"))
        for record in chosen
    ]
    stats = {
        "song_index": song.index,
        "title": song.title,
        "topic": song.topic,
        "bars_completed": len(chosen),
        "bars_expected": args.bars,
        "exact_nine_syllable_bars": sum(int(record["syllable_count"]) == 9 for record in chosen),
        "ascii_lyric_bars": sum(str(record["text"]).isascii() for record in chosen),
        "api_attempts": len(attempts),
        "api_errors": sum(bool(attempt.get("error")) for attempt in attempts),
        "requested_choices": sum(int(attempt["requested_choices"]) for attempt in attempts),
        "returned_candidates": sum(len(attempt.get("candidates", ())) for attempt in attempts),
        "generation_latency_ms": _distribution(float(attempt["latency_ms"]) for attempt in attempts),
        "selected_total_score": _distribution(selected_scores),
        "selected_stress_alignment": _distribution(stress_scores),
        "qualified_pool_size": _distribution(
            int(record["generation"]["flow_qualified_pool_count"]) for record in chosen
        ),
        "target_pool_met_bars": sum(bool(record["generation"]["target_pool_met"]) for record in chosen),
        "selection_mode_counts": dict(
            sorted(
                Counter(
                    str(record["generation"].get("selection_mode", "strict"))
                    for record in chosen
                ).items()
            )
        ),
        "oov_bars": sum(bool(record["oov_words"]) for record in chosen),
    }
    _write_json(directory / "generation_stats.json", stats)


def write_album_summary(args: argparse.Namespace) -> None:
    songs = []
    chosen_records: list[dict[str, Any]] = []
    for song in DEFAULT_SONGS:
        directory = _song_directory(args.output_dir, song)
        lyric_stats = _read_json(directory / "generation_stats.json")
        audio_path = directory / "audio_stats.json"
        chosen_records.extend(_read_jsonl(directory / "chosen_lyrics.jsonl"))
        songs.append(
            {
                "song_index": song.index,
                "title": song.title,
                "topic": song.topic,
                "directory": str(directory),
                "lyrics": lyric_stats,
                "audio": _read_json(audio_path) if audio_path.exists() else None,
            }
        )
    audio_stats = [song["audio"] for song in songs if song["audio"] is not None]
    aggregate = {
        "songs": len(songs),
        "bars": sum(int(song["lyrics"]["bars_completed"]) for song in songs),
        "exact_nine_syllable_bars": sum(int(song["lyrics"]["exact_nine_syllable_bars"]) for song in songs),
        "api_attempts": sum(int(song["lyrics"]["api_attempts"]) for song in songs),
        "api_errors": sum(int(song["lyrics"]["api_errors"]) for song in songs),
        "requested_choices": sum(int(song["lyrics"]["requested_choices"]) for song in songs),
        "returned_candidates": sum(int(song["lyrics"]["returned_candidates"]) for song in songs),
        "target_pool_met_bars": sum(int(song["lyrics"]["target_pool_met_bars"]) for song in songs),
        "selection_mode_counts": dict(
            sorted(
                sum(
                    (Counter(song["lyrics"]["selection_mode_counts"]) for song in songs),
                    Counter(),
                ).items()
            )
        ),
        "selected_total_score": _distribution(float(record["total_score"]) for record in chosen_records),
        "selected_stress_alignment": _distribution(
            float(
                next(
                    component["value"]
                    for component in record["score_components"]
                    if component["name"] == "stress_alignment"
                )
            )
            for record in chosen_records
        ),
        "wav_files": sum(song["audio"] is not None for song in songs),
        "total_audio_seconds": sum(
            float(song["audio"]["duration_seconds"]) for song in songs if song["audio"] is not None
        ),
        "max_software_timing_error_samples": max(
            (int(song["audio"]["max_software_timing_error_samples"]) for song in songs if song["audio"] is not None),
            default=None,
        ),
        "wav_bytes": sum(int(stats["file_bytes"]) for stats in audio_stats),
        "peak_max": max((float(stats["peak"]) for stats in audio_stats), default=None),
        "nonzero_samples": sum(int(stats["nonzero_samples"]) for stats in audio_stats),
        "overlap_slot_count": sum(int(stats["overlap_slot_count"]) for stats in audio_stats),
        "warning_counts": dict(
            sorted(sum((Counter(stats["warning_counts"]) for stats in audio_stats), Counter()).items())
        ),
        "pronunciation_source_counts": dict(
            sorted(
                sum(
                    (Counter(stats["pronunciation_source_counts"]) for stats in audio_stats),
                    Counter(),
                ).items()
            )
        ),
    }
    _write_json(args.output_dir / "album_stats.json", {"aggregate": aggregate, "songs": songs})


def write_artifact_index(args: argparse.Namespace) -> None:
    lines = [
        "# Offline Rap Album Artifacts",
        "",
        f"- Tempo: {args.tempo:g} BPM",
        f"- Songs: {len(DEFAULT_SONGS)}",
        f"- Bars per song: {args.bars}",
        "- Audio: 48 kHz stereo IEEE-float WAV",
        "- Qualification: exactly nine syllables, total score >= "
        f"{args.minimum_score:.2f}, stress alignment >= {args.minimum_stress:.2f}",
        "",
    ]
    for song in DEFAULT_SONGS:
        directory = _song_directory(args.output_dir, song)
        relative = directory.relative_to(args.output_dir)
        lines.extend(
            [
                f"## {song.index:02d}. {song.title}",
                "",
                f"Topic: {song.topic}",
                "",
                f"- Human-readable lyrics: `{relative}/lyrics.txt`",
                f"- Chosen schedule log: `{relative}/chosen_lyrics.jsonl`",
                f"- Generation attempts: `{relative}/generation_attempts.jsonl`",
                f"- Generation statistics: `{relative}/generation_stats.json`",
                f"- Audio render log: `{relative}/audio_render_log.jsonl`",
                f"- Audio statistics: `{relative}/audio_stats.json`",
                f"- Song audio: `{relative}/song.wav`",
                "",
            ]
        )
    (args.output_dir / "ARTIFACT_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def _ensure_manifest(args: argparse.Namespace) -> None:
    path = args.output_dir / "album_manifest.json"
    configuration = {
        "bars_per_song": args.bars,
        "tempo_bpm": args.tempo,
        "beats_per_bar": 4,
        "ticks_per_beat": 4,
        "choices_per_attempt": args.choices,
        "minimum_score": args.minimum_score,
        "minimum_stress": args.minimum_stress,
        "target_pool": args.target_pool,
        "max_attempts": args.max_attempts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "api_sampling_seed": None,
        "prompt_variation_seed": args.seed,
        "model": args.model,
        "base_url": args.base_url,
        "voice": args.voice,
        "voice_speed_wpm": args.voice_speed,
        "voice_pitch": args.voice_pitch,
        "audio_format": {"sample_rate_hz": 48_000, "channels": 2, "sample_width_bytes": 4},
        "rhyme_anchor_reset_bars": 4,
        "repetition_gate": "reject any candidate sharing a normalized word trigram with an earlier song bar",
        "opening_word_maximum_uses": 2,
        "closing_word_maximum_uses": 2,
        "narrative_focus": "five song stages crossed with ten rotating semantic lenses",
    }
    songs = [asdict(song) for song in DEFAULT_SONGS]
    if path.exists():
        existing = _read_json(path)
        if existing.get("configuration") != configuration or existing.get("songs") != songs:
            raise ValueError("existing album manifest does not match requested configuration")
        return
    _write_json(
        path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "offline stress-flow-aligned rap album research artifact",
            "configuration": configuration,
            "songs": songs,
            "assumptions": [
                "The request means ten songs containing fifty bars each.",
                "Each song holds one fixed topic for all fifty bars.",
                "The existing three nine-slot flow templates repeat in an eight-bar arrangement.",
                "Rhyme anchors reset every four bars to avoid one rhyme constraint across an entire song.",
                "Pronunciation uses the existing CMUdict-first eSpeak syllable renderer.",
                "Every syllable onset is validated against the exact template sample before WAV commit.",
            ],
        },
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.bars <= 0 or args.choices <= 0 or args.target_pool <= 0 or args.max_attempts <= 0:
        raise ValueError("bars, choices, target-pool, and max-attempts must be positive")
    if args.tempo <= 0 or args.timeout_s <= 0:
        raise ValueError("tempo and timeout must be positive")
    if not 0 <= args.minimum_score <= 1 or not 0 <= args.minimum_stress <= 1:
        raise ValueError("score and stress thresholds must be between zero and one")


def _template_payload(template) -> dict[str, Any]:
    return {
        "template_id": template.template_id,
        "ticks": [slot.tick_in_bar for slot in template.slots],
        "target_stress": [slot.target_stress for slot in template.slots],
        "boundary_strength": [slot.boundary_strength for slot in template.slots],
        "rhyme_group": [slot.rhyme_group for slot in template.slots],
    }


def _song_directory(root: Path, song: SongDefinition) -> Path:
    return root / f"{song.index:02d}_{song.slug}"


def _deduplicate(lines: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = normalize_text(line)
        if key and key not in seen:
            seen.add(key)
            output.append(line)
    return tuple(output)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

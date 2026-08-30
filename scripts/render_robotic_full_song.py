#!/usr/bin/env python3
"""Render one frozen lyric corpus with the pitch-preserving robotic backend."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from scipy.io import wavfile

from streammuse.application.rap.audio_rendering import bar_frame_count, bar_start_frame
from streammuse.application.rap.bar_renderer import DeterministicRapBarRenderer
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import (
    AudioFormat,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    PcmAudio,
    ScenarioSegment,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.artifacts import (
    file_sha256,
    listening_artifact_filename,
)
from streammuse.experiments.rap_audio_protocols.audio import (
    TARGET_SAMPLE_RATE_HZ,
    TARGET_STEREO_FORMAT,
    TARGET_VOCAL_FORMAT,
    mix_stems,
    write_listening_wav,
)
from streammuse.infrastructure.rap.speech import EspeakPhonemeSynthesizer
from streammuse.infrastructure.rap.time_stretch import RubberBandTimeStretcher


_RENDERER_LABEL = "robotic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--expected-bars", type=int, default=50)
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--voice-speed", type=int, default=175)
    parser.add_argument("--voice-pitch", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_song(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        song_id=args.song_id,
        expected_bars=args.expected_bars,
        voice=args.voice,
        voice_speed=args.voice_speed,
        voice_pitch=args.voice_pitch,
    )
    return 0


def build_planned_bar(record: Mapping[str, Any], *, total_bars: int) -> PlannedRapBar:
    bar = int(record["bar"])
    if bar < 0 or total_bars <= 0 or bar >= total_bars:
        raise ValueError("recorded bar lies outside the requested song")
    raw_schedule = record.get("schedule")
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError(f"bar {bar} requires a nonempty recorded schedule")

    ticks = tuple(int(item["tick_in_bar"]) for item in raw_schedule)
    if ticks != tuple(sorted(set(ticks))):
        raise ValueError(f"bar {bar} schedule ticks must be unique and increasing")
    if tuple(int(item["slot_index"]) for item in raw_schedule) != tuple(range(len(raw_schedule))):
        raise ValueError(f"bar {bar} schedule slot indexes must be contiguous and zero-based")
    if any(int(item["absolute_tick"]) != bar * 16 + tick for item, tick in zip(raw_schedule, ticks, strict=True)):
        raise ValueError(f"bar {bar} recorded absolute ticks disagree with tick_in_bar")

    flow_slots = tuple(
        FlowSlot(
            tick_in_bar=tick,
            duration_ticks=(ticks[index + 1] if index + 1 < len(ticks) else 16) - tick,
            target_stress=float(item["target_stress"]),
            boundary_strength=int(item.get("boundary_strength", 3 if index == len(ticks) - 1 else 0)),
            rhyme_group="A" if index == len(ticks) - 1 else None,
        )
        for index, (item, tick) in enumerate(zip(raw_schedule, ticks, strict=True))
    )
    template_id = str(record["template_id"])
    template = FlowTemplate(
        template_id=template_id,
        name=f"Recorded flow: {template_id}",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=flow_slots,
        provenance=FlowProvenance(
            kind=str((record.get("flow_retiming") or {}).get("template_provenance", "recorded_schedule")),
            source="frozen chosen_lyrics.jsonl schedule",
        ),
    )
    beat_slots = materialize_flow(template, bar)
    scheduled = tuple(
        ScheduledSyllable(
            slot=slot,
            syllable=_recorded_syllable(raw_schedule, index),
        )
        for index, slot in enumerate(beat_slots)
    )
    text = str(record["text"])
    topic = str(record["topic"])
    return PlannedRapBar(
        bar=bar,
        segment=ScenarioSegment(0, total_bars, topic, template_id, (text,)),
        template=template,
        analysis=_analysis_from_schedule(text, scheduled),
        scheduled=scheduled,
        text=text,
        source=str(record.get("source", "frozen_corpus")),
        fallback_reason=None,
        frozen=True,
    )


def render_song(
    *,
    input_dir: Path,
    output_dir: Path,
    song_id: str,
    expected_bars: int = 50,
    renderer: Any | None = None,
    voice: str = "en-us",
    voice_speed: int = 175,
    voice_pitch: int = 50,
) -> dict[str, Any]:
    if expected_bars <= 0:
        raise ValueError("expected_bars must be positive")
    common_dir = Path(input_dir).resolve() / "common" / song_id
    chosen_path = common_dir / "chosen_lyrics.jsonl"
    requests_path = common_dir / "requests.jsonl"
    drums_path = common_dir / "drums.wav"
    records = _read_jsonl(chosen_path)
    _validate_records(records, expected_bars=expected_bars)
    tempo_bpm = _read_tempo_bpm(requests_path)
    tempo = Tempo(tempo_bpm, 4, 4)
    audio_format = AudioFormat()

    if renderer is None:
        _require_audio_executables()
        renderer = DeterministicRapBarRenderer(
            tempo=tempo,
            audio_format=audio_format,
            synthesizer=EspeakPhonemeSynthesizer(cache_size=4096),
            drums=_SilentDrumRenderer(),
            time_stretcher=RubberBandTimeStretcher(),
            voice=voice,
            speed_wpm=voice_speed,
            pitch=voice_pitch,
        )

    title = str(records[0]["title"])
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    vocals_path = destination / listening_artifact_filename(title, _RENDERER_LABEL, "vocals")
    mix_path = destination / listening_artifact_filename(title, _RENDERER_LABEL, "mix")
    log_path = destination / listening_artifact_filename(title, _RENDERER_LABEL, "render log", extension=".jsonl")
    stats_path = destination / listening_artifact_filename(title, _RENDERER_LABEL, "stats", extension=".json")

    warning_counts: Counter[str] = Counter()
    pronunciation_counts: Counter[str] = Counter()
    render_latencies: list[float] = []
    vocal_bars: list[np.ndarray] = []
    overlap_slots = 0
    max_software_error = 0
    start_frame = 0
    with log_path.open("w", encoding="utf-8") as log:
        for record in records:
            plan = build_planned_bar(record, total_bars=expected_bars)
            prepared = renderer.render(plan)
            expected_bar_frames = bar_frame_count(plan.bar, tempo, audio_format)
            _validate_prepared_bar(prepared, plan, expected_frames=expected_bar_frames)
            stereo = np.frombuffer(prepared.audio.data, dtype=np.float32).reshape(
                prepared.audio.frame_count,
                prepared.audio.format.channels,
            )
            vocal_bars.append(stereo.mean(axis=1, dtype=np.float32))
            warning_counts.update(item.code.value for item in prepared.warnings)
            pronunciation_counts.update(item.pronunciation_source for item in prepared.diagnostics)
            overlap_slots += sum(item.overlap_frames > 0 for item in prepared.diagnostics)
            max_software_error = max(
                max_software_error,
                max((abs(item.software_error_samples) for item in prepared.diagnostics), default=0),
            )
            render_latencies.append(float(prepared.render_latency_ms))
            payload = {
                "bar": plan.bar,
                "text": plan.text,
                "template_id": plan.template.template_id,
                "recorded_ticks": [item.slot.tick for item in plan.scheduled],
                "start_frame": start_frame,
                "frame_count": prepared.audio.frame_count,
                "render_latency_ms": prepared.render_latency_ms,
                "warnings": [asdict(item) for item in prepared.warnings],
                "diagnostics": [asdict(item) for item in prepared.diagnostics],
            }
            log.write(json.dumps(payload, sort_keys=True) + "\n")
            log.flush()
            start_frame += prepared.audio.frame_count
            print(
                f"[robotic][{title}][bar {plan.bar + 1:02d}/{expected_bars}] "
                f"render_ms={prepared.render_latency_ms:.1f} warnings={len(prepared.warnings)}",
                flush=True,
            )

    vocal_samples = np.concatenate(vocal_bars).astype(np.float32, copy=False)
    expected_frames = bar_start_frame(expected_bars, tempo, TARGET_VOCAL_FORMAT)
    if vocal_samples.shape[0] != expected_frames:
        raise RuntimeError(
            f"assembled vocals contain {vocal_samples.shape[0]} frames, expected {expected_frames}"
        )
    vocals = PcmAudio(TARGET_VOCAL_FORMAT, expected_frames, vocal_samples.reshape(-1, 1).tobytes())
    drums = _load_common_drums(drums_path, expected_frames=expected_frames)
    mix = mix_stems(
        vocals,
        drums,
        vocal_gain=1.0,
        drum_gain=0.45,
        listening_wav_path=mix_path,
    )
    write_listening_wav(vocals_path, vocals)

    stats = {
        "schema_version": "streammuse.robotic_full_song.v1",
        "renderer": "espeak_phoneme_plus_rubberband_r3_pitch_preserving",
        "song_id": song_id,
        "song_index": int(records[0].get("song_index", 0)),
        "title": title,
        "topic": str(records[0]["topic"]),
        "bar_count": expected_bars,
        "tempo_bpm": tempo_bpm,
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "frame_count": expected_frames,
        "duration_seconds": expected_frames / TARGET_SAMPLE_RATE_HZ,
        "effective_vocal_gain": 0.80,
        "mix_drum_gain": 0.45,
        "mix_peak_before_limiter": mix.peak_before_limiter,
        "mix_limiter_gain": mix.applied_gain,
        "vocal_peak": float(np.max(np.abs(vocal_samples), initial=0.0)),
        "vocal_nonzero_samples": int(np.count_nonzero(vocal_samples)),
        "render_latency_ms": _distribution(render_latencies),
        "warning_counts": dict(sorted(warning_counts.items())),
        "pronunciation_source_counts": dict(sorted(pronunciation_counts.items())),
        "overlap_slot_count": overlap_slots,
        "max_software_timing_error_samples": max_software_error,
        "inputs": {
            "chosen_lyrics": str(chosen_path.resolve()),
            "chosen_lyrics_sha256": file_sha256(chosen_path),
            "requests": str(requests_path.resolve()),
            "requests_sha256": file_sha256(requests_path),
            "drums": str(drums_path.resolve()),
            "drums_sha256": file_sha256(drums_path),
        },
        "artifacts": {
            "vocals": str(vocals_path),
            "mix": str(mix_path),
            "render_log": str(log_path),
            "stats": str(stats_path),
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


class _SilentDrumRenderer:
    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio:
        del template
        frames = bar_frame_count(bar, tempo, audio_format)
        return PcmAudio(
            audio_format,
            frames,
            np.zeros((frames, audio_format.channels), dtype=np.float32).tobytes(),
        )


def _recorded_syllable(schedule: list[Mapping[str, Any]], index: int) -> Syllable:
    word = str(schedule[index]["word"])
    first = index
    while first > 0 and str(schedule[first - 1]["word"]) == word:
        first -= 1
    last = index
    while last + 1 < len(schedule) and str(schedule[last + 1]["word"]) == word:
        last += 1
    return Syllable(
        word=word,
        index_in_word=index - first,
        syllable_count=last - first + 1,
        stress=int(schedule[index]["lexical_stress"]),
        phonemes=tuple(str(item) for item in schedule[index].get("phonemes", ())),
        analysis_source=str(schedule[index].get("analysis_source", "recorded_schedule")),
    )


def _analysis_from_schedule(text: str, scheduled: tuple[ScheduledSyllable, ...]):
    from streammuse.domain.rap import ProsodyAnalysis

    return ProsodyAnalysis(
        text=text,
        normalized_text=" ".join(text.lower().split()),
        syllables=tuple(item.syllable for item in scheduled),
        end_rhyme_tail=scheduled[-1].syllable.phonemes,
        oov_words=(),
        heuristic_words=(),
        punctuation_boundary_after=(),
    )


def _validate_records(records: list[dict[str, Any]], *, expected_bars: int) -> None:
    if len(records) != expected_bars:
        raise ValueError(f"chosen lyric records contain {len(records)} bars, expected {expected_bars}")
    if [int(record["bar"]) for record in records] != list(range(expected_bars)):
        raise ValueError("chosen lyric records must contain contiguous bars starting at zero")
    for field in ("title", "topic", "song_index"):
        values = {str(record[field]) for record in records}
        if len(values) != 1:
            raise ValueError(f"chosen lyric records must use one {field}")


def _validate_prepared_bar(prepared, plan: PlannedRapBar, *, expected_frames: int) -> None:
    if prepared.bar != plan.bar or prepared.text != plan.text:
        raise RuntimeError(f"renderer returned mismatched metadata for bar {plan.bar}")
    if prepared.audio.format != TARGET_STEREO_FORMAT:
        raise RuntimeError("robotic renderer must return 48 kHz stereo float32 audio")
    if prepared.audio.frame_count != expected_frames:
        raise RuntimeError(
            f"bar {plan.bar} contains {prepared.audio.frame_count} frames, expected {expected_frames}"
        )


def _load_common_drums(path: Path, *, expected_frames: int) -> PcmAudio:
    sample_rate_hz, raw = wavfile.read(path)
    samples = np.asarray(raw)
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        raise ValueError(f"common drums must use {TARGET_SAMPLE_RATE_HZ} Hz audio")
    if samples.ndim != 2 or samples.shape[1] != TARGET_STEREO_FORMAT.channels:
        raise ValueError("common drums must be stereo")
    if samples.shape[0] != expected_frames:
        raise ValueError(f"common drums contain {samples.shape[0]} frames, expected {expected_frames}")
    if samples.dtype == np.int16:
        normalised = samples.astype(np.float32) / 32768.0
    elif np.issubdtype(samples.dtype, np.floating):
        normalised = samples.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported common drum WAV dtype: {samples.dtype}")
    return PcmAudio(TARGET_STEREO_FORMAT, expected_frames, normalised.tobytes())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    return records


def _read_tempo_bpm(path: Path) -> float:
    requests = _read_jsonl(path)
    values = {float(request.get("tempo_bpm", 90.0)) for request in requests}
    if len(values) != 1:
        raise ValueError("render requests must use one tempo")
    tempo_bpm = values.pop()
    if tempo_bpm <= 0:
        raise ValueError("tempo must be positive")
    return tempo_bpm


def _require_audio_executables() -> None:
    for executable in ("espeak-ng", "rubberband"):
        if shutil.which(executable) is None:
            raise OSError(f"robotic rendering requires the {executable} executable")


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "total": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "count": float(len(values)),
        "total": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "maximum": float(max(values)),
    }


if __name__ == "__main__":
    raise SystemExit(main())

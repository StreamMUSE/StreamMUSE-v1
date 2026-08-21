from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from collections.abc import Mapping
from pathlib import Path

from streammuse.application.rap.alignment import align_exact
from streammuse.domain.rap.flow import materialize_flow
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.experiments.rap_audio_protocols.warp import (
    RubberBandTimeMapStretcher,
)
from streammuse.infrastructure.rap.mms_forced_alignment import MmsForcedAligner
from streammuse.infrastructure.rap.moss_aligned_phrase import (
    MossAlignedPhraseRenderer,
)
from streammuse.infrastructure.rap.moss_tts import PersistentMossSynthesizer
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
TEMPO_BPM = 90.0
SAMPLE_RATE_HZ = 24_000
EXPECTED_FRAMES = 128_000
LINES = (
    "neon signs hum while the shadows creep",
    "cold neon hums while the city wakes",
)
TEMPLATE_IDS = ("baseline_syncopated_9", "baseline_staggered_9")


def _emit(event: str, **values: object) -> None:
    print(
        json.dumps(
            {"event": event, "monotonic": round(time.monotonic(), 6), **values},
            sort_keys=True,
        ),
        flush=True,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _request() -> TwoBarRenderRequest:
    analyzer = CmuProsodyAnalyzer()
    scheduled = []
    for bar, (line, template_id) in enumerate(zip(LINES, TEMPLATE_IDS, strict=True)):
        analysis = analyzer.analyze(line)
        slots = materialize_flow(BUILTIN_TEMPLATES.get(template_id), bar)
        scheduled.extend(align_exact(analysis, slots))
    seconds_per_tick = 60.0 / TEMPO_BPM / 4.0
    syllables = tuple(
        SyllableTarget(
            word=item.syllable.word,
            index_in_word=item.syllable.index_in_word,
            phonemes=item.syllable.phonemes,
            lexical_stress=item.syllable.stress,
            target_stress=item.slot.accent,
            boundary_strength=item.slot.boundary_strength,
            absolute_tick=item.slot.tick,
            tick_in_chunk=item.slot.tick,
            target_seconds=item.slot.tick * seconds_per_tick,
        )
        for item in scheduled
    )
    if len(syllables) != 18:
        raise RuntimeError(
            f"known smoke transcript must contain 18 syllables, got {len(syllables)}"
        )
    return TwoBarRenderRequest(
        song_id="task3-h200-real-smoke",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="\n".join(LINES),
        syllables=syllables,
        tempo_bpm=TEMPO_BPM,
    )


def _wav_evidence(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    with wave.open(str(path), "rb") as rendered:
        evidence = {
            "sample_rate_hz": rendered.getframerate(),
            "channels": rendered.getnchannels(),
            "sample_width_bytes": rendered.getsampwidth(),
            "frame_count": rendered.getnframes(),
            "compression": rendered.getcomptype(),
        }
    evidence["sha256"] = hashlib.sha256(payload).hexdigest()
    evidence["byte_count"] = len(payload)
    if evidence != {
        **evidence,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": 1,
        "sample_width_bytes": 2,
        "frame_count": EXPECTED_FRAMES,
        "compression": "NONE",
    }:
        raise RuntimeError(f"unexpected rendered WAV properties: {evidence}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--moss-device", default="cuda:0")
    parser.add_argument("--aligner-device", default="cuda:0")
    parser.add_argument("--rubberband-binary", default="rubberband")
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    request = _request()
    (artifact_root / "request.json").write_text(
        json.dumps(request.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit(
        "request_ready",
        request_sha256=request.sha256,
        tempo_bpm=request.tempo_bpm,
        syllables=len(request.syllables),
        target_frames=EXPECTED_FRAMES,
    )

    started = time.perf_counter()
    synthesizer = PersistentMossSynthesizer.load(
        model_id=MODEL_ID,
        device=args.moss_device,
        reference_wav=args.reference_wav,
    )
    moss_load_ms = (time.perf_counter() - started) * 1_000.0
    _emit("moss_loaded", elapsed_ms=round(moss_load_ms, 3))

    started = time.perf_counter()
    aligner = MmsForcedAligner.load(device=args.aligner_device)
    aligner_load_ms = (time.perf_counter() - started) * 1_000.0
    _emit("mms_loaded", elapsed_ms=round(aligner_load_ms, 3))

    warmup_root = artifact_root / "warmup"
    warmup_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    warmup_source = synthesizer.synthesize(request, warmup_root / "source.wav")
    moss_warmup_ms = (time.perf_counter() - started) * 1_000.0
    _emit(
        "moss_warmed",
        elapsed_ms=round(moss_warmup_ms, 3),
        source_sha256=warmup_source.source_wav_sha256,
    )
    started = time.perf_counter()
    aligner_warmup = aligner.warmup(warmup_source.output_wav, request.text)
    aligner_warmup_ms = (time.perf_counter() - started) * 1_000.0
    _emit(
        "mms_warmed",
        elapsed_ms=round(aligner_warmup_ms, 3),
        confidence=round(float(aligner_warmup["confidence"]), 6),
    )

    def stretcher_factory(**kwargs: object) -> RubberBandTimeMapStretcher:
        return RubberBandTimeMapStretcher(
            binary=str(args.rubberband_binary),
            **kwargs,
        )

    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=aligner,
        stretcher_factory=stretcher_factory,
        rubberband_version="rubberband 3.3.0 R3",
    )
    renders = []
    for index in (1, 2):
        workspace = artifact_root / f"render-{index}"
        _emit("render_started", index=index, workspace=str(workspace))
        started = time.perf_counter()
        result = renderer.render(request, workspace)
        wall_ms = (time.perf_counter() - started) * 1_000.0
        for required in ("source.wav", "mms_alignment.json", "vocal.wav"):
            if not (workspace / required).is_file():
                raise RuntimeError(
                    f"render {index} omitted canonical artifact {required}"
                )
        alignment = json.loads(
            (workspace / "mms_alignment.json").read_text(encoding="utf-8")
        )
        if len(alignment["mapping"]["anchors"]) != len(request.syllables):
            raise RuntimeError(
                f"render {index} alignment does not cover every syllable"
            )
        wav = _wav_evidence(workspace / "vocal.wav")
        render_evidence = {
            "index": index,
            "wall_ms": wall_ms,
            "stage_timings_ms": dict(result.stage_timings_ms),
            "warnings": list(result.warnings),
            "alignment_diagnostics": _jsonable(result.alignment_diagnostics),
            "audio_diagnostics": _jsonable(result.audio_diagnostics),
            "versions": _jsonable(result.model_tool_versions),
            "wav": wav,
            "source_sha256": alignment["source"]["sha256"],
            "alignment_confidence": alignment["aligner"]["confidence"],
            "mapping_method_counts": alignment["mapping"]["method_counts"],
            "endpoint_policy": alignment["mapping"]["endpoint_policy"],
        }
        renders.append(render_evidence)
        _emit(
            "render_complete",
            index=index,
            wall_ms=round(wall_ms, 3),
            stage_timings_ms=render_evidence["stage_timings_ms"],
            vocal_sha256=wav["sha256"],
            warning_count=len(result.warnings),
        )

    summary = {
        "request_sha256": request.sha256,
        "model_id": MODEL_ID,
        "moss_device": args.moss_device,
        "aligner_device": args.aligner_device,
        "rubberband_binary": str(args.rubberband_binary),
        "cold_load_ms": {"moss": moss_load_ms, "aligner": aligner_load_ms},
        "warmup_ms": {"moss": moss_warmup_ms, "aligner": aligner_warmup_ms},
        "renders": renders,
        "repeat_vocal_hash_equal": renders[0]["wav"]["sha256"]
        == renders[1]["wav"]["sha256"],
        "repeat_source_hash_equal": renders[0]["source_sha256"]
        == renders[1]["source_sha256"],
    }
    summary_path = artifact_root / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit("smoke_complete", summary=str(summary_path))


if __name__ == "__main__":
    main()

"""Client simulation: drive a live render server the way the Mac controller does.

Sends consecutive two-bar requests (default scenario, realtime policy, last
four committed lines as context) through the real client-side
RemoteMossChunkPreparationStrategy: HTTP, optional Opus decode, manifest
validation, (protocol v2) local Rubber Band R3 warp, resample, and local drum
mix. Records client wall time, transfer
timing, Mac-side validation/mix time, server stage timings from the manifest,
candidate statistics, and vLLM token counters (when the model server exposes
/metrics). The playback clock, audio device, and fallback substitution are not
simulated, so fallback rates and underruns need the real demo.

--budget-ms sets both the request's remaining budget and the client deadline:
5000 approximates realtime pressure; a large value (e.g. 60000) gives the
unconstrained reference run. --save-audio keeps each chunk's vocal and mixed
WAVs, and --replay-context reuses the context lines a previous run recorded,
so per-chunk comparisons are not confounded by a diverging context trajectory.
--protocol v2 (default) receives the raw MOSS phrase plus MMS onsets and warps
it on this machine with --warp-policy; v1 receives the server-warped vocal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def vllm_counters(url: str) -> dict[str, float]:
    import httpx

    try:
        response = httpx.get(url.rstrip("/").removesuffix("/v1") + "/metrics", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        # Not every OpenAI-compatible server exposes Prometheus metrics.
        return {}
    text = response.text
    wanted = {
        "vllm:prompt_tokens_total": "prompt_tokens",
        "vllm:generation_tokens_total": "generation_tokens",
        "vllm:request_success_total": "requests",
        "vllm:e2e_request_latency_seconds_sum": "e2e_seconds_sum",
        "vllm:e2e_request_latency_seconds_count": "e2e_count",
        "vllm:time_to_first_token_seconds_sum": "ttft_seconds_sum",
        "vllm:prefix_cache_hits_total": "prefix_cache_hits",
        "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    }
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in wanted:
            key = wanted[match.group(1)]
            out[key] = out.get(key, 0.0) + float(match.group(3))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-url", default="http://127.0.0.1:8020")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--budget-ms", type=int, default=5000)
    parser.add_argument("--transport", choices=("pcm", "opus"), default="opus")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--render-reserve-ms", type=int, default=3_000)
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--warp-policy",
        choices=("gentle_sparse_r3", "all_onsets_r3"),
        default="gentle_sparse_r3",
        help="local R3 policy for --protocol v2 (v1 uses the server's policy)",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="write chunk-NNN-vocals.wav (warped vocal), chunk-NNN-source.wav (v2 raw MOSS) "
        "and chunk-NNN-mix.wav (Mac mix) under --out",
    )
    parser.add_argument(
        "--replay-context",
        type=Path,
        help="records.jsonl of an earlier run; reuse its per-chunk context_lines instead of rolling",
    )
    args = parser.parse_args()
    replay_context = _load_replay_context(args.replay_context) if args.replay_context else None

    from streammuse.application.rap.chunk_audio import RemoteMossChunkPreparationStrategy
    from streammuse.domain.rap.audio import AudioFormat
    from streammuse.domain.rap.remote_chunk import (
        REMOTE_CHUNK_SCHEMA_VERSION,
        REMOTE_CHUNK_SCHEMA_VERSION_V2,
        RemoteCandidatePolicy,
        RemoteRapBarRequest,
        RemoteRapChunkRequest,
    )
    from streammuse.domain.timing import Tempo
    from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
    from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
    from streammuse.infrastructure.rap.remote_chunk_client import RemoteChunkClient
    from streammuse.infrastructure.rap.scenarios import default_scenario
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    args.out.mkdir(parents=True, exist_ok=False)
    scenario = default_scenario()
    tempo = Tempo(scenario.tempo_bpm, 4, 4)
    analyzer = CmuProsodyAnalyzer()
    client = RemoteChunkClient(args.render_url, audio_transport=args.transport)
    health = client.health()
    if not health.ready:
        raise SystemExit("render server not ready")
    schema_version = REMOTE_CHUNK_SCHEMA_VERSION
    phrase_warper = None
    warped: dict[str, object] = {}
    if args.protocol == "v2":
        from streammuse.infrastructure.rap.phrase_warp import (
            RubberBandPhraseWarper,
            probe_rubberband_r3,
        )

        schema_version = REMOTE_CHUNK_SCHEMA_VERSION_V2
        if schema_version not in health.supported_schema_versions:
            raise SystemExit("render server does not support protocol v2; use --protocol v1")
        local_warper = RubberBandPhraseWarper(
            policy=args.warp_policy, rubberband_version=probe_rubberband_r3()
        )

        class _RecordingWarper:
            def warp(self, *a, **k):
                result = local_warper.warp(*a, **k)
                warped["warp_ms"] = result.warp_ms
                warped["vocal_wav"] = result.vocal_wav
                return result

        phrase_warper = _RecordingWarper()
    strategy = RemoteMossChunkPreparationStrategy(
        client=client,
        audio_format=AudioFormat(),
        drums=ProceduralBoomBapRenderer(seed=args.seed),
        prosody=analyzer,
        tempo=tempo,
        phrase_warper=phrase_warper,
    )
    policy = RemoteCandidatePolicy.realtime_default(render_reserve_ms=args.render_reserve_ms)
    session = f"profile-{int(time.time())}"
    context: list[str] = []

    # Wrap the client to separate transport from Mac-side work.
    transfer: dict[str, object] = {}
    original_prepare = client.prepare

    def timed_prepare(*a, **k):
        started = time.perf_counter()
        response = original_prepare(*a, **k)
        transfer["client_prepare_ms"] = (time.perf_counter() - started) * 1000.0
        transfer["timing"] = response.timing
        transfer["manifest"] = response.package.manifest
        return response

    client.prepare = timed_prepare  # type: ignore[method-assign]

    from streammuse.application.rap import chunk_audio as ca

    drums = ProceduralBoomBapRenderer(seed=args.seed)
    captured: dict[str, object] = {}
    original_decode = client._decode_package

    def keep_package(*a, **k):
        started = time.perf_counter()
        package = original_decode(*a, **k)
        captured["decode_ms"] = (time.perf_counter() - started) * 1000.0
        captured["package"] = package
        return package

    client._decode_package = keep_package  # type: ignore[method-assign]

    def mac_mix_probe(request, _transfer):
        """Time the Mac decode/resample/drum-mix steps when validation rejects."""
        package = captured.get("package")
        vocal_wav = warped.get("vocal_wav") or getattr(package, "vocal_wav", None)
        if vocal_wav is None or (schema_version == REMOTE_CHUNK_SCHEMA_VERSION_V2 and "vocal_wav" not in warped):
            return None
        fmt = AudioFormat()
        started = time.perf_counter()
        vocals = ca._decode_pcm16_mono_wav(vocal_wav, request.expected_frame_count)
        frame_counts = tuple(ca.bar_frame_count(b.bar, tempo, fmt) for b in request.bars)
        full = ca._resample_to_output(vocals, fmt, sum(frame_counts))
        offset = 0
        for bar_request, frames in zip(request.bars, frame_counts):
            mixed = np.zeros((frames, fmt.channels), dtype=np.float32)
            mixed[:] = full[offset : offset + frames]
            drum = drums.render(bar_request.flow_template, tempo, fmt, bar_request.bar)
            ca.mix_at(mixed, drum, 0, 0.5)
            ca.limit_peak(mixed)
            offset += frames
        return (time.perf_counter() - started) * 1000.0

    total = args.warmups + args.chunks
    with (args.out / "records.jsonl").open("w", buffering=1) as ledger:
        for index in range(total):
            warmup = index < args.warmups
            start_bar = (index * 2) % scenario.total_bars
            bars = []
            for bar in (start_bar, start_bar + 1):
                segment = scenario.segment_for_bar(bar)
                bars.append(
                    RemoteRapBarRequest(bar, segment.topic, BUILTIN_TEMPLATES.get(segment.template_id))
                )
            request = RemoteRapChunkRequest.create(
                session_id=session,
                chunk_index=index,
                bars=tuple(bars),
                tempo_bpm=tempo.bpm,
                remaining_budget_ms=args.budget_ms,
                policy=policy,
                context_lines=(
                    replay_context[index] if replay_context is not None else tuple(context[-4:])
                ),
                seed=args.seed + index,
                schema_version=schema_version,
            )
            before = vllm_counters(args.vllm_url)
            transfer.clear()
            captured.clear()
            warped.clear()
            started = time.perf_counter()
            error = None
            try:
                prepared = strategy.prepare(
                    request, deadline_monotonic=time.monotonic() + args.budget_ms / 1000.0
                )
            except Exception as exc:  # record and continue
                prepared = None
                error = f"{type(exc).__name__}: {exc}"
            wall_ms = (time.perf_counter() - started) * 1000.0
            after = vllm_counters(args.vllm_url)
            row: dict[str, object] = {
                "index": index,
                "warmup": warmup,
                "bars": [start_bar, start_bar + 1],
                "topic": [b.topic for b in bars],
                "wall_ms": wall_ms,
                "error": error,
                "seed": request.seed,
                "context_lines": list(request.context_lines),
                "budget_ms": args.budget_ms,
                "protocol": args.protocol,
                "vllm_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after},
            }
            manifest = transfer.get("manifest")
            timing = transfer.get("timing")
            if timing is not None:
                row["transfer"] = {
                    "client_prepare_ms": transfer.get("client_prepare_ms"),
                    "request_ms": timing.request_ms,
                    "first_byte_ms": timing.first_byte_ms,
                    "download_ms": timing.download_ms,
                    "response_bytes": timing.response_bytes,
                    "attempts": timing.attempts,
                }
            if manifest is not None:
                payload = manifest.to_payload()
                diagnostics = payload.get("diagnostics", {})
                row["server_stage_ms"] = diagnostics.get("stage_timings_ms")
                row["candidate_stats"] = diagnostics.get("candidate_stats")
                row["selected"] = [b.get("text") for b in payload.get("selected_bars", [])]
                context.extend(str(t) for t in row["selected"])
            row["client_package_decode_ms"] = captured.get("decode_ms")
            if "warp_ms" in warped:
                row["mac_local_warp_ms"] = warped["warp_ms"]
            if args.save_audio:
                row["audio_files"] = _save_audio(
                    args.out, index, captured.get("package"), prepared, warped.get("vocal_wav")
                )
            if prepared is not None:
                row["mac_prepare_ms"] = wall_ms - float(transfer.get("client_prepare_ms", 0.0))
            elif manifest is not None:
                row["mac_mix_probe_ms"] = mac_mix_probe(request, transfer)
            ledger.write(json.dumps(row, default=str, sort_keys=True) + "\n")
            stage = row.get("server_stage_ms") or {}
            print(
                json.dumps(
                    {
                        "i": index,
                        "warmup": warmup,
                        "wall": round(wall_ms),
                        "gen": round(stage.get("generation", 0)),
                        "eval": round(stage.get("evaluation", 0)),
                        "moss": round(stage.get("moss", 0)),
                        "mms": round(stage.get("aligner", 0)),
                        "r3": round(stage.get("warp", row.get("mac_local_warp_ms", 0))),
                        "srv_total": round(stage.get("total", 0)),
                        "err": error,
                    }
                ),
                flush=True,
            )
    client.close()

    rows = [json.loads(line) for line in (args.out / "records.jsonl").open()]
    measured = [r for r in rows if not r["warmup"] and r.get("server_stage_ms")]

    def dist(values):
        values = sorted(values)
        if not values:
            return None
        return {
            "median": statistics.median(values),
            "p90": values[max(0, int(0.9 * len(values)) - 1)],
            "max": values[-1],
            "min": values[0],
        }

    summary: dict[str, object] = {
        "n": len(measured),
        "failures": sum(1 for r in rows if not r["warmup"] and r["error"]),
        "wall_ms": dist(r["wall_ms"] for r in measured),
    }
    for key in ("generation", "evaluation", "moss", "aligner", "warp", "total"):
        summary[f"server_{key}_ms"] = dist(
            r["server_stage_ms"][key] for r in measured if key in r["server_stage_ms"]
        )
    summary["mac_local_warp_ms"] = dist(
        r["mac_local_warp_ms"] for r in measured if "mac_local_warp_ms" in r
    )
    summary["server_http_minus_orchestrator_ms"] = dist(
        r["transfer"]["request_ms"] + r["transfer"]["first_byte_ms"] - r["server_stage_ms"]["total"]
        for r in measured
    )
    summary["download_ms"] = dist(r["transfer"]["download_ms"] for r in measured)
    summary["client_prepare_minus_ttfb_ms"] = dist(
        r["transfer"]["client_prepare_ms"]
        - r["transfer"]["request_ms"]
        - r["transfer"]["first_byte_ms"]
        - r["transfer"]["download_ms"]
        for r in measured
    )
    summary["mac_validation_mix_ms"] = dist(r["mac_prepare_ms"] for r in measured if "mac_prepare_ms" in r)
    summary["mac_mix_probe_ms"] = dist(r["mac_mix_probe_ms"] for r in measured if r.get("mac_mix_probe_ms"))
    summary["client_package_decode_ms"] = dist(
        r["client_package_decode_ms"] for r in measured if r.get("client_package_decode_ms")
    )
    summary["rejected_by_mac_validation"] = sum(1 for r in measured if r["error"])
    summary["vllm_requests_per_chunk"] = dist(r["vllm_delta"].get("requests", 0) for r in measured)
    summary["vllm_generation_tokens_per_chunk"] = dist(
        r["vllm_delta"].get("generation_tokens", 0) for r in measured
    )
    summary["vllm_prompt_tokens_per_chunk"] = dist(r["vllm_delta"].get("prompt_tokens", 0) for r in measured)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def _load_replay_context(path: Path) -> dict[int, tuple[str, ...]]:
    contexts: dict[int, tuple[str, ...]] = {}
    for line in path.open():
        row = json.loads(line)
        if "context_lines" not in row:
            raise SystemExit(f"{path} has no context_lines; record it with this script first")
        contexts[int(row["index"])] = tuple(str(item) for item in row["context_lines"])
    return contexts


def _save_audio(
    out: Path, index: int, package: object, prepared: object, local_vocal_wav: bytes | None
) -> dict[str, str]:
    from scipy.io import wavfile

    files: dict[str, str] = {}
    vocal_wav = getattr(package, "vocal_wav", None)
    if local_vocal_wav is not None:
        # v2: the package carries the raw MOSS phrase; the vocal is the Mac warp.
        if vocal_wav:
            source = out / f"chunk-{index:03d}-source.wav"
            source.write_bytes(vocal_wav)
            files["source"] = source.name
        vocal_wav = local_vocal_wav
    if vocal_wav:
        vocals = out / f"chunk-{index:03d}-vocals.wav"
        vocals.write_bytes(vocal_wav)
        files["vocals"] = vocals.name
    bars = getattr(prepared, "bars", None)
    if bars:
        # Mac-side bars are interleaved float32 PCM; write an IEEE-float WAV.
        fmt = bars[0].audio.format
        samples = np.concatenate(
            [np.frombuffer(bar.audio.data, dtype=np.float32).reshape(-1, fmt.channels) for bar in bars]
        )
        mix = out / f"chunk-{index:03d}-mix.wav"
        wavfile.write(mix, fmt.sample_rate_hz, samples)
        files["mix"] = mix.name
    return files


if __name__ == "__main__":
    main()

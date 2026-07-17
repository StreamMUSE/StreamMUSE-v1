"""Frozen acc-only triangle-listening contract for the July 2026 pilot.

This module is intentionally independent from the historical 24-clip quality
rating contract in :mod:`melody_robustness`.  It owns the deterministic v2
selection and the append-only, resumable response workflow.  Rendering stays
in ``scripts/prepare_robustness_listening.py`` because it depends on the local
FluidSynth executable.
"""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from streammuse.experiments.melody_robustness import (
    SEEDS,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    validate_staged_input_manifest,
    write_canonical_json,
)
from streammuse.experiments.robustness_metrics import Roll


TRIANGLE_SELECTION_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_selection.v2"
)
TRIANGLE_LEDGER_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_response.v2"
)
TRIANGLE_SITTING_LEDGER_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_sitting_event.v2"
)
TRIANGLE_SNAPSHOT_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_snapshot.v2"
)
TRIANGLE_UNBLINDED_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_unblinded.v2"
)
TRIANGLE_LISTENING_ATTEMPT_ID = "listening-attempt-001"
TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_retry_authorization.v2"
)
TRIANGLE_RETRY_LINEAGE_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_retry_lineage.v2"
)
TRIANGLE_RETRY_SEED_DOMAIN = "streammuse-triangle-retry-v2"

TRIANGLE_TRIAL_COUNT = 95
TRIANGLE_PRACTICE_COUNT = 3
TRIANGLE_PRESENTATION_COUNT = 285
TRIANGLE_CLIP_SECONDS = 8
TRIANGLE_CLIP_BEATS = 16
TRIANGLE_RENDER_BPM = 120
TRIANGLE_RENDER_SAMPLE_RATE = 44_100
TRIANGLE_SYNTH_GAIN = 0.5
TRIANGLE_GAIN_POLICY = "common_pair_gain_with_true_peak_protection_only"
TRIANGLE_RENDERER_SCHEMA_VERSION = (
    "streammuse.melody_robustness.listening_triangle_renderer.v2"
)
TRIANGLE_TRUE_PEAK_LIMIT_DBTP = -0.1
TRIANGLE_TRUE_PEAK_IMPLEMENTATION = "scipy_resample_poly_4x_kaiser_8.6"
TRIANGLE_COVERAGE_COLLAPSE_MAX_RATIO = 0.25
TRIANGLE_COVERAGE_REFERENCE_MIN_RATIO = 0.75
TRIANGLE_PROMPT = (
    "三段都只包含模型生成的伴奏。请选择听起来不同的一段；若确实无法听出差异，"
    "选择 ‘no audible difference’。不要评价哪段更好。"
)

TRIANGLE_PATTERNS = ("AAB", "ABA", "BAA", "BBA", "BAB", "ABB")
TRIANGLE_CHOICES = ("1", "2", "3", "no_difference")
TRIANGLE_TAGS = (
    "pitch_harmony",
    "rhythm_timing",
    "density",
    "texture_register",
    "silence_coverage",
    "other",
)
PRIMARY_CONDITIONS = ("pitch", "onset", "both")
BLOCK_COUNTS = {
    "medium_primary": 60,
    "high_exploratory": 10,
    "sham_sampling_baseline": 5,
    "identity_catch": 6,
    "known_different_control": 6,
    "exact_repeat": 8,
}
QC_THRESHOLDS = {
    "identity_correct": 5,
    "identity_total": 6,
    "known_different_correct": 5,
    "known_different_total": 6,
    "repeat_consistent": 6,
    "repeat_total": 8,
}

KNOWN_DIFFERENT_RECIPE = {
    "name": "fixed_four_bar_scale_v1",
    "pitches": [48, 55, 60, 64, 67, 72, 76, 79],
    "onset_step_model_ticks": 8,
    "duration_model_ticks": 6,
    "velocity": 96,
    "fail_if_equal_to_comparator": True,
}


def formal_listening_selectors(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the canonical unique formal-output selectors used by listening.

    A triangle source also contains selection-only metadata such as ``kind``.
    Campaign readiness, however, is bound to the seven fields that identify a
    generated accompaniment in the formal schedule. Keeping that projection
    here prevents the campaign auditor and report builder from hashing subtly
    different selector objects.
    """

    schema = selection.get("schema_version")
    selectors: list[dict[str, Any]] = []
    if schema == TRIANGLE_SELECTION_SCHEMA_VERSION:
        trials = selection.get("trials", selection.get("scored_trials"))
        if not isinstance(trials, list):
            raise ValueError("triangle selection lacks scored trials")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise ValueError("triangle listening trial is not an object")
            sources = trial.get("sources")
            if not isinstance(sources, Mapping):
                raise ValueError("triangle listening trial lacks sources")
            for side in ("a", "b"):
                source = sources.get(side)
                if not isinstance(source, Mapping):
                    raise ValueError(f"triangle listening trial lacks source {side}")
                if source.get("kind") != "formal":
                    continue
                selectors.append(
                    {
                        "formal_pipeline": source.get("formal_pipeline"),
                        "source_artifact": source.get("source_artifact"),
                        "presentation": source.get("presentation"),
                        "song": source.get("song"),
                        "condition": source.get("condition"),
                        "perturb_seed": source.get("perturb_seed"),
                        "sample_seed": source.get("sample_seed"),
                    }
                )
    else:
        clips = selection.get("clips")
        if not isinstance(clips, list):
            raise ValueError("listening selection lacks clips")
        for clip in clips:
            if not isinstance(clip, Mapping):
                raise ValueError("listening selection clip is not an object")
            pipeline = clip.get("pipeline")
            if pipeline not in {"rt_theoretical", "rt_combined"}:
                continue
            selectors.append(
                {
                    "formal_pipeline": "rt",
                    "source_artifact": (
                        "theoretical_model"
                        if pipeline == "rt_theoretical"
                        else "combined"
                    ),
                    "presentation": clip.get("block"),
                    "song": clip.get("song"),
                    "condition": clip.get("condition"),
                    "perturb_seed": clip.get("perturb_seed"),
                    "sample_seed": clip.get("sample_seed"),
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for selector in selectors:
        if selector["formal_pipeline"] != "rt":
            raise ValueError("listening source must use the formal RT pipeline")
        if selector["source_artifact"] not in {"theoretical_model", "combined"}:
            raise ValueError("listening source artifact is unsupported")
        key = json.dumps(selector, sort_keys=True, separators=(",", ":"))
        unique[key] = selector
    return [unique[key] for key in sorted(unique)]


def triangle_listening_attempt_id(number: int) -> str:
    if isinstance(number, bool) or not isinstance(number, int) or number < 1 or number > 999:
        raise ValueError("listening attempt number must be an integer in 1..999")
    return f"listening-attempt-{number:03d}"


def triangle_listening_attempt_number(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("listening attempt ID must be a string")
    matched = re.fullmatch(r"listening-attempt-([0-9]{3})", value)
    if matched is None or int(matched.group(1)) < 1:
        raise ValueError("listening attempt ID must match listening-attempt-NNN")
    return int(matched.group(1))


def derive_triangle_retry_seed(
    *, base_blind_seed: int, attempt_number: int
) -> tuple[int, str]:
    """Return the frozen domain-separated seed for attempt 2 and later."""

    if attempt_number < 2:
        raise ValueError("retry seed derivation requires attempt_number >= 2")
    material = (
        f"{TRIANGLE_RETRY_SEED_DOMAIN}\0{int(base_blind_seed)}\0{attempt_number}"
    ).encode("ascii")
    digest = hashlib.sha256(material).hexdigest()
    # A non-negative 63-bit integer works identically in Python's Random and
    # remains exactly representable by JSON implementations used by the tools.
    seed = int(digest[:16], 16) & ((1 << 63) - 1)
    return seed, digest


def validate_triangle_renderer_identity(
    identity: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Strictly validate the renderer record bound into C5."""

    expected_keys = {
        "schema_version",
        "fluidsynth",
        "soundfont",
        "midi_program",
        "midi_bank",
        "render_bpm",
        "sample_rate",
        "bit_depth",
        "synth_gain",
        "gain_policy",
        "true_peak_limit_dbtp",
        "true_peak_implementation",
        "command_template",
    }
    if set(identity) != expected_keys:
        raise ValueError("triangle renderer identity has missing or unexpected fields")
    expected_values = {
        "schema_version": TRIANGLE_RENDERER_SCHEMA_VERSION,
        "midi_program": 0,
        "midi_bank": 0,
        "render_bpm": TRIANGLE_RENDER_BPM,
        "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
        "bit_depth": 16,
        "synth_gain": TRIANGLE_SYNTH_GAIN,
        "gain_policy": TRIANGLE_GAIN_POLICY,
        "true_peak_limit_dbtp": TRIANGLE_TRUE_PEAK_LIMIT_DBTP,
        "true_peak_implementation": TRIANGLE_TRUE_PEAK_IMPLEMENTATION,
        "command_template": [
            "{fluidsynth_binary}",
            "-ni",
            "{soundfont}",
            "{input_midi}",
            "-F",
            "{output_wav}",
            "-r",
            str(TRIANGLE_RENDER_SAMPLE_RATE),
            "-g",
            str(TRIANGLE_SYNTH_GAIN),
        ],
    }
    for field, expected in expected_values.items():
        if identity.get(field) != expected:
            raise ValueError(f"triangle renderer {field} drifted from frozen contract")
    synth = identity.get("fluidsynth")
    soundfont = identity.get("soundfont")
    if not isinstance(synth, Mapping) or set(synth) != {
        "binary_path",
        "binary_sha256",
        "ld_library_path",
        "library_files",
        "version",
    }:
        raise ValueError("triangle renderer FluidSynth record is malformed")
    if not isinstance(soundfont, Mapping) or set(soundfont) != {"path", "size", "sha256"}:
        raise ValueError("triangle renderer soundfont record is malformed")
    if not isinstance(synth.get("version"), str) or not synth["version"]:
        raise ValueError("triangle renderer FluidSynth version is missing")
    if synth.get("ld_library_path") is not None and not isinstance(
        synth.get("ld_library_path"), str
    ):
        raise ValueError("triangle renderer LD_LIBRARY_PATH is malformed")
    records = [
        ("FluidSynth binary", synth),
        ("soundfont", soundfont),
    ]
    libraries = synth.get("library_files")
    if not isinstance(libraries, list):
        raise ValueError("triangle renderer library_files must be a list")
    for index, record in enumerate(libraries):
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            raise ValueError(f"triangle renderer library record {index} is malformed")
        records.append((f"library {index}", record))
    for label, record in records:
        path_key = "binary_path" if label == "FluidSynth binary" else "path"
        sha_key = "binary_sha256" if label == "FluidSynth binary" else "sha256"
        raw_path = record.get(path_key)
        digest = record.get(sha_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"triangle renderer {label} path is missing")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"triangle renderer {label} sha256 is invalid")
        path = Path(raw_path).resolve()
        if label != "FluidSynth binary":
            size = record.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"triangle renderer {label} size is invalid")
        if verify_files:
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"triangle renderer {label} file/hash mismatch")
            if label != "FluidSynth binary" and path.stat().st_size != record["size"]:
                raise ValueError(f"triangle renderer {label} size mismatch")
    return dict(identity)


def _song(entry: Mapping[str, Any]) -> str:
    return str(entry.get("song", entry.get("source_stem")))


def _formal_source(
    *, song: str, condition: str, perturb_seed: int | None, sample_seed: int
) -> dict[str, Any]:
    return {
        "kind": "formal",
        "formal_pipeline": "rt",
        "source_artifact": "theoretical_model",
        "presentation": "acc_solo",
        "song": song,
        "condition": condition,
        "perturb_seed": perturb_seed,
        "sample_seed": int(sample_seed),
    }


def _synthetic_source(*, song: str, sample_seed: int) -> dict[str, Any]:
    return {
        "kind": "synthetic_control",
        "source_artifact": "known_different_scale_v1",
        "presentation": "acc_solo",
        "song": song,
        "condition": "known_different_control",
        "perturb_seed": None,
        "sample_seed": int(sample_seed),
        "recipe": {
            **KNOWN_DIFFERENT_RECIPE,
            "pitches": list(KNOWN_DIFFERENT_RECIPE["pitches"]),
        },
    }


def build_triangle_control_roll(source: Mapping[str, Any]) -> tuple[Roll, int]:
    """Materialize a frozen synthetic/practice selector without hidden defaults.

    In particular, the known-different MIDI velocity is part of the selection
    recipe. Returning it beside the binary roll prevents rendering and the
    objective assay from silently replacing it with a writer default.
    """

    artifact = source.get("source_artifact")
    if source.get("kind") == "synthetic_control":
        expected_recipe = {
            **KNOWN_DIFFERENT_RECIPE,
            "pitches": list(KNOWN_DIFFERENT_RECIPE["pitches"]),
        }
        if artifact != "known_different_scale_v1":
            raise ValueError("unsupported known-different source artifact")
        if source.get("recipe") != expected_recipe:
            raise ValueError("known-different selector differs from its frozen recipe")
        pitches = [int(value) for value in expected_recipe["pitches"]]
        step = int(expected_recipe["onset_step_model_ticks"])
        duration = int(expected_recipe["duration_model_ticks"])
        velocity = int(expected_recipe["velocity"])
    elif source.get("kind") == "practice_tone" and artifact == "practice_tone_a":
        pitches, step, duration, velocity = [60, 64, 67, 72], 16, 12, 80
    elif source.get("kind") == "practice_tone" and artifact == "practice_tone_b":
        pitches, step, duration, velocity = [61, 66, 70, 73], 16, 12, 80
    else:
        raise ValueError(f"unsupported triangle control selector: {dict(source)}")

    sustain: set[tuple[int, int]] = set()
    onsets: set[tuple[int, int]] = set()
    for index, pitch in enumerate(pitches):
        start = index * step
        if start >= TRIANGLE_CLIP_BEATS * 4:
            break
        end = min(start + duration, TRIANGLE_CLIP_BEATS * 4)
        onsets.add((start, pitch))
        sustain.update((tick, pitch) for tick in range(start, end))
    return (
        Roll(
            end_tick=TRIANGLE_CLIP_BEATS * 4,
            sustain=frozenset(sustain),
            onsets=frozenset(onsets),
        ),
        velocity,
    )


def _trial(
    semantic_id: str,
    block: str,
    source_a: Mapping[str, Any],
    source_b: Mapping[str, Any],
    *,
    condition: str | None = None,
    repeat_of: str | None = None,
) -> dict[str, Any]:
    value = {
        "semantic_id": semantic_id,
        "block": block,
        "condition": condition,
        "sources": {"a": dict(source_a), "b": dict(source_b)},
        "repeat_of": repeat_of,
    }
    return value


def _pattern_odd_position(pattern: str) -> int:
    counts = Counter(pattern)
    odd_label = "A" if counts["A"] == 1 else "B"
    return pattern.index(odd_label) + 1


def _pattern_duplicated_label(pattern: str) -> str:
    counts = Counter(pattern)
    return "A" if counts["A"] == 2 else "B"


def _assign_patterns(trials: list[dict[str, Any]], seed: int) -> None:
    assignments: dict[str, str] = {}

    def assign_pool(
        selected: Sequence[dict[str, Any]], pool: Sequence[str], domain: int
    ) -> None:
        if len(selected) != len(pool):
            raise AssertionError("pattern pool size differs from frozen trial stratum")
        shuffled = list(pool)
        random.Random(seed ^ domain).shuffle(shuffled)
        for trial, pattern in zip(selected, shuffled, strict=True):
            assignments[str(trial["semantic_id"])] = pattern

    # Every primary condition has exact 10/10 sham-vs-treatment duplication;
    # odd positions are 6/7/7 and every pattern occurs three or four times.
    primary_pool = ["AAB"] * 4 + ["ABA"] * 3 + ["BAA"] * 3
    primary_pool += ["BBA"] * 3 + ["BAB"] * 4 + ["ABB"] * 3
    for condition_index, condition in enumerate(PRIMARY_CONDITIONS):
        selected = [
            trial
            for trial in trials
            if trial["block"] == "medium_primary" and trial["condition"] == condition
        ]
        assign_pool(selected, primary_pool, 0x1000 + condition_index)
    high = [trial for trial in trials if trial["block"] == "high_exploratory"]
    assign_pool(
        high,
        ["AAB"] * 2
        + ["ABA"] * 2
        + ["BAA"]
        + ["BBA"] * 2
        + ["BAB"]
        + ["ABB"] * 2,
        0x2000,
    )
    baseline = [
        trial for trial in trials if trial["block"] == "sham_sampling_baseline"
    ]
    assign_pool(baseline, ["AAB", "ABA", "BAA", "BBA", "BAB"], 0x3000)
    known = [
        trial for trial in trials if trial["block"] == "known_different_control"
    ]
    assign_pool(known, TRIANGLE_PATTERNS, 0x4000)

    by_semantic: dict[str, dict[str, Any]] = {}
    for trial_index, trial in enumerate(trials):
        if trial["block"] == "identity_catch":
            pattern = "AAA"
            odd_position: int | None = None
            correct_choice = "no_difference"
        elif trial["block"] == "exact_repeat":
            original = by_semantic[str(trial["repeat_of"])]
            original_pattern = str(original["presentation_pattern"])
            duplicate = _pattern_duplicated_label(original_pattern)
            candidates = [
                value
                for value in TRIANGLE_PATTERNS
                if value != original_pattern
                and _pattern_duplicated_label(value) == duplicate
            ]
            # Preserve the underlying odd source while moving its position.
            candidates = [
                value
                for value in candidates
                if _pattern_odd_position(value) != _pattern_odd_position(original_pattern)
            ]
            pattern = candidates[trial_index % len(candidates)]
            odd_position = _pattern_odd_position(pattern)
            correct_choice = str(odd_position)
        else:
            pattern = assignments[str(trial["semantic_id"])]
            odd_position = _pattern_odd_position(pattern)
            correct_choice = str(odd_position)
        trial["presentation_pattern"] = pattern
        trial["duplicated_source"] = (
            "a" if pattern == "AAA" or _pattern_duplicated_label(pattern) == "A" else "b"
        )
        trial["correct_choice"] = correct_choice
        trial["odd_position"] = odd_position
        trial["global_order_index"] = trial_index
        trial["question_id"] = f"Q{trial_index + 1:03d}"
        by_semantic[str(trial["semantic_id"])] = trial


def _practice_trials() -> list[dict[str, Any]]:
    # Practice never exposes or reuses a formal output.  The renderer creates
    # the two small deterministic tone examples directly from these selectors.
    tone_a = {
        "kind": "practice_tone",
        "source_artifact": "practice_tone_a",
        "presentation": "acc_solo",
    }
    tone_b = {
        "kind": "practice_tone",
        "source_artifact": "practice_tone_b",
        "presentation": "acc_solo",
    }
    specs = [
        ("P001", "AAA", tone_a, tone_a, "no_difference"),
        ("P002", "AAB", tone_a, tone_b, "3"),
        ("P003", "BAB", tone_a, tone_b, "2"),
    ]
    return [
        {
            "practice_id": practice_id,
            "presentation_pattern": pattern,
            "sources": {"a": dict(source_a), "b": dict(source_b)},
            "correct_choice": correct,
            "feedback_allowed": True,
            "scored": False,
        }
        for practice_id, pattern, source_a, source_b, correct in specs
    ]


def _analysis_horizons(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    grouped: dict[str, set[int]] = defaultdict(set)
    for entry in entries:
        raw = entry.get("analysis_end_tick")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{_song(entry)}: invalid analysis_end_tick")
        grouped[_song(entry)].add(raw)
    inconsistent = {song: values for song, values in grouped.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"analysis horizon differs across variants: {inconsistent}")
    return {song: next(iter(values)) for song, values in grouped.items()}


def build_triangle_selection_manifest(
    input_manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    excerpt_starts: Mapping[str, int],
    blind_order_seed: int = int(SEEDS["blind_order"]),
    verify_files: bool = True,
) -> dict[str, Any]:
    """Build the exact 95-trial, five-prefix-chunk selection.

    Each 19-trial chunk contains 12 medium trials (four per condition), two
    high-dose trials, one sham-sampling trial and four QC/repeat trials.  This
    preserves useful balance even when the listener stops early, without
    freezing artificial sitting boundaries.
    """

    resolved_manifest = Path(manifest_path).resolve()
    if int(blind_order_seed) != int(SEEDS["blind_order"]):
        raise ValueError("blind-order seed must match the frozen campaign contract")
    entries = validate_staged_input_manifest(
        input_manifest,
        manifest_path=resolved_manifest,
        verify_files=verify_files,
    )
    songs = sorted({_song(entry) for entry in entries})
    if len(songs) != 5:
        raise ValueError(f"triangle selection requires exactly five songs, got {songs}")
    if set(map(str, excerpt_starts)) != set(songs):
        raise ValueError("excerpt starts must name all five songs exactly")
    horizons = _analysis_horizons(entries)
    starts: dict[str, int] = {}
    for song in songs:
        start = excerpt_starts[song]
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError(f"{song}: excerpt start must be a non-negative integer")
        if (start + TRIANGLE_CLIP_BEATS) * 4 > horizons[song]:
            raise ValueError(f"{song}: 16-beat triangle excerpt exceeds analysis horizon")
        starts[song] = int(start)

    pseeds = [int(seed) for seed in SEEDS["perturb"]]
    sseeds = [int(seed) for seed in SEEDS["sample"]]
    seed_pairs = [(pseed, sseed) for pseed in pseeds for sseed in sseeds]

    medium_chunks: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for chunk_index in range(5):
        for pair_index, (pseed, sseed) in enumerate(seed_pairs):
            for condition_index, condition in enumerate(PRIMARY_CONDITIONS):
                song = songs[(chunk_index + condition_index + 2 * pair_index) % 5]
                sham = _formal_source(
                    song=song,
                    condition="sham",
                    perturb_seed=None,
                    sample_seed=sseed,
                )
                treatment = _formal_source(
                    song=song,
                    condition=condition,
                    perturb_seed=pseed,
                    sample_seed=sseed,
                )
                semantic_id = f"M:{song}:{condition}:p{pseed}:s{sseed}"
                medium_chunks[chunk_index].append(
                    _trial(
                        semantic_id,
                        "medium_primary",
                        sham,
                        treatment,
                        condition=condition,
                    )
                )

    high_chunks: list[list[dict[str, Any]]] = []
    baseline: list[dict[str, Any]] = []
    for chunk_index in range(5):
        high_rows = []
        # Across five chunks every song occurs once with each sample seed.
        for song_offset, sseed in ((0, sseeds[0]), (3, sseeds[1])):
            song = songs[(chunk_index + song_offset) % 5]
            high_rows.append(
                _trial(
                    f"H:{song}:s{sseed}",
                    "high_exploratory",
                    _formal_source(
                        song=song,
                        condition="sham",
                        perturb_seed=None,
                        sample_seed=sseed,
                    ),
                    _formal_source(
                        song=song,
                        condition="high",
                        perturb_seed=int(SEEDS["high_perturb"]),
                        sample_seed=sseed,
                    ),
                    condition="high",
                )
            )
        high_chunks.append(high_rows)
        baseline_song = songs[(chunk_index + 4) % 5]
        baseline.append(
            _trial(
                f"S:{baseline_song}",
                "sham_sampling_baseline",
                _formal_source(
                    song=baseline_song,
                    condition="sham",
                    perturb_seed=None,
                    sample_seed=sseeds[0],
                ),
                _formal_source(
                    song=baseline_song,
                    condition="sham",
                    perturb_seed=None,
                    sample_seed=sseeds[1],
                ),
                condition="sham_sampling",
            )
        )

    def identity_trial(index: int, song: str) -> dict[str, Any]:
        sseed = sseeds[index % 2]
        source = _formal_source(
            song=song,
            condition="sham",
            perturb_seed=None,
            sample_seed=sseed,
        )
        return _trial(
            f"I:{index + 1}:{song}:s{sseed}",
            "identity_catch",
            source,
            source,
            condition="identity",
        )

    def known_trial(index: int, song: str) -> dict[str, Any]:
        sseed = sseeds[index % 2]
        source = _formal_source(
            song=song,
            condition="sham",
            perturb_seed=None,
            sample_seed=sseed,
        )
        return _trial(
            f"K:{index + 1}:{song}:s{sseed}",
            "known_different_control",
            source,
            _synthetic_source(song=song, sample_seed=sseed),
            condition="known_different",
        )

    chunk_extras: list[list[dict[str, Any]]] = [
        [
            identity_trial(0, songs[0]),
            identity_trial(1, songs[2]),
            known_trial(0, songs[3]),
            known_trial(1, songs[4]),
        ]
    ]
    for chunk_index in range(1, 5):
        repeats = []
        previous_medium = medium_chunks[chunk_index - 1]
        repeat_songs = [songs[chunk_index], songs[(chunk_index + 2) % 5]]
        repeat_sources = []
        for repeat_song in repeat_songs:
            source = next(
                trial
                for trial in previous_medium
                if trial["sources"]["a"]["song"] == repeat_song
                and trial not in repeat_sources
            )
            repeat_sources.append(source)
        for repeat_index, source in enumerate(repeat_sources):
            repeated = _trial(
                f"R:{chunk_index}:{repeat_index + 1}:{source['semantic_id']}",
                "exact_repeat",
                source["sources"]["a"],
                source["sources"]["b"],
                condition=str(source["condition"]),
                repeat_of=str(source["semantic_id"]),
            )
            repeats.append(repeated)
        chunk_extras.append(
            [
                identity_trial(chunk_index + 1, songs[(chunk_index + 3) % 5]),
                known_trial(chunk_index + 1, songs[(chunk_index + 4) % 5]),
                *repeats,
            ]
        )

    # M/E/M/H/M/M/E/M/B/M/E/M/H/M/M/E/M/M/M: exactly 19 rows.
    layout = "MEMHMMEMBMEMHMMEMMM"
    ordered: list[dict[str, Any]] = []
    for chunk_index in range(5):
        queues = {
            "M": iter(medium_chunks[chunk_index]),
            "H": iter(high_chunks[chunk_index]),
            "B": iter([baseline[chunk_index]]),
            "E": iter(chunk_extras[chunk_index]),
        }
        chunk = [next(queues[kind]) for kind in layout]
        if len(chunk) != 19:
            raise AssertionError("internal prefix chunk construction drift")
        song_counts = Counter(str(trial["sources"]["a"]["song"]) for trial in chunk)
        if sorted(song_counts.values()) != [3, 4, 4, 4, 4]:
            raise AssertionError(f"internal prefix song balance drift: {song_counts}")
        ordered.extend(chunk)

    _assign_patterns(ordered, int(blind_order_seed))
    semantic_positions = {
        str(trial["semantic_id"]): int(trial["global_order_index"]) for trial in ordered
    }
    for trial in ordered:
        repeat_of = trial.get("repeat_of")
        if repeat_of is not None:
            trial["repeat_distance"] = int(trial["global_order_index"]) - semantic_positions[
                str(repeat_of)
            ]
            if trial["repeat_distance"] < 8:
                raise AssertionError("repeat must be separated by at least eight questions")
    for trial in ordered:
        song = str(trial["sources"]["a"].get("song", trial["sources"]["b"].get("song")))
        trial["excerpt"] = {
            "start_model_beat": starts[song],
            "end_model_beat": starts[song] + TRIANGLE_CLIP_BEATS,
            "start_model_tick": starts[song] * 4,
            "end_model_tick": (starts[song] + TRIANGLE_CLIP_BEATS) * 4,
            "analysis_end_tick": horizons[song],
        }

    counts = Counter(str(trial["block"]) for trial in ordered)
    if dict(counts) != BLOCK_COUNTS:
        raise AssertionError(f"internal triangle block-count drift: {counts}")
    payload = {
        "schema_version": TRIANGLE_SELECTION_SCHEMA_VERSION,
        "frozen_before_formal": True,
        "retry_reblind_after_formal_without_semantic_change": False,
        "supersedes_quality_selection_v1": True,
        "input_manifest_path": str(resolved_manifest),
        "input_manifest_sha256": file_sha256(resolved_manifest),
        "analysis_horizons_ticks": horizons,
        "default_excerpt_model_beats": [16, 32],
        "blind_order_seed": int(blind_order_seed),
        "base_blind_order_seed": int(blind_order_seed),
        "effective_blind_order_seed": int(blind_order_seed),
        "effective_blind_order_seed_sha256": None,
        "listening_attempt_number": 1,
        "retry_lineage": None,
        "retry_lineage_sha256": None,
        "trial_count": TRIANGLE_TRIAL_COUNT,
        "practice_count": TRIANGLE_PRACTICE_COUNT,
        "presentation_count": TRIANGLE_PRESENTATION_COUNT,
        "prefix_chunk_size": 19,
        "prefix_chunk_count": 5,
        "block_counts": dict(BLOCK_COUNTS),
        "primary_question": (
            "same song/excerpt/checkpoint/sample seed and RT-theoretical pipeline: "
            "is generated accompaniment discriminable after changing melody condition?"
        ),
        "question_id": "generated_acc_triangle_discrimination",
        "semantics_version": 2,
        "question_prompt": TRIANGLE_PROMPT,
        "render_policy": {
            "presentation": "acc_solo",
            "formal_pipeline": "rt",
            "source_artifact": "theoretical_model",
            "render_bpm": TRIANGLE_RENDER_BPM,
            "clip_seconds": TRIANGLE_CLIP_SECONDS,
            "clip_beats": TRIANGLE_CLIP_BEATS,
            "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
            "bit_depth": 16,
            "synth_gain": TRIANGLE_SYNTH_GAIN,
            "gain_policy": TRIANGLE_GAIN_POLICY,
            "normalization": "none",
            "pre_roll": "load_active_sustain_at_frozen_window_start",
            "crop": "half_open_frozen_window_then_exact_8_seconds",
            "fade": "none",
            "pad": "trailing_literal_silence",
        },
        "source_policy": {
            "attempt": "latest immutable content-valid formal attempt; do not substitute",
            "operational_invalid": "retain_and_report",
            "expected_empty": "retain_as_literal_silence_and_report",
            "nonempty_rendered_silent": "package_failure",
        },
        "coverage_policy": {
            "metric": "fraction_of_excerpt_model_ticks_with_any_sustain",
            "collapse_max_ratio": TRIANGLE_COVERAGE_COLLAPSE_MAX_RATIO,
            "reference_min_ratio": TRIANGLE_COVERAGE_REFERENCE_MIN_RATIO,
            "one_sided_empty_is_coverage_driven": True,
            "sensitivity_view_excludes_but_never_replaces_main_rows": True,
        },
        "response_schema": {
            "odd_choice": list(TRIANGLE_CHOICES),
            "confidence": [1, 2, 3, 4, 5],
            "difference_tags": list(TRIANGLE_TAGS),
            "optional_note": True,
            "play_counts": "three non-negative integers with at least one play each",
            "response_time_ms": "non-negative integer",
            "sitting_id": "opaque non-empty string",
            "persistence": "append_only_sha256_hash_chain_and_atomic_progress",
        },
        "collection_policy": {
            "flexible_sittings": True,
            "minimum_responses_for_snapshot": 1,
            "maximum_responses": TRIANGLE_TRIAL_COUNT,
            "resume_from_next_unanswered": True,
            "semantic_partial_unblind_allowed": True,
            "post_partial_unblind": "retain_but_mark_exploratory",
        },
        "sitting_policy": {
            "append_only_hash_chain": True,
            "start_required_before_response": True,
            "start_fields": ["sitting_id", "device", "environment", "note"],
            "end_fields": ["sitting_id", "anomalies", "note"],
            "snapshot_seals_sitting_prefix": True,
            "sitting_boundaries_frozen": False,
        },
        "listening_attempt_id": TRIANGLE_LISTENING_ATTEMPT_ID,
        "listening_attempt_policy": {
            "initial_attempt_id": TRIANGLE_LISTENING_ATTEMPT_ID,
            "qc_failure": (
                "seal_and_preserve_failed_attempt_then_build_a_new_complete_"
                "domain_separated_blind_attempt"
            ),
            "retry_attempt_id_format": "listening-attempt-NNN",
            "retry_seed_derivation": (
                "sha256('streammuse-triangle-retry-v2' || base_blind_seed || "
                "attempt_number)"
            ),
            "failed_attempt_must_remain_reported": True,
            "answers_never_carry_between_attempts": True,
        },
        "qc_rules": dict(QC_THRESHOLDS),
        "decision_rules": {
            "condition_total": 20,
            "condition_correct_threshold": 12,
            "song_total": 4,
            "song_correct_threshold": 2,
            "songs_meeting_threshold": 4,
            "requires_full_qc_pass": True,
            "requires_all_condition_rows_pre_first_semantic_unblind": True,
            "partial_label": "partial — preregistered decision pending",
        },
        "blinding_policy": {
            "public_ids_only": True,
            "semantic_preview_before_first_unblind": False,
            "six_patterns": list(TRIANGLE_PATTERNS),
            "literal_duplicates": True,
            "sitting_boundaries_frozen": False,
        },
        "practice_trials": _practice_trials(),
        "trials": ordered,
    }
    return payload


def validate_triangle_selection_manifest(
    selection: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Exact-rebuild the complete v2 selection; count-only validation is forbidden."""

    if selection.get("schema_version") != TRIANGLE_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported triangle listening selection schema")
    attempt_number = triangle_listening_attempt_number(
        selection.get("listening_attempt_id")
    )
    if selection.get("listening_attempt_number") != attempt_number:
        raise ValueError("triangle selection attempt ID/number mismatch")
    if attempt_number > 1:
        return _validate_retry_selection_manifest(
            selection,
            input_manifest,
            manifest_path=manifest_path,
            verify_files=verify_files,
        )
    trials = selection.get("trials")
    if not isinstance(trials, list) or len(trials) != TRIANGLE_TRIAL_COUNT:
        raise ValueError("triangle selection must contain exactly 95 trials")
    starts_by_song: dict[str, set[int]] = defaultdict(set)
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("triangle trial must be an object")
        excerpt = trial.get("excerpt")
        sources = trial.get("sources")
        if not isinstance(excerpt, Mapping) or not isinstance(sources, Mapping):
            raise ValueError("triangle trial lacks excerpt or sources")
        source_a = sources.get("a")
        source_b = sources.get("b")
        if not isinstance(source_a, Mapping) or not isinstance(source_b, Mapping):
            raise ValueError("triangle trial source selectors must be objects")
        song = source_a.get("song", source_b.get("song"))
        start = excerpt.get("start_model_beat")
        if not isinstance(song, str) or isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("triangle trial has invalid song/excerpt selector")
        starts_by_song[song].add(start)
    if any(len(values) != 1 for values in starts_by_song.values()):
        raise ValueError("triangle selection must freeze one excerpt per song")
    try:
        expected = build_triangle_selection_manifest(
            input_manifest,
            manifest_path=manifest_path,
            excerpt_starts={song: next(iter(values)) for song, values in starts_by_song.items()},
            blind_order_seed=int(selection["blind_order_seed"]),
            verify_files=verify_files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid triangle selection contract: {exc}") from exc
    if dict(selection) != expected:
        raise ValueError("triangle listening selection manifest mismatch with exact rebuild")
    return expected


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_utc_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(dict(value)))


def _selection_from_package(package: Path) -> tuple[Path, dict[str, Any]]:
    render = _read_json_object(package / "render_manifest.json")
    selection_path = Path(str(render.get("selection_path", ""))).resolve()
    if not selection_path.is_file():
        raise FileNotFoundError("triangle render manifest selection path is missing")
    if file_sha256(selection_path) != render.get("selection_sha256"):
        raise ValueError("triangle selection hash differs from render manifest")
    selection = _read_json_object(selection_path)
    if selection.get("schema_version") != TRIANGLE_SELECTION_SCHEMA_VERSION:
        raise ValueError("package is not a triangle-listening v2 package")
    attempt_number = triangle_listening_attempt_number(
        selection.get("listening_attempt_id")
    )
    if selection.get("listening_attempt_number") != attempt_number:
        raise ValueError("package selection attempt ID/number mismatch")
    if render.get("listening_attempt_id") != selection.get("listening_attempt_id"):
        raise ValueError("package render/selection listening attempt IDs differ")
    return selection_path, selection


def _durable_append_line(path: Path, line: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    original_size = os.fstat(descriptor).st_size
    try:
        written = os.write(descriptor, line)
        if written != len(line):
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
            raise OSError(f"short append to {path.name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def validate_sitting_ledger(
    package_dir: str | Path,
) -> tuple[list[dict[str, Any]], str | None, dict[str, dict[str, Any]]]:
    """Validate append-only sitting start/end metadata for the current attempt."""

    package = Path(package_dir).resolve()
    _selection_path, selection = _selection_from_package(package)
    ledger = package / "blind" / "sitting_ledger.jsonl"
    if not ledger.exists():
        return [], None, {}
    expected_fields = {
        "schema_version",
        "listening_attempt_id",
        "sequence",
        "event",
        "sitting_id",
        "device",
        "environment",
        "note",
        "anomalies",
        "recorded_at",
        "previous_record_hash",
        "record_hash",
    }
    rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    previous: str | None = None
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"sitting ledger contains blank record at line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"sitting ledger is truncated/corrupt at line {line_number}"
                ) from exc
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError(f"sitting ledger fields mismatch at line {line_number}")
            body = {key: value for key, value in row.items() if key != "record_hash"}
            if row.get("record_hash") != canonical_sha256(body):
                raise ValueError(f"sitting ledger hash mismatch at line {line_number}")
            if row.get("schema_version") != TRIANGLE_SITTING_LEDGER_SCHEMA_VERSION:
                raise ValueError(f"sitting ledger schema mismatch at line {line_number}")
            if row.get("listening_attempt_id") != selection.get("listening_attempt_id"):
                raise ValueError(f"sitting ledger attempt mismatch at line {line_number}")
            if row.get("sequence") != line_number or row.get(
                "previous_record_hash"
            ) != previous:
                raise ValueError(f"sitting ledger chain mismatch at line {line_number}")
            sitting_id = row.get("sitting_id")
            if not isinstance(sitting_id, str) or not sitting_id or len(sitting_id) > 128:
                raise ValueError(f"invalid sitting ID at line {line_number}")
            if row.get("event") not in {"start", "end"}:
                raise ValueError(f"invalid sitting event at line {line_number}")
            for field in ("device", "environment", "note"):
                value = row.get(field)
                limit = 10_000 if field == "note" else 1_000
                if not isinstance(value, str) or len(value) > limit:
                    raise ValueError(f"invalid sitting {field} at line {line_number}")
            if row["event"] == "start" and (
                not row["device"].strip() or not row["environment"].strip()
            ):
                raise ValueError("sitting start requires device and environment")
            anomalies = row.get("anomalies")
            if (
                not isinstance(anomalies, list)
                or any(not isinstance(value, str) or not value for value in anomalies)
                or len(anomalies) != len(set(anomalies))
            ):
                raise ValueError(f"invalid sitting anomalies at line {line_number}")
            _validate_utc_timestamp(row.get("recorded_at"), label="sitting.recorded_at")
            state = states.get(sitting_id)
            if row["event"] == "start":
                if state is not None or anomalies:
                    raise ValueError("sitting start must be unique and cannot contain anomalies")
                states[sitting_id] = {"start": row, "end": None}
            else:
                if state is None or state["end"] is not None:
                    raise ValueError("sitting end requires one unmatched start")
                if (
                    row["device"] != state["start"]["device"]
                    or row["environment"] != state["start"]["environment"]
                ):
                    raise ValueError("sitting end device/environment differ from start")
                state["end"] = row
            rows.append(row)
            previous = str(row["record_hash"])
    return rows, previous, states


def _append_sitting_event_unlocked(
    package_dir: str | Path,
    *,
    event: str,
    sitting_id: str,
    device: str | None = None,
    environment: str | None = None,
    note: str = "",
    anomalies: Sequence[str] = (),
    recorded_at: str | None = None,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    _selection_path, selection = _selection_from_package(package)
    rows, previous, states = validate_sitting_ledger(package)
    if event not in {"start", "end"}:
        raise ValueError("sitting event must be start or end")
    if not isinstance(sitting_id, str) or not sitting_id or len(sitting_id) > 128:
        raise ValueError("sitting_id must be a non-empty string of at most 128 characters")
    if not isinstance(note, str) or len(note) > 10_000:
        raise ValueError("sitting note must be a string of at most 10000 characters")
    if isinstance(anomalies, (str, bytes)) or not isinstance(anomalies, Sequence):
        raise ValueError("sitting anomalies must be a sequence of strings")
    if len(anomalies) != len(set(anomalies)) or any(
        not isinstance(value, str) or not value for value in anomalies
    ):
        raise ValueError("sitting anomalies must be unique non-empty strings")
    if event == "start":
        if sitting_id in states:
            raise ValueError("sitting_id was already used in this attempt")
        if not isinstance(device, str) or not device.strip() or len(device) > 1_000:
            raise ValueError("sitting start requires a non-empty device description")
        if (
            not isinstance(environment, str)
            or not environment.strip()
            or len(environment) > 1_000
        ):
            raise ValueError("sitting start requires a non-empty environment description")
        if anomalies:
            raise ValueError("sitting start cannot contain anomalies")
    else:
        state = states.get(sitting_id)
        if state is None or state["end"] is not None:
            raise ValueError("sitting end requires one active start")
        device = str(state["start"]["device"])
        environment = str(state["start"]["environment"])
    timestamp = recorded_at or _utc_now()
    _validate_utc_timestamp(timestamp, label="sitting.recorded_at")
    body = {
        "schema_version": TRIANGLE_SITTING_LEDGER_SCHEMA_VERSION,
        "listening_attempt_id": selection["listening_attempt_id"],
        "sequence": len(rows) + 1,
        "event": event,
        "sitting_id": sitting_id,
        "device": device,
        "environment": environment,
        "note": note,
        "anomalies": list(anomalies),
        "recorded_at": timestamp,
        "previous_record_hash": previous,
    }
    record = {**body, "record_hash": canonical_sha256(body)}
    _durable_append_line(
        package / "blind" / "sitting_ledger.jsonl", canonical_json_bytes(record)
    )
    validated, head, _states = validate_sitting_ledger(package)
    if len(validated) != len(rows) + 1 or head != record["record_hash"]:
        raise RuntimeError("sitting ledger did not validate after durable append")
    _atomic_json(package / "blind" / "progress_state.json", progress_summary(package))
    return record


def append_sitting_event(
    package_dir: str | Path,
    *,
    event: str,
    sitting_id: str,
    device: str | None = None,
    environment: str | None = None,
    note: str = "",
    anomalies: Sequence[str] = (),
    recorded_at: str | None = None,
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    lock_path = package / "private" / "listening_state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _append_sitting_event_unlocked(
                package,
                event=event,
                sitting_id=sitting_id,
                device=device,
                environment=environment,
                note=note,
                anomalies=anomalies,
                recorded_at=recorded_at,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_response_ledger(
    package_dir: str | Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate every ledger record and return rows plus current head hash."""

    package = Path(package_dir).resolve()
    _selection_path, selection = _selection_from_package(package)
    _unblind_state(package)
    trial_ids = [str(trial["question_id"]) for trial in selection["trials"]]
    ledger = package / "blind" / "response_ledger.jsonl"
    if not ledger.exists():
        validate_sitting_ledger(package)
        return [], None
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    seen: set[str] = set()
    expected_record_fields = {
        "schema_version",
        "listening_attempt_id",
        "sequence",
        "trial_id",
        "odd_choice",
        "confidence_1_to_5",
        "difference_tags",
        "note",
        "play_counts",
        "response_time_ms",
        "sitting_id",
        "submitted_at",
        "blinding_phase",
        "previous_record_hash",
        "record_hash",
    }
    with ledger.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"response ledger contains blank record at line {line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"response ledger is truncated/corrupt at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"response ledger line {line_number} is not an object")
            if set(row) != expected_record_fields:
                raise ValueError(
                    f"response ledger fields mismatch at line {line_number}"
                )
            record_hash = row.get("record_hash")
            body = {key: value for key, value in row.items() if key != "record_hash"}
            if record_hash != canonical_sha256(body):
                raise ValueError(f"response ledger hash mismatch at line {line_number}")
            if body.get("schema_version") != TRIANGLE_LEDGER_SCHEMA_VERSION:
                raise ValueError(f"response ledger schema mismatch at line {line_number}")
            if body.get("listening_attempt_id") != selection.get(
                "listening_attempt_id"
            ):
                raise ValueError(
                    f"response ledger attempt mismatch at line {line_number}"
                )
            if body.get("sequence") != line_number:
                raise ValueError(f"response ledger sequence mismatch at line {line_number}")
            if body.get("previous_record_hash") != previous:
                raise ValueError(f"response ledger chain mismatch at line {line_number}")
            trial_id = body.get("trial_id")
            if trial_id != trial_ids[line_number - 1]:
                raise ValueError("responses must follow the frozen global order without skips")
            if not isinstance(trial_id, str) or trial_id in seen:
                raise ValueError("response ledger contains a duplicate/orphan trial")
            choice = body.get("odd_choice")
            confidence = body.get("confidence_1_to_5")
            tags = body.get("difference_tags")
            play_counts = body.get("play_counts")
            response_time = body.get("response_time_ms")
            if choice not in TRIANGLE_CHOICES:
                raise ValueError(f"invalid odd choice for {trial_id}")
            if isinstance(confidence, bool) or confidence not in {1, 2, 3, 4, 5}:
                raise ValueError(f"invalid confidence for {trial_id}")
            if (
                not isinstance(tags, list)
                or len(tags) != len(set(tags))
                or any(tag not in TRIANGLE_TAGS for tag in tags)
            ):
                raise ValueError(f"invalid difference tags for {trial_id}")
            if (
                not isinstance(play_counts, list)
                or len(play_counts) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in play_counts)
            ):
                raise ValueError(f"invalid play counts for {trial_id}")
            if isinstance(response_time, bool) or not isinstance(response_time, int) or response_time < 0:
                raise ValueError(f"invalid response time for {trial_id}")
            if not isinstance(body.get("sitting_id"), str) or not body["sitting_id"]:
                raise ValueError(f"invalid sitting ID for {trial_id}")
            if len(body["sitting_id"]) > 128:
                raise ValueError(f"sitting ID is too long for {trial_id}")
            if not isinstance(body.get("note"), str):
                raise ValueError(f"invalid response note for {trial_id}")
            if len(body["note"]) > 10_000:
                raise ValueError(f"response note is too long for {trial_id}")
            _validate_utc_timestamp(body.get("submitted_at"), label="submitted_at")
            if body.get("blinding_phase") not in {
                "pre_semantic_unblind",
                "post_partial_unblind_exploratory",
            }:
                raise ValueError(f"invalid blinding phase for {trial_id}")
            rows.append(row)
            seen.add(trial_id)
            previous = str(record_hash)
    if len(rows) > TRIANGLE_TRIAL_COUNT:
        raise ValueError("response ledger exceeds the frozen 95-trial pool")
    _sitting_rows, _sitting_head, sitting_states = validate_sitting_ledger(package)
    missing_sittings = sorted(
        {
            str(row["sitting_id"])
            for row in rows
            if str(row["sitting_id"]) not in sitting_states
        }
    )
    if missing_sittings:
        raise ValueError(
            f"responses reference sitting IDs without start records: {missing_sittings}"
        )
    return rows, previous


def _unblind_state(package: Path) -> dict[str, Any] | None:
    path = package / "unblind_state.json"
    sidecar = path.with_name(path.name + ".sha256")
    semantic_outputs = list(
        (package / "snapshots").glob("*/partial_unblinded_scores.json")
    ) + list((package / "snapshots").glob("*/partial_discrimination_summary.json"))
    if (package / "generated_acc_after_unblind").exists():
        semantic_outputs.append(package / "generated_acc_after_unblind")
    if not path.is_file():
        if sidecar.exists() or semantic_outputs:
            raise ValueError(
                "semantic-unblind evidence exists but immutable unblind_state is missing"
            )
        return None
    if not sidecar.is_file():
        raise ValueError("unblind_state checksum sidecar is missing")
    parts = sidecar.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[0] != file_sha256(path) or parts[1] != path.name:
        raise ValueError("unblind_state checksum sidecar mismatch")
    state = _read_json_object(path)
    expected_keys = {
        "schema_version",
        "created_at",
        "listening_attempt_id",
        "first_snapshot_id",
        "first_snapshot_sha256",
        "first_snapshot_ledger_head_hash",
        "first_snapshot_answered_count",
        "selection_sha256",
        "private_key_sha256",
    }
    if set(state) != expected_keys or state.get("schema_version") != (
        "streammuse.melody_robustness.listening_triangle_unblind_state.v2"
    ):
        raise ValueError("unblind_state schema/fields are invalid")
    triangle_listening_attempt_number(state.get("listening_attempt_id"))
    _validate_utc_timestamp(state.get("created_at"), label="unblind_state.created_at")
    snapshot = package / "snapshots" / str(state.get("first_snapshot_id"))
    sealed_path = snapshot / "sealed_responses.json"
    if not sealed_path.is_file() or file_sha256(sealed_path) != state.get(
        "first_snapshot_sha256"
    ):
        raise ValueError("unblind_state first snapshot hash mismatch")
    sealed = _read_json_object(sealed_path)
    if (
        sealed.get("ledger_head_hash") != state.get("first_snapshot_ledger_head_hash")
        or sealed.get("answered_count") != state.get("first_snapshot_answered_count")
        or sealed.get("selection_sha256") != state.get("selection_sha256")
    ):
        raise ValueError("unblind_state first snapshot bindings mismatch")
    if sealed.get("listening_attempt_id") != state.get("listening_attempt_id"):
        raise ValueError("unblind_state first snapshot attempt binding mismatch")
    key_path = package / "private" / "private_key.json"
    if not key_path.is_file() or file_sha256(key_path) != state.get("private_key_sha256"):
        raise ValueError("unblind_state private key binding mismatch")
    return state


def progress_summary(package_dir: str | Path) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    _selection_path, selection = _selection_from_package(package)
    rows, head = validate_response_ledger(package)
    answered = len(rows)
    state = _unblind_state(package)
    sitting_counts = Counter(str(row["sitting_id"]) for row in rows)
    sitting_rows, sitting_head, sitting_states = validate_sitting_ledger(package)
    answered_trials = selection["trials"][:answered]
    block_coverage = Counter(str(trial["block"]) for trial in answered_trials)
    primary_coverage = Counter(
        str(trial["condition"])
        for trial in answered_trials
        if trial["block"] == "medium_primary"
    )
    return {
        "schema_version": "streammuse.melody_robustness.listening_triangle_progress.v2",
        "listening_attempt_id": selection["listening_attempt_id"],
        "collection_status": (
            "not_started"
            if answered == 0
            else "full"
            if answered == TRIANGLE_TRIAL_COUNT
            else "partial"
        ),
        "answered_count": answered,
        "pending_count": TRIANGLE_TRIAL_COUNT - answered,
        "total_count": TRIANGLE_TRIAL_COUNT,
        "next_trial_id": (
            selection["trials"][answered]["question_id"]
            if answered < TRIANGLE_TRIAL_COUNT
            else None
        ),
        "next_trial_public_path": (
            f"trials/{selection['trials'][answered]['question_id']}"
            if answered < TRIANGLE_TRIAL_COUNT
            else None
        ),
        "ledger_head_hash": head,
        "sitting_ledger_head_hash": sitting_head,
        "sitting_event_count": len(sitting_rows),
        "sitting_counts": dict(sorted(sitting_counts.items())),
        "sittings": {
            sitting_id: {
                "device": value["start"]["device"],
                "environment": value["start"]["environment"],
                "started_at": value["start"]["recorded_at"],
                "ended_at": (
                    None if value["end"] is None else value["end"]["recorded_at"]
                ),
                "anomalies": (
                    [] if value["end"] is None else value["end"]["anomalies"]
                ),
                "answered_count": sitting_counts[sitting_id],
                "active": value["end"] is None,
            }
            for sitting_id, value in sorted(sitting_states.items())
        },
        "blind_coverage": {
            "primary_answered": {
                condition: {
                    "answered": primary_coverage[condition],
                    "total": 20,
                }
                for condition in PRIMARY_CONDITIONS
            },
            "qc_presentations": {
                "identity": {
                    "answered": block_coverage["identity_catch"],
                    "total": 6,
                },
                "known_different": {
                    "answered": block_coverage["known_different_control"],
                    "total": 6,
                },
                "repeat": {
                    "answered": block_coverage["exact_repeat"],
                    "total": 8,
                },
            },
            "qc_status": "pending_until_semantic_unblind_and_all_qc_answered",
        },
        "blinding_status": (
            "fully_blind" if state is None else "partially_unblinded_during_collection"
        ),
        "first_semantic_unblind_at": None if state is None else state.get("created_at"),
        "can_snapshot": answered >= 1,
        "can_continue": answered < TRIANGLE_TRIAL_COUNT,
    }


def _append_response_unlocked(
    package_dir: str | Path,
    *,
    odd_choice: str,
    confidence_1_to_5: int,
    sitting_id: str,
    difference_tags: Sequence[str] = (),
    note: str = "",
    play_counts: Sequence[int] = (1, 1, 1),
    response_time_ms: int = 0,
    trial_id: str | None = None,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    """Append and fsync one response, then atomically refresh progress state."""

    package = Path(package_dir).resolve()
    _selection_path, selection = _selection_from_package(package)
    rows, previous = validate_response_ledger(package)
    sequence = len(rows) + 1
    if sequence > TRIANGLE_TRIAL_COUNT:
        raise ValueError("all 95 triangle trials are already answered")
    expected_id = str(selection["trials"][sequence - 1]["question_id"])
    if trial_id is not None and trial_id != expected_id:
        raise ValueError(f"next frozen trial is {expected_id}, not {trial_id}")
    if not isinstance(odd_choice, str) or odd_choice not in TRIANGLE_CHOICES:
        raise ValueError(f"odd_choice must be one of {TRIANGLE_CHOICES}")
    if isinstance(confidence_1_to_5, bool) or confidence_1_to_5 not in {1, 2, 3, 4, 5}:
        raise ValueError("confidence must be an integer in 1..5")
    if not isinstance(sitting_id, str) or not sitting_id or len(sitting_id) > 128:
        raise ValueError("sitting_id must be a non-empty string of at most 128 characters")
    if isinstance(difference_tags, (str, bytes)) or not isinstance(
        difference_tags, Sequence
    ):
        raise ValueError("difference_tags must be a sequence of frozen tag strings")
    if len(difference_tags) != len(set(difference_tags)) or any(
        not isinstance(tag, str) or tag not in TRIANGLE_TAGS
        for tag in difference_tags
    ):
        raise ValueError(f"difference tags must be unique members of {TRIANGLE_TAGS}")
    if not isinstance(note, str) or len(note) > 10_000:
        raise ValueError("note must be a string of at most 10000 characters")
    if isinstance(play_counts, (str, bytes)) or not isinstance(play_counts, Sequence):
        raise ValueError("play_counts must contain three positive integers")
    if len(play_counts) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in play_counts
    ):
        raise ValueError("play_counts must contain three positive integers")
    if isinstance(response_time_ms, bool) or not isinstance(response_time_ms, int) or response_time_ms < 0:
        raise ValueError("response_time_ms must be a non-negative integer")
    submitted = submitted_at or _utc_now()
    _validate_utc_timestamp(submitted, label="submitted_at")
    _sitting_rows, _sitting_head, sitting_states = validate_sitting_ledger(package)
    sitting = sitting_states.get(sitting_id)
    if sitting is None:
        raise ValueError("response sitting_id has no structured sitting start record")
    if sitting["end"] is not None:
        raise ValueError("response cannot be appended after the sitting was ended")
    state = _unblind_state(package)
    body = {
        "schema_version": TRIANGLE_LEDGER_SCHEMA_VERSION,
        "listening_attempt_id": selection["listening_attempt_id"],
        "sequence": sequence,
        "trial_id": expected_id,
        "odd_choice": odd_choice,
        "confidence_1_to_5": confidence_1_to_5,
        "difference_tags": list(difference_tags),
        "note": note,
        "play_counts": list(play_counts),
        "response_time_ms": response_time_ms,
        "sitting_id": sitting_id,
        "submitted_at": submitted,
        "blinding_phase": (
            "pre_semantic_unblind"
            if state is None
            else "post_partial_unblind_exploratory"
        ),
        "previous_record_hash": previous,
    }
    record = {**body, "record_hash": canonical_sha256(body)}
    ledger = package / "blind" / "response_ledger.jsonl"
    line = canonical_json_bytes(record)
    _durable_append_line(ledger, line)
    # Re-read the complete chain before publishing the derived resume state.
    validated, head = validate_response_ledger(package)
    if len(validated) != sequence or head != record["record_hash"]:
        raise RuntimeError("response ledger did not validate after durable append")
    summary = progress_summary(package)
    _atomic_json(package / "blind" / "progress_state.json", summary)
    return {"response": record, "progress": summary}


def append_response(
    package_dir: str | Path,
    *,
    odd_choice: str,
    confidence_1_to_5: int,
    sitting_id: str,
    difference_tags: Sequence[str] = (),
    note: str = "",
    play_counts: Sequence[int] = (1, 1, 1),
    response_time_ms: int = 0,
    trial_id: str | None = None,
    submitted_at: str | None = None,
) -> dict[str, Any]:
    """Serialize read/validate/append/fsync/revalidate as one ledger CAS."""

    package = Path(package_dir).resolve()
    lock_path = package / "private" / "listening_state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _append_response_unlocked(
                package,
                odd_choice=odd_choice,
                confidence_1_to_5=confidence_1_to_5,
                sitting_id=sitting_id,
                difference_tags=difference_tags,
                note=note,
                play_counts=play_counts,
                response_time_ms=response_time_ms,
                trial_id=trial_id,
                submitted_at=submitted_at,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def create_snapshot(package_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    package = Path(package_dir).resolve()
    selection_path, selection = _selection_from_package(package)
    rows, head = validate_response_ledger(package)
    if not rows or head is None:
        raise ValueError("at least one answered trial is required for a snapshot")
    package_audit_path = package / "package_audit.json"
    package_audit = _read_json_object(package_audit_path)
    if package_audit.get("valid") is not True:
        raise ValueError("cannot snapshot an invalid triangle package")
    answered_ids = [str(row["trial_id"]) for row in rows]
    all_ids = [str(trial["question_id"]) for trial in selection["trials"]]
    sitting_rows, sitting_head, _sitting_states = validate_sitting_ledger(package)
    snapshot_id = f"snapshot-{len(rows):03d}-{head[:12]}"
    destination = package / "snapshots" / snapshot_id
    payload = {
        "schema_version": TRIANGLE_SNAPSHOT_SCHEMA_VERSION,
        "listening_attempt_id": selection["listening_attempt_id"],
        "snapshot_id": snapshot_id,
        "created_at": _utc_now(),
        "answered_count": len(rows),
        "pending_count": TRIANGLE_TRIAL_COUNT - len(rows),
        "collection_status": (
            "full" if len(rows) == TRIANGLE_TRIAL_COUNT else "partial"
        ),
        "ledger_head_hash": head,
        "sitting_records": sitting_rows,
        "sitting_ledger_head_hash": sitting_head,
        "answered_trial_ids": answered_ids,
        "pending_trial_ids": all_ids[len(rows) :],
        "responses": rows,
        "selection_path": str(selection_path),
        "selection_sha256": file_sha256(selection_path),
        "package_audit_path": str(package_audit_path),
        "package_audit_sha256": file_sha256(package_audit_path),
        "render_manifest_sha256": file_sha256(package / "render_manifest.json"),
        "private_key_sha256": file_sha256(package / "private" / "private_key.json"),
        "unblind_state_at_snapshot": _unblind_state(package),
    }
    progress = progress_summary(package)
    if destination.exists():
        # created_at is intentionally part of an immutable first snapshot.  A
        # repeated request for the same ledger head returns the original.
        existing = validate_snapshot(package, destination)
        return destination, existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{snapshot_id}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"stale/in-progress snapshot temp directory: {temporary}")
    temporary.mkdir()
    try:
        write_canonical_json(temporary / "sealed_responses.json", payload)
        write_canonical_json(temporary / "blind_progress_summary.json", progress)
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError:
        shutil.rmtree(temporary)
        existing = validate_snapshot(package, destination)
        return destination, existing
    except Exception:
        shutil.rmtree(temporary)
        raise
    validated = validate_snapshot(package, destination)
    return destination, validated


def validate_snapshot(package_dir: str | Path, snapshot_dir: str | Path) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    snapshot_root = Path(snapshot_dir).resolve()
    if not snapshot_root.is_relative_to(package / "snapshots"):
        raise ValueError("snapshot directory escapes the package")
    sealed_path = snapshot_root / "sealed_responses.json"
    sealed = _read_json_object(sealed_path)
    checksum_path = sealed_path.with_name(sealed_path.name + ".sha256")
    if not checksum_path.is_file():
        raise ValueError("snapshot checksum sidecar is missing")
    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    if (
        len(checksum_parts) != 2
        or checksum_parts[0] != file_sha256(sealed_path)
        or checksum_parts[1] != sealed_path.name
    ):
        raise ValueError("snapshot checksum sidecar does not bind sealed_responses.json")
    if sealed.get("schema_version") != TRIANGLE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema mismatch")
    selection_path, selection = _selection_from_package(package)
    expected_snapshot_fields = {
        "schema_version",
        "listening_attempt_id",
        "snapshot_id",
        "created_at",
        "answered_count",
        "pending_count",
        "collection_status",
        "ledger_head_hash",
        "sitting_records",
        "sitting_ledger_head_hash",
        "answered_trial_ids",
        "pending_trial_ids",
        "responses",
        "selection_path",
        "selection_sha256",
        "package_audit_path",
        "package_audit_sha256",
        "render_manifest_sha256",
        "private_key_sha256",
        "unblind_state_at_snapshot",
    }
    if set(sealed) != expected_snapshot_fields:
        raise ValueError("snapshot has missing or unexpected fields")
    if sealed.get("listening_attempt_id") != selection.get("listening_attempt_id"):
        raise ValueError("snapshot listening attempt ID mismatch")
    _validate_utc_timestamp(sealed.get("created_at"), label="snapshot.created_at")
    if sealed.get("selection_sha256") != file_sha256(selection_path):
        raise ValueError("snapshot selection hash mismatch")
    if sealed.get("package_audit_sha256") != file_sha256(package / "package_audit.json"):
        raise ValueError("snapshot package audit hash mismatch")
    if sealed.get("render_manifest_sha256") != file_sha256(package / "render_manifest.json"):
        raise ValueError("snapshot render manifest hash mismatch")
    if sealed.get("private_key_sha256") != file_sha256(package / "private" / "private_key.json"):
        raise ValueError("snapshot private key hash mismatch")
    responses = sealed.get("responses")
    if not isinstance(responses, list) or not 1 <= len(responses) <= TRIANGLE_TRIAL_COUNT:
        raise ValueError("snapshot must seal 1..95 response rows")
    expected_ids = [str(trial["question_id"]) for trial in selection["trials"]]
    ids = [str(row.get("trial_id")) for row in responses]
    if ids != expected_ids[: len(ids)] or len(ids) != len(set(ids)):
        raise ValueError("snapshot rows are duplicate, orphaned, reordered, or skipped")
    previous: str | None = None
    for index, row in enumerate(responses, start=1):
        body = {key: value for key, value in row.items() if key != "record_hash"}
        if (
            row.get("record_hash") != canonical_sha256(body)
            or row.get("sequence") != index
            or row.get("previous_record_hash") != previous
        ):
            raise ValueError("snapshot response hash chain is invalid")
        previous = str(row["record_hash"])
    if previous != sealed.get("ledger_head_hash"):
        raise ValueError("snapshot ledger head does not match its response chain")
    sitting_records = sealed.get("sitting_records")
    if not isinstance(sitting_records, list):
        raise ValueError("snapshot sitting_records must be a list")
    durable_sittings, _current_sitting_head, _sitting_states = (
        validate_sitting_ledger(package)
    )
    if (
        len(durable_sittings) < len(sitting_records)
        or durable_sittings[: len(sitting_records)] != sitting_records
    ):
        raise ValueError("snapshot sitting records differ from the durable-ledger prefix")
    expected_sitting_head = (
        None if not sitting_records else sitting_records[-1].get("record_hash")
    )
    if sealed.get("sitting_ledger_head_hash") != expected_sitting_head:
        raise ValueError("snapshot sitting ledger head mismatch")
    started_sittings = {
        str(row.get("sitting_id"))
        for row in sitting_records
        if row.get("event") == "start"
    }
    if any(str(row.get("sitting_id")) not in started_sittings for row in responses):
        raise ValueError("snapshot response lacks a sealed sitting start record")
    if sealed.get("answered_count") != len(responses):
        raise ValueError("snapshot answered_count mismatch")
    if sealed.get("pending_trial_ids") != expected_ids[len(responses) :]:
        raise ValueError("snapshot pending set mismatch")
    if sealed.get("answered_trial_ids") != expected_ids[: len(responses)]:
        raise ValueError("snapshot answered ID set mismatch")
    if sealed.get("pending_count") != TRIANGLE_TRIAL_COUNT - len(responses):
        raise ValueError("snapshot pending_count mismatch")
    expected_status = "full" if len(responses) == TRIANGLE_TRIAL_COUNT else "partial"
    if sealed.get("collection_status") != expected_status:
        raise ValueError("snapshot collection_status mismatch")
    if sealed.get("snapshot_id") != snapshot_root.name:
        raise ValueError("snapshot ID differs from its immutable directory name")
    expected_snapshot_id = f"snapshot-{len(responses):03d}-{str(previous)[:12]}"
    if snapshot_root.name != expected_snapshot_id:
        raise ValueError("snapshot directory name does not bind count and ledger head")
    if Path(str(sealed.get("selection_path", ""))).resolve() != selection_path:
        raise ValueError("snapshot selection path mismatch")
    if Path(str(sealed.get("package_audit_path", ""))).resolve() != (
        package / "package_audit.json"
    ):
        raise ValueError("snapshot package-audit path mismatch")
    durable_rows, _durable_head = validate_response_ledger(package)
    if len(durable_rows) < len(responses) or durable_rows[: len(responses)] != responses:
        raise ValueError("snapshot responses differ from the exact durable-ledger prefix")
    state = _unblind_state(package)
    state_at_snapshot = sealed.get("unblind_state_at_snapshot")
    if state is None and state_at_snapshot is not None:
        raise ValueError("blind snapshot cannot claim a semantic-unblind state")
    if state is not None:
        first_count = int(state["first_snapshot_answered_count"])
        if len(responses) > first_count and state_at_snapshot != state:
            raise ValueError("post-unblind snapshot does not preserve the unblind boundary")
        if state_at_snapshot is not None and state_at_snapshot != state:
            raise ValueError("snapshot unblind state differs from immutable package state")
    return sealed


def _response_semantic_choice(trial: Mapping[str, Any], choice: str) -> str:
    if choice == "no_difference":
        return "no_difference"
    position = int(choice) - 1
    pattern = str(trial["presentation_pattern"])
    if position < 0 or position >= len(pattern):
        return "invalid"
    return pattern[position].lower()


def unblind_snapshot(
    package_dir: str | Path, snapshot_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    package = Path(package_dir).resolve()
    snapshot_root = Path(snapshot_dir).resolve()
    sealed = validate_snapshot(package, snapshot_root)
    key_path = package / "private" / "private_key.json"
    key = _read_json_object(key_path)
    if key.get("selection_sha256") != sealed.get("selection_sha256"):
        raise ValueError("private key and snapshot selection hashes differ")
    if key.get("listening_attempt_id") != sealed.get("listening_attempt_id"):
        raise ValueError("private key and snapshot listening attempt IDs differ")
    key_rows = key.get("trials")
    if not isinstance(key_rows, list) or len(key_rows) != TRIANGLE_TRIAL_COUNT:
        raise ValueError("private key does not contain 95 exact trial mappings")
    key_by_id = {str(row["question_id"]): row for row in key_rows}

    state_path = package / "unblind_state.json"
    state = _unblind_state(package)
    if state is None:
        state = {
            "schema_version": "streammuse.melody_robustness.listening_triangle_unblind_state.v2",
            "created_at": _utc_now(),
            "listening_attempt_id": sealed["listening_attempt_id"],
            "first_snapshot_id": sealed["snapshot_id"],
            "first_snapshot_sha256": file_sha256(
                snapshot_root / "sealed_responses.json"
            ),
            "first_snapshot_ledger_head_hash": sealed["ledger_head_hash"],
            "first_snapshot_answered_count": sealed["answered_count"],
            "selection_sha256": sealed["selection_sha256"],
            "private_key_sha256": file_sha256(key_path),
        }
        write_canonical_json(state_path, state)
        state = _unblind_state(package)
        if state is None:
            raise RuntimeError("failed to persist immutable semantic-unblind event")
        _atomic_json(
            package / "blind" / "progress_state.json",
            progress_summary(package),
        )
    rows = []
    for response in sealed["responses"]:
        trial = key_by_id[str(response["trial_id"])]
        objective_identity = bool(trial.get("objective_identity", False))
        if trial["block"] == "identity_catch":
            correct = response["odd_choice"] == "no_difference"
        elif objective_identity:
            # A trial whose two actual generated files are identical cannot be
            # evidence that the perturbation was audibly discriminated.
            correct = False
        else:
            correct = response["odd_choice"] == str(trial["correct_choice"])
        rows.append(
            {
                **response,
                **trial,
                "scored_correct": correct,
                "semantic_response": _response_semantic_choice(
                    trial, str(response["odd_choice"])
                ),
            }
        )
    payload = {
        "schema_version": TRIANGLE_UNBLINDED_SCHEMA_VERSION,
        "listening_attempt_id": sealed["listening_attempt_id"],
        "snapshot_id": sealed["snapshot_id"],
        "snapshot_sha256": file_sha256(snapshot_root / "sealed_responses.json"),
        "selection_sha256": sealed["selection_sha256"],
        "private_key_sha256": file_sha256(key_path),
        "first_semantic_unblind": state,
        "answered_count": len(rows),
        "pending_count": TRIANGLE_TRIAL_COUNT - len(rows),
        "rows": rows,
    }
    output = snapshot_root / "partial_unblinded_scores.json"
    if output.exists():
        if _read_json_object(output) != payload:
            raise ValueError("existing partial unblinded scores cannot be overwritten")
    else:
        write_canonical_json(output, payload)
    return output, payload


def _rate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answered = len(rows)
    correct = sum(bool(row.get("scored_correct")) for row in rows)
    no_difference = sum(row.get("odd_choice") == "no_difference" for row in rows)
    confidence = [int(row["confidence_1_to_5"]) for row in rows]

    def pair_hash_identity(row: Mapping[str, Any], field: str) -> bool:
        source_a = row.get("source_a")
        source_b = row.get("source_b")
        if not isinstance(source_a, Mapping) or not isinstance(source_b, Mapping):
            return False
        value_a = source_a.get(field)
        value_b = source_b.get(field)
        return (
            isinstance(value_a, str)
            and bool(value_a)
            and isinstance(value_b, str)
            and bool(value_b)
            and value_a == value_b
        )

    return {
        "answered": answered,
        "correct": correct,
        "hit_rate": None if answered == 0 else correct / answered,
        "no_difference": no_difference,
        "mean_confidence": None if not confidence else sum(confidence) / len(confidence),
        "pre_unblind_answered": sum(
            row.get("blinding_phase") == "pre_semantic_unblind" for row in rows
        ),
        "post_unblind_answered": sum(
            row.get("blinding_phase") == "post_partial_unblind_exploratory"
            for row in rows
        ),
        "objective_identity": sum(bool(row.get("objective_identity")) for row in rows),
        "source_empty_either": sum(
            bool(row.get("source_a", {}).get("source_empty"))
            or bool(row.get("source_b", {}).get("source_empty"))
            for row in rows
        ),
        "coverage_driven": sum(bool(row.get("coverage_driven")) for row in rows),
        "operational_invalid_either": sum(
            isinstance(row.get("source_a"), Mapping)
            and row["source_a"].get("operational_valid") is False
            or isinstance(row.get("source_b"), Mapping)
            and row["source_b"].get("operational_valid") is False
            for row in rows
        ),
        "raw_token_identity": sum(
            pair_hash_identity(row, "raw_token_payload_sha256") for row in rows
        ),
        "theoretical_midi_identity": sum(
            pair_hash_identity(row, "source_sha256") for row in rows
        ),
        "canonical_midi_identity": sum(
            pair_hash_identity(row, "excerpt_midi_sha256") for row in rows
        ),
        "rendered_wav_identity": sum(
            pair_hash_identity(row, "rendered_pair_wav_sha256") for row in rows
        ),
    }


def _split_rate_views(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "pre_unblind": _rate_summary(
            [row for row in rows if row.get("blinding_phase") == "pre_semantic_unblind"]
        ),
        "post_unblind_exploratory": _rate_summary(
            [
                row
                for row in rows
                if row.get("blinding_phase") == "post_partial_unblind_exploratory"
            ]
        ),
        "combined_descriptive": _rate_summary(rows),
    }


def _derive_summary_payload(
    unblinded: Mapping[str, Any], *, unblinded_scores_sha256: str
) -> dict[str, Any]:
    """Pure derivation used by both the writer and report-side validator."""

    rows = list(unblinded.get("rows", []))
    if len(rows) != unblinded.get("answered_count"):
        raise ValueError("unblinded answered_count mismatch")

    by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in PRIMARY_CONDITIONS
    }
    high: list[dict[str, Any]] = []
    sham_sampling: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    known: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    primary_by_song: dict[str, dict[str, list[dict[str, Any]]]] = {
        condition: defaultdict(list) for condition in PRIMARY_CONDITIONS
    }
    for row in rows:
        block = row.get("block")
        if block == "medium_primary":
            condition = str(row["condition"])
            by_condition[condition].append(row)
            song = str(row["source_b"]["song"])
            primary_by_song[condition][song].append(row)
        elif block == "high_exploratory":
            high.append(row)
        elif block == "sham_sampling_baseline":
            sham_sampling.append(row)
        elif block == "identity_catch":
            identities.append(row)
        elif block == "known_different_control":
            known.append(row)
        elif block == "exact_repeat":
            repeats.append(row)

    identity_correct = sum(bool(row["scored_correct"]) for row in identities)
    known_correct = sum(bool(row["scored_correct"]) for row in known)
    original_by_semantic = {
        str(row["semantic_id"]): row for row in rows if row.get("block") != "exact_repeat"
    }
    repeat_comparable = 0
    repeat_consistent = 0
    repeat_rows = []
    for repeat in repeats:
        original = original_by_semantic.get(str(repeat.get("repeat_of")))
        consistent: bool | None = None
        if original is not None:
            repeat_comparable += 1
            consistent = repeat.get("semantic_response") == original.get("semantic_response")
            repeat_consistent += bool(consistent)
        repeat_rows.append(
            {
                "trial_id": repeat["trial_id"],
                "repeat_of": repeat.get("repeat_of"),
                "original_answered": original is not None,
                "consistent": consistent,
            }
        )
    all_qc_answered = (
        len(identities) == QC_THRESHOLDS["identity_total"]
        and len(known) == QC_THRESHOLDS["known_different_total"]
        and repeat_comparable == QC_THRESHOLDS["repeat_total"]
    )
    if not all_qc_answered:
        qc_status = "pending"
    elif (
        identity_correct >= QC_THRESHOLDS["identity_correct"]
        and known_correct >= QC_THRESHOLDS["known_different_correct"]
        and repeat_consistent >= QC_THRESHOLDS["repeat_consistent"]
    ):
        qc_status = "pass"
    else:
        qc_status = "fail"

    condition_results: dict[str, Any] = {}
    for condition, condition_rows in by_condition.items():
        summary = _rate_summary(condition_rows)
        per_song = {
            song: {
                **_rate_summary(song_rows),
                "views": _split_rate_views(song_rows),
            }
            for song, song_rows in sorted(primary_by_song[condition].items())
        }
        complete = len(condition_rows) == 20
        all_pre = complete and all(
            row.get("blinding_phase") == "pre_semantic_unblind" for row in condition_rows
        )
        songs_passing = sum(
            value["answered"] == 4 and value["correct"] >= 2 for value in per_song.values()
        )
        if complete and all_pre and qc_status == "pass":
            decision = (
                "confirmed discriminable in this fixed listener/package"
                if summary["correct"] >= 12 and songs_passing >= 4
                else "not confirmed in this listening run"
            )
        else:
            decision = "partial — preregistered decision pending"
        condition_results[condition] = {
            **summary,
            "pending": 20 - len(condition_rows),
            "per_song": per_song,
            "songs_with_at_least_2_of_4_correct": songs_passing,
            "decision": decision,
            "decision_eligible": complete and all_pre and qc_status == "pass",
            "views": _split_rate_views(condition_rows),
        }

    answered = len(rows)
    payload = {
        "schema_version": "streammuse.melody_robustness.listening_triangle_summary.v2",
        "listening_attempt_id": unblinded["listening_attempt_id"],
        "snapshot_id": unblinded["snapshot_id"],
        "unblinded_scores_sha256": unblinded_scores_sha256,
        "collection_status": (
            "full" if answered == TRIANGLE_TRIAL_COUNT else "partial"
        ),
        "answered_count": answered,
        "pending_count": TRIANGLE_TRIAL_COUNT - answered,
        "qc_status": qc_status,
        "attempt_disposition": (
            "in_progress_qc_pending"
            if qc_status == "pending"
            else "eligible_for_preregistered_decisions"
            if qc_status == "pass"
            else "sealed_qc_failure_retry_required"
        ),
        "retry_required": qc_status == "fail",
        "blinding_status": (
            "partially_unblinded_during_collection"
            if any(
                row.get("blinding_phase") == "post_partial_unblind_exploratory"
                for row in rows
            )
            else "fully_blind_for_answered_rows"
        ),
        "conditions": condition_results,
        "high_exploratory": _rate_summary(high),
        "sham_sampling_baseline": _rate_summary(sham_sampling),
        "quality_control": {
            "identity": {
                "answered": len(identities),
                "correct_no_difference": identity_correct,
                "required": "5/6",
            },
            "known_different": {
                "answered": len(known),
                "correct_odd": known_correct,
                "required": "5/6",
            },
            "repeats": {
                "answered": len(repeats),
                "comparable": repeat_comparable,
                "consistent": repeat_consistent,
                "required": "6/8",
                "rows": repeat_rows,
            },
        },
        "views": {
            **_split_rate_views(rows),
            "non_coverage_driven_sensitivity": _rate_summary(
                [row for row in rows if not bool(row.get("coverage_driven"))]
            ),
        },
        "limitations": [
            "single listener",
            "five songs and fixed eight-second excerpts",
            "multiple trials share songs and generated sources",
            "discrimination does not measure quality, preference, harmony, or population effects",
            "partial denominators do not support the preregistered full-condition decision",
        ],
    }
    return payload


def summarize_unblinded(
    package_dir: str | Path, snapshot_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    package = Path(package_dir).resolve()
    snapshot_root = Path(snapshot_dir).resolve()
    unblinded_path = snapshot_root / "partial_unblinded_scores.json"
    if not unblinded_path.is_file():
        unblind_snapshot(package, snapshot_root)
    unblinded = _read_json_object(unblinded_path)
    validate_snapshot(package, snapshot_root)
    if unblinded.get("snapshot_sha256") != file_sha256(
        snapshot_root / "sealed_responses.json"
    ):
        raise ValueError("unblinded rows are not bound to the current immutable snapshot")
    payload = _derive_summary_payload(
        unblinded, unblinded_scores_sha256=file_sha256(unblinded_path)
    )
    answered = int(unblinded["answered_count"])
    output = snapshot_root / "partial_discrimination_summary.json"
    if output.exists():
        if _read_json_object(output) != payload:
            raise ValueError("existing discrimination summary cannot be overwritten")
    else:
        write_canonical_json(output, payload)
    if answered == TRIANGLE_TRIAL_COUNT:
        full = package / "full"
        full.mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (unblinded_path, full / "unblinded_scores.json"),
            (output, full / "discrimination_summary.json"),
        ):
            if destination.exists():
                if file_sha256(destination) != file_sha256(source):
                    raise ValueError(f"existing full result cannot be overwritten: {destination}")
            else:
                shutil.copyfile(source, destination)
    return output, payload


def validate_unblinded_summary(
    package_dir: str | Path, snapshot_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pure, exact validation for report builders; never creates or rewrites files."""

    package = Path(package_dir).resolve()
    snapshot_root = Path(snapshot_dir).resolve()
    sealed = validate_snapshot(package, snapshot_root)
    key_path = package / "private" / "private_key.json"
    key = _read_json_object(key_path)
    if key.get("selection_sha256") != sealed.get("selection_sha256"):
        raise ValueError("private key and snapshot selection hashes differ")
    if key.get("listening_attempt_id") != sealed.get("listening_attempt_id"):
        raise ValueError("private key and snapshot listening attempt IDs differ")
    key_rows = key.get("trials")
    if not isinstance(key_rows, list) or len(key_rows) != TRIANGLE_TRIAL_COUNT:
        raise ValueError("private key does not contain 95 exact trial mappings")
    key_by_id = {str(row["question_id"]): row for row in key_rows}
    state = _unblind_state(package)
    if state is None:
        raise ValueError("semantic unblind state is missing")
    rebuilt_rows = []
    for response in sealed["responses"]:
        trial = key_by_id[str(response["trial_id"])]
        objective_identity = bool(trial.get("objective_identity", False))
        if trial["block"] == "identity_catch":
            correct = response["odd_choice"] == "no_difference"
        elif objective_identity:
            correct = False
        else:
            correct = response["odd_choice"] == str(trial["correct_choice"])
        rebuilt_rows.append(
            {
                **response,
                **trial,
                "scored_correct": correct,
                "semantic_response": _response_semantic_choice(
                    trial, str(response["odd_choice"])
                ),
            }
        )
    expected_unblinded = {
        "schema_version": TRIANGLE_UNBLINDED_SCHEMA_VERSION,
        "listening_attempt_id": sealed["listening_attempt_id"],
        "snapshot_id": sealed["snapshot_id"],
        "snapshot_sha256": file_sha256(snapshot_root / "sealed_responses.json"),
        "selection_sha256": sealed["selection_sha256"],
        "private_key_sha256": file_sha256(key_path),
        "first_semantic_unblind": state,
        "answered_count": len(rebuilt_rows),
        "pending_count": TRIANGLE_TRIAL_COUNT - len(rebuilt_rows),
        "rows": rebuilt_rows,
    }
    unblinded_path = snapshot_root / "partial_unblinded_scores.json"
    unblinded = _read_json_object(unblinded_path)
    if unblinded != expected_unblinded:
        raise ValueError("partial unblinded scores differ from exact snapshot/key derivation")
    expected_summary = _derive_summary_payload(
        expected_unblinded,
        unblinded_scores_sha256=file_sha256(unblinded_path),
    )
    summary = _read_json_object(snapshot_root / "partial_discrimination_summary.json")
    if summary != expected_summary:
        raise ValueError("partial discrimination summary differs from exact score derivation")
    return unblinded, summary


_REBLIND_TRIAL_FIELDS = {
    "presentation_pattern",
    "duplicated_source",
    "correct_choice",
    "odd_position",
    "global_order_index",
    "question_id",
    "repeat_distance",
}


def _retry_semantic_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy_value
        for key, copy_value in json.loads(
            json.dumps(dict(trial), ensure_ascii=False)
        ).items()
        if key not in _REBLIND_TRIAL_FIELDS
    }


def _retry_order_key(
    *, effective_seed: int, nonce: int, chunk_index: int, semantic_id: str
) -> str:
    material = (
        f"{TRIANGLE_RETRY_SEED_DOMAIN}:order\0{effective_seed}\0{nonce}\0"
        f"{chunk_index}\0{semantic_id}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _reblind_retry_selection(
    previous: Mapping[str, Any],
    *,
    attempt_number: int,
    retry_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    previous_number = triangle_listening_attempt_number(
        previous.get("listening_attempt_id")
    )
    if attempt_number != previous_number + 1:
        raise ValueError("retry must advance the listening attempt by exactly one")
    base_seed = previous.get("base_blind_order_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("previous selection lacks the frozen base blind-order seed")
    effective_seed, seed_digest = derive_triangle_retry_seed(
        base_blind_seed=base_seed,
        attempt_number=attempt_number,
    )
    previous_trials = previous.get("trials")
    if not isinstance(previous_trials, list) or len(previous_trials) != TRIANGLE_TRIAL_COUNT:
        raise ValueError("previous retry source must contain exactly 95 trials")
    semantic_chunks = [
        [_retry_semantic_trial(row) for row in previous_trials[start : start + 19]]
        for start in range(0, TRIANGLE_TRIAL_COUNT, 19)
    ]
    ordered: list[dict[str, Any]] | None = None
    for nonce in range(10_000):
        candidate = [
            row
            for chunk_index, chunk in enumerate(semantic_chunks)
            for row in sorted(
                chunk,
                key=lambda value, ci=chunk_index: _retry_order_key(
                    effective_seed=effective_seed,
                    nonce=nonce,
                    chunk_index=ci,
                    semantic_id=str(value["semantic_id"]),
                ),
            )
        ]
        positions = {
            str(row["semantic_id"]): index for index, row in enumerate(candidate)
        }
        if all(
            row.get("repeat_of") is None
            or (
                positions[str(row["semantic_id"])]
                - positions[str(row["repeat_of"])]
                >= 8
            )
            for row in candidate
        ):
            ordered = candidate
            break
    if ordered is None:
        raise RuntimeError("could not derive a retry order satisfying repeat spacing")
    _assign_patterns(ordered, effective_seed)
    positions = {
        str(row["semantic_id"]): int(row["global_order_index"]) for row in ordered
    }
    for row in ordered:
        if row.get("repeat_of") is not None:
            row["repeat_distance"] = positions[str(row["semantic_id"])] - positions[
                str(row["repeat_of"])
            ]
    payload = json.loads(json.dumps(dict(previous), ensure_ascii=False))
    payload.update(
        {
            "frozen_before_formal": True,
            "retry_reblind_after_formal_without_semantic_change": True,
            "blind_order_seed": effective_seed,
            "base_blind_order_seed": base_seed,
            "effective_blind_order_seed": effective_seed,
            "effective_blind_order_seed_sha256": seed_digest,
            "listening_attempt_id": triangle_listening_attempt_id(attempt_number),
            "listening_attempt_number": attempt_number,
            "retry_lineage": dict(retry_lineage),
            "retry_lineage_sha256": canonical_sha256(dict(retry_lineage)),
            "trials": ordered,
        }
    )
    return payload


def _retry_failure_evidence(
    failed_package_dir: str | Path, failed_snapshot_dir: str | Path
) -> dict[str, Any]:
    package = Path(failed_package_dir).resolve()
    snapshot = Path(failed_snapshot_dir).resolve()
    sealed = validate_snapshot(package, snapshot)
    if sealed.get("collection_status") != "full" or sealed.get("answered_count") != 95:
        raise ValueError("QC retry requires an immutable full 95-response snapshot")
    unblinded, summary = validate_unblinded_summary(package, snapshot)
    if summary.get("qc_status") != "fail":
        raise ValueError("QC retry is only authorized when qc_status=fail")
    if summary.get("retry_required") is not True or summary.get(
        "attempt_disposition"
    ) != "sealed_qc_failure_retry_required":
        raise ValueError("failed summary does not preserve the preregistered retry decision")
    selection_path, selection = _selection_from_package(package)
    manifest_path = Path(str(selection.get("input_manifest_path", ""))).resolve()
    manifest = _read_json_object(manifest_path)
    validate_triangle_selection_manifest(
        selection,
        manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )
    previous_number = triangle_listening_attempt_number(
        selection.get("listening_attempt_id")
    )
    if sealed.get("listening_attempt_id") != selection.get("listening_attempt_id"):
        raise ValueError("failed snapshot and selection attempt IDs differ")
    audit_path = package / "package_audit.json"
    audit = _read_json_object(audit_path)
    if audit.get("valid") is not True or audit.get("accepted_final") is not True:
        raise ValueError("QC retry requires a valid accepted-final failed package")
    if audit.get("listening_attempt_id") != selection.get("listening_attempt_id"):
        raise ValueError("failed package audit attempt ID mismatch")
    state_path = package / "unblind_state.json"
    state = _unblind_state(package)
    if state is None:
        raise ValueError("QC retry requires an immutable semantic-unblind state")
    ledger_path = package / "blind" / "response_ledger.jsonl"
    rows, ledger_head = validate_response_ledger(package)
    if len(rows) != TRIANGLE_TRIAL_COUNT or ledger_head != sealed.get("ledger_head_hash"):
        raise ValueError("failed full snapshot does not match the complete durable ledger")
    sitting_ledger_path = package / "blind" / "sitting_ledger.jsonl"
    sitting_rows, sitting_ledger_head, _sitting_states = validate_sitting_ledger(package)
    if not sitting_rows or not sitting_ledger_path.is_file():
        raise ValueError("failed attempt lacks structured sitting provenance")
    render_path = package / "render_manifest.json"
    private_path = package / "private" / "private_key.json"
    unblinded_path = snapshot / "partial_unblinded_scores.json"
    summary_path = snapshot / "partial_discrimination_summary.json"
    previous_lineage = selection.get("retry_lineage")
    if previous_number == 1:
        base_selection_path = selection_path
        base_selection_sha256 = file_sha256(selection_path)
    else:
        if not isinstance(previous_lineage, Mapping):
            raise ValueError("retry selection lacks its previous lineage")
        base_selection_path = Path(str(previous_lineage["base_selection_path"])).resolve()
        base_selection_sha256 = str(previous_lineage["base_selection_sha256"])
        if not base_selection_path.is_file() or file_sha256(
            base_selection_path
        ) != base_selection_sha256:
            raise ValueError("retry chain base selection path/hash mismatch")
    return {
        "package": package,
        "snapshot": snapshot,
        "sealed": sealed,
        "selection_path": selection_path,
        "selection": selection,
        "previous_attempt_id": selection["listening_attempt_id"],
        "previous_attempt_number": previous_number,
        "next_attempt_id": triangle_listening_attempt_id(previous_number + 1),
        "next_attempt_number": previous_number + 1,
        "base_blind_order_seed": int(selection["base_blind_order_seed"]),
        "base_selection_path": base_selection_path,
        "base_selection_sha256": base_selection_sha256,
        "audit_path": audit_path,
        "render_path": render_path,
        "private_path": private_path,
        "ledger_path": ledger_path,
        "ledger_head": ledger_head,
        "sitting_ledger_path": sitting_ledger_path,
        "sitting_ledger_head": sitting_ledger_head,
        "state_path": state_path,
        "unblinded_path": unblinded_path,
        "summary_path": summary_path,
        "summary": summary,
        "unblinded": unblinded,
    }


def _retry_authorization_payload(
    evidence: Mapping[str, Any], *, created_at: str
) -> dict[str, Any]:
    effective_seed, seed_digest = derive_triangle_retry_seed(
        base_blind_seed=int(evidence["base_blind_order_seed"]),
        attempt_number=int(evidence["next_attempt_number"]),
    )
    sealed = evidence["sealed"]
    return {
        "schema_version": TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION,
        "created_at": created_at,
        "authorization_reason": "qc_failure",
        "previous_attempt_id": evidence["previous_attempt_id"],
        "previous_attempt_number": evidence["previous_attempt_number"],
        "next_attempt_id": evidence["next_attempt_id"],
        "next_attempt_number": evidence["next_attempt_number"],
        "base_blind_order_seed": evidence["base_blind_order_seed"],
        "effective_blind_order_seed": effective_seed,
        "effective_blind_order_seed_sha256": seed_digest,
        "failed_package_path": str(evidence["package"]),
        "base_selection_path": str(evidence["base_selection_path"]),
        "base_selection_sha256": evidence["base_selection_sha256"],
        "failed_selection_path": str(evidence["selection_path"]),
        "failed_selection_sha256": file_sha256(evidence["selection_path"]),
        "failed_package_audit_path": str(evidence["audit_path"]),
        "failed_package_audit_sha256": file_sha256(evidence["audit_path"]),
        "failed_render_manifest_path": str(evidence["render_path"]),
        "failed_render_manifest_sha256": file_sha256(evidence["render_path"]),
        "failed_private_key_path": str(evidence["private_path"]),
        "failed_private_key_sha256": file_sha256(evidence["private_path"]),
        "failed_response_ledger_path": str(evidence["ledger_path"]),
        "failed_response_ledger_sha256": file_sha256(evidence["ledger_path"]),
        "failed_sitting_ledger_path": str(evidence["sitting_ledger_path"]),
        "failed_sitting_ledger_sha256": file_sha256(evidence["sitting_ledger_path"]),
        "failed_sitting_ledger_head_hash": evidence["sitting_ledger_head"],
        "failed_snapshot_path": str(evidence["snapshot"]),
        "failed_snapshot_id": sealed["snapshot_id"],
        "failed_sealed_responses_sha256": file_sha256(
            evidence["snapshot"] / "sealed_responses.json"
        ),
        "failed_ledger_head_hash": evidence["ledger_head"],
        "failed_answered_count": sealed["answered_count"],
        "failed_unblind_state_path": str(evidence["state_path"]),
        "failed_unblind_state_sha256": file_sha256(evidence["state_path"]),
        "failed_unblinded_scores_path": str(evidence["unblinded_path"]),
        "failed_unblinded_scores_sha256": file_sha256(evidence["unblinded_path"]),
        "failed_summary_path": str(evidence["summary_path"]),
        "failed_summary_sha256": file_sha256(evidence["summary_path"]),
        "previous_qc_status": "fail",
        "previous_retry_required": True,
    }


def _read_retry_authorization(path: Path) -> dict[str, Any]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("retry authorization or checksum sidecar is missing")
    parts = sidecar.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[0] != file_sha256(path) or parts[1] != path.name:
        raise ValueError("retry authorization checksum sidecar mismatch")
    return _read_json_object(path)


def _validate_retry_authorization(
    path: Path, *, verify_files: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = _read_retry_authorization(path)
    _validate_utc_timestamp(
        authorization.get("created_at"), label="retry_authorization.created_at"
    )
    if authorization.get("schema_version") != TRIANGLE_RETRY_AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("retry authorization schema mismatch")
    if not verify_files:
        raise ValueError("retry authorization validation requires immutable failed artifacts")
    evidence = _retry_failure_evidence(
        authorization.get("failed_package_path"),
        authorization.get("failed_snapshot_path"),
    )
    expected = _retry_authorization_payload(
        evidence, created_at=str(authorization["created_at"])
    )
    if authorization != expected:
        raise ValueError("retry authorization differs from exact failed-artifact derivation")
    return authorization, evidence


def _retry_lineage_from_authorization(
    authorization_path: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": TRIANGLE_RETRY_LINEAGE_SCHEMA_VERSION,
        "authorization_reason": "qc_failure",
        "current_attempt_id": authorization["next_attempt_id"],
        "current_attempt_number": authorization["next_attempt_number"],
        "previous_attempt_id": authorization["previous_attempt_id"],
        "previous_attempt_number": authorization["previous_attempt_number"],
        "base_blind_order_seed": authorization["base_blind_order_seed"],
        "effective_blind_order_seed": authorization["effective_blind_order_seed"],
        "effective_blind_order_seed_sha256": authorization[
            "effective_blind_order_seed_sha256"
        ],
        "base_selection_path": authorization["base_selection_path"],
        "base_selection_sha256": authorization["base_selection_sha256"],
        "previous_package_path": authorization["failed_package_path"],
        "previous_selection_path": authorization["failed_selection_path"],
        "previous_selection_sha256": authorization["failed_selection_sha256"],
        "previous_package_audit_path": authorization["failed_package_audit_path"],
        "previous_package_audit_sha256": authorization[
            "failed_package_audit_sha256"
        ],
        "failed_snapshot_path": authorization["failed_snapshot_path"],
        "failed_snapshot_id": authorization["failed_snapshot_id"],
        "failed_sealed_responses_sha256": authorization[
            "failed_sealed_responses_sha256"
        ],
        "failed_ledger_head_hash": authorization["failed_ledger_head_hash"],
        "failed_response_ledger_path": authorization["failed_response_ledger_path"],
        "failed_response_ledger_sha256": authorization[
            "failed_response_ledger_sha256"
        ],
        "failed_sitting_ledger_path": authorization["failed_sitting_ledger_path"],
        "failed_sitting_ledger_sha256": authorization[
            "failed_sitting_ledger_sha256"
        ],
        "failed_sitting_ledger_head_hash": authorization[
            "failed_sitting_ledger_head_hash"
        ],
        "failed_unblind_state_path": authorization["failed_unblind_state_path"],
        "failed_unblind_state_sha256": authorization[
            "failed_unblind_state_sha256"
        ],
        "failed_unblinded_scores_path": authorization[
            "failed_unblinded_scores_path"
        ],
        "failed_unblinded_scores_sha256": authorization[
            "failed_unblinded_scores_sha256"
        ],
        "failed_summary_path": authorization["failed_summary_path"],
        "failed_summary_sha256": authorization["failed_summary_sha256"],
        "previous_qc_status": authorization["previous_qc_status"],
        "previous_retry_required": authorization["previous_retry_required"],
        "retry_authorization_path": str(authorization_path),
        "retry_authorization_sha256": file_sha256(authorization_path),
    }


def derive_triangle_retry_selection_manifest(
    failed_package_dir: str | Path,
    failed_snapshot_dir: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Authorize and deterministically derive the next complete blind attempt."""

    evidence = _retry_failure_evidence(failed_package_dir, failed_snapshot_dir)
    authorization_path = (
        Path(failed_package_dir).resolve()
        / "retry_authorizations"
        / f"authorize-{evidence['next_attempt_id']}.json"
    )
    if authorization_path.exists():
        authorization, validated_evidence = _validate_retry_authorization(
            authorization_path, verify_files=True
        )
        evidence = validated_evidence
    else:
        authorization = _retry_authorization_payload(
            evidence, created_at=_utc_now()
        )
        _atomic_json(authorization_path, authorization)
        authorization_digest = file_sha256(authorization_path)
        _atomic_bytes(
            authorization_path.with_name(authorization_path.name + ".sha256"),
            f"{authorization_digest}  {authorization_path.name}\n".encode("ascii"),
        )
        authorization, evidence = _validate_retry_authorization(
            authorization_path, verify_files=True
        )
    lineage = _retry_lineage_from_authorization(authorization_path, authorization)
    selection = _reblind_retry_selection(
        evidence["selection"],
        attempt_number=int(authorization["next_attempt_number"]),
        retry_lineage=lineage,
    )
    return selection, authorization_path, authorization


def _validate_retry_selection_manifest(
    selection: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    verify_files: bool,
) -> dict[str, Any]:
    if not verify_files:
        raise ValueError("retry selection validation requires immutable lineage files")
    lineage = selection.get("retry_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("retry selection lacks retry_lineage")
    if selection.get("retry_lineage_sha256") != canonical_sha256(dict(lineage)):
        raise ValueError("retry selection lineage hash mismatch")
    authorization_path = Path(str(lineage.get("retry_authorization_path", ""))).resolve()
    if not authorization_path.is_file() or file_sha256(authorization_path) != lineage.get(
        "retry_authorization_sha256"
    ):
        raise ValueError("retry selection authorization path/hash mismatch")
    authorization, evidence = _validate_retry_authorization(
        authorization_path, verify_files=True
    )
    expected_lineage = _retry_lineage_from_authorization(
        authorization_path, authorization
    )
    if dict(lineage) != expected_lineage:
        raise ValueError("retry selection lineage differs from exact authorization")
    resolved_manifest = Path(manifest_path).resolve()
    if resolved_manifest != Path(str(selection.get("input_manifest_path", ""))).resolve():
        raise ValueError("retry selection input-manifest path mismatch")
    if file_sha256(resolved_manifest) != selection.get("input_manifest_sha256"):
        raise ValueError("retry selection input-manifest hash mismatch")
    if dict(input_manifest) != _read_json_object(resolved_manifest):
        raise ValueError("retry validator input manifest differs from its pinned file")
    previous = evidence["selection"]
    if previous.get("input_manifest_sha256") != selection.get("input_manifest_sha256"):
        raise ValueError("retry changed the staged input manifest")
    # Validate staging independently even though the semantic selectors come
    # from the already exact-validated failed selection.
    validate_staged_input_manifest(
        input_manifest,
        manifest_path=resolved_manifest,
        verify_files=True,
    )
    expected = _reblind_retry_selection(
        previous,
        attempt_number=int(selection["listening_attempt_number"]),
        retry_lineage=expected_lineage,
    )
    if dict(selection) != expected:
        raise ValueError("retry triangle selection differs from exact reblind derivation")
    return expected


def _validate_existing_generated_acc_index(
    package: Path,
    export_root: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and reuse the immutable export created by any prior unblind."""

    json_path = export_root / "generated_acc_index.json"
    csv_path = export_root / "generated_acc_index.csv"
    index = _read_json_object(json_path)
    expected_fields = {
        "schema_version",
        "listening_attempt_id",
        "selection_sha256",
        "private_key_sha256",
        "unblind_state_sha256",
        "export_authorizing_snapshot_id",
        "sealed_responses_sha256",
        "unblinded_scores_sha256",
        "discrimination_summary_sha256",
        "answered_count",
        "collection_status",
        "row_count",
        "csv_path",
        "csv_sha256",
        "rows",
    }
    if set(index) != expected_fields or index.get("schema_version") != (
        "streammuse.melody_robustness.generated_acc_export_index.v2"
    ):
        raise ValueError("existing generated acc index schema/fields are invalid")

    key_path = package / "private" / "private_key.json"
    key = _read_json_object(key_path)
    state_path = package / "unblind_state.json"
    expected_invariants = {
        "listening_attempt_id": key.get("listening_attempt_id"),
        "selection_sha256": key.get("selection_sha256"),
        "private_key_sha256": file_sha256(key_path),
        "unblind_state_sha256": file_sha256(state_path),
        "row_count": len(expected_rows),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "rows": list(expected_rows),
    }
    for field, expected in expected_invariants.items():
        if index.get(field) != expected:
            raise ValueError(f"existing generated acc index {field} drifted")

    authorizing_id = index.get("export_authorizing_snapshot_id")
    if not isinstance(authorizing_id, str) or not authorizing_id:
        raise ValueError("existing generated acc index lacks an authorizing snapshot")
    snapshots_root = (package / "snapshots").resolve()
    authorizing_snapshot = (snapshots_root / authorizing_id).resolve()
    if authorizing_snapshot.parent != snapshots_root:
        raise ValueError("existing generated acc authorizing snapshot escapes snapshot root")
    sealed = validate_snapshot(package, authorizing_snapshot)
    unblinded, summary = validate_unblinded_summary(package, authorizing_snapshot)
    expected_snapshot_fields = {
        "sealed_responses_sha256": file_sha256(
            authorizing_snapshot / "sealed_responses.json"
        ),
        "unblinded_scores_sha256": file_sha256(
            authorizing_snapshot / "partial_unblinded_scores.json"
        ),
        "discrimination_summary_sha256": file_sha256(
            authorizing_snapshot / "partial_discrimination_summary.json"
        ),
        "answered_count": unblinded["answered_count"],
        "collection_status": summary["collection_status"],
    }
    if sealed.get("answered_count") != unblinded.get("answered_count"):
        raise ValueError("existing export authorizing snapshot score count drifted")
    for field, expected in expected_snapshot_fields.items():
        if index.get(field) != expected:
            raise ValueError(
                f"existing generated acc index authorizing snapshot {field} drifted"
            )
    return index

def export_generated_acc(
    package_dir: str | Path, *, snapshot_dir: str | Path
) -> tuple[Path, list[dict[str, Any]]]:
    """Export semantically named full MIDI and 8-second WAV after unblinding."""

    package = Path(package_dir).resolve()
    snapshot = Path(snapshot_dir).resolve()
    sealed = validate_snapshot(package, snapshot)
    state = _unblind_state(package)
    if state is None:
        raise ValueError("generated acc export is only allowed after semantic unblinding")
    unblinded, summary = validate_unblinded_summary(package, snapshot)
    key = _read_json_object(package / "private" / "private_key.json")
    sources: dict[str, dict[str, Any]] = {}
    for trial in key.get("trials", []):
        for side in ("source_a", "source_b"):
            source = trial.get(side)
            if isinstance(source, Mapping) and source.get("kind") == "formal":
                identity = ":".join(
                    str(source.get(field))
                    for field in ("song", "condition", "perturb_seed", "sample_seed")
                )
                sources.setdefault(identity, dict(source))
    export_root = package / "generated_acc_after_unblind"
    midi_root = export_root / "midi"
    wav_root = export_root / "wav_8s"
    midi_root.mkdir(parents=True, exist_ok=True)
    wav_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for identity, source in sorted(sources.items()):
        song = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source["song"]))
        pseed = "none" if source.get("perturb_seed") is None else str(source["perturb_seed"])
        stem = f"{song}__{source['condition']}__p-{pseed}__s-{source['sample_seed']}"
        original_midi = Path(str(source["source_path"])).resolve()
        canonical_midi = Path(str(source["excerpt_midi_path"])).resolve()
        canonical_wav = Path(str(source.get("canonical_wav_path", ""))).resolve()
        if not original_midi.is_file() or file_sha256(original_midi) != source.get("source_sha256"):
            raise ValueError(f"formal generated MIDI drifted before export: {identity}")
        if not canonical_midi.is_file() or file_sha256(canonical_midi) != source.get(
            "excerpt_midi_sha256"
        ):
            raise ValueError(
                f"canonical excerpt MIDI missing/corrupt before export: {identity}"
            )
        midi_out = midi_root / f"{stem}.mid"
        if midi_out.exists():
            if file_sha256(midi_out) != source.get("source_sha256"):
                raise ValueError(f"existing semantic MIDI export cannot be overwritten: {identity}")
        else:
            shutil.copyfile(original_midi, midi_out)
        if not canonical_wav.is_file() or file_sha256(canonical_wav) != source.get(
            "canonical_wav_sha256"
        ):
            raise ValueError(f"formal source canonical WAV is missing/corrupt: {identity}")
        wav_out = wav_root / f"{stem}.wav"
        if wav_out.exists():
            if file_sha256(wav_out) != source.get("canonical_wav_sha256"):
                raise ValueError(f"existing semantic WAV export cannot be overwritten: {identity}")
        else:
            shutil.copyfile(canonical_wav, wav_out)
        rows.append(
            {
                "song": source["song"],
                "condition": source["condition"],
                "perturb_seed": source.get("perturb_seed"),
                "sample_seed": source["sample_seed"],
                "run_id": source.get("run_id"),
                "attempt_id": source.get("attempt_id"),
                "operational_valid": source.get("operational_valid"),
                "source_empty": source.get("source_empty"),
                "raw_token_payload_sha256": source.get("raw_token_payload_sha256"),
                "output_event_payload_sha256": source.get(
                    "output_event_payload_sha256"
                ),
                "formal_theoretical_midi": str(original_midi),
                "formal_theoretical_midi_sha256": source.get("source_sha256"),
                "exported_midi": str(midi_out),
                "exported_midi_sha256": file_sha256(midi_out),
                "exported_excerpt_wav": str(wav_out),
                "exported_excerpt_wav_sha256": file_sha256(wav_out),
                "post_unblinding_qualitative_followup_only": True,
            }
        )
    fields = list(rows[0]) if rows else []
    csv_path = export_root / "generated_acc_index.csv"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    expected_csv = csv_buffer.getvalue().encode("utf-8")
    if csv_path.exists():
        if csv_path.read_bytes() != expected_csv:
            raise ValueError("existing generated_acc_index.csv cannot be overwritten")
    else:
        csv_path.write_bytes(expected_csv)
    index_payload = {
        "schema_version": (
            "streammuse.melody_robustness.generated_acc_export_index.v2"
        ),
        "listening_attempt_id": sealed["listening_attempt_id"],
        "selection_sha256": sealed["selection_sha256"],
        "private_key_sha256": sealed["private_key_sha256"],
        "unblind_state_sha256": file_sha256(package / "unblind_state.json"),
        "export_authorizing_snapshot_id": sealed["snapshot_id"],
        "sealed_responses_sha256": file_sha256(snapshot / "sealed_responses.json"),
        "unblinded_scores_sha256": file_sha256(
            snapshot / "partial_unblinded_scores.json"
        ),
        "discrimination_summary_sha256": file_sha256(
            snapshot / "partial_discrimination_summary.json"
        ),
        "answered_count": unblinded["answered_count"],
        "collection_status": summary["collection_status"],
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "rows": rows,
    }
    json_path = export_root / "generated_acc_index.json"
    if json_path.is_file():
        _validate_existing_generated_acc_index(package, export_root, rows)
    else:
        write_canonical_json(json_path, index_payload)
    return csv_path, rows

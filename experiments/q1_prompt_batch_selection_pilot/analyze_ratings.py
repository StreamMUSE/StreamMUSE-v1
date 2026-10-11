#!/usr/bin/env python
"""Analyze Q1 blind ratings against prompt-only symbolic music features."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from scipy.stats import spearmanr


SCORE_COLUMNS = [
    "overall_score_1_to_5",
    "melody_fit_1_to_5",
    "accompaniment_quality_1_to_5",
]
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
TIV_WEIGHTS = np.array([3.0, 8.0, 11.5, 15.0, 14.5, 7.5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--ratings-dir",
        help="Optional rating-sheet directory, relative to run-dir or absolute.",
    )
    parser.add_argument(
        "--ambiguous-x-midpoint",
        type=float,
        default=3.5,
        help="Numeric value assigned to ratings such as 3.x.",
    )
    parser.add_argument("--good-threshold", type=float, default=4.0)
    return parser.parse_args()


def parse_score(value: str, x_midpoint: float) -> tuple[float, str]:
    value = value.strip()
    if not value:
        return math.nan, "missing"
    try:
        return float(value), "exact"
    except ValueError:
        pass
    if re.fullmatch(r"\d+(?:\.\d+)?\.x", value):
        return x_midpoint, f"ambiguous_x_as_{x_midpoint:g}"
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
    if "~" in value and len(numbers) == 2:
        return sum(numbers) / 2.0, "range_midpoint"
    raise ValueError(f"unsupported rating value: {value!r}")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else math.nan


def weighted_pitch_class(notes: list[pretty_midi.Note]) -> np.ndarray:
    chroma = np.zeros(12, dtype=float)
    for note in notes:
        chroma[note.pitch % 12] += max(note.end - note.start, 1e-6)
    total = chroma.sum()
    return chroma / total if total else chroma


def estimate_scale(melody_notes: list[pretty_midi.Note]) -> tuple[int, str, set[int]]:
    chroma = weighted_pitch_class(melody_notes)
    if not chroma.any():
        return 0, "major", {0, 2, 4, 5, 7, 9, 11}
    best: tuple[float, int, str] | None = None
    for root in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = cosine_similarity(chroma, np.roll(profile, root))
            candidate = (score, root, mode)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    _, root, mode = best
    intervals = {0, 2, 4, 5, 7, 9, 11} if mode == "major" else {0, 2, 3, 5, 7, 8, 10}
    return root, mode, {(root + interval) % 12 for interval in intervals}


def tonal_centroid(chroma: np.ndarray) -> np.ndarray:
    pitch_classes = np.arange(12)
    basis = np.stack(
        [
            np.sin(7 * np.pi * pitch_classes / 6),
            np.cos(7 * np.pi * pitch_classes / 6),
            np.sin(3 * np.pi * pitch_classes / 2),
            np.cos(3 * np.pi * pitch_classes / 2),
            0.5 * np.sin(2 * np.pi * pitch_classes / 3),
            0.5 * np.cos(2 * np.pi * pitch_classes / 3),
        ]
    )
    return basis @ chroma


def tiv(chroma: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(chroma)[1:7] * TIV_WEIGHTS
    return np.concatenate([spectrum.real, spectrum.imag])


def get_named_notes(
    midi: pretty_midi.PrettyMIDI, name: str
) -> list[pretty_midi.Note]:
    matching = [
        note
        for instrument in midi.instruments
        if instrument.name.strip().lower() == name.lower()
        for note in instrument.notes
    ]
    if matching:
        return matching
    fallback_index = 0 if name.lower() == "melody" else 1
    if len(midi.instruments) > fallback_index:
        return list(midi.instruments[fallback_index].notes)
    return []


def active_on_beat(note: pretty_midi.Note, start: float, end: float) -> bool:
    return note.start < end - 1e-7 and note.end > start + 1e-7


def prompt_features(midi_path: Path, beats: int = 8, steps_per_beat: int = 4) -> dict:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    tempo_times, tempi = midi.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    beat_duration = 60.0 / tempo
    duration = beats * beat_duration
    melody = [note for note in get_named_notes(midi, "Melody") if note.start < duration]
    accompaniment = [
        note for note in get_named_notes(midi, "Accompaniment") if note.start < duration
    ]
    result = {
        "tempo_bpm": tempo,
        "acc_note_density": len(accompaniment) / beats,
        "acc_empty_beat_rate": sum(
            not any(
                active_on_beat(
                    note,
                    beat * beat_duration,
                    (beat + 1) * beat_duration,
                )
                for note in accompaniment
            )
            for beat in range(beats)
        )
        / beats,
    }
    if not accompaniment:
        result.update(
            {
                "acc_rhythm_density": 0.0,
                "acc_pitch_range": 0.0,
                "acc_average_pitch_interval": math.nan,
                "acc_pitch_class_entropy": 0.0,
                "acc_pitch_in_scale_rate": 0.0,
                "acc_qualified_note_rate": 0.0,
                "acc_average_voice_number": 0.0,
                "acc_groove_consistency": 1.0,
                "mel_acc_tonal_distance": math.nan,
                "mel_acc_tiv_similarity": math.nan,
                "mel_acc_rhythm_similarity": 0.0,
                "mel_acc_onset_js_distance": math.nan,
                "mel_acc_consonance_rate": math.nan,
            }
        )
        return result

    total_steps = beats * steps_per_beat
    acc_onset_steps = {
        min(total_steps - 1, max(0, int(round(note.start / beat_duration * steps_per_beat))))
        for note in accompaniment
    }
    result["acc_rhythm_density"] = len(acc_onset_steps) / total_steps
    pitches = np.array([note.pitch for note in accompaniment], dtype=float)
    result["acc_pitch_range"] = float(pitches.max() - pitches.min())

    grouped_pitches: dict[int, list[int]] = defaultdict(list)
    for note in accompaniment:
        onset_step = int(round(note.start / beat_duration * steps_per_beat))
        grouped_pitches[onset_step].append(note.pitch)
    onset_mean_pitches = np.array(
        [np.mean(grouped_pitches[step]) for step in sorted(grouped_pitches)]
    )
    result["acc_average_pitch_interval"] = (
        float(np.mean(np.abs(np.diff(onset_mean_pitches))))
        if len(onset_mean_pitches) > 1
        else 0.0
    )

    acc_chroma = weighted_pitch_class(accompaniment)
    nonzero = acc_chroma[acc_chroma > 0]
    result["acc_pitch_class_entropy"] = float(-np.sum(nonzero * np.log2(nonzero)))
    _, _, melody_scale = estimate_scale(melody)
    result["acc_pitch_in_scale_rate"] = sum(
        note.pitch % 12 in melody_scale for note in accompaniment
    ) / len(accompaniment)
    result["acc_qualified_note_rate"] = sum(
        (note.end - note.start) / beat_duration >= 0.75 for note in accompaniment
    ) / len(accompaniment)

    boundaries = sorted(
        {0.0, duration}
        | {
            min(duration, max(0.0, time))
            for note in accompaniment
            for time in (note.start, note.end)
            if 0.0 <= time <= duration
        }
    )
    voice_area = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2
        active = sum(note.start <= midpoint < note.end for note in accompaniment)
        voice_area += active * (end - start)
    result["acc_average_voice_number"] = voice_area / duration

    bar_steps = 4 * steps_per_beat
    bar_patterns = []
    for bar in range(max(1, beats // 4)):
        pattern = np.zeros(bar_steps, dtype=float)
        for step in acc_onset_steps:
            if bar * bar_steps <= step < (bar + 1) * bar_steps:
                pattern[step - bar * bar_steps] = 1.0
        bar_patterns.append(pattern)
    if len(bar_patterns) > 1:
        result["acc_groove_consistency"] = float(
            np.mean(
                [
                    1.0 - np.mean(left != right)
                    for left, right in zip(bar_patterns, bar_patterns[1:])
                ]
            )
        )
    else:
        result["acc_groove_consistency"] = 1.0

    mel_chroma = weighted_pitch_class(melody)
    result["mel_acc_tonal_distance"] = float(
        np.linalg.norm(tonal_centroid(mel_chroma) - tonal_centroid(acc_chroma))
    )
    result["mel_acc_tiv_similarity"] = cosine_similarity(tiv(mel_chroma), tiv(acc_chroma))

    mel_pattern = np.zeros(total_steps, dtype=float)
    acc_pattern = np.zeros(total_steps, dtype=float)
    for note in melody:
        step = min(
            total_steps - 1,
            max(0, int(round(note.start / beat_duration * steps_per_beat))),
        )
        mel_pattern[step] = 1.0
    for step in acc_onset_steps:
        acc_pattern[step] = 1.0
    result["mel_acc_rhythm_similarity"] = cosine_similarity(mel_pattern, acc_pattern)

    def within_beat_distribution(notes: list[pretty_midi.Note]) -> np.ndarray:
        counts = np.zeros(steps_per_beat, dtype=float)
        for note in notes:
            step = int(round(note.start / beat_duration * steps_per_beat))
            counts[step % steps_per_beat] += 1
        return counts / counts.sum() if counts.sum() else counts

    mel_onsets = within_beat_distribution(melody)
    acc_onsets = within_beat_distribution(accompaniment)
    midpoint = (mel_onsets + acc_onsets) / 2

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    result["mel_acc_onset_js_distance"] = math.sqrt(
        max(
            0.0,
            0.5 * kl_divergence(mel_onsets, midpoint)
            + 0.5 * kl_divergence(acc_onsets, midpoint),
        )
    )

    consonant_intervals = {0, 3, 4, 5, 7, 8, 9}
    simultaneous_intervals = []
    for mel_note in melody:
        for acc_note in accompaniment:
            if min(mel_note.end, acc_note.end) > max(mel_note.start, acc_note.start):
                simultaneous_intervals.append((mel_note.pitch - acc_note.pitch) % 12)
    result["mel_acc_consonance_rate"] = (
        sum(interval in consonant_intervals for interval in simultaneous_intervals)
        / len(simultaneous_intervals)
        if simultaneous_intervals
        else math.nan
    )
    return result


def piece_center(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] - frame.groupby("piece_id")[column].transform("mean")


def safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float, int]:
    valid = left.notna() & right.notna()
    count = int(valid.sum())
    if count < 3 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return math.nan, math.nan, count
    rho, p_value = spearmanr(left[valid], right[valid])
    return float(rho), float(p_value), count


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def load_rating_sheets(run_dir: Path, ratings_dir: Path | None) -> pd.DataFrame:
    multi_rater_files = sorted(ratings_dir.glob("*.csv")) if ratings_dir else []
    if not multi_rater_files:
        multi_rater_files = sorted((run_dir / "ratings").glob("*.csv"))
    paths = multi_rater_files or [run_dir / "blind_ratings.csv"]
    sheets = []
    for path in paths:
        sheet = pd.read_csv(path, dtype=str).fillna("")
        if "rater_id" not in sheet:
            sheet.insert(0, "rater_id", path.stem)
        missing_scores = (sheet[SCORE_COLUMNS] == "").all(axis=1)
        sheet = sheet.loc[~missing_scores].copy()
        sheets.append(sheet)
    if not sheets:
        raise RuntimeError("no rating sheets found")
    ratings = pd.concat(sheets, ignore_index=True)
    if ratings.empty:
        raise RuntimeError("rating sheets contain no completed rows")
    if ratings.duplicated(["rater_id", "blind_id"]).any():
        raise RuntimeError("duplicate (rater_id, blind_id) ratings found")
    return ratings


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings_dir = None
    if args.ratings_dir:
        ratings_dir = Path(args.ratings_dir)
        if not ratings_dir.is_absolute():
            ratings_dir = run_dir / ratings_dir
        ratings_dir = ratings_dir.resolve()
    ratings = load_rating_sheets(run_dir, ratings_dir)
    mapping = pd.read_csv(run_dir / "blind_mapping_private.csv", dtype=str)
    merged = ratings.merge(mapping, on="blind_id", validate="many_to_one")
    resolution_notes: list[str] = []
    for column in SCORE_COLUMNS:
        parsed = merged[column].map(
            lambda value: parse_score(value, args.ambiguous_x_midpoint)
        )
        merged[f"{column}_raw"] = merged[column]
        merged[column] = parsed.map(lambda item: item[0])
        merged[f"{column}_resolution"] = parsed.map(lambda item: item[1])
        for rater_id, blind_id, raw, resolution in merged.loc[
            merged[f"{column}_resolution"] != "exact",
            ["rater_id", "blind_id", f"{column}_raw", f"{column}_resolution"],
        ].itertuples(index=False, name=None):
            resolution_notes.append(
                f"{rater_id}/{blind_id} {column}: {raw} -> {resolution}"
            )

    merged["piece_id"] = merged["piece_id"].astype(str)
    merged["prompt_seed"] = merged["prompt_seed"].astype(int)
    merged["continuation_seed"] = merged["continuation_seed"].astype(int)
    merged["is_failure"] = merged["overall_score_1_to_5"] <= 0
    merged["is_good"] = merged["overall_score_1_to_5"] >= args.good_threshold
    merged.to_csv(output_dir / "sample_scores_resolved.csv", index=False)

    group_columns = ["piece_id", "npz_file", "prompt_index", "prompt_seed", "prompt_midi"]
    aggregate = (
        merged.groupby(group_columns, as_index=False)
        .agg(
            continuation_count=("blind_id", "count"),
            rater_count=("rater_id", "nunique"),
            overall_mean=("overall_score_1_to_5", "mean"),
            overall_std=("overall_score_1_to_5", "std"),
            melody_fit_mean=("melody_fit_1_to_5", "mean"),
            accompaniment_quality_mean=("accompaniment_quality_1_to_5", "mean"),
            good_rate=("is_good", "mean"),
            failure_rate=("is_failure", "mean"),
            prompt_ppl=("prompt_ppl", "first"),
        )
        .sort_values(["piece_id", "prompt_seed"])
    )
    aggregate["prompt_ppl"] = aggregate["prompt_ppl"].astype(float)
    feature_rows = []
    for row in aggregate.itertuples(index=False):
        features = prompt_features(run_dir / row.prompt_midi)
        features.update({"piece_id": row.piece_id, "prompt_seed": row.prompt_seed})
        feature_rows.append(features)
    features = pd.DataFrame(feature_rows)
    prompt_results = aggregate.merge(
        features, on=["piece_id", "prompt_seed"], validate="one_to_one"
    )
    prompt_results.to_csv(output_dir / "prompt_level_results.csv", index=False)

    target = "overall_mean"
    metric_columns = [
        "prompt_ppl",
        "acc_note_density",
        "acc_rhythm_density",
        "acc_empty_beat_rate",
        "acc_pitch_range",
        "acc_average_pitch_interval",
        "acc_pitch_class_entropy",
        "acc_pitch_in_scale_rate",
        "acc_qualified_note_rate",
        "acc_average_voice_number",
        "acc_groove_consistency",
        "mel_acc_tonal_distance",
        "mel_acc_tiv_similarity",
        "mel_acc_rhythm_similarity",
        "mel_acc_onset_js_distance",
        "mel_acc_consonance_rate",
    ]
    correlations = []
    centered_target = piece_center(prompt_results, target)
    for metric in metric_columns:
        raw_rho, raw_p, raw_n = safe_spearman(prompt_results[metric], prompt_results[target])
        centered_metric = piece_center(prompt_results, metric)
        centered_rho, centered_p, centered_n = safe_spearman(
            centered_metric, centered_target
        )
        per_piece_rhos = []
        for _, piece_rows in prompt_results.groupby("piece_id"):
            piece_rho, _, _ = safe_spearman(
                piece_rows[metric], piece_rows[target]
            )
            if not math.isnan(piece_rho):
                per_piece_rhos.append(piece_rho)
        correlations.append(
            {
                "metric": metric,
                "raw_spearman_rho": raw_rho,
                "raw_p_value": raw_p,
                "raw_n": raw_n,
                "piece_centered_spearman_rho": centered_rho,
                "piece_centered_p_value": centered_p,
                "piece_centered_n": centered_n,
                "mean_per_piece_spearman_rho": (
                    float(np.mean(per_piece_rhos)) if per_piece_rhos else math.nan
                ),
                "pieces_with_metric_variation": len(per_piece_rhos),
                "absolute_piece_centered_rho": abs(centered_rho)
                if not math.isnan(centered_rho)
                else math.nan,
            }
        )
    correlation_frame = pd.DataFrame(correlations).sort_values(
        ["pieces_with_metric_variation", "absolute_piece_centered_rho"],
        ascending=[False, False],
    )
    correlation_frame.to_csv(output_dir / "metric_correlations.csv", index=False)

    selector_decisions_path = (
        ratings_dir.parent / "_private" / "selector_decisions.csv"
        if ratings_dir
        else None
    )
    frozen_decisions = None
    if selector_decisions_path and selector_decisions_path.exists():
        frozen_decisions = pd.read_csv(selector_decisions_path, dtype={"piece_id": str})

    selector_rows = []
    for piece_id, piece_rows in prompt_results.groupby("piece_id"):
        oracle_choice = piece_rows.loc[piece_rows["overall_mean"].idxmax()]
        selector_row = {
            "piece_id": piece_id,
            "random_expected_score": piece_rows["overall_mean"].mean(),
        }
        if frozen_decisions is not None:
            piece_decisions = frozen_decisions[
                frozen_decisions["piece_id"] == str(piece_id)
            ]
            selectors_to_columns = {
                "raw_min_ppl": "selected_by_raw_min_ppl",
                "empty_filtered_min_ppl": "selected_by_empty_filtered_min_ppl",
                "music_rule": "selected_by_music_rule",
            }
            for selector_name, decision_column in selectors_to_columns.items():
                selected_rows = piece_decisions[piece_decisions[decision_column] == 1]
                if len(selected_rows) != 1:
                    raise RuntimeError(
                        f"{piece_id}: expected one {selector_name} decision"
                    )
                selected_seed = int(selected_rows.iloc[0]["prompt_seed"])
                selected_prompt = piece_rows[
                    piece_rows["prompt_seed"] == selected_seed
                ]
                if len(selected_prompt) != 1:
                    raise RuntimeError(
                        f"{piece_id}: selected Prompt seed {selected_seed} missing"
                    )
                selector_row[f"{selector_name}_selected_seed"] = selected_seed
                selector_row[f"{selector_name}_selected_score"] = selected_prompt.iloc[
                    0
                ]["overall_mean"]
        else:
            ppl_choice = piece_rows.loc[piece_rows["prompt_ppl"].idxmin()]
            selector_row["ppl_selected_seed"] = int(ppl_choice["prompt_seed"])
            selector_row["ppl_selected_score"] = ppl_choice["overall_mean"]

        selector_row["oracle_seed"] = int(oracle_choice["prompt_seed"])
        selector_row["oracle_score"] = oracle_choice["overall_mean"]
        for key in list(selector_row):
            if key.endswith("_selected_score"):
                selector_name = key.removesuffix("_selected_score")
                selector_row[f"{selector_name}_regret_to_oracle"] = (
                    oracle_choice["overall_mean"] - selector_row[key]
                )
        selector_rows.append(selector_row)
    selectors = pd.DataFrame(selector_rows)
    selectors.to_csv(output_dir / "selector_baselines.csv", index=False)
    selector_means = {
        "random": selectors["random_expected_score"].mean(),
        "oracle": selectors["oracle_score"].mean(),
    }
    for column in selectors.columns:
        if column.endswith("_selected_score"):
            selector_means[column.removesuffix("_selected_score")] = selectors[
                column
            ].mean()

    if frozen_decisions is not None:
        selector_table_columns = [
            "piece_id",
            "random_expected_score",
            "raw_min_ppl_selected_seed",
            "raw_min_ppl_selected_score",
            "empty_filtered_min_ppl_selected_seed",
            "empty_filtered_min_ppl_selected_score",
            "music_rule_selected_seed",
            "music_rule_selected_score",
            "oracle_seed",
            "oracle_score",
        ]
        selector_summary_text = (
            f"Mean scores: random `{selector_means['random']:.3f}`, raw minimum "
            f"PPL `{selector_means['raw_min_ppl']:.3f}`, empty-filtered minimum "
            f"PPL `{selector_means['empty_filtered_min_ppl']:.3f}`, frozen music "
            f"rule `{selector_means['music_rule']:.3f}`, and oracle "
            f"`{selector_means['oracle']:.3f}`."
        )
        selector_heading = "## Frozen selector comparison"
    else:
        selector_table_columns = [
            "piece_id",
            "random_expected_score",
            "ppl_selected_seed",
            "ppl_selected_score",
            "oracle_seed",
            "oracle_score",
            "ppl_regret_to_oracle",
        ]
        selector_summary_text = (
            "Across pieces, random selection has expected mean score "
            f"`{selector_means['random']:.3f}`, minimum-PPL selection scores "
            f"`{selector_means['ppl']:.3f}`, and the within-piece oracle scores "
            f"`{selector_means['oracle']:.3f}`."
        )
        selector_heading = "## PPL selector baseline"

    top_correlations = correlation_frame.head(8)
    report = [
        "# Q1 Prompt Batch Selection Pilot Analysis",
        "",
        "## Dataset",
        "",
        f"- Completed ratings: `{len(merged)}`.",
        f"- Raters: `{merged['rater_id'].nunique()}`.",
        f"- Blind samples covered: `{merged['blind_id'].nunique()}`.",
        "",
        "## Rating resolution",
        "",
        f"- `3.x` is provisionally mapped to `{args.ambiguous_x_midpoint:g}`.",
        "- Numeric ranges such as `4~3` are mapped to their midpoint.",
        f"- A good result is defined as overall score >= `{args.good_threshold:g}`.",
        "- A failure is defined as overall score <= `0`.",
        "",
    ]
    if resolution_notes:
        report.extend([f"- {note}" for note in resolution_notes])
        report.append("")
    report.extend(
        [
            "## Prompt-level raw results",
            "",
            markdown_table(
                prompt_results.sort_values(["piece_id", "prompt_seed"]),
                [
                    "piece_id",
                    "prompt_seed",
                    "overall_mean",
                    "overall_std",
                    "melody_fit_mean",
                    "accompaniment_quality_mean",
                    "good_rate",
                    "failure_rate",
                    "prompt_ppl",
                    "acc_note_density",
                    "acc_empty_beat_rate",
                ],
            ),
            "",
            selector_heading,
            "",
            markdown_table(
                selectors,
                selector_table_columns,
            ),
            "",
            selector_summary_text,
            "",
            "## Strongest exploratory correlations",
            "",
            "The piece-centered correlation removes each piece's mean before ranking.",
            "The per-piece value computes a separate 3-prompt rank correlation for each",
            "piece and averages only pieces where that metric varies.",
            "",
            markdown_table(
                top_correlations,
                [
                    "metric",
                    "piece_centered_spearman_rho",
                    "piece_centered_p_value",
                    "mean_per_piece_spearman_rho",
                    "pieces_with_metric_variation",
                    "raw_spearman_rho",
                ],
            ),
            "",
            "A metric supported by only one varying piece (for example empty-beat rate)",
            "is useful as a failure guard but is not evidence of a general ranking rule.",
            "",
            "## Scope",
            "",
            (
                f"This analysis contains {len(prompt_results)} prompts from "
                f"{prompt_results['piece_id'].nunique()} pieces. Correlations are"
            ),
            "exploratory and must not be treated as statistically established selection",
            "rules. The next experiment should add pieces and raters before fitting a",
            "combined selector.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()

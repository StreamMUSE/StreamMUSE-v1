"""Existing CR/UR plus explicitly defined descriptive metrics, without fake GT."""

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def clipped(notes, start, end):
    import pretty_midi
    return [pretty_midi.Note(n.velocity, n.pitch, max(start, n.start), min(end, n.end))
            for n in notes if n.start < end and n.end > start]


def describe(notes, start, end, step_seconds):
    """Occupancy counts sounding notes; exact adjacent patterns exclude empty pairs."""
    spans = sorted((max(start, n.start), min(end, n.end)) for n in notes
                   if n.start < end and n.end > start)
    occupied, right = 0., start
    for a, b in spans:
        occupied += max(0., b - max(a, right))
        right = max(right, b)
    beat = 4 * step_seconds
    onsets = [n for n in notes if start <= n.start < end]
    patterns = []
    for index in range(round((end - start) / beat)):
        left = start + index * beat
        pattern = tuple(sorted((n.pitch, round((n.start - left) / step_seconds, 6),
                                round((n.end - n.start) / step_seconds, 6))
                               for n in notes if left - 1e-8 <= n.start < left + beat - 1e-8))
        patterns.append(pattern)
    pairs = [(a, b) for a, b in zip(patterns, patterns[1:]) if a and b]
    return {'onset_count': len(onsets), 'nonempty': int(bool(onsets)),
            'onsets_per_beat': len(onsets) / ((end - start) / beat),
            'sounding_time_coverage': occupied / (end - start),
            'adjacent_nonempty_pairs': len(pairs),
            'adjacent_exact_onset_pattern_repeat': sum(a == b for a, b in pairs) / len(pairs) if pairs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path, required=True)
    args = parser.parse_args()
    import pretty_midi
    source = args.eval_root / 'evaluate_accompaniment_metrics.py'
    spec = importlib.util.spec_from_file_location('frozen_music_eval', source)
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    plan = json.loads((args.root / 'plan.json').read_text())
    complete = json.loads((args.root / 'complete.json').read_text())
    assert len(complete) == len(plan['pieces']) * len(plan['seeds'])
    rows = []
    step_seconds = 60 / plan['export_bpm'] / 4
    for piece in plan['pieces']:
        for seed in plan['seeds']:
            directory = args.root / piece['piece_id'] / f'seed{seed}'
            pair = json.loads((directory / 'comparison.json').read_text())
            for condition in ('baseline', 'repair'):
                midi = pretty_midi.PrettyMIDI(str(directory / f'{condition}_full.mid'))
                melody = [n for i in midi.instruments if i.name == 'Melody' for n in i.notes]
                acc = [n for i in midi.instruments if i.name == 'Accompaniment' for n in i.notes]
                for name, tick in (('primary_after_repair', 84), ('secondary_postjoin', 32)):
                    start, end = tick * step_seconds, piece['stop_tick'] * step_seconds
                    mel_clip, acc_clip = clipped(melody, start, end), clipped(acc, start, end)
                    harmonic = metrics._compute_harmonic_consonance(mel_clip, acc_clip, [0, 3, 4, 7, 8, 9])
                    rows.append({'piece_id': piece['piece_id'], 'title': piece['title'], 'seed': seed,
                        'condition': condition, 'window': name, 'start_tick': tick,
                        'end_tick': piece['stop_tick'], 'repair_status': pair['repair_status'],
                        'CR': harmonic['consonant_ratio'] if harmonic else None,
                        'UR': harmonic['unsupported_ratio'] if harmonic else None,
                        'harmonic_metric_defined': harmonic is not None,
                        **describe(acc, start, end, step_seconds)})
    with (args.root / 'music_metrics.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ['CR', 'UR', 'onsets_per_beat', 'sounding_time_coverage', 'adjacent_exact_onset_pattern_repeat']
    paired = []
    for piece in plan['pieces']:
        for window in ('primary_after_repair', 'secondary_postjoin'):
            item = {'piece_id': piece['piece_id'], 'title': piece['title'], 'window': window}
            for key in keys:
                deltas = []
                for seed in plan['seeds']:
                    subset = {r['condition']: r for r in rows if r['piece_id'] == piece['piece_id']
                              and r['seed'] == seed and r['window'] == window}
                    a, b = subset['baseline'][key], subset['repair'][key]
                    if a is not None and b is not None:
                        deltas.append(b - a)
                item[key + '_delta_repair_minus_baseline'] = statistics.mean(deltas) if deltas else None
                item[key + '_defined_seed_pairs'] = len(deltas)
            paired.append(item)
    summary = []
    for window in ('primary_after_repair', 'secondary_postjoin'):
        for condition in ('baseline', 'repair'):
            subset = [r for r in rows if r['condition'] == condition and r['window'] == window]
            item = {'window': window, 'condition': condition, 'all_trials': len(subset),
                    'nonempty_trials': sum(r['nonempty'] for r in subset),
                    'harmonic_defined_trials': sum(r['harmonic_metric_defined'] for r in subset)}
            for key in keys:
                by_piece = defaultdict(list)
                for r in subset:
                    if r[key] is not None:
                        by_piece[r['piece_id']].append(r[key])
                item[key] = statistics.mean(statistics.mean(v) for v in by_piece.values()) if by_piece else None
                item[key + '_defined_pieces'] = len(by_piece)
            summary.append(item)
    output = {'summary': summary, 'paired_piece_deltas': paired,
        'repair_applied_pairs': sum(p['repair_status'] == 'applied' for p in complete),
        'all_pairs': len(complete), 'eval_source': str(source),
        'eval_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'definitions': {
            'CR_UR': 'Existing _compute_harmonic_consonance; duration-weighted Melody notes, any overlap, intervals 0/3/4/7/8/9.',
            'empty_output': 'Original function returns null on empty accompaniment or melody. Retain trial, report null and defined counts; never impute a new CR/UR definition.',
            'aggregation': 'Mean seeds within piece, then mean pieces. Paired deltas use jointly defined seeds, counts explicit.',
            'coverage': 'Union of sounding accompaniment intervals / evaluation duration.',
            'density': 'Accompaniment onsets within window / window beats.',
            'repetition': 'Adjacent beats with nonempty onset patterns only: exact equality of pitch, local onset, note duration. Descriptive, not a previously published metric.',
            'export': 'Raw returned events; no playback lateness. Held notes closed at fixed song end for MIDI only.',
            'reference': 'No verified aligned GT; FMD/JSD intentionally not computed.'}}
    save(args.root / 'music_analysis.json', output)
    text = ['# Causal Offline History-Repair Pilot', '',
            'Shared Prompt per pair; same causal Melody; no clock deadlines or future Melody.',
            'One partial repair of [64,80), applied before tick 84. Original emitted events are not rewritten.',
            'Primary window: [84,song end); secondary: [32,song end).', '',
            f"Applied repairs: {output['repair_applied_pairs']}/{len(complete)}. No quality gate or best-candidate selection.", '',
            '| Window | Condition | Nonempty | CR | UR | Sounding coverage |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        fmt = lambda value: '--' if value is None else f'{value:.4f}'
        text.append(f"| {r['window']} | {r['condition']} | {r['nonempty_trials']}/{r['all_trials']} | {fmt(r['CR'])} | {fmt(r['UR'])} | {fmt(r['sounding_time_coverage'])} |")
    text += ['', 'CR/UR retain the existing evaluator definition; null trials and defined counts are in CSV/JSON.',
             'These metrics do not establish perceptual improvement. Listen to the paired full-song MIDI files.',
             'This is not a realtime delivery experiment and reports no ISR.']
    (args.root / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

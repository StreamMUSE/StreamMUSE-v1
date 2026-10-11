"""Audit full-song delivery using the existing, hash-pinned ISR evaluator."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import mido

HASHES = {
    'system_trace_v2.py': '1f3b8786d6e0e9b4cdb575c248b03e4aed1494173942465d5068a7a3ecb9207a',
    'beat_delivery.py': '628c8f975b32dc0ddd7f5788084c7b43da3c7e8ea234f105f665b9197e9abd07',
}


def read(path):
    return json.loads(path.read_text())


def playback_events(path):
    midi = mido.MidiFile(path)
    result = []
    for track in midi.tracks:
        if not any(m.type == 'track_name' and m.name.lower().startswith('accompaniment') for m in track):
            continue
        absolute = 0
        for msg in track:
            absolute += msg.time
            if msg.type not in ('note_on', 'note_off'):
                continue
            tick = absolute * 4 / midi.ticks_per_beat
            assert abs(tick - round(tick)) < 1e-6, tick
            result.append(dict(type='note_on' if msg.type == 'note_on' and msg.velocity else 'note_off',
                               tick=round(tick), pitch=msg.note, velocity=msg.velocity))
    return result


def onsets(events, start, end):
    return Counter((e['tick'], e['pitch']) for e in events
                   if e['type'] == 'note_on' and e.get('velocity', 100) > 0 and start <= e['tick'] < end)


def note_ends(events):
    active, closed = {}, {}
    for e in events:
        if e['type'] == 'note_on' and e.get('velocity', 100) > 0:
            active[e['pitch']] = e['tick']
        elif e['pitch'] in active:
            start = active.pop(e['pitch'])
            closed[start, e['pitch']] = e['tick']
    return closed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path, required=True)
    args = parser.parse_args()
    for name, expected in HASHES.items():
        assert hashlib.sha256((args.eval_root / 'src/eval_toolkit' / name).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(args.eval_root / 'src'))
    from eval_toolkit.system_trace_v2 import evaluate_session
    plan = read(args.root / 'plan.json')
    trials = read(args.root / 'complete.json')
    end = plan['case']['stop_tick']
    rows, frame_rows, traces, repair_audits = [], [], {}, []
    for trial in trials:
        session = Path(trial['session_dir'])
        _, frames = evaluate_session(session, observation_tick=32, end_tick=end)
        assert [f['tick'] for f in frames] == list(range(32, end))
        seconds_per_tick = 60 / plan['contract']['playback_bpm'] / 4
        assert all(abs((f['deadline_time_s'] - f['nominal_tick_time_s']) / seconds_per_tick - .1) < 1e-5 for f in frames)
        raw = read(session / 'prompt_continuation_raw_history.json')
        played = playback_events(session / 'combined.mid')
        traces[trial['condition'], trial['seed']] = read(session / 'prompt_continuation_model_trace.json')
        shutil.copy2(session / 'theoretical_model.mid', args.root / 'listening' /
                     f"{trial['condition']}_s{trial['seed']}_theoretical_model.mid")
        raw_on, played_on = onsets(raw, 32, end), onsets(played, 32, end)
        matched = sum((raw_on & played_on).values())
        raw_ends, played_ends = note_ends(raw), note_ends(played)
        comparable = [key for key in (raw_on & played_on) if key in raw_ends and key in played_ends
                      and raw_on[key] == 1 and played_on[key] == 1]
        differences = [dict(tick=k[0], pitch=k[1], raw_off=raw_ends[k], playback_off=played_ends[k])
                       for k in comparable if raw_ends[k] != played_ends[k]]
        (args.root / trial['condition'] / f"seed{trial['seed']}" / 'note_off_differences.json').write_text(
            json.dumps(differences, indent=2) + '\n')
        audit = read(args.root / trial['condition'] / f"seed{trial['seed']}" / 'history_repair_audit.json')
        if trial['condition'] == 'repair':
            repair_audits.append(dict(seed=trial['seed'], decisions=[
                {k: r[k] for k in ('status', 'reason', 'result', 'error') if k in r}
                for r in audit['controller']]))
        for applied in audit['applied']:
            if 'empty_guard_before' in applied:
                assert applied['empty_guard_before'] == applied['empty_guard_after']
            restored = applied['candidate']['provenance']['metadata']['restored_constraint_snapshot']
            assert restored == applied['ticket']['constraint_snapshot']
        row = dict(condition=trial['condition'], seed=trial['seed'],
                   repair_applications=trial['applied'], on_time_frames=sum(f['on_time'] for f in frames),
                   total_frames=len(frames), ISR_f=sum(f['on_time'] for f in frames) / len(frames),
                   raw_postjoin_onsets=sum(raw_on.values()), playback_postjoin_onsets=sum(played_on.values()),
                   matched_onsets=matched, raw_onsets_not_in_playback=sum(raw_on.values()) - matched,
                   playback_onsets_not_in_raw=sum(played_on.values()) - matched,
                   matched_closed_unique_notes=len(comparable), different_note_offs=len(differences))
        after_target = [f for f in frames if f['tick'] >= plan['bar'][1] + 4]
        boundary = [f for f in frames if plan['bar'][1] + 4 <= f['tick'] < plan['bar'][1] + 8]
        row['ISR_f_after_target'] = sum(f['on_time'] for f in after_target) / len(after_target)
        row['target_next_beat_on_time_frames'] = sum(f['on_time'] for f in boundary)
        row['target_next_beat_total_frames'] = len(boundary)
        rows.append(row)
        frame_rows.extend(dict(condition=trial['condition'], seed=trial['seed'], tick=f['tick'], on_time=f['on_time']) for f in frames)
    for filename, values in [('delivery.csv', rows), ('frames.csv', frame_rows)]:
        with (args.root / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = []
    for condition in plan['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        summary.append(dict(condition=condition, trials=len(selected),
            ISR_f=sum(r['on_time_frames'] for r in selected) / sum(r['total_frames'] for r in selected),
            applied=sum(r['repair_applications'] for r in selected)))
    paired = []
    for seed in plan['seeds']:
        a = {g['generation_start_tick']: g for g in traces['baseline', seed]['continuation_generations']}
        b = {g['generation_start_tick']: g for g in traces['repair', seed]['continuation_generations']}
        common = sorted(a.keys() & b.keys())
        item = dict(seed=seed, common_calls=len(common), baseline_calls=len(a), repair_calls=len(b),
                    baseline_only_ticks=sorted(a.keys() - b.keys()), repair_only_ticks=sorted(b.keys() - a.keys()))
        for field in ('part0_token_digest', 'part0_roll_digest', 'prompt_token_digest', 'raw_token_digest'):
            assert all(field in a[t] and field in b[t] for t in common)
            different = [t for t in common if a[t][field] != b[t][field]]
            item[field] = dict(equal_calls=len(common) - len(different), first_difference_tick=next(iter(different), None))
        paired.append(item)
    candidate_timings = []
    for path in sorted((args.root / 'repair/worker').glob('candidate_*.json')):
        candidate = read(path)
        tail = set(candidate['ticket']['initial_active_pitches'])
        for event in candidate.get('candidate', {}).get('events', []):
            if event['type'] == 'note_on':
                tail.add(event['pitch'])
            else:
                tail.discard(event['pitch'])
        projection = candidate.get('candidate', {}).get('provenance', {}).get('boundary_projection')
        candidate_timings.append(dict(file=path.name, elapsed_s=candidate['completed_at'] - candidate['started_at'],
                                     success='candidate' in candidate, error=candidate.get('error'),
                                     boundary_policy=candidate.get('boundary_policy', 'strict'),
                                     selected_interior_notes=projection['selected_interior_notes'] if projection else None,
                                     original_tail=candidate['ticket']['active_pitches_at_end'],
                                     candidate_tail=sorted(tail)))
    output = dict(summary=summary, trials=rows, paired_generation=paired, repair_audits=repair_audits,
        repair_policy=plan.get('repair_policy', 'strict'),
        candidate_timings=candidate_timings, eval_hashes=HASHES, window_ticks=[32, end],
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        secondary_window_ticks=[plan['bar'][1] + 4, end],
        target_next_beat_ticks=[plan['bar'][1] + 4, plan['bar'][1] + 8],
        definition='Original evaluator: earliest valid decision <= nominal + 0.1 tick; valid REST included.',
        limitations=['One piece and three seeds; no general quality claim.',
                    'Raw history is scheduler-returned generation, not the repaired backend conditioning history.',
                    'Onset comparison ignores velocity; offset comparison uses matched closed unique onsets only.',
                    'Export-time terminal note closure must be distinguished from internal offset shifts.'])
    (args.root / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Full-song guarded history-repair pilot', '',
        'One piece (Tonghua), three seeds per condition; this is NOT a quality benchmark.', '',
        '## Configuration', '',
        '- Production MIDI CLI -> HTTP backend -> scheduler -> actual playback exports.',
        '- Playback 90 BPM; model conditioning 80 BPM; temperature 1.1, top-p .95, top-k 50, repetition penalty 1.',
        '- P+C Rule If-Else N=10 in both groups; tonal constraint and EMPTY guard enabled.',
        '- One fixed historical bar [64,80) ticks; private repair seed 101. No automatic quality classifier.',
        '- Repair policy: ' + plan.get('repair_policy', 'strict') + '.',
        '- Only future Continuation context can change; already-played events are never rewritten.',
        '- ISR_f uses the original hash-verified evaluator over ticks [32,444), including valid REST.', '',
        '## Results', '', '| Condition | Seed | Applied | ISR_f | Raw onsets | Played onsets | Missing raw onsets |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['condition']} | {row['seed']} | {row['repair_applications']} | {row['ISR_f']:.4f} | "
                     f"{row['raw_postjoin_onsets']} | {row['playback_postjoin_onsets']} | {row['raw_onsets_not_in_playback']} |")
    lines += ['', 'See `analysis.json` for every repair decision, Melody/full-input/raw digest comparisons, and worker durations.',
              '`delivery.csv` and `frames.csv` retain the numerical results. Each trial has `note_off_differences.json`.',
              '`listening/*_playback.mid` is actual output; `*_theoretical_model.mid` is the existing model-history export.', '',
              '## Limits', '',
              'A successful application only verifies the history replacement path, not improved music quality.',
              'Timing comparisons have one observation per seed/condition and include wall-clock variability.',
              'Onset agreement does not prove offset agreement. Terminal note closure is not automatically an internal timing bug.',
              'No scheduler, late policy, quantization, Web deployment, training weights, or persistent cross-request KV cache was changed.', '']
    if plan.get('repair_policy') == 'preserve_cross_boundary':
        lines += ['## Partial-repair semantics', '',
                  'This is not whole-bar replacement. Original notes crossing the start or end are retained.',
                  'Only fully interior sampled notes that do not overlap retained notes of the same pitch are used.',
                  'No pitch, duration, or velocity is modified; no synthetic note-off is inserted.',
                  'The resulting context combines original boundary notes and selected candidate notes.',
                  'Original unprojected candidates and their token traces remain in the worker audit.',
                  'Projected events are explicitly NOT labeled as a direct decode of those sampled tokens.', '']
    (args.root / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

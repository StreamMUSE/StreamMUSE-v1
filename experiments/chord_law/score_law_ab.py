"""Score the strong-beat boost A/B/C exports: CR/UR on [84, stop_tick).

Same frozen evaluator (_compute_harmonic_consonance) and window as the
three-arm bundle; ch0=melody, ch1=accompaniment.
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

EVAL = '/data/home/yuanxin/StreamMUSE-eval-metrics-20260902/evaluate_accompaniment_metrics.py'


def load_metrics():
    spec = importlib.util.spec_from_file_location('law_eval', EVAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules['law_eval'] = module
    spec.loader.exec_module(module)
    return module


class N:
    __slots__ = ('start', 'end', 'pitch')

    def __init__(self, a, b, p):
        self.start, self.end, self.pitch = a, b, p


def score(path, stop_tick, step, metrics, start_tick=84):
    """CR/UR over event ticks [start_tick, stop_tick).

    step is seconds per event tick (quarter-beat). Note times convert via
    beats (a/tpb) * seconds-per-beat = a/tpb * step*4 — not a/tpb*step,
    which silently shrank the window by 4x in every score before 20261012
    (user10 exports were effectively scored on [336, stop), not [84, stop)).
    """
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    beat_sec = step * 4
    mel, acc, open_n = [], [], {}
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                open_n[(msg.channel, msg.note)] = t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in open_n:
                    a = open_n.pop(key)
                    note = N(a / tpb * beat_sec, t / tpb * beat_sec, msg.note)
                    (mel if msg.channel == 0 else acc).append(note)
    s, e = start_tick * step, stop_tick * step
    for n in mel + acc:
        n.end = max(n.end, n.start + 1e-6)
    mel_c = [n for n in mel if n.start < e and n.end > s]
    acc_c = [n for n in acc if n.start < e and n.end > s]
    harm = metrics._compute_harmonic_consonance(mel_c, acc_c, [0, 3, 4, 7, 8, 9])
    if harm is None:
        return None
    return {'CR': harm['consonant_ratio'], 'UR': harm['unsupported_ratio'],
            'melody_n': len(mel_c), 'acc_n': len(acc_c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / 'plan.json').read_text())
    step = 60.0 / (plan['export_bpm'] * 4)
    start_tick = plan.get('primary_metric_start_tick', 84)
    metrics = load_metrics()
    rows = []
    for piece in plan['pieces']:
        for seed in plan['seeds']:
            for arm in plan['arms']:
                path = args.root / piece['piece_id'] / f'seed{seed}' / arm / 'full.mid'
                if not path.exists():
                    continue
                s = score(path, piece['stop_tick'], step, metrics, start_tick)
                if s is None:
                    rows.append({'piece_id': piece['piece_id'], 'seed': seed, 'arm': arm,
                                 'error': 'empty_window'})
                else:
                    rows.append({'piece_id': piece['piece_id'], 'title': piece['title'],
                                 'seed': seed, 'arm': arm, **s})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))
    print(f"{'arm':20} {'CR':>8} {'UR':>8}  n")
    for arm in plan['arms']:
        sub = [r for r in rows if r['arm'] == arm and 'CR' in r]
        if not sub:
            print(f'{arm:20}  NO DATA')
            continue
        print(f"{arm:20} {statistics.mean(r['CR'] for r in sub):8.4f} "
              f"{statistics.mean(r['UR'] for r in sub):8.4f}  {len(sub)}")
    per_piece = {}
    for r in rows:
        if 'CR' not in r:
            continue
        per_piece.setdefault(r['piece_id'], {}).setdefault(r['arm'], []).append(r['CR'])
    print('\nper-piece CR (seed-mean):')
    for pid in sorted(per_piece):
        line = ' '.join(
            f"{arm.split('_')[0]}={statistics.mean(v):.3f}"
            for arm, v in sorted(per_piece[pid].items()))
        print(f'  {pid}: {line}')


if __name__ == '__main__':
    main()

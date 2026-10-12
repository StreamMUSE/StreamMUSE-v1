"""Paired baseline-vs-oracle summary from score_law_ab.py output.

Reads scores.json rows and reports aggregate + per-piece paired deltas
(oracle - baseline) on the [84, stop) window, same convention as the
three-arm bundle analysis.
"""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.scores.read_text()) if 'CR' in r]
    arms = sorted({r['arm'] for r in rows})
    print(f"{'arm':12} {'CR':>8} {'UR':>8}  n")
    for arm in arms:
        sub = [r for r in rows if r['arm'] == arm]
        print(f"{arm:12} {statistics.mean(r['CR'] for r in sub):8.4f} "
              f"{statistics.mean(r['UR'] for r in sub):8.4f}  {len(sub)}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r['piece_id'], r['seed']), {})[r['arm']] = r
    keys = sorted(k for k in by_key
                  if 'baseline' in by_key[k] and 'oracle' in by_key[k])
    deltas = [(by_key[k]['oracle']['CR'] - by_key[k]['baseline']['CR'],
               by_key[k]['oracle']['UR'] - by_key[k]['baseline']['UR'],
               k) for k in keys]
    if deltas:
        dcr = [d for d, _, _ in deltas]
        dur = [u for _, u, _ in deltas]
        win = sum(d > 0 for d in dcr)
        loss = sum(d < 0 for d in dcr)
        print(f'\npaired (n={len(deltas)}): dCR {statistics.mean(dcr):+.4f} '
              f'({win}W/{loss}L/{len(deltas)-win-loss}T), dUR {statistics.mean(dur):+.4f}')
        per_piece = {}
        for d, u, (pid, seed) in deltas:
            per_piece.setdefault(pid, []).append((d, u))
        print('per-piece dCR (seed-mean):')
        for pid in sorted(per_piece):
            ds = [d for d, _ in per_piece[pid]]
            print(f'  {pid}: {statistics.mean(ds):+.3f}')
    if args.output:
        args.output.write_text(json.dumps(
            {'aggregate': {arm: {'CR': statistics.mean(r['CR'] for r in rows if r['arm'] == arm),
                                 'UR': statistics.mean(r['UR'] for r in rows if r['arm'] == arm),
                                 'n': sum(1 for r in rows if r['arm'] == arm)} for arm in arms},
             'paired_deltas': [{'piece_id': k[0], 'seed': k[1], 'dCR': d, 'dUR': u}
                               for d, u, k in deltas]}, indent=1))


if __name__ == '__main__':
    main()

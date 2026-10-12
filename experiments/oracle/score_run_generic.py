"""Score a run_offline_paired / run_oracle results root under the plan window.

Reads <root>/plan.json (pieces, export_bpm, primary_metric_start_tick), scores
every <piece_id>/seed*/<arm>_full.mid found on disk, writes <root>/scores.json,
and prints per-arm CR/UR plus paired coverage W/L/T for every two-arm combo.

Usage: python -m experiments.oracle.score_run_generic --root <results_dir>
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.chord_law.score_law_ab import load_metrics, score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()

    metrics = load_metrics()
    plan = json.loads((args.root / 'plan.json').read_text())
    step = 60.0 / (plan['export_bpm'] * 4)
    start_tick = plan.get('primary_metric_start_tick', 84)
    pieces = {p['piece_id']: p for p in plan['pieces']}

    rows = []
    for pid_dir in sorted(args.root.iterdir()):
        if not pid_dir.is_dir() or pid_dir.name not in pieces:
            continue
        for seed_dir in sorted(pid_dir.glob('seed*')):
            # run_offline_paired: seed*/<arm>_full.mid; run_oracle: seed*/<arm>/full.mid
            paths = list(seed_dir.glob('*_full.mid'))
            paths += list(seed_dir.glob('*/full.mid'))
            for path in sorted(paths):
                if path.parent.name == seed_dir.name:
                    arm = path.stem[:-len('_full')]
                else:
                    arm = path.parent.name
                s = score(path, pieces[pid_dir.name]['stop_tick'], step, metrics,
                          start_tick=start_tick)
                if s:
                    rows.append({'piece_id': pid_dir.name, 'seed': int(seed_dir.name[4:]),
                                 'arm': arm, **s})
    (args.root / 'scores.json').write_text(json.dumps(rows, indent=1))

    print(f'window [{start_tick}, per-piece stop)  n_rows={len(rows)}')
    arms = sorted({r['arm'] for r in rows})
    by_key = {}
    for r in rows:
        by_key.setdefault((r['piece_id'], r['seed']), {})[r['arm']] = r
    for arm in arms:
        sub = [v[arm] for v in by_key.values() if arm in v]
        print(f"{arm:10} CR {statistics.mean(r['CR'] for r in sub):.4f} "
              f"UR {statistics.mean(r['UR'] for r in sub):.4f}  n={len(sub)}")
    for i, arm_a in enumerate(arms):
        for arm_b in arms[i + 1:]:
            pairs = [(v[arm_a], v[arm_b]) for v in by_key.values()
                     if arm_a in v and arm_b in v]
            if not pairs:
                continue
            d = [b['CR'] - a['CR'] for a, b in pairs]
            w = sum(1 for x in d if x > 1e-9)
            l = sum(1 for x in d if x < -1e-9)
            du = [b['UR'] - a['UR'] for a, b in pairs]
            print(f"paired {arm_a} -> {arm_b}: dCR {statistics.mean(d):+.4f} "
                  f"({w}W/{l}L/{len(pairs) - w - l}T)  dUR {statistics.mean(du):+.4f}")


if __name__ == '__main__':
    main()

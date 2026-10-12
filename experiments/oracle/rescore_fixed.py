"""Re-score historical user10 batches under the corrected [84, stop) window.

The pre-20261012 scorer shrank note times 4x, effectively scoring [336, stop).
This re-scores: chord_law A/B/C runs, the three-arm bundle (realtime /
offlineP+C / +repair), and the offline paired baseline/repair batch, writing
one scores.json per source. Paired deltas remain the primary evidence; these
numbers replace the absolute values in reports.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment')
from experiments.chord_law.score_law_ab import load_metrics, score  # noqa: E402

OUT = Path('/data/home/yuanxin/experiments/rescore_fixed_window_20261012')
OFFLINE_PLAN = Path('/data/home/yuanxin/experiments/bar_history_offline_user10_20261007/run_IkyAZT/results/plan.json')
BUNDLE = Path('/data/home/yuanxin/experiments/three_arm_bundle_20261008')
LAW_RUNS = [Path('/data/home/yuanxin/experiments/chord_law_20261008/run_fwSUhK/results'),
            Path('/data/home/yuanxin/experiments/chord_law_20261008/run_tTY9PI/results')]
OFFLINE = OFFLINE_PLAN.parent


def summarize(rows, label):
    print(f'\n== {label} ==')
    arms = sorted({r['arm'] for r in rows if 'CR' in r})
    for arm in arms:
        sub = [r for r in rows if r['arm'] == arm and 'CR' in r]
        if sub:
            print(f"{arm:22} CR {statistics.mean(r['CR'] for r in sub):.4f} "
                  f"UR {statistics.mean(r['UR'] for r in sub):.4f}  n={len(sub)}")


def main():
    metrics = load_metrics()
    plan = json.loads(OFFLINE_PLAN.read_text())
    step = 60.0 / (plan['export_bpm'] * 4)
    pieces = {p['piece_id']: p for p in plan['pieces']}

    rows = []
    for path in sorted(BUNDLE.glob('*/s*_*.mid')):
        stem = path.stem  # s{seed}_{arm}
        seed_s, arm = stem.split('_', 1)
        pid = path.parent.name.split('_')[0]
        piece = pieces[pid]
        s = score(path, piece['stop_tick'], step, metrics)
        if s:
            rows.append({'piece_id': pid, 'seed': int(seed_s[1:]), 'arm': arm, **s})
    (OUT / 'bundle_fixed.json').write_text(json.dumps(rows, indent=1))
    summarize(rows, 'three-arm bundle [84, stop) fixed')

    rows = []
    for pid_dir in sorted(OFFLINE.iterdir()):
        if not pid_dir.is_dir():
            continue
        for cond in ('baseline', 'repair'):
            for seed_dir in pid_dir.glob('seed*'):
                path = seed_dir / f'{cond}_full.mid'
                if not path.exists():
                    continue
                s = score(path, pieces[pid_dir.name]['stop_tick'], step, metrics)
                if s:
                    rows.append({'piece_id': pid_dir.name, 'seed': int(seed_dir.name[4:]),
                                 'arm': cond, **s})
    (OUT / 'offline_paired_fixed.json').write_text(json.dumps(rows, indent=1))
    summarize(rows, 'offline paired baseline/repair [84, stop) fixed')

    for run in LAW_RUNS:
        lp = run / 'plan.json'
        if not lp.exists():
            continue
        lplan = json.loads(lp.read_text())
        lstep = 60.0 / (lplan['export_bpm'] * 4)
        lpieces = {p['piece_id']: p for p in lplan['pieces']}
        rows = []
        for pid_dir in sorted(run.iterdir()):
            if not pid_dir.is_dir() or pid_dir.name not in lpieces:
                continue
            for seed_dir in pid_dir.glob('seed*'):
                for arm_dir in sorted(seed_dir.iterdir()):
                    path = arm_dir / 'full.mid'
                    if not path.exists():
                        continue
                    s = score(path, lpieces[pid_dir.name]['stop_tick'], lstep, metrics)
                    if s:
                        rows.append({'piece_id': pid_dir.name, 'seed': int(seed_dir.name[4:]),
                                     'arm': arm_dir.name, **s})
        (OUT / f'{run.parent.name}_{run.name}_fixed.json').write_text(json.dumps(rows, indent=1))
        summarize(rows, f'{run.parent.name}/{run.name} A/B/C [84, stop) fixed')


if __name__ == '__main__':
    main()

"""BEAT-40 exam for the match reward model.

Known-answer test: within each piece/seed, melody is identical between the
baseline and oracle arms — only the accompaniment differs. A useful judge
must rank reward(oracle) > reward(baseline) (target >= 80%), and its
per-pair delta should correlate with the CR delta the frozen evaluator
measures. Run over one or more run_oracle results roots.

Usage:
  python -m experiments.rl_reward.eval_exam --ckpt <pt> \
    --pairs-root /data/.../oracle_beat40_20261012/results \
    --pairs-root /data/.../oraclev2_melfirst_20261013/results \
    --output exam.json
"""

import argparse
import json
from pathlib import Path

import torch

from experiments.rl_reward.reward_model import MatchRewardModel
from experiments.rl_reward.reward_data import midi_pair_rolls


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--pairs-root', type=Path, action='append', required=True)
    parser.add_argument('--beats', type=int, default=32,
                        help='scoring window length in beats (default 32 = train windows)')
    parser.add_argument('--beat-offset', type=int, default=0,
                        help='first beat of the scoring window in the piece')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MatchRewardModel().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    rows = []
    for root in args.pairs_root:
        plan = json.loads((root / 'plan.json').read_text())
        seeds = plan['seeds']
        for piece in plan['pieces']:
            pid = piece['piece_id']
            for seed in seeds:
                arms = {}
                for arm in ('baseline', 'oracle'):
                    path = root / pid / f'seed{seed}' / arm / 'full.mid'
                    if not path.exists():
                        continue
                    mel, acc = midi_pair_rolls(path, beats=args.beats,
                                               start_beat=args.beat_offset)
                    pair = (mel.unsqueeze(0).to(device), acc.unsqueeze(0).to(device))
                    mel_e, acc_e = model.embed(*pair)
                    disc_in = torch.cat([mel_e, acc_e, mel_e * acc_e], dim=-1)
                    arms[arm] = {'cosine': float((mel_e * acc_e).sum()),
                                 'disc': float(model.discriminator(disc_in))}
                if len(arms) != 2:
                    continue
                scores_path = root / 'scores.json'
                cr = {}
                if scores_path.exists():
                    for r in json.loads(scores_path.read_text()):
                        if r['piece_id'] == pid and r['seed'] == seed:
                            cr[r['arm']] = r['CR']
                rows.append({'root': root.name, 'piece_id': pid, 'seed': seed,
                             'reward_baseline': arms['baseline'],
                             'reward_oracle': arms['oracle'], 'CR': cr})
        print(f'{root.name}: scored')

    for key in ('cosine', 'disc'):
        correct = sum(1 for r in rows
                      if r['reward_oracle'][key] > r['reward_baseline'][key])
        print(f'ranking accuracy [{key}]: {correct}/{len(rows)} = {correct / len(rows):.3f}')

    have_cr = [r for r in rows if len(r['CR']) == 2]
    if have_cr:
        for key in ('cosine', 'disc'):
            d_reward = [r['reward_oracle'][key] - r['reward_baseline'][key] for r in have_cr]
            d_cr = [r['CR']['oracle'] - r['CR']['baseline'] for r in have_cr]
            all_reward = [r['reward_oracle'][key] for r in have_cr] + \
                [r['reward_baseline'][key] for r in have_cr]
            all_cr = [r['CR']['oracle'] for r in have_cr] + \
                [r['CR']['baseline'] for r in have_cr]
            print(f'spearman d_reward~d_CR [{key}]: {spearman(d_reward, d_cr):.3f} '
                  f'(n={len(have_cr)})')
            print(f'spearman reward~CR [{key}]: {spearman(all_reward, all_cr):.3f}')

    args.output.write_text(json.dumps(rows, indent=1))
    print(f'wrote {args.output} ({len(rows)} pairs)')


if __name__ == '__main__':
    main()

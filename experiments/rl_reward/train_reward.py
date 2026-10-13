"""Train the melody-accompaniment match reward model.

Usage:
  python -m experiments.rl_reward.train_reward \
    --data-dir /data/home/yuanxin/data/allxml_npz_pop_dedup \
    --output /data/home/yuanxin/experiments/rl_reward_20261013 --epochs 10
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.rl_reward.reward_model import MatchRewardModel
from experiments.rl_reward.reward_data import NpzWindowDataset, scan_windows


@torch.no_grad()
def val_retrieval(model, loader, device):
    model.eval()
    correct = total = 0
    for mel, acc in loader:
        mel, acc = mel.to(device), acc.to(device)
        mel_e, acc_e = model.embed(mel, acc)
        logits = mel_e @ acc_e.t()
        correct += (logits.argmax(1) == torch.arange(logits.size(0), device=device)).sum().item()
        total += logits.size(0)
    model.train()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--val-frac', type=float, default=0.05)
    parser.add_argument('--limit-windows', type=int, default=0,
                        help='debug: cap windows after the val split')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output.mkdir(parents=True, exist_ok=True)

    print('scanning windows...', flush=True)
    windows = scan_windows(args.data_dir)
    print(f'{len(windows)} content windows from 7574 files', flush=True)
    rng = random.Random(args.seed)
    idx = list(range(len(windows)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    if args.limit_windows:
        idx = idx[:args.limit_windows]
    val_ds = NpzWindowDataset([windows[i] for i in idx[:n_val]])
    train_ds = NpzWindowDataset([windows[i] for i in idx[n_val:]])
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True)

    model = MatchRewardModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f'model params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    history = []
    best_val = -1.0
    for epoch in range(args.epochs):
        running = {'nce': 0.0, 'disc': 0.0, 'acc': 0.0, 'n': 0}
        for mel, acc in train_ld:
            mel, acc = mel.to(device), acc.to(device)
            nce, disc, logits = model.losses(mel, acc)
            loss = nce + 0.5 * disc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running['nce'] += nce.item()
            running['disc'] += disc.item()
            running['acc'] += (logits.argmax(1) ==
                               torch.arange(logits.size(0), device=device)).float().mean().item()
            running['n'] += 1
        sched.step()
        val_r1 = val_retrieval(model, val_ld, device)
        entry = {'epoch': epoch,
                 'nce': running['nce'] / running['n'],
                 'disc': running['disc'] / running['n'],
                 'train_r1': running['acc'] / running['n'],
                 'val_r1': val_r1}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if val_r1 > best_val:
            best_val = val_r1
            torch.save({'model': model.state_dict(), 'val_r1': val_r1,
                        'epoch': epoch}, args.output / 'reward_model_best.pt')
    torch.save({'model': model.state_dict(), 'history': history},
               args.output / 'reward_model_final.pt')
    (args.output / 'train_log.json').write_text(json.dumps(history, indent=1))
    print(f'DONE best_val_r1={best_val:.4f}', flush=True)


if __name__ == '__main__':
    main()

"""ReaLchords-faithful multi-scale match reward.

Completes the components the v0-v2 iterations omitted: per-scale models
(32/16/8/4 beats, trained separately as in the paper) with 50%-overlap
sliding scoring summed across scales, plus the paper's penalty terms
(silence / repetition / early-ending) as additive deductions.

Train one scale:   python -m experiments.rl_reward.multiscale train \
                     --data-dir ... --output DIR --beats 32
Exam (all scales): python -m experiments.rl_reward.multiscale exam \
                     --ckpt-dir DIR --pairs-root ... [--beat-offset N]

Faithful to the paper where specified (in-batch negative InfoNCE +
discriminator per scale, sliding-window summation); approximations are
flagged: penalty weights default to 0.25 each (paper coefficients not
fully public) and per-scale exam scores use the contrastive cosine.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TICKS_PER_BEAT = 24
MEASURE_TICKS = 96
FULL_BEATS = 32


def measure_roll(measure):
    return measure[:3].max(axis=0) > 0, measure[3:].max(axis=0) > 0


def _scan_one(job):
    path, measures = job
    try:
        z = np.load(path, allow_pickle=True)
        n = sum(1 for k in z.files if re.match(r'measure_\d+$', k))
        need = measures * MEASURE_TICKS
        starts = []
        for start in range(0, n - measures + 1):
            ok = True
            for i in range(start, start + measures):
                m = z[f'measure_{i}']
                if m.shape != (6, 88, 96) or m[:3].sum() == 0 or m[3:].sum() == 0:
                    ok = False
                    break
            if ok:
                starts.append(start)
        return path, starts
    except Exception as exc:  # noqa: BLE001
        print(f'skip {path}: {exc}')
        return path, []


def scan_windows(data_dir, measures, workers=16):
    from multiprocessing import Pool
    paths = sorted(Path(data_dir).glob('*.npz'))
    windows = []
    with Pool(workers) as pool:
        for k, (path, starts) in enumerate(
                pool.imap_unordered(_scan_one,
                                    [(p, measures) for p in paths], chunksize=32)):
            windows.extend((path, s) for s in starts)
            if (k + 1) % 1000 == 0:
                print(f'scanned {k + 1}/{len(paths)} files...', flush=True)
    return windows


def load_pair(z, start, beats):
    ticks = beats * TICKS_PER_BEAT
    mel = np.zeros((88, ticks), dtype=np.float32)
    acc = np.zeros((88, ticks), dtype=np.float32)
    for i in range(beats // 4):
        m, a = measure_roll(z[f'measure_{start + i}'])
        mel[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = m
        acc[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = a
    return mel, acc


class ScaleWindowDataset(Dataset):
    def __init__(self, windows, beats):
        self.windows = windows
        self.beats = beats

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, start = self.windows[idx]
        z = np.load(path, allow_pickle=True)
        mel, acc = load_pair(z, start, self.beats)
        return (torch.from_numpy(mel).unsqueeze(0),
                torch.from_numpy(acc).unsqueeze(0))


class PartEncoder(nn.Module):
    def __init__(self, beats, dim=128):
        super().__init__()
        self.conv_pitch = nn.Sequential(
            nn.Conv2d(1, 32, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, (22, 1)),
            nn.ReLU(inplace=True),
        )
        self.time_down = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, max(1, beats * 3), dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.out = nn.Linear(dim, dim)

    def forward(self, roll):
        h = self.conv_pitch(roll).squeeze(2)
        h = self.time_down(h).transpose(1, 2)
        h = self.transformer(h + self.pos[:, :h.size(1)])
        return F.normalize(self.out(h.mean(dim=1)), dim=-1)


class ScaleModel(nn.Module):
    """Twin-tower contrastive + discriminator, ReaLchords per-scale core."""

    def __init__(self, beats, dim=128):
        super().__init__()
        self.beats = beats
        self.mel_encoder = PartEncoder(beats, dim)
        self.acc_encoder = PartEncoder(beats, dim)
        self.discriminator = nn.Sequential(
            nn.Linear(dim * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def losses(self, mel, acc, temperature=0.1):
        mel_e, acc_e = self.mel_encoder(mel), self.acc_encoder(acc)
        logits = mel_e @ acc_e.t() / temperature
        target = torch.arange(logits.size(0), device=logits.device)
        nce = 0.5 * (F.cross_entropy(logits, target)
                     + F.cross_entropy(logits.t(), target))
        b = mel_e.size(0)
        roll = torch.randperm(b, device=mel_e.device)
        while True:
            clash = roll == torch.arange(b, device=roll.device)
            if not clash.any():
                break
            roll[clash] = roll.roll(1)[clash]
        fake = acc_e[roll]
        real_feat = torch.cat([mel_e, acc_e, mel_e * acc_e], dim=-1)
        fake_feat = torch.cat([mel_e, fake, mel_e * fake], dim=-1)
        disc = (F.binary_cross_entropy_with_logits(
            self.discriminator(real_feat), torch.ones(b, 1, device=mel_e.device))
            + F.binary_cross_entropy_with_logits(
                self.discriminator(fake_feat), torch.zeros(b, 1, device=mel_e.device)))
        return nce, disc, logits


@torch.no_grad()
def val_r1(model, loader, device):
    model.eval()
    correct = total = 0
    for mel, acc in loader:
        mel, acc = mel.to(device), acc.to(device)
        me, ae = model.mel_encoder(mel), model.acc_encoder(acc)
        correct += ((me @ ae.t()).argmax(1) == torch.arange(me.size(0), device=device)).sum().item()
        total += me.size(0)
    model.train()
    return correct / max(1, total)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output.mkdir(parents=True, exist_ok=True)
    measures = args.beats // 4
    print(f'== scale {args.beats} beats ({measures} measures) ==', flush=True)

    print('scanning windows...', flush=True)
    windows = scan_windows(args.data_dir, measures, args.workers)
    print(f'{len(windows)} windows', flush=True)
    rng = random.Random(args.seed)
    songs = sorted({p for p, _ in windows})
    rng.shuffle(songs)
    n_val = max(1, int(len(songs) * args.val_frac))
    val_songs = set(songs[:n_val])
    val_w = [w for w in windows if w[0] in val_songs]
    train_w = [w for w in windows if w[0] not in val_songs]
    if args.limit_windows:
        train_w = train_w[:args.limit_windows]
    print(f'{len(songs)} songs: val {len(val_w)} / train {len(train_w)}', flush=True)
    train_ld = DataLoader(ScaleWindowDataset(train_w, args.beats), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(ScaleWindowDataset(val_w, args.beats), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)

    model = ScaleModel(args.beats).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for epoch in range(args.epochs):
        run = {'nce': 0.0, 'disc': 0.0, 'acc': 0.0, 'n': 0}
        for mel, acc in train_ld:
            mel, acc = mel.to(device), acc.to(device)
            nce, disc, logits = model.losses(mel, acc)
            loss = nce + 0.5 * disc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run['nce'] += nce.item(); run['disc'] += disc.item()
            run['acc'] += (logits.argmax(1) == torch.arange(logits.size(0), device=device)).float().mean().item()
            run['n'] += 1
        sched.step()
        r1 = val_r1(model, val_ld, device)
        print(json.dumps({'epoch': epoch, 'nce': run['nce'] / run['n'],
                          'disc': run['disc'] / run['n'],
                          'train_r1': run['acc'] / run['n'], 'val_r1': r1}), flush=True)
        if r1 > best:
            best = r1
            torch.save({'model': model.state_dict(), 'beats': args.beats,
                        'val_r1': r1, 'epoch': epoch},
                       args.output / f'scale_b{args.beats}_best.pt')
    print(f'DONE scale_b{args.beats} best_val_r1={best:.4f}', flush=True)


def penalties(acc_roll):
    """ReaLchords penalty terms on one 32-beat acc roll (1, 88, T)."""
    beats = FULL_BEATS
    T = acc_roll.shape[-1]
    per_beat = acc_roll.reshape(acc_roll.shape[0], 88, beats, TICKS_PER_BEAT).sum(dim=(1, 3))
    silent = (per_beat[0] == 0).float()
    trailing = 0
    for b in range(beats - 1, -1, -1):
        if per_beat[0][b] == 0:
            trailing += 1
        else:
            break
    bars = acc_roll.reshape(acc_roll.shape[0], 88, beats // 4, MEASURE_TICKS).sum(dim=1) > 0
    rep = torch.randperm(beats // 4)
    repeats = 0
    for i in range(beats // 4 - 1):
        if (bars[0][i] == bars[0][i + 1]).all():
            repeats += 1
    return {'silence': float(silent.mean()), 'early_end': trailing / beats,
            'repetition': repeats / max(1, beats // 4 - 1)}


@torch.no_grad()
def exam(args):
    from experiments.rl_reward.reward_data import midi_pair_rolls

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_dir = args.ckpt_dir
    models = {}
    for beats in args.scales:
        path = ckpt_dir / f'scale_b{beats}_best.pt'
        if not path.exists():
            print(f'missing {path.name}, skipping scale')
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = ScaleModel(ck.get('beats', beats)).to(device)
        m.load_state_dict(ck['model'])
        m.eval()
        models[beats] = m
    print(f'loaded scales: {sorted(models)}', flush=True)

    rows = []
    for root in args.pairs_root:
        plan = json.loads((root / 'plan.json').read_text())
        for piece in plan['pieces']:
            pid = piece['piece_id']
            for seed in plan['seeds']:
                arms = {}
                for arm in ('baseline', 'oracle'):
                    path = root / pid / f'seed{seed}' / arm / 'full.mid'
                    if not path.exists():
                        continue
                    mel, acc = midi_pair_rolls(path)
                    total = 0.0
                    for beats, m in models.items():
                        w = beats * TICKS_PER_BEAT
                        step = max(1, w // 2)  # 50% overlap
                        vals = []
                        for s in range(0, mel.shape[-1] - w + 1, step):
                            me = m.mel_encoder(mel[..., s:s + w].unsqueeze(0).to(device))
                            ae = m.acc_encoder(acc[..., s:s + w].unsqueeze(0).to(device))
                            vals.append(float((me * ae).sum()))
                        total += float(np.mean(vals))
                    pens = penalties(acc)
                    total -= sum(args.w_sil * pens['silence'],
                                 args.w_end * pens['early_end'],
                                 args.w_rep * pens['repetition'])
                    arms[arm] = total
                if len(arms) == 2:
                    rows.append({'root': root.name, 'piece_id': pid, 'seed': seed, **arms})
        print(f'{root.name}: scored', flush=True)

    ok = sum(1 for r in rows if r['oracle'] > r['baseline'])
    print(f'ranking accuracy [multiscale]: {ok}/{len(rows)} = {ok / max(1, len(rows)):.3f}')
    Path(args.output).write_text(json.dumps(rows, indent=1))
    print(f'wrote {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'exam'))
    parser.add_argument('--data-dir')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ckpt-dir', type=Path)
    parser.add_argument('--pairs-root', type=Path, action='append')
    parser.add_argument('--beat-offset', type=int, default=0)
    parser.add_argument('--scales', type=int, nargs='+', default=[32, 16, 8, 4])
    parser.add_argument('--beats', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--val-frac', type=float, default=0.05)
    parser.add_argument('--limit-windows', type=int, default=0)
    parser.add_argument('--w-sil', type=float, default=0.25)
    parser.add_argument('--w-end', type=float, default=0.25)
    parser.add_argument('--w-rep', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        exam(args)


if __name__ == '__main__':
    main()

"""v2 reward model: single-tower joint local-window discriminator.

Diagnosis from the v0/v1 twin-encoder exams: corruption discrimination is
solvable via global texture statistics (register/density/rhythm), so the
judge never learns melody-harmony alignment — it scored the ground-truth
pairing BELOW model output at the deployment window and rated an empty
accompaniment (0.87) above every real one. ReaLchords saw the same
contrastive collapse at short windows and prescribe a discriminative
single tower at local scales.

This file: a 2-channel (melody, accompaniment) classifier over 8-beat
windows, real pair vs corrupted pair, with corruption modes that include
the empty roll (anti-silence). One window per item per step, random
offset inside the 32-beat roll; exam scoring averages 4 strided windows.

Train:  python -m experiments.rl_reward.joint mode=train --data-dir ... --output ...
Exam:   python -m experiments.rl_reward.joint mode=exam --ckpt ... --pairs-root ... [--beat-offset N]
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
WINDOW_MEASURES = 8
ROLL_BEATS = 32
ROLL_TICKS = ROLL_BEATS * TICKS_PER_BEAT


def measure_roll(measure):
    mel = measure[:3].max(axis=0) > 0
    acc = measure[3:].max(axis=0) > 0
    return mel, acc


def scan_windows(data_dir, workers=16):
    from multiprocessing import Pool
    paths = sorted(Path(data_dir).glob('*.npz'))

    def scan_one(path):
        try:
            z = np.load(path, allow_pickle=True)
            n = sum(1 for k in z.files if re.match(r'measure_\d+$', k))
            starts = []
            for start in range(0, n - WINDOW_MEASURES + 1):
                ok = True
                for i in range(start, start + WINDOW_MEASURES):
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

    windows = []
    with Pool(workers) as pool:
        for k, (path, starts) in enumerate(pool.imap_unordered(scan_one, paths, chunksize=32)):
            windows.extend((path, s) for s in starts)
            if (k + 1) % 1000 == 0:
                print(f'scanned {k + 1}/{len(paths)} files...', flush=True)
    return windows


def load_pair(z, start):
    mel = np.zeros((88, ROLL_TICKS), dtype=np.float32)
    acc = np.zeros((88, ROLL_TICKS), dtype=np.float32)
    for i in range(WINDOW_MEASURES):
        m, a = measure_roll(z[f'measure_{start + i}'])
        mel[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = m
        acc[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = a
    return mel, acc


def corrupt(acc, z, starts, start, rng):
    """Same-song hard negatives; empty attacks the silence-is-good pathology."""
    mode = rng.choices(('window', 'shuffle', 'transpose', 'empty'),
                       weights=(0.3, 0.25, 0.25, 0.2))[0]
    if mode == 'window' and len(starts) > 1:
        other = start
        while other == start:
            other = rng.choice(starts)
        return load_pair(z, other)[1]
    if mode == 'shuffle':
        perm = list(range(WINDOW_MEASURES))
        rng.shuffle(perm)
        return acc.reshape(88, WINDOW_MEASURES, MEASURE_TICKS)[:, perm, :] \
                  .reshape(88, ROLL_TICKS).copy()
    if mode == 'transpose':
        delta = rng.choice((-3, -2, -1, 1, 2, 3))
        neg = np.zeros_like(acc)
        if delta > 0:
            neg[delta:] = acc[:-delta]
        else:
            neg[:delta] = acc[-delta:]
        return neg
    return np.zeros_like(acc)


class JointWindowDataset(Dataset):
    """(x_pos, x_neg) 2-channel 8-beat windows at a random offset."""

    def __init__(self, windows, beats=8):
        self.windows = windows
        self.beats = beats
        self.song_starts = {}
        for path, start in windows:
            self.song_starts.setdefault(path, []).append(start)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, start = self.windows[idx]
        rng = random.Random(random.getrandbits(64))
        z = np.load(path, allow_pickle=True)
        mel, acc = load_pair(z, start)
        acc_neg = corrupt(acc, z, self.song_starts[path], start, rng)
        w = self.beats * TICKS_PER_BEAT
        u = rng.randint(0, ROLL_BEATS - self.beats) * TICKS_PER_BEAT
        x_pos = np.stack([mel[:, u:u + w], acc[:, u:u + w]])
        x_neg = np.stack([mel[:, u:u + w], acc_neg[:, u:u + w]])
        return torch.from_numpy(x_pos), torch.from_numpy(x_neg)


class JointDiscriminator(nn.Module):
    def __init__(self, beats=8, dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, (7, 5), stride=(2, 2), padding=(3, 2)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, (22, 1)),
            nn.ReLU(inplace=True),
        )
        self.time = nn.Conv1d(dim, dim, 4, stride=2, padding=1)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, beats * TICKS_PER_BEAT // 8, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):  # (B, 2, 88, T)
        h = self.conv(x).squeeze(2)
        h = self.time(h).transpose(1, 2)
        h = self.tr(h + self.pos[:, :h.size(1)])
        return self.head(h.mean(dim=1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x_pos, x_neg in loader:
        x_pos = x_pos.to(device)
        x_neg = x_neg.to(device)
        lp = model(x_pos)
        ln = model(x_neg)
        correct += (lp > ln).sum().item()
        total += lp.size(0)
    model.train()
    return correct / max(1, total)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output.mkdir(parents=True, exist_ok=True)

    print('scanning windows...', flush=True)
    windows = scan_windows(args.data_dir, workers=args.workers)
    print(f'{len(windows)} windows', flush=True)
    rng = random.Random(args.seed)
    songs = sorted({path for path, _ in windows})
    rng.shuffle(songs)
    n_val = max(1, int(len(songs) * args.val_frac))
    val_songs = set(songs[:n_val])
    val_w = [w for w in windows if w[0] in val_songs]
    train_w = [w for w in windows if w[0] not in val_songs]
    if args.limit_windows:
        train_w = train_w[:args.limit_windows]
    print(f'{len(songs)} songs: val {len(val_w)} windows / train {len(train_w)} windows', flush=True)
    train_ld = DataLoader(JointWindowDataset(train_w, args.beats), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(JointWindowDataset(val_w, args.beats), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=True)

    model = JointDiscriminator(args.beats).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    best = -1.0
    for epoch in range(args.epochs):
        running = {'loss': 0.0, 'acc': 0.0, 'n': 0}
        for x_pos, x_neg in train_ld:
            x_pos = x_pos.to(device)
            x_neg = x_neg.to(device)
            lp = model(x_pos)
            ln = model(x_neg)
            loss = (F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
                    + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running['loss'] += loss.item()
            running['acc'] += (lp > ln).float().mean().item()
            running['n'] += 1
        sched.step()
        val_acc = evaluate(model, val_ld, device)
        entry = {'epoch': epoch, 'loss': running['loss'] / running['n'],
                 'train_acc': running['acc'] / running['n'], 'val_acc': val_acc}
        print(json.dumps(entry), flush=True)
        if val_acc > best:
            best = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc,
                        'beats': args.beats, 'epoch': epoch},
                       args.output / 'joint_best.pt')
    torch.save({'model': model.state_dict(), 'beats': args.beats}, args.output / 'joint_final.pt')
    print(f'DONE best_val_acc={best:.4f}', flush=True)


@torch.no_grad()
def exam(args):
    import mido  # noqa: F401  (midi_pair_rolls needs it importable via reward_data)
    from experiments.rl_reward.reward_data import midi_pair_rolls

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    beats = ck.get('beats', args.beats)
    model = JointDiscriminator(beats).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

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
                    mel, acc = midi_pair_rolls(path, start_beat=args.beat_offset)
                    w = beats * TICKS_PER_BEAT
                    scores = []
                    for s in range(0, mel.shape[-1] - w + 1, w):
                        x = torch.stack([mel, acc], dim=1)[..., s:s + w].to(device)
                        scores.append(float(torch.sigmoid(model(x))))
                    arms[arm] = float(np.mean(scores))
                if len(arms) == 2:
                    rows.append({'root': root.name, 'piece_id': pid, 'seed': seed, **arms})
        print(f'{root.name}: scored', flush=True)

    ok = sum(1 for r in rows if r['oracle'] > r['baseline'])
    print(f'ranking accuracy [joint]: {ok}/{len(rows)} = {ok / max(1, len(rows)):.3f}')
    Path(args.output).write_text(json.dumps(rows, indent=1))
    print(f'wrote {args.output} ({len(rows)} pairs)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'exam'))
    parser.add_argument('--data-dir')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ckpt', type=Path)
    parser.add_argument('--pairs-root', type=Path, action='append')
    parser.add_argument('--beat-offset', type=int, default=0)
    parser.add_argument('--beats', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--val-frac', type=float, default=0.05)
    parser.add_argument('--limit-windows', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        exam(args)


if __name__ == '__main__':
    main()

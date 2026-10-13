"""Data plumbing for the match-reward model.

Training side: 8-measure binary windows from allxml_npz_pop_dedup npz
(per-measure keys `measure_i`, (6, 88, 96) = melody ch0-2 / acc ch3-5
x 88 pitch x 96 ticks). Exam side: generated MIDIs -> the same
representation, channel 0 = melody / other channels = accompaniment
(same convention as score_law_ab).
"""

import mido
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.rl_reward.reward_model import ROLL_BEATS, ROLL_TICKS, TICKS_PER_BEAT

MEASURE_TICKS = 96
WINDOW_MEASURES = ROLL_BEATS // 4  # 8 measures = 32 beats


def measure_roll(measure):
    """(6, 88, 96) uint8 -> two binary (88, 96) rolls (melody, acc)."""
    mel = measure[:3].max(axis=0) > 0
    acc = measure[3:].max(axis=0) > 0
    return mel, acc


def _scan_one(path):
    """All (start_measure) windows where 8 consecutive measures are exactly
    (6, 88, 96) with both parts nonempty. The npz corpus mixes measure tick
    counts (72/48/108...); strictest filter avoids resampling semantics."""
    import re
    measure_key = re.compile(r'measure_\d+$')
    try:
        z = np.load(path, allow_pickle=True)
        n = sum(1 for k in z.files if measure_key.match(k))
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


def scan_windows(data_dir, workers=16):
    import glob
    from multiprocessing import Pool
    from pathlib import Path
    paths = sorted(glob.glob(str(Path(data_dir) / '*.npz')))
    windows = []
    with Pool(workers) as pool:
        for k, (path, starts) in enumerate(pool.imap_unordered(_scan_one, paths, chunksize=32)):
            windows.extend((path, s) for s in starts)
            if (k + 1) % 1000 == 0:
                print(f'scanned {k + 1}/{len(paths)} files...', flush=True)
    return windows


class NpzWindowDataset(Dataset):
    """Returns (mel, acc, acc_neg): acc_neg is a hard negative built from the
    SAME song — a different window's accompaniment (right style/key, wrong
    harmony), a transposed copy (wrong key), or measure-shuffled (wrong chord
    order). Cross-song negatives are left to in-batch InfoNCE."""

    def __init__(self, windows):
        self.windows = windows
        self.song_starts = {}
        for path, start in windows:
            self.song_starts.setdefault(path, []).append(start)

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _load_pair(z, start):
        mel = np.zeros((88, ROLL_TICKS), dtype=np.float32)
        acc = np.zeros((88, ROLL_TICKS), dtype=np.float32)
        for i in range(WINDOW_MEASURES):
            m, a = measure_roll(z[f'measure_{start + i}'])
            mel[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = m
            acc[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = a
        return mel, acc

    def _hard_negative(self, z, path, start, acc):
        import random
        choices = ['transpose', 'shuffle']
        starts = self.song_starts[path]
        if len(starts) > 1:
            choices.append('window')
        mode = random.choice(choices)
        if mode == 'window':
            other = start
            while other == start:
                other = random.choice(starts)
            return self._load_pair(z, other)[1]
        if mode == 'transpose':
            delta = random.choice((-3, -2, -1, 1, 2, 3))
            neg = np.zeros_like(acc)
            if delta > 0:
                neg[delta:] = acc[:-delta]
            else:
                neg[:delta] = acc[-delta:]
            return neg
        perm = list(range(WINDOW_MEASURES))
        random.shuffle(perm)
        return acc.reshape(88, WINDOW_MEASURES, MEASURE_TICKS)[:, perm, :] \
                  .reshape(88, ROLL_TICKS).copy()

    def __getitem__(self, idx):
        path, start = self.windows[idx]
        z = np.load(path, allow_pickle=True)
        mel, acc = self._load_pair(z, start)
        acc_neg = self._hard_negative(z, path, start, acc)
        return (torch.from_numpy(mel).unsqueeze(0),
                torch.from_numpy(acc).unsqueeze(0),
                torch.from_numpy(acc_neg).unsqueeze(0))


def midi_pair_rolls(path, beats=ROLL_BEATS, start_beat=0):
    """Generated/GT MIDI -> (mel_roll, acc_roll), each (88, beats*24) binary.

    Channel 0 = melody, any other channel = accompaniment (score_law_ab
    convention). Ticks map to beats via the file's own ticks_per_beat;
    start_beat offsets the window into the file.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    ticks = beats * TICKS_PER_BEAT
    t0 = start_beat * TICKS_PER_BEAT
    mel = np.zeros((88, ticks), dtype=np.float32)
    acc = np.zeros((88, ticks), dtype=np.float32)
    open_notes = {}
    skipped = 0
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key not in open_notes:
                    continue
                start = open_notes.pop(key)
                pitch = msg.note - 21  # corpus vocabulary is MIDI 21..108
                if not 0 <= pitch < 88:
                    skipped += 1
                    continue
                pos = round(start / tpb * TICKS_PER_BEAT) - t0
                end = min(round(t / tpb * TICKS_PER_BEAT) - t0, ticks)
                roll = mel if msg.channel == 0 else acc
                for p in range(max(pos, 0), max(pos + 1, min(end, ticks))):
                    if 0 <= p < ticks:
                        roll[pitch, p] = 1.0
    if skipped:
        print(f'{path}: skipped {skipped} out-of-vocabulary notes')
    return (torch.from_numpy(mel).unsqueeze(0),
            torch.from_numpy(acc).unsqueeze(0))

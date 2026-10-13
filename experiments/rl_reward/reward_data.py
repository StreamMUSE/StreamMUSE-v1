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


def scan_windows(data_dir):
    """(file_idx, start_measure) for windows where both parts are active in
    at least half the beats. One pass over all files at scan time."""
    import glob
    from pathlib import Path
    windows = []
    import re
    measure_key = re.compile(r'measure_\d+$')
    for path in sorted(glob.glob(str(Path(data_dir) / '*.npz'))):
        try:
            z = np.load(path, allow_pickle=True)
            n = sum(1 for k in z.files if measure_key.match(k))
        except Exception as exc:  # noqa: BLE001
            print(f'skip {path}: {exc}')
            continue
        for start in range(0, max(0, n - WINDOW_MEASURES + 1)):
            ok_mel = ok_acc = 0
            for i in range(start, start + WINDOW_MEASURES):
                mel, acc = measure_roll(z[f'measure_{i}'])
                ok_mel += mel.sum() > 0
                ok_acc += acc.sum() > 0
            if ok_mel >= WINDOW_MEASURES // 2 and ok_acc >= WINDOW_MEASURES // 2:
                windows.append((path, start))
    return windows


class NpzWindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, start = self.windows[idx]
        z = np.load(path, allow_pickle=True)
        mel = np.zeros((88, ROLL_TICKS), dtype=np.float32)
        acc = np.zeros((88, ROLL_TICKS), dtype=np.float32)
        for i in range(WINDOW_MEASURES):
            m, a = measure_roll(z[f'measure_{start + i}'])
            mel[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = m
            acc[:, i * MEASURE_TICKS:(i + 1) * MEASURE_TICKS] = a
        return torch.from_numpy(mel).unsqueeze(0), torch.from_numpy(acc).unsqueeze(0)


def midi_pair_rolls(path, beats=ROLL_BEATS):
    """Generated/GT MIDI -> (mel_roll, acc_roll), each (88, beats*24) binary.

    Channel 0 = melody, any other channel = accompaniment (score_law_ab
    convention). Ticks map to beats via the file's own ticks_per_beat.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    ticks = beats * TICKS_PER_BEAT
    mel = np.zeros((88, ticks), dtype=np.float32)
    acc = np.zeros((88, ticks), dtype=np.float32)
    open_notes = {}
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
                pos = round(start / tpb * TICKS_PER_BEAT)
                end = min(round(t / tpb * TICKS_PER_BEAT), ticks)
                roll = mel if msg.channel == 0 else acc
                for p in range(pos, max(pos + 1, min(end, ticks))):
                    if 0 <= p < ticks:
                        roll[msg.note, p] = 1.0
    return (torch.from_numpy(mel).unsqueeze(0),
            torch.from_numpy(acc).unsqueeze(0))

"""Prepare BEAT-40 oracle inputs from the exact NPZ-aligned GT exports.

Splits each piece's seed0/full_gt.mid (round-trip-verified reconstruction:
Melody track ch0, Accompaniment track ch1, ppq 220, 32 beats = 128 ticks,
120 bpm) into single-track melody.mid + gt_acc.mid, hashes everything, and
emits a run_oracle.py-compatible plan (cohorts.beat40) plus a sha-pinned GT
manifest. The 32-beat generated region is [32, 128); the metric window is
set to the full generated region via primary_metric_start_tick=32.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mido

BEAT_GT_ROOT = Path('/data/home/yuanxin/streammuse_eval_results/'
                    'h200_trimmed_exact_beat_gt_40piece_3seed_20260903')
CHECKPOINTS = {
    'continuation': '/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors',
    'prompt': '/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors',
}
SAMPLING = {'temperature': 1.05, 'top_p': 0.98, 'top_k': 0, 'repetition_penalty': 1.0}
STOP_TICK = 128


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_track(source, wanted_name, out_path):
    midi = mido.MidiFile(source)
    tracks = [(i, tr) for i, tr in enumerate(midi.tracks)
              if next((e.name for e in tr if e.type == 'track_name'), '') == wanted_name]
    if len(tracks) != 1:
        raise ValueError(f'{source}: expected one {wanted_name!r} track, got {len(tracks)}')
    _, track = tracks[0]
    out = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat)
    new = mido.MidiTrack()
    out.tracks.append(new)
    new.append(mido.MetaMessage('track_name', name=wanted_name, time=0))
    absolute, previous, notes = 0, 0, 0
    for event in track:
        absolute += event.time
        if event.type in ('note_on', 'note_off'):
            new.append(event.copy(time=absolute - previous))
            previous = absolute
            notes += 1 if event.type == 'note_on' and event.velocity > 0 else 0
    out.save(out_path)
    if not notes:
        raise ValueError(f'{out_path}: no notes')
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('/data/home/yuanxin/experiments/oracle_beat40_prep_20261012'))
    parser.add_argument('--limit-pieces', type=int, default=40)
    args = parser.parse_args()
    pieces_dir = args.out / 'pieces'
    pieces_dir.mkdir(parents=True, exist_ok=False)
    pieces, manifest = [], {'accompaniment': {}}
    piece_dirs = sorted(p for p in BEAT_GT_ROOT.iterdir() if p.is_dir())[:args.limit_pieces]
    for order, piece_dir in enumerate(piece_dirs, 1):
        pid = piece_dir.name
        source = piece_dir / 'seed0' / 'full_gt.mid'
        target = pieces_dir / pid
        target.mkdir()
        mel_notes = split_track(source, 'Melody', target / 'melody.mid')
        acc_notes = split_track(source, 'Accompaniment', target / 'gt_acc.mid')
        melody = target / 'melody.mid'
        gt_acc = target / 'gt_acc.mid'
        pieces.append({'piece_id': pid, 'title': pid, 'order': order,
                       'melody_midi': str(melody), 'melody_midi_sha256': sha(melody),
                       'stop_tick': STOP_TICK, 'midi_ppq': 220,
                       'melody_note_count': mel_notes, 'gt_acc_note_count': acc_notes})
        manifest['accompaniment'][pid] = {'midi': str(gt_acc), 'sha256': sha(gt_acc)}
    checkpoints = {name: {'path': str(path), 'sha256': sha(path)}
                   for name, path in CHECKPOINTS.items()}
    plan = {'cohorts': {'beat40': {
        'pieces': pieces, 'sampling': SAMPLING,
        'model_bpm': 120, 'playback_bpm': 120,
        'source_root': str(BEAT_GT_ROOT),
        'note': 'melody+gt_acc split from round-trip-verified full_gt.mid; '
                'same source npz as the 20260903 beatgt eval'}},
        'checkpoints': checkpoints}
    (args.out / 'plan_beat40.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    (args.out / 'gt_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"pieces: {len(pieces)}, stop_tick: {STOP_TICK}")
    for p in pieces[:3]:
        print(' ', p['piece_id'], 'melody_notes', p['melody_note_count'],
              'acc_notes', p['gt_acc_note_count'])


if __name__ == '__main__':
    main()

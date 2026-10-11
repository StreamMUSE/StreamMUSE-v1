"""Strong-beat triad boost A/B/C: offline, shared prompt, no repair, no wall clock.

Arms per piece/seed share one Rule If-Else N=10 prompt, then differ only in the
env-driven tonal boost window: A tau=0, B tau in [32,256) ticks (bars 3-16),
C tau from tick 32 to song end. No history repair; history-repair stays off.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SAMPLING = ('temperature', 'top_p', 'top_k', 'repetition_penalty')
ARMS = {
    'A_baseline': {'tau': 0.0, 'start': 32, 'end': 256},
    'B_strong_to_bar16': {'tau': 2.0, 'start': 32, 'end': 256},
    'C_strong_full': {'tau': 2.0, 'start': 32, 'end': 0},
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_events(path):
    import mido
    midi = mido.MidiFile(path)
    tracks = [i for i, track in enumerate(midi.tracks)
              if any(e.type == 'note_on' and e.velocity > 0 for e in track)]
    if len(tracks) != 1:
        raise ValueError('Expected exactly one Melody note track; do not merge accompaniment')
    from experiments.bar_quality_history.live_gpu_smoke import midi_events
    return midi_events(path, tracks[0])


def increment(events, start, end):
    return [deepcopy(e) for e in events if start <= e['tick'] < end]


def export_midi(path, melody, accompaniment, stop, bpm):
    import mido
    midi = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    audit = {}
    for channel, (name, events) in enumerate((('Melody', melody), ('Accompaniment', accompaniment))):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        track.append(mido.Message('program_change', channel=channel, program=0, time=0))
        active, previous = set(), 0
        selected = sorted((e for e in events if e['tick'] < stop or
                           (e['tick'] == stop and e['type'] == 'note_off')),
                          key=lambda e: (e['tick'], 0 if e['type'] == 'note_off' else 1, e['pitch']))
        for event in selected:
            tick = int(event['tick']) * 120
            track.append(mido.Message(event['type'], channel=channel, note=event['pitch'],
                                      velocity=int(event.get('velocity', 80)), time=tick - previous))
            previous = tick
            if event['type'] == 'note_on':
                active.add(event['pitch'])
            else:
                active.discard(event['pitch'])
        audit[name] = {'terminal_note_off_pitches': sorted(active), 'terminal_tick': stop}
        for pitch in sorted(active):
            track.append(mido.Message('note_off', channel=channel, note=pitch, velocity=0,
                                      time=stop * 120 - previous))
            previous = stop * 120
        track.append(mido.MetaMessage('end_of_track', time=stop * 120 - previous))
    midi.save(path)
    return audit


def run_arm(engine, backend, config, melody, prompt, seed, stop, arm, arm_cfg, directory):
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['LEKAI_TONAL_BOOST_TAU'] = str(arm_cfg['tau'])
    os.environ['LEKAI_TONAL_BOOST_START_TICK'] = str(arm_cfg['start'])
    os.environ['LEKAI_TONAL_BOOST_END_TICK'] = str(arm_cfg['end'])
    engine.reset_session(seed)
    engine.set_session_generation_config(**{k: config[k] for k in SAMPLING})
    injected = engine.inject_history(increment(melody, 0, 32), deepcopy(prompt), 32)
    if not injected.get('success'):
        raise RuntimeError('Prompt injection failed')
    rows, returned, previous = [], deepcopy(prompt), 32
    for tick in range(32, stop, 4):
        incoming = increment(melody, previous, tick)
        previous = tick
        backend.input_token_calls.clear()
        began = time.perf_counter()
        events, timings = engine.generate(
            melody_events=incoming, generation_start_tick=tick,
            generation_length_frames=4, generation_interval_ticks=4,
            prompt_length_ticks=128, inference_mode='sliding_window', model_name='lekai',
            checkpoint_path=config['checkpoint_path'], bpm=config['bpm'])
        metadata = backend.generation_metadata_snapshot()[-1]
        if not metadata.get('part0_trace_available') or metadata.get('fallback_reason'):
            raise RuntimeError('Real-model token trace required')
        rows.append({'tick': tick, 'melody_increment': incoming, 'events': deepcopy(events),
                     'metadata': metadata, 'elapsed_seconds': time.perf_counter() - began,
                     'timings': timings})
        returned.extend(deepcopy(events))
    record = {'seed': seed, 'condition': arm, 'rows': rows, 'raw_events': returned,
              'runtime': backend.runtime_info()}
    save(directory / 'generation.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=['1', '2', '3'], required=True)
    parser.add_argument('--limit-pieces', type=int, default=10)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu:
        raise ValueError('GPU binding mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    cohort = plan['cohorts']['user10']
    pieces = cohort['pieces'][:args.limit_pieces]
    for checkpoint in plan['checkpoints'].values():
        assert sha(checkpoint['path']) == checkpoint['sha256']
    for piece in pieces:
        assert sha(piece['melody_midi']) == piece['melody_midi_sha256']
        assert piece['stop_tick'] > 84 and piece['stop_tick'] % 4 == 0
    config = {'mode': 'real', 'checkpoint_path': plan['checkpoints']['continuation']['path'],
              'device': 'cuda', 'dtype': 'float16', 'bpm': cohort['model_bpm'],
              'context_beats': 32, **cohort['sampling'], 'time_signature_index': 4,
              'tonal_constraint_enabled': True, 'empty_token_guard_enabled': True}
    save(args.output / 'plan.json', {'config': config, 'arms': ARMS, 'pieces': pieces,
        'seeds': args.seeds, 'source_plan': str(args.plan), 'source_plan_sha256': sha(args.plan),
        'checkpoints': plan['checkpoints'], 'export_bpm': cohort['playback_bpm'],
        'primary_metric_start_tick': 84, 'secondary_metric_start_tick': 32,
        'scope': 'causal offline, strong-beat triad boost only, no repair, no wall clock',
        'boost_semantics': 'additive tau on tonic-triad PIT onsets at bar beats 1/3, '
                           'all octaves via pitch class, only after key determination'})
    paths = list((ROOT / 'src/streammuse/infrastructure/inference').rglob('*.py'))
    paths += list((ROOT / 'experiments').rglob('*.py'))
    save(args.output / 'code_snapshot.json', {'sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
         'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)})
    os.environ.update(LEKAI_HISTORY_REPAIR_ENABLED='0', LEKAI_PROMPT_DEVICE='cuda',
        LEKAI_PROMPT_DTYPE='float16', LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS='1',
        LEKAI_PROMPT_TIME_SIGNATURE_INDEX='4', LEKAI_PROMPT_CONTEXT_BEATS='32',
        LEKAI_PROMPT_BPM=str(config['bpm']), LEKAI_DISABLE_FALLBACK='1',
        LEKAI_RT_LOG_DIR=str(args.output / 'model_logs'))
    from experiments.bar_quality_history.model_probe import SnapshotProbe, _verify_runtime
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import LekaiPromptEngine
    from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import LekaiContinuationEngine
    probe = SnapshotProbe(config, allow_gpu=True)
    summary = []
    try:
        probe.__enter__()
        with probe._environment():
            os.environ['LEKAI_REQUIRE_SESSION'] = '0'
            backend = probe._backend
            _verify_runtime(backend.runtime_info(), config)
            engine = LekaiContinuationEngine(backend=backend)
            prompt_engine = LekaiPromptEngine(checkpoint_path=plan['checkpoints']['prompt']['path'])
            save(args.output / 'prompt_runtime.json', prompt_engine.runtime_info())
            for piece in pieces:
                melody = source_events(piece['melody_midi'])
                for seed in args.seeds:
                    directory = args.output / piece['piece_id'] / f'seed{seed}'
                    directory.mkdir(parents=True)
                    save(directory / 'melody.json', melody)
                    print('START', piece['piece_id'], seed, flush=True)
                    prompt_engine.reset_session(seed)
                    prompt_engine.set_session_generation_config(prompt_selection_mode='rule_s_if_else',
                        prompt_batch_candidates=10, **{k: config[k] for k in SAMPLING})
                    prompt = prompt_engine.generate_prompt_accompaniment(increment(melody, 0, 32), 0, 32,
                                                                         bpm=config['bpm'])
                    save(directory / 'prompt.json', {'events': prompt, 'log': prompt_engine.last_generation_log()})
                    if prompt_engine.last_generated_acc_beats() < 8:
                        raise RuntimeError('Incomplete eight-beat Prompt; do not silently change join')
                    records = {}
                    for arm, arm_cfg in ARMS.items():
                        record = run_arm(engine, backend, config, melody, prompt, seed,
                                         piece['stop_tick'], arm, arm_cfg, directory / arm)
                        audit = export_midi(directory / arm / 'full.mid', melody, record['raw_events'],
                                            piece['stop_tick'], cohort['playback_bpm'])
                        save(directory / arm / 'export_audit.json', audit)
                        records[arm] = record
                    # Determinism guard: with tau=0 the arm must reproduce the
                    # paired-experiment baseline exactly (same seed and prompt
                    # procedure); divergence means the environment drifted.
                    save(directory / 'arms_done.json', {'arms': sorted(records)})
                    summary.append({'piece_id': piece['piece_id'], 'title': piece['title'],
                                    'seed': seed, 'arms': sorted(records)})
                    save(args.output / f'progress_{len(summary):02d}.json', summary[-1])
                    print('DONE', piece['piece_id'], seed, flush=True)
            save(args.output / 'complete.json', summary)
    finally:
        probe.close()


if __name__ == '__main__':
    main()

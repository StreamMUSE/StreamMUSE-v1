"""Pinned three-system batch using existing runners; no model/input hooks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_matched_system_eval as matched

HOME = Path('/data/home/yuanxin')
PYTHON = HOME / 'StreamMUSE-v1/.venv/bin/python'
LEGACY = HOME / 'StreamMUSE-legacy-duration-absolute-clock'
LEGACY_REF = '47496cbf'
MODERN_REF = 'ec843a79'
CONDITIONS = ('legacy_m2a', 'lekai_no_prompt', 'pc_rule_if_else_n10')
IDS = {'lekai_no_prompt': 'streammuse_v1_standard',
       'pc_rule_if_else_n10': 'streammuse_v2_prompt_continuation'}
CHECKPOINTS = {
    'legacy': (HOME / 'models/streammuse_v1_m2a/m2a_epoch00_val0.90296.ckpt',
               '6d097f9bbcdda75535551a5bd08f878cdd4f122170a85afe5c1d6da6451ea813'),
    'prompt': (HOME / 'RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors',
               '8d13485ccf5958dd3340d98123209f9f6e7ddb63494717af3c82b29db718fce7'),
    'continuation': (HOME / 'RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors',
               'd93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    matched.write_json(Path(path), value)


def source_identity():
    code = matched.code_identity()
    assert not subprocess.check_output(['git', '-C', str(matched.REPO_ROOT), 'diff', MODERN_REF, '--', 'src', 'transformers'])
    assert not subprocess.check_output(['git', '-C', str(LEGACY), 'diff', LEGACY_REF])
    return code


def midi_geometry(path):
    import mido
    midi = mido.MidiFile(path)
    notes = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == 'note_on' and message.velocity > 0:
                notes.append(Fraction(tick * 4, midi.ticks_per_beat))
    assert notes and min(notes) == 0, f'Input still has leading Melody silence: {path}'
    return {'note_on_count': len(notes), 'first_note_on_tick': 0, 'midi_ppq': midi.ticks_per_beat}


def prepare(args):
    root = args.root
    assert not root.exists(), root
    code = source_identity()
    gate_root = HOME / 'experiments/human10_pc_fixed_events_vs_midi_20260909'
    gate = [read(p) for p in sorted(gate_root.glob('cases/*/comparison_repeat_*.json'))]
    assert len(gate) == 20 and all(d['primary_raw_exact'] and d['model_inputs_exact'] and d['prompt_selection_exact'] for d in gate)
    assert all(read(gate_root / f'progress_gpu{i}.json')['status'] == 'completed' for i in (0, 1, 2))
    checkpoints = {}
    for name, (path, sha) in CHECKPOINTS.items():
        assert matched.file_sha256(path) == sha, name
        checkpoints[name] = {'path': str(path), 'sha256': sha}
    old_user_path = HOME / 'experiments/human10_legacy_m2a_duration_fixed_3seed_20260909/plan.json'
    old_user = read(old_user_path)
    test_path = HOME / 'data/streammuse_eval/beat_test_cohort_40_trimmed_npz_aligned_exactroll_20260903/cohort_manifest.json'
    assert matched.file_sha256(test_path) == 'a0b66d6049cba4b879fb1d90a32625e1caabb59568829d7da0cea5bd719967d8'
    old_test = read(test_path)
    root.mkdir(parents=True)
    cohorts = {}
    for name, source, rows in [('user10', old_user_path, old_user['pieces']), ('test40', test_path, old_test['samples'])]:
        prepared = []
        for row in rows:
            pid = row['piece_id']
            src = Path(row['midi_path']) if name == 'user10' else test_path.parent / row['melody_midi']
            sha = row['melody_input_sha256'] if name == 'user10' else row['melody_midi_sha256']
            assert matched.file_sha256(src) == sha
            dest = root / 'cohorts' / name / 'inputs' / pid / 'melody.mid'
            dest.parent.mkdir(parents=True)
            shutil.copy2(src, dest)
            case = {'piece_id': pid, 'order': row['order'], 'title': row.get('title', pid),
                    'melody_midi': str(dest), 'melody_midi_sha256': sha,
                    'source_melody': str(src), 'stop_tick': row['client_stop_tick'] if name == 'user10' else 128,
                    **midi_geometry(dest)}
            if name == 'test40':
                gt = test_path.parent / row['gt_midi']
                assert matched.file_sha256(gt) == row['gt_midi_sha256']
                target = dest.with_name('gt.mid')
                shutil.copy2(gt, target)
                case.update(gt_midi=str(target), gt_midi_sha256=row['gt_midi_sha256'])
            prepared.append(case)
        manifest = root / 'cohorts' / name / 'cohort_manifest.json'
        save(manifest, {'source_manifest': str(source), 'source_manifest_sha256': matched.file_sha256(source), 'samples': prepared})
        cohorts[name] = {'manifest': str(manifest), 'manifest_sha256': matched.file_sha256(manifest),
                         'pieces': prepared, 'playback_bpm': 90 if name == 'user10' else 120,
                         'model_bpm': 80 if name == 'user10' else 120,
                         'sampling': {'temperature': 1.1 if name == 'user10' else 1.05,
                                      'top_p': .95 if name == 'user10' else .98,
                                      'top_k': 50 if name == 'user10' else 0, 'repetition_penalty': 1.0},
                         'count_in_beats_modern': 4 if name == 'user10' else 0}
    plan = {'modern_code': code, 'modern_system_ref': MODERN_REF,
            'legacy_code': {'path': str(LEGACY), 'commit': subprocess.check_output(['git', '-C', str(LEGACY), 'rev-parse', 'HEAD'], text=True).strip()},
            'checkpoints': checkpoints, 'cohorts': cohorts, 'seeds': [0, 1, 2],
            'conditions': CONDITIONS, 'expected_trials': 450, 'allowed_gpus': [0, 1, 2],
            'time_signature_index': 4, 'steps_per_beat': 4, 'root': str(root),
            'legacy': {'I_ticks': 4, 'GL_interleaved_frames': 7, 'original_melody_duration': True,
                       'clock': 'absolute_monotonic', 'count_in_beats': 0},
            'modern': {'I_ticks': 4, 'GL_ticks': 4, 'prompt_beats': 8, 'context_beats': 32,
                       'boundary_order': 'single_executor_then_request', 'late_recovery': False,
                       'raw_observer': False, 'source_tick_override': False},
            'primary_playback': 'combined.mid is actual delivery, never replaced by raw reconstruction',
            'metrics': {'music': ['FMD', 'JSD-P', 'JSD-O', 'CR', 'UR'], 'system': 'per-beat ISR_f; nominal deadline + 0.1 tick', 'NLL': False},
            'raw_gate': {'pairs': len(gate), 'calls': sum(d['common_calls'] for d in gate), 'all_exact': True},
            'retry_policy': 'Stop worker on failed validation; retain all records, no automatic reruns'}
    save(root / 'plan.json', plan)
    shutil.copy2(matched.REPO_ROOT / 'docs/formal_user10_test40.md', root / 'README.md')
    save(root / '_metadata/raw_acceptance_gate.json', gate)
    print(json.dumps({'root': str(root), 'cohorts': {k: len(v['pieces']) for k, v in cohorts.items()}, 'trials': 450}))


def contract_for(cohort, condition, stop_tick):
    sampling = matched.SamplingConfig(**cohort['sampling'])
    return matched.EvaluationContract(playback_bpm=cohort['playback_bpm'], model_condition_bpm=cohort['model_bpm'],
        ticks_per_beat=4, window_end_tick=stop_tick, prompt_beats=8,
        generation_interval_ticks=4, generation_length_frames=4,
        prompt_sampling=sampling, continuation_sampling=sampling,
        count_in_beats=cohort['count_in_beats_modern'],
        continuation_constraints=condition == 'pc_rule_if_else_n10', deterministic_boundary_generation=False)


def validate_modern_artifacts(record, condition):
    session = Path(record['session_dir'])
    assert (session / 'combined.mid').is_file()
    if condition == 'pc_rule_if_else_n10':
        trace = read(session / 'prompt_continuation_model_trace.json')
        assert trace['trace_capture_complete'], 'P+C trace incomplete'
        assert not trace['runtime_info']['scheduler_is_failed'], trace['runtime_info'].get('scheduler_error')
        prompt = trace['prompt_generation_log']
        assert prompt['selection_mode'] == 'rule_s_if_else' and prompt['candidate_count'] == 10
        for name in ['prompt_continuation_raw_history.json', 'prompt_continuation_prompt_history.json',
                     'prompt_continuation_replay_requests.jsonl']:
            assert (session / name).is_file(), name
    else:
        for name in ['melody_history.json', 'accompaniment_history.json']:
            assert (session / name).is_file(), name
    return {'session_dir': str(session), 'combined': str(session / 'combined.mid'),
            'native_raw_logging': True, 'extra_model_observer': False}


def legacy_trial(root, cohort_name, cohort, case, seed, gpu):
    out = root / 'runs' / cohort_name / 'legacy_m2a' / case['piece_id'] / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    one = out / 'input_manifest.json'
    save(one, {'samples': [case]})
    command = [str(PYTHON), str(LEGACY / 'scripts/run_legacy_m2a_matched_eval.py'),
        '--cohort-manifest', str(one), '--output-root', str(out / 'run'), '--gpu', str(gpu),
        '--seeds', str(seed), '--checkpoint', str(CHECKPOINTS['legacy'][0]),
        '--tempo', str(cohort['playback_bpm']), '--ticks-per-beat', '4',
        '--generation-interval-ticks', '4', '--generation-length-frames', '7',
        '--client-stop-tick', str(case['stop_tick']), '--system-trace']
    save(out / 'command.json', command)
    with (out / 'runner.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=case['stop_tick'] * 60 / cohort['playback_bpm'] / 4 + 240)
    manifest = read(out / 'run/run_manifest.json')
    trial = manifest['trials'][0]
    assert trial['status'] == 'completed', trial.get('error')
    assert trial['effective_seed'] == seed
    assert trial['input_sha256_actual'] == case['melody_midi_sha256']
    assert trial['config']['melody_duration_policy'] == 'midi_file_original'
    assert '--midi-file-use-original-duration' in trial['commands']['client']
    return {'combined': trial['output_midi'], 'manifest': str(out / 'run/run_manifest.json'),
            'melody_notes': trial['midi']['melody_notes'], 'accompaniment_notes': trial['midi']['accompaniment_notes']}


def publish_playback(root, cohort_name, condition, case, seed, result):
    label = f"{case['order']:02d}_{case['title']}" if cohort_name == 'user10' else f"{case['order']:02d}_{case['piece_id']}"
    dest = root / 'listening_by_cohort' / cohort_name / label
    dest.mkdir(parents=True, exist_ok=True)
    source = Path(result['combined'])
    target = dest / f'{condition}_s{seed}.mid'
    assert not target.exists(), target
    shutil.copy2(source, target)
    if not (dest / '00_melody.mid').exists():
        shutil.copy2(case['melody_midi'], dest / '00_melody.mid')
    if 'gt_midi' in case and not (dest / '00_gt.mid').exists():
        shutil.copy2(case['gt_midi'], dest / '00_gt.mid')
    assert matched.file_sha256(source) == matched.file_sha256(target)
    return str(target)


def worker(args):
    assert args.gpu in (0, 1, 2)
    used = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert int(used.strip()) < 500, f'GPU {args.gpu} is busy'
    plan = read(args.root / 'plan.json')
    assert source_identity()['git_commit'] == plan['modern_code']['git_commit']
    for checkpoint in plan['checkpoints'].values():
        assert matched.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
    if not args.smoke:
        for gpu in (0, 1, 2):
            assert read(args.root / '_smoke/status' / f'gpu{gpu}.json')['status'] == 'completed'
    root = args.root / '_smoke' if args.smoke else args.root
    status_path = root / 'status' / f'gpu{args.gpu}.json'
    assert not status_path.exists(), status_path
    status = {'status': 'running', 'gpu': args.gpu, 'seed': 0 if args.smoke else args.gpu,
              'smoke': args.smoke, 'completed': [], 'started_unix': time.time()}
    save(status_path, status)
    handle = None
    try:
        for name, cohort in plan['cohorts'].items():
            assert matched.file_sha256(Path(cohort['manifest'])) == cohort['manifest_sha256']
            cases = cohort['pieces'][:1] if args.smoke else cohort['pieces']
            conditions = [CONDITIONS[args.gpu]] if args.smoke else CONDITIONS
            for condition in conditions:
                seed = status['seed']
                if condition != 'legacy_m2a':
                    handle = matched.start_server(IDS[condition], output_root=root / '_servers' / name / f'gpu{args.gpu}' / condition,
                        python_bin=str(PYTHON), gpu=str(args.gpu), timeout_s=180,
                        code=plan['modern_code'], prompt_checkpoint=plan['checkpoints']['prompt'],
                        continuation_checkpoint=plan['checkpoints']['continuation'], time_signature_index=4,
                        prompt_selection_mode='rule_s_if_else' if condition == 'pc_rule_if_else_n10' else 'single',
                        prompt_batch_candidates=10 if condition == 'pc_rule_if_else_n10' else 1,
                        contract=contract_for(cohort, condition, cases[0]['stop_tick']))
                for case in cases:
                    status['current'] = {'cohort': name, 'condition': condition, 'piece': case['piece_id']}
                    save(status_path, status)
                    assert matched.file_sha256(Path(case['melody_midi'])) == case['melody_midi_sha256']
                    print('START', status['current'], seed, flush=True)
                    if condition == 'legacy_m2a':
                        result = legacy_trial(root, name, cohort, case, seed, args.gpu)
                    else:
                        out = root / 'runs' / name / condition / case['piece_id'] / f'seed{seed}'
                        record = matched.run_trial(handle=handle,
                            piece=matched.CohortPiece(case['piece_id'], Path(case['melody_midi']), case['melody_midi_sha256']),
                            seed=seed, trial_dir=out, python_bin=str(PYTHON),
                            timeout_s=case['stop_tick'] * 60 / cohort['playback_bpm'] / 4 + 180,
                            code=plan['modern_code'], prompt_checkpoint=plan['checkpoints']['prompt'],
                            continuation_checkpoint=plan['checkpoints']['continuation'], time_signature_index=4,
                            prompt_selection_mode='rule_s_if_else' if condition == 'pc_rule_if_else_n10' else 'single',
                            prompt_batch_candidates=10 if condition == 'pc_rule_if_else_n10' else 1,
                            contract=contract_for(cohort, condition, case['stop_tick']))
                        assert record['run_status'] == 'complete', record.get('failure_reason')
                        result = validate_modern_artifacts(record, condition)
                    result.update(status['current'], seed=seed)
                    result['listening_midi'] = publish_playback(root, name, condition, case, seed, result)
                    status['completed'].append(result)
                    save(status_path, status)
                    print('DONE', status['current'], 'completed', len(status['completed']), flush=True)
                if handle is not None:
                    matched.stop_server(handle)
                    handle = None
        status.update(status='completed', finished_unix=time.time())
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if handle is not None:
            matched.stop_server(handle)
        save(status_path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()

"""Retest only ordinary Lekai after the playback-only late-onset drop fix."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import run_formal_user10_test40 as formal
import run_matched_system_eval as matched

CONDITION = 'lekai_no_prompt'
FIX = '13beb814'


def source_identity():
    code = matched.code_identity()
    assert code['git_clean'], code['git_status_porcelain']
    assert not subprocess.check_output(
        ['git', 'diff', FIX, 'HEAD', '--', 'src', 'transformers'], cwd=matched.REPO_ROOT,
    ), 'Production source differs from the accepted playback fix'
    return code


def prepare(args):
    assert not args.root.exists(), args.root
    baseline_path = args.baseline / 'plan.json'
    baseline = formal.read(baseline_path)
    plan = copy.deepcopy(baseline)
    code = source_identity()
    plan.update(modern_code=code, modern_system_ref=code['git_commit'], root=str(args.root),
                conditions=[CONDITION], expected_trials=150, playback_fix=FIX,
                baseline_plan=str(baseline_path), baseline_plan_sha256=matched.file_sha256(baseline_path))
    for key in ['legacy', 'legacy_code', 'raw_gate']:
        plan.pop(key, None)
    plan['checkpoints'] = {'continuation': baseline['checkpoints']['continuation']}
    plan['modern']['late_note_on_policy'] = 'drop_preserve_active_note_off_cleanup'
    plan['modern']['prompt_model_used'] = False
    plan['modern']['continuation_constraints'] = False
    for name, cohort in plan['cohorts'].items():
        assert len(cohort['pieces']) == (10 if name == 'user10' else 40)
        assert matched.file_sha256(Path(cohort['manifest'])) == cohort['manifest_sha256']
        for case in cohort['pieces']:
            assert matched.file_sha256(Path(case['melody_midi'])) == case['melody_midi_sha256']
    checkpoint = plan['checkpoints']['continuation']
    assert matched.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
    formal.save(args.root / 'plan.json', plan)
    shutil.copy2(matched.REPO_ROOT / 'docs/lekai_late_drop_retest.md', args.root / 'README.md')
    print(json.dumps({'root': str(args.root), 'trials': 150, 'code': code}))


def signatures(session):
    records = formal.read(Path(session) / 'inferences.json')
    result = {}
    for record in records:
        start = record['request_data']['generation_start_tick']
        response = record['response_data']
        meta = response['response_metadata']
        assert start not in result, (session, start)
        result[start] = {
            'part0_tokens': meta['part0_tokens'], 'raw_tokens': meta['raw_tokens'],
            'structural_tokens': meta['structural_tokens'], 'events': response['accompaniment'],
        }
    return result


def validate_trial(record, cohort, case, seed, stop_tick):
    assert record['run_status'] == 'complete', record.get('failure_reason')
    assert record['actual_seeds'] == {'continuation_seed': seed, 'prompt_seed': None}
    assert record['evaluation_contract']['prompt_model_used'] is False
    assert record['evaluation_contract']['continuation_constraints'] is False
    assert record['evaluation_contract']['continuation_sampling'] == cohort['sampling']
    assert record['melody_input_sha256'] == case['melody_midi_sha256']
    result = formal.validate_modern_artifacts(record, CONDITION)
    session = Path(record['session_dir'])
    trace = [json.loads(line) for line in (session / 'model_schedule_trace.jsonl').read_text().splitlines() if line]
    ons = [r for r in trace if r.get('type') == 'note_on' and r.get('velocity', 0) > 0]
    assert all(r['scheduled_tick'] == r['logical_tick'] for r in ons if r['action'] == 'scheduled')
    assert not any(str(r.get('policy', '')).startswith('clamped_') for r in trace)
    assert sorted(signatures(session)) == list(range(4, stop_tick + 1, 4)), 'Incomplete generation trace'
    result['late_note_on_drops'] = sum(r['action'] == 'dropped' for r in ons)
    return result


def worker(args):
    assert args.gpu in (0, 1, 2)
    used = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    assert int(used.strip()) < 500, f'GPU {args.gpu} is busy'
    plan = formal.read(args.root / 'plan.json')
    assert source_identity()['git_commit'] == plan['modern_code']['git_commit']
    assert matched.file_sha256(Path(plan['baseline_plan'])) == plan['baseline_plan_sha256']
    checkpoint = plan['checkpoints']['continuation']
    assert matched.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
    if not args.smoke:
        gate = formal.read(args.root / '_smoke/acceptance.json')
        assert gate['accepted'] and gate['git_commit'] == plan['modern_code']['git_commit']
    root = args.root / '_smoke' if args.smoke else args.root
    status_path = root / 'status' / f'gpu{args.gpu}.json'
    assert not status_path.exists(), status_path
    status = dict(status='running', gpu=args.gpu, seed=args.gpu, smoke=args.smoke,
                  completed=[], started_unix=time.time())
    formal.save(status_path, status)
    handle = None
    try:
        for name in ('user10', 'test40'):
            cohort = plan['cohorts'][name]
            assert matched.file_sha256(Path(cohort['manifest'])) == cohort['manifest_sha256']
            cases = cohort['pieces']
            if args.smoke:
                if name == 'user10':
                    cases = cases[args.gpu:args.gpu + 1]
                else:
                    pid = ['5849289', '6123939', '5509144'][args.gpu]
                    cases = [next(c for c in cases if c['piece_id'] == pid)]
            handle = matched.start_server(
                formal.IDS[CONDITION], output_root=root / '_servers' / name / f'gpu{args.gpu}',
                python_bin=str(formal.PYTHON), gpu=str(args.gpu), timeout_s=180,
                code=plan['modern_code'], prompt_checkpoint={}, continuation_checkpoint=checkpoint,
                time_signature_index=4, prompt_selection_mode='single', prompt_batch_candidates=1,
                contract=formal.contract_for(cohort, CONDITION, 128 if args.smoke else cases[0]['stop_tick']),
            )
            for case in cases:
                assert matched.file_sha256(Path(case['melody_midi'])) == case['melody_midi_sha256']
                for repeat in range(2 if args.smoke else 1):
                    stop_tick = min(case['stop_tick'], 128) if args.smoke else case['stop_tick']
                    status['current'] = dict(cohort=name, condition=CONDITION, piece=case['piece_id'],
                                             repeat=repeat, stop_tick=stop_tick)
                    formal.save(status_path, status)
                    print('START', status['current'], args.gpu, flush=True)
                    out = root / 'runs' / name / CONDITION / case['piece_id'] / f'seed{args.gpu}'
                    if args.smoke:
                        out = out / f'repeat{repeat}'
                    record = matched.run_trial(
                        handle=handle,
                        piece=matched.CohortPiece(case['piece_id'], Path(case['melody_midi']), case['melody_midi_sha256']),
                        seed=args.gpu, trial_dir=out, python_bin=str(formal.PYTHON),
                        timeout_s=stop_tick * 60 / cohort['playback_bpm'] / 4 + 180,
                        code=plan['modern_code'], prompt_checkpoint={}, continuation_checkpoint=checkpoint,
                        time_signature_index=4, prompt_selection_mode='single', prompt_batch_candidates=1,
                        contract=formal.contract_for(cohort, CONDITION, stop_tick),
                    )
                    result = validate_trial(record, cohort, case, args.gpu, stop_tick)
                    result.update(status['current'], seed=args.gpu)
                    if not args.smoke:
                        result['listening_midi'] = formal.publish_playback(root, name, CONDITION, case, args.gpu, result)
                    status['completed'].append(result)
                    formal.save(status_path, status)
                    print('DONE', status['current'], 'completed', len(status['completed']), flush=True)
            matched.stop_server(handle)
            handle = None
        status.update(status='completed', finished_unix=time.time())
    except BaseException as exc:
        status.update(status='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        if handle is not None:
            matched.stop_server(handle)
        formal.save(status_path, status)


def accept(args):
    plan = formal.read(args.root / 'plan.json')
    assert source_identity()['git_commit'] == plan['modern_code']['git_commit']
    comparisons = []
    baseline = Path(plan['baseline_plan']).parent
    for gpu in (0, 1, 2):
        status = formal.read(args.root / '_smoke/status' / f'gpu{gpu}.json')
        assert status['status'] == 'completed' and len(status['completed']) == 4
        for cohort in ('user10', 'test40'):
            trials = [r for r in status['completed'] if r['cohort'] == cohort]
            assert len(trials) == 2
            left, right = [signatures(r['session_dir']) for r in trials]
            assert left == right, (cohort, gpu, 'same-seed raw/input repeat mismatch')
            old_manifest = baseline / 'runs' / cohort / CONDITION / trials[0]['piece'] / f'seed{gpu}/trial_manifest.json'
            old = signatures(formal.read(old_manifest)['session_dir'])
            assert all(start in old and old[start] == value for start, value in left.items()), (cohort, gpu, 'old/new input/raw mismatch')
            comparisons.append(dict(cohort=cohort, piece=trials[0]['piece'], seed=gpu,
                                    calls=len(left), repeat_exact=True, baseline_prefix_exact=True))
    gate = dict(accepted=True, git_commit=plan['modern_code']['git_commit'], comparisons=comparisons)
    formal.save(args.root / '_smoke/acceptance.json', gate)
    print(json.dumps(gate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker', 'accept'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare' and args.baseline is None:
        parser.error('prepare requires --baseline')
    if args.action == 'worker' and args.gpu is None:
        parser.error('worker requires --gpu')
    globals()[args.action](args)


if __name__ == '__main__':
    main()

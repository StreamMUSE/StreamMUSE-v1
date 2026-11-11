#!/usr/bin/env python3
"""Offline H200 inference stack, restricted to physical GPUs 1 and 2.

Run on H200: prepare; start vllm; start tts; wait vllm; wait tts;
start render; wait render; verify. Services intentionally survive this controller.
Only stop <service> signals processes, using the pinned pidfd ownership helper.
No installation, SSH configuration, source sync, or unrelated process management.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

DEPLOY = Path('/data/home/Andrew.Yang/StreamMUSE/deploy')
ROOT = DEPLOY / 'rap_inference_20261111'
MOSS = DEPLOY / 'rap_mosslocal_20261028'
BASELINES = {
    'vllm': DEPLOY / 'rap_flow_audio_20261028/vllm.pid.json',
    'tts': MOSS / 'tts.pid.json',
    'render': DEPLOY / 'rap_research_20261105/render.pid.json',
}
SERVICES = {
    'vllm': (18711, 1, 'GPU-74ee34a6-23bd-7e60-e012-6bf3d30b4c3a'),
    'tts': (18730, 2, 'GPU-42777f32-16c2-0681-57bb-f72568e6ee57'),
    'render': (18721, 2, 'GPU-42777f32-16c2-0681-57bb-f72568e6ee57'),
}
HELPER = MOSS / 'rap_flow_audio_20261028_deploy.py'
HELPER_SHA = '01545d7c6a81e40a668dbb0063da562a64621a436feeef9d946d6eb0d0a3ec53'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safety_helper():
    if sha256(HELPER) != HELPER_SHA:
        raise RuntimeError('Pinned process-ownership helper changed; refusing operation')
    spec = importlib.util.spec_from_file_location('inference_safety', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SERVICES = SERVICES
    return module


def replace_options(argv, values):
    result = list(argv)
    for key, value in values.items():
        if result.count(key) != 1:
            raise RuntimeError(f'Expected one baseline option: {key}')
        result[result.index(key) + 1] = str(value)
    return result


def remap(name, baseline):
    plan = copy.deepcopy(baseline)
    port, index, gpu_uuid = SERVICES[name]
    values = {'--host': '127.0.0.1', '--port': port}
    if name == 'vllm':
        values['--gpu-memory-utilization'] = '0.25'
    elif name == 'tts':
        values.update({'--tts_engine.gpu_memory_fraction': '0.3',
                       '--tts_engine.engine.mem_fraction_static': '0.3', '--mps': 'off'})
    else:
        values.update({'--artifact-root': ROOT / 'render_artifacts',
                       '--aligner-device': 'cuda:0',
                       '--vllm-url': 'http://127.0.0.1:18711/v1',
                       '--moss-sglang-url': 'http://127.0.0.1:18730'})
    argv = replace_options(plan['argv'], values)
    environment = plan['environment']
    environment.pop('STREAMMUSE_DEPLOYMENT_TOKEN', None)
    environment.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    environment.pop('CUDA_MPS_LOG_DIRECTORY', None)
    environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       STREAMMUSE_DEPLOYMENT_ROOT=str(ROOT), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    cwd = plan['cwd']
    if name != 'tts':
        environment['PYTHONPATH'] = environment.get('PYTHONPATH', '').replace(cwd, str(ROOT / 'repo'))
        cwd = str(ROOT / 'repo')
    return {'service': name, 'root': str(ROOT), 'argv': argv, 'environment': environment,
            'cwd': cwd, 'port': port, 'gpu_index': index, 'gpu_uuid': gpu_uuid,
            'baseline': str(BASELINES[name]), 'evidence': plan.get('evidence', {})}


def port_preflight(port):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))


def validate_gpu(name, gpu):
    _, index, gpu_uuid = SERVICES[name]
    if gpu['index'] != index or gpu['uuid'] != gpu_uuid:
        raise RuntimeError(f'Physical GPU mapping changed for {name}')
    fraction = {'vllm': 0.25, 'tts': 0.30, 'render': 0}[name]
    required = fraction * gpu['total_mib'] + 8192
    if gpu['total_mib'] - gpu['memory_mib'] < required:
        raise RuntimeError(f'Insufficient shared GPU headroom for {name}: {gpu}')


def snapshot(safety):
    result = safety.gpu_snapshot()
    totals = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.total', '--format=csv,noheader,nounits'], text=True)
    total_by_uuid = {u.strip(): int(n.strip()) for u, n in
                     (line.split(',') for line in totals.strip().splitlines())}
    for gpu in result['gpus']:
        gpu['total_mib'] = total_by_uuid[gpu['uuid']]
    return result


def prepare(safety):
    if (ROOT / 'prepared.json').exists() or (ROOT / 'repo').exists():
        raise RuntimeError('Already prepared or partial preparation exists; never overwrite source')
    before = snapshot(safety)
    for name, (port, index, _) in SERVICES.items():
        port_preflight(port)
        validate_gpu(name, next(g for g in before['gpus'] if g['index'] == index))
    baseline_paths = {}
    for name, path in BASELINES.items():
        baseline = json.loads(path.read_text())
        baseline.pop('token', None)
        baseline['environment'].pop('STREAMMUSE_DEPLOYMENT_TOKEN', None)
        frozen = ROOT / f'{name}.baseline.json'
        safety.write_json(frozen, baseline)
        baseline_paths[name] = {'path': str(path), 'sha256': sha256(path)}
        plan = remap(name, baseline)
        if not Path(plan['argv'][0]).is_file():
            raise RuntimeError(f'Missing cached executable: {plan["argv"][0]}')
    tts = json.loads((ROOT / 'tts.baseline.json').read_text())
    evidence = tts['evidence']
    for key in ('environment', 'config', 'patch', 'reference_audio', 'reference_text'):
        if sha256(Path(evidence[key + '_path'])) != evidence[key + '_sha256']:
            raise RuntimeError(f'Cached MOSS evidence changed: {key}')
    for key in ('resolved_model_path', 'resolved_codec_model_path'):
        if not (Path(evidence[key]) / 'config.json').is_file():
            raise RuntimeError(f'Missing cached model: {key}')
    repo = Path(json.loads((ROOT / 'render.baseline.json').read_text())['cwd'])
    shutil.copytree(repo, ROOT / 'repo', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    versions = {}
    for name, python, packages in (
        ('qwen', Path(json.loads((ROOT / 'vllm.baseline.json').read_text())['argv'][0]).with_name('python'), ['vllm', 'torch', 'transformers']),
        ('moss', MOSS / 'env/bin/python', ['sglang-omni', 'sglang', 'torch', 'transformers']),
        ('renderer', Path(json.loads((ROOT / 'render.baseline.json').read_text())['argv'][0]), ['torch', 'torchaudio', 'transformers']),
    ):
        code = 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ' + repr(packages) + '}))'
        versions[name] = json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True))
    if versions['moss']['sglang'] != '0.5.21' or versions['moss']['sglang-omni'] != '0.1.8':
        raise RuntimeError('MOSS serving package versions changed')
    result = {'at': safety.now(), 'baseline_paths': baseline_paths, 'source_baseline': str(repo),
              'source_sha256': safety.source_manifest(), 'versions': versions,
              'safety_helper': str(HELPER), 'safety_helper_sha256': HELPER_SHA,
              'gpu_before': before}
    safety.write_json(ROOT / 'prepared.json', result)
    return result


def load_record(name):
    record = json.loads((ROOT / f'{name}.pid.json').read_text())
    if (record['root'] != str(ROOT) or record['service'] != name
            or record['gpu_uuid'] != SERVICES[name][2] or record['port'] != SERVICES[name][0]
            or record['safety_helper_sha256'] != HELPER_SHA):
        raise RuntimeError('Deployment ownership record mismatch')
    return record


def start(safety, name):
    if not (ROOT / 'prepared.json').is_file():
        raise RuntimeError('Run prepare first')
    if (ROOT / f'{name}.pid.json').exists():
        raise RuntimeError('Existing PID record; inspect/stop explicitly before relaunch')
    if name == 'render':
        for dependency in ('vllm', 'tts'):
            if not safety.alive(load_record(dependency)):
                raise RuntimeError(f'Owned dependency not alive: {dependency}')
            safety.fetch(f'http://127.0.0.1:{SERVICES[dependency][0]}/health')
    plan = remap(name, json.loads((ROOT / f'{name}.baseline.json').read_text()))
    port_preflight(plan['port'])
    before = snapshot(safety)
    validate_gpu(name, next(g for g in before['gpus'] if g['index'] == plan['gpu_index']))
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):
        raise RuntimeError('External MPS configured; refusing launch')
    environment = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
    changes = {**plan['environment'], 'STREAMMUSE_DEPLOYMENT_TOKEN': uuid.uuid4().hex}
    environment.update(changes)
    source_sha = safety.source_manifest()
    safety.write_json(ROOT / f'{name}.preflight.json', before)
    safety.write_json(ROOT / f'{name}.launch.json', {**plan, 'environment': changes})
    log = ROOT / f'{name}.log'
    with log.open('ab', buffering=0) as output:
        process = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = {**plan, 'environment': changes, 'pid': process.pid, 'token': changes['STREAMMUSE_DEPLOYMENT_TOKEN'],
              'started_at': safety.now(), 'log': str(log), 'source_sha256': source_sha,
              'launcher_sha256': sha256(Path(__file__)), 'safety_helper_sha256': HELPER_SHA}
    safety.write_json(ROOT / f'{name}.pid.json', record)
    record['identity'] = safety.proc_identity(process.pid)
    safety.write_json(ROOT / f'{name}.pid.json', record)
    return {'service': name, 'pid': process.pid, 'gpu_index': plan['gpu_index'],
            'gpu_uuid': plan['gpu_uuid'], 'port': plan['port'], 'log': str(log)}


def status(safety, name):
    record = load_record(name)
    owned = 'identity' in record and safety.alive(record)
    result = {'at': safety.now(), 'service': name, 'pid': record['pid'], 'owned_alive': owned,
              'port': record['port'], 'gpu_index': record['gpu_index'], 'gpu_uuid': record['gpu_uuid']}
    if owned:
        result['members'] = safety.group_members(record)
        result['process_environment'] = []
        for identity in result['members']:
            env = safety.proc_environment(identity['pid'])
            result['process_environment'].append({'pid': identity['pid'], **{
                key: env.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'STREAMMUSE_DEPLOYMENT_ROOT')}})
        try:
            result['health'] = safety.fetch(f'http://127.0.0.1:{record["port"]}/health')
        except (OSError, ValueError) as error:
            result['health_error'] = str(error)
    safety.write_json(ROOT / f'{name}.status.json', result)
    return result


def wait(safety, name, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        record = load_record(name)
        if not safety.alive(record):
            raise RuntimeError(f'{name} leader exited; inspect {record["log"]}; no signals sent')
        try:
            health = safety.fetch(f'http://127.0.0.1:{record["port"]}/health')
            if name != 'render' or health['body'].get('ready') is True:
                return status(safety, name)
        except (OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError(f'{name} warmup timeout; services left running')


def verify(safety):
    states = {name: status(safety, name) for name in SERVICES}
    for name, state in states.items():
        if not state['owned_alive'] or state.get('health', {}).get('status') != 200:
            raise RuntimeError(f'Service not healthy/owned: {name}')
    health = states['render']['health']['body']
    if health.get('ready') is not True or any(health.get(k, {}).get('ready') is not True for k in ('moss', 'aligner', 'warmup')):
        raise RuntimeError('Renderer warmup is incomplete')
    request = urllib.request.Request('http://127.0.0.1:18711/v1/chat/completions',
        data=json.dumps({'model': 'qwen-rap', 'messages': [{'role': 'user', 'content': 'Reply with one word: ready'}],
                         'max_tokens': 8, 'temperature': 0}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        completion = json.load(response)
    if not completion.get('choices'):
        raise RuntimeError('Qwen inference returned no choices')
    gpu = snapshot(safety)
    owned_gpu = []
    for name, state in states.items():
        members = {member['pid'] for member in state['members']}
        matched = [p for p in gpu['processes'] if p['pid'] in members]
        if not matched or any(p['gpu_uuid'] != SERVICES[name][2] for p in matched):
            raise RuntimeError(f'GPU process mapping invalid or absent: {name}')
        owned_gpu.extend({'service': name, **p} for p in matched)
    result = {'at': safety.now(), 'services': states, 'gpu': gpu, 'owned_gpu_processes': owned_gpu,
              'qwen_completion': completion, 'models': {name: safety.fetch(
                  f'http://127.0.0.1:{SERVICES[name][0]}/v1/models') for name in ('vllm', 'tts')}}
    safety.write_json(ROOT / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'plan', 'start', 'wait', 'status', 'stop', 'verify'])
    parser.add_argument('service', nargs='?', choices=list(SERVICES))
    parser.add_argument('--timeout', type=float, default=2400)
    args = parser.parse_args()
    if args.action not in ('prepare', 'verify') and args.service is None:
        parser.error('This action requires an exact service name')
    os.umask(0o077)
    safety = safety_helper()
    with safety.control_lock():
        if args.action == 'prepare':
            result = prepare(safety)
        elif args.action == 'plan':
            result = remap(args.service, json.loads((ROOT / f'{args.service}.baseline.json').read_text()))
        elif args.action == 'start':
            result = start(safety, args.service)
        elif args.action == 'wait':
            result = wait(safety, args.service, args.timeout)
        elif args.action == 'status':
            result = status(safety, args.service)
        elif args.action == 'stop':
            load_record(args.service)
            result = safety.stop(args.service)
        else:
            result = verify(safety)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Refused: {error}', file=sys.stderr)
        sys.exit(1)

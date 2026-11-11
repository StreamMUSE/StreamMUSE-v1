#!/usr/bin/env python3
"""Local-only MOSS-Local deployment controller; never uses SSH or manages Qwen.

Before stopping the old renderer, run `plan render` to preserve its manifest,
or pass --baseline /path/to/a/saved/render.pid.json. Plans require the prepared
runtime files, but never query GPUs or services. Start returns after spawning;
use status to inspect health. Stop uses the existing strict pidfd group helper.
Nothing in OLDROOT is modified, and no rollback files are deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path('/data/home/Andrew.Yang/StreamMUSE/deploy/rap_mosslocal_20261028')
OLDROOT = Path('/data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028')
GPU_UUID = 'GPU-74ee34a6-23bd-7e60-e012-6bf3d30b4c3a'
MODEL = 'OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5'
MODEL_REVISION = 'be7766a6735b98bd793f7c79fb720b4d0f5d13b8'
CODEC_REVISION = 'f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169'
OMNI_REVISION = '870d77446b3fae6e4f4f2a6964f7eae2f75e663a'
HF_HOME = Path.home() / '.cache/huggingface'
PORTS = {'tts': 18730, 'render': 18720}


def load_safety(root):
    path = Path(__file__).with_name('rap_flow_audio_20261028_deploy.py')
    if not path.is_file():
        path = OLDROOT / 'deploy.py'
    spec = importlib.util.spec_from_file_location('scoped_deployment_' + root.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


# Independent module globals keep Qwen identity checks bound to OLDROOT.
safety = load_safety(ROOT)
old_safety = load_safety(OLDROOT)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_snapshot():
    return ROOT / 'model' / MODEL_REVISION


def codec_snapshot():
    return ROOT / 'codec' / CODEC_REVISION


def validate_baseline(record):
    argv = record.get('argv', [])
    if (record.get('root') != str(OLDROOT) or record.get('service') != 'render'
            or record.get('port') != 18720 or record.get('gpu_uuid') != GPU_UUID
            or len(argv) < 3 or not Path(argv[0]).is_absolute()
            or 'streammuse.presentation.rap_render_server' not in argv
            or not isinstance(record.get('environment'), dict)):
        raise RuntimeError('Not the expected old renderer baseline manifest')


def load_baseline(path=None):
    saved = ROOT / 'render.baseline.json'
    source = Path(path) if path else (saved if saved.exists() else OLDROOT / 'render.pid.json')
    record = json.loads(source.read_text())
    validate_baseline(record)
    if saved.exists():
        if json.loads(saved.read_text()) != record:
            raise RuntimeError('Saved baseline differs; refusing to overwrite rollback evidence')
    else:
        safety.write_json(saved, record)
    return record


def replace_options(argv, values, flags=(), remove=()):
    result = []
    stripped = set(values) | set(remove)
    index = 0
    while index < len(argv):
        token = argv[index]
        key = token.split('=', 1)[0]
        if key in stripped:
            if '=' not in token and (index + 1 >= len(argv) or argv[index + 1].startswith('--')):
                raise RuntimeError(f'Missing value for baseline option {key}')
            index += 1 if '=' in token else 2
        elif key in flags:
            index += 1
        else:
            result.append(token)
            index += 1
    for key, value in values.items():
        result.extend([key, str(value)])
    return result + list(flags)


def runtime_evidence(environment_sha256=None):
    locks = [ROOT / 'environment.lock.json', ROOT / 'environment.lock.txt']
    lock = next((path for path in locks if path.is_file()), None)
    if lock is None:
        raise RuntimeError('Missing environment.lock.json or environment.lock.txt')
    config = ROOT / 'sglang-omni/examples/configs/moss_tts_local_non_streaming.yaml'
    paths = {'environment': lock, 'config': config,
             'patch': ROOT / 'moss-local-continuation.patch',
             'reference_audio': ROOT / 'media/reference.wav',
             'reference_text': ROOT / 'media/reference.txt'}
    evidence = {}
    for name, path in paths.items():
        evidence[name + '_path'] = str(path)
        evidence[name + '_sha256'] = sha256(path)
    if environment_sha256 is not None and environment_sha256 != evidence['environment_sha256']:
        raise RuntimeError('Supplied environment SHA-256 does not match the actual lock file')
    model = model_snapshot()
    if model.is_symlink():
        raise RuntimeError(f'Pinned materialized model must not be a symlink: {model}')
    if not model.is_dir():
        raise RuntimeError(f'Pinned materialized model is missing: {model}')
    # Transformers resolves Python symlinks into HF blobs, losing sibling imports.
    for path in model.rglob('*'):
        if path.is_symlink():
            raise RuntimeError(f'Pinned materialized model contains a symlink: {path}')
    for filename in ('config.json', 'configuration_moss_tts.py'):
        if not (model / filename).is_file():
            raise RuntimeError(f'Pinned materialized model is missing {filename}: {model}')
    codec = codec_snapshot()
    if codec.is_symlink():
        raise RuntimeError(f'Pinned materialized codec must not be a symlink: {codec}')
    if not codec.is_dir():
        raise RuntimeError(f'Pinned materialized codec is missing: {codec}')
    for path in codec.rglob('*'):
        if path.is_symlink():
            raise RuntimeError(f'Pinned materialized codec contains a symlink: {path}')
    if not (codec / 'config.json').is_file():
        raise RuntimeError(f'Pinned materialized codec is missing config.json: {codec}')
    evidence.update(model=MODEL, model_revision=MODEL_REVISION,
                    resolved_model_path=str(model.resolve()),
                    codec_revision=CODEC_REVISION, resolved_codec_model_path=str(codec.resolve()),
                    omni_version='0.1.8', omni_revision=OMNI_REVISION,
                    sglang_version='0.5.21', sglang_revision='pypi-0.5.21',
                    gpu_memory_fraction=0.3, mem_fraction_static=0.3,
                    mps='off', cpu_threads=8)
    return evidence


def build_plan(name, baseline=None, environment_sha256=None, *, wire_audio_codec='pcm'):
    if wire_audio_codec not in ('pcm', 'opus'):
        raise ValueError('Unsupported wire audio codec; expected pcm or opus')
    if name != 'render' and wire_audio_codec != 'pcm':
        raise ValueError('Wire audio codec selection is only supported for render')
    evidence = runtime_evidence(environment_sha256)
    if name == 'tts':
        argv = [str(ROOT / 'env/bin/sgl-omni'), 'serve',
                '--model-path', str(model_snapshot()), '--config', evidence['config_path'],
                '--model-name', MODEL,
                '--mps', 'off',
                '--preprocessing.factory.codec_model_path', str(codec_snapshot()),
                '--vocoder.factory.codec_model_path', str(codec_snapshot()),
                '--tts_engine.gpu_memory_fraction', '0.3',
                '--tts_engine.engine.mem_fraction_static', '0.3',
                '--host', '127.0.0.1', '--port', '18730',
                '--allowed-local-media-path', str(ROOT / 'media')]
        environment = {'PATH': f'{ROOT}/env/bin:/usr/local/cuda/bin:/usr/bin:/bin',
                       'SGLANG_OMNI_STARTUP_TIMEOUT': '2400',
                       'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                       'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8',
                       'PYTHONPATH': str(ROOT / 'sglang-omni')}
        cwd = ROOT / 'sglang-omni'
    elif name == 'render':
        validate_baseline(baseline)
        argv = replace_options(baseline['argv'], {
            '--host': '127.0.0.1', '--port': '18720',
            '--artifact-root': ROOT / 'render_artifacts',
            '--vllm-url': 'http://127.0.0.1:18711/v1',
            '--moss-model': MODEL, '--moss-model-revision': MODEL_REVISION,
            '--moss-serving-backend': 'sglang-omni',
            '--moss-sglang-url': 'http://127.0.0.1:18730',
            '--moss-reference-wav': evidence['reference_audio_path'],
            '--moss-reference-text-file': evidence['reference_text_path'],
            '--moss-sglang-reference-uri': Path(evidence['reference_audio_path']).as_uri(),
            '--moss-sglang-reference-sha256': evidence['reference_audio_sha256'],
            '--moss-runtime-version': '0.1.8', '--moss-runtime-revision': OMNI_REVISION,
            '--moss-sglang-version': '0.5.21', '--moss-sglang-revision': 'pypi-0.5.21',
            '--moss-runtime-environment-sha256': evidence['environment_sha256'],
            '--moss-runtime-config': evidence['config_path'],
            '--moss-runtime-config-sha256': evidence['config_sha256'],
            '--moss-runtime-patch-sha256': evidence['patch_sha256'],
            '--wire-audio-codec': wire_audio_codec, '--lyric-candidates': '32',
            **({'--opus-compression-level': '5'} if wire_audio_codec == 'opus' else {}),
        }, flags=('--lyric-stream', '--lyric-continuation', '--moss-sglang-resample-output'),
            remove=('--moss-device', '--moss-mlx-url', '--mlx-version', '--mlx-audio-version',
                    '--mlx-audio-commit', '--mlx-moss-quantization', '--mlx-moss-weights-sha256',
                    '--mlx-moss-audio-tokenizer-revision', '--opus-compression-level'))
        environment = dict(baseline['environment'])
        deps = [part for part in environment.get('PYTHONPATH', '').split(':')
                if part and not part.rstrip('/').endswith(('/repo', '/repo/src'))]
        environment['PYTHONPATH'] = ':'.join([str(ROOT / 'repo/src'), str(ROOT / 'repo'), *deps])
        environment.pop('STREAMMUSE_DEPLOYMENT_TOKEN', None)
        cwd = ROOT / 'repo'
    else:
        raise ValueError('Only tts and render are managed')
    environment.update(CUDA_VISIBLE_DEVICES=GPU_UUID, STREAMMUSE_DEPLOYMENT_ROOT=str(ROOT),
                       HF_HOME=str(HF_HOME), HF_HUB_CACHE=str(HF_HOME / 'hub'),
                       HUGGINGFACE_HUB_CACHE=str(HF_HOME / 'hub'),
                       TRANSFORMERS_CACHE=str(HF_HOME / 'hub'),
                       HF_MODULES_CACHE=str(HF_HOME / 'modules'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    return {'service': name, 'argv': argv, 'environment': environment, 'cwd': str(cwd),
            'evidence': evidence}


def verify_runtime():
    checkout = ROOT / 'sglang-omni'
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != OMNI_REVISION:
        raise RuntimeError('SGLang-Omni checkout revision mismatch')
    subprocess.run(['git', '-C', str(checkout), 'apply', '--reverse', '--check',
                    str(ROOT / 'moss-local-continuation.patch')], check=True, capture_output=True)
    code = ('import importlib.metadata as m, json; '
            'print(json.dumps({n:m.version(n) for n in ("sglang-omni", "sglang")}))')
    versions = json.loads(subprocess.check_output([str(ROOT / 'env/bin/python'), '-I', '-c', code], text=True))
    if versions != {'sglang-omni': '0.1.8', 'sglang': '0.5.21'}:
        raise RuntimeError(f'Runtime package version mismatch: {versions}')
    return {'revision': revision, 'versions': versions, 'patch_reverse_check': True}


def validate_gpu(name, gpu, processes, allowed_pids=()):
    if gpu['index'] != 1 or gpu['uuid'] != GPU_UUID:
        raise RuntimeError('GPU index/UUID changed; refusing launch')
    occupants = {process['pid'] for process in processes if process['gpu_uuid'] == GPU_UUID}
    if name == 'tts':
        if occupants or gpu['memory_mib'] > 16:
            raise RuntimeError('GPU1 is occupied; refusing TTS launch')
    elif not allowed_pids or occupants - set(allowed_pids) or (gpu['memory_mib'] > 16 and not occupants):
        raise RuntimeError('GPU1 has unrelated or unattributed occupancy; refusing renderer launch')


def preflight(name):
    allowed = set()
    if name == 'render':
        record = json.loads((ROOT / 'tts.pid.json').read_text())
        if not safety.alive(record):
            raise RuntimeError('Owned new TTS service is not running')
        if record.get('service') != 'tts' or record.get('evidence') != runtime_evidence():
            raise RuntimeError('TTS runtime evidence differs from current files')
        allowed = {identity['pid'] for identity in safety.group_members(record)}
        safety.fetch('http://127.0.0.1:18730/health')
        qwen = json.loads((OLDROOT / 'vllm.pid.json').read_text())
        if not old_safety.alive(qwen):
            raise RuntimeError('Owned old Qwen service is not running')
        old_safety.require_health('vllm')
    safety.port_preflight(PORTS[name])
    snapshot = safety.gpu_snapshot()
    gpu = next((gpu for gpu in snapshot['gpus'] if gpu['index'] == 1), None)
    if gpu is None:
        raise RuntimeError('GPU1 missing')
    validate_gpu(name, gpu, snapshot['processes'], allowed)
    return snapshot


def start(name, plan):
    path = ROOT / f'{name}.pid.json'
    if path.exists():
        raise RuntimeError('Existing identity record; inspect/stop before relaunch')
    if (os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
            or plan['environment'].get('CUDA_MPS_PIPE_DIRECTORY')):
        raise RuntimeError('External MPS pipe configured; refusing launch')
    verification = verify_runtime()
    if plan['evidence'] != runtime_evidence():
        raise RuntimeError('Runtime files changed since plan')
    source_sha256 = safety.source_manifest()
    snapshot = preflight(name)
    changes = {**plan['environment'], 'STREAMMUSE_DEPLOYMENT_TOKEN': uuid.uuid4().hex}
    environment = dict(os.environ)
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PYTHONHOME'):
        environment.pop(key, None)
    environment.update(changes)
    safety.write_json(ROOT / f'{name}.preflight.json', snapshot)
    safety.write_json(ROOT / f'{name}.launch.json', {**plan, 'environment': changes})
    with (ROOT / f'{name}.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(plan['argv'], cwd=plan['cwd'], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = {**plan, 'root': str(ROOT), 'pid': process.pid, 'environment': changes,
              'token': changes['STREAMMUSE_DEPLOYMENT_TOKEN'], 'started_at': safety.now(),
              'gpu_index': 1, 'gpu_uuid': GPU_UUID, 'port': PORTS[name],
              'log': str(ROOT / f'{name}.log'), 'source_sha256': source_sha256,
              'launcher_sha256': sha256(Path(__file__)),
              'safety_helper_sha256': sha256(Path(safety.__file__)),
              'runtime_verification': verification}
    # Retain the PID even if immediate process exit prevents identity capture.
    safety.write_json(path, record)
    record['identity'] = safety.proc_identity(process.pid)
    safety.write_json(path, record)
    return record


def status(name):
    record = json.loads((ROOT / f'{name}.pid.json').read_text())
    owned = 'identity' in record and safety.alive(record)
    result = {'record': record, 'owned_alive': owned,
              'members': safety.group_members(record) if owned else [], 'endpoints': []}
    if owned:
        paths = ['/health', '/v1/models'] if name == 'tts' else ['/health', '/v1/rap/capabilities']
        for path in paths:
            url = f'http://127.0.0.1:{PORTS[name]}{path}'
            try:
                result['endpoints'].append(safety.fetch(url))
            except (OSError, ValueError) as error:
                result['endpoints'].append({'url': url, 'error': str(error)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'start', 'status', 'stop'])
    parser.add_argument('service', choices=['tts', 'render'])
    parser.add_argument('--baseline', type=Path, help='Saved old renderer PID manifest; never overwritten')
    parser.add_argument('--moss-runtime-environment-sha256', help='Optional expected lock-file hash; always computed')
    parser.add_argument('--wire-audio-codec', choices=['pcm', 'opus'],
                        help='Transport for plan/start render (default: pcm; Opus compression level: 5)')
    args = parser.parse_args(argv)
    if args.wire_audio_codec is not None and (args.service != 'render' or args.action not in ('plan', 'start')):
        parser.error('--wire-audio-codec is only valid for plan/start render')
    try:
        with safety.control_lock():
            if args.action in ('plan', 'start'):
                baseline = load_baseline(args.baseline) if args.service == 'render' else None
                result = build_plan(args.service, baseline, args.moss_runtime_environment_sha256,
                                    wire_audio_codec=args.wire_audio_codec or 'pcm')
                if args.action == 'start':
                    result = start(args.service, result)
            elif args.action == 'stop':
                result = safety.stop(args.service)
            else:
                result = status(args.service)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Refused: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

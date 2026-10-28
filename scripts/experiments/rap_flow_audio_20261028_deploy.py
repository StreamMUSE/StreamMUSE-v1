#!/usr/bin/env python3
"""Isolated, offline H200 deployment. Run --self-test without H200 access."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path('/data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028')
OLD = Path('/data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014')
SERVICES = {
    'vllm': (18711, 0, 'GPU-4f4e1a5f-dfd1-e747-40d2-993bf2548918'),
    'render': (18720, 1, 'GPU-74ee34a6-23bd-7e60-e012-6bf3d30b4c3a'),
}


def isolated_plan(name, argv, environment):
    argv, environment = list(argv), dict(environment)
    port, _, gpu_uuid = SERVICES[name]
    argv[argv.index('--port') + 1] = str(port)
    environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                       VLLM_CACHE_ROOT=str(ROOT / 'vllm-cache'),
                       STREAMMUSE_DEPLOYMENT_ROOT=str(ROOT))
    if name == 'render':
        argv[argv.index('--artifact-root') + 1] = str(ROOT / 'render_artifacts')
        argv[argv.index('--vllm-url') + 1] = 'http://127.0.0.1:18711/v1'
        environment['PYTHONPATH'] = f'{OLD}/deps:{ROOT}/repo/src:{ROOT}/repo'
        environment['HF_MODULES_CACHE'] = str(ROOT / 'hf-modules')
    return argv, environment


def validate_empty_gpu(name, gpu, processes):
    _, index, gpu_uuid = SERVICES[name]
    if gpu['index'] != index or gpu['uuid'] != gpu_uuid:
        raise RuntimeError('GPU index/UUID changed; refusing launch')
    if gpu['memory_mib'] > 16 or any(p['gpu_uuid'] == gpu_uuid for p in processes):
        raise RuntimeError(f'GPU{index} is occupied; refusing launch: {gpu}')


def can_stop(record, identity, environment, uid):
    return (record['identity'] == identity and identity['uid'] == uid
            and identity['pgid'] == identity['sid'] == record['pid']
            and record['root'] == str(ROOT)
            and environment.get('STREAMMUSE_DEPLOYMENT_TOKEN') == record['token']
            and environment.get('STREAMMUSE_DEPLOYMENT_ROOT') == str(ROOT)
            and environment.get('CUDA_VISIBLE_DEVICES') == record['gpu_uuid'])


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


@contextmanager
def control_lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    if ROOT.is_symlink() or ROOT.stat().st_uid != os.getuid():
        raise RuntimeError('Deployment root must be an owned real directory')
    with (ROOT / '.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def command(name):
    spec = importlib.util.spec_from_file_location('verified_deployment', OLD / 'deploy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv, environment, _ = module.command(name)
    return isolated_plan(name, argv, environment)


def gpu_snapshot():
    def rows(query):
        output = subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True)
        return [[v.strip() for v in row] for row in csv.reader(output.splitlines()) if row]
    gpus = [{'index': int(i), 'uuid': u, 'memory_mib': int(m)}
            for i, u, m in rows('--query-gpu=index,uuid,memory.used')]
    processes = [{'gpu_uuid': u, 'pid': int(p), 'process_name': n, 'memory_mib': m}
                 for u, p, n, m in rows('--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory')]
    return {'at': now(), 'gpus': gpus, 'processes': processes}


def port_preflight(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', port))


def preflight(names):
    snapshot = gpu_snapshot()
    for name in names:
        port_preflight(SERVICES[name][0])
        gpu = next(g for g in snapshot['gpus'] if g['index'] == SERVICES[name][1])
        validate_empty_gpu(name, gpu, snapshot['processes'])
        if (ROOT / f'{name}.pid.json').exists():
            raise RuntimeError(f'{name} has an identity record; inspect/stop it before relaunch')
    return snapshot


def proc_identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ProcessLookupError(f'{pid} is a zombie')
    return {'pid': pid, 'pgid': int(fields[2]), 'sid': int(fields[3]),
            'uid': proc.stat().st_uid, 'start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def proc_environment(pid):
    return dict(item.decode().split('=', 1) for item in
                (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in item)


def alive(record):
    try:
        return can_stop(record, proc_identity(record['pid']), proc_environment(record['pid']), os.getuid())
    except (FileNotFoundError, ProcessLookupError):
        return False


def source_manifest():
    files = {}
    for folder in ('src', 'scripts/rap_audio_backends'):
        for path in sorted((ROOT / 'repo' / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[str(path.relative_to(ROOT / 'repo'))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise RuntimeError('No copied repository sources found')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {'at': now(), 'sha256': digest, 'files': files}
    write_json(ROOT / f'source-{digest}.json', manifest)
    return digest


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode()
        return {'url': url, 'status': response.status, 'body': json.loads(body) if body else None}


def require_health(name):
    response = fetch(f'http://127.0.0.1:{SERVICES[name][0]}/health')
    if name == 'render' and response['body'].get('ready') is not True:
        raise RuntimeError('Renderer is not ready')
    return response


def start(name):
    if name == 'render':
        record = json.loads((ROOT / 'vllm.pid.json').read_text())
        if not alive(record):
            raise RuntimeError('Owned Qwen service is not running')
        require_health('vllm')
    argv, changes = command(name)
    changes['STREAMMUSE_DEPLOYMENT_TOKEN'] = uuid.uuid4().hex
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.pop('LD_LIBRARY_PATH', None)
    environment.update(changes)
    source_sha256 = source_manifest()
    snapshot = preflight([name])
    write_json(ROOT / f'{name}.preflight.json', snapshot)
    (ROOT / f'{name}.command.txt').write_text(
        'cd ' + shlex.quote(str(ROOT / 'repo')) + '\n' +
        shlex.join(['env', *[f'{k}={v}' for k, v in changes.items()], *argv]) + '\n')
    with (ROOT / f'{name}.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(argv, cwd=ROOT / 'repo', env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = {'service': name, 'root': str(ROOT), 'pid': process.pid,
              'identity': proc_identity(process.pid), 'token': changes['STREAMMUSE_DEPLOYMENT_TOKEN'],
              'started_at': now(), 'argv': argv, 'environment': changes,
              'port': SERVICES[name][0], 'gpu_index': SERVICES[name][1],
              'gpu_uuid': SERVICES[name][2], 'cwd': str(ROOT / 'repo'),
              'log': str(ROOT / f'{name}.log'), 'source_sha256': source_sha256,
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'old_launcher_sha256': hashlib.sha256((OLD / 'deploy.py').read_bytes()).hexdigest()}
    write_json(ROOT / f'{name}.pid.json', record)
    return record


def descendant_of(pid, leader):
    visited = set()
    while pid > 1 and pid not in visited:
        if pid == leader:
            return True
        visited.add(pid)
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        pid = int(fields[1])
    return False


def member_owned(record, identity, environment, descendant):
    return (identity['uid'] == os.getuid() and identity['pgid'] == record['pid']
            and identity['sid'] == record['pid']
            and identity['boot_id'] == record['identity']['boot_id']
            and int(identity['start_ticks']) >= int(record['identity']['start_ticks'])
            and descendant
            and environment.get('STREAMMUSE_DEPLOYMENT_TOKEN', record['token']) == record['token']
            and environment.get('STREAMMUSE_DEPLOYMENT_ROOT', str(ROOT)) == str(ROOT)
            and environment.get('CUDA_VISIBLE_DEVICES', record['gpu_uuid']) == record['gpu_uuid'])


def group_members(record):
    members = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            identity = proc_identity(int(proc.name))
            if identity['pgid'] != record['pid']:
                continue
            env = proc_environment(identity['pid'])
            # vLLM's titled worker can have empty /proc/environ. Require live
            # ancestry to the token-verified leader, never just a matching UID.
            if not member_owned(record, identity, env, descendant_of(identity['pid'], record['pid'])):
                raise RuntimeError(f'Unowned process in recorded group: {identity}')
            members.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop(name):
    path = ROOT / f'{name}.pid.json'
    record = json.loads(path.read_text())
    if not alive(record):
        raise RuntimeError('Leader missing or identity mismatch; no signal sent, record retained')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required; no unsafe fallback')
    handles = []
    try:
        # Pin every identity before any signal. Never broadcast to a numeric PGID.
        for identity in group_members(record):
            fd = os.pidfd_open(identity['pid'])
            handles.append((identity, fd))
            if proc_identity(identity['pid']) != identity:
                raise RuntimeError('Process identity changed while opening pidfd; no signal sent')
        if not any(identity == record['identity'] for identity, _ in handles):
            raise RuntimeError('Recorded leader disappeared; no signal sent')
        write_json(ROOT / f'{name}.shutdown.json', {'at': now(), 'members': [i for i, _ in handles]})
        for _, fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        until = time.monotonic() + 30
        while time.monotonic() < until:
            remaining = []
            for identity, _ in handles:
                try:
                    if proc_identity(identity['pid']) == identity:
                        remaining.append(identity)
                except (ProcessLookupError, FileNotFoundError):
                    pass
            if not remaining:
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        if proc_identity(int(proc.name))['pgid'] == record['pid']:
                            raise RuntimeError('New group member remains; no further signal sent')
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                archive = path.with_name(f'{name}.stopped-{time.time_ns()}.pid.json')
                path.rename(archive)
                return {'stopped': name, 'pid': record['pid'], 'archive': str(archive)}
            time.sleep(0.2)
        raise RuntimeError('Owned processes remain after SIGTERM; no escalation, record retained')
    finally:
        for _, fd in handles:
            os.close(fd)


def status(name):
    record = json.loads((ROOT / f'{name}.pid.json').read_text())
    result = {'at': now(), 'record': record, 'owned_alive': alive(record), 'gpu': gpu_snapshot()}
    if result['owned_alive']:
        result['members'] = group_members(record)
    paths = ['/health', '/v1/models'] if name == 'vllm' else ['/health', '/v1/rap/capabilities']
    result['endpoints'] = []
    for path in paths:
        url = f'http://127.0.0.1:{SERVICES[name][0]}{path}'
        try:
            result['endpoints'].append(fetch(url))
        except (OSError, ValueError) as error:
            result['endpoints'].append({'url': url, 'error': str(error)})
    write_json(ROOT / f'{name}.status.json', result)
    return result


def wait_ready(name, timeout):
    record = json.loads((ROOT / f'{name}.pid.json').read_text())
    until = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < until:
        if not alive(record):
            raise RuntimeError(f'{name} exited or identity changed; inspect {record["log"]}')
        try:
            require_health(name)
        except (OSError, ValueError, RuntimeError) as error:
            last_error = error
        else:
            return status(name)
        time.sleep(2)
    raise RuntimeError(f'{name} not ready within {timeout}s: {last_error}; process left running')


def self_test():
    import tempfile
    import unittest

    class SafetyTests(unittest.TestCase):
        def test_isolation_does_not_mutate_old_command(self):
            argv = ['python', '--port', '8620', '--artifact-root', 'old',
                    '--vllm-url', 'http://127.0.0.1:8601/v1']
            environment = {'CUDA_VISIBLE_DEVICES': 'old', 'HF_HUB_OFFLINE': '1'}
            planned, env = isolated_plan('render', argv, environment)
            self.assertEqual(planned[planned.index('--port') + 1], '18720')
            self.assertEqual(planned[planned.index('--artifact-root') + 1], str(ROOT / 'render_artifacts'))
            self.assertEqual(planned[planned.index('--vllm-url') + 1], 'http://127.0.0.1:18711/v1')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], SERVICES['render'][2])
            self.assertEqual(env['PYTHONPATH'], f'{OLD}/deps:{ROOT}/repo/src:{ROOT}/repo')
            self.assertEqual(argv[2], '8620')
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'old')

        def test_qwen_uses_only_gpu_zero(self):
            argv, env = isolated_plan('vllm', ['vllm', '--port', '8601'], {})
            self.assertEqual(argv[-1], '18711')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], SERVICES['vllm'][2])
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')

        def test_gpu_preflight_rejects_busy_or_changed_device(self):
            gpu = {'index': 0, 'uuid': SERVICES['vllm'][2], 'memory_mib': 0}
            validate_empty_gpu('vllm', gpu, [])
            for changed, processes in [({**gpu, 'memory_mib': 20}, []),
                                       ({**gpu, 'uuid': 'other'}, []),
                                       (gpu, [{'gpu_uuid': gpu['uuid'], 'pid': 123}])]:
                with self.subTest(gpu=changed, processes=processes), self.assertRaises(RuntimeError):
                    validate_empty_gpu('vllm', changed, processes)

        def test_stop_rejects_reused_pid_wrong_boot_owner_group_or_token(self):
            identity = {'pid': 123, 'pgid': 123, 'sid': 123, 'uid': 42,
                        'start_ticks': '456', 'boot_id': 'boot'}
            record = {'pid': 123, 'identity': identity, 'token': 'secret',
                      'root': str(ROOT), 'gpu_uuid': SERVICES['vllm'][2]}
            env = {'STREAMMUSE_DEPLOYMENT_TOKEN': 'secret',
                   'STREAMMUSE_DEPLOYMENT_ROOT': str(ROOT),
                   'CUDA_VISIBLE_DEVICES': SERVICES['vllm'][2]}
            self.assertTrue(can_stop(record, identity, env, 42))
            for key, value in [('start_ticks', '457'), ('boot_id', 'reboot'),
                               ('uid', 43), ('pgid', 999), ('sid', 999)]:
                with self.subTest(key=key):
                    self.assertFalse(can_stop(record, {**identity, key: value}, env, 42))
            self.assertFalse(can_stop(record, identity, {}, 42))
            self.assertFalse(can_stop(record, identity, env, 43))

        def test_port_preflight_rejects_existing_listener(self):
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                listener.listen()
                with self.assertRaises(OSError):
                    port_preflight(listener.getsockname()[1])

        def test_environmentless_worker_needs_verified_ancestry(self):
            identity = {'pid': 124, 'pgid': 123, 'sid': 123, 'uid': os.getuid(),
                        'start_ticks': '457', 'boot_id': 'boot'}
            record = {'pid': 123, 'identity': {**identity, 'pid': 123, 'start_ticks': '456'},
                      'root': str(ROOT), 'token': 'owned', 'gpu_uuid': SERVICES['vllm'][2]}
            self.assertTrue(member_owned(record, identity, {}, True))
            self.assertFalse(member_owned(record, identity, {}, False))
            self.assertFalse(member_owned(record, {**identity, 'sid': 999}, {}, True))
            self.assertFalse(member_owned(record, identity, {'STREAMMUSE_DEPLOYMENT_TOKEN': 'other'}, True))

        @unittest.skipUnless(hasattr(os, 'pidfd_open') and Path('/proc').exists(), 'Linux pidfds required')
        def test_real_owned_shutdown_and_identity_mismatch_refusal(self):
            global ROOT
            previous_root = ROOT
            with tempfile.TemporaryDirectory(prefix='rap-deploy-safety-') as directory:
                ROOT = Path(directory)
                environment = {**os.environ, 'STREAMMUSE_DEPLOYMENT_TOKEN': uuid.uuid4().hex,
                               'STREAMMUSE_DEPLOYMENT_ROOT': str(ROOT),
                               'CUDA_VISIBLE_DEVICES': SERVICES['vllm'][2]}
                code = ('import subprocess, sys, time; '
                        'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env={}); '
                        'print(p.pid, flush=True); time.sleep(60)')
                process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                           env=environment, start_new_session=True, text=True)
                try:
                    child_pid = int(process.stdout.readline())
                    record = {'pid': process.pid, 'identity': proc_identity(process.pid),
                              'root': str(ROOT), 'gpu_uuid': SERVICES['vllm'][2],
                              'token': environment['STREAMMUSE_DEPLOYMENT_TOKEN']}
                    tampered = {**record, 'identity': {**record['identity'], 'start_ticks': '0'}}
                    write_json(ROOT / 'probe.pid.json', tampered)
                    with self.assertRaises(RuntimeError):
                        stop('probe')
                    self.assertIsNone(process.poll(), 'Identity mismatch must not send a signal')
                    write_json(ROOT / 'probe.pid.json', record)
                    self.assertIn(child_pid, [member['pid'] for member in group_members(record)])
                    stopped = stop('probe')
                    self.assertEqual(stopped['pid'], process.pid)
                    self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)
                    self.assertFalse((ROOT / 'probe.pid.json').exists())
                    self.assertTrue(Path(stopped['archive']).is_file())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    process.stdout.close()
                    ROOT = previous_root

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SafetyTests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('action', nargs='?', choices=['preflight', 'start', 'stop', 'status', 'wait'])
    parser.add_argument('service', nargs='?', choices=['vllm', 'render', 'all'], default='all')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.action is None:
        parser.error('action or --self-test is required')
    if args.action in ('start', 'wait') and args.service == 'all':
        parser.error('Start/wait vllm first, then render')
    names = ['render', 'vllm'] if args.service == 'all' else [args.service]
    with control_lock():
        if args.action == 'preflight':
            result = preflight(names)
            write_json(ROOT / 'preflight.json', result)
        elif args.action == 'wait':
            result = wait_ready(args.service, args.timeout)
        else:
            action = {'start': start, 'stop': stop, 'status': status}[args.action]
            result = [action(name) for name in names]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Isolated, offline Lekai snapshots; importing this module needs only stdlib.

Real runs require mode='real', allow_gpu=True and an externally set
CUDA_VISIBLE_DEVICES of exactly '1', '2', or '3'. Check occupancy externally
before authorizing a run; this adapter neither checks nor selects GPUs.
Injected factories are exclusively for mode='mock' CPU tests and receive a
copy of the validated config. They must create a fresh, privately owned backend,
never return a live service. Timing is offline wall time, not realtime evidence.
The standard backend in this checkout has no CONTINUATION_PROMPT_MODE switch.
Only explicit time_signature_index=4 (4/4) is supported. Both observed melody
and accompaniment prefixes are half-open: boundary note_off is accepted but
boundary note_on is rejected. No event sorting or synthetic releases occur.
"""

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid


_ENV_LOCK = threading.Lock()
_REQUIRED = {
    'mode', 'checkpoint_path', 'device', 'dtype', 'bpm', 'context_beats',
    'temperature', 'top_p', 'top_k', 'repetition_penalty',
    'tonal_constraint_enabled', 'empty_token_guard_enabled',
    'time_signature_index',
}


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _number(value, name, minimum, maximum=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (positive and value == minimum) or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f'{name} is out of range')


def _validate_config(config):
    if type(config) is not dict or set(config) != _REQUIRED:
        raise ValueError(f'config must contain exactly {sorted(_REQUIRED)}')
    c = deepcopy(config)
    if c['mode'] not in ('real', 'mock'):
        raise ValueError('mode must explicitly be real or mock')
    if type(c['checkpoint_path']) is not str or not c['checkpoint_path'].strip():
        raise ValueError('checkpoint_path must be an explicit nonempty string')
    if c['device'] not in ('cpu', 'cuda') or (
        c['mode'] == 'real' and c['device'] != 'cuda'
    ):
        raise ValueError('real runs require device=cuda; mock accepts cpu or cuda')
    if c['dtype'] not in ('float16', 'float32'):
        raise ValueError('dtype must explicitly be float16 or float32')
    _integer(c['bpm'], 'bpm', 1)
    _integer(c['context_beats'], 'context_beats', 1)
    if type(c['time_signature_index']) is not int or c['time_signature_index'] != 4:
        raise ValueError('only explicit time_signature_index=4 (4/4) is supported')
    _integer(c['top_k'], 'top_k')
    _number(c['temperature'], 'temperature', 0)
    _number(c['top_p'], 'top_p', 0, 1)
    _number(c['repetition_penalty'], 'repetition_penalty', 0, positive=True)
    for key in ('tonal_constraint_enabled', 'empty_token_guard_enabled'):
        if type(c[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    return c


def _events(events, name, cutoff, *, prefix=False):
    if type(events) is not list:
        raise ValueError(f'{name} must be a list')
    result = []
    for event in events:
        if type(event) is not dict or not {'type', 'pitch', 'tick'} <= set(event):
            raise ValueError(f'{name} event requires type, pitch, tick')
        if set(event) - {'type', 'pitch', 'tick', 'velocity'}:
            raise ValueError(f'{name} event has unknown fields')
        if event['type'] not in ('note_on', 'note_off'):
            raise ValueError(f'{name} requires note_on or note_off')
        tick = _integer(event['tick'], f'{name}.tick')
        _integer(event['pitch'], f'{name}.pitch', 0, 127)
        if event['type'] == 'note_on' and 'velocity' not in event:
            raise ValueError('note_on requires explicit positive velocity')
        if 'velocity' in event:
            _integer(event['velocity'], f'{name}.velocity',
                     1 if event['type'] == 'note_on' else 0, 127)
        if tick > cutoff or (prefix and tick == cutoff and event['type'] == 'note_on'):
            raise ValueError(f'{name} event exceeds permitted cutoff')
        result.append(dict(event))
    return result


def _json_safe(value):
    # Strict round-trip also detaches caller-owned objects; never stringify tensors.
    return json.loads(json.dumps(value, allow_nan=False))


def _authorize(config, allow_gpu):
    if allow_gpu is not True:
        raise ValueError('real model construction requires explicit allow_gpu=True')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1', '2', '3'):
        raise ValueError("explicit CUDA_VISIBLE_DEVICES must be exactly '1', '2', or '3'")
    if config['device'] != 'cuda':
        raise ValueError('real runs require cuda')


def _verify_runtime(runtime, config, *, generated=False):
    if (runtime.get('mode') != 'real_model'
            or runtime.get('has_real_model') is not True
            or runtime.get('fallback_reason') not in (None, '')
            or runtime.get('resolved_device') not in ('cuda', 'cuda:0')
            or runtime.get('resolved_dtype') != config['dtype']):
        raise RuntimeError('real checkpoint runtime required; fallback/device/dtype mismatch')
    if Path(runtime.get('checkpoint_path') or '').resolve() != Path(config['checkpoint_path']).resolve():
        raise RuntimeError('runtime checkpoint mismatch')
    if generated and (
        runtime.get('runtime_model_name') != 'lekai'
        or runtime.get('runtime_inference_mode') != 'sliding_window'
        or runtime.get('pending_boundary_generations') != 0
    ):
        raise RuntimeError('not a completed standard Lekai snapshot')


def _real_backend_factory(config, *, allow_gpu):
    """The sole heavyweight import site, behind authorization and path checks."""
    _authorize(config, allow_gpu)
    if not Path(config['checkpoint_path']).is_file():
        raise ValueError('actual checkpoint_path must name an existing file')
    from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend

    class PrivateSnapshotBackend(LekaiHttpBackend):
        def __init__(self):
            self.input_token_calls = []
            super().__init__(checkpoint_path=config['checkpoint_path'])

        def _generate_rule_based(self, *args, **kwargs):
            raise RuntimeError('rule-based fallback is forbidden in snapshot probes')

        def _generate_part1_tokens_from_prompt(self, prompt_tokens, **kwargs):
            self.input_token_calls.append([int(v) for v in prompt_tokens.tolist()])
            return super()._generate_part1_tokens_from_prompt(prompt_tokens, **kwargs)

        def reset_session(self, seed):
            result = super().reset_session(seed)
            self.input_token_calls.clear()
            return result

        def consume_generation_metadata(self, request_id):
            result = super().consume_generation_metadata(request_id)
            result['input_tokens'] = deepcopy(self.input_token_calls)
            result['prompt_tokens'] = (list(self.input_token_calls[-1])
                                       if self.input_token_calls else [])
            return result

        def close(self):
            # Standard backend owns an executor but exposes no public close API.
            try:
                self._resolve_pending_boundary_generations(store=False)
            finally:
                self._boundary_executor.shutdown(wait=True, cancel_futures=True)

    backend = PrivateSnapshotBackend()
    try:
        _verify_runtime(backend.runtime_info(), config)
        adapter = backend._model_adapter
        if not (hasattr(adapter, 'model') and hasattr(adapter, 'tokenizer')
                and backend._tokenizer is not None):
            raise RuntimeError('standard interleaved token decoder is required')
    except BaseException:
        backend.close()
        raise
    return backend


class SnapshotProbe:
    """Owns one private backend. Every generate resets and injects a full snapshot."""

    def __init__(self, config: dict, *, allow_gpu=False, backend_factory=None):
        self._config = _validate_config(config)
        if type(allow_gpu) is not bool:
            raise ValueError('allow_gpu must be boolean')
        if self._config['mode'] == 'mock':
            if not callable(backend_factory):
                raise ValueError('mock mode requires an explicit test backend_factory')
        elif backend_factory is not None:
            raise ValueError('backend_factory is permitted only in explicit mock mode')
        self.backend_kind = 'fake' if backend_factory is not None else 'real'
        self._factory = backend_factory
        self._allow_gpu = allow_gpu
        self._backend = None
        self._closed = False
        self._call_lock = threading.Lock()
        if self.backend_kind == 'real':
            _authorize(self._config, self._allow_gpu)

    @contextmanager
    def _environment(self, *, seed=0):
        # The backend reads process env during decoding as well as construction.
        # Serialize cooperating probes and restore before returning to the caller.
        with _ENV_LOCK:
            c = self._config
            values = {
                'LEKAI_DEVICE': c['device'], 'LEKAI_DTYPE': c['dtype'],
                'LEKAI_DEFAULT_BPM': str(c['bpm']),
                'LEKAI_PROMPT_CONTEXT_BEATS': str(c['context_beats']),
                'LEKAI_HISTORY_MAX_TICKS': str(c['context_beats'] * 4),
                'LEKAI_RT_TEMPERATURE': str(c['temperature']),
                'LEKAI_RT_TOP_P': str(c['top_p']),
                'LEKAI_RT_TOP_K': str(c['top_k']),
                'LEKAI_RT_REPETITION_PENALTY': str(c['repetition_penalty']),
                'LEKAI_CONTINUATION_TONAL_CONSTRAINT': str(int(c['tonal_constraint_enabled'])),
                'LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD': str(int(c['empty_token_guard_enabled'])),
                'LEKAI_ENABLE_MPS_FALLBACK': '0',
                'LEKAI_DETERMINISTIC_BOUNDARY_GENERATION': '1',
                'LEKAI_TIME_SIGNATURE_INDEX': str(c['time_signature_index']),
                'LEKAI_MEASURE_BEATS': '4',
                'LEKAI_RT_SEED': str(seed), 'LEKAI_REQUIRE_SESSION': '1',
                'LEKAI_USE_CACHE': '1', 'LEKAI_WARMUP_STEPS': '1',
                'LEKAI_MAX_GENERATION_LENGTH_FRAMES': None,
                'LEKAI_MAX_PROMPT_TICKS': None,
            }
            previous = {key: os.environ.get(key) for key in values}
            try:
                for key, value in values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                yield dict(values)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def _ensure_backend(self):
        if self._closed:
            raise RuntimeError('probe is closed')
        if self.backend_kind == 'real':
            _authorize(self._config, self._allow_gpu)
        if self._backend is None:
            c = deepcopy(self._config)
            self._backend = (self._factory(c) if self._factory is not None
                             else _real_backend_factory(c, allow_gpu=self._allow_gpu))

    def __enter__(self):
        with self._call_lock, self._environment():
            self._ensure_backend()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            backend, self._backend = self._backend, None
            if backend is not None:
                with self._environment():
                    backend.close()

    def generate(self, *, melody_events: list, accompaniment_events: list,
                 target_start_step: int, length_steps: int,
                 observed_until: int, seed: int, constraint_snapshot=None) -> dict:
        start = _integer(target_start_step, 'target_start_step')
        length = _integer(length_steps, 'length_steps', 1)
        cutoff = _integer(observed_until, 'observed_until')
        _integer(seed, 'seed', 0, 2**63 - 1)
        if start % 4 or length % 4:
            raise ValueError('target_start_step and length_steps must be multiples of 4; no rounding')
        melody = _events(melody_events, 'melody_events', cutoff, prefix=True)
        accompaniment = _events(accompaniment_events, 'accompaniment_events', start, prefix=True)
        with self._call_lock, self._environment(seed=seed) as forced_environment:
            self._ensure_backend()
            backend = self._backend
            c = self._config
            started = time.perf_counter()
            session = backend.reset_session(seed=seed)
            backend.set_session_generation_config(**{
                key: c[key] for key in ('temperature', 'top_p', 'top_k', 'repetition_penalty')
            })
            injection = backend.inject_history(
                melody_events=deepcopy(melody), accompaniment_events=deepcopy(accompaniment),
                injection_length_ticks=start,
            )
            if session.get('success') is not True or injection.get('success') is not True:
                raise RuntimeError('snapshot reset/injection failed')
            if constraint_snapshot is not None:
                state = _json_safe(constraint_snapshot)
                if set(state) != {'empty_guard', 'tonal_tracker'}:
                    raise ValueError('invalid constraint snapshot')
                backend._empty_token_guard.restore(state['empty_guard'])
                if set(state['tonal_tracker']) != set(vars(backend._tonal_tracker)):
                    raise ValueError('invalid tonal tracker snapshot')
                tonal = deepcopy(state['tonal_tracker'])
                # JSON transport converts tuple keys; restore tracker comparisons.
                if tonal['_boundary'] is not None:
                    tonal['_boundary'] = tuple(tonal['_boundary'])
                if tonal['_pending'] is not None:
                    tonal['_pending'] = (tuple(tonal['_pending'][0]), tonal['_pending'][1])
                vars(backend._tonal_tracker).update(tonal)
            request_id = uuid.uuid4().hex
            events, timings = backend.generate(
                melody_events=[], generation_start_tick=start,
                generation_length_frames=length, generation_interval_ticks=length,
                prompt_length_ticks=c['context_beats'] * 4,
                inference_mode='sliding_window', model_name='lekai',
                checkpoint_path=c['checkpoint_path'], bpm=c['bpm'],
                session_id=session['session_id'], session_epoch=session['session_epoch'],
                request_id=request_id,
            )
            elapsed = time.perf_counter() - started
            metadata = backend.consume_generation_metadata(request_id)
            runtime = backend.runtime_info()
            if self.backend_kind == 'real':
                _verify_runtime(runtime, c, generated=True)
                if (metadata.get('part0_trace_available') is not True
                        or len(metadata.get('token_decode_beats', [])) != length // 4
                        or len(metadata.get('input_tokens', [])) != length // 4
                        or not metadata.get('raw_tokens')
                        or metadata.get('fallback_reason')
                        or metadata.get('fallback')):
                    raise RuntimeError('missing real token trace or fallback generation')
            runtime = {**runtime, 'backend_kind': self.backend_kind}
            metadata = {**metadata, 'backend_kind': self.backend_kind,
                        'forced_environment': forced_environment,
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'config': deepcopy(c), 'backend_timings': timings,
                        'session': session, 'injection': injection,
                        'restored_constraint_snapshot': deepcopy(constraint_snapshot),
                        'snapshot': {'melody_events': melody,
                                     'accompaniment_events': accompaniment,
                                     'target_start_step': start, 'length_steps': length,
                                     'observed_until': cutoff, 'seed': seed},
                        'timing_scope': 'offline reset/inject/generate; excludes model construction'}
            return _json_safe({'events': events, 'elapsed_seconds': elapsed,
                               'metadata': metadata, 'runtime': runtime})

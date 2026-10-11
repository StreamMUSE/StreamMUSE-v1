"""Isolated stdlib snapshot tests. No real backend/model/GPU is invoked."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

from experiments.bar_quality_history import model_probe as module


def config(**changes):
    value = dict(mode='mock', checkpoint_path='/not-loaded/checkpoint.pt',
                 device='cpu', dtype='float32', bpm=120, context_beats=16,
                 temperature=0.8, top_p=0.95, top_k=50, repetition_penalty=1.2,
                 tonal_constraint_enabled=False, empty_token_guard_enabled=True,
                 time_signature_index=4)
    value.update(changes)
    return value


def on(tick, pitch=60):
    return dict(type='note_on', pitch=pitch, tick=tick, velocity=100)


def off(tick, pitch=60):
    return dict(type='note_off', pitch=pitch, tick=tick, velocity=0)


def request(**changes):
    value = dict(melody_events=[on(0), off(8)],
                 accompaniment_events=[on(0, 48), off(4, 48)],
                 target_start_step=4, length_steps=4, observed_until=8, seed=42)
    value.update(changes)
    return value


class FakeBackend:
    def __init__(self, c):
        self.config = deepcopy(c)
        self.constructor_env = dict(os.environ)
        self.calls = []
        self.epoch = 0
        self.closed = False

    def reset_session(self, seed):
        self.calls.append(('reset', seed))
        self.epoch += 1
        self.seed = seed
        return dict(success=True, session_id='fake-session', session_epoch=self.epoch)

    def set_session_generation_config(self, **kwargs):
        self.calls.append(('sampling', kwargs))

    def inject_history(self, **kwargs):
        self.calls.append(('inject', deepcopy(kwargs)))
        return dict(success=True)

    def generate(self, **kwargs):
        self.calls.append(('generate', deepcopy(kwargs)))
        self.generation_env = dict(os.environ)
        self.last_request = kwargs
        return [on(kwargs['generation_start_tick'], 48),
                off(kwargs['generation_start_tick'] + 3, 48)], {'inference_start_time': 1.0}

    def consume_generation_metadata(self, request_id):
        assert request_id == self.last_request['request_id']
        start = self.last_request['generation_start_tick']
        count = self.last_request['generation_length_frames'] // 4
        return dict(raw_tokens=[169, 170], prompt_tokens=[257, 255, 172],
                    input_tokens=[[257, 255, 172]] * count,
                    part0_tokens=[169, 171], part0_trace_available=True,
                    token_decode_beats=[dict(start_tick=start + i * 4) for i in range(count)])

    def runtime_info(self):
        return dict(mode='fake', has_real_model=False, resolved_device='cpu')

    def close(self):
        self.closed = True


class ProbeTests(unittest.TestCase):
    def make_probe(self, c=None):
        self.backends = []

        def factory(cfg):
            backend = FakeBackend(cfg)
            self.backends.append(backend)
            return backend

        return module.SnapshotProbe(c or config(), backend_factory=factory)

    def test_snapshot_contract_reset_sampling_injection_and_order(self):
        melody = [on(2, 65), off(2, 62), on(0), off(8)]
        acc = [on(0, 48), off(4, 48), off(4, 55)]
        original = deepcopy((melody, acc))
        with self.make_probe() as probe:
            first = probe.generate(**request(melody_events=melody, accompaniment_events=acc))
            second = probe.generate(**request(seed=43))
            fake = self.backends[0]
            self.assertEqual([name for name, _ in fake.calls],
                             ['reset', 'sampling', 'inject', 'generate'] * 2)
            self.assertEqual(fake.calls[0], ('reset', 42))
            self.assertEqual(fake.calls[4], ('reset', 43))
            self.assertEqual(fake.calls[1][1], dict(temperature=.8, top_p=.95,
                                                   top_k=50, repetition_penalty=1.2))
            self.assertEqual(fake.calls[2][1], dict(melody_events=melody,
                             accompaniment_events=acc, injection_length_ticks=4))
            sent = fake.calls[3][1]
            self.assertEqual(sent['melody_events'], [])
            self.assertEqual(sent['model_name'], 'lekai')
            self.assertEqual(sent['inference_mode'], 'sliding_window')
            self.assertEqual(sent['generation_start_tick'], 4)
            self.assertEqual(sent['generation_length_frames'], 4)
            self.assertEqual(sent['generation_interval_ticks'], 4)
            self.assertEqual(sent['prompt_length_ticks'], 64)
            self.assertEqual(sent['bpm'], 120)
            self.assertEqual(sent['session_epoch'], 1)
            self.assertEqual(fake.calls[7][1]['session_epoch'], 2)
            self.assertNotEqual(sent['request_id'], fake.calls[7][1]['request_id'])
            self.assertEqual(first['runtime']['backend_kind'], 'fake')
            self.assertEqual(first['metadata']['snapshot']['melody_events'], melody)
            self.assertEqual(first['metadata']['input_tokens'], [[257, 255, 172]])
            self.assertEqual(first['metadata']['raw_tokens'], [169, 170])
            self.assertEqual(first['events'], [on(4, 48), off(7, 48)])
            self.assertGreaterEqual(first['elapsed_seconds'], 0)
            json.dumps(second, allow_nan=False)
        self.assertTrue(fake.closed)
        self.assertEqual((melody, acc), original)

    def test_retrospective_melody_and_boundary_off_are_accepted(self):
        with self.make_probe() as probe:
            result = probe.generate(**request(length_steps=16, observed_until=20,
                                               melody_events=[on(19), off(20)]))
        self.assertEqual(result['metadata']['snapshot']['observed_until'], 20)

    def test_no_rounding_and_strict_input_validation_before_construction(self):
        invalid = [dict(target_start_step=1), dict(target_start_step=-4),
                   dict(target_start_step=True), dict(length_steps=5),
                   dict(length_steps=0), dict(observed_until=1.0),
                   dict(seed=True), dict(seed=-1), dict(seed=2**63),
                   dict(melody_events=(on(0),)),
                   dict(melody_events=[on(9)]),
                   dict(melody_events=[on(8)]),
                   dict(melody_events=[on(-1)]),
                   dict(melody_events=[on(0.5)]),
                   dict(melody_events=[on(True)]),
                   dict(melody_events=[on(0, 128)]),
                   dict(melody_events=[on(0, -1)]),
                   dict(melody_events=[dict(on(0), velocity=0)]),
                   dict(melody_events=[dict(on(0), velocity=True)]),
                   dict(melody_events=[dict(off(0), velocity=128)]),
                   dict(melody_events=[dict(off(0), type='control_change')]),
                   dict(melody_events=[dict(off(0), channel=1)]),
                   dict(accompaniment_events=[on(4)]),
                   dict(accompaniment_events=[off(5)])]
        for change in invalid:
            with self.subTest(change=change):
                probe = self.make_probe()
                with self.assertRaises(ValueError):
                    probe.generate(**request(**change))
                self.assertEqual(self.backends, [])
                probe.close()

    def test_off_velocity_optional_on_velocity_explicit(self):
        event = off(4)
        del event['velocity']
        with self.make_probe() as probe:
            probe.generate(**request(accompaniment_events=[event]))
            with self.assertRaises(ValueError):
                probe.generate(**request(melody_events=[dict(type='note_on', tick=0, pitch=60)]))

    def test_all_config_fields_required_and_unknowns_rejected(self):
        for field in config():
            c = config()
            del c[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.SnapshotProbe(c, backend_factory=FakeBackend)
        with self.assertRaises(ValueError):
            module.SnapshotProbe(config(extra=True), backend_factory=FakeBackend)

    def test_config_type_and_range_validation(self):
        invalid = dict(mode=['auto', None], checkpoint_path=['', None, Path('x')],
                       device=['auto', 'cuda:1', 'mps'], dtype=['auto', 'bfloat16'],
                       bpm=[0, True, 120.5], context_beats=[0, 2.5, True],
                       temperature=[-1, float('nan'), float('inf'), '1', True],
                       top_p=[-1, 1.1, float('nan'), True], top_k=[-1, 1.5, True],
                       repetition_penalty=[0, -1, float('inf')],
                       tonal_constraint_enabled=[1, 'false'],
                       empty_token_guard_enabled=[0, None],
                       time_signature_index=[0, 3, 4.0, True, '4'])
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.make_probe(config(**{key: value}))

    def test_explicit_mock_factory_no_gpu_permission_or_checkpoint_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.make_probe() as probe:
                self.assertEqual(probe.generate(**request())['runtime']['backend_kind'], 'fake')
        with self.assertRaises(ValueError):
            module.SnapshotProbe(config())
        with self.assertRaises(ValueError):
            module.SnapshotProbe(config(mode='real', device='cuda'), backend_factory=FakeBackend)

    def test_gpu_authorization_rejected_before_factory(self):
        c = config(mode='real', device='cuda')
        with patch.object(module, '_real_backend_factory') as factory:
            for visible in ('', '0', '4', '1,2', ' 1', 'cuda:1', '-1'):
                with self.subTest(visible=visible), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible):
                    with self.assertRaises(ValueError):
                        module.SnapshotProbe(c, allow_gpu=True)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1'):
                with self.assertRaises(ValueError):
                    module.SnapshotProbe(c)
                with self.assertRaises(ValueError):
                    module.SnapshotProbe(config(mode='real'), allow_gpu=True)
            factory.assert_not_called()

    def test_authorization_rechecked_at_construction(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'):
            probe = module.SnapshotProbe(config(mode='real', device='cuda'), allow_gpu=True)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            with patch.object(module, '_real_backend_factory') as factory:
                with self.assertRaises(ValueError):
                    probe.__enter__()
                factory.assert_not_called()

    def test_environment_applied_before_factory_and_restored_each_operation(self):
        with patch.dict(os.environ, LEKAI_RT_TEMPERATURE='old',
                        LEKAI_MAX_PROMPT_TICKS='1', LEKAI_DEVICE='mps',
                        LEKAI_RT_SEED='999', LEKAI_PROMPT_CONTEXT_BEATS='999',
                        LEKAI_HISTORY_MAX_TICKS='999', LEKAI_TIME_SIGNATURE_INDEX='1'):
            before = dict(os.environ)
            with self.make_probe() as probe:
                self.assertEqual(dict(os.environ), before)
                fake = self.backends[0]
                self.assertEqual(fake.constructor_env['LEKAI_RT_TEMPERATURE'], '0.8')
                self.assertEqual(fake.constructor_env['LEKAI_DEVICE'], 'cpu')
                self.assertEqual(fake.constructor_env['LEKAI_ENABLE_MPS_FALLBACK'], '0')
                self.assertEqual(fake.constructor_env['LEKAI_CONTINUATION_TONAL_CONSTRAINT'], '0')
                self.assertEqual(fake.constructor_env['LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD'], '1')
                self.assertNotIn('LEKAI_MAX_PROMPT_TICKS', fake.constructor_env)
                result = probe.generate(**request())
                self.assertEqual(fake.generation_env['LEKAI_RT_TOP_K'], '50')
                self.assertEqual(fake.generation_env['LEKAI_RT_SEED'], '42')
                forced = result['metadata']['forced_environment']
                self.assertEqual(forced['LEKAI_TIME_SIGNATURE_INDEX'], '4')
                self.assertEqual(forced['LEKAI_PROMPT_CONTEXT_BEATS'], '16')
                self.assertEqual(forced['LEKAI_HISTORY_MAX_TICKS'], '64')
                self.assertEqual(forced['LEKAI_RT_SEED'], '42')
                self.assertIsNone(forced['LEKAI_MAX_PROMPT_TICKS'])
                self.assertEqual(dict(os.environ), before)
            self.assertEqual(dict(os.environ), before)

    def test_environment_restored_on_constructor_generation_and_close_failure(self):
        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, 'factory'):
            with module.SnapshotProbe(config(), backend_factory=Mock(side_effect=RuntimeError('factory'))):
                pass
        self.assertEqual(dict(os.environ), before)
        with self.assertRaisesRegex(RuntimeError, 'generate'):
            with self.make_probe() as probe:
                self.backends[0].generate = Mock(side_effect=RuntimeError('generate'))
                probe.generate(**request())
        self.assertTrue(self.backends[0].closed)
        self.assertEqual(dict(os.environ), before)
        probe = self.make_probe().__enter__()
        self.backends[0].close = Mock(side_effect=RuntimeError('close'))
        with self.assertRaisesRegex(RuntimeError, 'close'):
            probe.close()
        self.assertEqual(dict(os.environ), before)

    def test_elapsed_is_measured_not_backend_timing(self):
        with self.make_probe() as probe, patch.object(module.time, 'perf_counter', side_effect=[10., 12.5]):
            self.assertEqual(probe.generate(**request())['elapsed_seconds'], 2.5)

    def test_close_is_idempotent_and_final(self):
        probe = self.make_probe().__enter__()
        backend = self.backends[0]
        backend.close = Mock()
        probe.close()
        probe.close()
        backend.close.assert_called_once()
        with self.assertRaises(RuntimeError):
            probe.generate(**request())

    def test_real_runtime_and_fallback_trace_checks_using_fake_factory_patch(self):
        c = config(mode='real', device='cuda')
        good = dict(mode='real_model', has_real_model=True, resolved_device='cuda',
                    resolved_dtype='float32', checkpoint_path=c['checkpoint_path'],
                    fallback_reason=None, runtime_model_name='lekai',
                    runtime_inference_mode='sliding_window', pending_boundary_generations=0)
        bad_runtime = [dict(mode='rule_stub'), dict(has_real_model=False),
                       dict(resolved_device='cpu'), dict(resolved_dtype='float16'),
                       dict(fallback_reason='load failed'), dict(checkpoint_path='/other'),
                       dict(pending_boundary_generations=1), dict(runtime_model_name='other')]
        for change in bad_runtime + [None]:
            fake = FakeBackend(c)
            fake.runtime_info = Mock(return_value={**good, **(change or {})})
            if change is None:
                fake.consume_generation_metadata = Mock(return_value={'fallback': True})
            with self.subTest(change=change), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='3'):
                with patch.object(module, '_real_backend_factory', return_value=fake):
                    with module.SnapshotProbe(c, allow_gpu=True) as probe:
                        with self.assertRaises(RuntimeError):
                            probe.generate(**request())

    def test_real_factory_authorization_path_and_private_trace_with_stub_module(self):
        # Substitute the entire heavy module. This exercises private subclassing
        # without importing production code or constructing a model.
        class BackendStub(FakeBackend):
            def __init__(self, checkpoint_path):
                super().__init__(config(mode='real', device='cuda', checkpoint_path=checkpoint_path))
                self._model_adapter = types.SimpleNamespace(model=object(), tokenizer=object())
                self._tokenizer = object()
                self._boundary_executor = Mock()

            def runtime_info(self):
                return dict(mode='real_model', has_real_model=True, resolved_device='cuda',
                            resolved_dtype='float32', checkpoint_path=self.config['checkpoint_path'])

            def _generate_part1_tokens_from_prompt(self, prompt_tokens, **kwargs):
                return [169, 170]

            def _resolve_pending_boundary_generations(self, **kwargs):
                pass

        stub = types.ModuleType('streammuse.infrastructure.inference.lekai_http_backend')
        stub.LekaiHttpBackend = BackendStub
        c = config(mode='real', device='cuda')
        with patch.dict(sys.modules, {stub.__name__: stub}), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1'):
            with patch.object(Path, 'is_file', return_value=False):
                with self.assertRaises(ValueError):
                    module._real_backend_factory(c, allow_gpu=True)
            with patch.object(Path, 'is_file', return_value=True):
                backend = module._real_backend_factory(c, allow_gpu=True)
            with self.assertRaises(RuntimeError):
                backend._generate_rule_based()
            tokens = types.SimpleNamespace(tolist=lambda: [257, 255, 172])
            backend._generate_part1_tokens_from_prompt(tokens, temperature=.8)
            backend.last_request = dict(request_id='r', generation_start_tick=4, generation_length_frames=4)
            trace = backend.consume_generation_metadata('r')
            self.assertEqual(trace['input_tokens'], [[257, 255, 172]])
            self.assertEqual(trace['prompt_tokens'], [257, 255, 172])
            backend.reset_session(42)
            self.assertEqual(backend.input_token_calls, [])
            backend.close()
            backend._boundary_executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_import_does_not_import_torch_or_backend(self):
        root = Path(__file__).resolve().parents[3]
        code = (
            'import sys, importlib.abc\n'
            'class Guard(importlib.abc.MetaPathFinder):\n'
            ' def find_spec(self, fullname, path=None, target=None):\n'
            '  if fullname.split(".")[0] in ("torch", "numpy", "streammuse"):\n'
            '   raise AssertionError("heavy import: " + fullname)\n'
            'sys.meta_path.insert(0, Guard())\n'
            'from experiments.bar_quality_history.model_probe import SnapshotProbe\n'
            'assert "torch" not in sys.modules\n'
        )
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=root,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

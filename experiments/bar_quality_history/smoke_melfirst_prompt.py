"""Prompt-layout smoke for the LEKAI_MEL_FIRST backend flag.

Mechanics check only (generation quality needs the mel_first-trained
checkpoint): drives the snapshot backend over a synthetic 36-beat melody,
generating beats [32, 36), then asserts the per-beat token order in the
built prompts (metadata input_tokens, one full prompt per generated beat).

  acc-first (default): [BEAT] <acc> ACC_END <mel> MEL_END
  mel-first          : [BEAT] <mel> MEL_END <acc> ACC_END
  mel-first tail     : final section is [BEAT] <mel_target> MEL_END

Also asserts the flag only REORDERS context beats: with identical history
and seed, flag-off and flag-on context-beat sections hold the same token
multiset, and the mel-first tail adds exactly the target-beat melody.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LEKAI_BEAT = 172
LEKAI_ACC_END = 170
LEKAI_MEL_END = 171


def synthetic_events():
    melody = []
    for beat in range(36):
        tick = beat * 4
        pitch = 60 + (beat % 8)
        melody.append({'type': 'note_on', 'pitch': pitch, 'tick': tick, 'velocity': 100})
        melody.append({'type': 'note_off', 'pitch': pitch, 'tick': tick + 4})
    accompaniment = []
    for beat in range(32):
        tick = beat * 4
        for pitch in (48, 55, 64):
            accompaniment.append({'type': 'note_on', 'pitch': pitch, 'tick': tick, 'velocity': 80})
            accompaniment.append({'type': 'note_off', 'pitch': pitch, 'tick': tick + 4})
    return melody, accompaniment


def sections(tokens):
    """Prompt tokens split at [BEAT] markers; pre-first-BEAT tokens dropped.

    A trailing empty section (acc-first tail after the final [BEAT]) is
    kept so both layouts yield one section per generated context beat.
    """
    out, current = [], None
    for token in tokens:
        if token == LEKAI_BEAT:
            if current is not None:
                out.append(current)
            current = []
        elif current is not None:
            current.append(token)
    if current is not None:
        out.append(current)
    return out


def first_end_marker(section):
    for token in section:
        if token in (LEKAI_ACC_END, LEKAI_MEL_END):
            return token
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', choices=['1', '2', '3'], required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['LEKAI_RT_LOG_DIR'] = 'logs/melfirst_prompt_smoke'

    from experiments.bar_quality_history.model_probe import SnapshotProbe

    config = {'mode': 'real', 'checkpoint_path': args.checkpoint, 'device': 'cuda',
              'dtype': 'float16', 'bpm': 120, 'context_beats': 32,
              'temperature': 1.05, 'top_p': 0.98, 'top_k': 0, 'repetition_penalty': 1.0,
              'time_signature_index': 4, 'tonal_constraint_enabled': True,
              'empty_token_guard_enabled': True}
    melody, accompaniment = synthetic_events()

    prompts = {}
    results = {}
    probe = SnapshotProbe(config, allow_gpu=True)
    try:
        probe.__enter__()
        for flag in ('0', '1'):
            os.environ['LEKAI_MEL_FIRST'] = flag
            result = probe.generate(
                melody_events=melody, accompaniment_events=accompaniment,
                target_start_step=128, length_steps=16, observed_until=144, seed=0)
            prompts[flag] = result['metadata']['input_tokens']
            results[flag] = result
    finally:
        probe.close()
    os.environ.pop('LEKAI_MEL_FIRST', None)

    forced = results['0']['metadata'].get('forced_environment') or {}
    print('forced LEKAI_PROMPT_CONTEXT_BEATS =', forced.get('LEKAI_PROMPT_CONTEXT_BEATS'),
          '| input_tokens lens =', [len(p) for p in prompts['0']],
          '| section counts =', [len(sections(p)) for p in prompts['0']])

    off, on = prompts['0'], prompts['1']
    assert len(off) == len(on) == 4, (len(off), len(on))

    for beat_index, tokens in enumerate(off):
        secs = sections(tokens)
        assert len(secs) == 33 + beat_index, (beat_index, len(secs))
        for section in secs[:32 + beat_index]:
            assert first_end_marker(section) == LEKAI_ACC_END, 'acc-first context violated'
        assert not secs[32 + beat_index], 'acc-first tail must be empty'
    print('PASS acc-first layout: context sections [acc|mel], empty tail')

    for beat_index, tokens in enumerate(on):
        secs = sections(tokens)
        assert len(secs) == 33 + beat_index, (beat_index, len(secs))
        for section in secs[:32 + beat_index]:
            assert first_end_marker(section) == LEKAI_MEL_END, 'mel-first context violated'
            assert section.index(LEKAI_MEL_END) < section.index(LEKAI_ACC_END), \
                'mel must precede acc inside beat section'
        tail = secs[32 + beat_index]
        assert first_end_marker(tail) == LEKAI_MEL_END and LEKAI_ACC_END not in tail, \
            'mel-first tail must be target melody only'
    print('PASS mel-first layout: context sections [mel|acc], tail = target melody')

    off_secs, on_secs = sections(off[0]), sections(on[0])
    for j in range(32):
        assert sorted(off_secs[j]) == sorted(on_secs[j]), f'context beat {j} changed beyond reorder'
    assert not off_secs[32] and on_secs[32], 'tail difference must be exactly the target melody'
    print('PASS reorder-only: first prompt identical as multiset per beat, tail adds target melody')
    print('SMOKE_OK')


if __name__ == '__main__':
    main()

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import run_formal_user10_test40 as formal


@pytest.mark.parametrize('user', [False, True])
@pytest.mark.parametrize('pc', [False, True])
def test_explicit_modern_contract(user, pc, tmp_path):
    cohort = {'playback_bpm': 90 if user else 120, 'model_bpm': 80 if user else 120,
              'sampling': {'temperature': 1.1 if user else 1.05, 'top_p': .95 if user else .98,
                           'top_k': 50 if user else 0, 'repetition_penalty': 1.0},
              'count_in_beats_modern': 4 if user else 0}
    condition = 'pc_rule_if_else_n10' if pc else 'lekai_no_prompt'
    c = formal.contract_for(cohort, condition, 444 if user else 128)
    c.validate()
    system = formal.IDS[condition]
    env = formal.matched.build_server_environment(system, port=18000, gpu='0', server_dir=tmp_path,
        code={'git_commit': 'abc'}, prompt_checkpoint={'path': 'prompt.safetensors'},
        continuation_checkpoint={'path': 'continuation.safetensors'}, time_signature_index=4,
        prompt_selection_mode='rule_s_if_else' if pc else 'single', prompt_batch_candidates=10 if pc else 1,
        contract=c)
    assert env['LEKAI_TIME_SIGNATURE_INDEX'] == '4'
    assert env['LEKAI_DETERMINISTIC_BOUNDARY_GENERATION'] == '0'
    assert env['LEKAI_CONTINUATION_TONAL_CONSTRAINT'] == env['LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD'] == str(int(pc))
    assert float(env['LEKAI_RT_TEMPERATURE']) == cohort['sampling']['temperature']
    if not pc:
        assert 'LEKAI_PROMPT_CHECKPOINT_PATH' not in env
    else:
        assert env['LEKAI_PROMPT_BATCH_CANDIDATES'] == '10'
    command = formal.matched.build_cli_command(system, python_bin='python', midi_path=Path('input.mid'),
        log_root=tmp_path, base_url='http://127.0.0.1:18000', contract=c,
        prompt_selection_mode='rule_s_if_else' if pc else 'single', prompt_batch_candidates=10 if pc else 1)
    value = lambda flag: command[command.index(flag) + 1]
    assert value('--input-mode') == 'midi_file'
    assert value('--count-in-beats') == str(cohort['count_in_beats_modern'])
    assert value('--continuation-mode') == ('prompt_continuation' if pc else 'standard')
    assert value('--temperature') == str(cohort['sampling']['temperature'])
    assert '--midi-file-source-tick-mode' not in command


def test_batch_trial_count():
    assert (10 + 40) * 3 * len(formal.CONDITIONS) == 450


def test_reject_untrimmed_input(tmp_path):
    import mido
    path = tmp_path / 'm.mid'
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.Message('note_on', note=60, time=480),
        mido.Message('note_off', note=60, time=480)]))
    midi.save(path)
    with pytest.raises(AssertionError, match='leading Melody silence'):
        formal.midi_geometry(path)

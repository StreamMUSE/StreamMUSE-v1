from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streammuse.infrastructure.inference.tonal_constraint import (
    melody_evidence, TwoBarKeyTracker, TonalBeatMask, infer_key_with_nan_fallback,
)
from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import PianoMusicTokenizer
from streammuse.infrastructure.inference.lekai_model.generation_utils import sample_token
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend


def chord(pitches, start=0, end=32):
    return ([dict(type="note_on",pitch=p,tick=start) for p in pitches]
            + [dict(type="note_off",pitch=p,tick=end) for p in pitches])


def test_duration_evidence_causal_sustain_and_velocity_zero():
    events=chord([60],0,50)+[dict(type="note_on",pitch=60,tick=18,velocity=0)]
    evidence=melody_evidence(events,16,32)
    assert evidence[0]==0 and evidence.sum()==0  # Capped note already expired at tick 16.
    assert melody_evidence(chord([60],0,50),8,32)[0]==8
    assert melody_evidence(chord([61],32,50),0,32).sum()==0


def test_same_tick_on_off_does_not_become_a_ghost_sustain():
    assert melody_evidence(chord([73],256,256),0,508).sum()==0
    # Conversely, an observed OFF then ON is a retrigger and must keep sounding.
    retrigger = [dict(type='note_off',pitch=73,tick=256),
                 dict(type='note_on',pitch=73,tick=256)]
    assert melody_evidence(retrigger,0,284)[1]==16


def test_four_beat_cap_is_per_onset_not_per_estimation_window():
    held = [dict(type='note_on',pitch=60,tick=0)]
    assert melody_evidence(held,0,8)[0]==8
    assert melody_evidence(held,0,16)[0]==16
    assert melody_evidence(held,0,1000)[0]==16
    assert melody_evidence(held,16,1000)[0]==0
    # Each retrigger is a new note; cap does not limit total pitch-class evidence.
    held.append(dict(type='note_on',pitch=60,tick=32))
    assert melody_evidence(held,0,1000)[0]==32
    assert melody_evidence(chord([60],0,5),0,1000)[0]==5


def test_preestimate_on_last_beat_apply_only_at_next_boundary():
    tracker=TwoBarKeyTracker()
    # Short C evidence followed by stronger (but capped) Db evidence.
    events=chord([60,64,67],0,4)+chord([61,65,68],32,64)
    c=tracker.update(events,32,4)
    assert c['key']['tonic_pitch_class']==0
    assert tracker.update(events,60,4)['key']==c['key']
    # New events arriving in the final beat cannot retroactively affect the frozen estimate.
    events+=chord([60,64,67],60,100)
    db=tracker.update(events,64,4)
    assert db['key']['tonic_pitch_class']==1
    assert db['estimated_at_tick']==60 and db['window_end_tick']==60
    assert db['window_start_tick']==0 and db['boundary_tick']==64
    assert tracker.update([],68,4)==db


def test_missing_evidence_or_reset_leaves_key_unconstrained():
    t=TwoBarKeyTracker()
    c=t.update(chord([60,64,67]),32,4)
    t.update([],60,4)
    assert t.update([],64,4)['key'] is None
    assert t.update([],64,4)['reason']=='insufficient_evidence_unconstrained'
    t.reset()
    assert t.update(chord([61]),32,4)['allowed_pitch_classes'] is None


@pytest.mark.parametrize('pcs', [[], [0], [0, 2]])
def test_only_fewer_than_three_pitch_classes_get_nan_fallback(pcs):
    evidence = np.zeros(12)
    evidence[pcs] = 16
    decision = infer_key_with_nan_fallback(evidence)
    assert decision['key'] is None
    assert decision['key_state'] == 'NaN'
    assert not decision['constraint_active']


@pytest.mark.parametrize('pcs', [[0, 1, 2], list(range(12)), [0, 4, 7, 9]])
def test_three_or_more_pitch_classes_use_original_winner_without_score_thresholds(pcs):
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import infer_tonal_key
    evidence = np.zeros(12)
    evidence[pcs] = 16
    decision = infer_key_with_nan_fallback(evidence)
    assert decision['key'] == infer_tonal_key(evidence)
    assert decision['constraint_active']
    assert decision['key_state'] == 'determined'
    assert 'minimum_key_confidence' not in decision
    assert 'minimum_key_score_margin' not in decision


def test_nan_then_enable_at_boundary_without_releasing_for_close_scores():
    tracker = TwoBarKeyTracker()
    events = chord([60, 62], 0, 1)
    assert tracker.update(events, 32, 4)['key'] is None
    events += chord([60, 64, 67], 32, 48)
    assert tracker.update(events, 60, 4)['key'] is None
    assert tracker.update(events, 64, 4)['key']['tonic_pitch_class'] == 0
    # Close template scores must NOT disable the mask once enough PCs exist.
    events += chord(list(range(60, 72)), 64, 80)
    assert tracker.update(events, 92, 4)['constraint_active']
    assert tracker.update(events, 96, 4)['constraint_active']


def test_unknown_decision_is_standard_json_not_float_nan():
    import json
    decision = TwoBarKeyTracker().update([], 32, 4)
    encoded = json.dumps(decision, allow_nan=False)
    assert json.loads(encoded)['key'] is None


def test_cumulative_key_keeps_early_evidence_instead_of_only_recent_bars():
    tracker = TwoBarKeyTracker()
    events = [event for start in range(0,128,16)
              for event in chord([60,64,67],start,start+8)] + chord([61,65,68],128,160)
    tracker.update(events,128,4)
    tracker.update(events,156,4)
    decision = tracker.update(events,160,4)
    evidence = decision['pitch_class_duration_evidence']
    assert evidence[0] == evidence[4] == evidence[7] == 64
    assert evidence[1] == evidence[5] == evidence[8] == 16
    assert decision['key']['tonic_pitch_class'] == 0
    assert decision['evidence_scope'] == 'cumulative_from_session_start'
    assert decision['window_start_tick'] == 0


def test_full_history_and_current_request_used_once_despite_context_trim(monkeypatch):
    backend = LekaiHttpBackend()
    original = chord([60,64],0,8)
    increment = chord([67],180,184)
    backend.inject_history(original,[],32)
    backend._melody_history = []  # Simulate old notes leaving the model context.
    def generate(**kwargs):
        backend._tonal_tracker.update(backend._tonal_input_snapshot,188,4)
        return [], {}
    monkeypatch.setattr(backend,'_generate_locked',generate)
    backend.generate(melody_events=increment,generation_start_tick=188,
                     generation_length_frames=4,generation_interval_ticks=2,
                     prompt_length_ticks=32,inference_mode='sliding_window',
                     model_name='lekai',checkpoint_path=None)
    decision = backend._tonal_tracker.update(backend._input_digest_history,192,4)
    evidence = decision['pitch_class_duration_evidence']
    assert evidence[0] == evidence[4] == 8
    assert evidence[7] == 4 and sum(evidence) == 20
    assert decision['key'] is not None
    assert backend._input_digest_history == original + increment
    assert backend._tonal_input_snapshot is None


def test_failed_generation_clears_snapshot_without_committing_input(monkeypatch):
    backend = LekaiHttpBackend()
    original = chord([60,64,67],0,32)
    backend.inject_history(original,[],32)
    def fail(**kwargs):
        raise RuntimeError('test failure')
    monkeypatch.setattr(backend,'_generate_locked',fail)
    with pytest.raises(RuntimeError,match='test failure'):
        backend.generate(melody_events=chord([61],40,44),generation_start_tick=44,
                         generation_length_frames=4,generation_interval_ticks=2,
                         prompt_length_ticks=32,inference_mode='sliding_window',
                         model_name='lekai',checkpoint_path=None)
    assert backend._tonal_input_snapshot is None
    assert backend._input_digest_history == original


@pytest.mark.parametrize('beats_per_bar',[2,3,4,6])
def test_update_period_uses_two_bars(beats_per_bar):
    t=TwoBarKeyTracker();span=beats_per_bar*8
    t.update(chord([60,64,67],0,span),span,beats_per_bar)
    t.update(chord([61,65,68],span,2*span),2*span-4,beats_per_bar)
    assert t.update([],2*span,beats_per_bar)['estimated_at_tick']==2*span-4


@pytest.mark.parametrize('tonic',range(12))
@pytest.mark.parametrize('degrees',[(0,2,4,5,7,9,11),(0,2,3,5,7,8,10)])
@pytest.mark.parametrize('temperature',[0.0,1.1])
def test_all_keys_sampling_and_decoding_stays_in_key(tonic,degrees,temperature):
    allowed={(tonic+x)%12 for x in degrees}
    state=TonalBeatMask(allowed)
    generator=torch.Generator().manual_seed(57)
    tokens=[]
    for step in range(100):
        logits=torch.randn(1,300,generator=generator)
        logits[:,169:]= -30 # Adversarial preference for dense notes, not an early exit.
        logits=state.apply(logits,step)
        value=int(sample_token(logits,temperature=temperature,top_k=50,top_p=.95,generator=generator).item())
        state.accept(value);tokens.append(value)
        if value==170:break
    assert tokens[-1]==170 and len(tokens)<=100
    roll=PianoMusicTokenizer().decode_beats_to_pianoroll([tokens],track_marker_id=170)
    assert all((i+21)%12 in allowed for i in np.flatnonzero(roll.any(axis=(0,2))))


def test_relative_mask_changes_with_position_and_keeps_zero_initial_offset():
    m=TonalBeatMask({9,0})
    assert 81 in m.legal_ids(0) # MIDI21=A, first distance zero is legal.
    m.accept(81);m.accept(67)
    assert 81 not in m.legal_ids(2) # A duplicate position is not legal.
    m.accept(84);m.accept(67) # MIDI24=C
    assert 87 not in m.legal_ids(4) # +6 => MIDI30=F#, forbidden.
    assert 90 in m.legal_ids(4) # +9 => MIDI33=A, allowed.


def test_empty_and_invalid_logit_fallback_is_legal():
    state=TonalBeatMask(set())
    logits=state.apply(torch.full((1,300),float('nan')),0)
    assert logits.argmax().item()==169
    state.accept(169)
    assert state.apply(torch.zeros(1,300),1).argmax().item()==170
    state=TonalBeatMask({0});state.accept(120)
    assert state.apply(torch.full((1,300),-float('inf')),1).argmax().item()==67


def test_backend_mask_closes_old_out_of_key_sustain_and_records_decision(monkeypatch):
    monkeypatch.setenv('LEKAI_CONTINUATION_TONAL_CONSTRAINT','1')
    monkeypatch.setenv('LEKAI_TIME_SIGNATURE_INDEX','0')
    backend=LekaiHttpBackend()
    class Model:
        def __call__(self,input_ids,**kwargs):
            logits=torch.zeros(1,input_ids.shape[1],300)
            logits[:,:,121]=20 # Prefer a relative token that initially means C#.
            logits[:,:,120]=19
            logits[:,:,67]=18
            logits[:,:,170]=1
            return SimpleNamespace(logits=logits,past_key_values=None)
    backend._tokenizer=PianoMusicTokenizer()
    backend._converter=MidiConverter(ticks_per_beat=4)
    backend._model_adapter=SimpleNamespace(model=Model(),device='cpu',use_cache=False)
    monkeypatch.setattr(backend._logger,'log_generation',lambda **kwargs:None)
    backend.inject_history(chord([60,64,67],0,32),[dict(type='note_on',pitch=61,tick=28)],32)
    backend.set_session_generation_config(temperature=0)
    events=backend._generate_with_interleaved_prompt(32,2,4)
    d=backend._current_generation_trace['tonal_decisions'][0]
    assert d['key']['tonic_pitch_class']==0
    assert any(e['type']=='note_off' and e['pitch']==61 and e['tick']==32 for e in events)
    assert all(e['pitch']%12 in d['allowed_pitch_classes'] for e in events if e['type']=='note_on')
    backend.clear_history()
    assert backend._tonal_tracker._decision is None

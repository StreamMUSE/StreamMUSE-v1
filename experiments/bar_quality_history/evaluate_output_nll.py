"""Teacher-forced BEAT ACC-token NLL of exported MIDI, not repaired hidden history.

This is not the legacy RoFormer NLL evaluator. It uses the current continuation
checkpoint, its ACC->MEL tokenizer and 32-beat causal context. No sampling occurs.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def midi_tracks(path):
    import mido
    result = {}
    midi = mido.MidiFile(path)
    for track in midi.tracks:
        name = track.name
        if name not in ('Melody', 'Accompaniment'):
            if any(e.type in ('note_on', 'note_off') for e in track):
                raise ValueError('Unrecognized note track')
            continue
        if name in result:
            raise ValueError('Duplicate track name')
        result[name] = []
        tick = 0
        ppq = midi.ticks_per_beat
        for event in track:
            tick += event.time
            if event.type not in ('note_on', 'note_off'):
                continue
            if tick * 4 % ppq:
                raise ValueError('MIDI is not on the exact four-step grid')
            if not 21 <= event.note <= 108:
                raise ValueError('Out-of-vocabulary pitch; do not silently discard')
            result[name].append({'tick': tick * 4 // ppq, 'pitch': event.note,
                'type': 'note_off' if event.type == 'note_off' or event.velocity == 0 else 'note_on',
                'velocity': event.velocity})
    if set(result) != {'Melody', 'Accompaniment'}:
        raise ValueError('Both named tracks required, including an empty accompaniment track')
    return result


def make_encoder(tokenizer):
    # Borrow only the backend's pure event encoding functions; no backend instance,
    # executor, model warmup, sampling, constraint tracking or repair state exists.
    from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
    from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter

    class Encoder:
        _event_sort_key = staticmethod(LekaiHttpBackend._event_sort_key)
        _advance_active_pitches = LekaiHttpBackend._advance_active_pitches
        _encode_beat_tokens = LekaiHttpBackend._encode_beat_tokens
        _continuation_initial_tokens = LekaiHttpBackend._continuation_initial_tokens

    encoder = Encoder()
    encoder._converter = MidiConverter(ticks_per_beat=4)
    encoder._tokenizer = tokenizer
    return encoder


def encode_tracks(encoder, events, stop):
    result = {}
    for name, marker in (('Melody', 171), ('Accompaniment', 170)):
        active, beats = set(), []
        for tick in range(0, stop, 4):
            tokens, active = encoder._encode_beat_tokens(events[name], tick, active, marker)
            beats.append(tokens.tolist())
        result[name] = beats
    return result


def prompt_for_target(initial, acc, mel, target, context_beats=32):
    start = max(0, target - context_beats)
    tokens = list(initial)
    for beat in range(start, target):
        if beat == start or beat % 4 == 0:
            tokens.append(255)
        tokens += [172] + acc[beat] + mel[beat]
    if target == start or target % 4 == 0:
        tokens.append(255)
    tokens.append(172)
    return tokens


def make_batch(items, device, pad=258):
    import torch
    length = max(len(i['prompt']) + len(i['target']) for i in items)
    ids = torch.full((len(items), length), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for row, item in enumerate(items):
        p, t = item['prompt'], item['target']
        ids[row, :len(p) + len(t)] = torch.tensor(p + t, device=device)
        mask[row, :len(p) + len(t)] = 1
        labels[row, len(p):len(p) + len(t)] = torch.tensor(t, device=device)
    return ids, mask, labels


def token_losses(logits, labels):
    import torch.nn.functional as F
    # The logit before each target token predicts it. Prompt and padding labels
    # are masked, including Melody, BOS, metadata, BAR and BEAT markers.
    return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:],
                           ignore_index=-100, reduction='none')


def score_batch(model, items):
    import torch
    ids, mask, labels = make_batch(items, 'cuda')
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        losses = token_losses(output.logits, labels)
        result = []
        for row, item in enumerate(items):
            begin = len(item['prompt']) - 1
            values = losses[row, begin:begin + len(item['target'])].cpu().tolist()
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('Non-finite NLL')
            result.append(values)
    return result


def validate_forward(model, batch, scores):
    import torch
    import torch.nn.functional as F
    item = batch[0]
    ids, mask, labels = make_batch([item], 'cuda')
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        single = token_losses(output.logits, labels)[0, len(item['prompt']) - 1:
                                                    len(item['prompt']) + len(item['target']) - 1]
        hf_loss = float(output.loss)
        assert abs(float(single.mean()) - hf_loss) < 1e-5
        sequential = []
        for index in range(min(8, len(item['target']))):
            prefix = item['prompt'] + item['target'][:index]
            logits = model(input_ids=torch.tensor([prefix], device='cuda'), use_cache=False).logits[0, -1]
            sequential.append(float(F.cross_entropy(logits.float().unsqueeze(0),
                                  torch.tensor([item['target'][index]], device='cuda'))))
    batch_error = max(abs(x - y) for x, y in zip(single.tolist(), scores[0]))
    step_error = max(abs(x - y) for x, y in zip(single.tolist(), sequential))
    if batch_error > .02 or step_error > .02:
        raise AssertionError('FP16 batch/stepwise scoring validation exceeded 0.02 nats/token')
    return {'hf_loss': hf_loss, 'direct_token_mean': float(single.mean()),
            'batch_vs_single_max_abs_nats': batch_error, 'stepwise_max_abs_nats': step_error,
            'stepwise_tokens': len(sequential), 'numerical_tolerance_nats': .02}


def aggregate(rows):
    trials = []
    groups = defaultdict(list)
    for r in rows:
        groups[r['piece_id'], r['title'], r['seed'], r['condition']].append(r)
    for (piece, title, seed, condition), values in groups.items():
        for window, start in (('primary_after_repair', 84), ('secondary_postjoin', 32)):
            selected = [r for r in values if r['tick'] >= start]
            count = sum(r['token_count'] for r in selected)
            total = math.fsum(r['nll_sum'] for r in selected)
            trials.append({'piece_id': piece, 'title': title, 'seed': seed, 'condition': condition,
                           'window': window, 'beats': len(selected), 'tokens': count,
                           'nll_sum': total, 'NLL': total / count, 'PPL': math.exp(total / count)})
    summaries, paired = [], []
    for window in ('primary_after_repair', 'secondary_postjoin'):
        for condition in ('baseline', 'repair'):
            selected = [r for r in trials if r['window'] == window and r['condition'] == condition]
            total, count = math.fsum(r['nll_sum'] for r in selected), sum(r['tokens'] for r in selected)
            piece_values = defaultdict(list)
            for r in selected:
                piece_values[r['piece_id']].append(r['NLL'])
            summaries.append({'window': window, 'condition': condition, 'trials': len(selected),
                'scored_tokens': count, 'nll_sum': total, 'token_weighted_NLL': total / count,
                'token_weighted_PPL': math.exp(total / count),
                'mean_piece_mean_seed_NLL': statistics.mean(statistics.mean(v) for v in piece_values.values())})
        for piece in sorted({r['piece_id'] for r in trials}):
            selected = [r for r in trials if r['piece_id'] == piece and r['window'] == window]
            deltas = []
            for seed in sorted({r['seed'] for r in selected}):
                pair = {r['condition']: r for r in selected if r['seed'] == seed}
                deltas.append(pair['repair']['NLL'] - pair['baseline']['NLL'])
            paired.append({'piece_id': piece, 'title': selected[0]['title'], 'window': window,
                'seed_pairs': len(deltas), 'mean_NLL_delta': statistics.mean(deltas),
                'std_seed_NLL_delta': statistics.stdev(deltas) if len(deltas) > 1 else None})
    return trials, summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=['1', '2', '3'], required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu or args.batch_size < 1:
        raise ValueError('Explicit authorized GPU binding and positive batch size required')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from streammuse.infrastructure.inference.lekai_continuation_model.config import ModelConfig
    from streammuse.infrastructure.inference.lekai_continuation_model.my_tokenizer import PianoMusicTokenizer
    from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import PianoContinuationAdapter
    plan = read(args.root / 'plan.json')
    checkpoint = plan['checkpoints']['continuation']
    assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
    assert len(read(args.root / 'complete.json')) == 30
    tokenizer = PianoMusicTokenizer(config=ModelConfig())
    encoder = make_encoder(tokenizer)
    initial = encoder._continuation_initial_tokens(bpm=plan['config']['bpm'], time_signature_idx=4)
    items, sources, baseline_checks = [], [], 0
    pairs_equal_before = 0
    for piece in plan['pieces']:
        for seed in plan['seeds']:
            directory = args.root / piece['piece_id'] / f'seed{seed}'
            pair_items = []
            for condition in ('baseline', 'repair'):
                path = directory / f'{condition}_full.mid'
                events = midi_tracks(path)
                beats = encode_tracks(encoder, events, piece['stop_tick'])
                source = read(directory / condition / 'generation.json')
                trace = {r['tick']: r for r in source['rows']}
                sources.append({'piece_id': piece['piece_id'], 'seed': seed, 'condition': condition,
                                'midi': str(path), 'sha256': sha(path)})
                song_items = []
                for tick in range(32, piece['stop_tick'], 4):
                    target = tick // 4
                    prompt = prompt_for_target(initial, beats['Accompaniment'], beats['Melody'], target)
                    if condition == 'baseline':
                        if trace[tick]['input_tokens'] != [prompt]:
                            save(args.output / 'encoding_mismatch.json', {'piece': piece, 'seed': seed,
                                 'tick': tick, 'rebuilt': prompt, 'recorded': trace[tick]['input_tokens']})
                            raise AssertionError('Reconstructed baseline context differs from actual model input')
                        baseline_checks += 1
                    output = beats['Accompaniment'][target]
                    if len(prompt) + len(output) > ModelConfig().max_position_embeddings:
                        raise ValueError('Context exceeds model capacity; no silent truncation')
                    song_items.append({'piece_id': piece['piece_id'], 'title': piece['title'], 'seed': seed,
                        'condition': condition, 'tick': tick, 'prompt': prompt, 'target': output})
                pair_items.append(song_items)
                items.extend(song_items)
            for a, b in zip(*pair_items):
                if a['tick'] < 84:
                    assert a['prompt'] == b['prompt'] and a['target'] == b['target']
                    pairs_equal_before += 1
            print('PREPARED', piece['piece_id'], seed, flush=True)
    source_root = Path(__file__).resolve().parents[2]
    tracked = [Path(__file__), source_root / 'src/streammuse/infrastructure/inference/lekai_http_backend.py',
               source_root / 'src/streammuse/infrastructure/inference/lekai_model/MidiConverter.py']
    tracked += list((source_root / 'src/streammuse/infrastructure/inference/lekai_continuation_model').glob('*.py'))
    record = {'source_root': str(args.root), 'checkpoint': checkpoint, 'gpu': args.gpu,
        'dtype': 'float16 weights; float32 CE, natural logarithm', 'bpm_conditioning': plan['config']['bpm'],
        'context_beats': 32, 'target': 'canonical exported accompaniment beat, including EMPTY and ACC_END',
        'excluded_targets': 'Prompt prefix, Melody, BOS, TS, BPM, BAR, BEAT, padding',
        'probability': 'unmodified model logits, no temperature/top-k/top-p/repetition/tonal/empty masks',
        'history': 'Actual MIDI output history only; not internally repaired candidate history',
        'visibility': 'ACC-first causal: preceding completed Melody beats only',
        'windows': {'primary': [84, 'song_end'], 'secondary': [32, 'song_end']},
        'legacy_eval': 'Old eval RoFormer cal_nll is incompatible; this is separately named BEAT ACC-token NLL',
        'baseline_context_exact_matches': baseline_checks, 'equal_pre84_paired_beats': pairs_equal_before,
        'calls': len(items), 'sources': sources, 'source_sha256': {str(p): sha(p) for p in tracked},
        'max_sequence_length': max(len(i['prompt']) + len(i['target']) for i in items),
        'above_training_cutoff_2048': sum(len(i['prompt']) + len(i['target']) > 2048 for i in items)}
    save(args.output / 'manifest.json', record)
    print('LOADING_MODEL', checkpoint['path'], flush=True)
    adapter = PianoContinuationAdapter.from_checkpoint(checkpoint['path'], device='cuda', dtype=torch.float16, use_cache=False)
    model = adapter.model
    model.eval()
    assert not any(m.training for m in model.modules())
    scored, started = [], time.perf_counter()
    with (args.output / 'per_beat_tokens.jsonl').open('x', encoding='utf-8') as log:
        for offset in range(0, len(items), args.batch_size):
            batch = items[offset:offset + args.batch_size]
            losses = score_batch(model, batch)
            if offset == 0:
                save(args.output / 'forward_validation.json', validate_forward(model, batch, losses))
            for item, values in zip(batch, losses):
                assert len(values) == len(item['target'])
                row = {k: v for k, v in item.items() if k not in ('prompt', 'target')}
                row.update(token_count=len(values), nll_sum=math.fsum(values), NLL=statistics.mean(values),
                           prompt_tokens=len(item['prompt']))
                scored.append(row)
                log.write(json.dumps({**row, 'prompt_token_ids': item['prompt'], 'target_token_ids': item['target'],
                                      'target_nll': values}, ensure_ascii=False) + '\n')
            if offset % (args.batch_size * 50) == 0:
                print('SCORED', len(scored), '/', len(items), flush=True)
    assert len(scored) == len(items)
    trials, summaries, paired = aggregate(scored)
    for name, values in (('per_beat.csv', scored), ('per_trial.csv', trials), ('paired_piece.csv', paired)):
        with (args.output / name).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for source in sources:
        assert sha(Path(source['midi'])) == source['sha256']
    save(args.output / 'summary.json', {'summary': summaries, 'paired_piece_deltas': paired,
         'scoring_seconds': time.perf_counter() - started, 'all_source_midis_unchanged': True})
    print('COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()

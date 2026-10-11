"""CPU-only packaging of completed artifacts, without modifying source MIDI."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / 'plan.json').read_text(encoding='utf-8'))
    complete = json.loads((args.root / 'complete.json').read_text(encoding='utf-8'))
    target = args.root.parent / 'listening_by_piece'
    target.mkdir(exist_ok=False)
    rows, total_calls = [], 0
    for piece in plan['pieces']:
        title = re.sub(r'[<>:"/\\|?*]', '_', piece['title']).rstrip(' .')
        folder = target / (piece['piece_id'] + '_' + title)
        folder.mkdir()
        for i, seed in enumerate(plan['seeds']):
            source = args.root / piece['piece_id'] / f'seed{seed}'
            a = json.loads((source / 'baseline/generation.json').read_text(encoding='utf-8'))
            b = json.loads((source / 'repair/generation.json').read_text(encoding='utf-8'))
            assert [r['tick'] for r in a['rows']] == [r['tick'] for r in b['rows']]
            for x, y in zip(a['rows'], b['rows']):
                for key in ('part0_token_digest', 'part0_roll_digest'):
                    assert x['metadata'][key] == y['metadata'][key]
                if x['tick'] < 84:
                    assert x['events'] == y['events']
                    assert x['input_tokens'] == y['input_tokens']
                if b['repair']['status'] != 'applied':
                    assert x['events'] == y['events']
            total_calls += len(a['rows'])
            for j, condition in enumerate(('baseline', 'repair')):
                origin = source / f'{condition}_full.mid'
                dest = folder / f'{2*i+j+1:02d}_{condition}_s{seed}.mid'
                shutil.copy2(origin, dest)
                assert origin.read_bytes() == dest.read_bytes()
                rows.append({'piece_id': piece['piece_id'], 'title': piece['title'], 'seed': seed,
                             'condition': condition, 'repair_status': b['repair']['status'],
                             'midi': str(dest.relative_to(target)).replace('\\', '/'),
                             'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    with (target / 'index.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis = json.loads((args.root / 'music_analysis.json').read_text(encoding='utf-8'))
    deltas = [p for p in analysis['paired_piece_deltas'] if p['window'] == 'primary_after_repair']
    output = {'pairs': len(complete), 'midi_files': len(rows), 'paired_calls': total_calls,
              'melody_digests_equal': True, 'pre84_full_inputs_and_events_equal': True,
              'rejected_repair_all_events_equal': True,
              'repair_status_counts': dict(Counter(p['repair_status'] for p in complete)),
              'primary_paired_piece_deltas': {}}
    for key in ('CR', 'UR', 'onsets_per_beat', 'sounding_time_coverage'):
        values = [p[key + '_delta_repair_minus_baseline'] for p in deltas]
        assert all(v is not None for v in values)
        output['primary_paired_piece_deltas'][key] = {
            'mean': statistics.mean(values), 'sample_std_across_pieces': statistics.stdev(values),
            'pieces': len(values), 'positive': sum(v > 0 for v in values),
            'negative': sum(v < 0 for v in values), 'zero': sum(v == 0 for v in values)}
    with (args.root.parent / 'integrity_and_dispersion.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()

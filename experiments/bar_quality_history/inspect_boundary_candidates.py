"""CPU-only structural check of already-generated candidates; no resampling."""
import argparse
import json
from pathlib import Path

from experiments.bar_quality_history.boundary_projection import preserve_cross_boundary
from streammuse.infrastructure.inference.history_repair import _replay, prepare_history_repair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(args.source.glob('candidate_*.json')):
        original = json.loads(path.read_text())
        ticket = original['ticket']
        try:
            projected = preserve_cross_boundary(ticket, original['candidate'])
            history = ticket['accompaniment_events']
            active = _replay(history, 0, ticket['completed_until'], duplicate_release_before=ticket['start_tick'])
            prepare_history_repair(history, projected['events'], ticket['start_tick'], ticket['end_tick'],
                                   active, ticket['completed_until'])
            (args.output / path.name).write_text(json.dumps(dict(ticket=ticket, projected=projected), indent=2))
            rows.append(dict(source=path.name, status='structurally_valid',
                             selected_notes=projected['provenance']['boundary_projection']['selected_interior_notes']))
        except ValueError as exc:
            rows.append(dict(source=path.name, status='rejected', reason=str(exc)))
    result = dict(scope='CPU structural check only; no live application or quality claim', candidates=rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

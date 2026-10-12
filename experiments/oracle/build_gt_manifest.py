"""Build the GT-accompaniment manifest for run_oracle.py from explicit paths.

Layout-agnostic on purpose: masdar's user10 GT accompaniment location is
resolved at discovery time and passed here piece by piece. Hashes each MIDI
and emits {"accompaniment": {piece_id: {"midi", "sha256", "track"?}}}.
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--piece', action='append', required=True, metavar='ID=PATH',
                        help='piece_id=accompaniment_midi_path, repeatable')
    parser.add_argument('--track', action='append', default=[], metavar='ID=N',
                        help='optional note-track index per piece, repeatable')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tracks = dict(item.split('=', 1) for item in args.track)
    manifest = {'accompaniment': {}}
    for item in args.piece:
        piece_id, path = item.split('=', 1)
        entry = {'midi': str(Path(path).resolve()), 'sha256': sha(path)}
        if piece_id in tracks:
            entry['track'] = int(tracks[piece_id])
        manifest['accompaniment'][piece_id] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()

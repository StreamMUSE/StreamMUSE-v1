#!/usr/bin/env python3
"""
Automated debug script to compare Offline and Fake Realtime inference results.
"""

import sys
import json
import mido
from pathlib import Path
from typing import Dict, List, Optional


class InferenceConsistencyDebugger:
    """Debug tool for comparing offline and fake realtime inference."""

    def __init__(self, offline_dir: str, fakert_dir: str):
        self.offline_dir = Path(offline_dir)
        self.fakert_dir = Path(fakert_dir)
        self.report = {
            'input_comparison': {},
            'token_comparison': {},
            'output_comparison': {},
            'per_song': {},
            'bugs_found': []
        }

    def load_track_events(self, mid_path: str, track_keywords: List[str]) -> List[Dict]:
        """Extract note events from tracks whose name contains any keyword."""
        mid = mido.MidiFile(mid_path)
        events = []
        for track in mid.tracks:
            current_time = 0
            track_name = ""
            for msg in track:
                current_time += msg.time
                if msg.type == 'track_name':
                    track_name = str(getattr(msg, 'name', ''))
                    continue

                if msg.type in ['note_on', 'note_off'] and any(k in track_name for k in track_keywords):
                    events.append({
                        'tick': current_time,
                        'type': msg.type,
                        'pitch': msg.note,
                        'velocity': msg.velocity if msg.type == 'note_on' else 0
                    })
        return sorted(events, key=lambda x: (x['tick'], x['type']))

    def list_tracks(self, mid_path: str) -> List[str]:
        """List all track names in a MIDI file."""
        mid = mido.MidiFile(mid_path)
        names = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'track_name':
                    names.append(str(getattr(msg, 'name', '')))
                    break
        return names

    @staticmethod
    def has_target_track(track_names: List[str], track_keywords: List[str]) -> bool:
        """Return True if any track name contains any target keyword."""
        for name in track_names:
            if any(keyword in name for keyword in track_keywords):
                return True
        return False

    def compare_event_lists(self, events_a: List[Dict], events_b: List[Dict],
                            label_a: str = "A", label_b: str = "B") -> Dict:
        """Compare two event lists and return statistics."""
        result = {
            'count_a': len(events_a),
            'count_b': len(events_b),
            'count_diff': len(events_a) - len(events_b),
            'matched': 0,
            'mismatched': 0,
            'only_in_a': [],
            'only_in_b': []
        }

        set_a = {(e['tick'], e['pitch'], e['type']) for e in events_a}
        set_b = {(e['tick'], e['pitch'], e['type']) for e in events_b}

        result['matched'] = len(set_a & set_b)
        result['only_in_a'] = sorted(list(set_a - set_b))
        result['only_in_b'] = sorted(list(set_b - set_a))
        result['mismatched'] = len(result['only_in_a']) + len(result['only_in_b'])

        total = max(len(set_a), len(set_b), 1)
        result['match_rate'] = round(result['matched'] / total * 100, 2)

        return result

    def compare_outputs(self, offline_mid: str, fakert_mid: str) -> Dict:
        """Compare generated MIDI outputs."""
        track_keywords = ['Accompaniment', 'Part1']
        offline_tracks = self.list_tracks(offline_mid)
        fakert_tracks = self.list_tracks(fakert_mid)

        # Empty accompaniment is a valid state and should be compared as 0 events.
        # We only raise when the target accompaniment track itself cannot be found.
        offline_has_track = self.has_target_track(offline_tracks, track_keywords)
        fakert_has_track = self.has_target_track(fakert_tracks, track_keywords)

        if not offline_has_track or not fakert_has_track:
            raise ValueError(
                f'Cannot find accompaniment tracks.\n'
                f'  Offline tracks: {offline_tracks}\n'
                f'  FakeRT tracks: {fakert_tracks}\n'
                f'  Please verify track names include "Accompaniment" or "Part1".'
            )

        offline_acc = self.load_track_events(offline_mid, track_keywords)
        fakert_acc = self.load_track_events(fakert_mid, track_keywords)

        comp = self.compare_event_lists(offline_acc, fakert_acc, "Offline", "FakeRT")
        comp['offline_tracks'] = offline_tracks
        comp['fakert_tracks'] = fakert_tracks
        return comp

    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("Inference Consistency Debug Report")
        lines.append("=" * 70)

        # Per-song comparison
        lines.append("\n## Per-Song Comparison")
        per_song = self.report.get('per_song', {})
        for key in sorted(per_song.keys()):
            comp = per_song[key]
            lines.append(f"\n  Song: {key}")
            lines.append(f"    Offline events:  {comp.get('count_a', 'N/A')}")
            lines.append(f"    FakeRT events:   {comp.get('count_b', 'N/A')}")
            lines.append(f"    Matched:         {comp.get('matched', 'N/A')}")
            lines.append(f"    Match rate:      {comp.get('match_rate', 'N/A')}%")
            if comp.get('only_in_a'):
                lines.append(f"    Only in Offline (first 3): {list(comp['only_in_a'])[:3]}")
            if comp.get('only_in_b'):
                lines.append(f"    Only in FakeRT  (first 3): {list(comp['only_in_b'])[:3]}")

        # Aggregate
        lines.append("\n## Aggregate Summary")
        comp = self.report['output_comparison']
        lines.append(f"  Total Offline events: {comp.get('count_a', 'N/A')}")
        lines.append(f"  Total FakeRT events:  {comp.get('count_b', 'N/A')}")
        lines.append(f"  Total Matched: {comp.get('matched', 'N/A')}")
        lines.append(f"  Total Mismatched: {comp.get('mismatched', 'N/A')}")
        lines.append(f"  Overall Match Rate: {comp.get('match_rate', 'N/A')}%")

        # Bugs found
        lines.append("\n## Bugs Detected")
        if self.report['bugs_found']:
            for bug in self.report['bugs_found']:
                lines.append(f"  - {bug}")
        else:
            lines.append("  No obvious bugs detected (check details above)")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def _offline_song_key(path: Path) -> Optional[str]:
    suffix = '_generated.mid'
    if not path.name.endswith(suffix):
        return None
    base = path.name[:-len(suffix)]  # e.g. 000_1
    parts = base.split('_', 1)
    return parts[1] if len(parts) == 2 else None


def _fakert_song_key(path: Path) -> Optional[str]:
    suffix = '_fake_realtime_combined.mid'
    if not path.name.endswith(suffix):
        return None
    return path.name[:-len(suffix)]  # e.g. 1


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_inference_consistency.py <offline_dir> <fakert_dir> [report_dir]")
        sys.exit(1)

    offline_dir = sys.argv[1]
    fakert_dir = sys.argv[2]
    report_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(offline_dir)

    debugger = InferenceConsistencyDebugger(offline_dir, fakert_dir)

    offline_map = {}
    for mid_path in debugger.offline_dir.glob('*_generated.mid'):
        key = _offline_song_key(mid_path)
        if key:
            offline_map[key] = mid_path

    fakert_map = {}
    for mid_path in debugger.fakert_dir.glob('*_fake_realtime_combined.mid'):
        key = _fakert_song_key(mid_path)
        if key:
            fakert_map[key] = mid_path

    print(f"Found offline songs: {sorted(offline_map.keys())}")
    print(f"Found fakert songs:  {sorted(fakert_map.keys())}")

    common_keys = sorted(set(offline_map) & set(fakert_map))
    if not common_keys:
        raise RuntimeError('No matched song stems found between offline and fake realtime outputs.')

    print(f"Comparing songs: {common_keys}")

    per_song = {}
    total_a = 0
    total_b = 0
    total_matched = 0
    total_mismatched = 0

    for key in common_keys:
        print(f"\n  Comparing song '{key}':")
        print(f"    Offline: {offline_map[key]}")
        print(f"    FakeRT:  {fakert_map[key]}")
        try:
            comp = debugger.compare_outputs(str(offline_map[key]), str(fakert_map[key]))
            per_song[key] = comp
            total_a += int(comp['count_a'])
            total_b += int(comp['count_b'])
            total_matched += int(comp['matched'])
            total_mismatched += int(comp['mismatched'])
            print(f"    Match rate: {comp['match_rate']}% ({comp['matched']}/{max(comp['count_a'], comp['count_b'])})")
        except ValueError as e:
            print(f"    ERROR: {e}")
            per_song[key] = {'error': str(e)}

    debugger.report['per_song'] = per_song

    total_events = max(total_a, total_b, 1)
    debugger.report['output_comparison'] = {
        'count_a': total_a,
        'count_b': total_b,
        'count_diff': total_a - total_b,
        'matched': total_matched,
        'mismatched': total_mismatched,
        'match_rate': round(total_matched / total_events * 100, 2),
    }

    # Auto-detect bugs
    overall_rate = debugger.report['output_comparison']['match_rate']
    if overall_rate < 50:
        debugger.report['bugs_found'].append(
            f"CRITICAL: overall match rate {overall_rate}% < 50% — likely systematic bug"
        )
    elif overall_rate < 90:
        debugger.report['bugs_found'].append(
            f"WARNING: overall match rate {overall_rate}% < 90% — significant discrepancy"
        )
    elif overall_rate < 95:
        debugger.report['bugs_found'].append(
            f"MINOR: overall match rate {overall_rate}% < 95% — minor discrepancy, check quantization"
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'inference_consistency_report.json'
    report_path.write_text(json.dumps(debugger.report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(debugger.generate_report())
    print(f"\nJSON report saved to: {report_path}")


if __name__ == "__main__":
    main()

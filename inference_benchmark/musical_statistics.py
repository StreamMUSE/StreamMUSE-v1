"""
Helper functions for calculating these statictics across a midi file or a collection of midi files.
Statistics include:
- total number of notes
- note density
- note durations
- note pitches
- note pitches without octaves
- note velocities
- harmonic scores

For a single file, the statistics are calculated and returned as a dictionary of values or frequency distirbutions.
For a collection of files, the statistics are calculated and returned as a dictionary of frequency distributions.
"""

import os
from pathlib import Path
import glob
import json
import pretty_midi
import music21 as m21
from itertools import combinations
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import seaborn as sns
import math
import copy
import tempfile
from tqdm import tqdm

def calculate_tension_scores(score, chord_list, cutoff_bar):
    """
    Calculates a tension score for each chord by counting dissonant intervals.
    The tension over time is calculated in absolute terms: 4 bins per bar, up to the cutoff_bar.
    
    Returns a list of integer scores and the distribution over time.
    """
    tension_scores = []
    
    # Default to 4/4 time signature if none is found in the score.
    time_signatures = score.getTimeSignatures()
    ts = time_signatures[0] if time_signatures else m21.meter.TimeSignature('4/4')
        
    # Calculate the duration of one bin in quarter-lengths (each bar has 4 bins).
    bar_duration_ql = ts.barDuration.quarterLength
    bin_duration_ql = bar_duration_ql / 4.0
    
    # Initialize bins for the "over time" distribution.
    num_bins = cutoff_bar * 4
    tension_over_time = [[] for _ in range(num_bins)]

    try:
        for c in chord_list:
            # Calculate tension for the current chord.
            chord_tension = 0
            if len(c.pitches) >= 2:
                for note1, note2 in combinations(c.pitches, 2):
                    note_interval = m21.interval.Interval(note1, note2)
                    if not note_interval.isConsonant():
                        chord_tension += 1
            tension_scores.append(chord_tension)
            
            # Calculate the absolute bin index from the chord's offset.
            bin_index = int(c.offset / bin_duration_ql)
            
            # Add the tension score to the correct bin if it's within the cutoff.
            if bin_index < num_bins:
                tension_over_time[bin_index].append(chord_tension)

        # Average the tension in each bin, using 0 for empty bins.
        tension_over_time = [np.mean(tension) if tension else 0 for tension in tension_over_time]

    except Exception as e:
        print(f"An error occurred in calculate_tension_scores: {e}")
        return [], []
        
    return tension_scores, tension_over_time

def calculate_functionality_scores(score, chord_list, key, cutoff_bar):
    """
    Calculates a functionality score for each chord based on its role
    within the song's key. The distribution over time is calculated in 
    absolute terms: 4 bins per bar, up to the cutoff_bar.

    Returns a list of integer scores and the distribution over time.
    """
    functionality_scores = []
    
    # Default to 4/4 time signature if none is found in the score.
    time_signatures = score.getTimeSignatures()
    ts = time_signatures[0] if time_signatures else m21.meter.TimeSignature('4/4')
        
    # Calculate the duration of one bin in quarter-lengths (each bar has 4 bins).
    bar_duration_ql = ts.barDuration.quarterLength
    bin_duration_ql = bar_duration_ql / 4.0
    
    # Initialize bins for the "over time" distribution.
    num_bins = cutoff_bar * 4
    functionality_over_time = [[] for _ in range(num_bins)]
                
    try:
        for c in chord_list:
            try:
                # Calculate functionality for the current chord.
                rn = m21.roman.romanNumeralFromChord(c, key)
                score_val = rn.functionalityScore
                functionality_scores.append(score_val)
                
                # Calculate the absolute bin index from the chord's offset.
                bin_index = int(c.offset / bin_duration_ql)

                # Add the functionality score to the correct bin if it's within the cutoff.
                if bin_index < num_bins:
                    functionality_over_time[bin_index].append(score_val)
                    
            except m21.roman.RomanNumeralException:
                # Skip chords that cannot be analyzed as a Roman Numeral.
                continue

        # Average the functionality in each bin, using 0 for empty bins.
        functionality_over_time = [np.mean(func) if func else 0 for func in functionality_over_time]
                
    except Exception as e:
        print(f"An error occurred in calculate_functionality_scores: {e}")
        return [], []
        
    return functionality_scores, functionality_over_time

def calculate_statistics(midi_file, acc_separated=False, cutoff_bar=25):
    # acc_separated = True does not currently work yet

    # 1. Load with pretty_midi to get accurate timing in seconds.
    pm = pretty_midi.PrettyMIDI(midi_file)
    if acc_separated:
        pm_acc = pretty_midi.PrettyMIDI(midi_file.replace('mel', 'acc'))
        pm.instruments.extend(pm_acc.instruments)

    # 2. Use get_downbeats() to find the time in seconds of the cutoff bar.
    downbeats = pm.get_downbeats()
    if len(downbeats) > cutoff_bar:
        max_time_sec = downbeats[cutoff_bar]
    else:
        max_time_sec = pm.get_end_time()

    midi_duration = max_time_sec
    if midi_duration <= 1:
        print(f"Warning: MIDI file {midi_file} has insufficient duration after cutoff. Skipping.")
        return {}

    all_notes = [note for inst in pm.instruments for note in inst.notes if note.start < max_time_sec]
    total_notes = len(all_notes)
    
    # Load and process with music21.
    score = m21.converter.parse(midi_file)
    score.makeRests(fillGaps=True, inPlace=True)
    # score = score.voicesToParts()
    score.stripTies(inPlace=True)
    key = score.analyze('key')

    # Ensure a tempo marking exists in the main score.
    # First, try to get the original tempo from the MIDI file
    original_tempo = 120  # Default fallback
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[1]) > 0:
        original_tempo = tempo_changes[1][0]  # Use the first (initial) tempo
    
    if not score.iter().getElementsByClass('MetronomeMark'):
        print(f"No tempo found in music21, using original MIDI tempo: {original_tempo:.1f} BPM")
        score.insert(0, m21.tempo.MetronomeMark(number=original_tempo))
    else:
        print(f"Using existing tempo from music21")

    # **FIXED SECTION: Copy tempo into the new chord stream.**
    # `chordify()` creates a new stream that needs the tempo info.
    chords_stream = score.chordify()
    for mm in score.iter().getElementsByClass('MetronomeMark'):
        chords_stream.insert(mm.offset, mm)

    # Get time signature to calculate bar boundaries
    time_signatures = score.getTimeSignatures()
    ts = time_signatures[0] if time_signatures else m21.meter.TimeSignature('4/4')
    bar_duration_ql = ts.barDuration.quarterLength
    max_offset_ql = cutoff_bar * bar_duration_ql

    # Now get chords from the stream that definitely has tempo.
    all_chords = list(chords_stream.recurse().getElementsByClass('Chord'))
    
    # Sort chords by offset to ensure proper chronological order
    all_chords_sorted = sorted(all_chords, key=lambda c: c.offset)
    
    # Filter chords by bar position (offset in quarter lengths) only
    # Note: We don't use c.seconds for filtering because music21's tempo conversion can be unreliable
    chord_list = []
    for c in all_chords_sorted:
        # Check if chord is within bar cutoff
        if c.offset < max_offset_ql:
            chord_list.append(c)
    
    print(f'Cutoff bar: {cutoff_bar}, max seconds: {max_time_sec:.2f}, max offset (QL): {max_offset_ql:.2f}')
    print(f'Bar duration (QL): {bar_duration_ql:.2f}, Time signature: {ts}')
    print(f'Total chords in file: {len(all_chords_sorted)}')
    print(f'Chords within {cutoff_bar} bars: {len(chord_list)}')
    if chord_list:
        print(f'First chord: offset {chord_list[0].offset:.2f}')
        print(f'Last chord: offset {chord_list[-1].offset:.2f}')
        print(f'Chord offset range: {chord_list[-1].offset - chord_list[0].offset:.2f} quarter lengths')
    if all_chords_sorted and len(all_chords_sorted) > len(chord_list):
        print(f'Filtered out {len(all_chords_sorted) - len(chord_list)} chords beyond {cutoff_bar} bars')
        if len(all_chords_sorted) > 0:
            last_chord = all_chords_sorted[-1]
            print(f'Last chord in file: offset {last_chord.offset:.2f}')

    if total_notes > 0:
        note_density = total_notes / midi_duration
        durations = [note.end - note.start for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        pitches = [note.pitch for note in all_notes]
        pitches_no_octaves = [note.pitch % 12 for note in all_notes]
        
        tension_distribution, tension_distribution_over_time = calculate_tension_scores(score, chord_list, cutoff_bar)
        functionality_distribution, functionality_distribution_over_time = calculate_functionality_scores(score, chord_list, key, cutoff_bar)
    else:
        return {}

    return {
        'file': midi_file,
        'total_notes': total_notes,
        'note_density': note_density,
        'duration_distribution': durations,
        'velocity_distribution': velocities,
        'pitch_distribution': pitches,
        'pitch_distribution_no_octaves': pitches_no_octaves,
        'tension_distribution': tension_distribution,
        'functionality_distribution': functionality_distribution,
        'tension_distribution_over_time': tension_distribution_over_time,
        'functionality_distribution_over_time': functionality_distribution_over_time,
        'mean_duration': np.mean(durations) if durations else 0,
        'mean_velocity': np.mean(velocities) if velocities else 0,
        'mean_pitch': np.mean(pitches) if pitches else 0,
        'mean_pitch_no_octaves': np.mean(pitches_no_octaves) if pitches_no_octaves else 0,
        'mean_tension': np.mean(tension_distribution) if tension_distribution else 0,
        'mean_functionality': np.mean(functionality_distribution) if functionality_distribution else 0,
    }

def calculate_statistics_collection(midi_files_dir, glob_string, acc_separated=False, cutoff_bar=None):
    all_stats = []
    
    midi_files = list(glob.glob(glob_string, root_dir=Path(midi_files_dir), recursive=False))
    midi_files = [Path(midi_files_dir) / midi_file for midi_file in midi_files]
    if len(midi_files) >= 100:
        midi_files = np.random.choice(midi_files, 100, replace=False)
    midi_files = sorted(midi_files)
    print(midi_files)

    for midi_file in tqdm(midi_files, desc="Calculating statistics"):
        stats = calculate_statistics(str(midi_file), acc_separated=acc_separated, cutoff_bar=cutoff_bar)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        return {}
    
    # Save distribution statistics for each file as json
    stats_dict = {}
    for stat in all_stats:
        stats_dict[stat['file']] = {
            'total_notes': stat['total_notes'],
            'note_density': stat['note_density'],
            'duration_distribution': stat['duration_distribution'],
            'velocity_distribution': stat['velocity_distribution'],
            'pitch_distribution': stat['pitch_distribution'],
            'pitch_distribution_no_octaves': stat['pitch_distribution_no_octaves'],
            'tension_distribution': stat['tension_distribution'],
            'functionality_distribution': stat['functionality_distribution'],
            'tension_distribution_over_time': stat['tension_distribution_over_time'],
            'functionality_distribution_over_time': stat['functionality_distribution_over_time'],
        }
    stats_file = Path(midi_files_dir) / 'file_independent_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    # Save distribution statistics for the collection as json
    collection_stats = {
        'total_notes': [],
        'note_density': [],
        'duration_distribution': [],
        'velocity_distribution': [],
        'pitch_distribution': [],
        'pitch_distribution_no_octaves': [],
        'tension_distribution': [],
        'functionality_distribution': [],
        'tension_distribution_over_time': [],
        'functionality_distribution_over_time': []
    }
    for stat in all_stats:
        collection_stats['total_notes'].append(stat['total_notes'])
        collection_stats['note_density'].append(stat['note_density'])
        collection_stats['duration_distribution'].extend(stat['duration_distribution'])
        collection_stats['velocity_distribution'].extend(stat['velocity_distribution'])
        collection_stats['pitch_distribution'].extend(stat['pitch_distribution'])
        collection_stats['pitch_distribution_no_octaves'].extend(stat['pitch_distribution_no_octaves'])
        collection_stats['tension_distribution'].extend(stat['tension_distribution'])
        collection_stats['functionality_distribution'].extend(stat['functionality_distribution'])
    
    # Calculate average tension and functionality distributions over time across all files
    collection_stats['tension_distribution_over_time'] = [0 for _ in range(100)]
    for i in range(len(all_stats[0]['tension_distribution_over_time'])):
        collection_stats['tension_distribution_over_time'][i] = np.mean([stat['tension_distribution_over_time'][i] for stat in all_stats])
    
    collection_stats['functionality_distribution_over_time'] = [0 for _ in range(100)]
    for i in range(len(all_stats[0]['functionality_distribution_over_time'])):
        collection_stats['functionality_distribution_over_time'][i] = np.mean([stat['functionality_distribution_over_time'][i] for stat in all_stats])
    
    # Save the collection statistics to a json file
    collection_stats_file = Path(midi_files_dir) / 'collection_distribution_statistics.json'
    with open(collection_stats_file, 'w') as f:
        json.dump(collection_stats, f, indent=4)

    # Aggregate statistics
    aggregated_stats = {
        'file': [],
        'total_notes': [],
        'note_density': [],
        'duration_distribution': [],
        'velocity_distribution': [],
        'pitch_distribution': [],
        'pitch_distribution_no_octaves': [],
        'tension_distribution': [],
        'functionality_distribution': []
    }

    for stat in all_stats:
        aggregated_stats['file'].append(stat['file'])
        aggregated_stats['total_notes'].append(stat['total_notes'])
        aggregated_stats['note_density'].append(stat['note_density'])
        aggregated_stats['duration_distribution'].append(stat['mean_duration'])
        aggregated_stats['velocity_distribution'].append(stat['mean_velocity'])
        aggregated_stats['pitch_distribution'].append(stat['mean_pitch'])
        aggregated_stats['pitch_distribution_no_octaves'].append(stat['mean_pitch_no_octaves'])
        aggregated_stats['tension_distribution'].append(stat['mean_tension'])
        aggregated_stats['functionality_distribution'].append(stat['mean_functionality'])


    collection_mean_stats_file = Path(midi_files_dir) / 'collection_mean_statistics.json'
    df_stats = pd.DataFrame(aggregated_stats)
    df_stats.to_csv(collection_mean_stats_file, index=False)
    print(f"Statistics saved to {collection_mean_stats_file}")
    return aggregated_stats

def main(path):
    # Example single file usage
    # stats = calculate_statistics(path)
    # print(stats)

    # Example collection usage
    aggregated_stats = calculate_statistics_collection(path, glob_string='*.mid', acc_separated=False, cutoff_bar=25)
    # print(aggregated_stats)

if __name__ == "__main__":
    # Example single file usage
    # midi_file = 'inputs/pop909_dataset/acc/001.mid'
    # main(midi_file)

    # Example collection usage
    # midi_files = '../../../original_dataset/POP909-Dataset/POP909'
    # midi_files = '../../../original_dataset/aria-midi-v1-unique-ext/data'
    # midi_files = 'outputs/fake_offline_no_latency/aria_unique_skyline_top2'
    # main(midi_files)
    
    paths = [
        # '../../../original_dataset/aria-midi-v1-unique-ext/data',
        # '../../../original_dataset/POP909-Dataset/POP909',
        # 'outputs/0722_full_run/fake_offline_script/aria_unique_skyline_top2',
        'outputs/0722_full_run/fake_offline_script/pop909_dataset',
        # 'outputs/0722_full_run/fake_offline_script_lantency2/aria_unique_skyline_top2',
        'outputs/0722_full_run/fake_offline_script_lantency2/pop909_dataset',
        # 'outputs/0722_full_run/real_offline_Xinyue_new_chord_script/aria_unique_skyline_top2',
        'outputs/0722_full_run/real_offline_Xinyue_new_chord_script/pop909_dataset',
        # 'outputs/0722_full_run/real_offline_script/aria_unique_skyline_top2',
        'outputs/0722_full_run/real_offline_script/pop909_dataset',
        # 'outputs/0722_full_run/real_offline_Xinyue_new_script/aria_unique_skyline_top2',
        'outputs/0722_full_run/real_offline_Xinyue_new_script/pop909_dataset',
    ]

    for midi_files in tqdm(paths, desc="Calculating statistics for collections"):
        print(f"Calculating statistics for {midi_files}")
        if not os.path.exists(midi_files):
            print(f"Path {midi_files} does not exist. Skipping.")
            continue
        aggregated_stats = calculate_statistics_collection(midi_files, glob_string='*.mid', acc_separated=False, cutoff_bar=25)

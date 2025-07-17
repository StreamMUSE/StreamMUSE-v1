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
from tqdm import tqdm

def calculate_tension_scores(chord_list):
    """
    Calculates a tension score for each chord in a MIDI file by counting
    its dissonant intervals.
    
    Returns a list of integer scores, where 0 is perfectly consonant. Lower is better.
    """
    tension_scores = []
    try:
        for c in chord_list:
            chord_tension = 0
            if len(c.pitches) >= 2:
                for note1, note2 in combinations(c.pitches, 2):
                    note_interval = m21.interval.Interval(note1, note2)
                    if not note_interval.isConsonant():
                        chord_tension += 1
            tension_scores.append(chord_tension)
            
    except Exception as e:
        print(f"An error occurred in calculate_tension_scores: {e}")
        return []
        
    return tension_scores

def calculate_functionality_scores(chord_list, key):
    """
    Calculates a functionality score for each chord based on its role
    within the song's automatically detected key.

    Returns a list of integer scores (typically 0-100). Higher is better.
    """
    functionality_scores = []
    try:
        for c in chord_list:
            try:
                rn = m21.roman.romanNumeralFromChord(c, key)
                functionality_scores.append(rn.functionalityScore)
            except m21.roman.RomanNumeralException:
                continue
                
    except Exception as e:
        print(f"An error occurred in calculate_functionality_scores: {e}")
        return []
        
    return functionality_scores

def calculate_statistics(midi_file, acc_separated=False):
    pm = pretty_midi.PrettyMIDI(midi_file)
    if acc_separated:
        pm_acc = pretty_midi.PrettyMIDI(midi_file.replace('mel', 'acc'))
        pm.instruments.extend(pm_acc.instruments)

    score = m21.converter.parse(midi_file)
    if acc_separated:
        score_acc = m21.converter.parse(midi_file.replace('mel', 'acc'))
        for part in score_acc.parts:
            score.append(part)
            
    key = score.analyze('key')
    
    chords = score.chordify()
    chord_list = chords.flatten().getElementsByClass('Chord')
    
    midi_duration = pm.get_end_time()

    if midi_duration <= 30:
        print(f"Warning: MIDI file {midi_file} has less than 30 duration. Skipping.")
        return {}

    all_notes = [note for inst in pm.instruments for note in inst.notes]
    total_notes = len(all_notes)

    if total_notes >= 0:
        note_density = total_notes / midi_duration
        durations = [note.end - note.start for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        pitches = [note.pitch for note in all_notes]
        pitches_no_octaves = [note.pitch % 12 for note in all_notes]
        tension_distribution = calculate_tension_scores(chord_list)
        functionality_distribution = calculate_functionality_scores(chord_list, key)
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

        'mean_duration': np.mean(durations) if durations else 0,
        'mean_velocity': np.mean(velocities) if velocities else 0,
        'mean_pitch': np.mean(pitches) if pitches else 0,
        'mean_pitch_no_octaves': np.mean(pitches_no_octaves) if pitches_no_octaves else 0,
        'mean_tension': np.mean(tension_distribution) if tension_distribution else 0,
        'mean_functionality': np.mean(functionality_distribution) if functionality_distribution else 0,
    }

def calculate_statistics_collection(midi_files_dir, glob_string, acc_separated=False):
    all_stats = []
    
    midi_files = list(glob.glob(glob_string, root_dir=Path(midi_files_dir), recursive=False))
    midi_files = [Path(midi_files_dir) / midi_file for midi_file in midi_files]
    if len(midi_files) >= 100:
        midi_files = np.random.choice(midi_files, 100, replace=False)
    print(midi_files)

    for midi_file in tqdm(midi_files, desc="Calculating statistics"):
        stats = calculate_statistics(str(midi_file), acc_separated=acc_separated)
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
            'functionality_distribution': stat['functionality_distribution']
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
        'functionality_distribution': []
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
    aggregated_stats = calculate_statistics_collection(path, glob_string='*.mid', acc_separated=False)
    # print(aggregated_stats)

if __name__ == "__main__":
    # Example single file usage
    # midi_file = 'inputs/pop909_dataset/acc/001.mid'
    # main(midi_file)

    # Example collection usage
    # midi_files = '../../../original_dataset/POP909-Dataset/POP909'
    # midi_files = '../../../original_dataset/aria-midi-v1-unique-ext/data'
    midi_files = 'outputs/fake_offline_no_latency/aria_unique_skyline_top2'
    main(midi_files)
    

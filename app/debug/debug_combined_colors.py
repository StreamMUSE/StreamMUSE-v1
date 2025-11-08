#!/usr/bin/env python3
"""
Debug the combined constraint analysis to check color mapping.
"""

import pandas as pd
import numpy as np
import glob
import os

def load_benchmark_data(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'round_trip_time' in df.columns:
                filename = os.path.basename(file)
                if filename.startswith('gen_') and filename.endswith('.csv'):
                    gen_length_frames = int(filename.replace('gen_', '').replace('.csv', ''))
                    gen_length_ticks = (gen_length_frames + 1) // 2
                    df['generation_length'] = gen_length_ticks
                    df['generation_length_frames'] = gen_length_frames
                    all_data.append(df[['round_trip_time', 'generation_length', 'generation_length_frames']])
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_percentiles_by_generation_length(data, percentiles=[50, 60, 70, 80, 90, 99.5]):
    percentile_data = {}
    
    for gen_length in sorted(data['generation_length'].unique()):
        gen_data = data[data['generation_length'] == gen_length]
        rtt_ms = gen_data['round_trip_time'] * 1000
        
        for p in percentiles:
            percentile_value = np.percentile(rtt_ms, p)
            percentile_data[(gen_length, p)] = percentile_value
    
    return percentile_data

def evaluate_constraints(inference_interval, generation_length, round_trip_time_ms):
    TICK_DURATION_MS = 125
    constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
    musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
    constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
    return constraint1_satisfied and constraint2_satisfied

def create_combined_constraint_matrix(percentile_data, inference_range, generation_range):
    percentiles = [50, 60, 70, 80, 90, 99.5]
    matrix = np.full((len(inference_range), len(generation_range)), -1, dtype=int)
    
    for i, inference_interval in enumerate(inference_range):
        for j, generation_length in enumerate(generation_range):
            if (generation_length, 50) not in percentile_data:
                matrix[i, j] = -2  # No data
                continue
            
            highest_valid_percentile = -1
            
            for p_idx, percentile in enumerate(percentiles):
                if (generation_length, percentile) in percentile_data:
                    rtt_ms = percentile_data[(generation_length, percentile)]
                    if evaluate_constraints(inference_interval, generation_length, rtt_ms):
                        highest_valid_percentile = p_idx
                    else:
                        break
            
            matrix[i, j] = highest_valid_percentile
    
    return matrix

def debug_combined_matrix():
    """Debug the combined matrix to understand the color mapping."""
    data_dir = "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/results_Stream_MUSE/bench_results/manual_local_server"
    
    print("=== DEBUGGING COMBINED CONSTRAINT MATRIX ===")
    
    data = load_benchmark_data(data_dir)
    percentile_data = calculate_percentiles_by_generation_length(data)
    
    inference_range = np.arange(1, 5)
    generation_range = np.arange(1, 9)
    
    matrix = create_combined_constraint_matrix(percentile_data, inference_range, generation_range)
    
    print("Matrix shape:", matrix.shape)
    print("Matrix values:")
    print(matrix)
    print()
    
    print("Unique values in matrix:", np.unique(matrix))
    print()
    
    # Map values to percentiles
    percentiles = [50, 60, 70, 80, 90, 99.5]
    
    print("Value mapping:")
    print("-2: No data")
    print("-1: Invalid at all percentiles")
    for i, p in enumerate(percentiles):
        print(f"{i}: Valid up to {p}th percentile")
    print()
    
    # Show specific examples
    print("Specific parameter combinations:")
    for i, inference_interval in enumerate(inference_range):
        for j, generation_length in enumerate(generation_range):
            value = matrix[i, j]
            if value == -2:
                description = "No data"
            elif value == -1:
                description = "Invalid at all percentiles"
            else:
                description = f"Valid up to {percentiles[value]}th percentile"
            
            print(f"({inference_interval}, {generation_length}): {value} -> {description}")

if __name__ == "__main__":
    debug_combined_matrix()
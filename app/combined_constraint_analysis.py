#!/usr/bin/env python3
"""
Combined Constraint Analysis for StreamMUSE

Creates a single heatmap showing the highest percentile at which each parameter 
combination remains valid. Uses a color spectrum from red (completely invalid) 
to green (valid even at the strictest 99.5th percentile).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

def load_benchmark_data(data_dir):
    """
    Load all benchmark CSV files and combine round trip time data.
    
    Args:
        data_dir: Directory containing benchmark CSV files
    
    Returns:
        Combined DataFrame with all round trip time measurements
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'round_trip_time' in df.columns:
                # Extract generation length from filename (e.g., gen_5.csv -> 5 frames)
                filename = os.path.basename(file)
                if filename.startswith('gen_') and filename.endswith('.csv'):
                    gen_length_frames = int(filename.replace('gen_', '').replace('.csv', ''))
                    # Convert frames to ticks: ticks = (frames + 1) / 2
                    gen_length_ticks = (gen_length_frames + 1) // 2
                    df['generation_length'] = gen_length_ticks
                    df['generation_length_frames'] = gen_length_frames
                    all_data.append(df[['round_trip_time', 'generation_length', 'generation_length_frames']])
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if not all_data:
        raise ValueError(f"No valid benchmark data found in {data_dir}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_percentiles_by_generation_length(data, percentiles=[50, 60, 70, 80, 90, 99.5]):
    """
    Calculate round trip time percentiles for each generation length separately.
    
    Args:
        data: DataFrame with round_trip_time and generation_length columns
        percentiles: List of percentiles to calculate
    
    Returns:
        Dictionary mapping (generation_length, percentile) to round trip time in milliseconds
    """
    percentile_data = {}
    
    for gen_length in sorted(data['generation_length'].unique()):
        gen_data = data[data['generation_length'] == gen_length]
        rtt_ms = gen_data['round_trip_time'] * 1000  # Convert to milliseconds
        
        for p in percentiles:
            percentile_value = np.percentile(rtt_ms, p)
            percentile_data[(gen_length, p)] = percentile_value
    
    return percentile_data

def evaluate_constraints(inference_interval, generation_length, round_trip_time_ms):
    """
    Evaluate parameter constraints.
    
    Args:
        inference_interval: Inference interval in ticks
        generation_length: Generation length in ticks
        round_trip_time_ms: Round trip time in milliseconds
    
    Returns:
        bool: True if both constraints are satisfied
    """
    TICK_DURATION_MS = 125  # 125ms per tick at 120 BPM
    
    # Constraint 1: Inference interval must be long enough to avoid overlap
    # inference_interval_ticks * 125ms >= round_trip_time
    constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
    
    # Constraint 2: Generated music must arrive before it's needed
    # round_trip_time < (generation_length - inference_interval) * 125ms
    musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
    constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
    
    return constraint1_satisfied and constraint2_satisfied

def create_combined_constraint_matrix(percentile_data, inference_range, generation_range):
    """
    Create a matrix showing the highest percentile at which each parameter combination is valid.
    
    Args:
        percentile_data: Dictionary mapping (generation_length, percentile) to RTT
        inference_range: Range of inference intervals to test
        generation_range: Range of generation lengths to test
    
    Returns:
        2D numpy array with values representing the highest valid percentile
        -1 = completely invalid (red)
        0 = valid at 50th percentile (lightest)
        1 = valid at 60th percentile
        2 = valid at 70th percentile
        3 = valid at 80th percentile
        4 = valid at 90th percentile
        5 = valid at 99.5th percentile (green)
        -2 = no data available
    """
    percentiles = [50, 60, 70, 80, 90, 99.5]
    matrix = np.full((len(inference_range), len(generation_range)), -1, dtype=int)
    
    for i, inference_interval in enumerate(inference_range):
        for j, generation_length in enumerate(generation_range):
            # Check if we have data for this generation length
            if (generation_length, 50) not in percentile_data:
                matrix[i, j] = -2  # No data
                continue
            
            # Find the highest percentile at which this combination is valid
            highest_valid_percentile = -1
            
            for p_idx, percentile in enumerate(percentiles):
                if (generation_length, percentile) in percentile_data:
                    rtt_ms = percentile_data[(generation_length, percentile)]
                    if evaluate_constraints(inference_interval, generation_length, rtt_ms):
                        highest_valid_percentile = p_idx
                    else:
                        # Once we hit an invalid percentile, stop checking higher ones
                        break
            
            matrix[i, j] = highest_valid_percentile
    
    return matrix

def plot_combined_constraint_analysis(percentile_data, output_dir):
    """
    Create a single heatmap showing the highest percentile at which each parameter combination is valid.
    
    Args:
        percentile_data: Dictionary mapping (generation_length, percentile) to RTT in ms
        output_dir: Directory to save the plot
    """
    # Define parameter ranges (remove 0s as they don't make sense)
    inference_range = np.arange(1, 5)  # 1-4 inclusive
    generation_range = np.arange(1, 9)  # 1-8 inclusive
    
    # Create combined constraint matrix
    combined_matrix = create_combined_constraint_matrix(percentile_data, inference_range, generation_range)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Combined Parameter Constraint Analysis\nColor shows highest percentile at which combination is valid', 
                fontsize=18, fontweight='bold', y=0.92)
    
    # Define colors for different validity levels using dark green to light green gradient
    # Dark green (relaxed percentiles) to light green (strict percentiles), with invalid as dark grey
    # Matrix has values: -1, 0, 1, 2, 3, 4 (no 5 or -2 in actual data)
    from matplotlib.colors import ListedColormap
    colors = [
        '#404040',  # -1: Completely invalid (dark grey)
        '#2E7D32',  # 0: Valid at 50th percentile (dark green - relaxed requirement)
        '#4CAF50',  # 1: Valid at 60th percentile (green)
        '#66BB6A',  # 2: Valid at 70th percentile (medium green)
        '#81C784',  # 3: Valid at 80th percentile (medium light green)
        '#A5D6A7',  # 4: Valid at 90th percentile (light green)
        '#C8E6C9'   # 5: Valid at 99.5th percentile (very light green - strict requirement, if it existed)
    ]
    
    # Create custom colormap
    constraint_cmap = ListedColormap(colors)
    
    # Create heatmap with same style as the other plots
    import seaborn as sns
    sns.heatmap(combined_matrix, 
               ax=ax,
               xticklabels=generation_range,
               yticklabels=inference_range,
               cmap=constraint_cmap,
               vmin=-1,  # Invalid
               vmax=5,   # Valid at 99.5th percentile (theoretical max)
               cbar=False,
               linewidths=1.0,
               linecolor='white',
               square=True,  # Make cells square
               annot=False)
    
    # Customize the plot
    ax.set_title('Highest Valid Percentile for Each Parameter Combination', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Generation Length (ticks)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Interval (ticks)', fontsize=14, fontweight='bold')
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Invert y-axis to match original plot style
    ax.invert_yaxis()
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with proper color mapping
    # Only include colors that actually appear in the data
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Invalid at all percentiles'),  # -1 -> index 0
        Patch(facecolor=colors[1], label='Valid ≤ 50th percentile'),    # 0 -> index 1
        Patch(facecolor=colors[2], label='Valid ≤ 60th percentile'),    # 1 -> index 2
        Patch(facecolor=colors[3], label='Valid ≤ 70th percentile'),    # 2 -> index 3
        Patch(facecolor=colors[4], label='Valid ≤ 80th percentile'),    # 3 -> index 4
        Patch(facecolor=colors[5], label='Valid ≤ 90th percentile'),    # 4 -> index 5
        Patch(facecolor=colors[6], label='Valid ≤ 99.5th percentile')   # 5 -> index 6 (theoretical)
    ]
    
    # Position legend to the right of the plot
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1.05, 0.5), 
             fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legend
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'combined_parameter_constraint_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined constraint analysis plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== COMBINED CONSTRAINT ANALYSIS SUMMARY ===")
    percentiles = [50, 60, 70, 80, 90, 99.5]
    
    # Count combinations by highest valid percentile
    unique_values, counts = np.unique(combined_matrix[combined_matrix >= -1], return_counts=True)
    total_valid_combinations = len(inference_range) * len(generation_range)
    
    print(f"Total parameter combinations analyzed: {total_valid_combinations}")
    for val, count in zip(unique_values, counts):
        if val == -1:
            print(f"  Invalid at all percentiles: {count} ({count/total_valid_combinations*100:.1f}%)")
        elif val == -2:
            print(f"  No data available: {count} ({count/total_valid_combinations*100:.1f}%)")
        else:
            percentile = percentiles[val]
            print(f"  Valid up to {percentile}th percentile: {count} ({count/total_valid_combinations*100:.1f}%)")
    
    # Show the most robust parameter combinations
    print("\nMost robust parameter combinations (valid at highest percentiles):")
    robust_combinations = []
    for i, inference_interval in enumerate(inference_range):
        for j, generation_length in enumerate(generation_range):
            validity_level = combined_matrix[i, j]
            if validity_level >= 3:  # Valid at 80th percentile or higher
                percentile = percentiles[validity_level] if validity_level >= 0 else "invalid"
                robust_combinations.append((inference_interval, generation_length, percentile, validity_level))
    
    # Sort by validity level (highest first)
    robust_combinations.sort(key=lambda x: x[3], reverse=True)
    
    for interval, gen_len, percentile, level in robust_combinations[:10]:  # Show top 10
        print(f"  ({interval}, {gen_len}): Valid up to {percentile}th percentile")

def main():
    """Main analysis function."""
    # Use local server data as it's most recent and comprehensive
    data_dir = "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/results_Stream_MUSE/bench_results/manual_local_server"
    output_dir = "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/results_Stream_MUSE/parameter_analysis"
    
    print("Loading benchmark data...")
    try:
        data = load_benchmark_data(data_dir)
        print(f"Loaded {len(data)} measurements from {len(data['generation_length'].unique())} generation lengths")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nCalculating round trip time percentiles by generation length...")
    percentile_data = calculate_percentiles_by_generation_length(data, percentiles=[50, 60, 70, 80, 90, 99.5])
    
    print("\nGenerating combined constraint analysis visualization...")
    plot_combined_constraint_analysis(percentile_data, output_dir)

if __name__ == "__main__":
    main()
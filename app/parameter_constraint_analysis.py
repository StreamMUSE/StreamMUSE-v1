#!/usr/bin/env python3
"""
Parameter Constraint Analysis for StreamMUSE

Analyzes the validity of inference interval and generation length parameter combinations
based on real-time musical timing constraints and measured round trip times.
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

def evaluate_constraints_detailed(inference_interval, generation_length, round_trip_time_ms):
    """
    Evaluate parameter constraints with detailed breakdown.
    
    Args:
        inference_interval: Inference interval in ticks
        generation_length: Generation length in ticks
        round_trip_time_ms: Round trip time in milliseconds
    
    Returns:
        tuple: (constraint1_satisfied, constraint2_satisfied, constraint_status)
        constraint_status: 0=both satisfied, 1=only C1 violated, 2=only C2 violated, 3=both violated
    """
    TICK_DURATION_MS = 125  # 125ms per tick at 120 BPM
    
    # Constraint 1: Inference interval must be long enough to avoid overlap
    # inference_interval_ticks * 125ms >= round_trip_time
    constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
    
    # Constraint 2: Generated music must arrive before it's needed
    # round_trip_time < (generation_length - inference_interval) * 125ms
    musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
    constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
    
    # Determine constraint status for coloring
    if constraint1_satisfied and constraint2_satisfied:
        constraint_status = 0  # Both satisfied (green)
    elif not constraint1_satisfied and constraint2_satisfied:
        constraint_status = 1  # Only constraint 1 violated (orange)
    elif constraint1_satisfied and not constraint2_satisfied:
        constraint_status = 2  # Only constraint 2 violated (yellow)
    else:
        constraint_status = 3  # Both violated (red)
    
    return constraint1_satisfied, constraint2_satisfied, constraint_status

def evaluate_constraints(inference_interval, generation_length, round_trip_time_ms):
    """
    Simple constraint evaluation for backward compatibility.
    """
    c1, c2, _ = evaluate_constraints_detailed(inference_interval, generation_length, round_trip_time_ms)
    return c1 and c2

def create_constraint_matrix_detailed(percentile_data, percentile, inference_range, generation_range):
    """
    Create a matrix showing detailed constraint status for parameter combinations.
    
    Args:
        percentile_data: Dictionary mapping (generation_length, percentile) to RTT
        percentile: Which percentile to use for this matrix
        inference_range: Range of inference intervals to test
        generation_range: Range of generation lengths to test
    
    Returns:
        2D numpy array with constraint status codes:
        0 = both satisfied (green), 1 = C1 violated (orange), 2 = C2 violated (yellow), 3 = both violated (red)
        -1 = no data (gray)
    """
    matrix = np.full((len(inference_range), len(generation_range)), -1, dtype=int)
    
    for i, inference_interval in enumerate(inference_range):
        for j, generation_length in enumerate(generation_range):
            # Get the RTT for this specific generation length and percentile
            if (generation_length, percentile) in percentile_data:
                rtt_ms = percentile_data[(generation_length, percentile)]
                _, _, constraint_status = evaluate_constraints_detailed(
                    inference_interval, generation_length, rtt_ms
                )
                matrix[i, j] = constraint_status
            else:
                # If we don't have data for this generation length, mark as no data
                matrix[i, j] = -1
    
    return matrix

def create_constraint_matrix(percentile_data, percentile, inference_range, generation_range):
    """
    Create a matrix showing constraint satisfaction for parameter combinations (backward compatibility).
    """
    detailed_matrix = create_constraint_matrix_detailed(percentile_data, percentile, inference_range, generation_range)
    return detailed_matrix == 0  # True only when both constraints satisfied

def plot_constraint_analysis(percentile_data, output_dir):
    """
    Create three side-by-side heatmaps showing parameter constraints for different percentiles.
    
    Args:
        percentile_data: Dictionary mapping (generation_length, percentile) to RTT in ms
        output_dir: Directory to save the plot
    """
    # Define parameter ranges (remove 0s as they don't make sense)
    inference_range = np.arange(1, 5)  # 1-4 inclusive (0 inference interval doesn't make sense)
    generation_range = np.arange(1, 9)  # 1-8 inclusive (0 generation length doesn't make sense)
    
    # Create figure with six subplots in 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Constraint Analysis: Inference Interval vs Generation Length\n' +
                'Green=Valid, Orange=C1 Violated, Yellow=C2 Violated, Red=Both Violated', 
                fontsize=20, fontweight='bold', y=0.95)
    
    percentiles = [50, 60, 70, 80, 90, 99.5]
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    # Define colors for different constraint statuses
    # Order must match: -1=no data (gray), 0=valid (green), 1=C1 violated (orange), 2=C2 violated (yellow), 3=both violated (red)
    # Since vmin=-1 and vmax=3, the colormap needs 5 colors in order: -1, 0, 1, 2, 3
    constraint_colors = ['#d3d3d3', '#90ee90', '#ffa500', '#ffff80', '#ff6b6b']  # Gray, Green, Orange, Yellow, Red
    
    for idx, percentile in enumerate(percentiles):
        ax = axes_flat[idx]
        
        # Create detailed constraint matrix showing which constraints are violated
        detailed_matrix = create_constraint_matrix_detailed(percentile_data, percentile, inference_range, generation_range)
        
        # Create custom colormap for constraint status
        from matplotlib.colors import ListedColormap
        constraint_cmap = ListedColormap(constraint_colors)
        
        # Create heatmap with square cells and constraint-specific colors
        sns.heatmap(detailed_matrix, 
                   ax=ax,
                   xticklabels=generation_range,
                   yticklabels=inference_range,
                   cmap=constraint_cmap,
                   vmin=-1,  # Gray for no data
                   vmax=3,   # Red for both violated
                   cbar=False,
                   linewidths=1.0,
                   linecolor='white',
                   square=True,  # Make cells square
                   annot=False)
        
        # Customize subplot
        ax.set_title(f'{percentile}th Percentile\n(Generation-Length-Specific RTT)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Generation Length (ticks)', fontsize=14, fontweight='bold')
        if idx in [0, 3]:  # Label y-axis on leftmost plots of each row
            ax.set_ylabel('Inference Interval (ticks)', fontsize=14, fontweight='bold')
        
        # Increase tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Invert y-axis so 0 is at bottom
        ax.invert_yaxis()
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    # Add a comprehensive legend for all constraint statuses
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=constraint_colors[1], label='Valid (Both Constraints Satisfied)'),  # Green
        Patch(facecolor=constraint_colors[2], label='Constraint 1 Violated (Interval Too Short)'),  # Orange
        Patch(facecolor=constraint_colors[3], label='Constraint 2 Violated (Buffer Too Small)'),  # Yellow
        Patch(facecolor=constraint_colors[4], label='Both Constraints Violated'),  # Red
        Patch(facecolor=constraint_colors[0], label='No Benchmark Data')  # Gray
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.18, hspace=0.3, wspace=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'parameter_constraint_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out to avoid display issues
    
    print(f"Constraint analysis plot saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== CONSTRAINT ANALYSIS SUMMARY ===")
    for percentile in percentiles:
        matrix = create_constraint_matrix(percentile_data, percentile, inference_range, generation_range)
        valid_combinations = np.sum(matrix)
        total_combinations = matrix.size
        valid_percentage = (valid_combinations / total_combinations) * 100
        
        print(f"{percentile}th Percentile (Generation-Length-Specific):")
        print(f"  Valid parameter combinations: {valid_combinations}/{total_combinations} ({valid_percentage:.1f}%)")
        
        # Show RTT values for available generation lengths (with frame info)
        print("  RTT values by generation length (ticks):")
        for gen_length in sorted(set(gen for gen, _ in percentile_data.keys())):
            if (gen_length, percentile) in percentile_data:
                rtt = percentile_data[(gen_length, percentile)]
                # Calculate corresponding frame value for display
                frames = (gen_length * 2) - 1  # Reverse conversion: frames = (ticks * 2) - 1
                print(f"    {gen_length} ticks ({frames} frames): {rtt:.1f}ms")
        print()

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
    
    print("Sample percentile values (showing frame->tick conversion):")
    for gen_length in sorted(data['generation_length'].unique())[:3]:  # Show first 3
        # Get corresponding frame value for display
        sample_row = data[data['generation_length'] == gen_length].iloc[0]
        gen_frames = sample_row['generation_length_frames'] if 'generation_length_frames' in sample_row else "N/A"
        print(f"  Generation {gen_length} ticks (from {gen_frames} frames):")
        for p in [50, 60, 70, 80, 90, 99.5]:
            if (gen_length, p) in percentile_data:
                rtt = percentile_data[(gen_length, p)]
                print(f"    {p}th percentile: {rtt:.1f}ms")
    
    print("\nGenerating constraint analysis visualization...")
    
    # Explain the data pattern with frame->tick conversion
    available_gen_lengths = sorted(data['generation_length'].unique())
    frame_mapping = {}
    for gen_length in available_gen_lengths:
        sample_row = data[data['generation_length'] == gen_length].iloc[0]
        gen_frames = sample_row['generation_length_frames'] if 'generation_length_frames' in sample_row else "N/A"
        frame_mapping[gen_length] = gen_frames
    
    print(f"Note: Benchmark data available for generation lengths (ticks): {available_gen_lengths}")
    print("Frame to tick conversion:")
    for ticks, frames in frame_mapping.items():
        print(f"  {frames} frames → {ticks} ticks")
    print("Data is available for consecutive tick values 1-8, so we should see a complete picture.")
    print()
    
    plot_constraint_analysis(percentile_data, output_dir)

if __name__ == "__main__":
    main()
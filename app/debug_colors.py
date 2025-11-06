#!/usr/bin/env python3
"""
Debug the color mapping to ensure constraint status matches colors correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap

def debug_color_mapping():
    """Debug the color mapping by creating a simple test visualization."""
    
    print("=== COLOR MAPPING DEBUG ===")
    
    # The constraint status codes
    constraint_statuses = {
        0: "Both satisfied (should be GREEN)",
        1: "C1 violated (should be ORANGE)", 
        2: "C2 violated (should be YELLOW)",
        3: "Both violated (should be RED)",
        -1: "No data (should be GRAY)"
    }
    
    # Current color definition
    constraint_colors = ['#90ee90', '#ffa500', '#ffff80', '#ff6b6b', '#d3d3d3']
    color_names = ['Light Green', 'Orange', 'Light Yellow', 'Light Red', 'Light Gray']
    
    print("Defined colors:")
    for i, (color, name) in enumerate(zip(constraint_colors, color_names)):
        status = i if i < 4 else -1
        print(f"  Status {status}: {color} ({name}) - {constraint_statuses.get(status, 'Unknown')}")
    
    # Create a test matrix to verify color mapping
    test_matrix = np.array([
        [0, 1, 2, 3, -1],  # One row with each status
        [3, 2, 1, 0, -1],  # Reverse order
    ])
    
    # Create the colormap
    constraint_cmap = ListedColormap(constraint_colors)
    
    # Create test visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    im = ax.imshow(test_matrix, cmap=constraint_cmap, vmin=-1, vmax=3, aspect='auto')
    
    # Add labels
    ax.set_title('Color Mapping Test\nEach column should show the constraint status color')
    ax.set_xlabel('Constraint Status')
    ax.set_ylabel('Test Rows')
    
    # Set tick labels
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0: Both OK\n(Green)', '1: C1 Violated\n(Orange)', '2: C2 Violated\n(Yellow)', 
                       '3: Both Violated\n(Red)', '-1: No Data\n(Gray)'])
    ax.set_yticks(range(2))
    ax.set_yticklabels(['Row 1', 'Row 2'])
    
    # Add text annotations
    for i in range(2):
        for j in range(5):
            status = test_matrix[i, j]
            ax.text(j, i, f'{status}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/results_Stream_MUSE/parameter_analysis/color_test.png', dpi=150)
    plt.close()
    
    print("\nColor test visualization saved to color_test.png")
    
    # Now test actual constraint evaluation
    print("\n=== TESTING ACTUAL CONSTRAINT EVALUATION ===")
    
    def evaluate_constraints_detailed(inference_interval, generation_length, round_trip_time_ms):
        TICK_DURATION_MS = 125
        constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
        musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
        constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
        
        if constraint1_satisfied and constraint2_satisfied:
            constraint_status = 0  # Both satisfied (green)
        elif not constraint1_satisfied and constraint2_satisfied:
            constraint_status = 1  # Only constraint 1 violated (orange)
        elif constraint1_satisfied and not constraint2_satisfied:
            constraint_status = 2  # Only constraint 2 violated (yellow)
        else:
            constraint_status = 3  # Both violated (red)
        
        return constraint1_satisfied, constraint2_satisfied, constraint_status
    
    # Test a few cases to see what statuses we actually get
    test_cases = [
        (1, 2, 77.7),   # Should be green (0)
        (1, 2, 200),    # Should be orange (1) - interval too short
        (3, 4, 200),    # Should be yellow (2) - buffer too small  
        (1, 2, 300),    # Should be red (3) - both violated
    ]
    
    print("Testing constraint evaluations:")
    for interval, gen_len, rtt in test_cases:
        c1, c2, status = evaluate_constraints_detailed(interval, gen_len, rtt)
        expected_color = color_names[status if status >= 0 else 4]
        print(f"  ({interval}, {gen_len}, {rtt}ms) → Status {status} ({expected_color}) - C1={c1}, C2={c2}")

if __name__ == "__main__":
    debug_color_mapping()
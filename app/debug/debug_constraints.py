#!/usr/bin/env python3
"""
Debug the constraint analysis to verify correct behavior.
"""

def debug_constraint_logic():
    """Debug a few specific parameter combinations to verify constraint logic."""
    
    # Sample RTT values from our analysis
    rtt_50th = {1: 77.7, 2: 110.6, 3: 155.4, 4: 201.9}
    rtt_90th = {1: 101.3, 2: 158.7, 3: 216.4, 4: 241.6}
    
    TICK_DURATION_MS = 125
    
    def evaluate_constraints_detailed(inference_interval, generation_length, round_trip_time_ms):
        """Copy of the constraint evaluation function for debugging."""
        # Constraint 1: inference_interval_ticks * 125ms >= round_trip_time
        constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
        
        # Constraint 2: round_trip_time < (generation_length - inference_interval) * 125ms
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
    
    print("=== CONSTRAINT LOGIC DEBUGGING ===")
    print()
    
    # Test cases that should show different constraint violations
    test_cases = [
        (1, 2, 50),  # Should be good at 50th percentile
        (1, 2, 90),  # Should have different constraints at 90th
        (1, 3, 50),  # Different generation length
        (2, 3, 50),  # Different interval
        (2, 4, 90),  # Should be borderline
    ]
    
    print("Testing specific parameter combinations:")
    print(f"{'Interval':<8} {'GenLen':<8} {'Percentile':<10} {'RTT':<8} {'C1':<8} {'C2':<8} {'Status':<8} {'Color'}")
    print("-" * 80)
    
    colors = ['Green', 'Orange', 'Yellow', 'Red']
    
    for interval, gen_len, percentile in test_cases:
        if percentile == 50:
            rtt_ms = rtt_50th.get(gen_len, 200)  # Default if not found
        else:
            rtt_ms = rtt_90th.get(gen_len, 300)  # Default if not found
            
        c1, c2, status = evaluate_constraints_detailed(interval, gen_len, rtt_ms)
        
        # Calculate the actual values for debugging
        c1_limit = interval * TICK_DURATION_MS
        c2_buffer = (gen_len - interval) * TICK_DURATION_MS
        
        print(f"{interval:<8} {gen_len:<8} {percentile}th{'':<6} {rtt_ms:<8.1f} {c1!s:<8} {c2!s:<8} {status:<8} {colors[status]}")
        print(f"         Details: C1={c1_limit}ms≥{rtt_ms}ms, C2={rtt_ms}ms<{c2_buffer}ms")
        print()
    
    print("=== EXPECTED DIFFERENCES BETWEEN PERCENTILES ===")
    print()
    print("Same parameters but different percentiles should show different results:")
    print()
    
    for interval in [1, 2]:
        for gen_len in [2, 3, 4]:
            if gen_len <= interval:
                continue  # Skip invalid combinations
                
            print(f"Parameter: (interval={interval}, gen={gen_len})")
            
            for perc_name, rtt_data in [("50th", rtt_50th), ("90th", rtt_90th)]:
                if gen_len in rtt_data:
                    rtt_ms = rtt_data[gen_len]
                    c1, c2, status = evaluate_constraints_detailed(interval, gen_len, rtt_ms)
                    print(f"  {perc_name}: RTT={rtt_ms:.1f}ms → {colors[status]} (C1={c1}, C2={c2})")
            print()

if __name__ == "__main__":
    debug_constraint_logic()
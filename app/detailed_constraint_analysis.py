#!/usr/bin/env python3
"""
Detailed Constraint Analysis for StreamMUSE

Provides detailed breakdown of which constraints are violated for each parameter combination.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def detailed_constraint_analysis():
    """Print detailed constraint analysis for key parameter combinations."""
    
    # Round trip times from our analysis
    percentile_data = {50: 237.8, 90: 404.9, 99.5: 509.6}
    TICK_DURATION_MS = 125
    
    print("\n=== DETAILED CONSTRAINT ANALYSIS ===")
    print("Constraint 1: inference_interval * 125ms >= round_trip_time")
    print("Constraint 2: round_trip_time < (generation_length - inference_interval) * 125ms")
    print()
    
    # Check some key parameter combinations
    test_combinations = [
        (1, 3), (1, 4), (1, 5),  # interval=1
        (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),  # interval=2
        (3, 6), (3, 7), (3, 8),  # interval=3
        (4, 8)  # interval=4
    ]
    
    for percentile, rtt_ms in percentile_data.items():
        print(f"\n--- {percentile}th Percentile (RTT = {rtt_ms:.1f}ms) ---")
        print(f"{'Interval':<8} {'GenLen':<8} {'Constraint1':<12} {'Constraint2':<12} {'Valid':<8} {'Reason'}")
        print("-" * 70)
        
        for interval, gen_length in test_combinations:
            # Constraint 1
            c1_threshold = interval * TICK_DURATION_MS
            c1_satisfied = c1_threshold >= rtt_ms
            
            # Constraint 2  
            musical_buffer = (gen_length - interval) * TICK_DURATION_MS
            c2_satisfied = rtt_ms < musical_buffer
            
            overall_valid = c1_satisfied and c2_satisfied
            
            # Determine reason for failure
            reason = ""
            if not c1_satisfied and not c2_satisfied:
                reason = "Both constraints violated"
            elif not c1_satisfied:
                reason = f"Interval too short (need ≥{rtt_ms/TICK_DURATION_MS:.1f})"
            elif not c2_satisfied:
                reason = f"Buffer too small ({musical_buffer:.0f}ms < {rtt_ms:.1f}ms)"
            else:
                reason = "Valid"
            
            print(f"{interval:<8} {gen_length:<8} {c1_satisfied!s:<12} {c2_satisfied!s:<12} {overall_valid!s:<8} {reason}")

def find_optimal_parameters():
    """Find the optimal parameter combinations for different use cases."""
    
    percentile_data = {50: 237.8, 90: 404.9, 99.5: 509.6}
    TICK_DURATION_MS = 125
    
    print("\n=== OPTIMAL PARAMETER RECOMMENDATIONS ===")
    
    for percentile, rtt_ms in percentile_data.items():
        print(f"\n{percentile}th Percentile (RTT = {rtt_ms:.1f}ms):")
        
        valid_combinations = []
        
        # Test all combinations in our range
        for interval in range(1, 5):  # 1-4 ticks
            for gen_length in range(interval + 1, 9):  # Must be > interval
                c1_satisfied = interval * TICK_DURATION_MS >= rtt_ms
                musical_buffer = (gen_length - interval) * TICK_DURATION_MS
                c2_satisfied = rtt_ms < musical_buffer
                
                if c1_satisfied and c2_satisfied:
                    # Calculate efficiency metrics
                    responsiveness = 1 / interval  # Lower interval = more responsive
                    efficiency = interval / gen_length  # Higher ratio = more efficient
                    buffer_margin = musical_buffer - rtt_ms  # Safety margin
                    
                    valid_combinations.append({
                        'interval': interval,
                        'gen_length': gen_length,
                        'responsiveness': responsiveness,
                        'efficiency': efficiency,
                        'buffer_margin': buffer_margin
                    })
        
        if valid_combinations:
            # Sort by responsiveness (prefer lower intervals)
            valid_combinations.sort(key=lambda x: (-x['responsiveness'], -x['buffer_margin']))
            
            print("  Valid combinations (sorted by responsiveness):")
            print(f"  {'Interval':<8} {'GenLen':<8} {'Responsiveness':<14} {'Efficiency':<10} {'Buffer Margin'}")
            print("  " + "-" * 60)
            
            for combo in valid_combinations[:5]:  # Show top 5
                print(f"  {combo['interval']:<8} {combo['gen_length']:<8} "
                      f"{combo['responsiveness']:.3f}{'':10} {combo['efficiency']:.3f}{'':6} "
                      f"{combo['buffer_margin']:.1f}ms")
        else:
            print("  No valid combinations found!")

if __name__ == "__main__":
    detailed_constraint_analysis()
    find_optimal_parameters()
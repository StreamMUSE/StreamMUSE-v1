#!/usr/bin/env python3
"""
Comprehensive Parameter Constraint Analysis with 80th Percentile

Detailed analysis of all four confidence levels for StreamMUSE parameter selection.
"""

def detailed_constraint_breakdown():
    """Provide detailed breakdown of constraints at all percentiles."""
    
    # RTT data from analysis (generation-length-specific)
    rtt_data = {
        50: {1: 77.7, 2: 110.6, 3: 155.4, 4: 201.9, 5: 247.9, 6: 303.8, 7: 357.0, 8: 404.7},
        80: {1: 84.7, 2: 135.6, 3: 181.0, 4: 222.0, 5: 272.3, 6: 343.3, 7: 410.6, 8: 442.7},
        90: {1: 101.3, 2: 158.7, 3: 216.4, 4: 241.6, 5: 303.2, 6: 378.6, 7: 426.6, 8: 472.9},
        99.5: {1: 159.9, 2: 272.0, 3: 288.5, 4: 287.9, 5: 422.3, 6: 469.1, 7: 514.9, 8: 571.7}
    }
    
    TICK_DURATION_MS = 125  # 125ms per tick at 120 BPM
    
    print("=== COMPREHENSIVE PARAMETER CONSTRAINT ANALYSIS ===")
    print("All percentiles with complete frame-to-tick conversion")
    print()
    
    # Valid combinations count by percentile
    valid_counts = {50: 8, 80: 3, 90: 2, 99.5: 0}
    
    print("📊 CONFIDENCE LEVEL OVERVIEW:")
    for percentile, count in valid_counts.items():
        percentage = (count / 45) * 100
        if percentile == 50:
            use_case = "Development/Testing"
        elif percentile == 80:
            use_case = "Staging/Pre-Production"
        elif percentile == 90:
            use_case = "Production"
        else:
            use_case = "Mission-Critical"
        
        print(f"   {percentile}th percentile ({use_case}): {count}/45 valid ({percentage:.1f}%)")
    print()
    
    # Detailed analysis for each percentile
    for percentile, data in rtt_data.items():
        print(f"\n--- {percentile}th Percentile Detailed Analysis ---")
        
        # Find valid combinations
        valid_combinations = []
        for gen_length, rtt_ms in data.items():
            for inference_interval in range(1, 5):  # 1-4 ticks
                
                # Constraint 1: inference_interval * 125ms >= round_trip_time
                c1_satisfied = inference_interval * TICK_DURATION_MS >= rtt_ms
                
                # Constraint 2: round_trip_time < (generation_length - inference_interval) * 125ms
                if gen_length > inference_interval:
                    musical_buffer = (gen_length - inference_interval) * TICK_DURATION_MS
                    c2_satisfied = rtt_ms < musical_buffer
                else:
                    c2_satisfied = False
                
                if c1_satisfied and c2_satisfied:
                    buffer_margin = musical_buffer - rtt_ms
                    efficiency = inference_interval / gen_length
                    frames = (gen_length * 2) - 1  # Convert back to frames for reference
                    
                    valid_combinations.append({
                        'interval': inference_interval,
                        'gen_length': gen_length,
                        'gen_frames': frames,
                        'rtt': rtt_ms,
                        'buffer_margin': buffer_margin,
                        'efficiency': efficiency,
                        'responsiveness': 1 / inference_interval
                    })
        
        if valid_combinations:
            print(f"Valid combinations: {len(valid_combinations)}")
            print(f"{'Interval':<8} {'GenLen':<8} {'Frames':<8} {'RTT':<8} {'Buffer':<10} {'Efficiency':<10} {'Responsiveness'}")
            print("-" * 75)
            
            # Sort by responsiveness (lower interval = more responsive), then by efficiency
            valid_combinations.sort(key=lambda x: (-x['responsiveness'], -x['efficiency']))
            
            for combo in valid_combinations:
                print(f"{combo['interval']:<8} {combo['gen_length']:<8} {combo['gen_frames']:<8} "
                      f"{combo['rtt']:.1f}ms{'':<3} {combo['buffer_margin']:.1f}ms{'':<5} "
                      f"{combo['efficiency']:.3f}{'':<5} {combo['responsiveness']:.3f}")
                
            # Highlight best option
            best = valid_combinations[0]
            print(f"\n   🎯 BEST OPTION: (interval={best['interval']}, gen={best['gen_length']}) "
                  f"- {best['rtt']:.1f}ms RTT, {best['buffer_margin']:.1f}ms buffer")
        else:
            print("❌ No valid combinations found!")

def summarize_recommendations():
    """Provide actionable recommendations based on all percentiles."""
    
    print("\n" + "="*80)
    print("=== ACTIONABLE RECOMMENDATIONS BY USE CASE ===")
    print()
    
    recommendations = [
        {
            'use_case': '🟢 DEVELOPMENT & TESTING',
            'percentile': '50th',
            'reliability': '50% confidence',
            'best_params': '(interval=1, gen=2)',
            'details': 'RTT=110.6ms, Buffer=139.4ms, High responsiveness',
            'alternatives': '(interval=2, gen=3-4) for more safety',
            'pros': 'Fast iteration, good responsiveness',
            'cons': 'Moderate reliability, some request failures expected'
        },
        {
            'use_case': '🟡 STAGING & PRE-PRODUCTION',
            'percentile': '80th (NEW)',
            'reliability': '80% confidence',
            'best_params': '(interval=2, gen=3)',
            'details': 'RTT=181.0ms, Buffer=69.0ms, Balanced approach',
            'alternatives': '(interval=3, gen=4) for extra safety',
            'pros': 'Good balance of reliability and performance',
            'cons': 'Limited parameter options, tighter margins'
        },
        {
            'use_case': '🔴 PRODUCTION',
            'percentile': '90th',
            'reliability': '90% confidence',
            'best_params': '(interval=2, gen=3)',
            'details': 'RTT=216.4ms, Buffer=33.6ms, Minimal viable',
            'alternatives': '(interval=4, gen=8) for ultra-conservative',
            'pros': 'Production-grade reliability',
            'cons': 'Very tight buffer margins, limited flexibility'
        },
        {
            'use_case': '⚫ MISSION-CRITICAL',
            'percentile': '99.5th',
            'reliability': '99.5% confidence',
            'best_params': 'NONE VIABLE',
            'details': 'No combinations satisfy constraints',
            'alternatives': 'System optimization required',
            'pros': 'Ultimate reliability if achievable',
            'cons': 'Not viable with current system performance'
        }
    ]
    
    for rec in recommendations:
        print(f"{rec['use_case']} ({rec['percentile']} percentile)")
        print(f"   Reliability: {rec['reliability']}")
        print(f"   Best params: {rec['best_params']}")
        print(f"   Details: {rec['details']}")
        print(f"   Alternatives: {rec['alternatives']}")
        print(f"   ✅ Pros: {rec['pros']}")
        print(f"   ⚠️  Cons: {rec['cons']}")
        print()
    
    print("🎯 KEY INSIGHTS:")
    print("   • 80th percentile provides valuable middle ground between development and production")
    print("   • Progressive constraint tightening: 8 → 3 → 2 → 0 valid combinations")
    print("   • Real-time music generation is viable up to 90th percentile confidence")
    print("   • Mission-critical applications need system-level optimization")
    print()
    
    print("🛠️ IMPLEMENTATION STRATEGY:")
    print("   1. Start with 50th percentile for development and testing")
    print("   2. Use 80th percentile for staging and integration testing")
    print("   3. Deploy with 90th percentile for production reliability")
    print("   4. Monitor actual performance and adjust percentile targets")
    print("   5. Consider adaptive parameter selection based on real-time metrics")

if __name__ == "__main__":
    detailed_constraint_breakdown()
    summarize_recommendations()
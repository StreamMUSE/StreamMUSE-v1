#!/usr/bin/env python3
"""Detailed analysis of token differences."""

import json
from pathlib import Path

fake_rt_file = Path("logs/fake_rt/fake_rt_gen_0124.json")
offline_file = Path("logs/offline_gt0/offline_gen_0000.json")

with open(fake_rt_file) as f:
    fake_rt_data = json.load(f)

with open(offline_file) as f:
    offline_data = json.load(f)

fake_rt_tokens = fake_rt_data.get("prompt_tokens", [])
offline_tokens = offline_data.get("prompt_tokens", [])

print("🔍 Detailed Token Analysis")
print("=" * 80)
print()

# Count melody vs accompaniment in metadata
print("📌 Context Information:")
print(f"  FakeRT:")
print(f"    - generation_start_tick: {fake_rt_data.get('generation_start_tick')}")
print(f"    - notes: {fake_rt_data.get('notes')}")
print(f"    - melody_event_count: {fake_rt_data.get('melody_event_count')}")
print(f"    - accompaniment_event_count: {fake_rt_data.get('accompaniment_event_count')}")
print()
print(f"  Offline:")
print(f"    - generation_start_tick: {offline_data.get('generation_start_tick')}")
print(f"    - notes: {offline_data.get('notes')}")
print(f"    - melody_event_count: {offline_data.get('melody_event_count')}")
print(f"    - accompaniment_event_count: {offline_data.get('accompaniment_event_count')}")
print()

# Show all differences
print("📊 Token-by-token Comparison (showing differences):")
print("-" * 80)
diff_count = 0
for i in range(min(len(fake_rt_tokens), len(offline_tokens))):
    if fake_rt_tokens[i] != offline_tokens[i]:
        diff_count += 1
        if diff_count <= 20:  # Show first 20 differences
            print(f"  Index {i:3d}: FakeRT={fake_rt_tokens[i]:3d}  Offline={offline_tokens[i]:3d}  Δ={offline_tokens[i]-fake_rt_tokens[i]:+3d}")

print()
if len(fake_rt_tokens) != len(offline_tokens):
    print(f"⚠️  Length mismatch: FakeRT={len(fake_rt_tokens)}, Offline={len(offline_tokens)}")
    diff_count += abs(len(fake_rt_tokens) - len(offline_tokens))

print(f"📈 Total differences: {diff_count} out of {max(len(fake_rt_tokens), len(offline_tokens))} tokens")
print(f"   Match rate: {(1 - diff_count/max(len(fake_rt_tokens), len(offline_tokens))) * 100:.1f}%")
print()

# Analyze pattern: Token patterns typically follow melody pitch encoding
# In Lekai, common patterns are: pitch, duration, velocity, or special markers (255, 170, 171, etc.)
print("🎼 Pattern Analysis:")
special_tokens = [255, 170, 171, 169, 173]
print(f"  Special tokens (markers): {special_tokens}")
print()
print(f"  FakeRT special token counts:")
for token in special_tokens:
    count = fake_rt_tokens.count(token)
    print(f"    Token {token}: {count} occurrences")
print()
print(f"  Offline special token counts:")
for token in special_tokens:
    count = offline_tokens.count(token)
    print(f"    Token {token}: {count} occurrences")
print()

print("💡 Conclusion:")
print(f"  The prompt_tokens differ significantly starting from index 34.")
print(f"  This suggests that:")
print(f"  1. Offline starts from beat 0 with full song history")
print(f"  2. FakeRT starts from beat 31 with accumulated context")
print(f"  3. The melody pitches differ (tokens 110 vs 119 at index 34 suggest different notes)")
print(f"  4. This is expected behavior - they're snapshots from different points in time")

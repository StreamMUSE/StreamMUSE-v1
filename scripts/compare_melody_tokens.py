#!/usr/bin/env python3
"""Compare melody tokens between fake_rt and offline generations."""

import json
from pathlib import Path

# Load both JSON files
fake_rt_file = Path("logs/fake_rt/fake_rt_gen_0124.json")
offline_file = Path("logs/offline_gt0/offline_gen_0000.json")

print(f"📖 Loading files...")
print(f"  FakeRT: {fake_rt_file}")
print(f"  Offline: {offline_file}")
print()

with open(fake_rt_file) as f:
    fake_rt_data = json.load(f)

with open(offline_file) as f:
    offline_data = json.load(f)

# Extract prompt tokens
fake_rt_tokens = fake_rt_data.get("prompt_tokens", [])
offline_tokens = offline_data.get("prompt_tokens", [])

print(f"📊 Token counts:")
print(f"  FakeRT prompt_tokens length: {len(fake_rt_tokens)}")
print(f"  Offline prompt_tokens length: {len(offline_tokens)}")
print()

# Extract metadata
print(f"📋 Metadata:")
print(f"  FakeRT generation_start_tick: {fake_rt_data.get('generation_start_tick')}")
print(f"  FakeRT notes: {fake_rt_data.get('notes')}")
print(f"  Offline generation_start_tick: {offline_data.get('generation_start_tick')}")
print(f"  Offline notes: {offline_data.get('notes')}")
print()

# Print first 50 tokens from each
print(f"🎼 First 50 prompt tokens:")
print(f"  FakeRT:  {fake_rt_tokens[:50]}")
print(f"  Offline: {offline_tokens[:50]}")
print()

# Check if they're exactly the same
if fake_rt_tokens == offline_tokens:
    print("✅ Prompt tokens are IDENTICAL!")
else:
    print("❌ Prompt tokens are DIFFERENT")
    
    # Find where they start differing
    for i, (f, o) in enumerate(zip(fake_rt_tokens, offline_tokens)):
        if f != o:
            print(f"   First difference at index {i}: fake_rt={f}, offline={o}")
            # Show context around difference
            start = max(0, i - 3)
            end = min(len(fake_rt_tokens), i + 4)
            print(f"   FakeRT context [{start}:{end}]: {fake_rt_tokens[start:end]}")
            print(f"   Offline context [{start}:{end}]: {offline_tokens[start:end]}")
            break
    
    # Check if offline tokens are contained in fake_rt
    if offline_tokens[0] in fake_rt_tokens:
        idx = fake_rt_tokens.index(offline_tokens[0])
        print(f"   Offline token sequence starts in FakeRT at index {idx}")
        print(f"   FakeRT[{idx}:{idx+len(offline_tokens)}]: {fake_rt_tokens[idx:idx+len(offline_tokens)]}")
        print(f"   Offline[0:{len(offline_tokens)}]:          {offline_tokens[0:len(offline_tokens)]}")
        if fake_rt_tokens[idx:idx+len(offline_tokens)] == offline_tokens:
            print("   ✅ Offline tokens are a subsequence of FakeRT!")
        else:
            print("   ❌ Different even at matching starting point")

print()
print("🎯 Conclusion:")
print(f"  {'Melody is compatible' if fake_rt_tokens == offline_tokens else 'Melody differs - check if FakeRT context is accumulated differently'}")

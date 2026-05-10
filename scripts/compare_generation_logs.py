#!/usr/bin/env python3
"""
对比 Fake Realtime 和 Offline 的生成日志
用于分析两种模式的一致性

用法:
    python scripts/compare_generation_logs.py \
        --fake-rt-log logs/generation/fake_rt_gen_0000.json \
        --offline-log logs/generation/offline_gen_0000.json

或者对比两个目录:
    python scripts/compare_generation_logs.py \
        --fake-rt-dir logs/fake_rt/ \
        --offline-dir logs/offline/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_log(log_path: str) -> Dict[str, Any]:
    """加载日志文件"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_token_sequences(fake_rt_tokens: List[int], offline_tokens: List[int]) -> Dict[str, Any]:
    """对比两个 token sequence"""
    result = {
        "fake_rt_length": len(fake_rt_tokens),
        "offline_length": len(offline_tokens),
        "length_match": len(fake_rt_tokens) == len(offline_tokens),
    }
    
    # 找到第一个不匹配的位置
    min_len = min(len(fake_rt_tokens), len(offline_tokens))
    mismatches = []
    
    for i in range(min_len):
        if fake_rt_tokens[i] != offline_tokens[i]:
            mismatches.append({
                "position": i,
                "fake_rt": fake_rt_tokens[i],
                "offline": offline_tokens[i],
            })
            if len(mismatches) >= 10:  # 最多显示10个不匹配
                break
    
    result["mismatches"] = mismatches
    result["mismatch_count"] = len(mismatches)
    result["prefix_match_length"] = next(
        (i for i in range(min_len) if fake_rt_tokens[i] != offline_tokens[i]),
        min_len
    )
    
    # 特殊 token 分析
    result["fake_rt_special_tokens"] = analyze_special_tokens(fake_rt_tokens)
    result["offline_special_tokens"] = analyze_special_tokens(offline_tokens)
    
    return result


def analyze_special_tokens(tokens: List[int]) -> Dict[str, Any]:
    """分析特殊 token 的分布"""
    special_tokens = {
        "bos": 1,
        "bar": 255,
        "pad": 173,
        "mel_end": 170,
        "acc_end": 171,
        "acc_empty": 169,
    }
    
    result = {}
    for name, token_id in special_tokens.items():
        count = sum(1 for t in tokens if t == token_id)
        positions = [i for i, t in enumerate(tokens) if t == token_id]
        result[name] = {"count": count, "positions": positions[:20]}  # 最多显示20个位置
    
    # 分析前几个 token
    result["first_20_tokens"] = tokens[:20]
    
    return result


def compare_logs(fake_rt_log: Dict[str, Any], offline_log: Dict[str, Any]) -> Dict[str, Any]:
    """对比两个完整的日志"""
    comparison = {
        "general": {
            "fake_rt_file": fake_rt_log.get("input_file", "unknown"),
            "offline_file": offline_log.get("input_file", "unknown"),
            "fake_rt_timestamp": fake_rt_log.get("timestamp"),
            "offline_timestamp": offline_log.get("timestamp"),
        },
        "generation_params": {
            "temperature_match": fake_rt_log.get("temperature") == offline_log.get("temperature"),
            "top_k_match": fake_rt_log.get("top_k") == offline_log.get("top_k"),
            "top_p_match": fake_rt_log.get("top_p") == offline_log.get("top_p"),
            "repetition_penalty_match": fake_rt_log.get("repetition_penalty") == offline_log.get("repetition_penalty"),
            "fake_rt_params": {
                "temperature": fake_rt_log.get("temperature"),
                "top_k": fake_rt_log.get("top_k"),
                "top_p": fake_rt_log.get("top_p"),
                "repetition_penalty": fake_rt_log.get("repetition_penalty"),
            },
            "offline_params": {
                "temperature": offline_log.get("temperature"),
                "top_k": offline_log.get("top_k"),
                "top_p": offline_log.get("top_p"),
                "repetition_penalty": offline_log.get("repetition_penalty"),
            },
        },
        "events": {
            "fake_rt_melody_count": fake_rt_log.get("melody_event_count", 0),
            "offline_melody_count": offline_log.get("melody_event_count", 0),
            "fake_rt_acc_count": fake_rt_log.get("accompaniment_event_count", 0),
            "offline_acc_count": offline_log.get("accompaniment_event_count", 0),
        },
    }
    
    # 对比 token sequences
    fake_rt_tokens = fake_rt_log.get("prompt_tokens", [])
    offline_tokens = offline_log.get("prompt_tokens", [])
    comparison["tokens"] = compare_token_sequences(fake_rt_tokens, offline_tokens)
    
    return comparison


def print_comparison(comparison: Dict[str, Any], verbose: bool = False) -> None:
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("GENERATION LOG COMPARISON")
    print("=" * 80)
    
    # 基本信息
    print("\n[General]")
    print(f"  Fake RT file:  {comparison['general']['fake_rt_file']}")
    print(f"  Offline file:  {comparison['general']['offline_file']}")
    
    # 生成参数
    print("\n[Generation Parameters]")
    params = comparison['generation_params']
    print(f"  Temperature match:      {params['temperature_match']}")
    print(f"  Top-k match:            {params['top_k_match']}")
    print(f"  Top-p match:            {params['top_p_match']}")
    print(f"  Repetition penalty match: {params['repetition_penalty_match']}")
    
    if verbose:
        print(f"  Fake RT params:  {params['fake_rt_params']}")
        print(f"  Offline params:  {params['offline_params']}")
    
    # 事件数量
    print("\n[Event Counts]")
    events = comparison['events']
    print(f"  Melody events:     FakeRT={events['fake_rt_melody_count']}, Offline={events['offline_melody_count']}")
    print(f"  Acc events:        FakeRT={events['fake_rt_acc_count']}, Offline={events['offline_acc_count']}")
    
    # Token 对比
    print("\n[Token Sequence Comparison]")
    tokens = comparison['tokens']
    print(f"  Fake RT length:    {tokens['fake_rt_length']}")
    print(f"  Offline length:    {tokens['offline_length']}")
    print(f"  Length match:      {tokens['length_match']}")
    print(f"  Prefix match len:  {tokens['prefix_match_length']}")
    print(f"  Mismatch count:    {tokens['mismatch_count']}")
    
    if tokens['mismatches']:
        print("\n  First 10 mismatches:")
        for m in tokens['mismatches']:
            print(f"    Position {m['position']}: FakeRT={m['fake_rt']}, Offline={m['offline']}")
    
    # 特殊 token 分析
    print("\n[Special Token Analysis]")
    print("  Fake RT first 20 tokens:")
    print(f"    {tokens['fake_rt_special_tokens']['first_20_tokens']}")
    print("  Offline first 20 tokens:")
    print(f"    {tokens['offline_special_tokens']['first_20_tokens']}")
    
    if verbose:
        print("\n  Bar token positions (first 20):")
        print(f"    FakeRT:  {tokens['fake_rt_special_tokens']['bar']['positions']}")
        print(f"    Offline: {tokens['offline_special_tokens']['bar']['positions']}")
        
        print("\n  Pad token positions (first 20):")
        print(f"    FakeRT:  {tokens['fake_rt_special_tokens']['pad']['positions']}")
        print(f"    Offline: {tokens['offline_special_tokens']['pad']['positions']}")
    
    # 总结
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_match = (
        params['temperature_match'] and
        params['top_k_match'] and
        params['top_p_match'] and
        tokens['length_match'] and
        tokens['mismatch_count'] == 0
    )
    
    if all_match:
        print("✓ All checks passed! Logs match perfectly.")
    else:
        print("✗ Differences found:")
        if not params['temperature_match']:
            print("  - Temperature mismatch")
        if not params['top_k_match']:
            print("  - Top-k mismatch")
        if not params['top_p_match']:
            print("  - Top-p mismatch")
        if not tokens['length_match']:
            print("  - Token sequence length mismatch")
        if tokens['mismatch_count'] > 0:
            print(f"  - {tokens['mismatch_count']} token mismatches")
    
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FakeRT and Offline generation logs")
    parser.add_argument("--fake-rt-log", type=str, help="Path to FakeRT log JSON")
    parser.add_argument("--offline-log", type=str, help="Path to Offline log JSON")
    parser.add_argument("--fake-rt-dir", type=str, help="Directory containing FakeRT logs")
    parser.add_argument("--offline-dir", type=str, help="Directory containing Offline logs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output", type=str, help="Output comparison result to JSON file")
    args = parser.parse_args()
    
    # 单文件对比模式
    if args.fake_rt_log and args.offline_log:
        fake_rt_log = load_log(args.fake_rt_log)
        offline_log = load_log(args.offline_log)
        comparison = compare_logs(fake_rt_log, offline_log)
        print_comparison(comparison, verbose=args.verbose)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2)
            print(f"Comparison saved to: {args.output}")
        
        return 0 if comparison['tokens']['mismatch_count'] == 0 else 1
    
    # 目录对比模式
    elif args.fake_rt_dir and args.offline_dir:
        fake_rt_dir = Path(args.fake_rt_dir)
        offline_dir = Path(args.offline_dir)
        
        fake_rt_logs = sorted(fake_rt_dir.glob("*_gen_*.json"))
        offline_logs = sorted(offline_dir.glob("*_gen_*.json"))
        
        print(f"Found {len(fake_rt_logs)} FakeRT logs and {len(offline_logs)} Offline logs")
        
        all_comparisons = []
        all_passed = True
        
        for i, (frt, off) in enumerate(zip(fake_rt_logs, offline_logs)):
            print(f"\nComparing pair {i+1}: {frt.name} vs {off.name}")
            fake_rt_log = load_log(str(frt))
            offline_log = load_log(str(off))
            comparison = compare_logs(fake_rt_log, offline_log)
            print_comparison(comparison, verbose=args.verbose)
            all_comparisons.append({
                "fake_rt": str(frt),
                "offline": str(off),
                "comparison": comparison,
            })
            if comparison['tokens']['mismatch_count'] > 0:
                all_passed = False
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "total_pairs": len(all_comparisons),
                        "all_passed": all_passed,
                    },
                    "comparisons": all_comparisons,
                }, f, indent=2)
            print(f"All comparisons saved to: {args.output}")
        
        return 0 if all_passed else 1
    
    else:
        print("Error: Must specify either --fake-rt-log + --offline-log or --fake-rt-dir + --offline-dir")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

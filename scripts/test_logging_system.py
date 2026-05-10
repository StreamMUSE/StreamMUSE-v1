#!/usr/bin/env python3
"""
测试日志系统是否正常工作

用法:
    # 测试 FakeRT 日志
    LEKAI_RT_TOP_K=1 uv run python scripts/test_logging_system.py --mode fake_rt

    # 测试 Offline 日志
    uv run python scripts/test_logging_system.py --mode offline
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streammuse.infrastructure.inference.generation_logger import GenerationLogger, compare_logs


def test_generation_logger() -> bool:
    """测试 GenerationLogger 基本功能"""
    print("\n" + "=" * 60)
    print("Testing GenerationLogger")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 logger
        logger = GenerationLogger(output_dir=tmpdir, mode="fake_rt")
        
        # 模拟一次生成
        log_path = logger.log_generation(
            input_file="test.mid",
            generation_start_tick=16,
            generation_length_frames=4,
            prompt_tokens=[1, 4, 5, 173, 255, 255, 10, 20, 30, 170, 40, 50, 171],
            temperature=0.8,
            top_k=1,
            top_p=0.95,
            repetition_penalty=1.2,
            melody_events=[
                {"type": "note_on", "pitch": 60, "tick": 0},
                {"type": "note_off", "pitch": 60, "tick": 4},
            ],
            accompaniment_events=[
                {"type": "note_on", "pitch": 48, "tick": 16, "velocity": 80},
            ],
            bpm=120,
            notes="test generation",
        )
        
        # 验证文件是否创建
        if not os.path.exists(log_path):
            print(f"✗ Log file not created: {log_path}")
            return False
        
        print(f"✓ Log file created: {log_path}")
        
        # 验证文件内容
        import json
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        
        required_fields = [
            "mode", "timestamp", "input_file", "generation_start_tick",
            "prompt_tokens", "prompt_token_count", "temperature", "top_k",
            "melody_events", "accompaniment_events"
        ]
        
        for field in required_fields:
            if field not in log_data:
                print(f"✗ Missing field: {field}")
                return False
            print(f"✓ Field '{field}' present")
        
        # 测试 summary 保存
        summary_path = logger.save_summary()
        if not os.path.exists(summary_path):
            print(f"✗ Summary file not created: {summary_path}")
            return False
        
        print(f"✓ Summary file created: {summary_path}")
    
    print("\n✓ All GenerationLogger tests passed!")
    return True


def test_compare_logs() -> bool:
    """测试 compare_logs 功能"""
    print("\n" + "=" * 60)
    print("Testing compare_logs")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建两个相似的日志
        logger1 = GenerationLogger(output_dir=tmpdir, mode="fake_rt")
        logger2 = GenerationLogger(output_dir=tmpdir, mode="offline")
        
        # 相同的 tokens
        tokens = [1, 4, 5, 173, 255, 255, 10, 20, 30, 170]
        
        log1_path = logger1.log_generation(
            input_file="test1.mid",
            generation_start_tick=16,
            generation_length_frames=4,
            prompt_tokens=tokens,
            temperature=0.8,
            top_k=1,
            top_p=0.95,
            repetition_penalty=1.2,
            melody_events=[{"type": "note_on", "pitch": 60, "tick": 0}],
            accompaniment_events=[{"type": "note_on", "pitch": 48, "tick": 16}],
        )
        
        log2_path = logger2.log_generation(
            input_file="test2.mid",
            generation_start_tick=16,
            generation_length_frames=4,
            prompt_tokens=tokens,  # 相同的 tokens
            temperature=0.8,
            top_k=1,
            top_p=0.95,
            repetition_penalty=1.2,
            melody_events=[{"type": "note_on", "pitch": 60, "tick": 0}],
            accompaniment_events=[{"type": "note_on", "pitch": 48, "tick": 16}],
        )
        
        # 对比
        comparison = compare_logs(log1_path, log2_path)
        
        if comparison["prompt_comparison"]["tokens_match"]:
            print("✓ Token sequences match")
        else:
            print(f"✗ Token sequences mismatch: {comparison['prompt_comparison']['mismatches']}")
            return False
        
        # 测试不匹配的 tokens
        logger3 = GenerationLogger(output_dir=tmpdir, mode="offline")
        log3_path = logger3.log_generation(
            input_file="test3.mid",
            generation_start_tick=16,
            generation_length_frames=4,
            prompt_tokens=[1, 4, 5, 999, 255],  # 不同的 token
            temperature=0.8,
            top_k=1,
            top_p=0.95,
            repetition_penalty=1.2,
            melody_events=[],
            accompaniment_events=[],
        )
        
        comparison2 = compare_logs(log1_path, log3_path)
        
        if not comparison2["prompt_comparison"]["tokens_match"]:
            print("✓ Mismatch detection works")
        else:
            print("✗ Should have detected mismatch")
            return False
        
        if len(comparison2["prompt_comparison"]["mismatches"]) > 0:
            print(f"✓ Found {len(comparison2['prompt_comparison']['mismatches'])} mismatches")
        else:
            print("✗ Should have found mismatches")
            return False
    
    print("\n✓ All compare_logs tests passed!")
    return True


def test_environment_variables() -> bool:
    """测试环境变量配置"""
    print("\n" + "=" * 60)
    print("Testing Environment Variables")
    print("=" * 60)
    
    # 检查 FakeRT 环境变量
    rt_vars = [
        "LEKAI_RT_TEMPERATURE",
        "LEKAI_RT_TOP_K",
        "LEKAI_RT_TOP_P",
        "LEKAI_RT_REPETITION_PENALTY",
        "LEKAI_RT_LOG_DIR",
    ]
    
    print("\nFakeRT Environment Variables:")
    for var in rt_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}={value}")
        else:
            print(f"  {var}=<not set>")
    
    # 检查 Offline 环境变量
    offline_vars = [
        "LEKAI_OFFLINE_LOG_DIR",
    ]
    
    print("\nOffline Environment Variables:")
    for var in offline_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}={value}")
        else:
            print(f"  {var}=<not set>")
    
    print("\n✓ Environment variable check completed")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test logging system")
    parser.add_argument("--mode", choices=["all", "logger", "compare", "env"], default="all",
                       help="Which tests to run")
    args = parser.parse_args()
    
    all_passed = True
    
    if args.mode in ("all", "logger"):
        if not test_generation_logger():
            all_passed = False
    
    if args.mode in ("all", "compare"):
        if not test_compare_logs():
            all_passed = False
    
    if args.mode in ("all", "env"):
        if not test_environment_variables():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

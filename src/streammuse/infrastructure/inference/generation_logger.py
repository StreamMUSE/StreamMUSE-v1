"""
Generation Logger - 记录 Fake Realtime 和 Offline 的生成过程
用于对比两种模式的一致性
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class GenerationLog:
    """单次生成的完整日志"""
    
    # 基本信息
    mode: str  # "fake_rt" 或 "offline"
    timestamp: str
    input_file: str
    generation_start_tick: int
    generation_length_frames: int
    
    # Prompt 信息
    prompt_tokens: List[int]
    prompt_token_count: int
    
    # 生成参数
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    
    # Melody Events (输入)
    melody_events: List[Dict[str, Any]] = field(default_factory=list)
    melody_event_count: int = 0
    
    # Accompaniment Events (输出)
    accompaniment_events: List[Dict[str, Any]] = field(default_factory=list)
    accompaniment_event_count: int = 0
    
    # 额外信息
    bpm: Optional[int] = None
    notes: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, output_dir: str, suffix: str = "") -> str:
        """保存为 JSON 文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 文件名格式: {mode}_gen_{start_tick}_{suffix}.json
        filename = f"{self.mode}_gen_{self.generation_start_tick:04d}{suffix}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)


class GenerationLogger:
    """生成过程日志记录器"""
    
    def __init__(self, output_dir: str, mode: str):
        """
        Args:
            output_dir: 日志输出目录
            mode: "fake_rt" 或 "offline"
        """
        self.output_dir = output_dir
        self.mode = mode
        self.logs: List[GenerationLog] = []
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def log_generation(
        self,
        input_file: str,
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_tokens: List[int],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        melody_events: List[Dict[str, Any]],
        accompaniment_events: List[Dict[str, Any]],
        bpm: Optional[int] = None,
        notes: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
        suffix: str = "",
    ) -> str:
        """
        记录一次生成过程
        
        Returns:
            保存的日志文件路径
        """
        log = GenerationLog(
            mode=self.mode,
            timestamp=datetime.now().isoformat(),
            input_file=input_file,
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
            prompt_tokens=prompt_tokens,
            prompt_token_count=len(prompt_tokens),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            melody_events=sorted(melody_events, key=lambda x: x.get('tick', 0)),
            melody_event_count=len(melody_events),
            accompaniment_events=sorted(accompaniment_events, key=lambda x: x.get('tick', 0)),
            accompaniment_event_count=len(accompaniment_events),
            bpm=bpm,
            notes=notes,
            diagnostics=dict(diagnostics or {}),
        )
        
        self.logs.append(log)
        
        # 保存到文件
        filepath = log.save(self.output_dir, suffix)
        
        # 同时打印摘要到控制台
        print(f"\n[GEN_LOG] {'='*60}")
        print(f"[GEN_LOG] Mode: {self.mode}")
        print(f"[GEN_LOG] Input: {input_file}")
        print(f"[GEN_LOG] Start tick: {generation_start_tick}")
        print(f"[GEN_LOG] Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:50]}...")
        print(f"[GEN_LOG] Melody events: {len(melody_events)}")
        print(f"[GEN_LOG] Acc events: {len(accompaniment_events)}")
        if diagnostics:
            print(f"[GEN_LOG] Diagnostics: {diagnostics}")
        print(f"[GEN_LOG] Saved to: {filepath}")
        print(f"[GEN_LOG] {'='*60}\n")
        
        return filepath
    
    def save_summary(self) -> str:
        """保存所有日志的摘要"""
        summary = {
            "mode": self.mode,
            "total_generations": len(self.logs),
            "timestamp": datetime.now().isoformat(),
            "logs": [log.to_dict() for log in self.logs],
        }
        
        filepath = Path(self.output_dir) / f"{self.mode}_summary.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(filepath)


def format_token_sequence(tokens: List[int], max_display: int = 100) -> str:
    """格式化 token sequence 用于显示"""
    if len(tokens) <= max_display:
        return str(tokens)
    else:
        return str(tokens[:max_display]) + f" ... ({len(tokens) - max_display} more)"


def compare_logs(fake_rt_log_path: str, offline_log_path: str) -> Dict[str, Any]:
    """
    对比 FakeRT 和 Offline 的日志
    
    Returns:
        对比结果字典
    """
    with open(fake_rt_log_path, 'r') as f:
        fake_rt_log = json.load(f)
    
    with open(offline_log_path, 'r') as f:
        offline_log = json.load(f)
    
    comparison = {
        "fake_rt_file": fake_rt_log_path,
        "offline_file": offline_log_path,
        "prompt_comparison": {
            "fake_rt_token_count": fake_rt_log.get("prompt_token_count", 0),
            "offline_token_count": offline_log.get("prompt_token_count", 0),
            "tokens_match": fake_rt_log.get("prompt_tokens") == offline_log.get("prompt_tokens"),
        },
        "event_comparison": {
            "fake_rt_melody_count": fake_rt_log.get("melody_event_count", 0),
            "offline_melody_count": offline_log.get("melody_event_count", 0),
            "fake_rt_acc_count": fake_rt_log.get("accompaniment_event_count", 0),
            "offline_acc_count": offline_log.get("accompaniment_event_count", 0),
        },
    }
    
    # 详细对比 token sequence
    fake_rt_tokens = fake_rt_log.get("prompt_tokens", [])
    offline_tokens = offline_log.get("prompt_tokens", [])
    
    min_len = min(len(fake_rt_tokens), len(offline_tokens))
    mismatches = []
    for i in range(min_len):
        if fake_rt_tokens[i] != offline_tokens[i]:
            mismatches.append({
                "position": i,
                "fake_rt": fake_rt_tokens[i],
                "offline": offline_tokens[i],
            })
    
    if len(fake_rt_tokens) != len(offline_tokens):
        mismatches.append({
            "position": "length",
            "fake_rt_len": len(fake_rt_tokens),
            "offline_len": len(offline_tokens),
        })
    
    comparison["prompt_comparison"]["mismatches"] = mismatches
    comparison["prompt_comparison"]["mismatch_count"] = len(mismatches)
    
    return comparison

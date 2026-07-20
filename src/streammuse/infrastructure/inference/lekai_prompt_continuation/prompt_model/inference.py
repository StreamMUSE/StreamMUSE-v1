"""
inference.py — 条件推理编排层

职责: 加载数据 → 2小节 mel 作为 prompt → model 生成 acc → 解析序列 → MIDI 输出
"""

from pathlib import Path
from datetime import datetime
import os
import torch
import numpy as np
import safetensors.torch

from .config import ModelConfig, create_llama_config
from .model import PianoLLaMA
from .Token2Midi import MidiConverter
from .PianoDataset import PianoDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 模型加载 ====================

def load_model(model_path, model_config, device='cuda', use_fp16=False):
    token_config = create_llama_config(model_config)
    model = PianoLLaMA(token_config)

    if model_path:
        weights = safetensors.torch.load_file(model_path)
        model.load_state_dict(weights, strict=True)

    if use_fp16 and torch.cuda.is_available():
        model = model.half()

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params, {'FP16' if use_fp16 else 'FP32'}")
    return model


# ==================== 数据准备 ====================

def prepare_condition(dataset, condition_idx, num_bars=2):
    """
    构建条件 prompt: [BOS, TS, BPM, mel_2bars]

    Returns:
        dict: prompt_tokens, gt_path, metadata
    """
    tokenizer = dataset.tokenizer
    v = tokenizer.vocab
    file_path = os.path.join(dataset.root_dir, dataset.data_files[condition_idx])
    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict['metadata'].item()

    if metadata['num_measures'] < num_bars:
        return None

    measures = [save_dict[f'measure_{i}'] for i in range(num_bars)]

    # 编码 mel 部分作为 prompt
    ts_token = tokenizer.encode_time_sig(metadata['time_signature_idx'])
    bpm_token = tokenizer.encode_bpm(metadata['bpm'])
    _, _, _, measure_beats = tokenizer._encode_measures(measures, metadata)

    mel_parts = []
    for beats in measure_beats:
        mel_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
        for mel, acc in beats:
            mel_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
            mel_parts.append(mel)

    mel_content = torch.cat(mel_parts)
    prefix = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)
    prompt = torch.cat([prefix, mel_content])

    return {
        'prompt_tokens': prompt,
        'gt_path': file_path,
        'metadata': {
            'bpm': metadata['bpm'],
            'num_measures': metadata['num_measures'],
        },
    }


# ==================== 批量生成 ====================

def batch_generate(model, dataset, converter, output_dir='generated_samples',
                   num_samples=50, **gen_kwargs):
    """
    条件生成: 2小节 mel → 生成 acc → MIDI
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = dataset.tokenizer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    generated_count = 0
    while generated_count < num_samples:
        idx = torch.randint(0, len(dataset), (1,)).item()

        prep = prepare_condition(dataset, idx)
        if prep is None:
            continue

        # prompt = [BOS TS BPM mel_2bars], 模型续写 acc
        generated = model.generate_music(
            initial_tokens=prep['prompt_tokens'],
            device=device,
            **gen_kwargs,
        )

        # 解析生成的 acc 部分
        mel_beats, acc_beats = tokenizer.parse_generated_sequence(generated.squeeze(0))

        gt_name = Path(prep['gt_path']).stem
        tempo = prep['metadata']['bpm'] or 120

        converter.gt_to_midi(
            prep['gt_path'],
            f"{output_dir}/{timestamp}_{generated_count}_{gt_name}_GT.mid")

        converter.beats_to_midi(
            mel_beats=mel_beats,
            acc_beats=acc_beats,
            tempo=tempo,
            save_path=f"{output_dir}/{timestamp}_{generated_count}_{gt_name}.mid")

        generated_count += 1
        print(f"[{generated_count}/{num_samples}] done")


# ==================== 入口 ====================

if __name__ == '__main__':
    model_config = ModelConfig()

    dataset = PianoDataset(
        '/DATA6_6T/cby/musicxml/allxml_npz_dual_track_optimized_no_underscore',
        config=model_config,
    )
    print(f"Using device: {device}")

    model = load_model(
        model_path="baseline-piano-icml-best-model-20260328-0931/model.safetensors",
        model_config=model_config,
        device=device,
    )

    converter = MidiConverter(dataset.tokenizer)

    batch_generate(
        model, dataset, converter,
        output_dir='generated_samples',
        num_samples=1000,
        temperature=1.1,
        top_k=0,
        top_p=0.98,
        repetition_penalty=1.0,
    )

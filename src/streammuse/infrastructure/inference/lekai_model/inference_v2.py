from pathlib import Path
import pretty_midi
import torch
from transformers import LlamaConfig
import safetensors.torch
import os
import numpy as np
from datetime import datetime

from ..runtime_device import resolve_device
from .PianoDataset import PianoDataset, encode_bpm, process_measure_with_beat_interleaving
from .config import ModelConfig, TrainingConfig
from .model import PianoLLaMA
from .Token2Midi import tokens_to_midi
from .generation_utils import sample_token

device = resolve_device("auto")


def setup_model_configs_llama(model_config: ModelConfig):
    """创建LLaMA配置"""
    token_config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        rope_theta=model_config.rope_theta,
        attention_dropout=model_config.dropout,
        use_cache=True,
        initializer_range=0.02,
    )
    return token_config


def save_gt_midi(save_path, gt_path, velocity=80, dual_track=True):
    """
    从GT npz文件加载pianoroll并保存为MIDI
    """
    # 加载npz文件
    save_dict = np.load(gt_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()

    # 获取基本信息
    num_measures = metadata["num_measures"]
    bpm = metadata["bpm"]
    if bpm is None or bpm <= 0:
        bpm = 120  # 默认BPM
    # 收集所有measure的pianoroll
    all_measures = []
    for i in range(num_measures):
        measure = save_dict[f"measure_{i}"]  # shape: (4, 88, t)
        all_measures.append(measure)

    # 拼接所有measure成完整pianoroll
    # 在时间维度（第2维）上拼接
    full_pianoroll = np.concatenate(all_measures, axis=2)  # shape: (4, 88, total_t)

    if dual_track:
        # 双track模式：保留part0和part1分离
        pianoroll_to_midi(full_pianoroll, save_path, tempo=bpm, velocity=velocity)
    else:
        # 单track模式：合并part0和part1
        merged_pianoroll = np.logical_or(full_pianoroll[:2], full_pianoroll[2:])
        pianoroll_to_midi(merged_pianoroll, save_path, tempo=bpm, velocity=velocity)

    print(f"✅ Saved GT MIDI: {save_path} (BPM={bpm}, measures={num_measures}, dual_track={dual_track})")


def pianoroll_to_midi(merged_pianoroll, save_path, tempo=120, velocity=80):
    """
    将pianoroll转换为MIDI文件（支持单track或双track模式）
    """
    # 创建MIDI对象
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # 计算每个时间单位的秒数（假设1个时间单位 = 1/16音符）
    seconds_per_16th = 60.0 / tempo / 4

    # 判断是单track还是双track模式
    if merged_pianoroll.shape[0] == 2:
        # 单track模式（向后兼容）
        sustain_roll = merged_pianoroll[0]  # shape: (88, t)
        onset_roll = merged_pianoroll[1]  # shape: (88, t)

        # 创建钢琴轨道
        piano = pretty_midi.Instrument(program=0, name="Piano")

        # 遍历每个音高
        note_count = 0
        for pitch_idx in range(88):
            pitch = pitch_idx + 21  # MIDI音高 (A0=21 到 C8=108)

            # 找到所有onset位置
            onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]

            for onset_pos in onset_positions:
                # 找到音符结束位置（sustain结束）
                end_pos = onset_pos + 1

                # 继续向后查找直到sustain结束
                while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                    end_pos += 1

                # 转换为时间（秒）
                start_time = onset_pos * seconds_per_16th
                end_time = end_pos * seconds_per_16th

                # 确保音符有最小长度
                if end_time - start_time < 0.01:
                    end_time = start_time + 0.01

                # 创建MIDI音符
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                piano.notes.append(note)
                note_count += 1

        # 添加乐器到MIDI
        midi.instruments.append(piano)
        print(f"  Created MIDI with {note_count} notes at {tempo} BPM (single track)")

    elif merged_pianoroll.shape[0] == 4:
        # 双track模式
        # Part0 (高声部)
        part0_sustain = merged_pianoroll[0]
        part0_onset = merged_pianoroll[1]
        part0_instrument = pretty_midi.Instrument(program=0, name="Part0 (Melody)")

        note_count_part0 = 0
        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            onset_positions = np.where(part0_onset[pitch_idx] > 0)[0]

            for onset_pos in onset_positions:
                end_pos = onset_pos + 1
                while end_pos < part0_sustain.shape[1] and part0_sustain[pitch_idx, end_pos] > 0:
                    end_pos += 1

                start_time = onset_pos * seconds_per_16th
                end_time = end_pos * seconds_per_16th

                if end_time - start_time < 0.01:
                    end_time = start_time + 0.01

                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                part0_instrument.notes.append(note)
                note_count_part0 += 1

        midi.instruments.append(part0_instrument)

        # Part1 (低声部)
        part1_sustain = merged_pianoroll[2]
        part1_onset = merged_pianoroll[3]
        part1_instrument = pretty_midi.Instrument(program=0, name="Part1 (Accompaniment)")

        note_count_part1 = 0
        for pitch_idx in range(88):
            pitch = pitch_idx + 21
            onset_positions = np.where(part1_onset[pitch_idx] > 0)[0]

            for onset_pos in onset_positions:
                end_pos = onset_pos + 1
                while end_pos < part1_sustain.shape[1] and part1_sustain[pitch_idx, end_pos] > 0:
                    end_pos += 1

                start_time = onset_pos * seconds_per_16th
                end_time = end_pos * seconds_per_16th

                if end_time - start_time < 0.01:
                    end_time = start_time + 0.01

                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time)
                part1_instrument.notes.append(note)
                note_count_part1 += 1

        midi.instruments.append(part1_instrument)
        print(
            f"  Created MIDI with Part0: {note_count_part0} notes, Part1: {note_count_part1} notes at {tempo} BPM (dual track)"
        )

    else:
        raise ValueError(f"Unexpected pianoroll shape: {merged_pianoroll.shape}. Expected (2, 88, t) or (4, 88, t)")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存MIDI文件
    midi.write(save_path)


def load_model(model_path: str, model_config, device: str = "cuda", use_fp16: bool = False):
    """推理示例代码"""
    # 1. 初始化模型
    token_config = setup_model_configs_llama(model_config)
    model = PianoLLaMA(token_config)

    # 2. 加载权重
    if model_path:
        weights = safetensors.torch.load_file(model_path)
        model.load_state_dict(weights, strict=True)

    # 3. 转换为半精度（如果启用）
    if use_fp16 and torch.cuda.is_available():
        model = model.half()
        print("✓ Using FP16 precision (半精度推理)")

    model = model.to(device)
    model.eval()

    # 计算参数量和显存占用估算
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = total_params * (2 if use_fp16 else 4) / (1024**2)

    print(f"Model loaded:")
    print(f"  - Total params: {total_params:,}")
    print(f"  - Model size: {param_size_mb:.1f} MB")
    print(f"  - Precision: {'FP16' if use_fp16 else 'FP32'}")

    return model


@torch.no_grad()
def generate_accompaniment_v2(
    model,
    dataset,
    condition_idx: int,
    delay_beats: int = -1,
    gt_prefix_beats: int = 12,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    device: str = "cuda",
    verbose: bool = True,
    generator: torch.Generator | None = None,
):
    """
    Modified version of generate_accompaniment that skips initial silence in Part 0.
    """
    model.eval()

    # ========== 1. 从数据集提取条件 ==========
    file_path = os.path.join(dataset.root_dir, dataset.data_files[condition_idx])

    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()

    idx_value = metadata["time_signature_idx"]
    if idx_value == 9:
        idx_value = 4
    bpm_value = metadata["bpm"]
    num_measures = metadata["num_measures"]

    # 提取所有part0的beat tokens（条件）和part1的beat tokens（GT）
    part0_beats_list = []
    part1_beats_gt_list = []  # 存储part1的GT beats

    # Use dataset.bar_token if available, else 255
    bar_token = getattr(dataset, "bar_token", 255)

    for i in range(num_measures):
        measure = save_dict[f"measure_{i}"]
        part0_beats, part1_beats = process_measure_with_beat_interleaving(
            measure, tokenizer=dataset.tokenizer, timesteps_per_beat=4
        )
        # 添加bar token
        part0_beats_list.append(torch.tensor([bar_token], dtype=torch.long))
        part0_beats_list.extend(part0_beats)

        # 同样处理part1 GT
        part1_beats_gt_list.append(torch.tensor([bar_token], dtype=torch.long))
        part1_beats_gt_list.extend(part1_beats)

    # --- SKIP SILENCE LOGIC ---
    start_idx = 0
    found_notes = False

    # Iterate through part0_beats_list to find first non-empty beat
    for i, beat_tensor in enumerate(part0_beats_list):
        # Skip Bar tokens
        if beat_tensor.numel() == 1 and beat_tensor.item() == bar_token:
            continue

        # Check if beat has notes
        # Assuming notes are tokens < 168 (since 168-173 are markers)
        has_notes = False
        for token in beat_tensor:
            if token.item() < 168:
                has_notes = True
                break

        if has_notes:
            # Found notes! Backtrack to the nearest Bar token
            # We want to start from the beginning of the measure containing this beat
            for j in range(i, -1, -1):
                if part0_beats_list[j].numel() == 1 and part0_beats_list[j].item() == bar_token:
                    start_idx = j
                    found_notes = True
                    break
            if found_notes:
                break

    if found_notes and start_idx > 0:
        if verbose:
            print(f"Skipping initial silence: starting from index {start_idx} (Measure {start_idx // 5})")
        part0_beats_list = part0_beats_list[start_idx:]
        part1_beats_gt_list = part1_beats_gt_list[start_idx:]
    elif not found_notes:
        if verbose:
            print("Warning: No notes found in Part 0. Using full sequence.")

    # --------------------------

    pad_marker = 173

    # 根据delay_beats决定padding位置
    if delay_beats >= 0:
        # part1延迟：part0后面添加padding
        part0_beats_list.append(torch.tensor([pad_marker] * delay_beats, dtype=torch.long))
        # part1_beats_gt_list前面添加padding
        part1_beats_gt_list.insert(0, torch.tensor([pad_marker] * delay_beats, dtype=torch.long))
    else:
        # part1提前：part0前面添加padding
        advance_beats = -delay_beats
        part0_beats_list.insert(0, torch.tensor([pad_marker] * advance_beats, dtype=torch.long))
        # part1_beats_gt_list后面添加padding
        part1_beats_gt_list.append(torch.tensor([pad_marker] * advance_beats, dtype=torch.long))

    if verbose:
        print(f"提取条件: {len(part0_beats_list)} 个part0 beats, {len(part1_beats_gt_list)} 个part1 GT beats")

    # ========== 2. 构造初始prompt ==========
    bpm_token = encode_bpm(bpm_value) + dataset.bpm_offset_id

    initial_tokens = [
        torch.tensor([dataset.bos_token], dtype=torch.long),
        torch.tensor([idx_value + dataset.time_sig_offset_id], dtype=torch.long),
        torch.tensor([bpm_token], dtype=torch.long),
    ]

    generated = torch.cat(initial_tokens, dim=0).unsqueeze(0).to(device)

    # ========== 3. 交错生成part0和part1 ==========
    part0_idx = 0  # 当前注入到第几拍part0
    part1_idx = 0  # 当前处理到第几拍part1（用于GT注入）
    part1_beats_generated = []  # 收集生成的part1 beats
    current_part1_beat = []  # 当前正在生成的part1 beat

    part1_end_marker = 171
    part1_empty_marker = 169
    bar_token_id = 255

    position = 0  # 0=part0位置, 1=part1位置（交错）
    past_key_values = None
    l = len(part0_beats_list)
    max_iterations = 7000  # 防止无限循环

    if verbose:
        print(f"开始生成 (总拍数: {l}, 序列将包含~{l * 2}拍)")

    # 定期清理显存的计数器
    cache_reset_count = 0

    with torch.no_grad():
        for iteration in range(max_iterations):
            # ===== 决定当前位置应该做什么 =====
            if position == 0:  # part0位置
                # 如果delay_beats < 0，前abs(delay_beats)拍的part0位置需要填充pad
                if delay_beats < 0 and part0_idx <= -delay_beats:
                    pad_token = torch.tensor([[pad_marker]], device=device)
                    generated = torch.cat([generated, pad_token], dim=1)
                    part0_idx += 1
                    position = 1  # 切换到part1位置
                    past_key_values = None
                    cache_reset_count += 1
                    continue

                if part0_idx < l:
                    # 注入下一拍part0
                    next_part0_beat = part0_beats_list[part0_idx].to(device)
                    generated = torch.cat([generated, next_part0_beat.unsqueeze(0)], dim=1)
                    part0_idx += 1
                    position = 1  # 切换到part1位置
                    past_key_values = None  # 重置cache（因为我们手动添加了tokens）
                    cache_reset_count += 1
                    continue

                else:
                    # part0全部注入完毕
                    break

            else:  # part1位置（需要生成或注入GT）
                # 如果delay_beats >= 0，前delay_beats拍的part1位置填充pad_marker
                if delay_beats >= 0 and part0_idx <= delay_beats:
                    # 直接注入pad marker
                    pad_token = torch.tensor([[pad_marker]], device=device)
                    generated = torch.cat([generated, pad_token], dim=1)
                    part1_idx += 1  # 更新part1计数
                    position = 0  # 切换回part0位置
                    past_key_values = None
                    cache_reset_count += 1
                    continue

                # 检查是否应该使用GT（前gt_prefix_beats拍）
                if part1_idx < gt_prefix_beats and part1_idx < len(part1_beats_gt_list):
                    # 直接注入GT beat
                    gt_beat = part1_beats_gt_list[part1_idx].to(device)
                    generated = torch.cat([generated, gt_beat.unsqueeze(0)], dim=1)
                    # 将GT beat记录到生成结果中
                    part1_beats_generated.append(gt_beat.cpu().tolist())
                    part1_idx += 1
                    position = 0  # 切换回part0位置
                    past_key_values = None
                    cache_reset_count += 1
                    continue

                # 开始真正生成part1（过了GT前缀后）
                # 生成一个token
                outputs = model.model(
                    input_ids=generated[:, -1:] if past_key_values else generated,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                next_token = sample_token(
                    logits,
                    generated_tokens=generated,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generator=generator,
                )

                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                current_part1_beat.append(next_token.item())

                # 检查是否完成一拍part1
                if next_token.item() in [part1_end_marker, part1_empty_marker, bar_token_id]:
                    # 一拍part1生成完毕
                    part1_beats_generated.append(current_part1_beat.copy())
                    current_part1_beat = []
                    part1_idx += 1  # 更新part1计数
                    position = 0  # 切换回part0位置，注入下一拍part0
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_gt_beats = min(gt_prefix_beats, len(part1_beats_generated))
    num_generated_beats = len(part1_beats_generated) - num_gt_beats

    return {
        "generated_sequence": generated.cpu(),
        "part0_beats": part0_beats_list,
        "part1_beats": part1_beats_generated,
        "GT_path": file_path,
        "metadata": {
            "time_signature_idx": idx_value,
            "bpm": bpm_value,
            "num_measures": num_measures,
            "num_part0_beats": part0_idx,
            "num_part1_beats": len(part1_beats_generated),
            "num_part1_gt_beats": num_gt_beats,
            "num_part1_generated_beats": num_generated_beats,
        },
    }


def batch_generate_all_samples(
    model,
    dataset,
    output_dir="generated_samples",
    gt_prefix_beats=16,
    delay_beats=-1,
    temperature=1.1,
    top_k=10,
    top_p=0.95,
    repetition_penalty=1.0,
):
    """遍历数据集中的所有歌曲并生成伴奏"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_songs = len(dataset)
    print(f"开始生成所有 {total_songs} 首歌曲（gt_prefix_beats={gt_prefix_beats}）...")

    for idx in range(total_songs):
        print(f"\n[{idx + 1}/{total_songs}] 正在处理...")

        seq = generate_accompaniment_v2(
            model,
            dataset,
            condition_idx=idx,
            delay_beats=delay_beats,
            gt_prefix_beats=gt_prefix_beats,
            temperature=temperature,
            device=device,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # 获取文件名
        gt_name = Path(seq["GT_path"]).stem
        
        # 保存 GT MIDI
        gt_save_path = f"{output_dir}/{timestamp}_{idx}_{gt_name}_GT.mid"
        save_gt_midi(save_path=gt_save_path, gt_path=seq["GT_path"], velocity=80)
        
        # 保存生成的 MIDI
        gen_save_path = f"{output_dir}/{timestamp}_{idx}_{gt_name}_generated.mid"
        tokens_to_midi(result_dict=seq, save_path=gen_save_path, velocity=80)

        print(f"  ✓ 已保存: {gt_name}")

    print(f"\n✅ 全部完成！共生成 {total_songs} 首歌曲到 {output_dir}/")


def batch_generate_50_samples_v2(model, dataset, output_dir="generated_samples", min_length=600):
    """批量生成50首音乐，长度少于500的舍弃"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_count = 0
    total_attempts = 0

    print(f"开始生成50首音乐（最小长度: {min_length}）...")

    while saved_count < 200:
        total_attempts += 1

        condition_idx = torch.randint(0, len(dataset), (1,)).item()
        print(condition_idx)

        # Use generate_accompaniment_v2
        seq = generate_accompaniment_v2(
            model,
            dataset,
            condition_idx=condition_idx,
            delay_beats=-1,  # part1提前1拍
            gt_prefix_beats=0,  # 使用前40拍的GT
            temperature=1.1,
            device=device,
            top_k=10,
            top_p=0.95,
            repetition_penalty=1.0,
        )

        # 去掉batch维度
        gt_name = Path(seq["GT_path"]).stem  # 只取文件名，不带路径和扩展名
        save_path = f"{output_dir}/{timestamp}_{total_attempts}_{gt_name}_GT.mid"
        save_gt_midi(save_path=save_path, gt_path=seq["GT_path"], velocity=80)
        tokens_to_midi(
            result_dict=seq, save_path=f"{output_dir}/{timestamp}_{total_attempts}_{gt_name}.mid", velocity=80
        )
        saved_count += 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate accompaniment from NPZ files")
    parser.add_argument("--npz_dir", type=str, default="npz", help="Path to NPZ directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model.safetensors")
    parser.add_argument("--output_dir", type=str, default="generated_output", help="Output directory")
    parser.add_argument("--gt_prefix_beats", type=int, default=16, help="Number of GT beats as prompt")
    parser.add_argument("--delay_beats", type=int, default=-1, help="Delay beats (-1 means acc leads by 1)")
    parser.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature")
    args = parser.parse_args()
    
    model_config = ModelConfig()
    
    my_dataset = PianoDataset(
        args.npz_dir,
        config=model_config,
        cache_lengths=False,
    )
    print(f"Using device: {device}")
    print(f"Loaded {len(my_dataset)} songs from {args.npz_dir}")

    # 加载模型
    model = load_model(
        model_path=args.model_path,
        model_config=model_config,
        device=device,
    )

    # 遍历所有歌曲生成
    batch_generate_all_samples(
        model,
        my_dataset,
        output_dir=args.output_dir,
        gt_prefix_beats=args.gt_prefix_beats,
        delay_beats=args.delay_beats,
        temperature=args.temperature,
    )

    # # 批量生成50首
    # batch_generate_50_samples_v2(model, my_dataset, output_dir="gen_input_5_v2-no-prompt", min_length=200)

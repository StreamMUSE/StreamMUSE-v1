from pathlib import Path
from typing import Any

import pretty_midi
import torch
from transformers import LlamaConfig
import safetensors.torch
import os
from datetime import datetime
import numpy as np

from ..runtime_device import dtype_to_name, resolve_device, resolve_dtype
from .PianoDataset import PianoDataset
from .config import ModelConfig, TrainingConfig
from .model import PianoLLaMA
from .Token2Midi import tokens_to_midi


device = resolve_device("auto")


def setup_model_configs_llama(model_config: ModelConfig, *, use_cache: bool = True):
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
        use_cache=use_cache,
        initializer_range=0.02,
    )
    return token_config


def infer_checkpoint_format(model_path: str) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".safetensors":
        return "safetensors"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return suffix[1:]
    return "unknown"


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and payload:
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload  # type: ignore[return-value]
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, dict) and candidate and all(isinstance(v, torch.Tensor) for v in candidate.values()):
                return candidate  # type: ignore[return-value]
    raise ValueError(
        "Unsupported checkpoint structure. Expected a tensor dict or a dict containing "
        "'state_dict' / 'model_state_dict' / 'model'."
    )


def _strip_prefix_all(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _add_prefix_missing(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k if k.startswith(prefix) else f"{prefix}{k}": v for k, v in state_dict.items()}


def load_state_dict_with_metadata(model_path: str) -> tuple[dict[str, torch.Tensor], str, int]:
    checkpoint_format = infer_checkpoint_format(model_path)
    if checkpoint_format == "unknown":
        raise ValueError(
            f"Unsupported checkpoint extension for {model_path}. "
            "Supported: .safetensors, .pt, .pth, .ckpt"
        )

    if checkpoint_format == "safetensors":
        raw = safetensors.torch.load_file(model_path)
    else:
        raw = torch.load(model_path, map_location="cpu")

    state_dict = _extract_state_dict(raw)
    tensor_count = len(state_dict)
    return state_dict, checkpoint_format, tensor_count


def _load_state_dict_strict_with_adapters(model: PianoLLaMA, state_dict: dict[str, torch.Tensor]) -> None:
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [
        ("original", state_dict),
        ("strip_module", _strip_prefix_all(state_dict, "module.")),
        (
            "strip_module_add_model",
            _add_prefix_missing(_strip_prefix_all(state_dict, "module."), "model."),
        ),
        ("add_model", _add_prefix_missing(state_dict, "model.")),
        ("strip_model", _strip_prefix_all(state_dict, "model.")),
    ]

    seen: set[tuple[str, ...]] = set()
    last_error: RuntimeError | None = None
    for name, candidate in candidates:
        key_tuple = tuple(candidate.keys())
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        try:
            model.load_state_dict(candidate, strict=True)
            if name != "original":
                print(f"[load_model] Applied checkpoint key adapter: {name}")
            return
        except RuntimeError as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(f"Failed to load checkpoint state_dict with strict adapters: {last_error}") from last_error


def save_gt_midi(save_path, gt_path, velocity=80, dual_track=True):
    """
    从GT npz文件加载pianoroll并保存为MIDI

    Args:
        save_path: MIDI文件保存路径
        gt_path: GT npz文件路径
        velocity: MIDI音符力度
        dual_track: 是否保存为双track模式（True=分离part0和part1，False=合并）
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

    Args:
        merged_pianoroll: shape (2, 88, t) 或 (4, 88, t)
                         如果是(2, 88, t)：[0]是sustain，[1]是onset（单track模式，合并的）
                         如果是(4, 88, t)：前2通道是part0，后2通道是part1（双track模式）
        save_path: MIDI保存路径
        tempo: BPM速度
        velocity: 音符力度
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


def load_model(
    model_path: str,
    model_config,
    device: str = "cuda",
    use_fp16: bool = False,
    dtype: torch.dtype | None = None,
    use_cache: bool = True,
):
    """推理示例代码

    Args:
        model_path: 模型权重路径
        model_config: 模型配置
        device: 设备
        use_fp16: 是否使用半精度推理（大幅减少显存占用）
    """
    # 1. 初始化模型
    token_config = setup_model_configs_llama(model_config, use_cache=use_cache)
    model = PianoLLaMA(token_config)

    chosen_dtype = dtype
    if chosen_dtype is None:
        if use_fp16 and device != "cpu":
            chosen_dtype = torch.float16
        else:
            chosen_dtype = resolve_dtype(device, "auto")
    if chosen_dtype == torch.float16 and device == "cpu":
        chosen_dtype = torch.float32

    checkpoint_format = "none"
    tensor_count = 0

    # 2. 加载权重
    if model_path:
        state_dict, checkpoint_format, tensor_count = load_state_dict_with_metadata(model_path)
        _load_state_dict_strict_with_adapters(model, state_dict)

    # 3. 设置精度并移动设备
    if chosen_dtype == torch.float16:
        model = model.half()

    model = model.to(device)
    model.eval()

    # 计算参数量和显存占用估算
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = total_params * (2 if chosen_dtype == torch.float16 else 4) / (1024**2)

    print(f"Model loaded:")
    print(f"  - Checkpoint format: {checkpoint_format}")
    print(f"  - Checkpoint tensors: {tensor_count}")
    print(f"  - Total params: {total_params:,}")
    print(f"  - Model size: {param_size_mb:.1f} MB")
    print(f"  - Device: {device}")
    print(f"  - Precision: {dtype_to_name(chosen_dtype)}")

    return model


def batch_generate_50_samples(model, dataset, output_dir="generated_samples", min_length=600):
    """批量生成50首音乐，长度少于500的舍弃

    Args:
        model: 加载好的模型
        output_dir: 输出目录
        min_length: 最小序列长度要求
    """
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
        # 生成序列  4/4  346
        # pre = torch.cat([pre, tensor_data]).unsqueeze(0)
        seq = model.generate_accompaniment(
            dataset,
            condition_idx=condition_idx,
            delay_beats=-1,  # part1提前1拍
            gt_prefix_beats=40,  # 使用前40拍的GT
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


# 使用示例
if __name__ == "__main__":
    model_config = ModelConfig()

    training_config = TrainingConfig()
    # my_dataset = PianoDataset(
    #     "rt/RT_Accompaniment/mini_dataset",
    #     config=model_config,
    #     cache_lengths=False,  # 启用长度缓存
    # )
    my_dataset = PianoDataset(
        "output_npz_test",
        config=model_config,
        cache_lengths=False,  # 启用长度缓存
    )
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(
        model_path="rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors",
        model_config=model_config,
        device=device,
    )

    # 批量生成50首
    # batch_generate_50_samples(model, my_dataset, output_dir="generated_samples_1030", min_length=500)
    batch_generate_50_samples(model, my_dataset, output_dir="gen_input_5", min_length=200)
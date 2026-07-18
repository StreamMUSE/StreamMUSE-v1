from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from transformers import LlamaConfig

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 基础配置
    time = datetime.now().strftime("%m%d_%H%M")
    output_dir: str = "./checkpoints"
    num_epochs: int = 40
    save_model_epochs: int = 5
    train_batch_size = 4
    gradient_accumulation_steps: int = 8
    lr_warmup_ratio: float = 0.01       # warmup 占总步数的比例
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # 学习率配置
    learning_rate: float = 6e-5

    # 日志配置
    log: bool = False
    tensorboard_log_dir: str = "./logs"
    tensorboard_log_name: str = f"music_transformer_{time}"
    data_dir = "/DATA6_6T/cby/musicxml/allxml_npz_dual_track_optimized_no_underscore"
    save_steps = -1

    # 测试集配置
    use_test_set: bool = True  # 是否使用测试集
    test_split_ratio: float = 0.10  # 测试集划分比例（5%作为测试集）
    test_frequency: float = 0.5  # 测试频率（每0.10个epoch测试一次）
    test_batch_size: int = 4  # 测试时的batch size
    test_save_results: bool = True  # 是否保存测试结果到tensorboard
    random_seed: int = 42  # 数据集划分的随机种子

@dataclass
class ModelConfig:
    """模型架构配置"""
    # Token-level GPT配置
    vocab_size: int = 268
    hidden_size: int = 768  #
    num_hidden_layers: int = 16
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 8192
    marker_offset: int = 81   # 相对位置标记起始 ID (81-168, 覆盖 88 个音高)
    empty_patch_id: int = 169 # 空拍占位 token
    measures_length = 88  # 音高维度 (钢琴 88 键)
    patch_h = 1  # 音高方向的patch大小
    patch_w = 4  # 时间方向的patch大小

    # Token IDs
    track_marker_acc_id: int = 170   # acc 轨起始标记
    track_marker_mel_id: int = 171   # mel 轨起始标记
    beat_marker_id: int = 172        # 拍分隔符
    bar_token_id: int = 255
    eos_token_id: int = 256
    bos_token_id: int = 257
    pad_token_id: int = 258
    time_sig_offset_id: int = 259
    bpm_offset_id: int = 264
    train_cutoff_len = 1024  # 训练时的截断长度
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # 拍号 191: 3/4 192: 4/4
    # 和弦作为小节bar  190


def create_llama_config(model_config: ModelConfig, attn_implementation: str = "sdpa") -> LlamaConfig:
    """从 ModelConfig 创建 LlamaConfig（训练和推理共用）"""
    return LlamaConfig(
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
        attn_implementation=attn_implementation,
        use_cache=True,
    )

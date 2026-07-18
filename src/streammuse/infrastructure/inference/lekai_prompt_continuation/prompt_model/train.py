"""
训练主程序
负责初始化数据集、模型和训练器，然后启动训练流程

使用方法:
  单GPU训练: python train.py
  多GPU训练: accelerate launch --multi_gpu --num_processes=3 train.py
  后台运行: nohup accelerate launch --multi_gpu --num_processes=2 train.py > training.log 2>&1 &
"""
import torch
import safetensors.torch
from torch.utils.data import DataLoader

from .config import TrainingConfig, ModelConfig, create_llama_config
from .PianoDataset import PianoDataset, StreamingDataset, PaddingCollator
from .trainer import TransformerTrainer
from .model import PianoLLaMA


def create_datasets(train_config: TrainingConfig, model_config: ModelConfig):
    """创建训练集和测试集"""
    train_dataset = PianoDataset(
        train_config.data_dir,
        config=model_config,
        mode='train',
        test_split_ratio=train_config.test_split_ratio,
        random_seed=train_config.random_seed,
        truncate=False,  # streaming 负责切片，不在 Dataset 层截断
    )

    test_dataset = None
    if train_config.use_test_set:
        test_dataset = PianoDataset(
            train_config.data_dir,
            config=model_config,
            mode='test',
            test_split_ratio=train_config.test_split_ratio,
            random_seed=train_config.random_seed,
        )
        print(f"训练集大小: {len(train_dataset)} 个样本")
        print(f"测试集大小: {len(test_dataset)} 个样本")

    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset, train_config, model_config):
    """创建数据加载器（训练集使用 streaming，测试集使用 padding）"""
    # 训练集：流式打包，每个 chunk 恰好 max_seq_len，零 padding
    streaming = StreamingDataset(
        base_dataset=train_dataset,
        max_seq_len=model_config.train_cutoff_len,
        pad_token_id=model_config.pad_token_id,
    )
    estimated_batches = streaming.estimate_num_batches(train_config.train_batch_size)
    print(f"Streaming: ~{len(streaming)} chunks/epoch, ~{estimated_batches} batches/epoch")

    train_dataloader = DataLoader(
        streaming,
        batch_size=train_config.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # 测试集：标准 padding
    test_dataloader = None
    if test_dataset is not None:
        test_collator = PaddingCollator(
            max_seq_len=model_config.train_cutoff_len,
            pad_token_id=model_config.pad_token_id,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config.test_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=test_collator,
            pin_memory=True,
        )

    return train_dataloader, test_dataloader


def initialize_model(llama_config, checkpoint_path: str = None) -> PianoLLaMA:
    """初始化模型并加载预训练权重

    Args:
        llama_config: LLaMA配置
        checkpoint_path: 预训练权重的路径（可选）

    Returns:
        PianoLLaMA: 初始化好的模型
    """
    model = PianoLLaMA(llama_config)

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 加载预训练权重（如果提供）
    if checkpoint_path:
        print(f"正在加载预训练权重: {checkpoint_path}")
        weights = safetensors.torch.load_file(checkpoint_path)
        model.load_state_dict(weights, strict=False)
        print("预训练权重加载完成")
    else:
        print("不加载预训练权重，从头训练")

    # model = torch.compile(model)  # FA2 + DDP 下暂不兼容，后续可开启
    return model


def main():
    """主函数：协调整个训练流程"""
    # 加载配置
    train_config = TrainingConfig()
    model_config = ModelConfig()

    llama_config = create_llama_config(model_config, attn_implementation="flash_attention_2")

    # 创建数据集
    train_dataset, test_dataset = create_datasets(train_config, model_config)

    print('创建数据加载器')
    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset, test_dataset, train_config, model_config
    )

    print('初始化模型')
    checkpoint_path = "/home/guhaoyu/qlk/real_time/icml_code_backup/baseline-piano-icml-best-model-20260328-0931/model.safetensors"
    resume_path = None          # 断点续训路径 (如 "./checkpoints/latest")
    model = initialize_model(llama_config, checkpoint_path)

    trainer = TransformerTrainer(
        config=train_config,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        resume_path=resume_path,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()
    print("\n训练完成!")


if __name__ == '__main__':
    main()

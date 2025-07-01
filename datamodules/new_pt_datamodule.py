import torch
from torch.utils.data import Dataset
from schema.dataset_schema import NewPtDatasetSchema,NewPtDataModuleSchema
from schema.model_io_schema import NewPtM2AModelInputData
import pytorch_lightning as pl

class NewPtDataset(Dataset):
    def __init__(self, config: NewPtDatasetSchema):
        self.file_path = config.file_path
        self.target_length = config.target_length
        self.split_ratio = config.split_ratio
        self.stage = config.stage
    
        self.sequence_shift = config.sequence_shift

        try:
            self.length = torch.load(self.file_path[:-3] + ".length.pt", mmap=True)
            self.start = torch.cumsum(self.length, dim=0) - self.length
            
            # **只加载 input 数据**
            self.input_acc_data_full = torch.load(self.file_path, mmap=True)  # 完整的 input 伴奏数据
            self.input_mel_data_full = torch.load(self.file_path.replace("acc.pt", "mel.pt"), mmap=True)  # 完整的 input 旋律数据
            
            self.pitch_shift_range_acc = torch.load(self.file_path[:-3] + ".pitch_shift_range.pt", mmap=True).reshape(-1, 2)
            self.pitch_shift_range_mel = torch.load(
                self.file_path.replace("acc.pt", "mel.pt")[:-3] + ".pitch_shift_range.pt", mmap=True
            ).reshape(-1, 2)
        except Exception as e:
            raise RuntimeError(f"Error loading .pt files for {self.file_path} in FramedDataset: {e}")

        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 0] < -5, 0] = -5
        self.pitch_shift_range_acc[self.pitch_shift_range_acc[:, 1] > 6, 1] = 6
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 0] < -5, 0] = -5
        self.pitch_shift_range_mel[self.pitch_shift_range_mel[:, 1] > 6, 1] = 6

        if self.stage == "val" or self.stage == "test":
            self.pitch_shift_range_acc = torch.zeros_like(self.pitch_shift_range_acc)
            self.pitch_shift_range_mel = torch.zeros_like(self.pitch_shift_range_mel)

        # 调整有效长度的判断：由于有 shift，实际可用的片段长度会减少
        # 确保 target 序列不会超出原始数据长度
        is_valid = (self.length - self.sequence_shift) >= self.target_length
        self.valid_indices = torch.arange(len(self.start))[is_valid]

        if self.stage == "all":
            pass
        elif self.stage == "train":
            self.valid_indices = self.valid_indices[self.valid_indices % self.split_ratio != 0]
        elif self.stage == "val" or self.stage == "test":
            self.valid_indices = self.valid_indices[self.valid_indices % self.split_ratio == 0]

        self.valid_song_count = len(self.valid_indices)
        self.target_length = self.target_length

        print(f"Metadata for dataset {self.file_path} loaded. self.stage: {self.stage}. Number of valid songs: {self.valid_song_count}")

    def __len__(self) -> int:
        """返回数据集中有效歌曲片段的总数。"""
        return self.valid_song_count

    def __getitem__(self, idx: int) -> NewPtM2AModelInputData:
        """
        根据索引 idx 获取单个样本（固定长度的片段及其音高偏移）。
        获取包含 input 和 target 的合并片段。
        Args:
            idx (int): self.valid_indices 中的索引。
        Returns:
            NewPtM2AModelInputData: 包含合并的旋律/伴奏片段和音高偏移的模型输入数据。
        """
        raw_id = self.valid_indices[idx]
        
        # 计算 input 片段的最大起始偏移量
        # 这确保了整个合并片段 (input + sequence_shift) 都在原始数据范围内
        max_input_start_offset = self.length[raw_id] - self.target_length - self.sequence_shift
        
        # 随机确定 input 片段在原始数据中的相对起始位置
        # max_input_start_offset + 1 用于确保 torch.rand 的范围包含最大偏移量
        segment_relative_input_start = torch.floor(torch.rand(1) * (max_input_start_offset + 1)).long()
        
        # 计算 input 片段在完整数据集中的实际起始位置
        actual_input_start_in_full_data = segment_relative_input_start + self.start[raw_id]

        # 计算 target 片段在完整数据集中的实际结束位置
        # 这覆盖了从 input 开始到 target 结束的整个范围
        actual_target_end_in_full_data = actual_input_start_in_full_data + self.target_length + self.sequence_shift
        
        # 确定合并片段的起始和结束索引
        # 这个片段将包含 input 和 target 两部分
        combined_segment_start_index = actual_input_start_in_full_data
        combined_segment_end_index = actual_target_end_in_full_data # 结束索引是排他的，与 Python 切片一致
        
        # 获取完整的合并旋律和伴奏片段
        # 这些片段将直接传递给您的模型
        combined_mel_segment = self.input_mel_data_full[combined_segment_start_index:combined_segment_end_index]
        combined_acc_segment = self.input_acc_data_full[combined_segment_start_index:combined_segment_end_index]

        # 计算音高偏移 (这部分保持不变)
        minmax = torch.minimum(self.pitch_shift_range_mel[raw_id, 1], self.pitch_shift_range_acc[raw_id, 1])
        maxmin = torch.maximum(self.pitch_shift_range_mel[raw_id, 0], self.pitch_shift_range_acc[raw_id, 0])
        single_pitch_shift = torch.floor(torch.rand(1) * (minmax - maxmin + 1)).long() + maxmin
        
        return NewPtM2AModelInputData(
            mel_data=combined_mel_segment,
            acc_data=combined_acc_segment,
            pitch_shift=single_pitch_shift,
        )


class NewPtDataModule(pl.LightningDataModule):
    def __init__(self, config: NewPtDataModuleSchema):
        super().__init__()
        self.config = config
        self.train_dataset = NewPtDataset(config.train_config)
        self.val_dataset = NewPtDataset(config.val_config)
        self.test_dataset = NewPtDataset(config.test_config)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.config.train_config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.config.val_config.batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.config.test_config.batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: list[NewPtM2AModelInputData]) -> NewPtM2AModelInputData:
        """
        Collate function to stack the tensors from a batch.
        Assumes all segments in the batch have the same target_length.
        """
        mel_data = torch.stack([item.mel_data for item in batch])
        acc_data = torch.stack([item.acc_data for item in batch])
        pitch_shift = torch.stack([item.pitch_shift for item in batch])
        
        return NewPtM2AModelInputData(
            mel_data=mel_data,
            acc_data=acc_data,
            pitch_shift=pitch_shift
        )
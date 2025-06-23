import torch
from torch.utils.data import Dataset
from schema.dataset_schema import OldPtDatasetSchema,OldPtDataModuleSchema
from schema.model_io_schema import M2AModelInputData
import pytorch_lightning as pl

class OldPtDataset(Dataset):
    def __init__(self, config: OldPtDatasetSchema):
        self.file_path = config.file_path
        self.target_length = config.target_length
        self.split_ratio = config.split_ratio
        self.stage = config.stage
        try:
            self.length = torch.load(self.file_path[:-3] + ".length.pt", weights_only=True)
            self.start = torch.cumsum(self.length, dim=0) - self.length
            self.data_acc = torch.load(self.file_path, weights_only=True)  # 伴奏
            self.data_mel = torch.load(self.file_path.replace("acc.pt", "mel.pt"), weights_only=True)  # 旋律

            self.pitch_shift_range_acc = torch.load(self.file_path[:-3] + ".pitch_shift_range.pt", weights_only=True).reshape(-1, 2)
            self.pitch_shift_range_mel = torch.load(
                self.file_path.replace("acc.pt", "mel.pt")[:-3] + ".pitch_shift_range.pt", weights_only=True
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

        is_valid = self.length >= self.target_length
        self.valid_indices = torch.arange(len(self.start))[is_valid]

        if self.stage == "all":
            pass
        elif self.stage == "train":
            self.valid_indices = self.valid_indices[self.valid_indices % self.split_ratio != 0]
        elif self.stage == "val" or self.stage == "test":
            self.valid_indices = self.valid_indices[self.valid_indices % self.split_ratio == 0]

        self.stage = self.stage
        self.valid_song_count = len(self.valid_indices)
        self.target_length = self.target_length

        print(f"Metadata for dataset {self.file_path} loaded. self.stage: {self.stage}. Number of valid songs: {self.valid_song_count}")

    def __len__(self) -> int:
        """返回数据集中有效歌曲片段的总数。"""
        return self.valid_song_count

    def __getitem__(self, idx: int) -> M2AModelInputData:
        """
        根据索引 idx 获取单个样本（固定长度的片段及其音高偏移）。
        Args:
            idx (int): self.valid_indices 中的索引。
        Returns:
            M2AModelInputData: 包含旋律片段、伴奏片段和音高偏移的模型输入数据。
        """
        raw_id = self.valid_indices[idx]
        segment_relative_start = torch.floor(torch.rand(1) * (self.length[raw_id] - self.target_length)).long()
        actual_segment_start_in_full_data = segment_relative_start + self.start[raw_id]
        index_vector = torch.arange(self.target_length) + actual_segment_start_in_full_data

        mel_segment = self.data_mel[index_vector]
        acc_segment = self.data_acc[index_vector]

        minmax = torch.minimum(self.pitch_shift_range_mel[raw_id, 1], self.pitch_shift_range_acc[raw_id, 1])
        maxmin = torch.maximum(self.pitch_shift_range_mel[raw_id, 0], self.pitch_shift_range_acc[raw_id, 0])
        single_pitch_shift = torch.floor(torch.rand(1) * (minmax - maxmin + 1)).long() + maxmin
        return M2AModelInputData(
            mel_data=mel_segment,
            acc_data=acc_segment,
            pitch_shift=single_pitch_shift,
        )


class OldPtDataModule(pl.LightningDataModule):
    def __init__(self, config: OldPtDataModuleSchema):
        super().__init__()
        self.config = config
        self.train_dataset = OldPtDataset(config.train_config)
        self.val_dataset = OldPtDataset(config.train_config)
        self.test_dataset = OldPtDataset(config.train_config)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.train_config.batch_size, shuffle=True,collate_fn=self._collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.val_config.batch_size, shuffle=False,collate_fn=self._collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.test_config.batch_size, shuffle=False,collate_fn=self._collate_fn)
    
    def _collate_fn(self, batch:list[M2AModelInputData]) -> M2AModelInputData:
        """
        Collate function to handle variable-length sequences.
        """
        mel_data = [item.mel_data for item in batch]
        acc_data = [item.acc_data for item in batch]
        pitch_shift = [item.pitch_shift for item in batch]

        # mel_data_padded = torch.nn.utils.rnn.pad_sequence(mel_data, batch_first=True)
        # acc_data_padded = torch.nn.utils.rnn.pad_sequence(acc_data, batch_first=True)
        # pitch_shift_tensor = torch.tensor(pitch_shift)

        # return M2AModelInputData(
        #     mel_data=mel_data_padded,
        #     acc_data=acc_data_padded,
        #     pitch_shift=pitch_shift_tensor
        # )
        mel_data = torch.stack(mel_data, dim=0)
        acc_data = torch.stack(acc_data, dim=0)
        pitch_shift_tensor = torch.tensor(pitch_shift)
        
        return M2AModelInputData(
            mel_data=mel_data,
            acc_data=acc_data,
            pitch_shift=pitch_shift_tensor
        )
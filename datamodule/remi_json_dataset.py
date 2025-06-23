import os
import json  # Import json module
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from schema.dataset_schema import MelAccRemiJsonDatasetSchema, MelAccRemiJsonDataModuleSchema
from schema.model_io_schema import ModelInputData
import numpy as np
import torch
import glob

BAR_NONE_TOKEN_ID = 4  # Assuming Bar_None is represented by 0 in the tokenizer
EXTRACT_BAR_NUM = 4  # Assuming we want to extract 4 bars
ACC_BAR_BIAS = 1


def random_clip_and_pad(
    mel_sequence: np.ndarray,
    acc_sequence: np.ndarray,
    target_length: int,
    padding_token_id: int = 0,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    padding_probability: float = 0.0,
    exp_bias_lambda: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机截取并填充序列，同时在开头和结尾添加 SOS 和 EOS 标记。
    """
    bar_indices = np.where(mel_sequence == BAR_NONE_TOKEN_ID)[0]

    mel_start_bar = np.random.choice(bar_indices[: -EXTRACT_BAR_NUM + 1]) if len(bar_indices) >= EXTRACT_BAR_NUM else 0
    mel_end_bar = mel_start_bar + EXTRACT_BAR_NUM
    mel_seq = mel_sequence[mel_start_bar:mel_end_bar]
    acc_start_bar = mel_start_bar + ACC_BAR_BIAS if mel_start_bar + ACC_BAR_BIAS < len(acc_sequence) else mel_start_bar
    acc_end_bar = mel_end_bar + ACC_BAR_BIAS if mel_end_bar + ACC_BAR_BIAS < len(acc_sequence) else mel_end_bar
    acc_seq = acc_sequence[acc_start_bar:acc_end_bar]

    max_content_len = target_length - 2  # 为 SOS 和 EOS 标记留出空间

    use_random_shortening = np.random.rand() < padding_probability
    len_shortening_rand_val = np.random.rand()

    def process_sequence(seq: np.ndarray) -> torch.Tensor:
        if use_random_shortening:
            content_len = int(max_content_len * (1 - np.exp(-len_shortening_rand_val * exp_bias_lambda))) + 1
            content_len = np.clip(content_len, 1, max_content_len)
        else:
            content_len = max_content_len

        if len(seq) > content_len:
            start_index = np.random.randint(0, len(seq) - content_len + 1)
            clipped_seq = seq[start_index : start_index + content_len]
        else:
            clipped_seq = seq

        final_seq = np.full(target_length, padding_token_id, dtype=np.int64)
        final_seq[0] = sos_token_id
        final_seq[1 : 1 + len(clipped_seq)] = clipped_seq
        final_seq[1 + len(clipped_seq)] = eos_token_id

        return torch.tensor(final_seq, dtype=torch.long)

    return process_sequence(mel_seq), process_sequence(acc_seq)


def deterministic_clip_and_pad(
    mel_sequence: np.ndarray,
    acc_sequence: np.ndarray,
    target_length: int,
    padding_token_id: int = 0,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    start_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    确定性地截取并填充序列，同时在开头和结尾添加 SOS 和 EOS 标记。
    """
    bar_indices = np.where(mel_sequence == BAR_NONE_TOKEN_ID)[0]

    start_bar_idx = start_index
    end_bar_idx = start_bar_idx + EXTRACT_BAR_NUM

    if len(bar_indices) > end_bar_idx:
        mel_start_token = bar_indices[start_bar_idx]
        mel_end_token = bar_indices[end_bar_idx]
    elif len(bar_indices) > start_bar_idx:
        # Not enough bars, take until the end of the sequence
        mel_start_token = bar_indices[start_bar_idx]
        mel_end_token = len(mel_sequence)
    else:
        # Fallback if start_index is out of bounds or no bars found
        mel_start_token = 0
        mel_end_token = 0

    mel_seq = mel_sequence[mel_start_token:mel_end_token]

    # Adapt original logic for accompaniment with a token bias
    acc_start_token = mel_start_token + ACC_BAR_BIAS if mel_start_token + ACC_BAR_BIAS < len(acc_sequence) else mel_start_token
    acc_end_token = mel_end_token + ACC_BAR_BIAS if mel_end_token + ACC_BAR_BIAS < len(acc_sequence) else mel_end_token
    acc_seq = acc_sequence[acc_start_token:acc_end_token]

    max_content_len = target_length - 2  # 为 SOS 和 EOS 标记留出空间

    def process_sequence(seq: np.ndarray) -> torch.Tensor:
        clipped_seq = seq[:max_content_len]
        final_seq = np.full(target_length, padding_token_id, dtype=np.int64)
        final_seq[0] = sos_token_id
        final_seq[1 : 1 + len(clipped_seq)] = clipped_seq
        if 1 + len(clipped_seq) < target_length:
            final_seq[1 + len(clipped_seq)] = eos_token_id

        return torch.tensor(final_seq, dtype=torch.long)

    return process_sequence(mel_seq), process_sequence(acc_seq)

class MelAccRemiJsonDataset(Dataset): # 类名已更改
    def __init__(self, config: MelAccRemiJsonDatasetSchema) -> None:
        # 直接使用 mel_dir 和 acc_dir
        self.mel_dir = config.mel_dir
        self.acc_dir = config.acc_dir
        self.file_pattern = config.file_pattern # 仍然保留 file_pattern 用于 glob

        self.transform = config.transform
        self.max_seq_len = config.max_seq_len
        self.tokenization_type = config.tokenization_type
        self.midi_file_pairs = []
        self.stage = config.stage
        # self.val_start_idx 似乎未被使用，可以考虑移除或说明其用途

        self.midi_file_pairs = self._collect_file_pairs()
        self.midi_file_pairs.sort() # 保持排序

        data_range = config.data_range
        # 应用 data_range 切片逻辑
        if isinstance(data_range[0], int) and isinstance(data_range[1], int):
            start_index, end_index = data_range
        elif isinstance(data_range[0], float) and isinstance(data_range[1], float):
            start_index = int(data_range[0] * len(self.midi_file_pairs))
            end_index = int(data_range[1] * len(self.midi_file_pairs))
        else:
            raise ValueError("data_range must be a tuple of two integers or two floats.")
        self.midi_file_pairs = self.midi_file_pairs[start_index:end_index]


    def _collect_file_pairs(self) -> list[tuple[str, str]]:
        """
        Collects pairs of melody and accompaniment JSON file paths.
        It expects mel_dir and acc_dir to contain files with matching names.
        """
        pairs = []
        mel_files = glob.glob(os.path.join(self.mel_dir, self.file_pattern))

        for mel_path in mel_files:
            file_name = os.path.basename(mel_path)
            acc_path = os.path.join(self.acc_dir, file_name)

            if os.path.exists(acc_path):
                pairs.append((mel_path, acc_path))
            else:
                print(f"Warning: Corresponding accompaniment file not found for {mel_path} at {acc_path}")
        return pairs

    def __len__(self) -> int:
        return len(self.midi_file_pairs)

    def __getitem__(self, idx: int) -> ModelInputData:
        mel_path, acc_path = self.midi_file_pairs[idx]

        try:
            with open(mel_path, "r") as f:
                mel_data = json.load(f)["ids"]
            with open(acc_path, "r") as f:
                acc_data = json.load(f)["ids"]

            # np.array(...).squeeze() 确保数据是 1D 数组
            mel_seq_np = np.array(mel_data).squeeze()
            acc_seq_np = np.array(acc_data).squeeze()

            if self.stage == "train":
                mel_tensor, acc_tensor = random_clip_and_pad(
                    mel_seq_np,
                    acc_seq_np,
                    target_length=self.max_seq_len,
                    padding_token_id=0,
                    padding_probability=0.2,
                    exp_bias_lambda=5.0,
                )
            else: # val, test, predict stages
                mel_tensor, acc_tensor = deterministic_clip_and_pad(
                    mel_seq_np,
                    acc_seq_np,
                    target_length=self.max_seq_len,
                    padding_token_id=0,
                )

            if self.transform:
                mel_tensor = self.transform(mel_tensor)
                acc_tensor = self.transform(acc_tensor)
            return ModelInputData(mel_data=mel_tensor, acc_data=acc_tensor)
        except Exception as e:
            print(f"Error loading JSON files, {mel_path}, {acc_path}: {e}")
            # Raising an error is generally better for debugging than returning None,None
            # as DataLoader will likely fail on None values anyway.
            raise RuntimeError(f"Error processing data for {mel_path} and {acc_path}: {e}")



class MelAccRemiJsonDataModule(pl.LightningDataModule):
    def __init__(self, config: MelAccRemiJsonDataModuleSchema):
        super().__init__()
        self.train_config = config.train_config
        self.val_config = config.val_config
        self.test_config = config.test_config
        self.predict_config = config.predict_config

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            # Setup for training and validation datasets
            if self.train_config:
                self.train = MelAccRemiJsonDataset(self.train_config)
            if self.val_config:
                self.val = MelAccRemiJsonDataset(self.val_config)
        elif stage == "test":
            # Setup for test dataset
            if self.test_config:
                self.test = MelAccRemiJsonDataset(self.test_config)
        elif stage == "predict":
            # Setup for prediction dataset
            if self.predict_config:
                self.predict = MelAccRemiJsonDataset(self.predict_config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.val_config.batch_size,
            num_workers=self.val_config.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        # For prediction, you might want to use the full dataset or a specific subset
        # For now, returning the test_dataloader for demonstration
        return DataLoader(
            self.test,
            batch_size=self.predict_config.batch_size,
            num_workers=self.predict_config.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[ModelInputData]) -> ModelInputData:
        """
        Custom collate function to handle the batch of data.
        This function can be modified to handle different data structures.
        """
        # print( lambda: item.shape for item in batch if item is not None)
        mel_data = [item.mel_data for item in batch if item is not None]
        acc_data = [item.acc_data for item in batch if item is not None]
        if not mel_data or not acc_data:
            raise ValueError("Batch contains no valid data items.")
        mel_data = torch.stack(mel_data)
        acc_data = torch.stack(acc_data)
        return ModelInputData(mel_data=mel_data, acc_data=acc_data)

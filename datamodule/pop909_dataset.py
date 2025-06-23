import os
import json  # Import json module
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from schema.dataset_schema import Pop909DatasetSchema, Pop909DataModuleSchema
from schema.model_io_schema import ModelInputData
import numpy as np
import torch

# def random_clip_and_pad(
#     token_sequence: np.ndarray, target_length: int, padding_token_id: int, padding_probability: float = 0.2, exp_bias_lambda: float = 5.0
# ) -> torch.Tensor:
#     """
#     对一个 Token 序列进行随机截取，并在长度不足时进行填充。
#     可以控制即使对于长序列，也以一定概率引入随机填充。

#     Args:
#         token_sequence (np.ndarray): 输入的 Token 序列，一维 NumPy 数组。
#         target_length (int): 目标长度，即截取或填充后的长度。
#         padding_token_id (int): 用于填充的 Token ID。
#         padding_probability (float): 当序列长度超过 target_length 时，
#                                      以该概率随机截取一个更短的序列并填充，
#                                      而非总是截取 target_length 长度。
#                                      值应在 0.0 到 1.0 之间。

#     Returns:
#         torch.Tensor: 截取并填充后的 Token 序列，长度为 target_length。
#     """
#     current_length = len(token_sequence)
#     clipped_sequence = None
#     if current_length <= target_length:
#         clipped_sequence = token_sequence
#     else:
#         if np.random.rand() < padding_probability:
#             random_val = np.random.rand()
#             actual_clip_length = int(target_length * (1 - np.exp(-random_val * exp_bias_lambda))) + 1
#             actual_clip_length = np.clip(actual_clip_length, 1, target_length)  # 确保在有效范围内
#             if current_length < actual_clip_length:
#                 actual_clip_length = current_length
#             start_index = np.random.randint(0, current_length - actual_clip_length + 1)
#             clipped_sequence = token_sequence[start_index : start_index + actual_clip_length]
#         else:
#             start_index = np.random.randint(0, current_length - target_length + 1)
#             clipped_sequence = token_sequence[start_index : start_index + target_length]

#     padded_sequence = np.full(target_length, padding_token_id, dtype=token_sequence.dtype)
#     padded_sequence[: len(clipped_sequence)] = clipped_sequence

#     return torch.tensor(padded_sequence, dtype=torch.long)  # 返回 PyTorch Tensor


# def clip_and_pad(token_sequence: np.ndarray, start_index: int, target_length: int, padding_token_id: int) -> torch.Tensor:
#     """
#     对一个 Token 序列进行截取，并在长度不足时进行填充。

#     Args:
#         token_sequence (np.ndarray): 输入的 Token 序列，一维 NumPy 数组。
#         start_index (int): 截取的起始索引。
#         target_length (int): 目标长度，即截取或填充后的长度。
#         padding_token_id (int): 用于填充的 Token ID。

#     Returns:
#         torch.Tensor: 截取并填充后的 Token 序列，长度为 target_length。
#     """
#     current_length = len(token_sequence)
#     if start_index < 0 or start_index >= current_length:
#         raise ValueError(f"start_index {start_index} is out of bounds for token_sequence of length {current_length}")
#     if start_index + target_length > current_length:
#         clipped_sequence = token_sequence[start_index:current_length]
#     else:
#         clipped_sequence = token_sequence[start_index : start_index + target_length]
#     padded_sequence = np.full(target_length, padding_token_id, dtype=token_sequence.dtype)
#     padded_sequence[: len(clipped_sequence)] = clipped_sequence

#     return torch.tensor(padded_sequence, dtype=torch.long)  # 返回 PyTorch Tensor


# class REMITokenDataManager:
#     def __init__(
#         self,
#         melody_paths: list[str],
#         accompaniment_paths: list[str],
#         max_seq_len: int,
#         padding_token_id: int,
#         stage: str, # 'train', 'val', 'test'
#         padding_probability: float = 0.0,
#         exp_bias_lambda: float = 5.0,
#     ):
#         self.melody_paths = melody_paths
#         self.accompaniment_paths = accompaniment_paths
#         self.max_seq_len = max_seq_len
#         self.padding_token_id = padding_token_id
#         self.stage = stage
#         self.padding_probability = padding_probability
#         self.exp_bias_lambda = exp_bias_lambda


#     def _precompute_bar_indices(self):
#         """预计算所有歌曲的 Bar_0 Token 索引"""
#         bar_0_token_id = self.tokenizer.vocab.get('Bar_None')
#         for file_idx in range(len(self.full_data_paths)):
#             try:
#                 melody_tokens, _ = self._load_full_tokens(file_idx) # 只加载旋律来找 Bar
#                 self.all_bar_start_indices[file_idx] = [
#                     i for i, token_id in enumerate(melody_tokens) if token_id == bar_0_token_id
#                 ]
#                 if not self.all_bar_start_indices[file_idx]: # 如果没有 Bar_0，至少从开头开始
#                     self.all_bar_start_indices[file_idx].append(0)
#             except Exception as e:
#                 print(f"Error precomputing bar indices for file {self.full_data_paths[file_idx]}: {e}")

#     def _random_clip_and_pad(self, token_sequence: np.ndarray, target_length: int) -> torch.Tensor:
#         current_length = len(token_sequence)
#         clipped_sequence = None

#         if current_length <= target_length:
#             clipped_sequence = token_sequence
#         else:
#             if np.random.rand() < self.padding_probability:
#                 random_val = np.random.rand()
#                 actual_clip_length = int(target_length * (1 - np.exp(-random_val * self.exp_bias_lambda))) + 1
#                 actual_clip_length = np.clip(actual_clip_length, 1, target_length)

#                 start_index = np.random.randint(0, current_length - actual_clip_length + 1)
#                 clipped_sequence = token_sequence[start_index : start_index + actual_clip_length]
#             else:
#                 start_index = np.random.randint(0, current_length - target_length + 1)
#                 clipped_sequence = token_sequence[start_index : start_index + target_length]

#         padded_sequence = np.full(target_length, self.padding_token_id, dtype=token_sequence.dtype)
#         padded_sequence[:len(clipped_sequence)] = clipped_sequence

#         return torch.tensor(padded_sequence, dtype=torch.long)

#     def _deterministic_clip_and_pad(self, token_sequence: np.ndarray, target_length: int) -> torch.Tensor:
#         current_length = len(token_sequence)
#         padded_sequence = np.full(target_length, self.padding_token_id, dtype=token_sequence.dtype)
#         actual_clip_len = min(current_length, target_length)
#         padded_sequence[:actual_clip_len] = token_sequence[:actual_clip_len]
#         return torch.tensor(padded_sequence, dtype=torch.long)

#     def get_segment(self, file_idx: int, segment_start_offset: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         根据文件索引和片段起始偏移量获取处理后的旋律和伴奏片段。
#         如果 stage='train' 且 segment_start_offset 为 None，则内部随机选择。
#         """
#         full_melody_tokens, full_accompaniment_tokens = self._load_full_tokens(file_idx)

#         melody_tokens_np = np.array(full_melody_tokens, dtype=np.longlong)
#         acc_tokens_np = np.array(full_accompaniment_tokens, dtype=np.longlong)

#         shared_start_idx = 0

#         if self.stage == "train":
#             # 训练模式下，如果外部没有指定起始偏移，则内部随机生成
#             if segment_start_offset is None:
#                 min_available_len_for_acc = len(acc_tokens_np) - self.delay_tokens
#                 effective_min_len = min(len(melody_tokens_np), min_available_len_for_acc)
#                 max_possible_start_idx = effective_min_len - self.max_seq_len

#                 if max_possible_start_idx < 0:
#                     shared_start_idx = 0
#                 else:
#                     bar_starts = self.all_bar_start_indices[file_idx]
#                     valid_bar_starts = [
#                         i for i in bar_starts
#                         if i >= 0 and i <= max_possible_start_idx
#                     ]
#                     if valid_bar_starts:
#                         shared_start_idx = np.random.choice(valid_bar_starts)
#                     else:
#                         shared_start_idx = np.random.randint(0, max_possible_start_idx + 1)
#             else: # 如果外部指定了 segment_start_offset (用于特定采样策略)
#                 shared_start_idx = segment_start_offset

#         else: # val / test stages
#             # 验证/测试模式下，segment_start_offset 应该由外部 (Dataset.__init__ 中的 segments_info) 明确指定
#             if segment_start_offset is None:
#                 raise ValueError("In validation/test stage, segment_start_offset must be provided.")
#             shared_start_idx = segment_start_offset

#         # 截取旋律输入片段
#         melody_input_segment = melody_tokens_np[shared_start_idx : shared_start_idx + self.max_seq_len]

#         # 截取伴奏目标片段 (考虑延迟)
#         acc_target_segment = acc_tokens_np[
#             shared_start_idx + self.delay_tokens : shared_start_idx + self.delay_tokens + self.max_seq_len
#         ]

#         # 应用填充和截取逻辑
#         if self.stage == "train":
#             processed_melody = self._random_clip_and_pad(melody_input_segment, self.max_seq_len)
#             processed_accompaniment = self._random_clip_and_pad(acc_target_segment, self.max_seq_len)
#         else:
#             processed_melody = self._deterministic_clip_and_pad(melody_input_segment, self.max_seq_len)
#             processed_accompaniment = self._deterministic_clip_and_pad(acc_target_segment, self.max_seq_len)

#         return processed_melody, processed_accompaniment

BAR_NONE_TOKEN_ID = 4  # Assuming Bar_None is represented by 0 in the tokenizer
EXTRACT_BAR_NUM = 4  # Assuming we want to extract 4 bars
ACC_BAR_BIAS = 1


def random_clip_and_pad(
    mel_sequence: np.ndarray,
    acc_sequence: np.ndarray,
    target_length: int,
    padding_token_id: int=0,
    sos_token_id: int=1,
    eos_token_id: int=2,
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


class Pop909Dataset(Dataset):
    def __init__(self, config: Pop909DatasetSchema) -> None:
        # Removed debugging print statement to avoid cluttering logs
        self.mel_dir = config.mel_dir
        self.acc_dir = config.acc_dir
        self.transform = config.transform
        self.max_seq_len = config.max_seq_len
        self.tokenization_type = config.tokenization_type
        self.midi_file_pairs = []
        self.stage = config.stage
        self.val_start_idx = 0  # Placeholder for validation start index, if needed

        # Collect all MIDI file pairs (now JSON files)
        for i in range(1, 910):  # Assuming POP909 has 909 songs, numbered 001 to 909
            mel_path = os.path.join(self.mel_dir, f"{i:03d}.json")  # Change extension to .json
            acc_path = os.path.join(self.acc_dir, f"{i:03d}.json")  # Change extension to .json

            if os.path.exists(mel_path) and os.path.exists(acc_path):
                self.midi_file_pairs.append((mel_path, acc_path))
            else:
                # For debugging, if some files are missing
                if not os.path.exists(mel_path):
                    print(f"Warning: Melody JSON file not found at {mel_path}")
                if not os.path.exists(acc_path):
                    print(f"Warning: Accompaniment JSON file not found at {acc_path}")
        self.midi_file_pairs.sort()
        data_range = config.data_range
        if isinstance(data_range[0], int) and isinstance(data_range[1], int):
            start_index, end_index = data_range
        elif isinstance(data_range[0], float) and isinstance(data_range[1], float):
            start_index = int(data_range[0] * len(self.midi_file_pairs))
            end_index = int(data_range[1] * len(self.midi_file_pairs))
        self.midi_file_pairs = self.midi_file_pairs[start_index:end_index]  # Slice based on start and end index

    def __len__(self) -> int:
        return len(self.midi_file_pairs)

    def __getitem__(self, idx: int) -> ModelInputData:
        mel_path, acc_path = self.midi_file_pairs[idx]

        try:
            with open(mel_path, "r") as f:
                mel_data = json.load(f)["ids"]
            with open(acc_path, "r") as f:
                acc_data = json.load(f)["ids"]

            if self.stage == "train":
                mel_data, acc_data = random_clip_and_pad(
                    np.array(mel_data).squeeze(),
                    np.array(acc_data).squeeze(),
                    target_length=self.max_seq_len,
                    padding_token_id=0,  # Assuming 0 is the padding token ID
                    padding_probability=0.2,
                    exp_bias_lambda=5.0,
                )
            else:
                # # For validation or test, we can clip from a specific start index
                # start_index = self.val_start_idx if self.stage == "val" else 0
                # start_index = min(start_index, len(mel_data) - self.max_seq_len, len(acc_data) - self.max_seq_len)  # Ensure we don't exceed bounds
                # mel_data = clip_and_pad(
                #     np.array(mel_data),
                #     start_index=start_index,
                #     target_length=self.max_seq_len,
                #     padding_token_id=0  # Assuming 0 is the padding token ID
                # )
                # acc_data = clip_and_pad(
                #     np.array(acc_data),
                #     start_index=start_index,
                #     target_length=self.max_seq_len,
                #     padding_token_id=0  # Assuming 0 is the padding token ID
                # )
                # self.val_start_idx += self.max_seq_len  # Update start index for next validation item

                # start_index = 0  # For validation or test, we can start from the beginning
                # mel_data = clip_and_pad(
                #     np.array(mel_data),
                #     start_index=start_index,
                #     target_length=self.max_seq_len,
                #     padding_token_id=0,  # Assuming 0 is the padding token ID
                # )
                # acc_data = clip_and_pad(
                #     np.array(acc_data),
                #     start_index=start_index,
                #     target_length=self.max_seq_len,
                #     padding_token_id=0,  # Assuming 0 is the padding token ID
                # )

                mel_data, acc_data = deterministic_clip_and_pad(
                    np.array(mel_data).squeeze(),
                    np.array(acc_data).squeeze(),
                    target_length=self.max_seq_len,
                    padding_token_id=0,  # Assuming 0 is the padding token ID
                )

            if self.transform:
                mel_data = self.transform(mel_data)
                acc_data = self.transform(acc_data)
            return ModelInputData(mel_data=mel_data, acc_data=acc_data)
        except Exception as e:
            print(f"Error loading JSON files, {mel_path}, {acc_path}: {e}")
            # Return None or handle the error appropriately
            raise RuntimeError(f"Error loading JSON files: {mel_path}, {acc_path}. Details: {e}")


class Pop909DataModule(pl.LightningDataModule):
    def __init__(self, config: Pop909DataModuleSchema):
        super().__init__()
        self.train_config = config.train_config
        self.val_config = config.val_config
        self.test_config = config.test_config
        self.predict_config = config.predict_config

    def setup(self, stage: str = None) -> None:
        if stage == "fit":
            # Setup for training and validation datasets
            if self.train_config:
                self.train = Pop909Dataset(self.train_config)
            if self.val_config:
                self.val = Pop909Dataset(self.val_config)
        elif stage == "test":
            # Setup for test dataset
            if self.test_config:
                self.test = Pop909Dataset(self.test_config)
        elif stage == "predict":
            # Setup for prediction dataset
            if self.predict_config:
                self.predict = Pop909Dataset(self.predict_config)

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

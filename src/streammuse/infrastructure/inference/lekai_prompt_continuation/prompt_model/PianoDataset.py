# PianoDataset.py — 瘦数据集层
# 只负责: 文件 I/O、数据增强、截断、batching
# 所有 tokenization 逻辑委托给 PianoMusicTokenizer

import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict
from .my_tokenizer import PianoMusicTokenizer


class PianoDataset(Dataset):
    """钢琴音乐数据集，不依赖预计算长度缓存"""

    def __init__(self, data_dir, config, mode='train',
                 test_split_ratio=0.05, random_seed=42, truncate=True):
        self.root_dir = data_dir
        self.max_seq_len = config.train_cutoff_len
        self.truncate = truncate
        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed
        self.tokenizer = PianoMusicTokenizer(config=config)

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        print(f"找到 {len(self.data_files)} 个有效的npz文件")

        self._split_train_test()

    def _split_train_test(self):
        """根据mode划分训练集和测试集"""
        total = len(self.data_files)
        np.random.seed(self.random_seed)
        indices = np.arange(total)
        np.random.shuffle(indices)

        test_size = int(total * self.test_split_ratio)
        train_size = total - test_size

        if self.mode == 'train':
            sel = indices[:train_size]
            print(f"使用训练集: {len(sel)} 个文件 ({train_size}/{total})")
        elif self.mode == 'test':
            sel = indices[train_size:]
            print(f"使用测试集: {len(sel)} 个文件 ({test_size}/{total})")
        else:
            raise ValueError(f"mode必须是'train'或'test'，当前为: {self.mode}")

        self.data_files = [self.data_files[i] for i in sel]

    def __len__(self):
        return len(self.data_files)



    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.data_files[idx])

        # 1. 加载原始数据
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()
        num_measures = metadata['num_measures']

        if num_measures < 3:
            # 不足 3 小节（mel 2 + acc 需要第3小节的额外 beat），跳到下一个样本
            return self.__getitem__((idx + 1) % len(self))

        # 2. 数据增强: 音高移调
        pitch_shift = 0
        if np.random.random() < 0.7:
            pitch_shift = np.random.randint(-5, 6)

        # 3. 收集小节
        measures = [save_dict[f'measure_{i}'] for i in range(num_measures)]

        # 4. 条件生成序列: 2小节 mel(不计loss) → 2小节 acc(计loss)
        input_ids, labels = self.tokenizer.build_conditional_sequence(
            measures=measures,
            metadata=metadata,
            num_condition_bars=2,
            pitch_shift=pitch_shift,
        )

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class StreamingDataset(IterableDataset):
    """流式打包：所有序列拼成连续流，按 max_seq_len 切片，~100% 利用率

    每个 epoch 重新 shuffle，保留底层 Dataset 的数据增强（pitch shift 等）。
    底层 Dataset 应设置 truncate=False，让长序列自然跨 chunk 流动。
    """

    def __init__(self, base_dataset, max_seq_len, pad_token_id, n_estimate=200):
        self.base_dataset = base_dataset
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        # 采样估算总 chunk 数（供 DataLoader.__len__ 和 Accelerate 使用）
        n = min(n_estimate, len(base_dataset))
        sample_indices = np.random.choice(len(base_dataset), n, replace=False)
        total_tokens = sum(len(base_dataset[int(i)]['input_ids']) for i in sample_indices)
        self._estimated_num_chunks = int(len(base_dataset) * (total_tokens / n) / max_seq_len)

    def __len__(self):
        return self._estimated_num_chunks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = np.random.RandomState(torch.initial_seed() % (2**31))
        indices = rng.permutation(len(self.base_dataset)).tolist()

        if worker_info is not None:
            indices = indices[worker_info.id::worker_info.num_workers]

        buffer_ids = []
        buffer_labels = []
        buffer_seq_ids = []   # 每个 token 所属的序列编号
        seq_counter = 1

        for idx in indices:
            sample = self.base_dataset[idx]
            tokens = sample['input_ids'].tolist()
            labels = sample['labels'].tolist()

            buffer_ids.extend(tokens)
            buffer_labels.extend(labels)
            buffer_seq_ids.extend([seq_counter] * len(tokens))
            seq_counter += 1

            while len(buffer_ids) >= self.max_seq_len:
                chunk_seq_ids = buffer_seq_ids[:self.max_seq_len]

                # 重编号：chunk 内从 1 开始连续递增
                _, inverse = np.unique(chunk_seq_ids, return_inverse=True)
                renumbered = (inverse + 1).tolist()

                # position_ids：每条序列内部从 0 递增
                position_ids = []
                prev_sid = -1
                pos = 0
                for sid in renumbered:
                    if sid != prev_sid:
                        pos = 0
                        prev_sid = sid
                    position_ids.append(pos)
                    pos += 1

                yield {
                    'input_ids': torch.tensor(buffer_ids[:self.max_seq_len], dtype=torch.long),
                    'labels': torch.tensor(buffer_labels[:self.max_seq_len], dtype=torch.long),
                    'attention_mask': torch.tensor(renumbered, dtype=torch.long),
                    'position_ids': torch.tensor(position_ids, dtype=torch.long),
                }
                buffer_ids = buffer_ids[self.max_seq_len:]
                buffer_labels = buffer_labels[self.max_seq_len:]
                buffer_seq_ids = buffer_seq_ids[self.max_seq_len:]
        # 尾部不足一个 chunk 的 token 丢弃

    def estimate_num_batches(self, batch_size):
        """估算每 epoch 的 batch 数"""
        return max(1, self._estimated_num_chunks // batch_size)


class PaddingCollator:
    """简单 padding：变长序列填充到 batch 内最大长度（用于测试集）"""

    def __init__(self, max_seq_len, pad_token_id):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(f['input_ids']) for f in features), self.max_seq_len)

        batch_ids, batch_labels, batch_mask = [], [], []
        for f in features:
            ids = f['input_ids'][:max_len]
            lbl = f['labels'][:max_len]
            seq_len = len(ids)
            pad_len = max_len - seq_len

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
                mask = torch.cat([torch.ones(seq_len, dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
            else:
                mask = torch.ones(seq_len, dtype=torch.long)

            batch_ids.append(ids)
            batch_labels.append(lbl)
            batch_mask.append(mask)

        return {
            'input_ids': torch.stack(batch_ids),
            'labels': torch.stack(batch_labels),
            'attention_mask': torch.stack(batch_mask),
        }

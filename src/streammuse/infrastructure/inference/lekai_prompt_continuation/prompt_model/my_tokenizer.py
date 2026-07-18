"""
PianoMusicTokenizer - 钢琴音乐完整编解码系统

三层架构：
  Layer 1 (PatchCodec):       Piano roll <-> patch token matrix (三进制编码)
  Layer 2 (Beat encoding):    Measure (4,88,t) -> 逐拍压缩 token 列表
  Layer 3 (SequenceBuilder):  Measures + metadata -> 训练/推理就绪的 token 序列

序列格式 (无条件生成):
  [BOS][TS][BPM] [bar][beat][TRK_MEL][mel...][TRK_ACC][acc...] [beat]... [bar]... [EOS]

编码特点:
  - 轨道标签在内容前面 (TRK_X 作为起始标记)
  - 旋律在前，伴奏在后
  - 相对位置编码：只编码非零音高，用相对距离标记位置
  - 空拍压缩为 [TRK_X, EMPTY]
  - 无条件自回归生成，所有 token 参与 loss
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass


# ============================================================================
#  Vocabulary — 所有 token ID 的唯一定义处
# ============================================================================

@dataclass(frozen=True)
class Vocabulary:
    """
    集中管理所有 token ID，消除 magic number。
    frozen=True 防止运行时意外修改。
    """
    # --- Patch codec 标记 ---
    empty_marker: int = 169
    track_marker_acc: int = 170       # acc 轨起始标记
    track_marker_mel: int = 171       # mel 轨起始标记
    beat_marker: int = 172            # 拍分隔符

    # --- Codec 参数 ---
    marker_offset: int = 81           # 相对位置标记起始 ID (81-168)
    measures_length: int = 88
    img_h: int = 88

    # --- 序列级 token ---
    bar_token_id: int = 255
    eos_token_id: int = 256
    bos_token_id: int = 257           # codec 不用，保留给 AR 生成
    pad_token_id: int = 258
    time_sig_offset_id: int = 259     # codec 不用，保留给 AR 生成
    bpm_offset_id: int = 264          # = 259 + 5, codec 不用

    # --- BPM 分桶阈值 ---
    bpm_slow_threshold: int = 90
    bpm_fast_threshold: int = 200

    # --- Patch 默认尺寸 ---
    default_patch_h: int = 1
    default_patch_w: int = 4

    @classmethod
    def from_config(cls, config) -> 'Vocabulary':
        """从 ModelConfig 创建 Vocabulary，用 config 值覆盖默认值。"""
        return cls(
            track_marker_acc=config.track_marker_acc_id,
            track_marker_mel=config.track_marker_mel_id,
            beat_marker=config.beat_marker_id,
            bar_token_id=config.bar_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.bos_token_id,
            pad_token_id=config.pad_token_id,
            time_sig_offset_id=config.time_sig_offset_id,
            bpm_offset_id=config.bpm_offset_id,
            marker_offset=config.marker_offset,
            empty_marker=config.empty_patch_id,
            measures_length=config.measures_length,
            default_patch_h=config.patch_h,
            default_patch_w=config.patch_w,
        )


# ============================================================================
#  PatchCodec — Layer 1: piano roll <-> patch token matrix
# ============================================================================

class PatchCodec:
    """
    底层 patch 编解码器。
    处理 (sustain, onset) 双通道 piano roll 与三进制 patch token 矩阵之间的转换。
    """

    def __init__(
        self,
        patch_h: int = 1,
        patch_w: int = 4,
        img_h: int = 88,
    ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.img_h = img_h

        self.patch_size = patch_h * patch_w
        self.powers_3 = 3 ** np.arange(self.patch_size - 1, -1, -1)

        # strict 模式的特殊 token 替换规则
        self.special_token_ids = [
            13, 12, 59, 31, 64, 11, 55, 73, 37, 30,
            28, 5, 15, 46, 16, 17, 10, 14, 32, 19,
            3, 9, 1, 57, 4
        ]
        self.replacement_ids = [0, 67, 7, 40, 63]

    # ---- 编码 ----

    def image_to_patch_tokens(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True,
    ) -> np.ndarray:
        """
        双通道 piano roll → patch token 矩阵（三进制编码）。

        Args:
            image: (2, 88, t) — ch0: sustain, ch1: onset
            strict_mode: 是否替换特殊 token

        Returns:
            tokens: (num_time_patches, num_pitch_patches)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain = image[0].copy()
        onset = image[1].copy()
        onset[sustain == 0] = 0

        img_h, img_w = sustain.shape

        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain = np.pad(sustain, ((0, 0), (0, padding_w)), constant_values=0)
            onset = np.pad(onset, ((0, 0), (0, padding_w)), constant_values=0)
            img_w = sustain.shape[1]

        n_rows = img_h // self.patch_h
        n_cols = img_w // self.patch_w

        sustain_p = self._reshape_to_patches(sustain, n_rows, n_cols)
        onset_p = self._reshape_to_patches(onset, n_rows, n_cols)

        combined = sustain_p.astype(np.int64) + onset_p.astype(np.int64)
        tokens = np.dot(combined, self.powers_3)

        if strict_mode:
            tokens = self._replace_special_tokens(tokens)

        return tokens

    def _reshape_to_patches(self, ch: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
        p = ch.reshape(n_rows, self.patch_h, n_cols, self.patch_w)
        p = p.transpose(2, 0, 1, 3)
        return p.reshape(n_cols, n_rows, self.patch_size)

    def _replace_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        mask = np.isin(tokens, self.special_token_ids)
        if np.any(mask):
            tokens = tokens.copy()
            tokens[mask] = np.random.choice(self.replacement_ids, size=int(np.sum(mask)))
        return tokens

    # ---- 解码 ----

    def patch_tokens_to_image(self, tokens: np.ndarray) -> np.ndarray:
        """
        token 矩阵 → 双通道 piano roll (2, 88, t)。
        """
        n_cols, n_rows = tokens.shape

        combined = np.zeros((n_cols, n_rows, self.patch_size), dtype=np.int64)
        tmp = tokens.copy()
        for i in range(self.patch_size):
            combined[:, :, i] = tmp // self.powers_3[i]
            tmp = tmp % self.powers_3[i]

        sustain_p = (combined >= 1).astype(np.float32)
        onset_p = (combined == 2).astype(np.float32)

        sustain_ch = self._patches_to_channel(sustain_p, n_cols, n_rows)
        onset_ch = self._patches_to_channel(onset_p, n_cols, n_rows)

        return np.stack([sustain_ch, onset_ch], axis=0)

    def _patches_to_channel(self, patches: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
        p = patches.reshape(n_cols, n_rows, self.patch_h, self.patch_w)
        p = p.transpose(1, 2, 0, 3)
        return p.reshape(self.img_h, n_cols * self.patch_w)


# ============================================================================
#  PianoMusicTokenizer — 完整的音乐 tokenizer
# ============================================================================

class PianoMusicTokenizer:
    """
    钢琴音乐的完整 tokenizer。

    组合 PatchCodec + Vocabulary，提供从原始 piano roll 到训练就绪
    token 序列的全部编解码功能。这是所有 tokenization 操作的唯一入口。

    序列格式 (无条件生成):
      [BOS][TS][BPM] [bar][beat][TRK_MEL][mel...][TRK_ACC][acc...] [beat]... [bar]... [EOS]
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        config=None,
    ):
        if vocab is None and config is not None:
            self.vocab = Vocabulary.from_config(config)
        elif vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()

        self._codec = PatchCodec(
            patch_h=self.vocab.default_patch_h,
            patch_w=self.vocab.default_patch_w,
            img_h=self.vocab.img_h,
        )

    # ===================== Layer 1: 相对位置压缩 / 解压 =====================

    def compress_tokens(
        self,
        token_matrix: np.ndarray,
        track_marker: int,
    ) -> np.ndarray:
        """
        使用相对位置编码压缩 token 矩阵，轨道标记在内容前面。

        非空行: [TRACK_MARKER, POS+delta, value, POS+delta, value, ...]
        空行:   [TRACK_MARKER, EMPTY_MARKER]

        Args:
            token_matrix: (num_time_patches, measures_length) 的 token 矩阵
            track_marker: 轨道起始标记 (track_marker_acc 或 track_marker_mel)

        Returns:
            compressed: 压缩后的一维序列
        """
        v = self.vocab

        compressed_seqs = []
        for row in token_matrix:
            nz = np.where(row != 0)[0]
            if len(nz) == 0:
                compressed_seqs.append(
                    np.array([track_marker, v.empty_marker], dtype=np.int64))
            else:
                parts = [track_marker]
                prev = 0
                for idx in nz:
                    parts.extend([v.marker_offset + (idx - prev), row[idx]])
                    prev = idx
                compressed_seqs.append(np.array(parts, dtype=np.int64))

        return np.concatenate(compressed_seqs)

    def decompress_tokens(
        self,
        compressed: Union[np.ndarray, list],
        track_marker_id: int,
    ) -> np.ndarray:
        """
        相对位置 token 序列 → token 矩阵。

        Args:
            compressed: 压缩 token 序列
            track_marker_id: 轨道起始标记 ID

        Returns:
            (num_time_patches, measures_length)
        """
        v = self.vocab
        if isinstance(compressed, list):
            compressed = np.array(compressed, dtype=np.int64)

        rows = []
        i = 0

        while i < len(compressed):
            if compressed[i] == track_marker_id:
                i += 1
                row = np.zeros(v.measures_length, dtype=np.int64)
                abs_pos = 0
                while i < len(compressed) and compressed[i] != track_marker_id:
                    tok = compressed[i]
                    if tok == v.empty_marker:
                        # 空行标记，跳过
                        i += 1
                        break
                    if v.marker_offset <= tok < v.marker_offset + v.measures_length:
                        # 相对位置标记
                        abs_pos += tok - v.marker_offset
                        i += 1
                        if i < len(compressed) and compressed[i] != track_marker_id:
                            if 0 <= abs_pos < v.measures_length:
                                row[abs_pos] = compressed[i]
                            i += 1
                    else:
                        # 非预期 token，跳过
                        break
                rows.append(row)
            else:
                i += 1

        if len(rows) == 0:
            return np.zeros((0, v.measures_length), dtype=np.int64)

        return np.stack(rows, axis=0)

    # ===================== Layer 2: Beat 级编码 =====================

    def encode_measure(
        self,
        measure: np.ndarray,
        timesteps_per_beat: int = 4,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        将 4 通道 measure 编码为逐拍 (mel, acc) 固定展开 token 对。

        Args:
            measure: (4, 88, t) — ch0:2 = mel(part0), ch2:4 = acc(part1)
            timesteps_per_beat: 每拍的时间步数

        Returns:
            beats: List[(mel_tensor, acc_tensor)] — 每元素对应一拍
        """
        v = self.vocab
        t = measure.shape[2]
        beat_len = timesteps_per_beat
        num_beats = (t + beat_len - 1) // beat_len

        beats = []

        for b in range(num_beats):
            s = b * beat_len
            e = min(s + beat_len, t)
            beat = measure[:, :, s:e]

            if e - s < beat_len:
                pad_w = beat_len - (e - s)
                beat = np.pad(beat, ((0, 0), (0, 0), (0, pad_w)), constant_values=0)

            # 翻转音高轴: 高音在前 (idx 0 = C8, idx 87 = A0)
            beat = beat[:, ::-1, :].copy()

            # mel: channels 0-1
            tokens_mel = self._codec.image_to_patch_tokens(beat[:2], strict_mode=True)
            comp_mel = self.compress_tokens(tokens_mel, track_marker=v.track_marker_mel)

            # acc: channels 2-3
            tokens_acc = self._codec.image_to_patch_tokens(beat[2:], strict_mode=True)
            comp_acc = self.compress_tokens(tokens_acc, track_marker=v.track_marker_acc)

            beats.append((
                torch.tensor(comp_mel, dtype=torch.long),
                torch.tensor(comp_acc, dtype=torch.long),
            ))

        return beats

    def encode_bpm(self, bpm) -> int:
        """BPM 值 → token ID（已包含 offset）。"""
        v = self.vocab
        if bpm is None:
            bucket = 3  # UNK
        else:
            bpm_int = int(bpm)
            if bpm_int < v.bpm_slow_threshold:
                bucket = 0
            elif bpm_int <= v.bpm_fast_threshold:
                bucket = 1
            else:
                bucket = 2
        return bucket + v.bpm_offset_id

    def encode_time_sig(self, time_sig_idx: int) -> int:
        """拍号索引 → token ID（已包含 offset，含 2/2 拍特殊映射）。"""
        if time_sig_idx == 9:
            time_sig_idx = 4
        return time_sig_idx + self.vocab.time_sig_offset_id

    # ===================== 内部: 编码所有小节 =====================

    def _encode_measures(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        timesteps_per_beat: int = 4,
        pitch_shift: int = 0,
    ) -> Tuple[int, int, bool, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        编码所有小节，返回按小节分组的 (mel, acc) 拍对。

        Returns:
            (ts_token, bpm_token, is_continuation, measure_beats)
            measure_beats: List[List[(mel_tensor, acc_tensor)]]
                第一层按小节分组，第二层按拍分组
        """
        ts_token = self.encode_time_sig(metadata['time_signature_idx'])
        bpm_token = self.encode_bpm(metadata['bpm'])
        is_continuation = metadata.get('is_continuation', False)

        measure_beats = []
        for measure in measures:
            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            beats = self.encode_measure(measure, timesteps_per_beat)
            measure_beats.append(beats)

        return ts_token, bpm_token, is_continuation, measure_beats

    # ===================== Layer 3: 完整序列构建 =====================

    def build_training_sequence(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        add_bos: bool = True,
        timesteps_per_beat: int = 4,
        pitch_shift: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建完整的训练序列 (input_ids, labels)。

        序列格式 (无条件生成):
          [BOS][TS][BPM] [bar][beat][TRK_MEL][mel...][TRK_ACC][acc...] [beat]... [bar]... [EOS]

        无条件生成：所有 token 参与 loss (labels = input_ids)。

        Returns:
            (input_ids, labels): 均为 1D torch.long tensor。
        """
        v = self.vocab
        ts_token, bpm_token, is_continuation, measure_beats = self._encode_measures(
            measures, metadata, timesteps_per_beat, pitch_shift)

        parts = []

        for beats in measure_beats:
            # [bar]
            parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))

            for mel, acc in beats:
                # [beat_marker]
                parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
                # [TRK_MEL][mel content...] — mel 在前
                parts.append(mel)
                # [TRK_ACC][acc content...] — acc 在后
                parts.append(acc)

        content = torch.cat(parts)

        # [BOS] + TS + BPM + content + [EOS]
        seq_parts = []

        if add_bos:
            seq_parts.append(torch.tensor([v.bos_token_id], dtype=torch.long))

        seq_parts.append(torch.tensor([ts_token], dtype=torch.long))
        seq_parts.append(torch.tensor([bpm_token], dtype=torch.long))
        seq_parts.append(content)

        if not is_continuation:
            seq_parts.append(torch.tensor([v.eos_token_id], dtype=torch.long))

        input_ids = torch.cat(seq_parts)
        labels = input_ids.clone()

        return input_ids, labels

    def build_conditional_sequence(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        num_condition_bars: int = 2,
        extra_acc_beats: int = 4,
        timesteps_per_beat: int = 4,
        pitch_shift: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        条件生成序列: mel 作为条件 → 生成 acc（acc 比 mel 多 extra_acc_beats 拍）。

        序列格式:
          [BOS][TS][BPM] [mel num_condition_bars 小节] [acc num_condition_bars 小节 + extra beats] [EOS]
                         ←── -100 不计loss ──────────→ ←── 计loss ──────────────────────────────→

        mel 部分: [bar][beat][TRK_MEL mel...]... × num_condition_bars
        acc 部分: [bar][beat][TRK_ACC acc...]... × num_condition_bars + 下一小节的前 extra_acc_beats 拍
        """
        v = self.vocab

        # acc 需要多一小节的数据来提取额外的 beat
        acc_bars_needed = num_condition_bars + (1 if extra_acc_beats > 0 else 0)
        mel_measures = measures[:num_condition_bars]
        acc_measures = measures[:acc_bars_needed]

        ts_token, bpm_token, _, mel_measure_beats = self._encode_measures(
            mel_measures, metadata, timesteps_per_beat, pitch_shift)
        _, _, _, acc_measure_beats = self._encode_measures(
            acc_measures, metadata, timesteps_per_beat, pitch_shift)

        # mel: num_condition_bars 小节
        mel_parts = []
        for beats in mel_measure_beats:
            mel_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
            for mel, acc in beats:
                mel_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
                mel_parts.append(mel)

        # acc: num_condition_bars 小节 + 下一小节的前 extra_acc_beats 拍
        acc_parts = []
        for bar_idx, beats in enumerate(acc_measure_beats):
            acc_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
            for beat_idx, (mel, acc) in enumerate(beats):
                # 最后一小节只取前 extra_acc_beats 拍
                if bar_idx == num_condition_bars and beat_idx >= extra_acc_beats:
                    break
                acc_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
                acc_parts.append(acc)

        mel_content = torch.cat(mel_parts)
        acc_content = torch.cat(acc_parts)

        # [BOS TS BPM] + mel + acc + [EOS]
        prefix = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)
        eos = torch.tensor([v.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prefix, mel_content, acc_content, eos])

        # labels: 条件部分 = -100, 目标部分 = 真实 token
        condition_len = len(prefix) + len(mel_content)
        labels = input_ids.clone()
        labels[:condition_len] = -100

        return input_ids, labels

    # ===================== 序列解析 =====================

    def parse_generated_sequence(
        self,
        tokens: Union[torch.Tensor, list],
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        解析生成的 token 序列，提取 mel 和 acc 的逐拍 token 列表。

        从序列中按 TRK_MEL / TRK_ACC 标记分段提取各轨道内容。
        相对位置编码格式，每段长度可变。

        Args:
            tokens: 生成的完整 token 序列

        Returns:
            (mel_beats, acc_beats): 各轨道的逐拍 token 列表
        """
        v = self.vocab
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        mel_beats = []
        acc_beats = []

        structural = {v.beat_marker, v.bar_token_id, v.eos_token_id, v.bos_token_id}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == v.track_marker_mel:
                # 收集 mel tokens 直到 TRK_ACC 或结构标记
                beat_tokens = [tok]
                i += 1
                while i < len(tokens) and tokens[i] != v.track_marker_acc and tokens[i] not in structural:
                    beat_tokens.append(tokens[i])
                    i += 1
                mel_beats.append(beat_tokens)
            elif tok == v.track_marker_acc:
                # 收集 acc tokens 直到 TRK_MEL 或结构标记
                beat_tokens = [tok]
                i += 1
                while i < len(tokens) and tokens[i] != v.track_marker_mel and tokens[i] not in structural:
                    beat_tokens.append(tokens[i])
                    i += 1
                acc_beats.append(beat_tokens)
            else:
                i += 1

        return mel_beats, acc_beats

    # ===================== 解码 =====================

    def decode_beats_to_pianoroll(
        self,
        beats_list: list,
        track_marker_id: int,
    ) -> np.ndarray:
        """
        beat token 列表 → piano roll (2, 88, t)。

        Args:
            beats_list: beat token 的列表（tensor / list / int 混合均可）
            track_marker_id: 轨道起始标记 ID（track_marker_acc 或 track_marker_mel）

        Returns:
            pianoroll: (2, 88, t)
        """
        v = self.vocab

        # 展平
        flat = []
        for beat in beats_list:
            if isinstance(beat, torch.Tensor):
                flat.extend(beat.cpu().tolist())
            elif isinstance(beat, (list, np.ndarray)):
                flat.extend(beat if isinstance(beat, list) else beat.tolist())
            else:
                flat.append(beat)

        # 过滤掉 >= beat_marker 的结构标记（bar, bos, eos, pad, ts, bpm 等）
        # 保留: patch tokens(0-80), position markers(81-168), empty(169), track markers(170-171)
        filtered = np.array([t for t in flat if t < v.beat_marker], dtype=np.int64)

        if len(filtered) == 0:
            return np.zeros((2, v.img_h, 0), dtype=np.float32)

        # 解压 → token 矩阵 → piano roll
        mat = self.decompress_tokens(filtered, track_marker_id=track_marker_id)
        pr = self._codec.patch_tokens_to_image(mat)
        # 翻转音高轴还原: 编码时高音在前，解码后需翻回低音在前
        return pr[:, ::-1, :].copy()

    # ===================== 工具方法 =====================

    def estimate_sequence_length(
        self,
        measures: List[np.ndarray],
        timesteps_per_beat: int = 4,
    ) -> int:
        """
        估算 token 序列长度（用于长度缓存预计算）。

        格式: [BOS][TS][BPM] + {[bar] + num_beats × ([beat] + mel + acc)} × num_measures + [EOS]
        """
        total = 4  # BOS + time_sig + BPM + EOS

        for measure in measures:
            beats = self.encode_measure(measure, timesteps_per_beat)
            total += 1  # bar token
            for mel, acc in beats:
                total += 1  # beat_marker
                total += len(mel)
                total += len(acc)

        return total

    def get_config(self) -> dict:
        """返回配置字典。"""
        v = self.vocab
        return {
            'patch_h': v.default_patch_h,
            'patch_w': v.default_patch_w,
            'measures_length': v.measures_length,
            'track_marker_acc': v.track_marker_acc,
            'track_marker_mel': v.track_marker_mel,
            'beat_marker': v.beat_marker,
            'empty_marker': v.empty_marker,
            'img_h': v.img_h,
        }

    def __repr__(self) -> str:
        v = self.vocab
        return (
            f"PianoMusicTokenizer("
            f"patch={v.default_patch_h}x{v.default_patch_w}, "
            f"img_h={v.img_h}, "
            f"empty_marker={v.empty_marker})"
        )

import music21 as m21
import numpy as np
from typing import List, Tuple, Dict, Optional

import torch


class MusicXMLtoPianoRoll:
    def __init__(self, resolution: int = 16):
        """
        初始化转换器
        Args:
            resolution: 时间分辨率，默认16表示1/16音符为最小单位
        """
        self.resolution = resolution
        self.quarter_length = 4
        self.pitch_range = 88
        self.min_pitch = 21
        self.max_pitch = 108
        
    def get_time_signature_idx(self, time_signature: m21.meter.TimeSignature) -> int:
        """将拍号转换为索引"""
        ts_map = {
            '4/4': 0, '3/4': 1, '2/4': 2, '6/8': 3, '3/8': 4,
            '5/4': 5, '7/4': 6, '9/8': 7, '12/8': 8, '2/2': 9,
            '6/4': 10, '1/4': 11, '5/8': 12, '7/8': 13
        }
        return ts_map.get(time_signature.ratioString, -1)
    
    def get_key_signature_idx(self, key_signature: m21.key.KeySignature) -> int:
        """将调号转换为索引"""
        if key_signature is None:
            return -1  # 占位符
        
        # 用sharps数量作为索引 (-7到+7, 0=C major/A minor)
        # -7=7个降号, 0=无升降号, +7=7个升号
        return key_signature.sharps
    
    def get_tempo_info(self, score: m21.stream.Score) -> tuple:
        """
        提取速度信息
        Returns:
            (bpm_value, tempo_text): BPM数值和速度术语
        """
        # 查找速度标记
        tempo_marks = score.flatten().getElementsByClass(m21.tempo.TempoIndication)
        
        if tempo_marks:
            tempo = tempo_marks[0]
            
            # 获取BPM数值
            bpm_value = None
            if hasattr(tempo, 'number') and tempo.number:
                bpm_value = tempo.number
            elif hasattr(tempo, 'numberImplicit') and tempo.numberImplicit:
                bpm_value = tempo.numberImplicit
            
            # 获取速度术语
            tempo_text = None
            if hasattr(tempo, 'text') and tempo.text:
                tempo_text = tempo.text
            elif isinstance(tempo, m21.tempo.MetronomeMark):
                # MetronomeMark可能没有文字描述
                tempo_text = None
            elif hasattr(tempo, 'name'):
                # 一些预定义的速度术语如Allegro, Andante等
                tempo_text = tempo.name
            
            return bpm_value, tempo_text
        
        # 如果没有找到速度标记，返回占位符
        return None, None
    
    def get_absolute_note_positions(self, score: m21.stream.Score, measure_boundaries: List[Dict]) -> List[Dict]:
        """获取所有音符的绝对位置信息，并标记连音符组"""
        notes_info = []
        # 使用索引作为key，避免多乐章文件中小节编号重复导致的覆盖问题
        measure_time_map = {mb['index']: mb for mb in measure_boundaries}

        for part_idx, part in enumerate(score.parts):
            active_ties = {}
            tie_group_id = 0

            for measure_idx, measure in enumerate(part.getElementsByClass('Measure')):
                if measure_idx not in measure_time_map:
                    continue

                measure_start = measure_time_map[measure_idx]['start']
                
                for element in measure.flatten().notes:
                    if isinstance(element, (m21.note.Note, m21.chord.Chord)):
                        abs_start = measure_start + int(element.offset * self.quarter_length)
                        duration = int(element.duration.quarterLength * self.quarter_length)
                        
                        if isinstance(element, m21.note.Note):
                            pitches = [element.pitch.midi]
                            ties = [element.tie]
                        else:
                            pitches = [p.midi for p in element.pitches]
                            ties = [None] * len(pitches)
                        
                        for pitch, tie in zip(pitches, ties):
                            if self.min_pitch <= pitch <= self.max_pitch:
                                note_data = {
                                    'pitch': pitch,
                                    'abs_start': abs_start,
                                    'duration': duration,
                                    'tie_group': None,
                                    'is_tie_start': False
                                }
                                
                                if tie:
                                    if tie.type == 'start':
                                        tie_group_id += 1
                                        active_ties[pitch] = tie_group_id
                                        note_data['tie_group'] = tie_group_id
                                        note_data['is_tie_start'] = True
                                    elif tie.type in ['continue', 'stop']:
                                        if pitch in active_ties:
                                            note_data['tie_group'] = active_ties[pitch]
                                            note_data['is_tie_start'] = False
                                            if tie.type == 'stop':
                                                del active_ties[pitch]
                                
                                notes_info.append(note_data)
        
        return notes_info
    
    def convert(self, musicxml_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        将MusicXML文件转换为piano roll片段列表
        
        Returns:
            pianoroll_segments: 小节片段列表，每个片段shape为(2, 88, t)
            metadata: 元数据字典（包含BPM、调号、拍号等所有信息）
        """
        score = m21.converter.parse(musicxml_path)
        
        parts = score.parts
        # 获取第一个拍号
        first_time_sig = None
        for part in parts:
            for measure in part.getElementsByClass('Measure'):
                if measure.timeSignature:
                    first_time_sig = measure.timeSignature
                    break
            if first_time_sig:
                break
        
        
        time_signature_idx = self.get_time_signature_idx(first_time_sig)
        
        # 获取调号
        key_sig = None
        for part in parts:
            key_sigs = part.flatten().getElementsByClass('KeySignature')
            if key_sigs:
                key_sig = key_sigs[0]
                break
        
        key_signature_idx = self.get_key_signature_idx(key_sig)
        
        # 获取BPM和速度信息
        bpm_value, tempo_text = self.get_tempo_info(score)
        
        # 计算小节边界
        measure_boundaries = []
        current_time = 0

        for measure_idx, measure in enumerate(parts[0].getElementsByClass('Measure')):
            measure_duration_ticks = int(measure.duration.quarterLength * self.quarter_length)
            measure_boundaries.append({
                'index': measure_idx,  # 使用索引避免多乐章小节编号重复问题
                'number': measure.number,
                'start': current_time,
                'end': current_time + measure_duration_ticks,
                'duration': measure_duration_ticks
            })
            current_time += measure_duration_ticks
        
        total_length = current_time
        
        # 获取所有音符信息并生成完整piano roll
        notes_info = self.get_absolute_note_positions(score, measure_boundaries)
        
        full_onset = np.zeros((self.pitch_range, total_length), dtype=np.float32)
        full_sustain = np.zeros((self.pitch_range, total_length), dtype=np.float32)
        
        for note in notes_info:
            pitch_idx = note['pitch'] - self.min_pitch
            start = note['abs_start']
            if start >= total_length:
                continue
            end = min(start + note['duration'], total_length)
            
            full_sustain[pitch_idx, start:end] = 1.0
            
            if note['tie_group'] is None or note['is_tie_start']:
                full_onset[pitch_idx, start] = 1.0
        
        # 按小节边界切分
        pianoroll_segments = []
        for mb in measure_boundaries:
            segment = np.stack([
                full_sustain[:, mb['start']:mb['end']],
                full_onset[:, mb['start']:mb['end']]
            ], axis=0)
            pianoroll_segments.append(segment)
        
        metadata = {
            'time_signature': first_time_sig.ratioString,
            'time_signature_idx': time_signature_idx,
            'key_signature_idx': key_signature_idx,
            'key_signature': key_sig.asKey().name if key_sig else None,
            'bpm': bpm_value,
            'tempo_text': tempo_text,
            'num_measures': len(pianoroll_segments),
            'resolution': self.resolution,
            'total_length': total_length
        }
        
        return pianoroll_segments, metadata


def save_pianoroll_segments(segments: List[np.ndarray], 
                           metadata: Dict,
                           output_path: str):
    """保存piano roll片段到npz文件"""
    save_dict = {f'measure_{i}': segment.astype(np.uint8) for i, segment in enumerate(segments)}
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def load_pianoroll_segments(npz_path: str) -> Tuple[List[np.ndarray], Dict]:
    """从npz文件加载piano roll片段"""
    data = np.load(npz_path, allow_pickle=True)
    return data['measure'].tolist(), data['metadata'].item()



# 使用示例
def main():
    # 创建转换器实例
    converter = MusicXMLtoPianoRoll(resolution=16)
    
    # 转换MusicXML文件
    musicxml_file = "/home/lab-wei.zhenao/boyu/Dataset/successful_xmls/103370.musicxml"  # 替换为你的文件路径
    
    try:
        segments, metadata = converter.convert(musicxml_file)
        visualize_pianoroll(segments, measure_indices=[1,2,3,4,5,6])

        
        print(f"转换成功！")
        print(f"第一个小节拍号: {metadata['time_signature']}")
        print(f"小节数量: {metadata['num_measures']}")
        print(f"总长度(1/16音符): {metadata['total_length']}")
            
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()


# 可视化工具
def visualize_pianoroll(segments: List[np.ndarray], measure_indices: List[int] = None):
    """
    可视化多个小节的piano roll
    
    Args:
        segments: piano roll片段列表
        measure_indices: 要可视化的小节索引列表（默认前4个）
    """
    import matplotlib.pyplot as plt
    
    if measure_indices is None:
        measure_indices = list(range(min(8, len(segments))))
    
    n_measures = len(measure_indices)
    fig, axes = plt.subplots(n_measures, 2, figsize=(15, 4*n_measures))
    
    if n_measures == 1:
        axes = axes.reshape(1, -1)
    
    for idx, measure_idx in enumerate(measure_indices):
        segment = segments[measure_idx]

        n_time = segment.shape[2]  # 横向时间步数量，例如 16
        x_ticks = list(range(n_time + 1))  # 网格线要从0到n_time

        # Sustain通道
        axes[idx, 0].imshow(
            segment[0],
            aspect='auto',
            origin='lower',
            cmap='Blues',
            interpolation='nearest',
            extent=[0, n_time, 0, segment.shape[1]]   # 👈 关键
        )
        axes[idx, 0].set_title(f'Measure {measure_idx+1} - Sustain')
        axes[idx, 0].set_ylabel('Pitch (0-87)')
        axes[idx, 0].set_xlabel('Time (1/16 notes)')
        axes[idx, 0].set_xticks(x_ticks)
        axes[idx, 0].grid(True, axis='both', linestyle='--', linewidth=0.5)  # 👈 打开网格
        axes[idx, 0].set_xlim(0, n_time)

        # Onset通道
        axes[idx, 1].imshow(
            segment[1],
            aspect='auto',
            origin='lower',
            cmap='Reds',
            interpolation='nearest',
            extent=[0, n_time, 0, segment.shape[1]]   # 👈 同样
        )
        axes[idx, 1].set_title(f'Measure {measure_idx+1} - Onset')
        axes[idx, 1].set_ylabel('Pitch (0-87)')
        axes[idx, 1].set_xlabel('Time (1/16 notes)')
        axes[idx, 1].set_xticks(x_ticks)
        axes[idx, 1].grid(True, axis='both', linestyle='--', linewidth=0.5)
        axes[idx, 1].set_xlim(0, n_time)

    plt.tight_layout()
    plt.show()
    plt.savefig('pianoroll_visualization.png')


# 保存和加载功能
def save_pianoroll_segments(segments: List[np.ndarray], 
                           time_signature_idx: int,
                           metadata: Dict,
                           output_path: str):
    """保存piano roll片段到npz文件"""
    save_dict = {
        f'measure_{i}': segment for i, segment in enumerate(segments)
    }
    save_dict['time_signature_idx'] = time_signature_idx
    save_dict['metadata'] = metadata
    
    np.savez_compressed(output_path, **save_dict)
    print(f"已保存到 {output_path}")


def load_pianoroll_segments(npz_path: str) -> Tuple[List[np.ndarray], int, Dict]:
    """从npz文件加载piano roll片段"""
    data = np.load(npz_path, allow_pickle=True)
    
    segments = []
    i = 0
    while f'measure_{i}' in data:
        segments.append(data[f'measure_{i}'])
        i += 1
    
    time_signature_idx = int(data['time_signature_idx'])
    metadata = data['metadata'].item()
    
    return segments, time_signature_idx, metadata

if __name__ == "__main__":
    main()
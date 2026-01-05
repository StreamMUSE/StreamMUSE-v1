import music21 as m21
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil
#nohup setsid python batch_process.py > training.log 2>&1 &

class MusicXMLtoPianoRoll:
    RESOLUTION = 16
    QUARTER_LENGTH = 4
    PITCH_RANGE = 88
    MIN_PITCH = 21
    MAX_PITCH = 108
    
    @staticmethod
    def get_time_signature_idx(time_signature: m21.meter.TimeSignature) -> int:
        ts_map = {
            '4/4': 0, '3/4': 1, '2/4': 2, '6/8': 3, '3/8': 4,
            '5/4': 5, '7/4': 6, '9/8': 7, '12/8': 8, '2/2': 9,
            '6/4': 10, '1/4': 11, '5/8': 12, '7/8': 13
        }
        return ts_map.get(time_signature.ratioString, -1)
    
    @staticmethod
    def get_key_signature_idx(key_signature: m21.key.KeySignature) -> int:
        return key_signature.sharps if key_signature else -1
    
    @staticmethod
    def get_tempo_info(score: m21.stream.Score) -> tuple:
        tempo_marks = score.flatten().getElementsByClass(m21.tempo.TempoIndication)
        if tempo_marks:
            tempo = tempo_marks[0]
            bpm_value = getattr(tempo, 'number', None) or getattr(tempo, 'numberImplicit', None)
            tempo_text = getattr(tempo, 'text', None) or getattr(tempo, 'name', None)
            return bpm_value, tempo_text
        return None, None
    
    @staticmethod
    def extract_part_notes(part: m21.stream.Part, measure_boundaries: List[Dict], 
                          quarter_length: int) -> List[Dict]:
        """提取单个part的所有音符信息"""
        notes_info = []
        measure_time_map = {mb['number']: mb for mb in measure_boundaries}
        active_ties = {}
        tie_group_id = 0
        
        for measure in part.getElementsByClass('Measure'):
            if measure.number not in measure_time_map:
                continue
            measure_start = measure_time_map[measure.number]['start']
            
            for element in measure.flatten().notes:
                if isinstance(element, (m21.note.Note, m21.chord.Chord)):
                    abs_start = measure_start + int(element.offset * quarter_length)
                    duration = int(element.duration.quarterLength * quarter_length)
                    
                    pitches = [element.pitch.midi] if isinstance(element, m21.note.Note) else [p.midi for p in element.pitches]
                    ties = [element.tie] if isinstance(element, m21.note.Note) else [None] * len(pitches)
                    
                    for pitch, tie in zip(pitches, ties):
                        if MusicXMLtoPianoRoll.MIN_PITCH <= pitch <= MusicXMLtoPianoRoll.MAX_PITCH:
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
                                        if tie.type == 'stop':
                                            del active_ties[pitch]
                            
                            notes_info.append(note_data)
        
        return notes_info
    
    @staticmethod
    def notes_to_pianoroll(notes_info: List[Dict], total_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """将音符信息转换为pianoroll (sustain, onset)"""
        sustain = np.zeros((MusicXMLtoPianoRoll.PITCH_RANGE, total_length), dtype=np.float32)
        onset = np.zeros((MusicXMLtoPianoRoll.PITCH_RANGE, total_length), dtype=np.float32)
        
        for note in notes_info:
            pitch_idx = note['pitch'] - MusicXMLtoPianoRoll.MIN_PITCH
            start = note['abs_start']
            if start >= total_length:
                continue
            end = min(start + note['duration'], total_length)
            
            sustain[pitch_idx, start:end] = 1.0
            if note['tie_group'] is None or note['is_tie_start']:
                onset[pitch_idx, start] = 1.0
        
        return sustain, onset
    
    @staticmethod
    def convert(musicxml_path: str, resolution: int = 16) -> Tuple[List[np.ndarray], Dict]:
        """将MusicXML转换为双轨道4-channel piano roll片段"""
        score = m21.converter.parse(musicxml_path)
        parts = score.parts
        
        # 验证
        measure_count = len(parts[0].getElementsByClass('Measure'))
        if not (16 <= measure_count <= 300):
            raise ValueError(f"Skipped: {measure_count} measures (need 16-300)")
        
        if len(parts) < 2:
            raise ValueError("Skipped: Need at least 2 parts")
        
        # 获取拍号
        first_time_sig = None
        for part in parts:
            for measure in part.getElementsByClass('Measure'):
                if measure.timeSignature:
                    first_time_sig = measure.timeSignature
                    break
            if first_time_sig:
                break
        
        first_time_sig = first_time_sig or m21.meter.TimeSignature('4/4')
        
        if first_time_sig.ratioString not in ['2/2', '3/4', '4/4', '2/4', '6/8']:
            raise ValueError(f"Skipped: Time signature {first_time_sig.ratioString} not allowed")
        
        # 计算小节边界
        quarter_length = resolution // 4
        measure_boundaries = []
        current_time = 0
        
        for measure in parts[0].getElementsByClass('Measure'):
            measure_duration_ticks = int(measure.duration.quarterLength * quarter_length)
            measure_boundaries.append({
                'number': measure.number,
                'start': current_time,
                'end': current_time + measure_duration_ticks,
                'duration': measure_duration_ticks
            })
            current_time += measure_duration_ticks
        
        total_length = current_time
        
        # 分别处理两个轨道
        part0_notes = MusicXMLtoPianoRoll.extract_part_notes(parts[0], measure_boundaries, quarter_length)
        part1_notes = MusicXMLtoPianoRoll.extract_part_notes(parts[1], measure_boundaries, quarter_length)
        
        # 生成pianoroll
        p0_sustain, p0_onset = MusicXMLtoPianoRoll.notes_to_pianoroll(part0_notes, total_length)
        p1_sustain, p1_onset = MusicXMLtoPianoRoll.notes_to_pianoroll(part1_notes, total_length)
        
        # 按小节切分：4 channels = [p0_sustain, p0_onset, p1_sustain, p1_onset]
        pianoroll_segments = []
        for mb in measure_boundaries:
            segment = np.stack([
                p0_sustain[:, mb['start']:mb['end']],
                p0_onset[:, mb['start']:mb['end']],
                p1_sustain[:, mb['start']:mb['end']],
                p1_onset[:, mb['start']:mb['end']]
            ], axis=0)  # shape: (4, 88, time_steps)
            pianoroll_segments.append(segment)
        
        # 元数据
        key_sig = None
        for part in parts:
            key_sigs = part.flatten().getElementsByClass('KeySignature')
            if key_sigs:
                key_sig = key_sigs[0]
                break
        
        bpm_value, tempo_text = MusicXMLtoPianoRoll.get_tempo_info(score)
        
        metadata = {
            'time_signature': first_time_sig.ratioString,
            'time_signature_idx': MusicXMLtoPianoRoll.get_time_signature_idx(first_time_sig),
            'key_signature_idx': MusicXMLtoPianoRoll.get_key_signature_idx(key_sig),
            'key_signature': key_sig.asKey().name if key_sig else None,
            'bpm': bpm_value,
            'tempo_text': tempo_text,
            'num_measures': len(pianoroll_segments),
            'resolution': resolution,
            'total_length': total_length,
            'num_parts': 2,
            'num_channels': 4
        }
        
        return pianoroll_segments, metadata


def save_pianoroll_segments(segments: List[np.ndarray], metadata: Dict, output_path: str):
    """保存到npz"""
    save_dict = {f'measure_{i}': segment.astype(np.uint8) for i, segment in enumerate(segments)}
    save_dict['metadata'] = metadata
    np.savez_compressed(output_path, **save_dict)


def load_pianoroll_segments(npz_path: str) -> Tuple[List[np.ndarray], Dict]:
    """从npz加载"""
    data = np.load(npz_path, allow_pickle=True)
    measures = [data[f'measure_{i}'] for i in range(len(data.files) - 1)]
    return measures, data['metadata'].item()


def process_single_file(args):
    """处理单个文件"""
    xml_path, output_dir, resolution = args
    
    try:
        xml_name = Path(xml_path).stem
        output_path = os.path.join(output_dir, f"{xml_name}.npz")
        
        if os.path.exists(output_path):
            return xml_path, "skipped", "Already exists"
        
        segments, metadata = MusicXMLtoPianoRoll.convert(xml_path, resolution)
        save_pianoroll_segments(segments, metadata, output_path)
        
        return xml_path, "success", metadata
        
    except Exception as e:
        return xml_path, "error", str(e)


def batch_process_musicxml(input_dir: str, output_dir: str, xml_copy_dir: str = None,
                          resolution: int = 16, max_workers: int = 8, timeout_seconds: int = 30):
    """批量处理"""
    os.makedirs(output_dir, exist_ok=True)
    if xml_copy_dir:
        os.makedirs(xml_copy_dir, exist_ok=True)
    
    xml_files = []
    for ext in ['*.xml', '*.musicxml', '*.mxl']:
        xml_files.extend(Path(input_dir).glob(ext))
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files")
    
    args_list = [(str(xml_file), output_dir, resolution) for xml_file in xml_files]
    results = {'success': [], 'error': [], 'timeout': [], 'skipped': []}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, args): args[0] for args in args_list}
        
            
        for future in as_completed(future_to_file):
            xml_path = future_to_file[future]
            
            try:
                file_path, status, info = future.result(timeout=timeout_seconds)
                
                if status == "success":
                    results['success'].append(file_path)
                    if xml_copy_dir:
                        try:
                            dest_path = os.path.join(xml_copy_dir, Path(file_path).name)
                            shutil.copy2(file_path, dest_path)
                        except Exception:
                            pass
                            
                elif status == "error":
                    results['error'].append((file_path, info))
                elif status == "skipped":
                    results['skipped'].append(file_path)
                    
            except Exception as e:
                results['error'].append((xml_path, str(e)))
                
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Success: {len(results['success'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Errors: {len(results['error'])}")
    print(f"Timeouts: {len(results['timeout'])}")
    
    if xml_copy_dir:
        print(f"Successfully copied XMLs to: {xml_copy_dir}")
    
    return results


# 使用示例
if __name__ == "__main__":
    # 单文件测试
    # segments, metadata = MusicXMLtoPianoRoll.convert("your_file.xml")
    # print(f"Segment shape: {segments[0].shape}")  # 应该是 (4, 88, time_steps)
    
    input_folder = f"/home/lab-wei.zhenao/boyu/Dataset/successful_xmls"
    output_folder = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track"

    results = batch_process_musicxml(
        input_dir=input_folder,
        output_dir=output_folder,
        xml_copy_dir=None,
        resolution=16,
        max_workers=32,
        timeout_seconds=10
    )
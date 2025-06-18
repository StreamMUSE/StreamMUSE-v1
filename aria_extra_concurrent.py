# 导入必要的库
from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import glob
import concurrent.futures # <--- 导入并发库
from tqdm import tqdm

# --- 您的 extract_midi 和 save_notes_to_midi 函数保持不变 ---
# (我将它们放在这里以便代码完整)

def extract_midi(midi_file_path: str) -> Tuple[Score, List[Note], List[Note]]:
    try:
        score = Score.from_midi(midi_file_path)
    except Exception as e:
        # 在多进程中，最好不要打印太多，可以返回错误信息
        # print(f"Error loading MIDI file '{midi_file_path}': {e}")
        return None, [], []

    right_hand_notes: List[Note] = []
    left_hand_notes: List[Note] = []
    C4_PITCH = 60

    for track in score.tracks:
        for note in track.notes:
            if note.pitch >= C4_PITCH:
                right_hand_notes.append(note)
            else:
                left_hand_notes.append(note)

    return score, right_hand_notes, left_hand_notes

def save_notes_to_midi(
    notes: list[Note],
    original_score: Score,
    output_path: str,
    track_name: str
):
    new_score = Score(ticks_per_quarter=original_score.ticks_per_quarter)
    track = Track(name=track_name, program=0, is_drum=False)
    track.notes.extend(notes)
    track.notes.sort(key=lambda note: note.start)
    new_score.tracks.append(track)
    try:
        # 目录创建可以放在主逻辑中一次性完成，这里可以省略
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_score.dump_midi(output_path)
    except Exception as e:
        print(f"Error saving MIDI file to '{output_path}': {e}")

# --- 封装的“工作函数” ---
def process_single_file(input_path: str, args):
    """
    处理单个文件的完整逻辑，方便并行调用。
    """
    try:
        original_score, right_hand_notes, left_hand_notes = extract_midi(input_path)

        if original_score is None or (not right_hand_notes and not left_hand_notes):
            return f"Skipped (no data): {input_path}"

        # --- 计算输出路径 ---
        left_hand_dir = os.path.join(args.output_dir, "acc")
        right_hand_dir = os.path.join(args.output_dir, "mel")
        
        relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
        output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

        right_hand_output_path = os.path.join(right_hand_dir, output_filename)
        left_hand_output_path = os.path.join(left_hand_dir, output_filename)
        
        # --- 保存文件 ---
        if right_hand_notes:
            save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Right Hand")
        if left_hand_notes:
            save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Left Hand")
            
        return f"Success: {input_path}"
    except Exception as e:
        return f"Failed: {input_path} with error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract right-hand (>=C4) and left-hand (<C4) notes from MIDI files concurrently.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    # 添加一个控制进程数的参数
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes to use. Defaults to the number of CPU cores.")
    args = parser.parse_args()

    left_hand_dir = os.path.join(args.output_dir, "acc")
    right_hand_dir = os.path.join(args.output_dir, "mel")

    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    print(f"Searching for MIDI files in '{args.input_dir}'...")
    search_pattern = os.path.join(args.input_dir, '**', '*.mid')
    all_midi_files = glob.glob(search_pattern, recursive=True)
    search_pattern_midi = os.path.join(args.input_dir, '**', '*.midi')
    all_midi_files.extend(glob.glob(search_pattern_midi, recursive=True))

    if not all_midi_files:
        print(f"No MIDI files found in '{args.input_dir}'.")
    else:
        print(f"Found {len(all_midi_files)} total MIDI files.")
        
        # --- 这里是并发处理的核心 ---
        print(f"\nProcessing {len(all_midi_files)} files using multiple processes...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # 创建一个 future 列表，将每个文件的处理任务提交给进程池
            futures = [executor.submit(process_single_file, midi_file, args) for midi_file in all_midi_files]
            
            # 使用 tqdm 来显示处理进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_midi_files)):
                # (可选) 您可以在这里处理每个任务返回的结果，例如记录失败的文件
                # print(future.result())
                pass

        print("\n--- Concurrency Process Complete! ---")
        print(f"Left hand files saved in: '{left_hand_dir}'")
        print(f"Right hand files saved in: '{right_hand_dir}'")
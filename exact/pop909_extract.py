import os
from mido import MidiFile, MidiTrack, Message, merge_tracks
from tqdm import tqdm
import shutil

# Input and output directories
INPUT_DIR = 'POP909-Dataset/POP909'          # Replace with actual path
OUTPUT_DIR = 'datasets/Separated-POP909-Dataset'          # Where to save the melody-only MIDI files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# def extract_acc_track(mid_path, output_path):
#     midi = MidiFile(mid_path)
#     for i, track in enumerate(midi.tracks):
#         if 'piano' in track.name.lower() or 'bridge' in track.name.lower():
#             new_midi = MidiFile()
#             new_track = MidiTrack()
#             new_midi.tracks.append(new_track)
#             for msg in track:
#                 new_track.append(msg)
#             new_midi.save(output_path)
#             print(f"Saved acc track to: {output_path}")
#             return
#     print(f"No acc track found in: {mid_path}")

# def process_POP909_dataset(input_dir, output_dir):
#     output_mel_dir = os.path.join(output_dir, 'mel')
#     output_acc_dir = os.path.join(output_dir, 'acc')
#     os.makedirs(output_mel_dir, exist_ok=True)
#     os.makedirs(output_acc_dir, exist_ok=True)
#     for dirname in tqdm(os.listdir(input_dir)):
#         dir_path = os.path.join(input_dir, dirname)
#         if os.path.isdir(dir_path):
#             for filename in os.listdir(dir_path):
#                 if filename.endswith('.mid'):
#                     mid_path = os.path.join(dir_path, filename)
#                     output_mel_path = os.path.join(output_mel_dir, f"{filename}")
#                     output_acc_path = os.path.join(output_acc_dir, f"{filename}")
                    
#                     # Extract melody track
#                     extract_acc_track(mid_path, output_acc_path)

#                     # Copy the original MIDI file to the melody directory
#                     shutil.copy(mid_path, output_mel_path)
#                     print(f"Copied melody track to: {output_mel_path}")


def extract_and_merge_tracks(mid_path, output_path, track_keywords: list[str]):
    """
    查找MIDI文件中所有轨道名称包含任一指定关键字的轨道，
    将它们合并，并保存到一个新的MIDI文件中。
    """
    try:
        midi = MidiFile(mid_path)
    except Exception as e:
        print(f"Error reading MIDI file {mid_path}: {e}")
        return

    # 找到所有轨道名包含任一关键字的轨道
    tracks_to_merge = [track for track in midi.tracks if any(keyword in track.name.lower() for keyword in track_keywords)]

    if not tracks_to_merge:
        print(f"No tracks with keywords {track_keywords} found in: {mid_path}")
        return

    new_midi = MidiFile(type=0, ticks_per_beat=midi.ticks_per_beat)
    merged_track = merge_tracks(tracks_to_merge)

    # --- 新增代码：将所有音符统一到一个乐器（钢琴） ---
    # 过滤掉所有原有的乐器变更（program_change）消息
    messages_without_instrument_change = [msg for msg in merged_track if msg.type != "program_change"]

    # 创建一个新的轨道，并在开头插入一个单一的乐器定义（program=0，即声学大钢琴）
    final_track = MidiTrack()
    final_track.append(Message("program_change", program=0, time=0))
    final_track.extend(messages_without_instrument_change)
    # --- 新增代码结束 ---

    # --- 再次新增：将所有消息强制统一到通道0 ---
    # 这是解决 miditok 检测到多个相同乐器的关键
    track_on_channel_0 = MidiTrack()
    for msg in final_track:
        # 如果消息有 'channel' 属性，就将其设置为0
        if hasattr(msg, "channel"):
            msg.channel = 0
        track_on_channel_0.append(msg)
    # --- 再次新增结束 ---

    # Uncomment the following line for debugging during development
    # print(len(final_track))
    new_midi.tracks.append(final_track)

    new_midi.save(output_path)
    print(f"Saved {len(tracks_to_merge)} merged '{','.join(track_keywords)}' track(s) to: {output_path}")

def process_POP909_dataset(input_dir, output_dir):
    output_mel_dir = os.path.join(output_dir, "mel")
    output_acc_dir = os.path.join(output_dir, "acc")
    output_orig_dir = os.path.join(output_dir, "original")  # 为原始文件创建新目录

    os.makedirs(output_mel_dir, exist_ok=True)
    os.makedirs(output_acc_dir, exist_ok=True)
    os.makedirs(output_orig_dir, exist_ok=True)  # 创建新目录

    for dirname in tqdm(os.listdir(input_dir)):
        dir_path = os.path.join(input_dir, dirname)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".mid"):
                    mid_path = os.path.join(dir_path, filename)

                    # 定义所有输出文件的路径
                    output_mel_path = os.path.join(output_mel_dir, filename)
                    output_acc_path = os.path.join(output_acc_dir, filename)
                    output_orig_path = os.path.join(output_orig_dir, filename)

                    # 提取旋律轨道 (通常名为 'MELODY')
                    extract_and_merge_tracks(mid_path, output_mel_path, track_keywords=["melody"])

                    # 提取并合并伴奏轨道 ('piano', 'bridge')
                    extract_and_merge_tracks(mid_path, output_acc_path, track_keywords=["piano", "bridge"])

                    # 将原始MIDI文件复制到 'original' 目录
                    shutil.copy(mid_path, output_orig_path)
                    print(f"Copied original file to: {output_orig_path}")

if __name__ == "__main__":
    process_POP909_dataset(INPUT_DIR, OUTPUT_DIR)
    
    
    
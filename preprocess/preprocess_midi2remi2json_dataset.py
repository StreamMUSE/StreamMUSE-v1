import os
from miditok import REMI  # 假设您使用 REMI 编码
from miditok.midi_tokenizer import MusicTokenizer  # 导入 MIDITokenizer 基类以进行类型提示
from pathlib import Path


def preprocess_midi_dataset(root_dir: str, output_dir: str, tokenizer: MusicTokenizer = REMI()):
    """
    Preprocess MIDI files in the specified directory and save the tokenized output.

    Args:
        root_dir (str): Directory containing MIDI files.
        output_dir (str): Directory to save the processed MIDI files.
        tokenizer (MusicTokenizer): Tokenizer to use for processing MIDI files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    midi_files = [Path(root_dir) / f for f in os.listdir(root_dir) if f.endswith(".mid")]

    if isinstance(tokenizer, REMI):
        tokenizer.tokenize_dataset(midi_files, output_dir)
    else:
        raise NotImplementedError("Only REMI tokenizer is currently supported for preprocessing.")


if __name__ == "__main__":
    ROOT_MELODY_DIR = "datasets/Seperated-POP909-Dataset/mel"
    OUTPUT_MELODY_DIR = "datasets/Processed-POP909-Dataset-REMI/mel"
    ROOT_ACC_DIR = "datasets/Seperated-POP909-Dataset/acc"
    OUTPUT_ACC_DIR = "datasets/Processed-POP909-Dataset-REMI/acc"
    preprocess_midi_dataset(ROOT_MELODY_DIR, OUTPUT_MELODY_DIR)
    preprocess_midi_dataset(ROOT_ACC_DIR, OUTPUT_ACC_DIR)

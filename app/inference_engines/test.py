import numpy as np
import torch
from transformer_engine import TransformerInferenceEngine
from preprocess.xf_midi import XFMidi

def midi_to_node(midi_path, min_pitch=0, max_pitch=127, beat_div=4, program=None):
    """
    用 XFMidi 读取 midi 文件，返回 notes 列表，每个元素是 {'pitch', 'tick', 'duration'} 字典。
    """
    midi = XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    notes = []
    for inst in midi.instruments:
        # 如果只想要某种 program，可以加判断
        if program is not None and inst.program != program:
            continue
        for note in inst.notes:
            if min_pitch <= note.pitch <= max_pitch:
                # tick = round(time * resolution)
                start_tick = int(round(note.start))
                end_tick = int(round(note.end))
                if start_tick >= 0 and end_tick < 1163:
                    notes.append({
                        'pitch': note.pitch,
                        'tick': start_tick,
                        'duration': end_tick - start_tick
                    })
    # 按tick排序
    notes.sort(key=lambda x: x['tick'])
    
    return notes

if __name__ == "__main__":
    midi_path_mel = '/home/ubuntu/ugrip/stanleyz/StreamMUSE/input/mel/001.mid'
    midi_path_acc = '/home/ubuntu/ugrip/stanleyz/StreamMUSE/input/acc/001.mid'
    notes_mel = midi_to_node(midi_path_mel, min_pitch=0, max_pitch=127)
    notes_acc = midi_to_node(midi_path_acc, min_pitch=0, max_pitch=127)
    print("notes after midi_to_node():", notes_acc)
    print("Number of notes:", len(notes_acc))

    # Example usage of TransformerInferenceEngine
    model_path = '/home/ubuntu/ugrip/stanleyz/StreamMUSE/results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt'
    engine = TransformerInferenceEngine(checkpoint_path=model_path)
    rolls_mel = engine._notes_to_rolls(notes_mel, max_tick=1163, program=0)
    rolls_acc = engine._notes_to_rolls(notes_acc, max_tick=1163, program=1)
    print("rolls acc shape:", rolls_acc.shape)
    # 保存到 log 文件
    # log_path = "rolls_acc_log.txt"
    # with open(log_path, "w") as f:
    #     f.write(f"rolls shape: {rolls_acc.shape}\n")
    #     for row in rolls_acc.tolist():
    #         f.write(str(row) + "\n")
    # print(f"rolls 已保存到 {log_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_mel_raw = rolls_mel.unsqueeze(0).to(device)
    x_acc_raw = rolls_acc.unsqueeze(0).to(device)
    x_mel, x_acc = engine.model.preprocess(x_mel_raw.long(), pitch_shift=torch.zeros(1, dtype=torch.int8).to(device), y=x_acc_raw.long())
    # log_path = "acc_log.txt"
    # with open(log_path, "w") as f:
    #     f.write(f"rolls shape: {x_acc.shape}\n")
    #     for row in x_acc.tolist():
    #         f.write(str(row) + "\n")
    # print(f"rolls 已保存到 {log_path}")

    batch_size, seq_len, subseq_len = x_mel.shape
    stacked = torch.stack([x_acc, x_mel], dim=2)
    x = stacked.view(batch_size, seq_len * 2, subseq_len)

    with torch.no_grad():
        output_tensors = engine.model.global_sampling(x, x_mel_gt=None, temperature=1, max_seq_len=engine.generation_length_frames)

    
    # # Assuming you have some input data for the model
    # input_data = torch.randn(1, 10, 512)  # Example input tensor
    # output = engine.infer(input_data)
    # print(output)
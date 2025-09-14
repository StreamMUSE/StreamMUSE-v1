import sys
import numpy as np
import xf_midi
from settings import RWC_DATASET_PATH, LA_DATASET_PATH, NOTTINGHAM_DATASET_PATH
import os
from joblib import Parallel, delayed
import torch
import shutil
import json
import pretty_midi
import argparse

sys.path.append("/home/ubuntu/stanleyz/")
from StreamMUSE.m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
# from preprocess_midi2pt_dataset import preprocess_midi

midi_path = "input/mel/001.mid"

tokenize_dict = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
tokenize_count = [-1, -1, -1]

DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])

def preprocess_midi(midi_path, max_polyphony, beat_div=4, ins_ids='all'):
    print(midi_path)
    try:
        midi = xf_midi.XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    except Exception as e:
        print(f"Error processing {midi_path}: Invalid MIDI file. Error: {e}")
        return None
    print(midi)
    midi_end_time = int(midi.get_end_time())
    print("midi_end_time:", midi_end_time)
    if midi_end_time <= 0:
        return None
    if not isinstance(ins_ids, list):
        ins_ids = [ins_ids]
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    print("durantion boundaries:", duration_boundaries)
    min_pitch = 127
    max_pitch = 0
    result_rolls = []
    print("ins_ids:", ins_ids)
    for ins_id in ins_ids:
        has_any_note = False
        rolls = np.full((midi_end_time, max_polyphony, 3), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(midi_end_time, dtype=np.uint8)
        for i, ins in enumerate(midi.instruments):
            print(i, ins)
            program = ins.program
            if ins.is_drum:
                program = 127
            for note in ins.notes:
                start_time = int(round(note.start))
                end_time = int(round(note.end))
                if start_time >= 0 and end_time < midi_end_time and polyphony_counts[start_time] < max_polyphony:
                    if ins.is_drum:
                        duration = 0
                    else:
                        duration = np.searchsorted(duration_boundaries, end_time - start_time) #为了做四舍五入的quantize
                        min_pitch = min(min_pitch, note.pitch)
                        max_pitch = max(max_pitch, note.pitch)
                    add_note = False
                    if ins_id == 'all':
                        add_note = True
                    elif isinstance(ins_id, int):
                        raise NotImplementedError
                    elif isinstance(ins_id, str):
                        if '-' in ins_id:
                            task, num = ins_id.split('-')
                            num = int(num)
                            if task == 'track':
                                add_note = i == num
                            elif task == 'upto':
                                add_note = i <= num
                            elif task == 'from':
                                add_note = i >= num
                            elif task == 'notrack':
                                add_note = i != num
                            else:
                                raise NotImplementedError
                        elif ins_id == 'drum':
                            add_note = ins.is_drum
                        elif ins_id == 'nondrum':
                            add_note = not ins.is_drum
                        elif ins_id == 'empty':
                            add_note = False
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    if add_note:
                        has_any_note = True
                        rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration]
                        # [program, pitch, duration]
                        polyphony_counts[start_time] += 1
        if not has_any_note and ins_id != 'empty':
            return None  # invalid midi file
        for i in range(midi_end_time):
            # Sort notes by ins first, then by pitch, then by duration
            rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][np.lexsort((rolls[i, :polyphony_counts[i], 2], rolls[i, :polyphony_counts[i], 1], rolls[i, :polyphony_counts[i], 0]))]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254  # EOS token
        result_rolls.append(rolls)
        print("polyphony_counts:", polyphony_counts[0:100])
    result_rolls = np.concatenate(result_rolls, axis=1) 
    print(result_rolls)
    # Get song-level pitch shift range
    pitch_shift_max = 127 - max_pitch
    pitch_shift_min = -min_pitch
    print(midi_path, ": final", torch.tensor(result_rolls.reshape(midi_end_time, -1)).shape, torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8).shape)
    return torch.tensor(result_rolls.reshape(midi_end_time, -1)), torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8)

def decompress(model, byte_arr_mel, byte_arr_acc):
    x = torch.tensor(byte_arr_mel).unsqueeze(0)
    x = x.cuda()
    y = torch.tensor(byte_arr_acc).unsqueeze(0)
    y = y.cuda()
    return model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda(), y=y)

test_res1_mel = preprocess_midi(midi_path, 4)
test_res1_acc = preprocess_midi(midi_path.replace("mel", "acc"), 4)

# print("test_res1_mel:", test_res1_mel[0], test_res1_mel[1])
print("test_res1_mel shape:", test_res1_mel[0].shape, test_res1_mel[1].shape)
# 保存到 log 文件
# log_path = "test_res1_mel_log.txt"
# with open(log_path, "w") as f:
#     f.write(f"rolls shape: {test_res1_mel[0].shape}\n")
#     for row in test_res1_mel[0].tolist():
#         f.write(str(row) + "\n")
# print(f"rolls 已保存到 {log_path}")

# print("test_res1_acc:", test_res1_acc[0], test_res1_acc[1])
print("test_res1_acc shape:", test_res1_acc[0].shape, test_res1_acc[1].shape)
# 保存到 log 文件
# log_path = "test_res1_acc_log.txt"
# with open(log_path, "w") as f:
#     f.write(f"rolls shape: {test_res1_acc[0].shape}\n")
#     for row in test_res1_acc[0].tolist():
#         f.write(str(row) + "\n")
# print(f"rolls 已保存到 {log_path}")

model_path = "results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt"
model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path)

test_res2_mel, test_res2_acc = decompress(model, test_res1_mel[0], test_res1_acc[0])
# print("test_res2_mel:", test_res2_mel)
print("test_res2_mel shape:", test_res2_mel.shape)

# # print("test_res2_acc:", test_res2_acc)
print("test_res2_acc shape:", test_res2_acc.shape)
# log_path = "test_res2_acc.txt"
# with open(log_path, "w") as f:
#     f.write(f"test_res2_acc shape: {test_res2_acc.shape}\n")
#     for row in test_res2_acc.tolist():
#         f.write(str(row) + "\n")
# print(f"rolls 已保存到 {log_path}")

first_timestep = 1 # prompt_length != 0
prompt_length = 5

x_mel_gt = test_res2_mel.clone()
print("x_mel_gt shape:", x_mel_gt.shape)
test_res2_mel = test_res2_mel[:, :prompt_length]
print("test_res2_mel shape after slicing:", test_res2_mel.shape)
test_res2_acc = test_res2_acc[:, :prompt_length]
print("test_res2_acc shape after slicing:", test_res2_acc.shape)
batch_size, seq_len, subseq_len = test_res2_mel.shape  # 10*384*8
stacked = torch.stack([test_res2_acc, test_res2_mel], dim=2)
x = stacked.view(batch_size, seq_len * 2, subseq_len)
print("x:", x)
print("x shape:", x.shape)

output = model.global_sampling(x, temperature=1.0, max_seq_len=2)
print("output:", output)
print("output len:", len(output))
output_0 = [output[j][0 : 0 + 1, :] for j in range(len(output))]
print("output:", output_0)
print("output len:", len(output_0))

output_tensor = torch.cat(output_0, dim=0).unsqueeze(0)
print("output_tensor shape:", output_tensor.shape)
print("output_tensor:", output_tensor)

print("x_mel_gt 0-6:", x_mel_gt[:, :prompt_length+1])
x_mel_i = x_mel_gt[:, prompt_length]
output_tensor[0, -1] = x_mel_i
print("output_tensor after setting x_mel_i:", output_tensor)
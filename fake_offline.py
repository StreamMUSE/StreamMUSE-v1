import os
import torch
from m2a_transformer_inference import continuation, decompress, decode_output
from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from preprocess.preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES
import pretty_midi
import argparse
from typing import Literal
import numpy as np
import pdb

def fake_offline_sampling(
    model_path,
    init_midi_path,
    n_rounds=5,
    n_samples=1,
    prompt_len=75,
    temperature=1.0,
    gen_interval_ticks=1,  # every {gen_interval_ticks} logical ticks, we send a request
    gen_seq_len=2,  # in each round, we generate {gen_seq_len} logical ticks
    latency=0,  # means each round, we throw the first {latency} logical ticks
    save_dir="offline_fake"
):
    # 加载模型
    if "small" in model_path:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=False)
    else:
        model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=True)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True) # make sure the save directory exists
    
    # set the input MIDI path, and check if the init_midi_path is a valid MIDI file
    midi_path = init_midi_path 
    if os.path.isfile(midi_path):
        pass
    else:
        print(f"Error: {midi_path} is not a valid file.")

    # midi file -> a list of subseq: [num_logical_ticks, max_polyphony*3]
    byte_arr_mel = preprocess_midi(midi_path, 4)
    byte_arr_acc = preprocess_midi(midi_path.replace("mel", "acc"), 4)
    # check if the preprocessing returned None
    if byte_arr_mel is None:
        print(f"Error: preprocess_midi returned None for mel file: {midi_path}")
        return  # Skip this MIDI file
    if byte_arr_acc is None:
        print(f"Error: preprocess_midi returned None for acc file: {midi_path.replace('mel', 'acc')}")
        return  # Skip this MIDI file

    # a list of subseq: [num_logical_ticks, max_polyphony*3] 
    # ->
    # [1, num_logical_ticks, max_polyphony*2]
    x_mel, x_acc = decompress(model, byte_arr_mel[0], byte_arr_acc[0])

    first_timestep = 1 # prompt_length != 0
    prompt_length = prompt_len

    # clone the original x_acc, 
    # so that we can use them in the fake inference rounds
    x_acc_pool = x_acc.clone()
    x_mel_pool = x_mel.clone()
    
    # do the fake inference round, use inference rate to control each time 
    # how many tokens to generate
    output_tensor = None
    x_mel_pre = None
    x_acc_pre = None
    x_mel_next = None
    x_acc_next = None
    for i in range(n_rounds):
        print(f"Round {i + 1}/{n_rounds}...")
        # pdb.set_trace()
        # # if prompt_length == 0, we use the whole sequence
        # if prompt_length == 0:
        #     B, S, L = x_mel.shape
        #     valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)
        x_mel_pre = x_mel_pool[:, :prompt_length + i*gen_interval_ticks]  # [1, prompt_length + i*gengen_interval_ticks, polyphony*2]
        batch_size, seq_len, subseq_len = x_mel_pre.shape  # [1, prompt_length + i*gen_interval_ticks, polyphony*2]

        # it is for initialization, we only use the first prompt_length tokens
        if i == 0:
            # slice the x_mel to the correct length
            x_acc_pre = x_acc_pool[:, :prompt_length]
            stacked = torch.stack([x_acc_pre, x_mel_pre], dim=2)
            x = stacked.view(batch_size, seq_len * 2, subseq_len) # [1, prompt_length*2, polyphony*2]

            output = model.global_sampling(x, temperature=1.0, max_seq_len=gen_seq_len) # a list of tensors, each tensor is [num_samples, polyphony*2]
            output_0 = [output[j][0 : 0 + 1, :] for j in range(len(output))] # a list of tensors, each tensor is [1, polyphony*2]
            output_tensor = torch.cat(output_0, dim=0).unsqueeze(0) # [1, prompt_length*2+generate_len, polyphony*2]
            x_acc_pre = output_tensor[:, ::2, :]  # 取偶数位
            x_mel_pre = output_tensor[:, 1::2, :] # 取奇数位

            # 第一次的 lantency 换成空
            for j in range(latency):
                x_acc_pre[:, prompt_length + j, 0] = EOS_TOKEN
                x_acc_pre[:, prompt_length + j, 1:] = PAD_TOKEN
 
            x_acc_next = x_acc_pre
            x_acc_pre = x_acc_next[:, :prompt_length + (i+1)*gen_interval_ticks]
            # x_mel_i = x_mel[:, prompt_length]
            # output_tensor[0, -1] = x_mel_i
            continue

        stacked = torch.stack([x_acc_pre, x_mel_pre], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len) # [1, prompt_length*2, polyphony*2]

        output = model.global_sampling(x, temperature=1.0, max_seq_len=gen_seq_len) # a list of tensors, each tensor is [num_samples, polyphony*2]
        output_0 = [output[j][0 : 0 + 1, :] for j in range(len(output))] # a list of tensors, each tensor is [1, polyphony*2]
        output_tensor = torch.cat(output_0, dim=0).unsqueeze(0) # [1, prompt_length*2+generate_len, polyphony*2]
        x_acc_pre = output_tensor[:, ::2, :]  # 取偶数位
        x_mel_pre = output_tensor[:, 1::2, :] # 取奇数位

        tem_start = prompt_length + i*gen_interval_ticks + latency
        tem_end = prompt_length + i*gen_interval_ticks + latency + gen_interval_ticks

        # 假设 x_acc_next.shape = [1, cur_len, polyphony*2]
        needed_len = tem_end
        cur_len = x_acc_next.shape[1]
        if cur_len < needed_len:
            pad_len = needed_len - cur_len
            # 用 PAD_TOKEN 补齐
            pad_tensor = torch.full((x_acc_next.shape[0], pad_len, x_acc_next.shape[2]), PAD_TOKEN, dtype=x_acc_next.dtype, device=x_acc_next.device)
            x_acc_next = torch.cat([x_acc_next, pad_tensor], dim=1)

        x_acc_next[:, tem_start : tem_end, :] = x_acc_pre[:, tem_start : tem_end, :]
        x_acc_pre = x_acc_next[:, :prompt_length + (i+1)*gen_interval_ticks]

        # the input of the decode_output() is a list of tensors, so we need to break the loop earlier to 
        # avoid change the shape of the output of the global_sampling()
        if i == n_rounds - 1:
            output_tensor = output_0
            break
        # output_tensor = torch.cat(output_0, dim=0).unsqueeze(0)
        # x_mel_i = x_mel[:, prompt_length + i]
        # output_tensor[0, -1] = x_mel_i
    
    # decode the output
    decode_output(output_tensor, f"temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_v1.4.mid", tempo=90.0)


if __name__ == "__main__":
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt")
    parser.add_argument("--init_midi", type=str, default="input/mel/001.mid")
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gen_interval_ticks", type=float, default=1) # every {gen_interval_ticks} logical ticks, we send a request
    parser.add_argument("--gen_seq_len", type=float, default=2) # in each round, we generate {gen_seq_len} logical ticks
    parser.add_argument("--latency", type=str, default=0) # means each round, we throw the first {latency} logical ticks
    args = parser.parse_args()

    fake_offline_sampling(
        model_path=args.model_path,
        init_midi_path=args.init_midi,
        n_rounds=args.n_rounds,
        n_samples=args.n_samples,
        prompt_len=args.prompt_len,
        temperature=args.temperature,
        gen_interval_ticks=args.gen_interval_ticks,
        gen_seq_len=args.gen_seq_len,
        latency=args.latency
    )
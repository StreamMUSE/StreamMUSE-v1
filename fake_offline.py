import os
import torch
from m2a_transformer_inference import continuation, decompress, decode_output
from m2a_transformer import RoFormerSymbolicTransformer, EOS_TOKEN, PAD_TOKEN
from preprocess.preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES
import pretty_midi
import argparse
from typing import Literal

def fake_offline_sampling(
    model_path,
    init_midi_path,
    n_rounds=5,
    n_samples=1,
    prompt_len=75,
    temperature=1.0,
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
    x_acc_i = x_acc.clone()
    x_acc_i = x_acc_i[:, :prompt_length]
    # do the fake inference round, use inference rate to control each time 
    # how many tokens to generate
    output_tensor = None
    for i in range(n_rounds):
        print(f"Round {i + 1}/{n_rounds}...")
        # # if prompt_length == 0, we use the whole sequence
        # if prompt_length == 0:
        #     B, S, L = x_mel.shape
        #     valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)
        
        if i == 0:
            # slice the x_mel to the correct length
            x_mel_i = x_mel[:, :prompt_length]
            
            batch_size, seq_len, subseq_len = x_mel_i.shape  # 10*384*8
            stacked = torch.stack([x_acc_i, x_mel_i], dim=2)
            x = stacked.view(batch_size, seq_len * 2, subseq_len)

            output = model.global_sampling(x, temperature=1.0, max_seq_len=2)
            output_0 = [output[j][0 : 0 + 1, :] for j in range(len(output))]
            output_tensor = torch.cat(output_0, dim=0).unsqueeze(0)
            x_mel_i = x_mel[:, prompt_len]
            output_tensor[0, -1] = x_mel_i
            continue
        output = model.global_sampling(output_tensor, temperature=1.0, max_seq_len=2)
        output_0 = [output[j][0 : 0 + 1, :] for j in range(len(output))]
        if i == n_rounds - 1:
            output_tensor = output_0
            break
        output_tensor = torch.cat(output_0, dim=0).unsqueeze(0)
        x_mel_i = x_mel[:, prompt_len + i]
        output_tensor[0, -1] = x_mel_i
    
    # decode the output
    decode_output(output_tensor, f"temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_test1.mid", tempo=90.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt")
    parser.add_argument("--init_midi", type=str, default="input/mel/001.mid")
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    fake_offline_sampling(
        model_path=args.model_path,
        init_midi_path=args.init_midi,
        n_rounds=args.n_rounds,
        n_samples=args.n_samples,
        prompt_len=args.prompt_len,
        temperature=args.temperature
    )
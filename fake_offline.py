import os
import torch
from m2a_transformer_inference import continuation, decompress
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


    x_mel, x_acc = decompress(model, byte_arr_mel[0], byte_arr_acc[0])

    if prompt_length == 0:
        B, S, L = x_mel.shape
        valid = (x_mel != EOS_TOKEN) & (x_mel != PAD_TOKEN)  # [B, S, L]，True 表示该 token 有效
        frame_has = valid.any(dim=2)  # [B, S]，True 表示这一帧有非 (EOS 或 PAD) 的 token
        idx = torch.arange(S, device=x_mel.device).unsqueeze(0).expand(B, S)  # [B, S]
        idx_masked = torch.where(frame_has, idx, torch.full_like(idx, S))  # [B, S]
        first_timestep = idx_masked.min(dim=1).values
    else:
        first_timestep = 1
    if not gt_mel:
        decode_output(
            [x_mel[:, i, :] for i in range(x_mel.shape[1])],
            f"temp/{model.save_name}/{os.path.basename(midi_path)}_originalmelody.mid",
            single=True,
            tempo=90.0,
        )
    x_mel_gt = x_mel.clone()
    x_mel_gt = x_mel_gt[:, first_timestep - 1 + prompt_length :]
    x_mel = x_mel[:, :prompt_length]
    x_acc = x_acc[:, :prompt_length]
    batch_size, seq_len, subseq_len = x_mel.shape  # 10*384*8
    stacked = torch.stack([x_acc, x_mel], dim=2)
    x = stacked.view(batch_size, seq_len * 2, subseq_len)

    if prompt_length != 0:
        decode_output(
            [x[:, i, :] for i in range(x.shape[1])], f"temp/{model.save_name}/{os.path.basename(midi_path)}_promptlen{prompt_length}.mid", tempo=90.0
        )

    with torch.no_grad():
        x = x.repeat(n_samples, 1, 1)
        x_mel_gt = x_mel_gt.repeat(n_samples, 1, 1)
        import time

        start_time = time.time()
        if prompt_length == 0:
            output = model.global_sampling_from_scratch(x_mel_gt, temperature=temperature, max_seq_len=generation_length)
        else:
            output = model.global_sampling(x, x_mel_gt=x_mel_gt if gt_mel else None, temperature=temperature, max_seq_len=generation_length)
        end_time = time.time()
        print(f"Generation time: {end_time - start_time:.2f} seconds")

    for i in range(n_samples):
        output_i = [output[j][i : i + 1, :] for j in range(len(output))]
        decode_output(output_i, f"temp/{model.save_name}/prompt{prompt_length}/{os.path.basename(midi_path)}_temp{temperature}_{i}.mid", tempo=90.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--init_midi", type=str, required=True)
    parser.add_argument("--n_rounds", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=75)
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
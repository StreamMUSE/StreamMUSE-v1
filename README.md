# StreamMUSE 🎹⏳
This final goal of this task is to generate accompniament given melody by user input in a real-time streaming fashion. We use 'mel' as short for melody and 'acc' as short for accompaniment. 


## Environment 

Install required packages before you start: 

    pip install -r requirements.txt

We adapted the transformers package for it to take care of the special positional encoding:
    
    cd transformers
    pip install -e .

## Data Preparation

Download POP909 dataset:

    git clone https://github.com/music-x-lab/POP909-Dataset.git

![TODO](https://img.shields.io/badge/TODO-important-red): Think of a way to separate the melody track and accompaniment track in the MIDI files!

Make sure your MIDI dataset follow the below data structure and make sure the same song shares same name in both folders:

/name_of_dataset
    /mel
        001.mid
        002.mid
        ...
    /acc
        001.mid
        002.mid
        ...

To preprocess the midi files into tensor (input to the model), run
    
    python preprocess_large_midi_dataset.py --folders /path/to/folder1 /path/to/folder2 --name name_of_dataset --polyphony num_of_notes

You can process multiple folders together and combine them to one tensor. 

The `--polyphony` let you define how many notes are allowed to play simutaneously on one timestep. 

The output tensor will be stored under `/data`, find it there.

## Training 

Train model with:

    python m2a_transformer.py --batch_size 10 --model_size small --path_to_dataset xxx_acc.pt

Make sure the path to dataset points to the tensor of accompaniment (the model will automatically load the corresponding melody tensor at `xxx_mel.pt`)

The default log is Tensorboard, but if you are familiar with wandb, you can turn on wandb logging by passing `--wandb` at the end of the command above. 

If you want to save the model with customized name, pass `--model_name`, but make sure you include the model size in the model name.


## Inference

Download checkpoint from [here](https://drive.google.com/file/d/1mX4I8EsyKN5xjYqfDHhBS6ukGDN7kosY/view?usp=sharing) and place it under `/ckpt`

For inference, place all the melody and accompaniment midi under the `/input` folder:

/input
    /mel
        001.mid
        ...
    /acc
        001.mid
        ...

Run inferencen with:

    python m2a_transformer_inference.py --model_path /home/coder/laopo/StreamMUSE/ckpt/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt --prompt_len 75 --n_samples 2 --temperature 1.0

The result will be saved in `\temp.`
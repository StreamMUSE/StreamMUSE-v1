# StreamMUSE


Install required packages before you start: 

    pip install -r requirements.txt

We adapted the transformers package for it to take care of the special positional encoding:
    
    cd transformers
    pip install -e .

To preprocess the midi files into tensor (input to the model), run
    
    python preprocess_large_midi_dataset.py --folders /path/to/folder1 /path/to/folder2 --name mydataset --polyphony num_of_notes

The polyphony argument let you define how many notes are allowed to play simutaneously on one timestep. 

The output tensor will be stored under /data, find it there.

Align the generated tensor:
    
    python align_tensors.py --tensors /path/to/tensor1 /path/to/tensor2



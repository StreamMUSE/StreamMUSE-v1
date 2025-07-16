"""
Template Benchmark Script
"""

class template_benchmark_script:
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self,
            input_midi,
            output_midi,
            prompt_len=200,
            n_samples=1,
            generate_length=200,
            temperature=1.0
        ):
        print(f'Read input MIDI from {input_midi}')
        # Create a fake file 
        save_path = output_midi
        with open(save_path, 'w') as f:
            f.write("This is a template benchmark script. Replace with actual implementation.")
            f.write(f"{save_path}")
        print(f'Save output MIDI to {save_path}')



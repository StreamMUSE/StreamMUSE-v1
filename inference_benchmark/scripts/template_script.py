"""
Template Benchmark Script
"""
from fake_offline import fake_offline_sampling

class template_benchmark_script:
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self,
            input_midi,
            output_midi,
            prompt_len=200,
            n_samples=1,
            n_rounds=5,
            generate_length=2,
            temperature=1.0
        ):

        print(f'Read input MIDI from {input_midi}')
        fake_offline_sampling(
            model_path="path/to/model.pth",
            init_midi_path=input_midi,
            n_rounds=n_rounds,
            n_samples=n_samples,
            prompt_len=prompt_len,
            temperature=temperature,
            generation_length=generate_length
        )

        # Create a fake file 
        save_path = output_midi
        with open(save_path, 'w') as f:
            f.write("This is a template benchmark script. Replace with actual implementation.")
            f.write(f"{save_path}")

        """
        STOP ADDING YOUR SCRIPT HERE
        """
        
        print(f'Save output MIDI to {save_path}')

if __name__ == "__main__":
    # Example usage
    script = template_benchmark_script()
    script.run(
        input_midi="path/to/input.mid",
        output_midi="path/to/output.mid",
        prompt_len=200,
        n_samples=1,
        generate_length=200,
        temperature=1.0
    )

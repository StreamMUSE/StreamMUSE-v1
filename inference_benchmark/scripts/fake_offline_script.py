"""
Template Benchmark Script
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from fake_offline import fake_offline_sampling

class fake_offline_script:
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self,
            input_midi,
            output_midi,
            prompt_len=200,
            n_rounds=200,
            temperature=1.0,
            gen_interval_ticks=1,
            output_len=200,
            latency=0,
            n_samples=1
        ):
        model_path = "../results/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt"
        gen_seq_len = (gen_interval_ticks + latency) * 2
        n_rounds = int(output_len // gen_interval_ticks)

        print(f'Read input MIDI from {input_midi}')
        fake_offline_sampling(
            model_path=model_path,
            init_midi_path=input_midi,
            n_rounds=n_rounds,
            id_num=0,
            prompt_len=prompt_len,
            temperature=temperature,
            gen_interval_ticks=gen_interval_ticks,
            gen_seq_len=gen_seq_len,
            latency=latency,
            save_path=output_midi
        )

        print(f'Save output MIDI to {output_midi}')

if __name__ == "__main__":
    # Example usage
    script = fake_offline_script()
    script.run(
        input_midi="inputs/pop909_dataset/mel/001.mid",
        output_midi="outputs/testtttt.mid",
        prompt_len=200,
        gen_seq_len=2, # 2 frames means 1 logical tick
        temperature=1.0,
        n_samples=1
    )

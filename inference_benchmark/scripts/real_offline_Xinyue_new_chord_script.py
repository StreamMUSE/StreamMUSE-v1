import os
from .m2a_transformer_inference_w_chord_for_benchmark import continuation
from m2a_transformer_w_chord import RoFormerSymbolicTransformer

class real_offline_Xinyue_new_chord_script:
    def __init__(self):
        self.name = self.__class__.__name__

    def run(self,
            input_midi,
            output_midi,
            prompt_len=75,
            n_samples=1,
            output_len=192,
            temperature=1.0
        ):
        model_path="../results/ModelXinyueNewChord/m2a_transformer_v0.4_chord_small_batch_20_schedule.epoch=00.val_loss=0.72838.ckpt"
        if "small" in model_path:
            model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=False)
        else:
            model = RoFormerSymbolicTransformer.load_from_checkpoint(model_path, large=True)
        model.save_name = os.path.basename(model_path)
        model.cuda()
        model.eval()

        print(f'Read input MIDI from {input_midi}')
        continuation(
            model=model,
            midi_path=input_midi,
            prompt_length=prompt_len,
            generation_length=output_len*2,
            temperature=temperature,
            n_samples=n_samples,
            output_path=output_midi
        )
        print(f'Save output MIDI to {output_midi}')
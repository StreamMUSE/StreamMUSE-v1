# This is the fake real time

import pretty_midi
from .inference_engines.transformer_engine import TransformerInferenceEngine
import os
import json
from preprocess.xf_midi import XFMidi

def midi_to_note(midi_path, min_pitch=0, max_pitch=127, beat_div=4, program=None):
    """
    用 XFMidi 读取 midi 文件，返回 notes 列表，每个元素是 {'pitch', 'tick', 'duration'} 字典。
    """
    midi = XFMidi(midi_path, constant_tempo=60.0 / beat_div)
    max_tick = int(midi.get_end_time())
    notes = []
    for inst in midi.instruments:
        # 如果只想要某种 program，可以加判断
        if program is not None and inst.program != program:
            continue
        for note in inst.notes:
            if min_pitch <= note.pitch <= max_pitch:
                # tick = round(time * resolution)
                start_tick = int(round(note.start))
                end_tick = int(round(note.end))
                if start_tick >= 0 and end_tick < max_tick:
                    notes.append({
                        'pitch': note.pitch,
                        'tick': start_tick,
                        'duration': end_tick - start_tick
                    })
    # 按tick排序
    notes.sort(key=lambda x: x['tick'])
    
    return notes, midi.resolution

def note_list_to_pretty_midi(notes, resolution, program=0, name="track"):
    instrument = pretty_midi.Instrument(program=program, name=name, is_drum=(program==127))
    for note in notes:
        start = note['tick'] 
        end = (note['tick'] + note['duration']) 
        midi_note = pretty_midi.Note(
            velocity=100,
            pitch=note['pitch'],
            start=start,
            end=end
        )
        instrument.notes.append(midi_note)
    return instrument

if __name__ == "__main__":
    # 1. 初始化 engine
    checkpoint_path = os.getenv('CHECKPOINT_PATH')
    if not checkpoint_path:
        print('Fatal Error: CHECKPOINT_PATH environment variable is not set')
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    # Get model parameters from environment variables with defaults
    try:
        model_max_seq_len_frames = int(os.getenv('MODEL_MAX_SEQ_LEN_FRAMES', 96))
        generation_length_frames = int(os.getenv('GENERATION_LENGTH_FRAMES', 20))
    except ValueError:
        print("Fatal Error: Invalid integer value for model parameters in environment variables.")
        exit()

    try:
        print(f"Loading model from {checkpoint_path}...")
        print(f"Using Model Max Sequence Length (Frames): {model_max_seq_len_frames}")
        print(f"Using Generation Length (Frames): {generation_length_frames}")
        inference_engine = TransformerInferenceEngine(
            checkpoint_path=checkpoint_path,
            model_max_seq_len_frames=model_max_seq_len_frames,
            generation_length_frames=generation_length_frames
        )
    except FileNotFoundError as e:
            print(f"Fatal Error: {e}")
            exit()

    # 2. 读取 melody midi，转为 note list
    melody_notes, resolution = midi_to_note("input/mel/001.mid", max_tick=None)
    melody_notes = sorted(melody_notes, key=lambda n: n['tick'])

    # 3. 获取所有 tick
    all_ticks = sorted(set(n['tick'] for n in melody_notes))
    max_steps = 100  # 最多生成多少步（tick）

    # 4. 逐tick送入 melody，收集伴奏
    melody_history = []
    acc_history = []

    for i, tick in enumerate(all_ticks):
        if i >= max_steps:
            break
        # 当前 tick 的所有 note
        current_notes = [n for n in melody_notes if n['tick'] == tick]
        melody_history.extend(current_notes)

        # 关键：传入完整的 melody_history
        acc_notes, *_ = inference_engine.generate_accompaniment(melody_history, generation_start_tick=tick + 1)
        acc_history.extend(acc_notes)

        print(f"Step {i}, tick={tick}, melody={current_notes}, generated acc={acc_notes}")
        # time.sleep(0.05)  # 如需模拟实时可加延迟

    # 5. 输出为两个track的midi文件
    midi_out = pretty_midi.PrettyMIDI(resolution=resolution)
    melody_instr = note_list_to_pretty_midi(melody_notes, resolution, program=0, name="melody")
    acc_instr = note_list_to_pretty_midi(acc_history, resolution, program=1, name="accompaniment")
    midi_out.instruments.append(melody_instr)
    midi_out.instruments.append(acc_instr)
    midi_out.write("fake_client_output.mid")
    print("已保存为 fake_client_output.mid")
import pretty_midi
from app.inference_engines.transformer_engine import TransformerInferenceEngine
import os
import json

def midi_to_note_list(midi_path, tick_resolution=1, max_tick=40):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi.instruments:
        program = instrument.program if not instrument.is_drum else 127
        for note in instrument.notes:
            tick = int(note.start * midi.resolution / tick_resolution)
            duration = int((note.end - note.start) * midi.resolution / tick_resolution)
            if max_tick is not None and tick >= max_tick:
                continue
            notes.append({
                'pitch': int(note.pitch),
                'tick': int(tick),
                'duration': int(duration),
                'program': int(program)
            })
    return notes

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
melody_notes = midi_to_note_list("your_melody.mid", max_tick=None)
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

# 5. 可选：保存最终伴奏到文件
with open("fake_client_acc_history.json", "w") as f:
    json.dump(acc_history, f, ensure_ascii=False, indent=2)
# StreamMUSE Running Instructions

## 1. Start Inference Server

You can run the server with different engines (Baseline/Stanley or Lekai).

### Option A: Run Baseline (Stanley) Server

```bash
# Set environment variables
# for example ckpt_path 的转义符 \ 不可去掉
export CUDA_VISIBLE_DEVICES=1 # only needed when you try to run server on multiple GPU platform and you want to run the server on a specific GPU.
export CHECKPOINT_PATH=~/ugrip/shared_models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch\=00.val_loss\=0.90296.ckpt
export MODEL_MAX_SEQ_LEN_FRAMES=384
export ENGINE_TYPE=stanley  # Default

# Run Server
PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988
```

### Option B: Run Lekai Server (LLaMA-based)

#### Mode 1: Sliding Window (Default, Stable)
Recomputes context every time. Slower but safer for long sessions.
```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_PATH=rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors
export MODEL_SIZE=llama
export ENGINE_TYPE=lekai
export INFERENCE_MODE=sliding_window

PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988
```

#### Mode 2: Stateful (KV Cache, Faster)
Uses KV cache to speed up generation. History is cleared automatically when client restarts (switching songs).
*Note: May hit context length limit (3500 tokens) if the song is extremely long.*
```bash
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_PATH=rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors
export MODEL_SIZE=llama
export ENGINE_TYPE=lekai
export INFERENCE_MODE=stateful

PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988
```

## 2. Run Client / Experiment Runner

Once the server is running, you can run the experiment runner in a separate terminal.

```bash
# Run Experiment Runner
uv run real_time_experiment_runner.py \
--dataset-dir input/mel \
--injection-length 128 \
--generation-length 576 \
--out-root experiments1/realtime/lekai_test/interval_4_gen_frame_5 \
--server-url http://localhost:8988/generate_accompaniment \
--generation-interval-ticks 4 \
--generation-length-per-request 6
```

### Notes
- **Critical for Lekai Model**: You **MUST** set `--generation-interval-ticks 4` (matching the beat length). The Lekai engine generates music beat-by-beat. Setting this to 1 will cause duplicate generation and corrupt the context.
- Ensure `real_time_experiment_runner.py` exists in your path.
- Adjust `--out-root` to avoid overwriting previous results.
- For Lekai model, `generation-length-per-request` is ignored by the server (it always generates 1 beat), but kept for client compatibility.
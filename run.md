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

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=1
# Update this path to your actual Lekai model checkpoint
export CHECKPOINT_PATH=rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors
export MODEL_SIZE=llama
export ENGINE_TYPE=lekai
export INFERENCE_MODE=sliding_window  # Options: sliding_window (default), stateful

# Run Server
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
--out-root experiments1/realtime/lekai_test/interval_1_gen_frame_5 \
--server-url http://localhost:8988/generate_accompaniment \
--generation-interval-ticks 1 \
--generation-length-per-request 5
```

### Notes
- Ensure `real_time_experiment_runner.py` exists in your path.
- Adjust `--out-root` to avoid overwriting previous results.
- For Lekai model, `generation-length-per-request` might need tuning based on performance.
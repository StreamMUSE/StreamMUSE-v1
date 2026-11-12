# ZipZapZop Memory Sweep 实现报告（2026-07-09）

## 结论

本次已经完成 memory sweep 计划里可在当前仓库内落地的实现部分：生产代码调用链、单模型 sweep 脚本、7 模型 full-matrix wrapper、矩阵汇总脚本和对应单测。HF 网页授权、数百 GB 模型下载、单模型 GPU pilot、正式 7 模型 sweep 没有在本次执行，因为它们需要外部账号授权、模型文件和正在运行的 vLLM/GPU 环境。

## 已实现内容

### 1. Local chat client

文件：`src/streammuse/infrastructure/inference/local_chat_client.py`

- `LocalChatModelClientConfig` 新增 `top_p` 和 `extra_payload`。
- `LocalChatModelClient` 改为持有 `requests.Session()`，每次 `generate()` 复用同一个 session。
- `generate()` 会在 `top_p is not None` 时把 `top_p` 放进 OpenAI-compatible payload。
- `extra_payload` 会浅合并进 payload，但如果试图覆盖 `model/messages/temperature/max_tokens/top_p` 等已有 key，会直接 `ValueError`。
- 新增 `close()` 和 context manager，调用方可以显式释放 session。

### 2. task CLI / run_task 调用链

文件：`src/streammuse/presentation/task/cli.py`

- `streammuse-task run` 新增 `--top-p`。
- `run_task()` 新增 `top_p`、`extra_payload`、`oracle_history` 参数。
- `_build_client()` 透传 `top_p` / `extra_payload` 到 `LocalChatModelClientConfig`。
- `run_task()` / `play_task()` 在结束时关闭 client session。
- `create_task()` 支持把 `oracle_history` 传给 `ZipZapZopTask`。

### 3. ZipZapZop oracle history

文件：`src/streammuse/domain/tasks/zip_zap_zop.py`

- `ZipZapZopTask.__init__()` 新增 `oracle_history=False`。
- 默认模式仍然把模型实际 response 写进 history。
- oracle 模式下，history 的 `content` 写 expected answer，用于分离“上下文长度影响”和“模型错误自污染影响”。

### 4. batch runtime raw response trace

文件：`src/streammuse/application/tasks/runtime.py`

- batch artifact 现在额外保存 `model_response_raw`。
- `response_trace.jsonl` 的轻量 schema 没改，仍然是 `turn_id/prompt/response`。
- 这个 raw 字段主要给 gpt-oss 这类 reasoning/final 分离模型做后处理诊断。

### 5. 单模型 sweep 脚本

文件：`scripts/run_zzz_memory_sweep.py`

功能：

- CLI 参数覆盖 plan 中的单模型 sweep：`--model`、`--model-url`、`--turns`、`--history-limits`、`--temperatures`、`--top-p`、`--repeats-nonzero-temp`、`--pilot`、`--apc-label`、`--spec-label` 等。
- preflight：请求 `/v1/models`，确认模型服务可用且模型名匹配。
- reset prefix cache：从 `model-url` 推出 server root，探测/调用 `POST /reset_prefix_cache`。
- warmup：正式统计前发 2 个丢弃请求。
- schedule：正式模式按 repeat 平衡 history 档位顺序，pilot 模式固定顺序。
- per-model special config：
  - `Qwen/Qwen3-8B` 自动发送 `chat_template_kwargs: {enable_thinking: false}`。
  - `openai/gpt-oss-120b` 默认 `max_tokens=512`，并标记 `answer_extract_status`。
- 非 pilot 默认追加 `all + greedy + oracle_history=True` 对照。
- 输出：
  - `per_turn.csv`
  - `sweep_summary.json`
  - `sweep_summary.md`
  - `run_config.json`
  - 原始 run 目录在 `runs/` 下。

### 6. 矩阵汇总与 full-matrix wrapper

文件：

- `scripts/merge_zzz_memory_summaries.py`
- `scripts/run_zzz_full_matrix.sh`

`merge_zzz_memory_summaries.py` 会递归收集所有 `sweep_summary.json`，输出：

- `matrix_summary.json`
- `matrix_summary.csv`
- `matrix_summary.md`

`run_zzz_full_matrix.sh` 实现：

- 7 个模型按计划顺序启动/停止 vLLM。
- #6/#7 自动设置 `--tensor-parallel-size 2`。
- 每个模型跑 APC-on 主 sweep。
- 重启 vLLM 加 `--no-enable-prefix-caching` 跑 APC-off 的 `recent-32/all + greedy` 对照。
- gpt-oss 额外跑 ngram speculative decoding 组。
- gpt-oss ngram greedy 输出会和 baseline greedy 逐项比较，不一致直接报错。
- 每个模型完成后写 `done` marker，重跑 wrapper 会跳过已完成模型。

## 运行方式

### 单模型 pilot

```bash
uv run python scripts/run_zzz_memory_sweep.py   --model Qwen/Qwen3-8B   --model-url http://127.0.0.1:8000/v1   --pilot   --turns 6   --history-limits 0,2,all   --temperatures 0,0.7   --repeats-nonzero-temp 2
```

### 单模型正式 sweep

```bash
uv run python scripts/run_zzz_memory_sweep.py   --model Qwen/Qwen3-8B   --model-url http://127.0.0.1:8000/v1
```

### 7 模型 full matrix

```bash
nohup bash scripts/run_zzz_full_matrix.sh > matrix.log 2>&1 &
```

可选环境变量：

```bash
ROOT_DIR=task_runs/zzz_memory_full_matrix_custom
STREAMMUSE_MATRIX_GPUS=0,1,2
VLLM=$HOME/mbzuai-projects/llm-serving/.venv/bin/vllm
MODEL_URL=http://127.0.0.1:8000/v1
```

## 验证结果

已执行：

```bash
python3 -m py_compile scripts/run_zzz_memory_sweep.py scripts/merge_zzz_memory_summaries.py
bash -n scripts/run_zzz_full_matrix.sh
uv run pytest tests/unit/infrastructure/inference/test_local_chat_model_client.py tests/unit/presentation/task/test_task_cli.py tests/unit/presentation/task/test_zzz_memory_sweep_script.py tests/unit/domain/tasks/test_zip_zap_zop_task.py tests/unit/application/tasks/test_task_runtime.py -q
uv run pytest tests/ -q
uv run python scripts/run_zzz_memory_sweep.py --help
uv run python scripts/merge_zzz_memory_summaries.py --help
```

结果：

- targeted tests：`26 passed`
- full tests：`229 passed, 1 skipped, 1 warning`
- 两个 Python 脚本语法检查通过
- full-matrix bash wrapper 语法检查通过
- 两个新增 CLI 脚本 help 能正常启动

## 未执行项

以下仍需要在有外部条件时执行：

- HF gated model license accept：`meta-llama/Llama-3.3-70B-Instruct`、`google/gemma-3-27b-it`。
- 7 个模型下载与本地磁盘校验。
- 单模型 GPU pilot。
- 7 模型正式 full matrix sweep。
- 正式实验结果报告 `developing-logs/reports/2026-07-03-zzz-memory-sweep-report.md`。

## 注意事项

- 如果当前 vLLM 版本没有 `/reset_prefix_cache`，单模型脚本会记录 unavailable；full-matrix 的干净 APC 行为仍可通过重启 server 获得。
- `--max-tokens` 如果不显式传，普通模型默认 8，gpt-oss 默认 512。
- `Qwen/Qwen3.6-*` 是否需要禁思考还需要实际 smoke 后确认；目前只有 `Qwen/Qwen3-8B` 自动加了 `enable_thinking=false`。
- `task_runs/` 是运行输出目录，本次没有把已有输出纳入实现改动。

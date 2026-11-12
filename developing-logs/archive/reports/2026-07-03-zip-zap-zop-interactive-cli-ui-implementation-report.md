# ZipZapZop Interactive CLI UI Implementation Report

## 实现范围

本次实现新增了 `streammuse-task play`，用于让真人和本地 OpenAI-compatible LLM 在 terminal 里一人一轮玩 Zip-Zap-Zop。原来的 `streammuse-task run` 仍然保留 batch benchmark / realtime loop 语义，不被 interactive UI 改写。

新增核心文件：

- `src/streammuse/application/tasks/interactive_runtime.py`
- `tests/unit/application/tasks/test_interactive_runtime.py`

主要修改文件：

- `src/streammuse/domain/tasks/models.py`
- `src/streammuse/domain/tasks/zip_zap_zop.py`
- `src/streammuse/presentation/task/cli.py`
- `src/streammuse/application/tasks/__init__.py`
- `src/streammuse/domain/tasks/__init__.py`

## `run` 和 `play` 的区别

`streammuse-task run` 是非交互模式，适合 benchmark 和自动 trace：

- `offline_benchmark`：尽快连续跑完若干 turn。
- `realtime_loop`：按固定 tick/deadline 跑若干 turn。

`streammuse-task play` 是真人交互模式：

- human 和 LLM 交替回答。
- 支持 `--human-first` / `--llm-first`。
- 支持 terminal command：`:help`、`:hint`、`:expected`、`:summary`、`:quit`。
- 每轮都会校验答案、显示 OK/MISS，并写 interactive trace。

## 运行方式

需要先启动一个 OpenAI-compatible `/v1/chat/completions` 服务，例如 vLLM：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct   --host 0.0.0.0   --port 8000
```

然后运行：

```bash
uv run streammuse-task play   --task zip_zap_zop   --model-url http://127.0.0.1:8000/v1   --model Qwen/Qwen2.5-7B-Instruct   --max-turns 30   --max-tokens 8   --temperature 0   --deadline-ms 3000   --output-dir task_runs
```

LLM 先手：

```bash
uv run streammuse-task play   --task zip_zap_zop   --llm-first   --model-url http://127.0.0.1:8000/v1   --model Qwen/Qwen2.5-7B-Instruct   --max-turns 30   --output-dir task_runs
```

## Interactive trace 结构

每次 `play` 会创建：

```text
task_runs/{task_name}_interactive_{timestamp}_{id}/
  manifest.json
  response_trace.jsonl
  run_summary.json
  artifacts/turn/*.json
```

`response_trace.jsonl` 每行是一个 turn，字段包括：

- `turn_id`
- `actor`: `human` 或 `llm`
- `number`
- `prompt`: human prompt string 或 LLM messages list
- `response`
- `expected`
- `is_valid`
- `latency_ms`
- `deadline_missed`
- `metadata`: failure reason、token usage 等

注意：batch `TaskRuntime.run()` 的 `response_trace.jsonl` 仍然保持轻量 schema：`turn_id`、`prompt`、`response`。interactive schema 更丰富，两者用于不同运行模式。

## Soft deadline 行为

`--deadline-ms` 是 soft deadline：

- human input 不会被强制打断。
- LLM request 也不会被 runtime 强杀，真正的 HTTP 超时由 `--timeout-s` 传给 `LocalChatModelClientConfig.timeout_s` 控制。
- runtime 会记录每轮 latency，并把超时标成 `deadline_missed=true`。

## 多游戏上下文隔离

Interactive runtime 保存完整 transcript，但不解释游戏规则，也不决定给模型看多少上下文。

每个 task 自己实现：

- `build_human_prompt(state, transcript)`
- `build_llm_messages(state, transcript)`
- `build_hint(state, transcript)`
- `expected_for_state(state, transcript)`

ZipZapZop 当前使用 `history_limit` 控制最近多少轮进入 LLM prompt。未来别的游戏如果需要完整历史、摘要历史、隐藏部分信息或自定义 role，都可以在自己的 task/adapter 里实现，不需要改 runtime。

## 验证

已完成自动测试：

```bash
uv run pytest tests/unit/domain/tasks tests/unit/application/tasks tests/unit/presentation/task -q
# 18 passed

uv run pytest tests/unit tests/integration -q
# 212 passed, 1 warning
```

未执行真实本地 LLM server 的人工 20-turn 试玩，因为当前实现验证环境里没有启动用户指定的 OpenAI-compatible model server。CLI、runtime、trace、commands、deadline、valid/invalid path 均已由 fake terminal/fake model 单测覆盖。

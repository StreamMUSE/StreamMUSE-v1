# ZipZapZop Deadline Modes Implementation Report

## Summary

本次按照 `2026-07-03-zip-zap-zop-deadline-modes-plan.md` 完成了 `streammuse-task play --task zip_zap_zop` 的三种 deadline mode：

- `soft`：超时/答错只记录，游戏继续。
- `hard`：任一方超时或答错，立即结束，犯错者输。
- `challenge`：从宽松 deadline 开始，每完成一个 clean stage 就降低 deadline；任一方超时或答错立即输。

`play` 默认 `--deadline-mode menu`，启动后会先让用户选择模式；脚本和测试可以显式传 `--deadline-mode soft|hard|challenge` 跳过菜单。

## Changed Files

核心实现：

- `src/streammuse/application/tasks/interactive_runtime.py`
- `src/streammuse/presentation/task/cli.py`
- `src/streammuse/domain/tasks/models.py`
- `src/streammuse/domain/tasks/__init__.py`
- `src/streammuse/application/tasks/__init__.py`
- `src/streammuse/infrastructure/inference/local_chat_client.py`

测试：

- `tests/unit/application/tasks/test_interactive_runtime.py`
- `tests/unit/presentation/task/test_task_cli.py`
- `tests/unit/infrastructure/inference/test_local_chat_model_client.py`

## Runtime Behavior

### Soft Mode

Soft mode 保留之前的行为，但 summary 更详细：

- `deadline_missed=true` 会写入每一轮 trace。
- `invalid_responses` 会写入 summary。
- 游戏不会因为超时或答错停止。

### Hard Mode

Hard mode 下：

- human 超时：`stop_reason=deadline_loss`，`loser=human`。
- LLM 超时：`stop_reason=deadline_loss`，`loser=llm`。
- human 答错：`stop_reason=invalid_response_loss`，`loser=human`。
- LLM 答错：`stop_reason=invalid_response_loss`，`loser=llm`。

如果同一 turn 同时超时且答错，按 plan 以 deadline loss 优先，同时保留 `is_valid=false`。

### Challenge Mode

Challenge mode 下：

- 默认 deadline schedule 是 `10000,5000,3000,2000,1000` ms。
- 默认每个 stage 需要 `20` 个 clean turns。
- stage 通过后 deadline 会切换到下一档。
- 数字、state、transcript 不重置。
- schedule 到最后一档后继续使用最后一档，直到达到 `max_turns` 或有人输。

## CLI

新增参数：

```bash
--deadline-mode {menu,soft,hard,challenge}
--challenge-stage-turns 20
--challenge-deadline-ms-list 10000,5000,3000,2000,1000
```

示例：

```bash
uv run streammuse-task play   --task zip_zap_zop   --deadline-mode hard   --deadline-ms 3000   --model-url http://127.0.0.1:8000/v1   --model Qwen/Qwen2.5-7B-Instruct
```

Challenge mode：

```bash
uv run streammuse-task play   --task zip_zap_zop   --deadline-mode challenge   --challenge-stage-turns 20   --challenge-deadline-ms-list 10000,5000,3000,2000,1000   --model-url http://127.0.0.1:8000/v1   --model Qwen/Qwen2.5-7B-Instruct
```

## Trace And Summary

`response_trace.jsonl` 每一行的 `metadata` 现在会包含：

- `deadline_mode`
- `deadline_ms`
- `challenge_stage_index`
- `challenge_stage_turn_count`
- `model_error`，例如 LLM request timeout

`run_summary.json` / `manifest.json` 会记录：

- `deadline_mode`
- `final_deadline_ms`
- `winner`
- `loser`
- `stop_reason`
- `deadline_misses`
- `invalid_responses`

## Timeout Implementation

Human input：

- `TerminalIO` 新增 `prompt_with_timeout()`。
- `StdTerminalIO` 使用 `select.select([sys.stdin], [], [], timeout_s)` 实现 Linux terminal timeout。

LLM request：

- `LocalChatModel.generate(..., timeout_s=None)` 支持 per-call timeout。
- hard/challenge mode 下 runtime 会传 `current_deadline_ms / 1000.0`。
- `requests.Timeout` 会被转换成一次 LLM deadline loss turn，并写入 trace。

## Validation

已通过：

```bash
uv run pytest tests/unit/application/tasks/test_interactive_runtime.py tests/unit/presentation/task/test_task_cli.py tests/unit/infrastructure/inference/test_local_chat_model_client.py -q
# 23 passed

uv run pytest tests/unit/domain/tasks tests/unit/application/tasks tests/unit/presentation/task tests/unit/infrastructure/inference -q
# 71 passed, 1 warning

uv run pytest tests/unit tests/integration -q
# 220 passed, 1 warning
```

没有跑真实 vLLM 的人工试玩；代码路径由 fake terminal 和 fake model 覆盖，包括 menu、soft、hard、challenge、human timeout、LLM timeout、答错判负、stage 切换和 CLI 解析。

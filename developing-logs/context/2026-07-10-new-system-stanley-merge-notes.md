# New System Stanley Merge 注意事项

日期：2026-07-10

## 1. `new_system_stanley` 当前缺少 two-stage prompt-continuation 推理

当前 `new_system_stanley` 里已经合入了一些 task/debugger 相关功能，但还没有完整的 two-stage prompt-continuation inference framework。具体来说，当前 branch 没有 tracked 下列关键源码：

- `src/streammuse/infrastructure/inference/lekai_prompt_continuation/`
- `src/streammuse/application/services/prompt_continuation_realtime_service.py`
- `src/streammuse/infrastructure/inference/prompt_continuation_http_client.py`
- `scripts/run_lekai_prompt_continuation_offline.py`
- `tests/consistency/test_two_stage_prompt_continuation_consistency.py`
- `tests/consistency/two_stage_runners.py`

之前修过 scheduler 的 two-stage inference framework 在这个 branch 里：

```text
integrate-prompt-continuation-switch
```

确认到该 branch 最新相关提交是：

```text
3a264d56 Fix prompt-continuation streaming scheduling and add sweep tooling
```

因此后续如果要在 `new_system_stanley` 上使用 two-stage prompt+continuation，不能只依赖当前已经合入的 replay/debugger/task runtime。需要从 `integrate-prompt-continuation-switch` 继续 selective merge / cherry-pick two-stage 相关代码。

合入时要特别注意这些位置可能和当前 `new_system_stanley` 已有改动发生交叉：

- `pyproject.toml` / console script entrypoints
- `src/streammuse/presentation/cli/cli.py`
- `src/streammuse/presentation/cli/config_parser.py`
- `src/streammuse/presentation/web/server.py`
- `src/streammuse/application/runtime/`
- `src/streammuse/application/tasks/`
- `tests/consistency/`

合入后至少需要重新验证：

```bash
uv run pytest tests/ -q
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q
```

第二条 consistency test 需要真实模型 checkpoint 和对应环境变量；如果 checkpoint 没配置，测试会 skip 或无法完整验证。

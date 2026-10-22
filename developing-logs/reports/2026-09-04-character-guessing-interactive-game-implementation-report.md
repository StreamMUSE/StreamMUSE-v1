# Character Guessing 双向语音游戏实现报告

> 日期：2026-09-04
>
> 项目：`/Users/zhengbowen/Projects/Research Project 2025/StreamMUSE-v1`
>
> 分支：`feature/game`
>
> 实现基线 HEAD：`7ea0866020a836a9b5342665348506f71c982011`
>
> 状态：代码与自动化验证完成，待真实 Qwen/STT/TTS/Web qualification

## 1. 完成内容

新增 play-only 任务 `character_guessing`。玩家从一个公开、有限的人物列表中私下选择人物，但不把目标输入程序；LLM 从第一轮开始，以 yes/no question 逐步排除人物，也可以正式猜测。玩家对正式猜测回答 yes 后立即结束。

V1 内置 `world_historical_figures_en` scope，共 44 位英文历史人物，支持规范化名字和显式 aliases。目标人物不会出现在 `TaskState`、LLM prompt、trace、manifest、WebSocket 或进程参数中。

LLM 输出使用严格单行协议：

```text
ASK: <one yes-or-no question?>
GUESS: <one candidate name>
```

Human text/STT 只接受受控的简短 yes/no 表达。非法 LLM 输出和非法 Human 回答都由同一 actor 重试，不推进 logical round；soft mode 下同一 phase 连续三次非法会以 `retry_limit_reached` 结束。

为避免模型先用 `ASK: Is this person <name>?` 再重复正式猜测，`ASK` 不允许包含任何 catalog display name 或 alias；只要点名候选人就必须使用 `GUESS`。

## 2. Domain 与 runtime 设计

新增通用 optional protocols：

- `FlowControlledInteractiveTask`：task 自己决定 next actor、logical round 和 completion；旧游戏继续使用 parity scheduling。
- `SetupAwareInteractiveTask`：在第一回合前显示候选范围和 ready prompt。
- `PublicTaskStateProvider`：只向 Web/manifest 投影安全的公开状态。
- `InteractiveResponsePresenter`：把内部 `ASK:`/`GUESS:` wire format 转为玩家可读、可朗读文本。

一个 logical round 被严格定义为一个有效 LLM action 加一个有效 Human yes/no answer。`--max-rounds` 只会在 Human 完成回答后检查，因此不会留下 unanswered question。Runtime 另设 `6 * max_rounds` 的 attempt safety bound，允许每个 phase 最多两次非法后再成功，同时防止无界循环。

Deadline precedence 已统一：在 hard/challenge 模式下，同一 Human answer 如果既确认猜中又超时，最终结果是 `deadline_loss`；summary、Web progress 和 successful-guess metrics 都不会误报猜中。

## 3. Voice 与 TTS

Human voice 复用现有常驻 `faster-whisper tiny.en int8` 输入层。Character task 提供固定 hotwords 和严格 spoken parser：`yes/yeah/yep/correct/right` 归一为 yes，`no/nope/incorrect/wrong` 归一为 no；`maybe`、解释性长句、空转录和病态转录不会进入事实 history。

LLM 的 ASK/ GUESS 内容是动态的，因此 `speech_vocabulary()` 保持为空，不能完整 prewarm。启用音频输出时，CLI 强制 `--speech-cache-miss synthesize`。只有通过 protocol validation 的 action 才会朗读；机器音频 drain 完成并经过 speech guard 后，才会打开下一次 Human microphone。

## 4. Terminal、Web 与事件

Setup 阶段会显示 44 人名单并要求玩家只按 Enter。非空输入会被忽略，避免目标人物进入记录；`:quit` 可以在 setup 阶段退出。Setup 不属于 actor turn，也不启动 deadline。

Terminal 以 `[Round N]` 显示逻辑轮次，并展示清理后的问题或人物猜测。Web UI 新增：

- setup scope 和 responsive candidate list；
- current/max logical round；
- pending question 或 guess；
- Human/LLM 最近结果；
- final stop reason 或猜中人物与轮数。

新增 `session_setup` 和 `task_progress` 事件，snapshot schema 升为 2。动态文本继续只通过 `textContent` 写入 DOM，候选列表使用 `createElement`，不使用 `innerHTML`。

## 5. Trace、summary 与 latency analysis

每个 turn artifact 和 `response_trace.jsonl` 新增：

- `logical_round`；
- `phase_before` / `phase_after`；
- `action_kind`；
- `canonical_character_id`；
- `canonical_answer`；
- `round_completed`；
- `task_completed` / `task_stop_reason`；
- `display_text`。

`run_summary.json` 与 manifest 新增 logical round、question、guess、rejected guess、successful guess、character id 和 successful round 统计。

`scripts/analyze_interactive_voice_latency.py` 保持旧 `breakdown_turns.csv` 输出，并新增 `breakdown_rounds.csv`。Round aggregation 只组合相同 logical round 中有效的 LLM/Human turns，非法 retries 不会污染完整 round latency。

## 6. 主要文件

- `src/streammuse/domain/tasks/character_scopes.py`：catalog、validation、canonicalization。
- `src/streammuse/domain/tasks/character_guessing.py`：protocol parser、state machine、voice/TTS rendering。
- `src/streammuse/domain/tasks/models.py`：通用 flow/setup/public-state/presenter protocols。
- `src/streammuse/application/tasks/interactive_runtime.py`：task-owned scheduling、setup、retry、completion、metrics、events。
- `src/streammuse/presentation/task/cli.py`：task registration、专属参数/defaults/incompatibility validation。
- `src/streammuse/infrastructure/task_view/snapshot.py`：setup/progress reconnect snapshot。
- `src/streammuse/presentation/task_web/static/`：setup、round、pending action 和 final result UI。
- `scripts/analyze_interactive_voice_latency.py`：logical-round CSV 和统计。
- `docs/user-guide/character-guessing.md`：完整用户指南。

## 7. 自动化验证结果

通过：

- Character domain/runtime/CLI/Web/analyzer targeted suite：`99 passed`。
- 全部 task、voice、speech、task-view、Web 相关 suite：`422 passed`。
- 新增完整 hermetic session：`1 passed`，覆盖 scripted LLM、fake voice、fake TTS、真实 queue snapshot reducer、retries、trace、summary 和 cleanup。
- `node --check .../main.js`：通过。
- `git diff --check`：通过。
- `uv build`：成功生成 sdist 和 wheel；wheel 已确认包含 Character modules 与 Web static assets。

完整 `tests/` suite 当前结果为 `772 passed, 4 skipped, 15 failed`。15 个失败均位于本次未修改的 Lekai session contract 和 melody robustness campaign 测试，主要是已有 campaign status/qualification API 漂移。裸 `pytest` 还会额外收集仓库中的 vendored `transformers/` 测试，并因当前 Keras 3 缺少兼容 `tf_keras` 在 collection 阶段失败。这两组问题不由 Character Guessing 变更引起，但意味着全仓绿色 acceptance gate 尚未满足。

## 8. 尚未完成与限制

以下内容需要真实设备或外部服务，未在本次自动化环境完成：

- 真实 Qwen 27B endpoint 与 SSH forwarding qualification；
- 真实麦克风 `tiny.en int8` yes/no accuracy；
- 真实 TTS pronunciation、drain、guard 和回声测试；
- Web desktop/mobile 实机截图检查；
- 至少 10 个人物、每局最多 20 rounds 的统计实验；
- guess rate、protocol valid rate、ASR accuracy 和 end-to-end latency qualification report。

V1 还有这些明确限制：

- Human 是事实 oracle，程序不验证回答是否符合历史事实；
- 只支持英文、44 人内置 scope；
- 不支持 unknown/uncertain answer；
- 问题质量和猜中率依赖 LLM，本版没有 guided decoding、candidate elimination engine 或 retrieval；
- 动态 TTS 首次 cache miss 会产生 synthesis latency，cache 仍是 session 内存级。

## 9. 如何测试

### 9.1 文字模式

```bash
uv run --frozen streammuse-task play \
  --task character_guessing \
  --character-scope world_historical_figures_en \
  --max-rounds 20 \
  --model-url http://127.0.0.1:8010/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 64 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-character-text
```

看到名单后私下选一人，只按 Enter。之后只输入 yes/no。不要传 `--human-first`、`--start-number` 或 `--max-turns`。

### 9.2 双向语音 + TTS + Web

先确认 LLM tunnel：

```bash
curl http://127.0.0.1:8010/v1/models
```

然后运行：

```bash
uv run --frozen --extra voice --extra speech streammuse-task play \
  --task character_guessing \
  --character-scope world_historical_figures_en \
  --max-rounds 20 \
  --human-input voice \
  --microphone-device 2 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --voice-max-utterance-ms 1000 \
  --speech-output audio \
  --speech-backend system \
  --speech-cache-miss synthesize \
  --speech-guard-ms 200 \
  --web-ui \
  --web-port 8002 \
  --model-url http://127.0.0.1:8010/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 64 \
  --timeout-s 30 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-character-voice
```

打开 CLI 输出的 tokenized Web URL，等待页面完成首次 render；随后在终端 setup 阶段只按 Enter。建议戴耳机，避免 TTS 被麦克风重新收录。

### 9.3 自动化测试

```bash
uv run --frozen python -m pytest \
  tests/unit/domain/tasks/test_character_guessing_task.py \
  tests/unit/application/tasks/test_character_guessing_runtime.py \
  tests/integration/test_character_guessing_session.py -q
```

### 9.4 Latency breakdown

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  /private/tmp/streammuse-character-voice \
  --output-dir /private/tmp/streammuse-character-analysis
```

重点查看 `breakdown_turns.csv`、`breakdown_rounds.csv`、`breakdown_summary.json` 和 `breakdown_summary.md`。

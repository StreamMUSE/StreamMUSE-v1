# Animal Naming 双向语音实现报告

日期：2026-08-13

仓库：`/Users/zhengbowen/nondefault/Research 2026/StreamMUSE-v1`

分支：`feature/voice`

实现所基于的 HEAD：`86ed4652`

对应计划：`developing-logs/plans/2026-08-13-animal-naming-interactive-voice-plan.md`

## 1. 结论

Animal Naming 的软件实现阶段已经完成。现有 `AnimalNamingTask` 现在同时支持：

- 原有 `streammuse-task run --task animal_naming` batch benchmark；
- `streammuse-task play --task animal_naming` Human / LLM 交替对局；
- Human terminal input；
- Human voice input，经 `faster-whisper tiny.en` 转写；
- LLM silent output；
- LLM TTS audio output；
- shared whitelist、shared no-repeat state、open-ended referee output；
- raw STT、canonical response、normalized animal、referee、LLM、TTS 和 latency trace。

代码、focused automated tests、CLI one-turn smoke 和 analyzer compatibility smoke 均已完成。真实 microphone + remote LLM + speaker 的多轮 qualification 尚未执行，因此本报告是 **implementation report**，不是最终的 hardware qualification report。

## 2. V1 规则

V1 是“每轮说一个新动物”，不是累计复述型 memory game。

- Human 和 LLM 每轮只回答一个英文动物名称。
- 两个 actor 共用同一份 `used_animals` 和 `attempted_animals` state。
- 判定使用现有 91 项 exact whitelist。
- whitelist 外的真实动物仍判为 `UNKNOWN_ANIMAL`。
- 已经成功使用过的动物判为 `REPEATED_ANIMAL`。
- 空答案判为 `EMPTY_RESPONSE`。
- 不做 synonym、plural、fuzzy matching、semantic correction 或 LLM-based validation。
- Human cue 只显示文字；只有 LLM answer 可以使用 TTS。

## 3. 实现内容

### 3.1 Domain task

主要文件：`src/streammuse/domain/tasks/animal_naming.py`

完成内容：

1. 保留原有 91 项 `DEFAULT_ANIMALS` 和 batch API。
2. `validate_response()`、`advance_state()` 增加 optional `actor` / `transcript` 参数，同时保留旧 positional 调用方式。
3. 新增 Human prompt、LLM messages、hint、remaining animals 和 open-ended expected contract。
4. Human history 使用 `human` role，LLM history 使用 `assistant` role；batch 的旧行为保持不变。
5. Human 和 LLM 使用同一个 referee，只有 valid animal 进入 `used_animals`。
6. 所有非空 normalized attempt 进入 `attempted_animals`，让 LLM prompt 避免再次输出已失败或已使用的词。
7. 新增 speech context、spoken response parser 和 spoken-text renderer。

文本处理链路如下：

```text
raw response
  -> lowercase / trim / first line / punctuation removal
  -> optional a|an|the removal
  -> short 1-3 word speech grammar check
  -> canonical candidate
  -> exact whitelist + shared used-state referee
```

parser 只判断输入是否像一个简短动物候选，不查询 whitelist，也不判断是否重复。例如 `Dragon.` 会先解析为 `dragon`，再由 referee 判为 `UNKNOWN_ANIMAL`。这样 trace 可以区分 STT/parser 问题和游戏规则问题。

### 3.2 STT context

Animal Naming 的 `SpeechContext` 包含：

- 一个简短的英文 `initial_prompt`；
- 尚未使用动物的 deterministic hotwords；
- 已使用动物会从下一回合 hotwords 中移除。

production hotword renderer 的单元测试确认首轮完整传入 91 个词，顺序稳定，渲染结果小于 1024 个字符。当前没有任意截断 hotwords，也没有加入 fuzzy correction。

需要说明：这里只验证了 context 构造和传参。尚未使用固定真实音频比较 hotwords on/off 的识别率、latency 和 hallucination，因此该部分仍属于 Phase 7 qualification 工作。

### 3.3 TTS rendering 与 cache 策略

Animal Naming 的 LLM response 会先经过 `build_spoken_text()`：

- 单个结构合法的候选词返回 canonical animal text；
- explanation、列表和空输出返回 `None`，不会朗读；
- renderer 不修改 referee 结果，也不重新调用 LLM。

`speech_vocabulary()` 固定返回空 tuple，因此启动时不会预合成完整 91 项 whitelist。每个新的 LLM animal 在当前 session 内按需合成和缓存。

当前 audio cache 是 **session 内存 cache**，不是跨 session 的 disk cache。重新开一局后，新动物会重新按需合成；文档已避免把它描述成持久化 cache。

### 3.4 Open-ended interactive runtime

主要文件：`src/streammuse/application/tasks/interactive_runtime.py`

runtime 现在同时支持：

- Zip Zap Zop 这类有唯一 expected answer 的 task；
- Animal Naming 这类没有唯一 expected answer 的 task。

具体改动：

- `max_turns` 必须大于 0；
- open-ended invalid response 显示 `MISS reason=... actual=...`；
- 不向用户显示 `expected=None` 或 `number=None`；
- hard-mode loss 使用 `failure_reason`；
- `:expected` 对 open-ended task 解释“没有唯一答案”；
- summary 和 invalid details 只在字段有值时打印 `number` / `expected`；
- `TaskRefereeResult.metadata` 写入 `metadata.referee_metadata`；
- 旧 trace 字段保留，新增 metadata 为 optional，schema version 不变。

Animal Naming trace 可观察：

```text
metadata.human_input.raw_transcript
metadata.human_input.canonical_response
metadata.referee_metadata.normalized_animal
metadata.failure_reason
metadata.timing_breakdown
metadata.speech_output
```

### 3.5 CLI 与边界校验

主要文件：`src/streammuse/presentation/task/cli.py`

完成内容：

- `animal_naming` 加入 interactive task registry；
- `play` 和 `run` 使用同一个 domain task；
- `max_turns <= 0` 在资源初始化前失败；
- Animal Naming 的 `max_turns > 91` 在资源初始化前给出 task-specific error；
- 通用 `--start-number` 参数继续被接受，但不影响 Animal Naming state；
- 现有 STT、TTS、deadline 和 model server flags 原样复用；
- 没有新增 Animal-only CLI flag，也没有改变全局 voice defaults。

已验证 `--max-turns 92` 返回 exit code 2，并显示：

```text
invalid interactive task configuration: Animal Naming max_turns cannot exceed the whitelist size (91)
```

## 4. 文档改动

新增：

- `docs/user-guide/animal-naming.md`

更新：

- `README.md`
- `docs/index.md`
- `docs/reference/cli-reference.md`
- `docs/user-guide/voice-input.md`
- `docs/user-guide/speech-output.md`
- `docs/user-guide/voice-latency-breakdown.md`

文档覆盖 exact whitelist 语义、terminal、STT、TTS、完整双向语音命令、headphones 建议、session cache 语义和 trace/analyzer 用法。

## 5. 测试与验证

### 5.1 Focused automated tests

最终 focused command：

```bash
uv run pytest \
  tests/unit/domain/tasks/test_animal_naming_task.py \
  tests/unit/domain/tasks/test_speech_rendering.py \
  tests/unit/application/tasks/test_interactive_runtime.py \
  tests/unit/application/tasks/test_interactive_speech_runtime.py \
  tests/unit/infrastructure/voice/test_faster_whisper.py \
  tests/unit/presentation/task/test_task_cli.py \
  tests/unit/presentation/task/test_speech_cli.py \
  tests/unit/scripts/test_analyze_interactive_voice_latency.py -q
```

结果：

```text
161 passed in 3.00s
```

覆盖范围包括：

- batch compatibility；
- Human/LLM shared state；
- known、unknown、repeated、empty 和 malformed speech；
- actor-aware history；
- 91 项 deterministic hotword rendering；
- TTS empty prewarm vocabulary 和 on-demand synthesis；
- soft/hard invalid behavior；
- open-ended trace 和 summary；
- unknown LLM response；
- terminal、voice、silent、audio 以及组合 STT/TTS CLI config；
- max-turn boundaries 和 ignored start number；
- latency analyzer 对 `number=null` / `expected=null` 的兼容性；
- Zip Zap Zop regression paths。

### 5.2 全量回归

命令：

```bash
uv run pytest tests/ -q
```

结果：

```text
709 passed, 4 skipped, 15 failed, 1 warning in 19.92s
```

15 个失败都位于本 feature 未修改的模块：

- 1 个 `tests/integration/test_lekai_session_contract.py` failure；
- 14 个 melody robustness campaign / qualification / report failures。

主要 failure patterns 是：

- `qualification_candidate` 与测试预期的 `qualified_frozen` 不一致；
- 部分测试构造的 `argparse.Namespace` 缺少 `config_sha256`；
- direct campaign freeze 已被生产代码禁用，但旧测试仍调用该入口；
- Lekai history-trim contract 的 content validity 为 false。

这些文件不在本次 Animal Naming diff 中。本次没有顺带修改这些独立问题，以避免扩大 feature scope。由于没有“修改前的全量 suite 零失败”证据，本报告只声明它们与本 feature 的修改范围无关，不把全量 suite 写成 pass。

4 个 skip 是显式 opt-in：两个 consistency checkpoint tests，以及两个需要真实 local faster-whisper model / prerecorded audio 的 integration tests，不是 silent pass。

### 5.3 Static checks

- Python `compileall`：通过。
- `git diff --check`：通过。
- accidental audio/cache/trace tracked files：未发现。
- repository formatter command：未配置，N/A。
- repository lint command：未配置，N/A。
- repository type-check command：未配置，N/A。

### 5.4 真实 CLI smoke

已经通过真实 `streammuse-task` entry point 完成一轮 terminal Human smoke；输入为 `lion`，结果为 1 valid、0 invalid、0 deadline miss，run status 为 `completed`。

trace：

```text
/private/tmp/streammuse-animal-terminal-smoke/
  animal_naming_interactive_20260813-131546_0fd3860e/
```

关键 trace 内容：

```json
{
  "actor": "human",
  "number": null,
  "response": "lion",
  "expected": null,
  "is_valid": true,
  "metadata": {
    "failure_reason": "NONE",
    "referee_metadata": {"normalized_animal": "lion"}
  }
}
```

这个 trace 也成功运行了现有 latency analyzer，输出位于：

```text
/private/tmp/streammuse-animal-terminal-smoke-analysis/
```

这证明 analyzer 不依赖 Zip-specific `number` 或唯一 `expected`；它不构成 voice latency qualification。

## 6. 尚未完成

以下内容在 plan 中保持未勾选：

1. 保存当前 Zip Zap Zop 实机双向语音 trace 作为 reference。
2. 用固定真实 Animal Naming 音频比较 hotwords on/off。
3. 测量 hotword context 对真实 STT latency 的影响。
4. 审计是否出现 hotword repetition hallucination。
5. 完成 terminal-only 10-turn game。
6. 完成 terminal + real TTS 10-turn game。
7. 完成至少一局 microphone + remote LLM + speaker 的 20-turn smoke。
8. 完成 5 局、每局 20 turns 的正式 qualification。
9. 计算 exact canonical recognition rate 和各阶段 p50/p95。
10. 实机检查 TTS echo、audio truncation、model reload 和 playback failure。

因此以下目标目前不能宣称达标：

- recognition rate >= 90%；
- Human end-to-end latency p95 <= 2500 ms；
- 无系统性 echo / hallucination；
- 所有真实 TTS playback outcome 正常。

## 7. 建议的下一步测试

### 7.1 先跑 terminal 10 turns

```bash
uv run --frozen streammuse-task play \
  --task animal_naming \
  --human-input terminal \
  --human-first \
  --max-turns 10 \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-text
```

这局应主动覆盖 valid、repeated、真实但不在 whitelist 三种 Human answer，并检查 LLM 是否重复 Human 已用动物。

### 7.2 再跑完整双向语音

先确认 SSH forwarding：

```bash
curl http://127.0.0.1:8101/v1/models
```

再运行：

```bash
uv run --frozen --extra voice --extra speech streammuse-task play \
  --task animal_naming \
  --human-input voice \
  --human-first \
  --max-turns 20 \
  --microphone-device 2 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --voice-max-utterance-ms 1500 \
  --voice-save-audio \
  --speech-output audio \
  --speech-backend system \
  --speech-cache-miss synthesize \
  --speech-guard-ms 200 \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --timeout-s 30 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-bidirectional
```

建议使用 headphones。每局完成后保留整个 run directory，不要只保存 terminal output。

### 7.3 分析 trace

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  /private/tmp/streammuse-animal-bidirectional/<run-directory> \
  --output-dir /private/tmp/streammuse-animal-latency-analysis
```

正式 qualification 应收集 5 局、每局 20 turns，并为每个 Human turn 补充 intended animal，才能计算真实 recognition accuracy 和区分 capture、STT、parser、referee 四类错误。

## 8. 当前状态判定

| 项目 | 状态 | 说明 |
|---|---|---|
| Domain implementation | PASS | shared state、exact whitelist、parser/referee 分层完成 |
| Runtime implementation | PASS | open-ended 与 unique-answer task 共存 |
| CLI integration | PASS | `run`/`play`、STT/TTS flags 和边界校验完成 |
| Focused automated tests | PASS | 161 passed |
| Real CLI terminal smoke | PASS | one-turn valid path + trace + analyzer |
| Full repository suite | BLOCKED OUTSIDE SCOPE | 709 passed、4 skipped、15 个未改动模块失败 |
| Real-model STT comparison | PENDING | 需要固定 audio 与本地 model opt-in |
| Real TTS / speaker test | PENDING | 尚未实机播放 |
| 5-game qualification | PENDING | 尚无 recognition/latency gate 结论 |

软件实现可以进入实机 qualification，但整个 feature 尚未满足 plan 第 15 节的最终完成定义。

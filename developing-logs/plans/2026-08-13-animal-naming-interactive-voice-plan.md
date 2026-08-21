# Animal Naming 双向语音交互实现计划

> 日期：2026-08-13
>
> 状态：Implemented / 实机 qualification 待完成
>
> 目标分支：`feature/voice`
>
> 目标游戏：`animal_naming`
> 参考实现：当前已经支持双向语音的 `zip_zap_zop`

## 1. 背景与目标

当前项目已经具备一条完整的双向语音链路：

```text
Human speech
  -> microphone capture
  -> VAD / utterance segmentation
  -> faster-whisper STT
  -> task-specific speech parser
  -> game referee
  -> shared game state

LLM text
  -> task-specific text parser / referee
  -> task-specific spoken-text renderer
  -> TTS synthesis
  -> audio playback
  -> shared game state
```

`zip_zap_zop` 已经接入这条链路，但 `animal_naming` 当前只支持 `streammuse-task run` 的 benchmark 模式，不能通过 `streammuse-task play` 与真人交替进行双向语音游戏。

本功能的目标是把 `Animal Naming` 变成第二个双向语音游戏，使真人和 LLM 轮流说出一个尚未使用过的动物名称，并由同一个 domain referee 判断答案是否有效。

### 1.1 用户体验目标

一个典型回合应当是：

```text
[1] You > Name one unused animal:
    ASR > raw="A lion." parsed="lion"
    OK

[2] LLM thinking...
    LLM > elephant
    TTS > synthesized and played
    OK

[3] You > Name one unused animal:
    ASR > raw="Lion." parsed="lion"
    MISS reason=REPEATED_ANIMAL actual=lion
```

### 1.2 V1 规则定义

V1 延续当前 `AnimalNamingTask` 的规则，不改成累计复述型的 “I went to the zoo...” memory game：

1. Human 和 LLM 交替进行。
2. 每个回合只能回答一个常见英文动物名称。
3. 动物必须存在于当前 `DEFAULT_ANIMALS` whitelist 中。
4. 整场游戏中不能重复已经成功使用过的动物。
5. Human 和 LLM 共享同一份 `used_animals` 状态。
6. 已出现的非法但非空答案继续进入 `attempted_animals`，供后续 LLM prompt 避免重复尝试。
7. `soft` deadline mode 下非法答案只记录失败，游戏继续。
8. `hard` / challenge mode 下沿用现有 runtime 的失败终止规则。
9. 画面上的 turn cue 只使用文字；TTS 只朗读 LLM 的游戏答案。
10. V1 仅支持英文，和 `faster-whisper tiny.en` 以及当前英文 whitelist 保持一致。

## 2. 当前实现审计

### 2.1 已经可以复用的能力

- `InteractiveTaskRuntime` 已经负责 Human / LLM 轮换、deadline、trace 和 summary。
- `SpeechAwareInteractiveTask` 已经定义 Human speech 的上下文和解析接口。
- `SpeechRenderableTask` 已经定义 LLM 输出到 TTS 文本的接口。
- Voice input 已支持 microphone、VAD、`faster-whisper`、raw transcript 保存和 audio 保存。
- Speech output 已支持 prepare、cache、on-demand synthesis、playback timing 和失败策略。
- `zip_zap_zop` 可作为 task-level speech adapter 的直接参考。
- breakdown analyzer 已能分析 capture、STT、LLM、TTS synthesis 和 playback 阶段。

### 2.2 当前缺口

1. CLI 的 `INTERACTIVE_TASKS` 只有 `zip_zap_zop`，会拒绝 `play --task animal_naming`。
2. `AnimalNamingTask` 尚未实现 `InteractiveTask`、`SpeechAwareInteractiveTask` 和 `SpeechRenderableTask` 所需行为。
3. 当前 task history 默认把所有答案记为 `assistant`，不能表达 Human / LLM 两种 actor。
4. Animal Naming 没有唯一的 `expected` 答案，而 runtime 的部分终端输出仍假定 `expected` 一定存在。
5. 当前 TTS `prepare()` 会预先合成 `speech_vocabulary()` 的全部短语；如果把约百个动物名全部返回，会显著拖慢启动。
6. 当前 trace 会记录 `failure_reason`，但 task referee 的 `normalized_animal` 等 metadata 没有完整进入 response trace。
7. 现有 CLI 测试明确断言 Animal Naming 不能 interactive，需要替换为正向覆盖。

## 3. 设计原则

### 3.1 Speech layer 只负责转写和规范化

语音输入层不能决定一个动物是否符合游戏规则。职责必须保持为：

```text
STT raw transcript
  -> speech parser: 得到候选动物文本
  -> referee: 判断 whitelist / repetition / validity
```

例如：

| Raw transcript | Speech parser | Referee |
|---|---|---|
| `A lion.` | `lion` | valid 或 `REPEATED_ANIMAL` |
| `Dragon.` | `dragon` | `UNKNOWN_ANIMAL` |
| `I think maybe lion.` | unrecognized | `EMPTY_RESPONSE` 或 speech parse failure |

尤其不能让 speech parser 先用 whitelist 把 `dragon` 清空，否则 referee 无法产生正确的 `UNKNOWN_ANIMAL`，语音模式也会和文字模式产生不同的规则语义。

### 3.2 不使用 fuzzy matching

V1 不做以下自动纠错：

- edit-distance correction；
- embedding / LLM semantic correction；
- synonym collapsing；
- 把相似声音强行映射到某个 whitelist animal。

原因是这类行为会把 STT 错误和游戏判定混在一起，并可能把玩家没有说过的词判成正确答案。所有原始文本、解析文本和 referee 结果都应保留在 trace 中，先通过数据确认真实错误类型。

### 3.3 保持现有 whitelist 语义

V1 保留当前 whitelist，包括 `rhino` 和 `rhinoceros` 这种可能具有同义关系的独立条目。这样不会改变既有 benchmark 结果。是否引入 canonical animal ontology 留到后续独立设计。

### 3.4 Open-ended task 不伪造 expected answer

Animal Naming 在任一回合都有多个合法答案，所以：

- `expected_for_state()` 返回 `None`；
- referee 通过 whitelist 和 `used_animals` 判断；
- CLI 不显示 `expected=None`；
- hint 只提示剩余数量和规则，不直接提供某个正确答案。

## 4. Domain Task 改造

目标文件：

- `src/streammuse/domain/tasks/animal_naming.py`

### 4.1 保持 batch API 向后兼容

现有 `run --task animal_naming` 及其测试不能被 interactive feature 破坏。以下方法增加可选参数，但保留已有 positional 调用方式：

```python
validate_response(state, response, *, actor=None, transcript=None)
advance_state(state, referee_result, response_text, *, actor=None, transcript=None)
```

`actor=None` 时保持当前 batch 行为；interactive runtime 传入 `human` 或 `llm` 时，把真实 actor 写入 history。

### 4.2 抽取共享规范化 helper

把当前动物名称的规范化行为集中到一个纯函数，供文字判定、speech parser 和 spoken renderer 复用：

```text
- 只读取第一行
- trim whitespace
- lowercase
- 去掉首尾 punctuation
- 去掉单个英文 article: a / an / the
- 压缩连续空格
```

这个 helper 只做文本规范化，不做 whitelist membership 判断。

### 4.3 实现 InteractiveTask 行为

需要增加：

- `initial_state()`：返回空的 `used_animals`、`attempted_animals` 和 history。
- `build_human_prompt()`：返回简短文字 cue，例如 `Name one unused animal:`。
- `build_llm_messages()`：构造面向对局的 system / user messages。
- `expected_for_state()`：固定返回 `None`。
- `build_hint()`：返回剩余动物数量和“不重复”的提醒，不直接泄漏答案。

LLM prompt 必须明确：

1. 正在与 Human 轮流玩 Animal Naming。
2. 只输出一个英文动物名称。
3. 不要解释、加标点、编号或 thinking text。
4. `used_animals` 和 `attempted_animals` 都不可再输出。
5. 输出必须来自允许的 whitelist。

`build_llm_messages()` 应复用 batch `build_turn()` 的核心 message builder，避免两个模式以后产生不同规则。

### 4.4 修正 actor-aware history

成功 advance 时：

- Human 回合记录 role `human`；
- LLM 回合记录 role `assistant`；
- batch 模式保持现有 role 行为。

只有 referee 判定成功的动物进入 `used_animals`。非空但非法的规范化动物进入 `attempted_animals`，维持当前任务的行为。

### 4.5 明确定义 exhaustion

默认交互局数远小于 whitelist 大小，但 finite whitelist 仍需显式处理：

- task 增加 `remaining_animals(state)` helper；
- `play --task animal_naming` 在启动前校验 `max_turns <= len(task.animals)`；
- batch `run` 行为不变；
- 不在 V1 增加新的“提前胜利”状态机。

该校验防止配置要求的成功 turn 数从理论上超过合法唯一答案总数，同时不改变 invalid turn 的处理方式。

## 5. Human Speech Input 设计

### 5.1 实现 SpeechAwareInteractiveTask

增加：

```python
build_speech_context(state, prompt) -> SpeechContext
parse_spoken_response(state, transcript) -> ParsedSpokenResponse
```

### 5.2 SpeechContext

建议配置：

```text
initial_prompt:
  "The speaker will say exactly one common English animal name."

hotwords:
  当前仍未使用的动物 whitelist
```

第一版使用完整 remaining whitelist，而不是先引入任意截断策略。词表规模仍处于可控范围，并且完整列表可以避免只对某一部分动物产生识别偏置。

需要用测试和实测确认：

- hotwords 顺序 deterministic；
- 不包含重复项；
- 已使用动物会从 hotwords 中移除；
- prompt / hotword 渲染没有超过 `faster-whisper` 实际可接受范围；
- hotword 注入没有明显增加 STT latency 或重复 hallucination。

如果数据表明完整词表产生退化，再单独增加 deterministic cap，而不是在没有测量的情况下提前截断。

### 5.3 Spoken response parser

Parser 接受一个受限的英文动物候选短语：

- 允许大小写差异；
- 允许尾部标点；
- 允许 `a` / `an` / `the` article；
- 允许一到三个 alphabetic words，为未来多词动物名保留空间；
- 不要求候选词已经存在于 whitelist；
- 拒绝空文本、明显解释性句子和 generic transcript quality gate 判定的 pathological output。

示例：

| Raw | Parsed |
|---|---|
| `Lion.` | `lion` |
| `A rhinoceros.` | `rhinoceros` |
| `Dragon.` | `dragon`，随后由 referee 判 unknown |
| `Lion, lion, lion...` | unrecognized / pathological |
| `I would like to say lion.` | unrecognized |

Parser 返回结果必须保留：

- raw transcript；
- canonical response；
- parse status；
- parse failure reason。

### 5.4 VAD 参数建议

Animal name 的长度比 `Zip` / `Zap` / `Zop` 更不稳定，特别是 `hippopotamus`、`rhinoceros` 等词，因此不把全局默认值强制改为 1 秒。

第一轮 manual qualification 使用：

```text
--voice-max-utterance-ms 1500
```

如果保存的音频显示长词被截断，再升到 1800–2000 ms；如果多数回合总是撞到 1500 ms，则优先检查 VAD end-of-speech 参数，而不是继续增加 hard maximum。

## 6. LLM Text 与 TTS Output 设计

### 6.1 LLM response validation

LLM 输出经过同一个规范化 helper 和同一个 referee，不允许为 LLM 设置更宽松的规则。

以下均判非法：

- explanation；
- 多个动物；
- whitelist 之外的动物；
- Human 或 LLM 之前已经成功使用的动物；
- 之前已经非法尝试且 prompt 明确禁止再次输出的动物。

### 6.2 实现 SpeechRenderableTask

增加：

```python
build_spoken_text(state, response, referee_result) -> str | None
speech_vocabulary(state) -> tuple[str, ...]
```

`build_spoken_text()` 的规则：

- 对可解析的单个动物短语返回规范化动物名；
- repeated 或 unknown 的短动物词仍可朗读，因为它确实是 LLM 给出的游戏答案；
- explanation、多词列表、空输出等不可安全渲染的内容返回 `None`；
- TTS 朗读不改变 referee 结果。

### 6.3 禁止预热整个动物词表

`speech_vocabulary()` 对 Animal Naming 返回空 tuple。

原因：当前 `AudioSpeechOutput.prepare()` 会同步预合成 vocabulary。Zip Zap Zop 的固定 vocabulary 很小，适合预热；Animal Naming 有约百个候选词，并且每个有效词在一局中最多使用一次，启动时预合成全部词汇成本高、复用率低。

Animal Naming 使用现有 on-demand synthesis：

```text
--speech-cache-miss synthesize
```

这会让每个 LLM 回合遇到某个动物时即时合成。当前 audio cache 是 session 内存缓存，并不会跨局持久化；现有 breakdown trace 继续记录每回合的 synthesis latency。

正常游戏不建议使用：

```text
--speech-cache-miss skip
```

因为大多数动物在启动时不会预热，`skip` 会导致 LLM 答案没有音频。

### 6.4 不在 V1 增加 TTS prefetch

V1 不增加后台合成线程、rolling prefetch 或预测下一个 LLM 答案。这些会显著增加并发、取消和 deadline 语义复杂度。先收集 on-demand synthesis 数据，再决定是否值得优化。

## 7. Interactive Runtime 通用化

目标文件：

- `src/streammuse/application/tasks/interactive_runtime.py`

### 7.1 修复 open-ended task 的终端输出

当前部分输出默认游戏有唯一 expected value。需要改为条件渲染：

当 `expected is not None`：

```text
MISS expected=Zip actual=Zap
```

当 `expected is None`：

```text
MISS reason=REPEATED_ANIMAL actual=lion
```

同样处理：

- hard-mode loss message；
- invalid response summary；
- turn detail 中的 `number` / `expected` 字段；
- `:expected` 命令在 open-ended task 中给出 task-neutral 信息，而不是打印 `None`。

### 7.2 通用化 help 文案

把只适用于 Zip Zap Zop 的 “answer the next value” 改为 task-neutral 文案，例如：

```text
Answer the task prompt. Commands: :help, :hint, :expected, :summary, :quit
```

Zip Zap Zop 的已有输出和逻辑在存在 `expected` 时保持不变。

### 7.3 扩充 trace referee metadata

在 response trace metadata 中增加 task referee metadata，例如：

```json
{
  "failure_reason": "REPEATED_ANIMAL",
  "referee_metadata": {
    "normalized_animal": "lion"
  }
}
```

已有字段不删除、不改名；新字段保持 optional，因此无需提升现有 manifest schema version。

Animal Naming 的每个 Human voice turn 最终应能关联：

- audio artifact；
- raw transcript；
- parsed / canonical response；
- normalized animal；
- referee validity 和 failure reason；
- capture / STT / total latency。

LLM turn 还应包含：

- raw model response；
- normalized animal；
- LLM latency；
- TTS synthesis latency；
- playback latency / drained status。

## 8. CLI 接入

目标文件：

- `src/streammuse/presentation/task/cli.py`

改动：

1. 将 `animal_naming` 加入 `INTERACTIVE_TASKS`。
2. 让 `streammuse-task play --task animal_naming` 创建已有 `AnimalNamingTask`。
3. 保持 `streammuse-task run --task animal_naming` 完全可用。
4. 对 interactive Animal Naming 添加 `max_turns <= whitelist size` 校验。
5. 沿用现有通用 STT、TTS、deadline 和 model flags，不新增 Animal-only CLI flag。
6. `--start-number` 对 Animal Naming 没有业务含义；V1 继续容忍这个通用参数，但不使用它，避免扩大 CLI 重构范围。

推荐的默认运行策略：

- `--deadline-mode soft` 先做可观察性测试；
- `--voice-max-utterance-ms 1500`；
- `--max-tokens 8`；
- `--speech-cache-miss synthesize`；
- 使用 headphones，避免 TTS 回放进入下一次 microphone capture。

## 9. 自动化测试计划

### 9.1 Domain tests

目标文件：

- `tests/unit/domain/tasks/test_animal_naming_task.py`
- `tests/unit/domain/tasks/test_speech_rendering.py`

新增或调整测试：

- [x] `AnimalNamingTask` 具备 `InteractiveTask` 全部方法，并满足两个 runtime-checkable speech protocols。
- [x] batch 的旧 positional API 和既有测试保持通过。
- [x] initial state 正确。
- [x] Human prompt 简洁且不泄漏答案。
- [x] LLM messages 包含完整规则、used 和 attempted animals。
- [x] Human 与 LLM 成功答案写入同一份 used state。
- [x] history role 根据 actor 正确记录。
- [x] valid、empty、unknown、repeated 的 referee 结果正确。
- [x] unknown speech candidate 先被 parser 保留，再由 referee 判 `UNKNOWN_ANIMAL`。
- [x] repeated speech candidate 先解析成功，再由 referee 判 `REPEATED_ANIMAL`。
- [x] article、case 和 punctuation 规范化正确。
- [x] explanation、multiple animals、pathological repetition 被 speech parser 拒绝。
- [x] speech context hotwords deterministic、无重复、排除 used animals。
- [x] `expected_for_state()` 返回 `None`。
- [x] hint 只报告剩余数量和规则。
- [x] spoken renderer 对短候选词返回规范化文本，对 explanation 返回 `None`。
- [x] `speech_vocabulary()` 为空，防止全词表 TTS prewarm。

### 9.2 Runtime tests

目标文件：

- `tests/unit/application/tasks/test_interactive_runtime.py`
- `tests/unit/application/tasks/test_interactive_speech_runtime.py`

覆盖：

- [x] 4–6 turn 的 Human / LLM alternating happy path。
- [x] Human 用过的动物会导致 LLM repeat 失败，反向亦然。
- [x] voice raw transcript、parsed response 和 referee metadata 均写入 trace。
- [x] Animal Naming 的 audio prepare 收到空 vocabulary。
- [x] LLM animal 使用 on-demand synthesis 并正常 playback。
- [x] `cache_miss_synthesized` 等现有 speech status 正确记录。
- [x] speech output failure 的 `warn` / `fail` 策略不变。
- [x] soft mode 非法答案后继续下一 turn。
- [x] hard mode 非法答案终止且不出现 `expected=None`。
- [x] open-ended summary 不打印 `number=None` / `expected=None`。
- [x] Zip Zap Zop 的 existing formatted output 和 deadline semantics 不回归。

### 9.3 CLI tests

目标文件：

- `tests/unit/presentation/task/test_task_cli.py`

覆盖：

- [x] 替换“Animal Naming interactive 被拒绝”的旧断言。
- [x] `play --task animal_naming --human-input terminal` 可启动。
- [x] `play --task animal_naming --human-input voice` 正确传递 voice config。
- [x] speech output config 正确传递到 runtime。
- [x] 超过 whitelist size 的 `max_turns` 给出明确 CLI error。
- [x] `run --task animal_naming` 的既有路径不变。
- [x] manifest / trace 中 task name 和 provenance 正确。

### 9.4 Regression suite

- [x] 当前 Animal Naming batch tests 全部通过。
- [x] 当前 Zip Zap Zop interactive tests 全部通过。
- [x] Voice input unit tests 和不依赖真实模型的 integration coverage 通过。
- [ ] 真实 `faster-whisper` model integration tests；当前因未提供 opt-in model 环境变量而明确 skip。
- [x] Speech output unit tests 和 fake sink integration coverage 通过。
- [ ] 真实 TTS engine / speaker integration；保留到 Phase 7 实机 qualification。
- [x] CLI tests 通过。
- [x] 已运行全量 `pytest`：709 passed、4 skipped、15 个位于未改动模块的既有失败。
- [x] 仓库未配置 formatter / lint / type-check 命令；已通过 `compileall` 和 `git diff --check`。

## 10. Manual Qualification

### 10.1 阶段 A：纯文字、无 TTS

先排除 voice 和 audio 变量：

```bash
uv run --frozen --extra voice streammuse-task play \
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

检查：turn alternation、共享去重状态、unknown/repeated 原因、trace 和 summary。

### 10.2 阶段 B：文字输入 + LLM TTS

启用现有 speech output flags，验证：

- LLM 每次只读动物名；
- prepare 阶段没有预合成全部 whitelist；
- 首次 animal 为 on-demand synthesis；
- 新 session 重新按需合成，且不会误报不存在的 disk cache hit；
- synthesis 或 playback 失败符合配置策略。

### 10.3 阶段 C：完整双向语音

最终命令以项目实际 speech engine flags 为准，核心参数如下：

```bash
uv run --frozen --extra voice streammuse-task play \
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
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --timeout-s 30 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-voice
```

该命令需要补上已经实现的 TTS engine / voice / cache flags。使用 headphones，防止扬声器回放被下一轮 STT 收录。

### 10.4 数据采集

至少完成 5 局，每局 20 turns；Human 共获得约 50 个 voice turns。记录：

- Human 说出的 intended animal；
- raw transcript；
- parsed animal；
- referee result；
- capture / voiced utterance / STT / total Human latency；
- LLM latency；
- TTS synthesis / playback latency；
- cache hit / miss；
- 是否出现 TTS echo、hallucination 或 audio truncation。

运行现有 breakdown analyzer，并确认其对 `animal_naming` trace 没有依赖 Zip-specific `number` / `expected` 字段。若 analyzer 只依赖 actor 和 timing metadata，则无需修改 analyzer；若存在 task-specific 假设，再用最小改动通用化。

## 11. Acceptance Criteria

### 11.1 软件正确性

- [x] `streammuse-task play --task animal_naming` 可以启动并完成多轮对局。
- [x] Human 和 LLM 共用同一份 no-repeat state。
- [x] terminal、voice 和 LLM response 使用同一个 referee。
- [x] unknown 和 repeated response 得到准确且可观察的失败原因。
- [x] open-ended 游戏输出不出现 `expected=None` 或 `number=None`。
- [x] LLM 的 animal answer 能被 TTS 朗读。
- [x] 启动时不预合成完整动物 whitelist。
- [x] trace 能还原 raw speech、canonical response、normalized animal、referee 和 latency breakdown。
- [x] 当前 Animal Naming batch mode 与 Zip Zap Zop 双向语音均无回归。

### 11.2 实机 qualification 建议门槛

在安静环境、固定 microphone 和 50 个 Human voice turns 上：

- [ ] whitelist animal 的 exact canonical recognition rate >= 90%。
- [ ] Human turn end-to-end latency p95 <= 2500 ms，使用 1500 ms max utterance 配置。
- [ ] 不出现 voice infrastructure crash 或 model reload。
- [ ] LLM 所有可渲染答案都有 speech outcome，失败时 trace 原因明确。
- [ ] 未观察到系统性 TTS echo 被下一回合 STT 当作 Human 输入。
- [ ] 不出现因 hotword 列表导致的长重复 hallucination；若出现，保存对应 audio 和 raw transcript。

这些实机指标是 qualification gate，不替代 deterministic automated tests。

## 12. 风险与缓解措施

### 12.1 `tiny.en` 对少见动物名识别较差

缓解：使用 remaining whitelist hotwords；保存原始音频；先依据 confusion matrix 决定是否调整 whitelist、prompt 或模型，不做静默 fuzzy correction。

### 12.2 Hotwords 导致错误偏置或重复 hallucination

缓解：对比启用/禁用 hotwords 的固定音频集；记录 raw segment；如果完整列表退化，再引入有测试的 deterministic cap。

### 12.3 长动物名被 VAD / hard max 截断

缓解：第一版使用 1500 ms；通过 `--voice-save-audio` 检查；必要时提升到 1800–2000 ms，并单独分析 voiced utterance 和 trailing silence。

### 12.4 TTS on-demand synthesis 占据 LLM turn latency

缓解：保留 synthesis breakdown；先测量真实 p50/p95。只有数据证明需要时再设计持久化 audio cache 或 async prefetch。

### 12.5 Speaker echo 进入下一轮 microphone

缓解：runtime 必须在 playback 完成后才打开 Human input；qualification 使用 headphones。AEC 和 barge-in 不属于 V1。

### 12.6 Whitelist 与真实动物语义不完全一致

缓解：V1 明确使用 exact whitelist，保证 benchmark comparability；synonym ontology 和复数形式支持另开 plan。

### 12.7 LLM 输出 explanation 或 thinking text

缓解：严格 system prompt、`temperature=0`、小 `max_tokens`、task parser 和 referee；无论模型输出什么都不能绕过 domain validation。

## 13. 明确不在本次范围内的内容

- 累计复述型 zoo memory game；
- 多语言动物名称；
- fuzzy matching、embedding 或 LLM-based correction；
- animal synonym ontology；
- automatic plural / taxonomy inference；
- Human turn cue 的语音播报；
- barge-in、full-duplex capture 或 acoustic echo cancellation；
- TTS background prefetch；
- GUI；
- 对全局 voice defaults 的强制修改。

## 14. 实施顺序与 Todo List

### Phase 0：冻结基线

- [x] 记录当前 branch、working tree 和基线测试结果。
- [x] 单独运行现有 Animal Naming、Zip Zap Zop、voice 和 speech output 测试。
- [ ] 保存一个现有 Zip Zap Zop 双向语音 trace 作为 regression reference。

### Phase 1：Domain interactive contract

- [x] 抽取 Animal Naming 共享文本规范化 helper。
- [x] 保持 batch API 向后兼容。
- [x] 实现 Human prompt、LLM messages、expected 和 hint。
- [x] 实现 actor-aware history 和 shared state advance。
- [x] 增加 remaining animals / exhaustion helper。
- [x] 完成 domain unit tests。

### Phase 2：Speech input adapter

- [x] 实现 `build_speech_context()`。
- [x] 实现不依赖 whitelist membership 的 spoken parser。
- [x] 接入 existing generic transcript quality gate。
- [x] 覆盖 known、unknown、repeated、explanation 和 pathological transcript tests。

### Phase 3：Speech output adapter

- [x] 实现 `build_spoken_text()`。
- [x] 明确 `speech_vocabulary()` 返回空 tuple。
- [x] 覆盖 on-demand synthesis 和 unrenderable response tests。

### Phase 4：Runtime open-ended support

- [x] 条件渲染 `expected` / `number`。
- [x] 通用化 help、hard loss 和 summary 文案。
- [x] 把 referee metadata 加入 response trace。
- [x] 增加 Animal Naming alternating runtime tests。
- [x] 保持 Zip Zap Zop output regression tests 通过。

### Phase 5：CLI 与文档

- [x] 将 `animal_naming` 注册为 interactive task。
- [x] 增加 interactive max-turn validation。
- [x] 替换旧的 CLI rejection test。
- [x] 更新 voice input、speech output 和 task user guide。
- [x] 添加 terminal、TTS 和 full voice 示例命令。

### Phase 6：自动验证

- [x] 运行 focused domain tests。
- [x] 运行 runtime / speech tests。
- [x] 运行 CLI tests。
- [x] 运行完整 test suite；最终结果为 709 passed、4 skipped、15 个无关既有失败。
- [x] 运行 repository 可用的 compile check 和 `git diff --check`；仓库未配置 formatter/lint/type check 命令。
- [x] 检查 diff，确认没有修改无关模块或生成物。

### Phase 7：实机 qualification

- [ ] 完成 terminal-only 10-turn game。
- [ ] 完成 text + TTS 10-turn game。
- [ ] 完成 5 局 full voice、每局 20 turns。
- [ ] 运行 latency breakdown analyzer。
- [ ] 审计 recognition errors、unknown/repeated 和 TTS outcomes。
- [ ] 形成 qualification report，并列出是否达到 acceptance criteria。

## 15. 完成定义

只有同时满足以下条件，本 feature 才算完成：

1. 所有 Phase 1–6 automated tasks 完成且无 regression。
2. 至少完成一局真实的 Human STT + LLM + TTS 多轮游戏。
3. trace 能解释每个失败发生在 capture、STT、parser、referee、LLM、TTS synthesis 或 playback 的哪个环节。
4. `Animal Naming` 的 batch benchmark 行为没有改变。
5. qualification 中发现的已知问题已记录，不以不可复现的“听起来可以”作为验收依据。

## 16. Detailed Todo List

本节是实际实施时使用的完整 checklist。任务按依赖顺序排列；除非任务明确说明可以并行，否则后一 Phase 默认依赖前一 Phase 的结果。

### Phase 0：基线冻结与变更边界

#### T0.1 确认工作目录与分支

- [x] 确认实施发生在 `/Users/zhengbowen/nondefault/Research 2026/StreamMUSE-v1`。
- [x] 确认当前 branch 是 `feature/voice`。
- [x] 运行 `git status --short`，记录已有的用户改动。
- [x] 确认本 feature 不修改或回退与 Animal Naming 无关的改动。

#### T0.2 记录当前行为

- [x] 运行现有 `AnimalNamingTask` domain tests。
- [x] 运行现有 `ZipZapZopTask` domain tests。
- [x] 运行 interactive runtime tests。
- [x] 运行 voice input tests。
- [x] 运行 speech output tests。
- [x] 运行 task CLI tests。
- [x] 保存各组测试的通过数量、跳过数量和运行时间。
- [ ] 保存一次当前 `zip_zap_zop` 双向语音 trace 作为输出格式和 timing regression reference。

#### T0.3 冻结 V1 规则

- [x] 明确 V1 是“每轮说一个新动物”，不是累计复述型 zoo memory game。
- [x] 明确 V1 使用现有 exact whitelist。
- [x] 明确 V1 仅支持英文。
- [x] 明确 synonym、plural、fuzzy matching 和 semantic correction 不在本次范围。
- [x] 明确 Human cue 只显示文字，只有 LLM answer 使用 TTS。

#### Phase 0 退出条件

- [x] 基线测试结果已记录。
- [x] Existing working-tree changes 已识别且不会被覆盖。
- [x] V1 scope 没有未决的规则问题。

### Phase 1：Animal Naming Domain Contract

目标文件：

- `src/streammuse/domain/tasks/animal_naming.py`
- `tests/unit/domain/tasks/test_animal_naming_task.py`

#### T1.1 整理类型与 imports

- [x] 引入 `ChatMessage`。
- [x] 引入 `InteractiveActor`。
- [x] 引入 `InteractiveTurnRecord`。
- [x] 引入 speech protocol 所需的数据类型。
- [x] 保持 `RealtimeTask` 所需的现有类型和方法可用。
- [x] 不把 microphone、STT、TTS 或 runtime dependency 引入 domain module。

#### T1.2 统一 animal normalization

- [x] 保留 `_normalize_animal()` 为纯函数，或提取等价的私有 pure helper。
- [x] 处理 `None` 和空字符串。
- [x] 只使用第一行文本。
- [x] 执行 trim 和 lowercase。
- [x] 去掉首尾 whitespace 和 punctuation。
- [x] 去掉开头单个 `a`、`an` 或 `the`。
- [x] 压缩连续空格。
- [x] 确认 normalization 不查询 whitelist。
- [x] 确认 `Dragon.` 会规范化为 `dragon`，而不是空字符串。
- [x] 增加 normalization parameterized tests。

#### T1.3 保持 batch API 兼容

- [x] 把 `validate_response()` 扩展为接受 optional `actor` 和 `transcript` keyword arguments。
- [x] 把 `advance_state()` 扩展为接受 optional `actor` 和 `transcript` keyword arguments。
- [x] 保留 `advance_state(state, referee_result, response_text)` 的旧调用方式。
- [x] 保留 `build_turn()`。
- [x] 保留现有 batch prompt 的有效规则。
- [x] 确认 `streammuse-task run --task animal_naming` 无需传 interactive-only 参数。
- [x] 增加旧 positional API regression tests。

#### T1.4 实现共享 prompt builder

- [x] 抽取一个供 `build_turn()` 和 `build_llm_messages()` 复用的规则 prompt helper。
- [x] prompt 明确只允许一个 common real English animal name。
- [x] prompt 明确禁止 explanation、编号和额外文本。
- [x] prompt 包含 deterministic forbidden-animal list。
- [x] forbidden list 包含 `attempted_animals`。
- [x] interactive prompt 明确游戏对手是 Human。
- [x] 避免直接把未经处理的 STT raw transcript 写进 LLM prompt。
- [x] 测试 batch 和 interactive prompt 使用同一套核心规则。

#### T1.5 实现 Human prompt

- [x] 增加 `build_human_prompt(state, transcript)`。
- [x] 返回简短、task-specific 的文字 cue。
- [x] cue 不提供某个具体合法动物。
- [x] cue 不包含 keyboard shortcut 或语音使用说明。
- [x] 测试首轮和后续轮次 prompt 均稳定。

#### T1.6 实现 LLM messages

- [x] 增加 `build_llm_messages(state, transcript)`。
- [x] 使用 `system` 和 `user` Chat API roles。
- [x] 包含当前 forbidden animals。
- [x] 不把合法 whitelist 误当作已经使用的动物。
- [x] 保持 message ordering deterministic。
- [x] 控制 prompt 增长，使用 state 中的 compact animal lists，而不是重复整份 turn trace。
- [x] 测试 Human 和 LLM 已尝试的动物都进入 forbidden list。

#### T1.7 Referee 判定

- [x] 保持空答案返回 `EMPTY_RESPONSE`。
- [x] 保持 whitelist 之外答案返回 `UNKNOWN_ANIMAL`。
- [x] 保持已经成功使用的动物返回 `REPEATED_ANIMAL`。
- [x] 合法且未使用的 whitelist animal 返回 valid。
- [x] 所有结果 metadata 包含 `normalized_animal`。
- [x] `expected_output` 保持 `None`。
- [x] `actor` 不改变判定标准。
- [x] Human 和 LLM 使用完全相同的 referee tests。

#### T1.8 Actor-aware state advance

- [x] Human 回合在 domain history 中记录 role `human`。
- [x] LLM 回合在 domain history 中记录 role `assistant`。
- [x] batch 的 `actor=None` 保持原有 assistant-style history 行为。
- [x] 只有 valid animal 进入 `used_animals`。
- [x] 所有非空 normalized attempt 进入 `attempted_animals`。
- [x] 不重复插入相同的 used 或 attempted animal。
- [x] valid、unknown、repeated、empty 回合都正确增加 `turn_index`。
- [x] history 继续遵守 `max_history`。
- [x] history 保存原 response、valid flag 和 normalized animal。

#### T1.9 Open-ended helpers

- [x] 增加 `remaining_animals(state)`。
- [x] remaining set 等于 whitelist 减去 `used_animals`。
- [x] remaining order 在对外输出时 deterministic。
- [x] 增加 `expected_for_state()` 并固定返回 `None`。
- [x] 增加 `build_hint()`。
- [x] hint 显示 remaining count。
- [x] hint 提醒不能重复。
- [x] hint 不直接给出一个动物答案。

#### T1.10 Protocol compatibility tests

- [x] 验证 task 满足 `InteractiveTask` 的全部方法签名。
- [x] 验证 task 满足 `SpeechAwareInteractiveTask`。
- [x] 验证 task 满足 `SpeechRenderableTask`。
- [x] 验证 task 仍满足 existing realtime / batch usage。

#### Phase 1 退出条件

- [x] 所有 domain tests 通过。
- [x] Human 和 LLM 可以用相同 referee 更新同一份 state。
- [x] 现有 Animal Naming batch tests 无回归。

### Phase 2：Human Speech Input Adapter

目标文件：

- `src/streammuse/domain/tasks/animal_naming.py`
- `tests/unit/domain/tasks/test_animal_naming_task.py`
- 必要时使用现有 speech parsing test module

#### T2.1 构造 SpeechContext

- [x] 实现 `build_speech_context(state, transcript)`。
- [x] 设置简短英文 `initial_prompt`，说明输入是一个动物名称。
- [x] 从 `remaining_animals(state)` 构造 hotwords。
- [x] hotwords 不包含已使用动物。
- [x] hotwords 去重。
- [x] hotwords 顺序 deterministic。
- [x] 不在首版任意截断 whitelist。
- [x] 测试第一轮和使用若干动物后的 speech context。

#### T2.2 实现 spoken response parser

- [x] 实现 `parse_spoken_response(state, transcript, raw_text)`。
- [x] 空 transcript 返回 status `empty`。
- [x] 可接受短动物候选词返回 status `ok` 和 canonical text。
- [x] 允许 `a`、`an`、`the`。
- [x] 允许大小写和尾部 punctuation。
- [x] 允许一到三个 alphabetic words。
- [x] 拒绝 explanation sentence。
- [x] 拒绝一次列出多个动物。
- [x] 拒绝 pathological repetition。
- [x] parse failure 返回 stable reason code。
- [x] parser 不检查 whitelist membership。
- [x] parser 不检查 repetition。
- [x] parser 不执行 fuzzy correction。

#### T2.3 验证 parser/referee 分层

- [x] `Lion.` 解析为 `lion`，referee 判 valid。
- [x] `A rhinoceros.` 解析为 `rhinoceros`。
- [x] `Dragon.` 解析为 `dragon`，referee 判 `UNKNOWN_ANIMAL`。
- [x] 已用过的 `lion` 仍解析为 `lion`，referee 判 `REPEATED_ANIMAL`。
- [x] explanation 在 parser 阶段成为 unrecognized，而不是由 whitelist 误判。
- [x] raw transcript、canonical response 和 referee result 可以分别观察。

#### T2.4 检查 faster-whisper context 规模

- [x] 用 production renderer 检查完整 hotword 的 91 个词、字符长度和实际 decode kwargs。
- [ ] 运行固定 audio sample，比较有/无 hotwords 的 transcription。
- [ ] 测量 context injection 是否显著增加 STT latency。
- [ ] 检查是否出现 animal hotword repetition hallucination。
- [ ] 只有实测退化时才设计 deterministic hotword cap。

#### Phase 2 退出条件

- [x] Known、unknown 和 repeated animal 的 parser/referee 责任边界正确。
- [x] Speech context deterministic。
- [x] 没有 whitelist-based silent correction。

### Phase 3：LLM Speech Rendering 与 TTS 策略

目标文件：

- `src/streammuse/domain/tasks/animal_naming.py`
- `tests/unit/domain/tasks/test_speech_rendering.py`
- `tests/unit/application/tasks/test_interactive_speech_runtime.py`

#### T3.1 实现 spoken-text renderer

- [x] 实现 `build_spoken_text(state, transcript, response_text, actor=...)`。
- [x] 对单个可解析动物候选返回 canonical animal text。
- [x] valid animal 可以朗读。
- [x] repeated animal仍可以朗读。
- [x] unknown 但结构合法的短 animal candidate 可以朗读。
- [x] explanation、列表和空输出返回 `None`。
- [x] renderer 不修改 referee result。
- [x] renderer 不重新调用 LLM。

#### T3.2 避免完整 whitelist prewarm

- [x] 实现 `speech_vocabulary(state, max_turns=...)`。
- [x] Animal Naming 固定返回空 tuple。
- [x] 添加测试防止未来误把完整 whitelist 返回给 `prepare()`。
- [x] 确认 runtime 允许空 vocabulary。
- [x] 确认 Zip Zap Zop 的固定 vocabulary prewarm 行为不变。

#### T3.3 On-demand synthesis tests

- [x] fake speech sink 记录 `prepare(())`。
- [x] LLM 首次 animal response 触发 cache-miss synthesis。
- [x] synthesis 完成后触发 playback。
- [x] speech outcome 记录 spoken text、status 和 synthesis timing。
- [x] `cache_miss=skip` 时明确记录 skipped，不假装成功播放。
- [x] `warn` 和 `fail` speech error policies 保持原行为。

#### Phase 3 退出条件

- [x] LLM 可解析答案能够被 TTS 朗读。
- [x] 游戏启动时间不随 whitelist 大小线性增加。
- [x] speech failure 不会改变 referee 的确定性判定。

### Phase 4：Interactive Runtime 的 Open-ended Task 支持

目标文件：

- `src/streammuse/application/tasks/interactive_runtime.py`
- `tests/unit/application/tasks/test_interactive_runtime.py`
- `tests/unit/application/tasks/test_interactive_speech_runtime.py`

#### T4.1 通用化 turn display

- [x] 当 `expected` 存在时保留当前 `expected=... actual=...` 格式。
- [x] 当 `expected` 为 `None` 时显示 `reason=... actual=...`。
- [x] 不输出 `expected=None`。
- [x] `number` 为 `None` 时使用 turn display index，但不把 `number=None` 写入用户可见摘要。
- [x] 保持 Human ASR raw / parsed output 可见。
- [x] 保持 LLM raw response 可见。

#### T4.2 通用化命令输出

- [x] `:help` 使用 task-neutral 文案。
- [x] `:hint` 调用 task-specific hint。
- [x] `:expected` 在 open-ended task 中说明没有唯一答案。
- [x] `:summary` 不显示 meaningless expected / number fields。
- [x] `:quit` 行为不变。

#### T4.3 通用化 hard-mode loss

- [x] Animal Naming hard-mode invalid response 显示 failure reason。
- [x] 不打印 `Wrong answer ... expected None`。
- [x] deadline loss behavior 不变。
- [x] speech failure 的 hard/soft behavior 不变。
- [x] Zip Zap Zop hard-mode output 保持现状。

#### T4.4 Referee metadata trace

- [x] 将 `TaskRefereeResult.metadata` 复制到 response trace 的 nested metadata 字段。
- [x] 保留现有 `failure_reason` 字段。
- [x] 保留现有 raw / parsed voice metadata。
- [x] Animal Naming trace 包含 `normalized_animal`。
- [x] 新字段为 optional，不破坏旧 trace reader。
- [x] 检查是否需要 schema version；如果只是 optional metadata，则不提升。

#### T4.5 Alternating runtime tests

- [x] 构造 Human `lion`、LLM `elephant`、Human `tiger`、LLM `bear` happy path。
- [x] 验证四个 turn 全部使用同一个 state progression。
- [x] 验证 Human 重复 LLM animal。
- [x] 验证 LLM 重复 Human animal。
- [x] 验证 unknown Human animal。
- [x] 验证 unknown LLM animal。
- [x] 验证 soft mode invalid 后继续。
- [x] 验证 hard mode invalid 后终止。
- [x] 验证 final summary 的 valid / invalid count。

#### T4.6 Zip Zap Zop regression

- [x] 运行 existing terminal interactive tests。
- [x] 运行 existing voice interactive tests。
- [x] 运行 existing speech output tests。
- [x] 检查 expected / number 输出没有变化。
- [x] 检查 deadline basis 和 playback completion semantics 没有变化。

#### Phase 4 退出条件

- [x] Runtime 可以同时支持 unique-answer 和 open-ended tasks。
- [x] Trace 中可以还原完整 Animal Naming 判定链路。
- [x] Zip Zap Zop 没有行为或输出回归。

### Phase 5：CLI、Factory 与文档接入

目标文件：

- `src/streammuse/presentation/task/cli.py`
- `tests/unit/presentation/task/test_task_cli.py`
- `docs/user-guide/voice-input.md`
- `docs/user-guide/speech-output.md`
- 其他现有 task usage 文档

#### T5.1 注册 interactive task

- [x] 将 `animal_naming` 加入 `INTERACTIVE_TASKS`。
- [x] 确认 play path 使用现有 `AnimalNamingTask` factory。
- [x] 确认 run path 仍使用相同 domain task。
- [x] 删除或替换“Animal Naming interactive 必须被拒绝”的旧测试。

#### T5.2 校验 CLI 参数

- [x] terminal Human input 可以用于 Animal Naming。
- [x] voice Human input 可以用于 Animal Naming。
- [x] silent speech output 可以用于 Animal Naming。
- [x] audio speech output 可以用于 Animal Naming。
- [x] `max_turns` 必须为正数。
- [x] interactive `max_turns` 不超过 whitelist size。
- [x] `--start-number` 被容忍但不影响 Animal Naming state。
- [x] STT model/cache/revision/device flags 正确传递。
- [x] TTS engine/voice/cache/error-policy flags 正确传递。
- [x] deadline 和 model server flags 正确传递。

#### T5.3 CLI error messages

- [x] Unsupported task error 不再包含 `animal_naming`。
- [x] 超过 whitelist size 时给出 task-specific、可执行的错误信息。
- [x] Voice dependency 缺失时保持现有安装提示。
- [x] TTS dependency 或 engine 缺失时保持现有错误策略。

#### T5.4 CLI tests

- [x] parse terminal-only Animal Naming command。
- [x] parse full voice + TTS Animal Naming command。
- [x] mock runtime 并检查 task/config 注入。
- [x] 测试 max-turn boundary：1、whitelist size、whitelist size + 1。
- [x] 测试 manifest task name 和 provenance。
- [x] 测试 `run --task animal_naming` regression。

#### T5.5 更新文档

- [x] 说明 Animal Naming 的 exact whitelist 规则。
- [x] 说明 whitelist 之外的真实动物仍会得到 `UNKNOWN_ANIMAL`。
- [x] 说明 V1 不合并 synonym 和 plural。
- [x] 添加 terminal-only 命令。
- [x] 添加 terminal + TTS 命令。
- [x] 添加 full voice + TTS 命令。
- [x] 推荐 `--voice-max-utterance-ms 1500` 作为初始实验值。
- [x] 推荐使用 headphones。
- [x] 说明 `--speech-cache-miss synthesize` 是正常运行所需策略。
- [x] 说明 `--speech-cache-miss skip` 会使未缓存动物没有语音。
- [x] 说明如何定位 trace、audio artifacts 和 breakdown report。

#### Phase 5 退出条件

- [x] CLI 可启动两种 Human input mode 和两种 speech output mode。
- [x] 文档命令与实际 `--help` 完全一致。
- [x] Existing `run` mode 仍可用。

### Phase 6：自动化验证与代码质量

#### T6.1 Focused tests

- [x] 运行 Animal Naming domain tests。
- [x] 运行 speech parsing / rendering tests。
- [x] 运行 interactive runtime tests。
- [x] 运行 interactive speech runtime tests。
- [x] 运行 task CLI tests。
- [x] 运行 trace / analyzer tests。

#### T6.2 Full regression

- [x] 运行完整 unit test suite。
- [x] 运行不依赖真实 microphone/model 的 integration tests。
- [x] 对需要硬件或本地模型的 tests 确认是明确 skip，而不是 silent pass。
- [x] 记录 skipped tests 及其手工验证替代方案。

#### T6.3 Static checks

- [x] 确认仓库没有配置可运行的 formatter check，并在报告中标记为 N/A。
- [x] 确认仓库没有配置可运行的 lint check，并在报告中标记为 N/A。
- [x] 确认仓库没有配置可运行的 type check，并在报告中标记为 N/A。
- [x] 运行 `git diff --check`。
- [x] 检查没有 accidental generated files、audio files、cache 或 trace 被加入版本控制。

#### T6.4 Diff review

- [x] 检查所有修改都属于 domain、interactive runtime、CLI、tests 或 docs。
- [x] 检查没有改变 Zip Zap Zop referee。
- [x] 检查没有改变全局 STT/TTS defaults。
- [x] 检查没有增加 fuzzy matching 或 LLM-based validation。
- [x] 检查没有在 audio callback 中引入新的共享可变状态。
- [x] 检查 optional trace metadata 对旧 reader 向后兼容。

#### Phase 6 退出条件

- [x] Focused tests 全部通过。
- [x] Full suite 已执行；15 个失败均位于本 feature 未修改的 Lekai / melody robustness 模块，并已记录。
- [x] 仓库可用的 static checks 通过；未配置项已明确标记 N/A。
- [x] Diff review 无越界改动。

### Phase 7：真实设备与模型 Qualification

#### T7.1 环境准备

- [ ] 确认 `faster-whisper tiny.en int8` snapshot 已在本地 cache。
- [ ] 使用 `--voice-local-files-only` 验证不会临时联网下载。
- [ ] 确认 microphone device index。
- [ ] 确认 TTS engine、voice 和 cache 可用。
- [ ] 建立 SSH forwarding，将远端 OpenAI-compatible LLM server 映射到本地端口。
- [ ] 使用 `/v1/models` 检查 forwarding 和 model id。
- [ ] 使用 headphones，避免 speaker echo。

#### T7.2 Terminal-only smoke test

- [ ] 玩一局 10 turns。
- [x] 在真实 CLI one-turn smoke 中测试一个 valid animal。
- [ ] 主动测试一个 repeated animal。
- [ ] 主动测试一个真实但不在 whitelist 的 animal。
- [x] 检查 terminal output 没有 `expected=None`。
- [ ] 检查 summary 和 trace failure reasons。

#### T7.3 Text + TTS smoke test

- [ ] 玩一局至少 10 turns。
- [ ] 确认启动时没有等待完整 animal whitelist synthesis。
- [ ] 确认每个 LLM animal 只朗读动物名。
- [ ] 确认 on-demand synthesis status。
- [ ] 确认 playback 完成后才开始下一 Human turn。
- [ ] 重玩一局并确认新 session 会重新按需合成，不误报 disk cache hit。

#### T7.4 Full voice smoke test

- [ ] 使用 `--voice-max-utterance-ms 1500`。
- [ ] 使用 `--voice-save-audio`。
- [ ] 完成一局 20 turns。
- [ ] 检查 Human raw transcript 和 parsed animal。
- [ ] 检查长动物名是否被截断。
- [ ] 检查 quiet/no-speech 行为。
- [ ] 检查 TTS echo 是否进入下一轮 capture。
- [ ] 检查任何 pathological repetition transcript。

#### T7.5 正式数据采集

- [ ] 完成 5 局、每局 20 turns。
- [ ] 保留所有 session manifest 和 response trace。
- [ ] 保留 Human voice audio artifacts。
- [ ] 为每个 Human turn 标注 intended animal。
- [ ] 标注 raw transcript 是否正确。
- [ ] 标注 canonical parse 是否正确。
- [ ] 标注 referee 是否正确。
- [ ] 标注错误属于 capture、STT、parser 还是 game rule。
- [ ] 标注 LLM explanation、repeat 和 unknown 输出。
- [ ] 标注 TTS synthesis / playback failure。

#### T7.6 Latency breakdown

- [ ] 对 5 局 trace 运行现有 analyzer。
- [ ] 统计 Human capture latency p50/p95。
- [ ] 统计 voiced utterance latency p50/p95。
- [ ] 统计 STT latency p50/p95。
- [ ] 统计 Human end-to-end latency p50/p95。
- [ ] 统计 LLM latency p50/p95。
- [ ] 统计 TTS synthesis latency p50/p95。
- [ ] 分开统计 TTS cache hit 和 cache miss。
- [ ] 统计 playback duration 和 completion status。
- [x] 用 one-turn Animal Naming smoke trace 确认 analyzer 不依赖 `number` 或唯一 `expected`。

#### T7.7 质量指标

- [ ] 计算 known whitelist animal exact canonical recognition rate。
- [ ] 检查目标是否达到至少 90%。
- [ ] 检查 Human end-to-end p95 是否不高于 2500 ms。
- [ ] 检查是否发生 voice infrastructure crash 或 model reload。
- [ ] 检查所有 TTS failure 是否都有明确 trace reason。
- [ ] 检查是否存在系统性 echo 或 hallucination。

#### T7.8 Qualification report

- [ ] 汇总软件版本、branch 和 commit。
- [ ] 汇总 microphone、STT model、TTS engine 和 LLM model 配置。
- [ ] 汇总五局的 valid / invalid / deadline counts。
- [ ] 汇总 recognition accuracy。
- [ ] 汇总 latency breakdown。
- [ ] 列出所有 failure categories 和代表性 trace。
- [ ] 判断每条 acceptance criterion 是 pass、fail 还是 blocked。
- [ ] 把非阻塞问题登记为 follow-up，而不是静默忽略。

#### Phase 7 退出条件

- [ ] 至少一局完整双向语音游戏成功完成。
- [ ] 5 局 qualification 数据已分析。
- [ ] Acceptance criteria 有明确结论。
- [ ] 包含 5 局实机数据的 qualification detailed report 已写入 `developing-logs/reports/`。

### 最终交付检查

- [x] `animal_naming` 同时支持 `run` 和 `play`。
- [x] `play` 同时支持 terminal 与 voice Human input。
- [x] LLM answer 同时支持 silent 与 audio output。
- [x] Human 和 LLM 共享 whitelist 与 no-repeat referee。
- [x] Domain、speech、runtime、CLI 和 trace 的责任边界清楚。
- [x] Animal Naming focused tests 与仓库可用 static checks 通过；全量 suite 的无关失败已记录。
- [x] 文档命令与当前 CLI `--help` 一致。
- [ ] 真实设备 qualification 已完成并形成报告。

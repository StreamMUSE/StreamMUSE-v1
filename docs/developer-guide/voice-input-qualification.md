---
title: 语音输入离线资格验证
description: 使用冻结的本地语料库验证 faster-whisper 与 Zip-Zap-Zop 规范化链路
---

# 语音输入离线资格验证

`scripts/qualify_voice_input.py` 重放经过同意采集的本地语音文件。它不打开麦克风、
不下载模型（除非显式传入 `--allow-model-download`）、不复制原始音频，也不会把
`speaker` 或 `session` 写入逐样本结果。

## Manifest

输入可以是 JSONL，也可以是 JSON 数组，或包含 `samples`/`entries` 数组的 JSON 对象。
每条记录必须包含：

```json
{
  "audio_path": "audio/acceptance-0001.wav",
  "expected": "ZipZap",
  "split": "acceptance",
  "category": "combination",
  "speaker": "consented-speaker-id",
  "session": "session-id",
  "distance": "near",
  "environment": "quiet_room"
}
```

- `audio_path` 必须是本地路径；相对路径以 manifest 所在目录为基准。PCM WAV 由脚本
  直接解码，其他本地音频格式需要 `voice` 可选依赖组。同一路径或相同 SHA-256 内容
  不能出现两次，包括跨 `split` 复用；因此复制记录不能增加说话人或类别计数。
- `expected` 是可由 `ZipZapZopTask.parse_spoken_response()` 解析的答案。负样本使用
  `null` 或空字符串。
- `split` 是数据划分名称。默认仅对 `acceptance` 划分执行门槛判断，可通过
  `--acceptance-split` 修改。
- 正样本的 `category` 可省略，工具会从规范答案推断 `zip`、`zap`、`zop`、
  `combination` 或 `number`。若提供，必须与推断值完全一致。
- 负样本必须将 `category` 明确设为 `silence`、`noise`、`playback` 或
  `non_command` 之一。工具拒绝其他字符串，也不会把 manifest 中的任意类别原样写入
  结果，因此不能借该字段持久化说话人身份。
- `speaker`、`session`、`distance` 和 `environment` 可在探索运行中省略，但正式验收的
  所有记录都必须非空。语料必须包含非空的 `dev` 划分，且 `dev` 与验收划分的说话人和
  session 集合必须完全分离。
- `distance` 和 `environment` 只能使用小写字母/数字开头、最长 32 字符的安全类别 token，
  后续字符仅允许小写字母、数字、`_` 或 `-`。逐样本结果不写这些字段；摘要只保存
  去身份化的类别聚合计数。

调优前应冻结 manifest 和音频。脚本在摘要中记录 manifest、每段音频、已解析模型路径
和版本、`uv.lock` 路径与 SHA-256；逐样本输出使用序号与音频哈希，不复制身份字段、
音频路径或任意用户自定义类别。

内容哈希去重只能发现逐字节相同的文件。语料整理时仍应检查由重新编码、加静音或轻微
扰动得到的同源录音，避免它们跨说话人、类别或数据划分重复计数。

工具在运行开始时还记录资格脚本、`ZipZapZopTask` 解析器和实际 recognizer 源文件的
SHA-256。manifest 解析后、每段音频转录前后、生成摘要前以及发布证据前，工具会复核
源码、manifest、音频和 `uv.lock`；任一内容变化都会使资格运行失败。

## 运行

`--output-dir` 必须是不存在的新目录，并位于冻结语料目录之外。最终目录及同父 staging
目录都不能等于、包含或位于 manifest、`uv.lock`、任一音频或源码路径之下。识别器启动后，
解析出的模型目录也加入碰撞检查。已有输出不会被覆盖。

先确保 `tiny.en` 模型已经存在于指定缓存。无提示基线仅用于探索比较：

```bash
uv run --frozen --extra voice python scripts/qualify_voice_input.py \
  --manifest <corpus>/manifest.jsonl \
  --output-dir <results>/baseline \
  --model tiny.en \
  --device cpu \
  --compute-type int8 \
  --model-cache <model-cache> \
  --model-revision <frozen-revision>
```

任务静态上下文和自定义配置必须写入不同的新输出目录，便于比较冻结结果。只有
`--context-profile task` 与游戏生产路径使用相同的静态提示词和热词，才可能通过正式 gate：

```bash
# Zip/Zap/Zop 任务自带的静态提示词与热词
uv run --frozen --extra voice python scripts/qualify_voice_input.py \
  --manifest <corpus>/manifest.jsonl \
  --output-dir <results>/task-context \
  --model-cache <model-cache> \
  --model-revision <frozen-revision> \
  --context-profile task

# 单独评估候选提示词/热词
uv run --frozen --extra voice python scripts/qualify_voice_input.py \
  --manifest <corpus>/manifest.jsonl \
  --output-dir <results>/custom \
  --model-cache <model-cache> \
  --model-revision <frozen-revision> \
  --context-profile custom \
  --initial-prompt "Valid game words are Zip, Zap, and Zop." \
  --hotword Zip --hotword Zap --hotword Zop
```

只有首次有意填充缓存时才使用 `--allow-model-download`。正式资格运行应恢复默认的
`--local-files-only`。对于 Hugging Face `snapshots/<commit>` 目录，`commit` 必须是严格的
40 位小写十六进制提交号，识别器解析版本必须与目录一致，且 `--model-revision` 必须使用
同一个提交号；`main`、tag 或任意目录名都不算 pin。启动后工具为 HF snapshot 和普通本地
模型都建立 `model.bin`、`config.json` 等关键文件树内容哈希基线，并在转录后和发布前复核。
普通本地目录可产生可复现探索证据，但正式 gate 还要求配置恰为 `tiny.en`、CPU、int8、
`local_files_only=true`，并绑定 `Systran/faster-whisper-tiny.en` 仓库和上述 commit snapshot。

## 输出和门槛

成功完成的运行生成一个不可覆盖的 artifact-set 目录。工具先在同一父目录的 staging 目录中
生成两份文件，校验后再原子重命名整个目录；失败会清理 staging，因此不会出现新
`samples.jsonl` 搭配旧 `summary.json`：

- `samples.jsonl`：原始 ASR、规范化结果、解析状态、逐样本 ASR 延迟和音频哈希；
- `summary.json`：总体、按 `split`、按 `category`、按 `split/category` 的准确率、
  Wilson 95% 置信区间、混淆计数、ASR p50/p95、源码、模型溯源和依赖锁证据；其中
  `artifact_set.id` 绑定 artifact set，`artifact_set.samples.sha256` 绑定同目录逐样本文件；
- 非零退出码：输入无效返回 2，门槛未通过返回 1，通过离线语料库门槛返回 0。

默认门槛对应语音功能计划：验收集至少 5 名说话人，Zip/Zap/Zop 各 100 条、组合词
100 条、数字 150 条和负样本 200 条。负样本总数之外，`silence`、`noise`、`playback`
和 `non_command` 分别有独立计数门槛，默认每类至少 50 条，总计至少 200 条；可用
`--min-negative-category-samples` 显式调整冻结协议的每类下限。总体、各基础词、组合词与数字
合计的规范化精确率至少 95%；负样本总体以及四个负样本类别各自的错误命令率都不超过
1%，因此某一类别的失败不能被其他类别的低错误率掩盖。正样本空转录率
`positive_empty_transcript_rate` 不超过 5%（可用
`--max-positive-empty-transcript-rate` 调整），热启动 ASR p95 不超过 300 ms。这里的空转录
仅表示识别器对已经提供的音频返回空文本，不表示麦克风或 VAD 将正样本判定为无语音。

生产识别器的 transcript quality gate 同样参与重放：超长、连续 token 重复或异常高
compression ratio 的输出会保留原始文本和诊断信息，但 `predicted_canonical` 固定为 null，
并以 `transcript_quality_rejected=true` 记录，因此不能通过 parser 偶然变成正确命令。
overall、split 和 category summaries 还会分别汇总 rejection count/rate、正负样本数量及
reason histogram。它没有独立的 formal pass/fail 门槛：正样本拒绝由 canonical accuracy
门槛约束，负样本拒绝是安全结果，但仍通过该聚合指标暴露模型退化频率。

这些默认值共同构成 `formal` threshold profile。任何 CLI 门槛覆盖都会将运行标记为
`exploratory`，`formal_threshold_profile` gate 失败，因而 `passed` 永远不会变为 true；降低
样本数或放宽准确率只能用于开发诊断。正式 profile 同时要求 `task` context 和上述固定的
模型、设备、计算类型、离线模式及 HF 仓库/commit 证据。

通过还要求识别器记录可验证的模型证据以及当前仓库 `uv.lock` 的 SHA-256。工具解析锁文件
中适用于当前 Python/操作系统的版本，并逐项对比当前安装的 `faster-whisper`、
`ctranslate2`、`av`、`onnxruntime`、`tokenizers`、`huggingface-hub`、`sounddevice`、
`webrtcvad-wheels`、`numpy` 和 `scipy`；缺包、版本不唯一或任何版本漂移都会使 gate 失败。
摘要中的 `checks` 保存所有准确率、覆盖率和可复现性门槛的实际值、比较运算与阈值；
`environment.voice_dependencies` 和 `reproducibility` 保存对应证据。

`passed` 只表示 `offline_asr_corpus` 范围通过。预录音频无法验证端点造成的起始截断、
采集队列溢出、包含端点延迟的最后语音帧到文本延迟、麦克风/VAD 的正样本
false-no-speech rate，以及脚本化麦克风回合的截止时间成功率；这些项目会列在
`acceptance.not_measured` 中，完整功能资格状态始终明确为 false，必须由麦克风/端点测试
和目标 Mac 上的硬件验证补齐。

## 2026-07-16 实现验证记录

下面的记录证明生产接线、固定模型加载和自动化契约可运行，不代表正式语料库或物理麦克风
验收已经通过。

| 项目 | 实测值 |
|---|---|
| 系统 | macOS 15.1（24B83），arm64，Apple M1 Pro 10 核，16 GB |
| Python | 3.10.18 |
| 模型 | `tiny.en`，CPU，int8，`local_files_only=true` |
| 仓库/快照 | `Systran/faster-whisper-tiny.en` / `0d3d19a32d3338f10357c0889762bd8d64bbdeba` |
| 缓存来源 | 本机 `/private/tmp/streammuse-whisper-cache` 中的已有 snapshot |
| `uv.lock` SHA-256 | `08155ed0a91102efd9506fb6031c42f2772c417f53581bd9f333a490f326c2e4` |

锁定环境中的语音相关版本为：`faster-whisper 1.2.1`、`ctranslate2 4.8.1`、
`av 17.1.0`、`onnxruntime 1.23.2`、`tokenizers 0.21.1`、
`huggingface-hub 0.33.0`、`sounddevice 0.5.5`、`webrtcvad-wheels 2.0.14`、
`numpy 2.1.3` 和 `scipy 1.15.3`。

一次缓存已存在的启动样本记录到模型解析 1309.0 ms、模型加载 180.4 ms、预热
180.7 ms。该单次烟雾测试只用于确认固定 snapshot 能加载、预热并完成转录调用，不能替代
语料库 p50/p95 或准确率结果。

最终自动化结果：语音聚焦套件 `214 passed, 2 skipped`；指定上述模型和 commit 后，
真实模型集成测试 `1 passed, 1 skipped`，其中跳过项需要经过同意采集的预录语音。
`uv lock --check`、`compileall` 和 `git diff --check` 均通过。

全仓单元测试为 `582 passed, 14 failed`，14 个失败均位于本功能未修改的 melody robustness /
perturbation campaign 路径。全仓集成测试为 `4 passed, 1 failed, 2 skipped`，唯一失败位于未修改的
Lekai session contract。当前执行环境的 `voice-devices` 未发现输入设备，也没有正式同意语料库，
因此真实麦克风 20 回合、混淆矩阵、端点/队列指标、最后语音帧到文本 p95 和脚本化截止时间
成功率仍是外部验收项，不能据此宣称第 10 节产品门槛已经通过。

---
title: 交互式游戏语音输入
description: 在 Zip-Zap-Zop 游戏中使用 faster-whisper tiny.en 进行本地语音输入
---

# 交互式游戏语音输入

`streammuse-task play` 可以在轮到人类玩家时打开麦克风，将一段发言转成文本，再交给现有游戏裁判判定。语音输入当前只支持 `zip_zap_zop`，使用常驻的 `faster-whisper` `tiny.en` 模型，默认在 CPU 上以 `int8` 运行。

语音层不判断答案是否正确，也不会根据当前正确答案修改识别结果。它返回原始转录文本，`ZipZapZopTask` 只执行确定性的标点、空白、大小写、组合词和英语数字规范化，然后调用原有裁判逻辑。

识别器会把明显退化的模型输出标记为 `rejected_transcript`，而不是把它送给游戏 parser。当前防护覆盖超过 512 字符、同一 token 连续重复至少 16 次、任一 Whisper segment 的 compression ratio 超过 10，或 compression ratio 是非有限/不可解析值的输出。原始转录仍保留在 `metadata.human_input.raw_transcript`，触发原因和测量值记录在 ASR quality-gate metadata 中，便于排查而不会静默改写玩家说过的内容。

口语整数语法的范围为 `-999999999` 到 `999999999`。语音模式会在启动前检查本局从 `--start-number` 到 `--max-turns` 覆盖的范围；超出时直接拒绝配置，不会开始麦克风采集。

## 安装

语音依赖是可选的；普通终端输入不需要安装这些包。

```bash
uv sync --extra voice
```

该依赖组包含 `faster-whisper`、`sounddevice` 和 `webrtcvad-wheels`。SciPy 重采样器由主项目依赖提供。

## 检查麦克风

设备枚举不会加载 Whisper 模型，也不需要 `--task` 或模型服务器：

```bash
uv run --extra voice streammuse-task voice-devices
```

使用输出中的索引或完整设备名称选择麦克风：

```bash
--microphone-device 2
# 或
--microphone-device "Built-in Microphone"
```

macOS 首次访问麦克风时可能要求授权。若授权被拒绝，请在“系统设置 → 隐私与安全性 → 麦克风”中允许启动终端或 Python 的应用。Linux 需要可用的 PortAudio 输入设备；容器或远程会话通常不会自动暴露主机麦克风。

## 开始游戏

先启动 OpenAI 兼容的本地聊天模型服务，然后运行：

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

模型会在游戏开始前加载并预热一次。麦克风只在人类回合打开，每个回合检测到尾部静音后关闭；模型在整局游戏中保持常驻。

建议先使用 `soft` 和至少 3000 ms 的截止时间。等待开口、发言、端点静音和 ASR 都计入人类回合耗时。`hard`/`challenge` 会在采集阶段达到截止时间时停止录音，但进程内的 CTranslate2 转录不能安全取消，因此最终返回时间可能超过名义截止时间。

## 模型缓存和离线运行

默认情况下，首次启动可能在游戏开始前从 Hugging Face 下载模型。可以显式指定缓存和模型版本：

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --voice-model tiny.en \
  --voice-model-cache /path/to/model-cache \
  --voice-model-revision <commit-or-tag> \
  --voice-local-files-only \
  --deadline-mode soft
```

`--voice-local-files-only` 禁止下载；缓存中没有请求的模型或版本时，游戏会在创建任何玩家回合之前失败。也可以把 `--voice-model` 直接设置为本地 faster-whisper 模型目录。

## 记录和隐私

默认不会保存麦克风音频。运行目录仍会记录：

- 原始 ASR 转录和规范化后的游戏响应；
- 等待语音、发言、端点静音和 ASR 延迟；
- 麦克风设备、采样率、模型配置和端点原因；
- 音频溢出、无语音和空转录状态。

只有显式添加 `--voice-save-audio` 时，才会把每个有界发言保存到运行目录的 `artifacts/turn/`。WAV 在采集和 ASR 计时结束后写入，文件 I/O 不参与回合截止时间计分。这些 WAV 文件可能包含敏感语音，不应默认提交或共享。

冻结语料库的准确率、混淆和延迟验收流程见[语音输入离线资格验证](../developer-guide/voice-input-qualification.md)。

## 故障诊断

| 现象 | 处理方式 |
|---|---|
| 提示缺少语音依赖 | 运行 `uv sync --extra voice` |
| 没有输入设备 | 运行 `voice-devices`，检查系统权限和 PortAudio 设备 |
| 设备只接受 44.1 kHz | 选择支持 8/16/32/48 kHz 的输入配置；首版不在 VAD 前做流式 44.1 kHz 重采样 |
| 离线模型未命中 | 去掉 `--voice-local-files-only` 完成一次预下载，或修正缓存/本地模型路径 |
| Zip/Zap 混淆 | 查看 `response_trace.jsonl` 中的原始转录；系统不会用预期答案自动纠错 |
| 一秒挑战经常超时 | 先使用 3000 ms `soft` 模式，依据目标设备的 p95 指标再缩短时间 |
| 采集期间需要退出 | 按 `Ctrl-C`；语音模式采集中不并发处理 `:quit` |

LLM 回答的可选 TTS 播放见[交互式游戏语音输出](speech-output.md)。当前仍不包含声学回声消除。

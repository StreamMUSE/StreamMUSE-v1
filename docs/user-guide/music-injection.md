---
title: 音乐注入（music injection）
description: 使用预置 MIDI 文件为模型提供历史上下文，改善冷启动时的伴奏质量
---

# 音乐注入

在会话开始时，将一段预录制的旋律/伴奏注入模型历史，使模型不需要从零开始"热身"，从第一个音符起就能生成高质量的伴奏。

---

## 为什么需要注入？

RoFormer 模型需要足够的历史上下文（最多 96 帧）才能准确预测下一段伴奏。在会话开始时历史为空，最初几秒的伴奏质量通常较差。音乐注入通过预先填充历史记录解决这一问题。

---

## 基本用法

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --injection-file prompts/C_major/pop909_216_mel.mid \
    --injection-length 50
```

| 参数 | 说明 |
|---|---|
| `--injection-file` | 旋律 MIDI 文件路径（用于填充历史） |
| `--injection-length` | 注入长度（ticks）；`None` 表示使用全部文件 |

---

## 预置 prompts 目录

`prompts/` 目录按调性组织了来自 POP909 数据集的旋律和伴奏文件：

| 目录 | 调性 | 文件数 |
|---|---|---|
| `prompts/A_major/` | A 大调 | 4 首（mel + acc） |
| `prompts/A_minor/` | A 小调 | 6 首 |
| `prompts/B_minor/` | B 小调 | 10 首 |
| `prompts/C_major/` | C 大调 | 6 首 |
| `prompts/D_major/` | D 大调 | 6 首 |
| `prompts/D_minor/` | D 小调 | 2 首 |
| `prompts/E_minor/` | E 小调 | 4 首 |
| `prompts/F_major/` | F 大调 | 6 首 |
| `prompts/G_major/` | G 大调 | 10 首 |
| `prompts/manual/` | 手工编写 | 若干 |

每个目录下 `*_mel.mid` 为旋律文件，`*_acc.mid` 为对应的伴奏文件。

---

## 使用建议

**调性匹配**：建议选择与演奏调性相符的注入文件。例如，若计划演奏 C 大调，使用 `prompts/C_major/` 下的文件：

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --injection-file prompts/C_major/pop909_216_mel.mid \
    --injection-length 50
```

**注入长度**：
- 较短（20–30 ticks）：快速注入，历史较少但启动更快
- 较长（50–80 ticks）：充分填充历史，伴奏效果更好，但启动时间略长

**与 `--max-ticks` 配合**：对于基准测试，固定运行时长：

```bash
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file prompts/C_major/pop909_216_mel.mid \
    --injection-file prompts/C_major/pop909_216_mel.mid \
    --injection-length 50 \
    --max-ticks 200 \
    --output-type composite \
    --log-dir logs/injected_test
```

---

## 内部实现

CLI 在 `service.start()` 之前调用 `inference_engine.inject_history()`：

```python
# cli.py main() 中
if args.injection_file:
    melody_events = load_midi_as_events(args.injection_file, args.injection_length)
    acc_events = load_midi_as_events(args.injection_acc_file, args.injection_length)
    inference_engine.inject_history(melody_events, acc_events, args.injection_length)
```

`inject_history()` 对 HTTP 客户端是一个 POST 请求到 `/inject_notes`，对本地 Stanley 引擎则直接填充内存历史列表。

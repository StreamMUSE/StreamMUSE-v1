# 分支总结：`feature/voice` 与 `feature/web-host-bind-option`

日期：2026-07-16

两个分支共享同一个合并基线 `e44df2c1 merge`，且彼此独立、没有文件重叠。

---

## `feature/web-host-bind-option`

### 目的

为 web viewer 增加可由用户配置的 host 和 port 绑定参数，使 StreamMUSE 的 web UI 可以监听非默认接口（例如用 `0.0.0.0` 允许局域网访问）或自定义端口。

### 改动

- 修改文件：`src/streammuse/presentation/cli/config_parser.py`
- 在主解析器里新增两个 CLI 参数：
  - `--web-host`（字符串，默认 `"127.0.0.1"`）
  - `--web-port`（整数，默认 `8001`）

```python
parser.add_argument("--web-host", type=str, default="127.0.0.1",
    help="Host/interface for the web viewer to bind (use 0.0.0.0 to allow LAN access)")
parser.add_argument("--web-port", type=int, default=8001,
    help="Port for the web viewer")
```

### 范围与风险

- 改动非常小，风险很低。
- 目前只影响参数解析；实际 web viewer 需要在别处消费这两个值（或后续再接线）。
- 分支内没有包含测试、文档或运行时逻辑改动。

### 后续工作

解析后的 `args.web_host` 和 `args.web_port` 需要传给 web viewer server 的启动调用。集成时可以搜索代码中硬编码的 `127.0.0.1:8001` 或 `8001`，把参数替换进去。

---

## `feature/voice`

### 目的

引入一个独立的语音微基准测试脚本，在 StreamMUSE runtime 之外测量短句语音识别（STT）和语音合成（TTS）后端的延迟与准确率。这是未来集成语音功能的前瞻性探索工作。

### 关键文件

| 文件 | 说明 |
|---|---|
| `scripts/voice_microbench.py` | 主基准脚本（约 812 行）。对 10 个固定的 1-3 词短语跑 STT/TTS 后端，输出延迟、内存和准确率。 |
| `tests/unit/scripts/test_voice_microbench.py` | 11 个单元测试，覆盖统计/渲染辅助函数和 batch-code builder。 |
| `docs/developer-guide/voice-microbenchmark-results.md` | Mac 和 H200 上的 benchmark 结果报告。 |
| `docs/developer-guide/assets/voice_*.png` | 报告引用的图表。 |
| `.gitignore` | 忽略 `voice-bench-env*/`、`voice_bench_models/`、`voice_bench_runs/`。 |

### 支持的后端

- **STT**：`whisper_cpp`、`faster_whisper`、`sherpa_onnx`、`vosk`
- **TTS**：`system_tts`（macOS `say`）、`espeak-ng`、`piper`、`kokoro`、`sherpa_onnx`

### 测量指标

- `setup_ms`：模型/客户端初始化时间
- `first_run_ms`：初始化后第一次推理（包含懒加载 warmup）
- `steady_mean_ms`：第 2-10 个样本的平均延迟
- `all-run_mean_ms`：全部 10 个样本的平均延迟
- `peak_rss_mb`：峰值常驻内存
- `gpu_peak_mb`：峰值 GPU 显存（如可获取）
- STT 准确率：通过精确归一化匹配和 Levenshtein 编辑距离评估

### 运行方式

```bash
uv run python scripts/voice_microbench.py \
  --device my-machine \
  --kinds stt,tts \
  --stt-backends faster_whisper,vosk \
  --tts-backends piper,kokoro \
  --output-dir voice_bench_runs
```

- 如果未通过 `--samples-dir` 提供样本，会自动生成 STT 用的 WAV 样本。
- 每次运行的产物写入 `voice_bench_runs/voice_microbench_YYYYMMDD-HHMMSS/`。
- 在 stdout 打印 markdown 表格。

### 范围与风险

- 自包含：不触碰音乐推理、实时循环或 prompt-continuation 代码。
- 新增脚本和 `tests/unit/scripts/` 下的测试。
- 基准测试有意与 StreamMUSE 解耦，只使用本地工作区存放输出和样本。
- 集成时可能的摩擦：脚本可能依赖独立的隔离虚拟环境和特定系统包（如 `espeak-ng`、whisper.cpp 二进制等）。benchmark 结果文档已记录 Mac/H200 上哪些配置可用。

### 集成注意事项

- 如果合并到已有 CLI 改动的分支，新脚本不会与 `streammuse-task` 或 `streammuse-cli` 冲突。
- 单元测试通过 `importlib.util` 动态导入 `scripts/voice_microbench.py`，因此脚本必须保持 import-safe（没有顶层执行代码），否则测试会失败。

---

## 与当前分支的关系

当前工作分支是 `new_system_stanley`，是 StreamMUSE 的主开发分支，包含音乐实时推理、two-stage prompt+continuation 以及 task runtime 等最新改动。`feature/voice` 和 `feature/web-host-bind-option` 都不与这些文件重叠，因此可以独立合并，冲突风险很低。

| 分支 | 与当前分支的文件重叠 | 合并风险 |
|---|---|---|
| `feature/web-host-bind-option` | 无 | 极低 |
| `feature/voice` | 无 | 低 |

---

## 建议合并顺序

1. 先合并 `feature/web-host-bind-option`——只有两行的 trivial 改动。
2. 再合并 `feature/voice`——改动较大但自包含；合并后跑 `uv run pytest tests/unit/scripts/test_voice_microbench.py -q` 确认测试通过。

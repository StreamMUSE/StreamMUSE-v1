---
title: 安装
description: StreamMUSE 环境搭建与依赖安装指南
---

# 安装

## 前置条件

- Python 3.10 或更高版本
- [uv](https://github.com/astral-sh/uv) 包管理器（推荐）
- Git

## 步骤一：克隆仓库

```bash
git clone <repository-url>
cd StreamMUSE-new-sys
```

## 步骤二：安装主包依赖

使用 `uv sync` 安装 `pyproject.toml` 中声明的所有依赖并创建虚拟环境：

```bash
uv sync
```

## 步骤三：安装改版 Transformers

StreamMUSE 使用了经过修改的 HuggingFace Transformers，以支持 RoFormer 的位置编码。需要单独以 editable 模式安装：

```bash
cd transformers
pip install -e .
cd ..
```

## 步骤四：验证安装

```bash
uv run streammuse-cli --help
```

若安装成功，将看到完整的参数帮助信息，以 `usage: streammuse-cli` 开头。

---

## 可选依赖

### 交互式游戏语音输入

Zip-Zap-Zop 的本地语音输入使用独立的 `voice` 可选依赖组：

```bash
uv sync --extra voice
uv run --extra voice streammuse-task voice-devices
```

安装后参见[交互式游戏语音输入](../user-guide/voice-input.md)，了解模型缓存、麦克风权限、隐私和离线运行。

### 实时音频输出（`--output-type audio`）

实时音频播放依赖 `python-rtmidi` 提供的虚拟 MIDI 端口，以及系统 MIDI 合成器（如 macOS 自带的 FluidSynth 或 DLSMusicDevice）。

```bash
# macOS 通常无需额外安装；Linux 可能需要：
sudo apt-get install fluidsynth
```

### GPU 推理（Stanley 本地模式，`--inference-type stanley`）

本地模式需要 CUDA 12.8+ 及对应版本的 PyTorch。若仅使用 HTTP 模式（对接推理服务器），则不需要 GPU。

### macOS Apple Silicon（Lekai 本地推理）

在 M1/M2/M3 上运行 Lekai 本地服务时，建议：

1. 使用项目虚拟环境中的 PyTorch（`uv sync` 后自动安装）。
2. 启动前确认 MPS 可用：

```bash
uv run python -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

3. 如果遇到 MPS 不支持的算子，可设置 `LEKAI_ENABLE_MPS_FALLBACK=true`，让服务自动回退到 CPU。
4. 首次验证建议用较短生成长度（如 `generation-length-frames=8~16`），降低内存压力。

常见问题：

1. `ModuleNotFoundError`：请确保从仓库根目录执行命令，或使用 `uv run ...`。
2. `state_dict` key mismatch：确认 checkpoint 与当前模型结构匹配，优先使用 `.safetensors`。
3. 推理延迟过高：先降低 `generation-length-frames`，再提高 `generation-interval-ticks`。

---

## 下一步

- [快速上手](quickstart.md)：启动 fake 服务器，运行第一个示例
- [配置项](configuration.md)：了解所有可用的参数

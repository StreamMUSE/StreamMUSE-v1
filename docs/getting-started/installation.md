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

### 实时音频输出（`--output-type audio`）

实时音频播放依赖 `python-rtmidi` 提供的虚拟 MIDI 端口，以及系统 MIDI 合成器（如 macOS 自带的 FluidSynth 或 DLSMusicDevice）。

```bash
# macOS 通常无需额外安装；Linux 可能需要：
sudo apt-get install fluidsynth
```

### GPU 推理（Stanley 本地模式，`--inference-type stanley`）

本地模式需要 CUDA 12.8+ 及对应版本的 PyTorch。若仅使用 HTTP 模式（对接推理服务器），则不需要 GPU。

---

## 下一步

- [快速上手](quickstart.md)：启动 fake 服务器，运行第一个示例
- [配置项](configuration.md)：了解所有可用的参数

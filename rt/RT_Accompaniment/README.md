# RT_Accompaniment

基于 LLaMA 架构的实时钢琴伴奏生成模型。给定主旋律（Part0），自动生成伴奏声部（Part1）。

## 项目结构

```
RT_Accompaniment/
├── config.py          # 模型和训练配置
├── model.py           # PianoLLaMA 模型定义
├── my_tokenizer.py    # 钢琴卷帘编解码器
├── PianoDataset.py    # 数据集加载与处理
├── Token2Midi.py      # Token 转 MIDI 工具
├── inference.py       # 推理脚本
├── mini_dataset/      # 示例数据集 (51个NPZ文件)
└── checkpoints/       # 模型权重
    └── epoch_4_1104_1204/
        ├── model.safetensors
        └── config.json
```

## 环境依赖

```bash
pip install torch transformers safetensors numpy pretty_midi
```

## 使用方式

### 推理生成

运行 `inference.py` 进行伴奏生成：

```python
from inference import load_model, setup_model_configs_llama
from PianoDataset import PianoDataset
from config import ModelConfig
from Token2Midi import tokens_to_midi

# 1. 加载模型
model_config = ModelConfig()
llama_config = setup_model_configs_llama(model_config)
model = load_model(
    checkpoint_path="checkpoints/epoch_4_1104_1204",
    llama_config=llama_config,
    device="cuda"
)

# 2. 加载数据集
dataset = PianoDataset(
    data_path="mini_dataset",
    model_config=model_config,
    is_train=False
)

# 3. 生成伴奏
result, part1_beats, metadata = model.generate_accompaniment(
    dataset=dataset,
    condition_index=0,           # 选择条件样本索引
    delay_beats=-1,              # Part1 提前1拍 (负数提前，正数延迟)
    gt_prefix_beats=12,          # 使用前12拍的GT伴奏作为引导
    temperature=1.1,
    top_k=10,
    top_p=0.95,
    repetition_penalty=1.0,
    max_new_tokens=2048,
    device="cuda"
)

# 4. 转换为 MIDI
tokens_to_midi(
    part0_beats=metadata["part0_beats"],
    part1_beats=part1_beats,
    bpm=metadata["bpm"],
    output_path="output.mid"
)
```

### 批量生成

使用内置的批量生成函数：

```python
from inference import batch_generate_50_samples

batch_generate_50_samples(
    model=model,
    dataset=dataset,
    output_dir="./generated_outputs",
    delay_beats=-1,
    gt_prefix_beats=12
)
```

### 主要参数说明

| 参数 | 说明 |
|------|------|
| `delay_beats` | Part1 相对 Part0 的延迟拍数。负数表示 Part1 提前，正数表示延迟 |
| `gt_prefix_beats` | 使用多少拍的 Ground Truth 伴奏作为生成引导 |
| `temperature` | 采样温度，越高越随机 |
| `top_k` | Top-K 采样参数 |
| `top_p` | Nucleus 采样参数 |
| `repetition_penalty` | 重复惩罚系数 |

## 数据格式

NPZ 文件包含：
- `metadata`: 包含 `time_signature_idx`, `bpm`, `num_measures`
- `measure_0`, `measure_1`, ...: 形状为 `(4, 88, t)` 的钢琴卷帘
  - 通道 0-1: Part0 (主旋律) 的 sustain 和 onset
  - 通道 2-3: Part1 (伴奏) 的 sustain 和 onset

## 模型架构

- 基于 LLaMA 的因果语言模型
- 隐层维度: 768
- 注意力头: 6
- 层数: 18
- 词表大小: 268

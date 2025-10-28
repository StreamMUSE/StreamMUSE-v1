<!--
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 导出为 TorchScript

<Tip>

这是开始使用 TorchScript 进行实验的起点，我们仍在探索其在变量输入大小模型中的能力。
这是我们关注的焦点，我们将在即将发布的版本中深入分析，提供更多的代码示例、更灵活的实现以及比较
Python 代码与编译 TorchScript 的性能基准。

</Tip>

根据 [TorchScript 文档](https://pytorch.org/docs/stable/jit.html)：

> TorchScript 是从 PyTorch 代码创建可序列化和可优化的模型的一种方式。

有两个 PyTorch 模块：[JIT 和 TRACE](https://pytorch.org/docs/stable/jit.html)。
这两个模块允许开发人员将其模型导出到其他程序中重用，比如面向效率的 C++ 程序。

我们提供了一个接口，允许您将 🤗 Transformers 模型导出为 TorchScript，
以便在与基于 PyTorch 的 Python 程序不同的环境中重用。
本文解释如何使用 TorchScript 导出并使用我们的模型。

导出模型需要两个步骤：

- 使用 `torchscript` 参数实例化模型
- 使用虚拟输入进行前向传递

这些必要条件意味着开发人员应该注意以下详细信息。

## TorchScript 参数和绑定权重

`torchscript` 参数是必需的，因为大多数 🤗 Transformers 语言模型的 `Embedding` 层和
`Decoding` 层之间有绑定权重。TorchScript 不允许导出具有绑定权重的模型，因此必须事先解绑和克隆权重。

使用 `torchscript` 参数实例化的模型将其 `Embedding` 层和 `Decoding` 层分开，
这意味着它们不应该在后续进行训练。训练将导致这两层不同步，产生意外结果。

对于没有语言模型头部的模型，情况不同，因为这些模型没有绑定权重。
这些模型可以安全地导出而无需 `torchscript` 参数。

## 虚拟输入和标准长度

虚拟输入用于模型的前向传递。当输入的值传播到各层时，PyTorch 会跟踪在每个张量上执行的不同操作。
然后使用记录的操作来创建模型的 *trace* 。

跟踪是相对于输入的维度创建的。因此，它受到虚拟输入的维度限制，对于任何其他序列长度或批量大小都不起作用。
当尝试使用不同大小时，会引发以下错误：

```text
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```

我们建议使用至少与推断期间将馈送到模型的最大输入一样大的虚拟输入大小进行跟踪。
填充可以帮助填补缺失的值。然而，由于模型是使用更大的输入大小进行跟踪的，矩阵的维度也会很大，导致更多的计算。

在每个输入上执行的操作总数要仔细考虑，并在导出不同序列长度模型时密切关注性能。

## 在 Python 中使用 TorchScript

本节演示了如何保存和加载模型以及如何使用 trace 进行推断。

### 保存模型

要使用 TorchScript 导出 `BertModel`，请从 `BertConfig` 类实例化 `BertModel`，
然后将其保存到名为 `traced_bert.pt` 的磁盘文件中：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# 对输入文本分词
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# 屏蔽一个输入 token
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 创建虚拟输入
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# 使用 torchscript 参数初始化模型
# 即使此模型没有 LM Head，也将参数设置为 True。
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# 实例化模型
model = BertModel(config)

# 模型需要处于评估模式
model.eval()

# 如果您使用 *from_pretrained* 实例化模型，还可以轻松设置 TorchScript 参数
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)

# 创建 trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### 加载模型

现在，您可以从磁盘加载先前保存的 `BertModel`、`traced_bert.pt`，并在先前初始化的 `dummy_input` 上使用：

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### 使用 trace 模型进行推断

通过使用其 `__call__` dunder 方法使用 trace 模型进行推断：

```python
traced_model(tokens_tensor, segments_tensors)
```

## 使用 Neuron SDK 将 Hugging Face TorchScript 模型部署到 AWS

AWS 引入了用于云端低成本、高性能机器学习推理的
[Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) 实例系列。
Inf1 实例由 AWS Inferentia 芯片提供支持，这是一款专为深度学习推理工作负载而构建的定制硬件加速器。
[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) 是
Inferentia 的 SDK，支持对 transformers 模型进行跟踪和优化，以便在 Inf1 上部署。Neuron SDK 提供：

1. 简单易用的 API，只需更改一行代码即可为云端推理跟踪和优化 TorchScript 模型。
2. 针对[改进的性能成本](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/)的即插即用性能优化。
3. 支持使用 [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html)
   或 [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html)
   构建的 Hugging Face transformers 模型。

### 影响

基于 [BERT（来自 Transformers 的双向编码器表示）](https://huggingface.co/docs/transformers/main/model_doc/bert)架构的
transformers 模型，或其变体，如 [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert)
和 [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta) 在 Inf1 上运行最佳，
可用于生成抽取式问答、序列分类和标记分类等任务。然而，文本生成任务仍可以适应在 Inf1 上运行，
如这篇 [AWS Neuron MarianMT 教程](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html)所述。
有关可以直接在 Inferentia 上转换的模型的更多信息，请参阅 Neuron 文档的[模型架构适配](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia)章节。

### 依赖关系

使用 AWS Neuron 将模型转换为模型需要一个
[Neuron SDK 环境](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide)，
它已经预先配置在 [AWS 深度学习 AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html)上。

### 将模型转换为 AWS Neuron

使用与 [Python 中使用 TorchScript](torchscript#using-torchscript-in-python) 相同的代码来跟踪
`BertModel` 以将模型转换为 AWS NEURON。导入 `torch.neuron` 框架扩展以通过 Python API 访问 Neuron SDK 的组件：

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

您只需要修改下面这一行：

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

这样就能使 Neuron SDK 跟踪模型并对其进行优化，以在 Inf1 实例上运行。

要了解有关 AWS Neuron SDK 功能、工具、示例教程和最新更新的更多信息，
请参阅 [AWS NeuronSDK 文档](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)。

<!--
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 如何创建自定义流水线？

在本指南中，我们将演示如何创建一个自定义流水线并分享到 [Hub](https://hf.co/models)，或将其添加到 🤗 Transformers 库中。

首先，你需要决定流水线将能够接受的原始条目。它可以是字符串、原始字节、字典或任何看起来最可能是期望的输入。
尽量保持输入为纯 Python 语言，因为这样可以更容易地实现兼容性（甚至通过 JSON 在其他语言之间）。
这些将是流水线 (`preprocess`) 的 `inputs`。

然后定义 `outputs`。与 `inputs` 相同的策略。越简单越好。这些将是 `postprocess` 方法的输出。

首先继承基类 `Pipeline`，其中包含实现 `preprocess`、`_forward`、`postprocess` 和 `_sanitize_parameters` 所需的 4 个方法。

```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

这种分解的结构旨在为 CPU/GPU 提供相对无缝的支持，同时支持在不同线程上对 CPU 进行预处理/后处理。

`preprocess` 将接受最初定义的输入，并将其转换为可供模型输入的内容。它可能包含更多信息，通常是一个 `Dict`。

`_forward` 是实现细节，不应直接调用。`forward` 是首选的调用方法，因为它包含保障措施，以确保一切都在预期的设备上运作。
如果任何内容与实际模型相关，它应该属于 `_forward` 方法，其他内容应该在 preprocess/postprocess 中。

`postprocess` 方法将接受 `_forward` 的输出，并将其转换为之前确定的最终输出。

`_sanitize_parameters` 存在是为了允许用户在任何时候传递任何参数，无论是在初始化时 `pipeline(...., maybe_arg=4)`
还是在调用时 `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`。

`_sanitize_parameters` 的返回值是将直接传递给 `preprocess`、`_forward` 和 `postprocess` 的 3 个关键字参数字典。
如果调用方没有使用任何额外参数调用，则不要填写任何内容。这样可以保留函数定义中的默认参数，这总是更"自然"的。

在分类任务中，一个经典的例子是在后处理中使用 `top_k` 参数。

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

为了实现这一点，我们将更新我们的 `postprocess` 方法，将默认参数设置为 `5`，
并编辑 `_sanitize_parameters` 方法，以允许这个新参数。

```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

尽量保持简单输入/输出，最好是可 JSON 序列化的，因为这样可以使流水线的使用非常简单，而不需要用户了解新的对象类型。
通常也相对常见地支持许多不同类型的参数以便使用（例如音频文件，可以是文件名、URL 或纯字节）。

## 将其添加到支持的任务列表中

要将你的 `new-task` 注册到支持的任务列表中，你需要将其添加到 `PIPELINE_REGISTRY` 中：

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

如果需要，你可以指定一个默认模型，此时它应该带有一个特定的修订版本（可以是分支名称或提交哈希，这里我们使用了 `"abcdef"`），以及类型：

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## 在 Hub 上分享你的流水线

要在 Hub 上分享你的自定义流水线，你只需要将 `Pipeline` 子类的自定义代码保存在一个 Python 文件中。
例如，假设我们想使用一个自定义流水线进行句对分类，如下所示：

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

这个实现与框架无关，适用于 PyTorch 和 TensorFlow 模型。如果我们将其保存在一个名为
`pair_classification.py` 的文件中，然后我们可以像这样导入并注册它：

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

完成这些步骤后，我们可以将其与预训练模型一起使用。例如，`sgugger/finetuned-bert-mrpc`
已经在 MRPC 数据集上进行了微调，用于将句子对分类为是释义或不是释义。

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

然后，我们可以通过在 `Repository` 中使用 `save_pretrained` 方法将其分享到 Hub 上：

```py
from huggingface_hub import Repository

repo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")
classifier.save_pretrained("test-dynamic-pipeline")
repo.push_to_hub()
```

这将会复制包含你定义的 `PairClassificationPipeline` 的文件到文件夹 `"test-dynamic-pipeline"` 中，
同时保存流水线的模型和分词器，然后将所有内容推送到仓库 `{your_username}/test-dynamic-pipeline` 中。
之后，只要提供选项 `trust_remote_code=True`，任何人都可以使用它：

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## 将流水线添加到 🤗 Transformers

如果你想将你的流水线贡献给 🤗 Transformers，你需要在 `pipelines` 子模块中添加一个新模块，
其中包含你的流水线的代码，然后将其添加到 `pipelines/__init__.py` 中定义的任务列表中。

然后，你需要添加测试。创建一个新文件 `tests/test_pipelines_MY_PIPELINE.py`，其中包含其他测试的示例。

`run_pipeline_test` 函数将非常通用，并在每种可能的架构上运行小型随机模型，如 `model_mapping` 和 `tf_model_mapping` 所定义。

这对于测试未来的兼容性非常重要，这意味着如果有人为 `XXXForQuestionAnswering` 添加了一个新模型，
流水线测试将尝试在其上运行。由于模型是随机的，所以不可能检查实际值，这就是为什么有一个帮助函数 `ANY`，它只是尝试匹配流水线的输出类型。

你还 **需要** 实现 2（最好是 4）个测试。

- `test_small_model_pt`：为这个流水线定义一个小型模型（结果是否合理并不重要），并测试流水线的输出。
  结果应该与 `test_small_model_tf` 的结果相同。
- `test_small_model_tf`：为这个流水线定义一个小型模型（结果是否合理并不重要），并测试流水线的输出。
  结果应该与 `test_small_model_pt` 的结果相同。
- `test_large_model_pt`（可选）：在一个真实的流水线上测试流水线，结果应该是有意义的。
  这些测试速度较慢，应该被如此标记。这里的目标是展示流水线，并确保在未来的发布中没有漂移。
- `test_large_model_tf`（可选）：在一个真实的流水线上测试流水线，结果应该是有意义的。
  这些测试速度较慢，应该被如此标记。这里的目标是展示流水线，并确保在未来的发布中没有漂移。

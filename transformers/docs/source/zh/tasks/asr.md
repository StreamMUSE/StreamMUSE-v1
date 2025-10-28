<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 自动语音识别

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

自动语音识别（ASR）将语音信号转换为文本，将一系列音频输入映射到文本输出。
Siri 和 Alexa 这类虚拟助手使用 ASR 模型来帮助用户日常生活，还有许多其他面向用户的有用应用，如会议实时字幕和会议纪要。

本指南将向您展示如何：

1. 在 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集上对
   [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) 进行微调，以将音频转录为文本。
2. 使用微调后的模型进行推断。

<Tip>

如果您想查看所有与本任务兼容的架构和检查点，最好查看[任务页](https://huggingface.co/tasks/automatic-speech-recognition)。

</Tip>

在开始之前，请确保您已安装所有必要的库：

```bash
pip install transformers datasets evaluate jiwer
```

我们鼓励您登录自己的 Hugging Face 账户，这样您就可以上传并与社区分享您的模型。
出现提示时，输入您的令牌登录：

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 加载 MInDS-14 数据集

首先从🤗 Datasets 库中加载 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14)
数据集的一个较小子集。这将让您有机会先进行实验，确保一切正常，然后再花更多时间在完整数据集上进行训练。

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

使用 [`~Dataset.train_test_split`] 方法将数据集的 `train` 拆分为训练集和测试集：

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

然后看看数据集：

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

虽然数据集包含 `lang_id `和 `english_transcription` 等许多有用的信息，但在本指南中，
您将专注于 `audio` 和 `transcription`。使用 [`~datasets.Dataset.remove_columns`] 方法删除其他列：

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

再看看示例：

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

有 2 个字段：

- `audio`：由语音信号形成的一维 `array`，用于加载和重新采样音频文件。
- `transcription`：目标文本。

## 预处理

下一步是加载一个 Wav2Vec2 处理器来处理音频信号：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 数据集的采样率为 8000kHz（您可以在其[数据集卡片](https://huggingface.co/datasets/PolyAI/minds14)中找到此信息），
这意味着您需要将数据集重新采样为 16000kHz 以使用预训练的 Wav2Vec2 模型：

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

如您在上面的 `transcription` 中所看到的，文本包含大小写字符的混合。
Wav2Vec2 分词器仅训练了大写字符，因此您需要确保文本与分词器的词汇表匹配：

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

现在创建一个预处理函数，该函数应该：

1. 调用 `audio` 列以加载和重新采样音频文件。
2. 从音频文件中提取 `input_values` 并使用处理器对 `transcription` 列执行 tokenizer 操作。

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

要在整个数据集上应用预处理函数，可以使用🤗 Datasets 的 [`~datasets.Dataset.map`] 函数。
您可以通过增加 `num_proc` 参数来加速 `map` 的处理进程数量。
使用 [`~datasets.Dataset.remove_columns`] 方法删除不需要的列：

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers 没有用于 ASR 的数据整理器，因此您需要调整 [`DataCollatorWithPadding`] 来创建一个示例批次。
它还会动态地将您的文本和标签填充到其批次中最长元素的长度（而不是整个数据集），以使它们具有统一的长度。
虽然可以通过在 `tokenizer` 函数中设置 `padding=True` 来填充文本，但动态填充更有效。

与其他数据整理器不同，这个特定的数据整理器需要对 `input_values` 和 `labels `应用不同的填充方法：

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

现在实例化您的 `DataCollatorForCTCWithPadding`：

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## 评估

在训练过程中包含一个指标通常有助于评估模型的性能。
您可以通过🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 库快速加载一个评估方法。
对于这个任务，加载 [word error rate](https://huggingface.co/spaces/evaluate-metric/wer)（WER）指标
（请参阅🤗 Evaluate [快速上手](https://huggingface.co/docs/evaluate/a_quick_tour)以了解如何加载和计算指标）：

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

然后创建一个函数，将您的预测和标签传递给 [`~evaluate.EvaluationModule.compute`] 来计算 WER：

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

您的 `compute_metrics` 函数现在已经准备就绪，当您设置好训练时将返回给此函数。

## 训练

<frameworkcontent>
<pt>
<Tip>

如果您不熟悉使用[`Trainer`]微调模型，请查看这里的基本教程[here](../training#train-with-pytorch-trainer)！

</Tip>

现在您已经准备好开始训练您的模型了！使用 [`AutoModelForCTC`] 加载 Wav2Vec2。
使用 `ctc_loss_reduction` 参数指定要应用的减少方式。通常最好使用平均值而不是默认的求和：

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
)
```

此时，只剩下 3 个步骤：

1. 在 [`TrainingArguments`] 中定义您的训练参数。唯一必需的参数是 `output_dir`，用于指定保存模型的位置。
   您可以通过设置 `push_to_hub=True` 将此模型推送到 Hub（您需要登录到 Hugging Face 才能上传您的模型）。
   在每个 epoch 结束时，[`Trainer`] 将评估 WER 并保存训练检查点。
2. 将训练参数与模型、数据集、分词器、数据整理器和 `compute_metrics` 函数一起传递给 [`Trainer`]。
3. 调用 [`~Trainer.train`] 来微调您的模型。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

训练完成后，使用 [`~transformers.Trainer.push_to_hub`] 方法将您的模型分享到 Hub，方便大家使用您的模型：

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

要深入了解如何微调模型进行自动语音识别，
请查看这篇博客[文章](https://huggingface.co/blog/fine-tune-wav2vec2-english)以了解英语 ASR，
还可以参阅[这篇文章](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)以了解多语言 ASR。

</Tip>

## 推断

很好，现在您已经微调了一个模型，您可以用它进行推断了！

加载您想要运行推断的音频文件。请记住，如果需要，将音频文件的采样率重新采样为与模型匹配的采样率！

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

尝试使用微调后的模型进行推断的最简单方法是使用 [`pipeline`]。
使用您的模型实例化一个用于自动语音识别的 `pipeline`，并将您的音频文件传递给它：

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

转录结果还不错，但可以更好！尝试用更多示例微调您的模型，以获得更好的结果！

</Tip>

如果您愿意，您也可以手动复制 `pipeline` 的结果：

<frameworkcontent>
<pt>

加载一个处理器来预处理音频文件和转录，并将 `input` 返回为 PyTorch 张量：

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

将您的输入传递给模型并返回 logits：

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

获取具有最高概率的预测 `input_ids`，并使用处理器将预测的 `input_ids` 解码回文本：

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt>
</frameworkcontent>

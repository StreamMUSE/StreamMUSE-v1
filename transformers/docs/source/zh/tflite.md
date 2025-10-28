<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 导出为 TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) 是一个轻量级框架，用于资源受限的设备上，如手机、嵌入式系统和物联网（IoT）设备，部署机器学习模型。TFLite 旨在在计算能力、内存和功耗有限的设备上优化和高效运行模型。模型以一种特殊的高效可移植格式表示，其文件扩展名为 `.tflite`。

🤗 Optimum 通过 `exporters.tflite` 模块提供将 🤗 Transformers 模型导出至 TFLite 格式的功能。请参考 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/exporters/tflite/overview) 以获取支持的模型架构列表。

要将模型导出为 TFLite 格式，请安装所需的依赖项：

```bash
pip install optimum[exporters-tf]
```

请参阅 [🤗 Optimum 文档](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model) 以查看所有可用参数，或者在命令行中查看帮助：

```bash
optimum-cli export tflite --help
```

运行以下命令，以从 🤗 Hub 导出模型的检查点（checkpoint），以 `google-bert/bert-base-uncased` 为例：

```bash
optimum-cli export tflite --model google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```

你应该能在日志中看到导出进度以及生成的 `model.tflite` 文件的保存位置，如下所示：

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
```

上面的示例说明了从 🤗 Hub 导出检查点的过程。导出本地模型时，首先需要确保将模型的权重和分词器文件保存在同一目录（`local_path`）中。在使用 CLI（命令行）时，将 `local_path` 传递给 `model` 参数，而不是 🤗 Hub 上的检查点名称。
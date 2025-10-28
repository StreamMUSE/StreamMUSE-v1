<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 导出 🤗 Transformers 模型到 ONNX

🤗 Transformers提供了一个`transformers.onnx`包，通过利用配置对象，您可以将模型checkpoints转换为ONNX图。

有关更多详细信息，请参阅导出 🤗 Transformers 模型的[指南](../serialization)。

## ONNX Configurations

我们提供了三个抽象类，取决于您希望导出的模型架构类型：

* 基于编码器的模型继承 [`~onnx.config.OnnxConfig`]
* 基于解码器的模型继承 [`~onnx.config.OnnxConfigWithPast`]
* 编码器-解码器模型继承 [`~onnx.config.OnnxSeq2SeqConfigWithPast`]

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX Features

每个ONNX配置与一组 _特性_ 相关联，使您能够为不同类型的拓扑结构或任务导出模型。

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager


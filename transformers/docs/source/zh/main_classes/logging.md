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

# Logging

🤗 Transformers拥有一个集中式的日志系统，因此您可以轻松设置库输出的日志详细程度。

当前库的默认日志详细程度为`WARNING`。

要更改日志详细程度，只需使用其中一个直接的setter。例如，以下是如何将日志详细程度更改为INFO级别的方法：

```python
import transformers

transformers.logging.set_verbosity_info()
```

您还可以使用环境变量`TRANSFORMERS_VERBOSITY`来覆盖默认的日志详细程度。您可以将其设置为以下级别之一：`debug`、`info`、`warning`、`error`、`critical`。例如：

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

此外，通过将环境变量`TRANSFORMERS_NO_ADVISORY_WARNINGS`设置为`true`（如*1*），可以禁用一些`warnings`。这将禁用[`logger.warning_advice`]记录的任何警告。例如：

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

以下是如何在您自己的模块或脚本中使用与库相同的logger的示例：

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```


此日志模块的所有方法都在下面进行了记录，主要的方法包括 [`logging.get_verbosity`] 用于获取logger当前输出日志详细程度的级别和 [`logging.set_verbosity`] 用于将详细程度设置为您选择的级别。按照顺序（从最不详细到最详细），这些级别（及其相应的整数值）为：

- `transformers.logging.CRITICAL` 或 `transformers.logging.FATAL`（整数值，50）：仅报告最关键的errors。
- `transformers.logging.ERROR`（整数值，40）：仅报告errors。
- `transformers.logging.WARNING` 或 `transformers.logging.WARN`（整数值，30）：仅报告error和warnings。这是库使用的默认级别。
- `transformers.logging.INFO`（整数值，20）：报告error、warnings和基本信息。
- `transformers.logging.DEBUG`（整数值，10）：报告所有信息。

默认情况下，将在模型下载期间显示`tqdm`进度条。[`logging.disable_progress_bar`] 和 [`logging.enable_progress_bar`] 可用于禁止或启用此行为。

## `logging` vs `warnings`

Python有两个经常一起使用的日志系统：如上所述的`logging`，和对特定buckets中的警告进行进一步分类的`warnings`，例如，`FutureWarning`用于输出已经被弃用的功能或路径，`DeprecationWarning`用于指示即将被弃用的内容。

我们在`transformers`库中同时使用这两个系统。我们利用并调整了`logging`的`captureWarning`方法，以便通过上面的详细程度setters来管理这些警告消息。

对于库的开发人员，这意味着什么呢？我们应该遵循以下启发法则：
- 库的开发人员和依赖于`transformers`的库应优先使用`warnings`
- `logging`应该用于在日常项目中经常使用它的用户

以下是`captureWarnings`方法的参考。

[[autodoc]] logging.captureWarnings

## Base setters

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## Other functions

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar

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

# 共享自定义模型

🤗 Transformers 库设计得易于扩展。每个模型的代码都在仓库给定的子文件夹中，没有进行抽象，因此你可以轻松复制模型代码文件并根据需要进行调整。

如果你要编写全新的模型，从头开始可能更容易。在本教程中，我们将向你展示如何编写自定义模型及其配置，以便可以在 Transformers 中使用它；以及如何与社区共享它（及其依赖的代码），以便任何人都可以使用，即使它不在 🤗 Transformers 库中。

我们将以 ResNet 模型为例，通过将 [timm 库](https://github.com/rwightman/pytorch-image-models) 的 ResNet 类封装到 [`PreTrainedModel`] 中来进行说明。

## 编写自定义配置

在深入研究模型之前，让我们首先编写其配置。模型的配置是一个对象，其中包含构建模型所需的所有信息。我们将在下一节中看到，模型只能接受一个 `config` 来进行初始化，因此我们很需要使该对象尽可能完整。

我们将采用一些我们可能想要调整的 ResNet 类的参数举例。不同的配置将为我们提供不同类型可能的 ResNet 模型。在确认其中一些参数的有效性后，我们只需存储这些参数。

```python
from transformers import PretrainedConfig
from typing import List


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

编写自定义配置时需要记住的三个重要事项如下：
- 必须继承自 `PretrainedConfig`，
- `PretrainedConfig` 的 `__init__` 方法必须接受任何 kwargs，
- 这些 `kwargs` 需要传递给超类的 `__init__` 方法。

继承是为了确保你获得来自 🤗 Transformers 库的所有功能，而另外两个约束源于 `PretrainedConfig` 的字段比你设置的字段多。在使用 `from_pretrained` 方法重新加载配置时，这些字段需要被你的配置接受，然后传递给超类。

为你的配置定义 `model_type`（此处为 `model_type="resnet"`）不是必须的，除非你想使用自动类注册你的模型（请参阅最后一节）。

做完这些以后，就可以像使用库里任何其他模型配置一样，轻松地创建和保存配置。以下代码展示了如何创建并保存 resnet50d 配置：

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

这行代码将在 `custom-resnet` 文件夹内保存一个名为 `config.json` 的文件。然后，你可以使用 `from_pretrained` 方法重新加载配置：

```py
resnet50d_config = ResnetConfig.from_pretrained("custom-resnet")
```

你还可以使用 [`PretrainedConfig`] 类的任何其他方法，例如 [`~PretrainedConfig.push_to_hub`]，直接将配置上传到 Hub。

## 编写自定义模型

有了 ResNet 配置后，就可以继续编写模型了。实际上，我们将编写两个模型：一个模型用于从一批图像中提取隐藏特征（类似于 [`BertModel`]），另一个模型适用于图像分类（类似于 [`BertForSequenceClassification`]）。

正如之前提到的，我们只会编写一个松散的模型包装，以使示例保持简洁。在编写此类之前，只需要建立起块类型（block types）与实际块类（block classes）之间的映射。然后，通过将所有内容传递给ResNet类，从配置中定义模型：

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig


BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

对用于进行图像分类的模型，我们只需更改前向方法：

```py
import torch


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

在这两种情况下，请注意我们如何继承 `PreTrainedModel` 并使用 `config` 调用了超类的初始化（有点像编写常规的torch.nn.Module）。设置 `config_class` 的那行代码不是必须的，除非你想使用自动类注册你的模型（请参阅最后一节）。

<Tip>

如果你的模型与库中的某个模型非常相似，你可以重用与该模型相同的配置。

</Tip>

你可以让模型返回任何你想要的内容，但是像我们为 `ResnetModelForImageClassification` 做的那样返回一个字典，并在传递标签时包含loss，可以使你的模型能够在 [`Trainer`] 类中直接使用。只要你计划使用自己的训练循环或其他库进行训练，也可以使用其他输出格式。

现在我们已经有了模型类，让我们创建一个：

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

同样的，你可以使用 [`PreTrainedModel`] 的任何方法，比如 [`~PreTrainedModel.save_pretrained`] 或者 [`~PreTrainedModel.push_to_hub`]。我们将在下一节中使用第二种方法，并了解如何如何使用我们的模型的代码推送模型权重。但首先，让我们在模型内加载一些预训练权重。

在你自己的用例中，你可能会在自己的数据上训练自定义模型。为了快速完成本教程，我们将使用 resnet50d 的预训练版本。由于我们的模型只是它的包装，转移这些权重将会很容易：

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

现在让我们看看，如何确保在执行 [`~PreTrainedModel.save_pretrained`] 或 [`~PreTrainedModel.push_to_hub`] 时，模型的代码被保存。

## 将代码发送到 Hub

<Tip warning={true}>

此 API 是实验性的，未来的发布中可能会有一些轻微的不兼容更改。

</Tip>

首先，确保你的模型在一个 `.py` 文件中完全定义。只要所有文件都位于同一目录中，它就可以依赖于某些其他文件的相对导入（目前我们还不为子模块支持此功能）。对于我们的示例，我们将在当前工作目录中名为 `resnet_model` 的文件夹中定义一个 `modeling_resnet.py` 文件和一个 `configuration_resnet.py` 文件。 配置文件包含 `ResnetConfig` 的代码，模型文件包含 `ResnetModel` 和 `ResnetModelForImageClassification` 的代码。

```
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

`__init__.py` 可以为空，它的存在只是为了让 Python 检测到 `resnet_model` 可以用作模块。

<Tip warning={true}>

如果从库中复制模型文件，你需要将文件顶部的所有相对导入替换为从 `transformers` 包中的导入。

</Tip>

请注意，你可以重用（或子类化）现有的配置/模型。

要与社区共享您的模型，请参照以下步骤：首先从新创建的文件中导入ResNet模型和配置：

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

接下来，你需要告诉库，当使用 `save_pretrained` 方法时，你希望复制这些对象的代码文件，并将它们正确注册到给定的 Auto 类（特别是对于模型），只需要运行以下代码：

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

请注意，对于配置（只有一个自动类 [`AutoConfig`]），不需要指定自动类，但对于模型来说情况不同。 你的自定义模型可能适用于许多不同的任务，因此你必须指定哪一个自动类适合你的模型。

接下来，让我们像之前一样创建配置和模型：

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

现在要将模型推送到集线器，请确保你已登录。你看可以在终端中运行以下命令：

```bash
huggingface-cli login
```

或者在笔记本中运行以下代码：

```py
from huggingface_hub import notebook_login

notebook_login()
```

然后，可以这样将模型推送到自己的命名空间（或你所属的组织）：

```py
resnet50d.push_to_hub("custom-resnet50d")
```

除了模型权重和 JSON 格式的配置外，这行代码也会复制 `custom-resnet50d` 文件夹内的模型以及配置的 `.py` 文件并将结果上传至 Hub。你可以在此[模型仓库](https://huggingface.co/sgugger/custom-resnet50d)中查看结果。

有关推推送至 Hub 方法的更多信息，请参阅[共享教程](model_sharing)。

## 使用带有自定义代码的模型

可以使用自动类（auto-classes）和 `from_pretrained` 方法，使用模型仓库里带有自定义代码的配置、模型或分词器文件。所有上传到 Hub 的文件和代码都会进行恶意软件扫描（有关更多信息，请参阅 [Hub 安全](https://huggingface.co/docs/hub/security#malware-scanning) 文档）, 但你仍应查看模型代码和作者，以避免在你的计算机上执行恶意代码。 设置 `trust_remote_code=True` 以使用带有自定义代码的模型：

```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

我们强烈建议为 `revision` 参数传递提交哈希（commit hash），以确保模型的作者没有使用一些恶意的代码行更新了代码（除非您完全信任模型的作者）。

```py
commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
model = AutoModelForImageClassification.from_pretrained(
    "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
)
```

在 Hub 上浏览模型仓库的提交历史时，有一个按钮可以轻松复制任何提交的提交哈希。

## 将自定义代码的模型注册到自动类

如果你在编写一个扩展 🤗 Transformers 的库，你可能想要扩展自动类以包含您自己的模型。这与将代码推送到 Hub 不同，因为用户需要导入你的库才能获取自定义模型（与从 Hub 自动下载模型代码相反）。

只要你的配置 `model_type` 属性与现有模型类型不同，并且你的模型类有正确的 `config_class` 属性，你可以像这样将它们添加到自动类中：

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

请注意，将自定义配置注册到 [`AutoConfig`] 时，使用的第一个参数需要与自定义配置的 `model_type` 匹配；而将自定义模型注册到任何自动模型类时，使用的第一个参数需要与 `config_class` 匹配。

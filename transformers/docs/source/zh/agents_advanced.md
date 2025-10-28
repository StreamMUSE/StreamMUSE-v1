<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
# 智能体，超强版 - 多智能体、外部工具等

[[open-in-colab]]

### 什么是智能体？

> [!TIP]
> 如果你是 `transformers.agents` 的新手，请先阅读主文档 [智能体文档 ](./agents).
在本页面中，我们将重点介绍 `transformers.agents` 的几种高级用法。

## 多智能体

多智能体功能是微软框架 [Autogen](https://huggingface.co/papers/2308.08155) 中引入的。
它的意思是让多个智能体一起工作来解决任务，而不是只有一个智能体。
经验表明，在大多数基准测试中，这种方法能带来更好的性能。之所以有更好的性能，原因很简单：对于许多任务，通常我们更愿意让多个单独的单元专注于子任务，而不是让一个系统做所有事情。这里，拥有不同工具集和记忆的多个智能体可以实现高效的专业化。

你可以轻松地用 `transformers.agents` 构建层次化的多智能体系统。

为此，需要将智能体封装在 [`ManagedAgent`] 对象中。这个对象需要 `agent`、`name` 和 `description` 这几个参数，这些信息会嵌入到管理智能体的系统提示中，帮助它知道如何调用这个管理的智能体，就像我们对工具所做的那样。

下面是一个通过使用我们的 [`DuckDuckGoSearchTool`] 创建一个管理特定网络搜索智能体的示例：


```py
from transformers.agents import ReactCodeAgent, HfApiEngine, DuckDuckGoSearchTool, ManagedAgent

llm_engine = HfApiEngine()

web_agent = ReactCodeAgent(tools=[DuckDuckGoSearchTool()], llm_engine=llm_engine)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)

manager_agent = ReactCodeAgent(
    tools=[], llm_engine=llm_engine, managed_agents=[managed_web_agent]
)

manager_agent.run("Who is the CEO of Hugging Face?")
```

> [!TIP]
> 如果你想深入了解如何高效地实现多智能体系统，请查看 [how we pushed our multi-agent system to the top of the GAIA leaderboard](https://huggingface.co/blog/beating-gaia).

## 高级工具使用

### 通过子类化 Tool 来直接定义工具，并将其共享到 Hub

让我们再次使用主文档中的工具示例，我们已经实现了一个 `tool` 装饰器。

如果你需要添加一些变化，比如为工具自定义属性，可以按照更细粒度的方法构建工具：构建一个继承自 [`Tool`] 超类的类。

自定义工具需要：
- `name` 属性：表示工具本身的名称，通常描述工具的作用。由于代码返回了针对任务下载量最多的模型，我们将其命名为 model_download_counter。
- `description` 属性：用于填充智能体的系统提示。
- `inputs` 属性：这是一个包含 "type" 和 "description" 键的字典。它包含了有助于 Python 解释器做出选择的输入信息。
- `output_type` 属性：指定输出类型。
- `forward` 方法：其中包含执行推理代码。

`inputs` 和 `output_type` 的类型应当是 [Pydantic 格式](https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema)。

```python
from transformers import Tool
from huggingface_hub import list_models

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""

    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

现在，自定义的 `HfModelDownloadsTool` 类已经准备好，可以将其保存到名为 `model_downloads.py` 的文件中，并导入使用。


```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

你还可以通过调用 [`~Tool.push_to_hub`] 将自定义工具推送到 Hub。确保你已经为该工具创建了一个仓库，并使用具有读取访问权限的许可。

```python
tool.push_to_hub("{your_username}/hf-model-downloads")
```

通过 [`~Tool.load_tool`] 函数加载工具，并将其传递给智能体的 tools 参数。

```python
from transformers import load_tool, CodeAgent

model_download_tool = load_tool("m-ric/hf-model-downloads")
```

### 将 Space 导入为工具 🚀

你可以直接通过 [`Tool.from_space`] 方法将 Hub 上的 Space 导入为工具！

只需要提供 Space 在 Hub 上的 ID、名称和描述，帮助智能体理解工具的作用。在幕后，这将使用 [`gradio-client`](https://pypi.org/project/gradio-client/) 库来调用 Space。

例如，下面是从 Hub 导入 `FLUX.1-dev` Space 并用其生成图像的示例：

```
from transformers import Tool
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-dev",
    name="image_generator",
    description="Generate an image from a prompt")
image_generation_tool("A sunny beach")
```
看！这就是你生成的图像！🏖️

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sunny_beach.webp">

然后，你可以像使用其他工具一样使用这个工具。例如，改进提示 `穿宇航服的兔子` 并生成其图像：

```python
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[image_generation_tool])

agent.run(
    "Improve this prompt, then generate an image of it.", prompt='A rabbit wearing a space suit'
)
```

```text
=== Agent thoughts:
improved_prompt could be "A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background"
Now that I have improved the prompt, I can use the image generator tool to generate an image based on this prompt.
>>> Agent is executing the code below:
image = image_generator(prompt="A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background")
final_answer(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit_spacesuit_flux.webp">

这真酷吧？🤩

### 使用 gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) 是一个强大的库，允许使用 Hugging Face Spaces 作为工具。它支持许多现有的 Spaces，也支持自定义 Spaces。

transformers 支持通过 [`Tool.from_gradio`] 方法使用 `gradio_tools`。例如，下面是如何使用来自 `gradio-tools` 工具包的 [`StableDiffusionPromptGeneratorTool`](https://github.com/freddyaboulton/gradio-tools/blob/main/gradio_tools/tools/prompt_generator.py) 来改进提示，以生成更好的图像：

导入和实例化工具，并将其传递给 `Tool.from_gradio` 方法:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool
from transformers import Tool, load_tool, CodeAgent

gradio_prompt_generator_tool = StableDiffusionPromptGeneratorTool()
prompt_generator_tool = Tool.from_gradio(gradio_prompt_generator_tool)
```

> [!WARNING]
> gradio-tools 需要 **文本** 输入和输出，即使在处理像图像和音频这样的不同模态时也是如此。目前，图像和音频的输入输出与此不兼容。
### 使用 LangChain 工具

我们很喜欢 LangChain，并认为它有一套非常有吸引力的工具。
要从 LangChain 导入工具，可以使用 `from_langchain()` 方法。

例如，下面是如何使用它来重新创建上面介绍的搜索结果，使用一个 LangChain 网络搜索工具。该工具需要 `pip install google-search-results` 来正常工作。

```python
from langchain.agents import load_tools
from transformers import Tool, ReactCodeAgent

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = ReactCodeAgent(tools=[search_tool])

agent.run("How many more blocks (also denoted as layers) are in BERT base encoder compared to the encoder from the architecture proposed in Attention is All You Need?")
```

## 在酷炫的 Gradio 界面中展示智能体运行

你可以利用 `gradio.Chatbot` 来展示智能体的思考过程，通过 `stream_to_gradio`，下面是一个示例：

```py
import gradio as gr
from transformers import (
    load_tool,
    ReactCodeAgent,
    HfApiEngine,
    stream_to_gradio,
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

llm_engine = HfApiEngine("meta-llama/Meta-Llama-3-70B-Instruct")

# Initialize the agent with the image generation tool
agent = ReactCodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)


def interact_with_agent(task):
    messages = []
    messages.append(gr.ChatMessage(role="user", content=task))
    yield messages
    for msg in stream_to_gradio(agent, task):
        messages.append(msg)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ Task not finished yet!")
        ]
    yield messages


with gr.Blocks() as demo:
    text_input = gr.Textbox(lines=1, label="Chat Message", value="Make me a picture of the Statue of Liberty.")
    submit = gr.Button("Run illustrator agent!")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
    )
    submit.click(interact_with_agent, [text_input], [chatbot])

if __name__ == "__main__":
    demo.launch()
```
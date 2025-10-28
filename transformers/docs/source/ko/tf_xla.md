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

# TensorFlow 모델을 위한 XLA 통합 [[xla-integration-for-tensorflow-models]]

[[open-in-colab]]

XLA(Accelerated Linear Algebra)는 TensorFlow 모델의 실행 시간을 가속화하기 위한 컴파일러입니다. [공식 문서](https://www.tensorflow.org/xla)에 따르면 다음과 같습니다:

XLA(Accelerated Linear Algebra)는 선형 대수를 위한 도메인 특화 컴파일러로, TensorFlow 모델을 소스 코드 변경 없이 가속화할 수 있습니다.

TensorFlow에서 XLA를 사용하는 것은 간단합니다. XLA는 `tensorflow` 라이브러리 내에 패키지로 제공되며, [`tf.function`](https://www.tensorflow.org/guide/intro_to_graphs)과 같은 그래프 생성 함수에서 `jit_compile` 인수를 사용하여 활성화할 수 있습니다. `fit()` 및 `predict()`와 같은 Keras 메소드를 사용하는 경우, `jit_compile` 인수를 `model.compile()`에 전달하여 XLA를 간단하게 활성화할 수 있습니다. 그러나 XLA는 이러한 메소드에 국한되지 않고 임의의 `tf.function`을 가속화하는 데에도 사용할 수 있습니다.

🤗 Transformers에서는 [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2), [T5](https://huggingface.co/docs/transformers/model_doc/t5), [OPT](https://huggingface.co/docs/transformers/model_doc/opt)와 같은 모델의 텍스트 생성, 그리고 [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)와 같은 모델의 음성 처리를 포함하여 여러 TensorFlow 메소드가 XLA와 호환되도록 다시 작성되었습니다.

정확한 속도 향상은 모델에 따라 다르지만, 🤗 Transformers 내의 TensorFlow 텍스트 생성 모델의 경우 최대 100배의 속도 향상을 확인했습니다. 이 문서에서는 이러한 모델에 대해 XLA를 사용하여 최대 성능을 얻는 방법을 설명합니다. 또한 XLA 통합의 벤치마크 및 디자인 철학에 대한 추가 자료 링크도 제공할 것입니다.

## XLA를 사용하여 TF 함수 실행하기 [[running-tf-functions-with-xla]]

TensorFlow에서 다음과 같은 모델을 고려해 봅시다:

```py
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

위 모델은 차원이 `(10, )`인 입력을 받습니다. 다음과 같이 모델을 사용하여 순전파를 실행할 수 있습니다:

```py
# 모델에 대한 임의의 입력을 생성합니다.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# 순전파를 실행합니다.
_ = model(random_inputs)
```

XLA로 컴파일된 함수로 순전파를 실행하려면 다음과 같이 해야 합니다:

```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

`model`의 기본 `call()` 함수는 XLA 그래프를 컴파일하는 데 사용됩니다. 그러나 다른 모델 함수를 XLA로 컴파일하려면 다음과 같이 할 수도 있습니다:

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## 🤗 Transformers에서 XLA를 사용하여 TF 텍스트 생성 모델 실행하기 [[running-a-tf-text-generation-model-with-xla-from-transformers]]

🤗 Transformers에서 XLA로 가속화된 생성을 활성화하려면 최신 버전의 `transformers`가 설치되어 있어야 합니다. 다음과 같이 설치할 수 있습니다:

```bash
pip install transformers --upgrade
```

그리고 다음 코드를 실행할 수 있습니다:

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# 최소 버전의 Transformers가 설치되어 있지 않다면 오류가 발생합니다.
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

# XLA 생성 함수를 만들기 위한 한 줄
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# Generated -- TensorFlow is an open-source, open-source, distributed-source application # framework for the
```

알 수 있듯이, `generate()`에서 XLA를 활성화하는 것은 단 한 줄의 코드입니다. 코드의 나머지 부분은 변경되지 않습니다. 그러나 위 코드 스니펫에서는 XLA에 특정한 몇 가지 주의할 점이 있습니다. XLA가 가져다줄 속도 향상을 실현하기 위해서는 이를 알고 있어야 합니다. 다음 섹션에서 이에 대해 논의합니다.

## 주의할 점 [[gotchas-to-be-aware-of]]

XLA 활성화 함수(`xla_generate()`와 같은)를 처음 실행할 때 내부적으로 계산 그래프를 추론하려고 하며, 이는 시간이 소요됩니다. 이 과정은 [“추적(tracing)”](https://www.tensorflow.org/guide/intro_to_graphs#when_is_a_function_tracing)이라고 알려져 있습니다.

생성 시간이 빠르지 않다는 것을 알 수 있을 것입니다. `xla_generate()`(또는 다른 XLA 활성화 함수)의 연속 호출은 함수에 전달된 입력이 초기에 구축된 계산 그래프와 동일한 형태를 따른다면, 계산 그래프를 추론할 필요가 없습니다. 이는 입력 형태가 고정된 모달리티(예: 이미지)에는 문제가 되지 않지만, 가변 입력 형태 모달리티(예: 텍스트)를 사용할 때 주의해야 합니다.

`xla_generate()`가 항상 동일한 입력 형태로 동작하도록 하려면, 토크나이저를 호출할 때 `padding` 인수를 지정할 수 있습니다.

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# 여기서, padding 옵션이 있는 토크나이저를 호출합니다.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

이렇게 하면 `xla_generate()`에 대한 입력이 항상 추적된 형태로 전달되어 생성 시간이 가속화됩니다. 다음 코드로 이를 확인할 수 있습니다:

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
    tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
    start = time.time_ns()
    generated_tokens = xla_generate(**tokenized_input, num_beams=2)
    end = time.time_ns()
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

Tesla T4 GPU에서는 다음과 같은 출력을 예상할 수 있습니다:

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```
`xla_generate()`의 첫 번째 호출은 추적 때문에 시간이 오래 걸리지만, 연속 호출은 몇 배나 빠릅니다. 생성 옵션에 대한 어떤 변경이든 다시 추적을 유발하므로 생성 시간이 느려질 수 있음을 명심하세요.

이 문서에서는 🤗 Transformers에서 제공하는 모든 텍스트 생성 옵션을 다루지 않았습니다. 고급 사용 사례에 대해 문서를 참조하시기 바랍니다.

## 추가 자료 [[additional-resources]]

여기에 🤗 Transformers와 XLA에 대해 더 자세히 알고 싶은 경우 도움이 될 수 있는 몇 가지 추가 자료를 제공합니다. 
 
* [이 Colab 노트북](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb)은 XLA와 호환되는 인코더-디코더([T5](https://huggingface.co/docs/transformers/model_doc/t5)와 같은) 및 디코더 전용([GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)와 같은) 텍스트 생성 모델을 실험해 볼 수 있는 대화형 데모를 제공합니다.
* [이 블로그 글](https://huggingface.co/blog/tf-xla-generate)은 TensorFlow에서 XLA에 대한 친절한 소개와 함께 XLA와 호환되는 모델의 비교 벤치마크에 대한 개요를 제공합니다.
* [이 블로그 글](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html)은 🤗 Transformers의 TensorFlow 모델에 XLA 지원을 추가하는 것에 대한 디자인 철학을 논의합니다.
* XLA와 TensorFlow 그래프에 대해 더 자세히 알고 싶은 경우 추천하는 글:
    * [XLA: 기계 학습을 위한 최적화 컴파일러](https://www.tensorflow.org/xla)
    * [그래프 및 tf.function 소개](https://www.tensorflow.org/guide/intro_to_graphs)
    * [tf.function으로 성능 향상하기](https://www.tensorflow.org/guide/function) 
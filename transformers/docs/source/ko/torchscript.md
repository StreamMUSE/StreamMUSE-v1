<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TorchScript로 내보내기[[export-to-torchscript]]

<Tip>

TorchScript를 활용한 실험은 아직 초기 단계로, 가변적인 입력 크기 모델들을 통해 그 기능성을 계속 탐구하고 있습니다. 
이 기능은 저희가 관심을 두고 있는 분야 중 하나이며, 
앞으로 출시될 버전에서 더 많은 코드 예제, 더 유연한 구현, 그리고 Python 기반 코드와 컴파일된 TorchScript를 비교하는 벤치마크를 등을 통해 분석을 심화할 예정입니다.

</Tip>

[TorchScript 문서](https://pytorch.org/docs/stable/jit.html)에서는 이렇게 말합니다.

> TorchScript는 PyTorch 코드에서 직렬화 및 최적화 가능한 모델을 생성하는 방법입니다.

[JIT과 TRACE](https://pytorch.org/docs/stable/jit.html)는 개발자가 모델을 내보내서 효율 지향적인 C++ 프로그램과 같은 다른 프로그램에서 재사용할 수 있도록 하는 PyTorch 모듈입니다.

PyTorch 기반 Python 프로그램과 다른 환경에서 모델을 재사용할 수 있도록, 🤗 Transformers 모델을 TorchScript로 내보낼 수 있는 인터페이스를 제공합니다. 
이 문서에서는 TorchScript를 사용하여 모델을 내보내고 사용하는 방법을 설명합니다.

모델을 내보내려면 두 가지가 필요합니다:

- `torchscript` 플래그로 모델 인스턴스화
- 더미 입력을 사용한 순전파(forward pass)

이 필수 조건들은 아래에 자세히 설명된 것처럼 개발자들이 주의해야 할 여러 사항들을 의미합니다.

## TorchScript 플래그와 묶인 가중치(tied weights)[[torchscript-flag-and-tied-weights]]

`torchscript` 플래그가 필요한 이유는 대부분의 🤗 Transformers 언어 모델에서 `Embedding` 레이어와 `Decoding` 레이어 간의 묶인 가중치(tied weights)가 존재하기 때문입니다.
TorchScript는 묶인 가중치를 가진 모델을 내보낼 수 없으므로, 미리 가중치를 풀고 복제해야 합니다.

`torchscript` 플래그로 인스턴스화된 모델은 `Embedding` 레이어와 `Decoding` 레이어가 분리되어 있으므로 이후에 훈련해서는 안 됩니다.
훈련을 하게 되면 두 레이어 간 동기화가 해제되어 예상치 못한 결과가 발생할 수 있습니다.

언어 모델 헤드를 갖지 않은 모델은 가중치가 묶여 있지 않아서 이 문제가 발생하지 않습니다.
이러한 모델들은 `torchscript` 플래그 없이 안전하게 내보낼 수 있습니다.

## 더미 입력과 표준 길이[[dummy-inputs-and-standard-lengths]]

더미 입력(dummy inputs)은 모델의 순전파(forward pass)에 사용됩니다. 
입력 값이 레이어를 통해 전파되는 동안, PyTorch는 각 텐서에서 실행된 다른 연산을 추적합니다. 
이러한 기록된 연산은 모델의 *추적(trace)*을 생성하는 데 사용됩니다.

추적은 입력의 차원을 기준으로 생성됩니다. 
따라서 더미 입력의 차원에 제한되어, 다른 시퀀스 길이나 배치 크기에서는 작동하지 않습니다. 
다른 크기로 시도할 경우 다음과 같은 오류가 발생합니다:

```
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```
추론 중 모델에 공급될 가장 큰 입력만큼 큰 더미 입력 크기로 모델을 추적하는 것이 좋습니다. 
패딩은 누락된 값을 채우는 데 도움이 될 수 있습니다. 
그러나 모델이 더 큰 입력 크기로 추적되기 때문에, 행렬의 차원이 커지고 계산량이 많아집니다.

다양한 시퀀스 길이 모델을 내보낼 때는 각 입력에 대해 수행되는 총 연산 횟수에 주의하고 성능을 주의 깊게 확인하세요.

## Python에서 TorchScript 사용하기[[using-torchscript-in-python]]

이 섹션에서는 모델을 저장하고 가져오는 방법, 추적을 사용하여 추론하는 방법을 보여줍니다.

### 모델 저장하기[[saving-a-model]]

`BertModel`을 TorchScript로 내보내려면 `BertConfig` 클래스에서 `BertModel`을 인스턴스화한 다음, `traced_bert.pt`라는 파일명으로 디스크에 저장하면 됩니다.

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# 입력 텍스트 토큰화하기
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# 입력 토큰 중 하나를 마스킹하기
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 더미 입력 만들기
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# torchscript 플래그로 모델 초기화하기
# 이 모델은 LM 헤드가 없으므로 필요하지 않지만, 플래그를 True로 설정합니다.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# 모델을 인스턴트화하기
model = BertModel(config)

# 모델을 평가 모드로 두어야 합니다.
model.eval()

# 만약 *from_pretrained*를 사용하여 모델을 인스턴스화하는 경우, TorchScript 플래그를 쉽게 설정할 수 있습니다
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)

# 추적 생성하기
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### 모델 가져오기[[loading-a-model]]

이제 이전에 저장한 `BertModel`, 즉 `traced_bert.pt`를 디스크에서 가져오고, 이전에 초기화한 `dummy_input`에서 사용할 수 있습니다.

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### 추적된 모델을 사용하여 추론하기[[using-a-traced-model-for-inference]]

`__call__` 이중 언더스코어(dunder) 메소드를 사용하여 추론에 추적된 모델을 사용하세요:

```python
traced_model(tokens_tensor, segments_tensors)
```

## Neuron SDK로 Hugging Face TorchScript 모델을 AWS에 배포하기[[deploy-hugging-face-torchscript-models-to-aws-with-the-neuron-sdk]]

AWS가 클라우드에서 저비용, 고성능 머신 러닝 추론을 위한 [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) 인스턴스 제품군을 출시했습니다. 
Inf1 인스턴스는 딥러닝 추론 워크로드에 특화된 맞춤 하드웨어 가속기인 AWS Inferentia 칩으로 구동됩니다. 
[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#)은 Inferentia를 위한 SDK로, Inf1에 배포하기 위한 transformers 모델 추적 및 최적화를 지원합니다. 
Neuron SDK는 다음과 같은 기능을 제공합니다:

1. 코드 한 줄만 변경하면 클라우드 추론를 위해 TorchScript 모델을 추적하고 최적화할 수 있는 쉬운 API
2. 즉시 사용 가능한 성능 최적화로 [비용 효율 향상](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/>)
3. [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) 또는 [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html)로 구축된 Hugging Face transformers 모델 지원

### 시사점[[implications]]

[BERT (Bidirectional Encoder Representations from Transformers)](https://huggingface.co/docs/transformers/main/model_doc/bert) 아키텍처 또는 그 변형인 [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) 및 [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta)를 기반으로 한 Transformers 모델은 추출 기반 질의응답, 시퀀스 분류 및 토큰 분류와 같은 비생성 작업 시 Inf1에서 최상의 성능을 보입니다. 
그러나 텍스트 생성 작업도 [AWS Neuron MarianMT 튜토리얼](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html)을 따라 Inf1에서 실행되도록 조정할 수 있습니다.

Inferentia에서 바로 변환할 수 있는 모델에 대한 자세한 정보는 Neuron 문서의 [Model Architecture Fit](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) 섹션에서 확인할 수 있습니다.

### 종속성[[dependencies]]

AWS Neuron을 사용하여 모델을 변환하려면 [Neuron SDK 환경](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide)이 필요합니다.
 이는 [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html)에 미리 구성되어 있습니다.

### AWS Neuron으로 모델 변환하기[[converting-a-model-for-aws-neuron]]

`BertModel`을 추적하려면, [Python에서 TorchScript 사용하기](torchscript#using-torchscript-in-python)에서와 동일한 코드를 사용해서 AWS NEURON용 모델을 변환합니다. 
`torch.neuron` 프레임워크 익스텐션을 가져와 Python API를 통해 Neuron SDK의 구성 요소에 접근합니다:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

다음 줄만 수정하면 됩니다:

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

이로써 Neuron SDK가 모델을 추적하고 Inf1 인스턴스에 최적화할 수 있게 됩니다.

AWS Neuron SDK의 기능, 도구, 예제 튜토리얼 및 최신 업데이트에 대해 자세히 알아보려면 [AWS NeuronSDK 문서](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html)를 참조하세요.

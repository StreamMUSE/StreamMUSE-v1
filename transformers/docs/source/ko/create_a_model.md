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

# 맞춤형 아키텍처 만들기[[create-a-custom-architecture]]

[`AutoClass`](model_doc/auto)는 모델 아키텍처를 자동으로 추론하고 미리 학습된 configuration과 가중치를 다운로드합니다. 일반적으로 체크포인트에 구애받지 않는 코드를 생성하려면 `AutoClass`를 사용하는 것이 좋습니다. 하지만 특정 모델 파라미터를 보다 세밀하게 제어하고자 하는 사용자는 몇 가지 기본 클래스만으로 커스텀 🤗 Transformers 모델을 생성할 수 있습니다. 이는 🤗 Transformers 모델을 연구, 교육 또는 실험하는 데 관심이 있는 모든 사용자에게 특히 유용할 수 있습니다. 이 가이드에서는 'AutoClass'를 사용하지 않고 커스텀 모델을 만드는 방법에 대해 알아보겠습니다:

- 모델 configuration을 가져오고 사용자 지정합니다.
- 모델 아키텍처를 생성합니다.
- 텍스트에 사용할 느리거나 빠른 토큰화기를 만듭니다.
- 비전 작업을 위한 이미지 프로세서를 생성합니다.
- 오디오 작업을 위한 특성 추출기를 생성합니다.
- 멀티모달 작업용 프로세서를 생성합니다.

## Configuration[[configuration]]

[configuration](main_classes/configuration)은 모델의 특정 속성을 나타냅니다. 각 모델 구성에는 서로 다른 속성이 있습니다. 예를 들어, 모든 NLP 모델에는 `hidden_size`, `num_attention_heads`, `num_hidden_layers` 및 `vocab_size` 속성이 공통으로 있습니다. 이러한 속성은 모델을 구성할 attention heads 또는 hidden layers의 수를 지정합니다.

[DistilBERT](model_doc/distilbert) 속성을 검사하기 위해 [`DistilBertConfig`]에 접근하여 자세히 살펴봅니다:

```py
>>> from transformers import DistilBertConfig

>>> config = DistilBertConfig()
>>> print(config)
DistilBertConfig {
  "activation": "gelu",
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

[`DistilBertConfig`]는 기본 [`DistilBertModel`]을 빌드하는 데 사용되는 모든 기본 속성을 표시합니다. 모든 속성은 커스터마이징이 가능하므로 실험을 위한 공간을 만들 수 있습니다. 예를 들어 기본 모델을 다음과 같이 커스터마이즈할 수 있습니다:

- `activation` 파라미터로 다른 활성화 함수를 사용해 보세요.
- `attention_dropout` 파라미터를 사용하여 어텐션 확률에 더 높은 드롭아웃 비율을 사용하세요.

```py
>>> my_config = DistilBertConfig(activation="relu", attention_dropout=0.4)
>>> print(my_config)
DistilBertConfig {
  "activation": "relu",
  "attention_dropout": 0.4,
  "dim": 768,
  "dropout": 0.1,
  "hidden_dim": 3072,
  "initializer_range": 0.02,
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "transformers_version": "4.16.2",
  "vocab_size": 30522
}
```

사전 학습된 모델 속성은 [`~PretrainedConfig.from_pretrained`] 함수에서 수정할 수 있습니다:

```py
>>> my_config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased", activation="relu", attention_dropout=0.4)
```

모델 구성이 만족스러우면 [`~PretrainedConfig.save_pretrained`]로 저장할 수 있습니다. 설정 파일은 지정된 작업 경로에 JSON 파일로 저장됩니다:

```py
>>> my_config.save_pretrained(save_directory="./your_model_save_path")
```

configuration 파일을 재사용하려면 [`~PretrainedConfig.from_pretrained`]를 사용하여 가져오세요:

```py
>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
```

<Tip>

configuration 파일을 딕셔너리로 저장하거나 사용자 정의 configuration 속성과 기본 configuration 속성의 차이점만 저장할 수도 있습니다! 자세한 내용은 [configuration](main_classes/configuration) 문서를 참조하세요.

</Tip>

## 모델[[model]]

다음 단계는 [모델(model)](main_classes/models)을 만드는 것입니다. 느슨하게 아키텍처라고도 불리는 모델은 각 계층이 수행하는 동작과 발생하는 작업을 정의합니다. configuration의 `num_hidden_layers`와 같은 속성은 아키텍처를 정의하는 데 사용됩니다. 모든 모델은 기본 클래스 [`PreTrainedModel`]과 입력 임베딩 크기 조정 및 셀프 어텐션 헤드 가지 치기와 같은 몇 가지 일반적인 메소드를 공유합니다. 또한 모든 모델은 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 또는 [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)의 서브클래스이기도 합니다. 즉, 모델은 각 프레임워크의 사용법과 호환됩니다.

<frameworkcontent>
<pt>
사용자 지정 configuration 속성을 모델에 가져옵니다:

```py
>>> from transformers import DistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/config.json")
>>> model = DistilBertModel(my_config)
```

이제 사전 학습된 가중치 대신 임의의 값을 가진 모델이 생성됩니다. 이 모델을 훈련하기 전까지는 유용하게 사용할 수 없습니다. 훈련은 비용과 시간이 많이 소요되는 프로세스입니다. 일반적으로 훈련에 필요한 리소스의 일부만 사용하면서 더 나은 결과를 더 빨리 얻으려면 사전 훈련된 모델을 사용하는 것이 좋습니다.

사전 학습된 모델을 [`~PreTrainedModel.from_pretrained`]로 생성합니다:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

🤗 Transformers에서 제공한 모델의 사전 학습된 가중치를 사용하는 경우 기본 모델 configuration을 자동으로 불러옵니다. 그러나 원하는 경우 기본 모델 configuration 속성의 일부 또는 전부를 사용자 지정으로 바꿀 수 있습니다:

```py
>>> model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```
</pt>
<tf>
사용자 지정 configuration 속성을 모델에 불러옵니다:

```py
>>> from transformers import TFDistilBertModel

>>> my_config = DistilBertConfig.from_pretrained("./your_model_save_path/my_config.json")
>>> tf_model = TFDistilBertModel(my_config)
```

이제 사전 학습된 가중치 대신 임의의 값을 가진 모델이 생성됩니다. 이 모델을 훈련하기 전까지는 유용하게 사용할 수 없습니다. 훈련은 비용과 시간이 많이 소요되는 프로세스입니다. 일반적으로 훈련에 필요한 리소스의 일부만 사용하면서 더 나은 결과를 더 빨리 얻으려면 사전 훈련된 모델을 사용하는 것이 좋습니다.

사전 학습된 모델을 [`~TFPreTrainedModel.from_pretrained`]로 생성합니다:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
```

🤗 Transformers에서 제공한 모델의 사전 학습된 가중치를 사용하는 경우 기본 모델 configuration을 자동으로 불러옵니다. 그러나 원하는 경우 기본 모델 configuration 속성의 일부 또는 전부를 사용자 지정으로 바꿀 수 있습니다:

```py
>>> tf_model = TFDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", config=my_config)
```
</tf>
</frameworkcontent>

### 모델 헤드[[model-heads]]

이 시점에서 *은닉 상태(hidden state)*를 출력하는 기본 DistilBERT 모델을 갖게 됩니다. 은닉 상태는 최종 출력을 생성하기 위해 모델 헤드에 입력으로 전달됩니다. 🤗 Transformers는 모델이 해당 작업을 지원하는 한 각 작업마다 다른 모델 헤드를 제공합니다(즉, 번역과 같은 시퀀스 간 작업에는 DistilBERT를 사용할 수 없음).

<frameworkcontent>
<pt>
예를 들어, [`DistilBertForSequenceClassification`]은 시퀀스 분류 헤드가 있는 기본 DistilBERT 모델입니다. 시퀀스 분류 헤드는 풀링된 출력 위에 있는 선형 레이어입니다.

```py
>>> from transformers import DistilBertForSequenceClassification

>>> model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

다른 모델 헤드로 전환하여 이 체크포인트를 다른 작업에 쉽게 재사용할 수 있습니다. 질의응답 작업의 경우, [`DistilBertForQuestionAnswering`] 모델 헤드를 사용할 수 있습니다. 질의응답 헤드는 숨겨진 상태 출력 위에 선형 레이어가 있다는 점을 제외하면 시퀀스 분류 헤드와 유사합니다.

```py
>>> from transformers import DistilBertForQuestionAnswering

>>> model = DistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</pt>
<tf>
예를 들어, [`TFDistilBertForSequenceClassification`]은 시퀀스 분류 헤드가 있는 기본 DistilBERT 모델입니다. 시퀀스 분류 헤드는 풀링된 출력 위에 있는 선형 레이어입니다.

```py
>>> from transformers import TFDistilBertForSequenceClassification

>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
```

다른 모델 헤드로 전환하여 이 체크포인트를 다른 작업에 쉽게 재사용할 수 있습니다. 질의응답 작업의 경우, [`TFDistilBertForQuestionAnswering`] 모델 헤드를 사용할 수 있습니다. 질의응답 헤드는 숨겨진 상태 출력 위에 선형 레이어가 있다는 점을 제외하면 시퀀스 분류 헤드와 유사합니다.

```py
>>> from transformers import TFDistilBertForQuestionAnswering

>>> tf_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```
</tf>
</frameworkcontent>

## 토크나이저[[tokenizer]]

텍스트 데이터에 모델을 사용하기 전에 마지막으로 필요한 기본 클래스는 원시 텍스트를 텐서로 변환하는 [토크나이저](main_classes/tokenizer)입니다. 🤗 Transformers에 사용할 수 있는 토크나이저는 두 가지 유형이 있습니다:

- [`PreTrainedTokenizer`]: 파이썬으로 구현된 토크나이저입니다.
- [`PreTrainedTokenizerFast`]: Rust 기반 [🤗 Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/) 라이브러리로 만들어진 토크나이저입니다. 이 토크나이저는 Rust로 구현되어 배치 토큰화에서 특히 빠릅니다. 빠른 토크나이저는 토큰을 원래 단어나 문자에 매핑하는 *오프셋 매핑*과 같은 추가 메소드도 제공합니다.
두 토크나이저 모두 인코딩 및 디코딩, 새 토큰 추가, 특수 토큰 관리와 같은 일반적인 방법을 지원합니다.

<Tip warning={true}>

모든 모델이 빠른 토크나이저를 지원하는 것은 아닙니다. 이 [표](index#supported-frameworks)에서 모델의 빠른 토크나이저 지원 여부를 확인하세요.

</Tip>

토크나이저를 직접 학습한 경우, *어휘(vocabulary)* 파일에서 토크나이저를 만들 수 있습니다:

```py
>>> from transformers import DistilBertTokenizer

>>> my_tokenizer = DistilBertTokenizer(vocab_file="my_vocab_file.txt", do_lower_case=False, padding_side="left")
```

사용자 지정 토크나이저의 어휘는 사전 학습된 모델의 토크나이저에서 생성된 어휘와 다를 수 있다는 점을 기억하는 것이 중요합니다. 사전 학습된 모델을 사용하는 경우 사전 학습된 모델의 어휘를 사용해야 하며, 그렇지 않으면 입력이 의미를 갖지 못합니다. [`DistilBertTokenizer`] 클래스를 사용하여 사전 학습된 모델의 어휘로 토크나이저를 생성합니다:

```py
>>> from transformers import DistilBertTokenizer

>>> slow_tokenizer = DistilBertTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

[`DistilBertTokenizerFast`] 클래스로 빠른 토크나이저를 생성합니다:

```py
>>> from transformers import DistilBertTokenizerFast

>>> fast_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-uncased")
```

<Tip>

[`AutoTokenizer`]는 기본적으로 빠른 토크나이저를 가져오려고 합니다. 이 동작을 비활성화하려면 `from_pretrained`에서 `use_fast=False`를 설정하면 됩니다.

</Tip>

## 이미지 프로세서[[image-processor]]

이미지 프로세서(image processor)는 비전 입력을 처리합니다. 기본 [`~image_processing_utils.ImageProcessingMixin`] 클래스에서 상속합니다.

사용하려면 사용 중인 모델과 연결된 이미지 프로세서를 생성합니다. 예를 들어, 이미지 분류에 [ViT](model_doc/vit)를 사용하는 경우 기본 [`ViTImageProcessor`]를 생성합니다:

```py
>>> from transformers import ViTImageProcessor

>>> vit_extractor = ViTImageProcessor()
>>> print(vit_extractor)
ViTImageProcessor {
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "ViTImageProcessor",
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "size": 224
}
```

<Tip>

사용자 지정을 원하지 않는 경우 `from_pretrained` 메소드를 사용하여 모델의 기본 이미지 프로세서 매개변수를 불러오면 됩니다.

</Tip>

사용자 지정 이미지 프로세서를 생성하려면 [`ViTImageProcessor`] 파라미터를 수정합니다:

```py
>>> from transformers import ViTImageProcessor

>>> my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])
>>> print(my_vit_extractor)
ViTImageProcessor {
  "do_normalize": false,
  "do_resize": true,
  "feature_extractor_type": "ViTImageProcessor",
  "image_mean": [
    0.3,
    0.3,
    0.3
  ],
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": "PIL.Image.BOX",
  "size": 224
}
```

## 특성 추출기[[feature-extractor]]

특성 추출기(feature extractor)는 오디오 입력을 처리합니다. 기본 [`~feature_extraction_utils.FeatureExtractionMixin`] 클래스에서 상속되며, 오디오 입력을 처리하기 위해 [`SequenceFeatureExtractor`] 클래스에서 상속할 수도 있습니다.

사용하려면 사용 중인 모델과 연결된 특성 추출기를 생성합니다. 예를 들어, 오디오 분류에 [Wav2Vec2](model_doc/wav2vec2)를 사용하는 경우 기본 [`Wav2Vec2FeatureExtractor`]를 생성합니다:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor()
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 16000
}
```

<Tip>

사용자 지정이 필요하지 않은 경우 `from_pretrained` 메소드를 사용하여 모델의 기본 특성 추출기 ㅁ개변수를 불러 오면 됩니다.

</Tip>

사용자 지정 특성 추출기를 만들려면 [`Wav2Vec2FeatureExtractor`] 매개변수를 수정합니다:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> w2v2_extractor = Wav2Vec2FeatureExtractor(sampling_rate=8000, do_normalize=False)
>>> print(w2v2_extractor)
Wav2Vec2FeatureExtractor {
  "do_normalize": false,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": false,
  "sampling_rate": 8000
}
```


## 프로세서[[processor]]

멀티모달 작업을 지원하는 모델의 경우, 🤗 Transformers는 특성 추출기 및 토크나이저와 같은 처리 클래스를 단일 객체로 편리하게 래핑하는 프로세서 클래스를 제공합니다. 예를 들어, 자동 음성 인식 작업(Automatic Speech Recognition task (ASR))에 [`Wav2Vec2Processor`]를 사용한다고 가정해 보겠습니다. 자동 음성 인식 작업은 오디오를 텍스트로 변환하므로 특성 추출기와 토크나이저가 필요합니다.

오디오 입력을 처리할 특성 추출기를 만듭니다:

```py
>>> from transformers import Wav2Vec2FeatureExtractor

>>> feature_extractor = Wav2Vec2FeatureExtractor(padding_value=1.0, do_normalize=True)
```

텍스트 입력을 처리할 토크나이저를 만듭니다:

```py
>>> from transformers import Wav2Vec2CTCTokenizer

>>> tokenizer = Wav2Vec2CTCTokenizer(vocab_file="my_vocab_file.txt")
```

[`Wav2Vec2Processor`]에서 특성 추출기와 토크나이저를 결합합니다:

```py
>>> from transformers import Wav2Vec2Processor

>>> processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

configuration과 모델이라는 두 가지 기본 클래스와 추가 전처리 클래스(토크나이저, 이미지 프로세서, 특성 추출기 또는 프로세서)를 사용하면 🤗 Transformers에서 지원하는 모든 모델을 만들 수 있습니다. 이러한 각 기본 클래스는 구성이 가능하므로 원하는 특정 속성을 사용할 수 있습니다. 학습을 위해 모델을 쉽게 설정하거나 기존의 사전 학습된 모델을 수정하여 미세 조정할 수 있습니다.

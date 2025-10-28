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

# 🤗 Transformers로 할 수 있는 것[[what__transformers_can_do]]

🤗 Transformers는 자연어처리(NLP), 컴퓨터 비전, 오디오 및 음성 처리 작업에 대한 사전훈련된 최첨단 모델 라이브러리입니다. 
이 라이브러리는 트랜스포머 모델뿐만 아니라 컴퓨터 비전 작업을 위한 현대적인 합성곱 신경망과 같은 트랜스포머가 아닌 모델도 포함하고 있습니다. 

스마트폰, 앱, 텔레비전과 같은 오늘날 가장 인기 있는 소비자 제품을 살펴보면, 딥러닝 기술이 그 뒤에 사용되고 있을 확률이 높습니다. 
스마트폰으로 촬영한 사진에서 배경 객체를 제거하고 싶다면 어떻게 할까요? 이는 파놉틱 세그멘테이션 작업의 예입니다(아직 이게 무엇인지 모른다면, 다음 섹션에서 설명하겠습니다!).

이 페이지는 다양한 음성 및 오디오, 컴퓨터 비전, NLP 작업을 🤗 Transformers 라이브러리를 활용하여 다루는 간단한 예제를 3줄의 코드로 제공합니다. 

## 오디오[[audio]]


음성 및 오디오 처리 작업은 다른 모달리티와 약간 다릅니다. 이는 주로 오디오가 연속적인 신호로 입력되기 때문입니다. 
텍스트와 달리 원본 오디오 파형(waveform)은 문장이 단어로 나눠지는 것처럼 깔끔하게 이산적인 묶음으로 나눌 수 없습니다. 
이를 극복하기 위해 원본 오디오 신호는 일정한 간격으로 샘플링됩니다. 해당 간격 내에서 더 많은 샘플을 취할 경우 샘플링률이 높아지며, 오디오는 원본 오디오 소스에 더 가까워집니다.

과거의 접근 방식은 오디오에서 유용한 특징을 추출하기 위해 오디오를 전처리하는 것이었습니다. 
하지만 현재는 원본 오디오 파형을 특성 인코더에 직접 넣어서 오디오 표현(representation)을 추출하는 것이 더 일반적입니다. 
이렇게 하면 전처리 단계가 단순해지고 모델이 가장 중요한 특징을 학습할 수 있습니다.

### 오디오 분류[[audio_classification]]


오디오 분류는 오디오 데이터에 미리 정의된 클래스 집합의 레이블을 지정하는 작업입니다. 이는 많은 구체적인 응용 프로그램을 포함한 넓은 범주입니다.

일부 예시는 다음과 같습니다:

* 음향 장면 분류: 오디오에 장면 레이블("사무실", "해변", "경기장")을 지정합니다.
* 음향 이벤트 감지: 오디오에 소리 이벤트 레이블("차 경적", "고래 울음소리", "유리 파손")을 지정합니다.
* 태깅: 여러 가지 소리(새 지저귐, 회의에서의 화자 식별)가 포함된 오디오에 레이블을 지정합니다.
* 음악 분류: 음악에 장르 레이블("메탈", "힙합", "컨트리")을 지정합니다.

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### 자동 음성 인식[[automatic_speech_recognition]]


자동 음성 인식(ASR)은 음성을 텍스트로 변환하는 작업입니다. 
음성은 인간의 자연스러운 의사소통 형태이기 때문에 ASR은 가장 일반적인 오디오 작업 중 하나입니다. 
오늘날 ASR 시스템은 스피커, 전화 및 자동차와 같은 "스마트" 기술 제품에 내장되어 있습니다. 
우리는 가상 비서에게 음악 재생, 알림 설정 및 날씨 정보를 요청할 수 있습니다.

하지만 트랜스포머 아키텍처가 해결하는 데 도움을 준 핵심 도전 과제 중 하나는 양이 데이터 양이 적은 언어(low-resource language)에 대한 것입니다. 대량의 음성 데이터로 사전 훈련한 후 데이터 양이 적은 언어에서 레이블이 지정된 음성 데이터 1시간만으로 모델을 미세 조정하면 이전의 100배 많은 레이블이 지정된 데이터로 훈련된 ASR 시스템보다 훨씬 더 높은 품질의 결과를 얻을 수 있습니다. 
```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## 컴퓨터 비전[[computer_vision]]

컴퓨터 비전 작업 중 가장 초기의 성공적인 작업 중 하나는 [합성곱 신경망(CNN)](glossary#convolution)을 사용하여 우편번호 숫자 이미지를 인식하는 것이었습니다. 이미지는 픽셀로 구성되어 있으며 각 픽셀은 숫자 값으로 표현됩니다. 이로써 이미지를 픽셀 값의 행렬로 나타내는 것이 쉬워집니다. 특정한 픽셀 값의 조합은 이미지의 색상을 의미합니다.

컴퓨터 비전 작업은 일반적으로 다음 두 가지 방법으로 접근 가능합니다:

1. 합성곱을 사용하여 이미지의 낮은 수준 특징에서 높은 수준의 추상적인 요소까지 계층적으로 학습합니다.

2. 이미지를 패치로 나누고 트랜스포머를 사용하여 점진적으로 각 이미지 패치가 서로 어떠한 방식으로 연관되어 이미지를 형성하는지 학습합니다. `CNN`에서 선호하는 상향식 접근법과는 달리, 이 방식은 흐릿한 이미지로 초안을 그리고 점진적으로 선명한 이미지로 만들어가는 것과 유사합니다.

### 이미지 분류[[image_classification]]


이미지 분류는 한 개의 전체 이미지에 미리 정의된 클래스 집합의 레이블을 지정하는 작업입니다. 

대부분의 분류 작업과 마찬가지로, 이미지 분류에는 다양한 실용적인 용도가 있으며, 일부 예시는 다음과 같습니다:


* 의료: 질병을 감지하거나 환자 건강을 모니터링하기 위해 의료 이미지에 레이블을 지정합니다.
* 환경: 위성 이미지를 분류하여 산림 벌채를 감시하고 야생 지역 관리를 위한 정보를 제공하거나 산불을 감지합니다. 
* 농업: 작물 이미지를 분류하여 식물 건강을 확인하거나 위성 이미지를 분류하여 토지 이용 관찰에 사용합니다.
* 생태학: 동물이나 식물 종 이미지를 분류하여 야생 동물 개체군을 조사하거나 멸종 위기에 처한 종을 추적합니다.

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### 객체 탐지[[object_detection]]


이미지 분류와 달리 객체 탐지는 이미지 내에서 여러 객체를 식별하고 바운딩 박스로 정의된 객체의 위치를 파악합니다. 

객체 탐지의 몇 가지 응용 예시는 다음과 같습니다:

* 자율 주행 차량: 다른 차량, 보행자 및 신호등과 같은 일상적인 교통 객체를 감지합니다.
* 원격 감지: 재난 모니터링, 도시 계획 및 기상 예측 등을 수행합니다.
* 결함 탐지: 건물의 균열이나 구조적 손상, 제조 결함 등을 탐지합니다.


```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### 이미지 분할[[image_segmentation]]


이미지 분할은 픽셀 차원의 작업으로, 이미지 내의 모든 픽셀을 클래스에 할당합니다. 이는 객체 탐지와 다릅니다. 객체 탐지는 바운딩 박스를 사용하여 이미지 내의 객체를 레이블링하고 예측하는 반면, 분할은 더 세분화된 작업입니다. 분할은 픽셀 수준에서 객체를 감지할 수 있습니다. 

이미지 분할에는 여러 유형이 있습니다:

* 인스턴스 분할: 개체의 클래스를 레이블링하는 것 외에도, 개체의 각 구분된 인스턴스에도 레이블을 지정합니다 ("개-1", "개-2" 등).
* 파놉틱 분할: 의미적 분할과 인스턴스 분할의 조합입니다. 각 픽셀을 의미적 클래스로 레이블링하는 **동시에** 개체의 각각 구분된 인스턴스로도 레이블을 지정합니다.

분할 작업은 자율 주행 차량에서 유용하며, 주변 환경의 픽셀 수준 지도를 생성하여 보행자와 다른 차량 주변에서 안전하게 탐색할 수 있습니다. 또한 의료 영상에서도 유용합니다. 분할 작업이 픽셀 수준에서 객체를 감지할 수 있기 때문에 비정상적인 세포나 장기의 특징을 식별하는 데 도움이 될 수 있습니다. 이미지 분할은 의류 가상 시착이나 카메라를 통해 실제 세계에 가상 개체를 덧씌워 증강 현실 경험을 만드는 등 전자 상거래 분야에서도 사용될 수 있습니다.

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### 깊이 추정[[depth_estimation]]

깊이 추정은 카메라로부터 이미지 내부의 각 픽셀의 거리를 예측합니다. 이 컴퓨터 비전 작업은 특히 장면 이해와 재구성에 중요합니다. 예를 들어, 자율 주행 차량은 보행자, 교통 표지판 및 다른 차량과 같은 객체와의 거리를 이해하여 장애물과 충돌을 피해야 합니다. 깊이 정보는 또한 2D 이미지에서 3D 표현을 구성하는 데 도움이 되며 생물학적 구조나 건물의 고품질 3D 표현을 생성하는 데 사용될 수 있습니다.

깊이 추정에는 두 가지 접근 방식이 있습니다:

* 스테레오: 약간 다른 각도에서 촬영된 동일한 이미지 두 장을 비교하여 깊이를 추정합니다.
* 단안: 단일 이미지에서 깊이를 추정합니다.


```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## 자연어처리[[natural_language_processing]]

텍스트는 인간이 의사 소통하는 자연스러운 방식 중 하나이기 때문에 자연어처리 역시 가장 일반적인 작업 유형 중 하나입니다. 모델이 인식하는 형식으로 텍스트를 변환하려면 토큰화해야 합니다. 이는 텍스트 시퀀스를 개별 단어 또는 하위 단어(토큰)로 분할한 다음 이러한 토큰을 숫자로 변환하는 것을 의미합니다. 결과적으로 텍스트 시퀀스를 숫자 시퀀스로 표현할 수 있으며, 숫자 시퀀스를 다양한 자연어처리 작업을 해결하기 위한 모델에 입력할 수 있습니다!

### 텍스트 분류[[text_classification]]

다른 모달리티에서의 분류 작업과 마찬가지로 텍스트 분류는 미리 정의된 클래스 집합에서 텍스트 시퀀스(문장 수준, 단락 또는 문서 등)에 레이블을 지정합니다. 텍스트 분류에는 다양한 실용적인 응용 사례가 있으며, 일부 예시는 다음과 같습니다:

* 감성 분석: 텍스트를 `긍정` 또는 `부정`과 같은 어떤 극성에 따라 레이블링하여 정치, 금융, 마케팅과 같은 분야에서 의사 결정에 정보를 제공하고 지원할 수 있습니다.
* 콘텐츠 분류: 텍스트를 주제에 따라 레이블링(날씨, 스포츠, 금융 등)하여 뉴스 및 소셜 미디어 피드에서 정보를 구성하고 필터링하는 데 도움이 될 수 있습니다.

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### 토큰 분류[[token_classification]]

모든 자연어처리 작업에서는 텍스트가 개별 단어나 하위 단어로 분리되어 전처리됩니다. 분리된 단어를 [토큰](/glossary#token)이라고 합니다. 토큰 분류는 각 토큰에 미리 정의된 클래스 집합의 레이블을 할당합니다.

토큰 분류의 두 가지 일반적인 유형은 다음과 같습니다:

* 개체명 인식 (NER): 토큰을 조직, 인물, 위치 또는 날짜와 같은 개체 범주에 따라 레이블링합니다. NER은 특히 유전체학적인 환경에서 유전자, 단백질 및 약물 이름에 레이블을 지정하는 데 널리 사용됩니다.
* 품사 태깅 (POS): 명사, 동사, 형용사와 같은 품사에 따라 토큰에 레이블을 할당합니다. POS는 번역 시스템이 동일한 단어가 문법적으로 어떻게 다른지 이해하는 데 도움이 됩니다 (명사로 사용되는 "bank(은행)"과 동사로 사용되는 "bank(예금을 예치하다)"과 같은 경우).


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### 질의응답[[question_answering]]

질의응답은 또 하나의 토큰 차원의 작업으로, 문맥이 있을 때(개방형 도메인)와 문맥이 없을 때(폐쇄형 도메인) 질문에 대한 답변을 반환합니다. 이 작업은 가상 비서에게 식당이 영업 중인지와 같은 질문을 할 때마다 발생할 수 있습니다. 고객 지원 또는 기술 지원을 제공하거나 검색 엔진이 요청한 정보를 검색하는 데 도움을 줄 수 있습니다.

질문 답변에는 일반적으로 두 가지 유형이 있습니다:

* 추출형: 질문과 문맥이 주어졌을 때, 모델이 주어진 문맥의 일부에서 가져온 텍스트의 범위를 답변으로 합니다.
* 생성형: 질문과 문맥이 주어졌을 때, 주어진 문맥을 통해 답변을 생성합니다. 이 접근 방식은 [`QuestionAnsweringPipeline`] 대신 [`Text2TextGenerationPipeline`]을 통해 처리됩니다.

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### 요약[[summarization]]

요약은 원본 문서의 의미를 최대한 보존하면서 긴 문서를 짧은 문서로 만드는 작업입니다. 요약은 `sequence-to-sequence` 작업입니다. 입력보다 짧은 텍스트 시퀀스를 출력합니다. 요약 작업은 독자가 장문 문서들의 주요 포인트를 빠르게 이해하는 데 도움을 줄 수 있습니다. 입법안, 법률 및 금융 문서, 특허 및 과학 논문은 요약 작업이 독자의 시간을 절약하고 독서 보조 도구로 사용될 수 있는 몇 가지 예시입니다.

질문 답변과 마찬가지로 요약에는 두 가지 유형이 있습니다:

* 추출형: 원본 텍스트에서 가장 중요한 문장을 식별하고 추출합니다.
* 생성형: 원본 텍스트에서 목표 요약을 생성합니다. 입력 문서에 없는 새로운 단어를 포함할 수도 있습니다. [`SummarizationPipeline`]은 생성형 접근 방식을 사용합니다.

```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### 번역[[translation]]

번역은 한 언어로 된 텍스트 시퀀스를 다른 언어로 변환하는 작업입니다. 이는 서로 다른 배경을 가진 사람들이 서로 소통하는 데 도움을 주는 중요한 역할을 합니다. 더 넓은 대중에게 콘텐츠를 번역하여 전달하거나, 새로운 언어를 배우는 데 도움이 되는 학습 도구가 될 수도 있습니다. 요약과 마찬가지로, 번역은 `sequence-to-sequence` 작업입니다. 즉, 모델은 입력 시퀀스를 받아서 출력이 되는 목표 시퀀스를 반환합니다.

초기의 번역 모델은 대부분 단일 언어로 이루어져 있었지만, 최근에는 많은 언어 쌍 간에 번역을 수행할 수 있는 다중 언어 모델에 대한 관심이 높아지고 있습니다.

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### 언어 모델링[[language_modeling]]

언어 모델링은 텍스트 시퀀스에서 단어를 예측하는 작업입니다. 사전 훈련된 언어 모델은 많은 다른 하위 작업에 따라 미세 조정될 수 있기 때문에 매우 인기 있는 자연어처리 작업이 되었습니다. 최근에는 제로 샷(zero-shot) 또는 퓨 샷(few-shot) 학습이 가능한 대규모 언어 모델(Large Language Models, LLM)에 대한 많은 관심이 발생하고 있습니다. 이는 모델이 명시적으로 훈련되지 않은 작업도 해결할 수 있다는 것을 의미합니다! 언어 모델은 유창하고 설득력 있는 텍스트를 생성하는 데 사용될 수 있지만, 텍스트가 항상 정확하지는 않을 수 있으므로 주의가 필요합니다.

언어 모델링에는 두 가지 유형이 있습니다:

* 인과적 언어 모델링: 이 모델의 목적은 시퀀스에서 다음 토큰을 예측하는 것이며, 미래 토큰이 마스킹 됩니다.
    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

* 마스킹된 언어 모델링: 이 모델의 목적은 시퀀스 내의 마스킹된 토큰을 예측하는 것이며, 시퀀스 내의 모든 토큰에 대한 접근이 제공됩니다.
    
    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
    ```

이 페이지를 통해 각 모달리티의 다양한 작업 유형과 각 작업의 실용적 중요성에 대해 추가적인 배경 정보를 얻으셨기를 바랍니다. 다음 [섹션](tasks_explained)에서는 🤗 Transformer가 이러한 작업을 해결하는 **방법**에 대해 알아보실 수 있습니다.
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

# 객관식 문제[[multiple-choice]]

[[open-in-colab]]

객관식 과제는 문맥과 함께 여러 개의 후보 답변이 제공되고 모델이 정답을 선택하도록 학습된다는 점을 제외하면 질의응답과 유사합니다.

진행하는 방법은 아래와 같습니다:

1. [SWAG](https://huggingface.co/datasets/swag) 데이터 세트의 'regular' 구성으로 [BERT](https://huggingface.co/google-bert/bert-base-uncased)를 미세 조정하여 여러 옵션과 일부 컨텍스트가 주어졌을 때 가장 적합한 답을 선택합니다.
2. 추론에 미세 조정된 모델을 사용합니다.

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

모델을 업로드하고 커뮤니티와 공유할 수 있도록 허깅페이스 계정에 로그인하는 것이 좋습니다. 메시지가 표시되면 토큰을 입력하여 로그인합니다:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## SWAG 데이터 세트 가져오기[[load-swag-dataset]]

먼저 🤗 Datasets  라이브러리에서 SWAG 데이터셋의 '일반' 구성을 가져옵니다:

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

이제 데이터를 살펴봅니다:

```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

여기에는 많은 필드가 있는 것처럼 보이지만 실제로는 매우 간단합니다:

- `sent1` 및 `sent2`: 이 필드는 문장이 어떻게 시작되는지 보여주며, 이 두 필드를 합치면 `시작 구절(startphrase)` 필드가 됩니다.
- `종료 구절(ending)`: 문장이 어떻게 끝날 수 있는지에 대한 가능한 종료 구절를 제시하지만 그 중 하나만 정답입니다.
- `레이블(label)`: 올바른 문장 종료 구절을 식별합니다.

## 전처리[[preprocess]]

다음 단계는 문장의 시작과 네 가지 가능한 구절을 처리하기 위해 BERT 토크나이저를 불러옵니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

생성하려는 전처리 함수는 다음과 같아야 합니다:

1. `sent1` 필드를 네 개 복사한 다음 각각을 `sent2`와 결합하여 문장이 시작되는 방식을 재현합니다.
2. `sent2`를 네 가지 가능한 문장 구절 각각과 결합합니다.
3. 이 두 목록을 토큰화할 수 있도록 평탄화(flatten)하고, 각 예제에 해당하는 `input_ids`, `attention_mask` 및 `labels` 필드를 갖도록 다차원화(unflatten) 합니다.

```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

전체 데이터 집합에 전처리 기능을 적용하려면 🤗 Datasets [`~datasets.Dataset.map`] 메소드를 사용합니다. `batched=True`를 설정하여 데이터 집합의 여러 요소를 한 번에 처리하면 `map` 함수의 속도를 높일 수 있습니다:

```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

[`DataCollatorForMultipleChoice`]는 모든 모델 입력을 평탄화하고 패딩을 적용하며 그 결과를 결과를 다차원화합니다:
```py
>>> from transformers import DataCollatorForMultipleChoice
>>> collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
```

## 평가 하기[[evaluate]]

훈련 중에 메트릭을 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다. 🤗[Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하여 평가 방법을 빠르게 가져올 수 있습니다. 이 작업에서는 [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) 지표를 가져옵니다(🤗 Evaluate [둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하여 지표를 가져오고 계산하는 방법에 대해 자세히 알아보세요):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

그리고 예측과 레이블을 [`~evaluate.EvaluationModule.compute`]에 전달하여 정확도를 계산하는 함수를 만듭니다:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 훈련을 설정할 때 이 함수로 돌아가게 됩니다.

## 훈련 하기[[train]]

<frameworkcontent>
<pt>
<Tip>

[`Trainer`]로 모델을 미세 조정하는 데 익숙하지 않다면 기본 튜토리얼 [여기](../training#train-with-pytorch-trainer)를 살펴보세요!

</Tip>

이제 모델 훈련을 시작할 준비가 되었습니다! [`AutoModelForMultipleChoice`]로 BERT를 로드합니다:

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

이제 세 단계만 남았습니다:

1. 훈련 하이퍼파라미터를 [`TrainingArguments`]에 정의합니다. 유일한 필수 매개변수는 모델을 저장할 위치를 지정하는 `output_dir`입니다. `push_to_hub=True`를 설정하여 이 모델을 허브에 푸시합니다(모델을 업로드하려면 허깅 페이스에 로그인해야 합니다). 각 에폭이 끝날 때마다 [`Trainer`]가 정확도를 평가하고 훈련 체크포인트를 저장합니다.
2. 모델, 데이터 세트, 토크나이저, 데이터 콜레이터, `compute_metrics` 함수와 함께 훈련 인자를 [`Trainer`]에 전달합니다.
3. [`~Trainer.train`]을 사용하여 모델을 미세 조정합니다.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     processing_class=tokenizer,
...     data_collator=collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

훈련이 완료되면 모든 사람이 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 모델을 허브에 공유하세요:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

Keras로 모델을 미세 조정하는 데 익숙하지 않다면 기본 튜토리얼 [여기](../training#train-a-tensorflow-model-with-keras)를 살펴보시기 바랍니다!

</Tip>
TensorFlow에서 모델을 미세 조정하려면 최적화 함수, 학습률 스케쥴 및 몇 가지 학습 하이퍼파라미터를 설정하는 것부터 시작하세요:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

그리고 [`TFAutoModelForMultipleChoice`]로 BERT를 가져올 수 있습니다:

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]을 사용하여 데이터 세트를 `tf.data.Dataset` 형식으로 변환합니다:

```py
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"],
...     shuffle=False,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)을 사용하여 훈련 모델을 구성합니다:

```py
>>> model.compile(optimizer=optimizer)
```

훈련을 시작하기 전에 설정해야 할 마지막 두 가지는 예측의 정확도를 계산하고 모델을 허브로 푸시하는 방법을 제공하는 것입니다. 이 두 가지 작업은 모두 [Keras 콜백](../main_classes/keras_callbacks)을 사용하여 수행할 수 있습니다.

`compute_metrics`함수를 [`~transformers.KerasMetricCallback`]에 전달하세요:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

모델과 토크나이저를 업로드할 위치를 [`~transformers.PushToHubCallback`]에서 지정하세요:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

그리고 콜백을 함께 묶습니다:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

이제 모델 훈련을 시작합니다! 훈련 및 검증 데이터 세트, 에폭 수, 콜백을 사용하여 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)을 호출하고 모델을 미세 조정합니다:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

훈련이 완료되면 모델이 자동으로 허브에 업로드되어 누구나 사용할 수 있습니다!
</tf>
</frameworkcontent>


<Tip>

객관식 모델을 미세 조정하는 방법에 대한 보다 심층적인 예는 아래 문서를 참조하세요.
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

</Tip>

## 추론 하기[[inference]]

이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

텍스트와 두 개의 후보 답안을 작성합니다:

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

<frameworkcontent>
<pt>
각 프롬프트와 후보 답변 쌍을 토큰화하여 PyTorch 텐서를 반환합니다. 또한 `labels`을 생성해야 합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

입력과 레이블을 모델에 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

가장 높은 확률을 가진 클래스를 가져옵니다:

```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```
</pt>
<tf>
각 프롬프트와 후보 답안 쌍을 토큰화하여 텐서플로 텐서를 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

모델에 입력을 전달하고 `logits`를 반환합니다:

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

가장 높은 확률을 가진 클래스를 가져옵니다:

```py
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```
</tf>
</frameworkcontent>

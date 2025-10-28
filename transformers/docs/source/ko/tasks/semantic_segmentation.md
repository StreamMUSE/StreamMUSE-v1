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

# 의미적 분할(Semantic segmentation)[[semantic-segmentation]]

[[open-in-colab]]

<Youtube id="dKE8SIt9C-w"/>

의미적 분할(semantic segmentation)은 이미지의 각 픽셀에 레이블 또는 클래스를 할당합니다. 분할(segmentation)에는 여러 종류가 있으며, 의미적 분할의 경우 동일한 물체의 고유 인스턴스를 구분하지 않습니다. 두 물체 모두 동일한 레이블이 지정됩니다(예시로, "car-1" 과 "car-2" 대신 "car"로 지정합니다).
실생활에서 흔히 볼 수 있는 의미적 분할의 적용 사례로는 보행자와 중요한 교통 정보를 식별하는 자율 주행 자동차 학습, 의료 이미지의 세포와 이상 징후 식별, 그리고 위성 이미지의 환경 변화 모니터링등이 있습니다.

이번 가이드에서 배울 내용은 다음과 같습니다:

1. [SceneParse150](https://huggingface.co/datasets/scene_parse_150) 데이터 세트를 이용해 [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) 미세 조정하기.
2. 미세 조정된 모델을 추론에 사용하기.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/image-segmentation)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 모든 라이브러리가 설치되었는지 확인하세요:

```bash
pip install -q datasets transformers evaluate
```
커뮤니티에 모델을 업로드하고 공유할 수 있도록 Hugging Face 계정에 로그인하는 것을 권장합니다. 프롬프트가 나타나면 토큰을 입력하여 로그인하세요:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## SceneParse150 데이터 세트 불러오기[[load-sceneparse150-dataset]]

🤗 Datasets 라이브러리에서 SceneParse150 데이터 세트의 더 작은 부분 집합을 가져오는 것으로 시작합니다. 이렇게 하면 데이터 세트 전체에 대한 훈련에 많은 시간을 할애하기 전에 실험을 통해 모든 것이 제대로 작동하는지 확인할 수 있습니다.

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

데이터 세트의 `train`을 [`~datasets.Dataset.train_test_split`] 메소드를 사용하여 훈련 및 테스트 세트로 분할하세요:

```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

그리고 예시를 살펴보세요:

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}
```

- `image`: 장면의 PIL 이미지입니다.
- `annotation`: 분할 지도(segmentation map)의 PIL 이미지입니다. 모델의 타겟이기도 합니다.
- `scene_category`: "주방" 또는 "사무실"과 같이 이미지 장면을 설명하는 카테고리 ID입니다. 이 가이드에서는 둘 다 PIL 이미지인 `image`와 `annotation`만을 사용합니다.

나중에 모델을 설정할 때 유용하게 사용할 수 있도록 레이블 ID를 레이블 클래스에 매핑하는 사전도 만들고 싶을 것입니다. Hub에서 매핑을 다운로드하고 `id2label` 및 `label2id` 사전을 만드세요:

```py
>>> import json
>>> from pathlib import Path
>>> from huggingface_hub import hf_hub_download

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

## 전처리하기[[preprocess]

다음 단계는 모델에 사용할 이미지와 주석을 준비하기 위해 SegFormer 이미지 프로세서를 불러오는 것입니다. 우리가 사용하는 데이터 세트와 같은 일부 데이터 세트는 배경 클래스로 제로 인덱스를 사용합니다. 하지만 배경 클래스는 150개의 클래스에 실제로는 포함되지 않기 때문에 `do_reduce_labels=True` 를 설정해 모든 레이블에서 배경 클래스를 제거해야 합니다. 제로 인덱스는 `255`로 대체되므로 SegFormer의 손실 함수에서 무시됩니다:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
```

<frameworkcontent>
<pt>

이미지 데이터 세트에 데이터 증강을 적용하여 과적합에 대해 모델을 보다 강건하게 만드는 것이 일반적입니다. 이 가이드에서는 [torchvision](https://pytorch.org/vision/stable/index.html)의 [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html)를 사용하여 이미지의 색상 속성을 임의로 변경합니다. 하지만, 자신이 원하는 이미지 라이브러리를 사용할 수도 있습니다.

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

이제 모델에 사용할 이미지와 주석을 준비하기 위해 두 개의 전처리 함수를 만듭니다. 이 함수들은 이미지를 `pixel_values`로, 주석을 `labels`로 변환합니다. 훈련 세트의 경우 이미지 프로세서에 이미지를 제공하기 전에 `jitter`를 적용합니다. 테스트 세트의 경우 이미지 프로세서는 `images`를 자르고 정규화하며, 테스트 중에는 데이터 증강이 적용되지 않으므로 `labels`만 자릅니다.

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

모든 데이터 세트에 `jitter`를 적용하려면, 🤗 Datasets [`~datasets.Dataset.set_transform`] 함수를 사용하세요. 즉시 변환이 적용되기 때문에 더 빠르고 디스크 공간을 덜 차지합니다:

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>

이미지 데이터 세트에 데이터 증강을 적용하여 과적합에 대해 모델을 보다 강건하게 만드는 것이 일반적입니다. 이 가이드에서는 [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image)를 사용하여 이미지의 색상 속성을 임의로 변경합니다. 하지만, 자신이 원하는 이미지 라이브러리를 사용할 수도 있습니다.

별개의 두 변환 함수를 정의합니다:
- 이미지 증강을 포함하는 학습 데이터 변환
- 🤗 Transformers의 컴퓨터 비전 모델은 채널 우선 레이아웃을 기대하기 때문에, 이미지만 바꾸는 검증 데이터 변환

```py
>>> import tensorflow as tf


>>> def aug_transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.image.random_brightness(image, 0.25)
...     image = tf.image.random_contrast(image, 0.5, 2.0)
...     image = tf.image.random_saturation(image, 0.75, 1.25)
...     image = tf.image.random_hue(image, 0.1)
...     image = tf.transpose(image, (2, 0, 1))
...     return image


>>> def transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.transpose(image, (2, 0, 1))
...     return image
```

그런 다음 모델을 위해 두 개의 전처리 함수를 만들어 이미지 및 주석 배치를 준비합니다. 이 함수들은 이미지 변환을 적용하고 이전에 로드한 `image_processor`를 사용하여 이미지를 `pixel_values`로, 주석을 `label`로 변환합니다. `ImageProcessor` 는 이미지의 크기 조정과 정규화도 처리합니다.

```py
>>> def train_transforms(example_batch):
...     images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

전체 데이터 집합에 전처리 변환을 적용하려면 🤗 Datasets [`~datasets.Dataset.set_transform`] 함수를 사용하세요.
즉시 변환이 적용되기 때문에 더 빠르고 디스크 공간을 덜 차지합니다:

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```
</tf>
</frameworkcontent>

## 평가하기[[evaluate]]

훈련 중에 메트릭을 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하여 평가 방법을 빠르게 로드할 수 있습니다. 이 태스크에서는 [mean Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy) (IoU) 메트릭을 로드하세요 (메트릭을 로드하고 계산하는 방법에 대해 자세히 알아보려면 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)를 살펴보세요).

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

그런 다음 메트릭을 [`~evaluate.EvaluationModule.compute`]하는 함수를 만듭니다. 예측을 먼저 로짓으로 변환한 다음, 레이블의 크기에 맞게 모양을 다시 지정해야 [`~evaluate.EvaluationModule.compute`]를 호출할 수 있습니다:

<frameworkcontent>
<pt>

```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if isinstance(value, np.ndarray):
...                 metrics[key] = value.tolist()
...         return metrics
```

</pt>
</frameworkcontent>


<frameworkcontent>
<tf>

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     logits = tf.transpose(logits, perm=[0, 2, 3, 1])
...     logits_resized = tf.image.resize(
...         logits,
...         size=tf.shape(labels)[1:],
...         method="bilinear",
...     )

...     pred_labels = tf.argmax(logits_resized, axis=-1)
...     metrics = metric.compute(
...         predictions=pred_labels,
...         references=labels,
...         num_labels=num_labels,
...         ignore_index=-1,
...         reduce_labels=image_processor.do_reduce_labels,
...     )

...     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
...     per_category_iou = metrics.pop("per_category_iou").tolist()

...     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
...     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
...     return {"val_" + k: v for k, v in metrics.items()}
```

</tf>
</frameworkcontent>

이제 `compute_metrics` 함수를 사용할 준비가 되었습니다. 트레이닝을 설정할 때 이 함수로 돌아가게 됩니다.

## 학습하기[[train]]
<frameworkcontent>
<pt>
<Tip>

만약 [`Trainer`]를 사용해 모델을 미세 조정하는 것에 익숙하지 않다면, [여기](../training#finetune-with-trainer)에서 기본 튜토리얼을 살펴보세요!

</Tip>

이제 모델 학습을 시작할 준비가 되었습니다! [`AutoModelForSemanticSegmentation`]로 SegFormer를 불러오고, 모델에 레이블 ID와 레이블 클래스 간의 매핑을 전달합니다:

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

이제 세 단계만 남았습니다:

1. 학습 하이퍼파라미터를 [`TrainingArguments`]에 정의합니다. `image` 열이 삭제되기 때문에 사용하지 않는 열을 제거하지 않는 것이 중요합니다. `image` 열이 없으면 `pixel_values`을 생성할 수 없습니다. 이런 경우를 방지하려면 `remove_unused_columns=False`로 설정하세요! 유일하게 필요한 다른 매개변수는 모델을 저장할 위치를 지정하는 `output_dir`입니다. `push_to_hub=True`를 설정하여 이 모델을 Hub에 푸시합니다(모델을 업로드하려면 Hugging Face에 로그인해야 합니다). 각 에포크가 끝날 때마다 [`Trainer`]가 IoU 메트릭을 평가하고 학습 체크포인트를 저장합니다.
2. 모델, 데이터 세트, 토크나이저, 데이터 콜레이터, `compute_metrics` 함수와 함께 학습 인자를 [`Trainer`]에 전달하세요.
3. 모델을 미세 조정하기 위해 [`~Trainer.train`]를 호출하세요.

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     eval_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```
학습이 완료되면, 누구나 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메서드를 사용해 Hub에 모델을 공유하세요:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
<Tip>

Keras로 모델을 미세 조정하는 데 익숙하지 않은 경우, 먼저 [기본 튜토리얼](../training#train-a-tensorflow-model-with-keras)을 확인해보세요!

</Tip>

TensorFlow에서 모델을 미세 조정하려면 다음 단계를 따르세요:
1. 학습 하이퍼파라미터를 정의하고 옵티마이저와 학습률 스케쥴러를 설정하세요.
2. 사전 학습된 모델을 인스턴스화하세요.
3. 🤗 Dataset을 `tf.data.Dataset`로 변환하세요.
4. 모델을 컴파일하세요.
5. 콜백을 추가하여 메트릭을 계산하고 🤗 Hub에 모델을 업로드하세요.
6. `fit()` 메서드를 사용하여 훈련을 실행하세요.

하이퍼파라미터, 옵티마이저, 학습률 스케쥴러를 정의하는 것으로 시작하세요:

```py
>>> from transformers import create_optimizer

>>> batch_size = 2
>>> num_epochs = 50
>>> num_train_steps = len(train_ds) * num_epochs
>>> learning_rate = 6e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

그런 다음 레이블 매핑과 함께 [`TFAutoModelForSemanticSegmentation`]을 사용하여 SegFormer를 불러오고 옵티마이저로 컴파일합니다. 트랜스포머 모델은 모두 디폴트로 태스크 관련 손실 함수가 있으므로 원치 않으면 지정할 필요가 없습니다:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
>>> model.compile(optimizer=optimizer)  # 손실 함수 인자가 없습니다!
```

[`~datasets.Dataset.to_tf_dataset`] 와 [`DefaultDataCollator`]를 사용해 데이터 세트를 `tf.data.Dataset` 포맷으로 변환하세요:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")

>>> tf_train_dataset = train_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_eval_dataset = test_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

예측으로 정확도를 계산하고 모델을 🤗 Hub로 푸시하려면 [Keras callbacks](../main_classes/keras_callbacks)를 사용하세요. `compute_metrics` 함수를 [`KerasMetricCallback`]에 전달하고, 모델 업로드를 위해 [`PushToHubCallback`]를 사용하세요:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
...     metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
... )

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

이제 모델을 훈련할 준비가 되었습니다! 훈련 및 검증 데이터 세트, 에포크 수와 함께 `fit()`을 호출하고, 콜백을 사용하여 모델을 미세 조정합니다:

```py
>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     callbacks=callbacks,
...     epochs=num_epochs,
... )
```

축하합니다! 모델을 미세 조정하고 🤗 Hub에 공유했습니다. 이제 추론에 사용할 수 있습니다!

</tf>
</frameworkcontent>


## 추론하기[[inference]]

이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

추론할 이미지를 로드하세요:

```py
>>> image = ds[0]["image"]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="Image of bedroom"/>
</div>

<frameworkcontent>
<pt>

추론을 위해 미세 조정한 모델을 시험해 보는 가장 간단한 방법은 [`pipeline`]에서 사용하는 것입니다. 모델을 사용하여 이미지 분할을 위한 `pipeline`을 인스턴스화하고 이미지를 전달합니다:

```py
>>> from transformers import pipeline

>>> segmenter = pipeline("image-segmentation", model="my_awesome_seg_model")
>>> segmenter(image)
[{'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062690>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A50>},
 {'score': None,
  'label': 'floor',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062B50>},
 {'score': None,
  'label': 'ceiling',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A10>},
 {'score': None,
  'label': 'bed ',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E90>},
 {'score': None,
  'label': 'windowpane',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062390>},
 {'score': None,
  'label': 'cabinet',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062550>},
 {'score': None,
  'label': 'chair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062D90>},
 {'score': None,
  'label': 'armchair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E10>}]
```
원하는 경우 `pipeline`의 결과를 수동으로 복제할 수도 있습니다. 이미지 프로세서로 이미지를 처리하고 `pixel_values`을 GPU에 배치합니다:

```py
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 가능하다면 GPU를 사용하고, 그렇지 않다면 CPU를 사용하세요
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

모델에 입력을 전달하고 `logits`를 반환합니다:

```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```
그런 다음 로짓의 크기를 원본 이미지 크기로 다시 조정합니다:

```py
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```

</pt>
</frameworkcontent>

<frameworkcontent>
<tf>
이미지 프로세서를 로드하여 이미지를 전처리하고 입력을 TensorFlow 텐서로 반환합니다:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

모델에 입력을 전달하고 `logits`를 반환합니다:

```py
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

그런 다음 로그를 원본 이미지 크기로 재조정하고 클래스 차원에 argmax를 적용합니다:

```py
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     # `image.size`가 너비와 높이를 반환하기 때문에 `image`의 모양을 반전시킵니다
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

</tf>
</frameworkcontent>

결과를 시각화하려면 [dataset color palette](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51)를 각 클래스를 RGB 값에 매핑하는 `ade_palette()`로 로드합니다. 그런 다음 이미지와 예측된 분할 지도(segmentation map)을 결합하여 구성할 수 있습니다:

```py
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  # BGR로 변환

>>> img = np.array(image) * 0.5 + color_seg * 0.5  # 분할 지도으로 이미지 구성
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>

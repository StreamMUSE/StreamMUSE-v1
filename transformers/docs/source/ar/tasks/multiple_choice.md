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

# الاختيار من متعدد (Multiple choice)

[[open-in-colab]]

مهمة الاختيار من متعدد مشابهة لمهمة الإجابة على الأسئلة، ولكن مع توفير عدة إجابات محتملة مع سياق، ويُدرّب النموذج على تحديد الإجابة الصحيحة.

سيوضح لك هذا الدليل كيفية:

1. ضبط نموذج [BERT](https://huggingface.co/google-bert/bert-base-uncased)  باستخدام الإعداد `regular` لمجموعة بيانات [SWAG](https://huggingface.co/datasets/swag) لاختيار الإجابة الأفضل من بين الخيارات المتعددة المتاحة مع السياق.
2. استخدام النموذج المضبوط للاستدلال.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل الرمز المميز الخاص بك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات SWAG

ابدأ بتحميل تهيئة `regular` لمجموعة بيانات SWAG من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

ثم ألق نظرة على مثال:

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

على الرغم من أن الحقول تبدو كثيرة، إلا أنها في الواقع بسيطة جداً:

- `sent1` و `sent2`: يعرض هذان الحقلان بداية الجملة، وبدمجهما معًا، نحصل على حقل `startphrase`.
- `ending`: يقترح نهاية محتملة للجملة، واحدة منها فقط هي الصحيحة.
- `label`: يحدد نهاية الجملة الصحيحة.

## المعالجة المسبقة (Preprocess)

الخطوة التالية هي استدعاء مُجزئ BERT لمعالجة بدايات الجمل والنهايات الأربع المحتملة:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

تحتاج دالة المعالجة المسبقة التي تريد إنشاءها إلى:

1.  إنشاء أربع نسخ من حقل `sent1` ودمج كل منها مع `sent2` لإعادة إنشاء كيفية بدء الجملة.
2. دمج `sent2` مع كل من نهايات الجمل الأربع المحتملة.
3. تتجميع هاتين القائمتين لتتمكن من تجزئتهما، ثم إعادة ترتيبها بعد ذلك بحيث يكون لكل مثال حقول `input_ids` و `attention_mask` و `labels` مقابلة.


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

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] الخاصة بـ 🤗 Datasets. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

لا يحتوي 🤗 Transformers على مجمع بيانات للاختيار من متعدد، لذلك ستحتاج إلى تكييف [`DataCollatorWithPadding`] لإنشاء دفعة من الأمثلة. من الأكفأ إضافة حشو (padding) ديناميكي للجمل إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الحد الأقصى للطول.

يقوم `DataCollatorForMultipleChoice` بتجميع جميع مدخلات النموذج، ويطبق الحشو، ثم يعيد تجميع النتائج في شكلها الأصلي:

<frameworkcontent>
<pt>

```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import torch

>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="pt",
...         )

...         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
...         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
...         return batch
```
</pt>
<tf>
 
```py
>>> from dataclasses import dataclass
>>> from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
>>> from typing import Optional, Union
>>> import tensorflow as tf

>>> @dataclass
... class DataCollatorForMultipleChoice:
...     """
...     Data collator that will dynamically pad the inputs for multiple choice received.
...     """

...     tokenizer: PreTrainedTokenizerBase
...     padding: Union[bool, str, PaddingStrategy] = True
...     max_length: Optional[int] = None
...     pad_to_multiple_of: Optional[int] = None

...     def __call__(self, features):
...         label_name = "label" if "label" in features[0].keys() else "labels"
...         labels = [feature.pop(label_name) for feature in features]
...         batch_size = len(features)
...         num_choices = len(features[0]["input_ids"])
...         flattened_features = [
...             [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
...         ]
...         flattened_features = sum(flattened_features, [])

...         batch = self.tokenizer.pad(
...             flattened_features,
...             padding=self.padding,
...             max_length=self.max_length,
...             pad_to_multiple_of=self.pad_to_multiple_of,
...             return_tensors="tf",
...         )

...         batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
...         batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
...         return batch
```
</tf>
</frameworkcontent>

## التقييم (Evaluate)

يُفضل غالبًا تضمين مقياس أثناء التدريب لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (انظر إلى [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لـ 🤗 Evaluate لمعرفة المزيد حول كيفية تحميل المقياس وحسابه):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم أنشئ دالة لتمرير التنبؤات والتسميات إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np

>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

دالتك `compute_metrics` جاهزة الآن، وستعود إليها عند إعداد تدريبك.

## التدريب (Train)

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`], فراجع الدرس الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت جاهز لبدء تدريب نموذجك الآن! قم بتحميل BERT باستخدام [`AutoModelForMultipleChoice`]:

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. حدد معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة الوحيدة المطلوبة هي `output_dir` التي تحدد مكان حفظ نموذجك. ستدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب عليك تسجيل الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم الدقة وحفظ نقطة فحص التدريب.
2. مرر معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومُجمِّع البيانات والمعالج ودالة تجميع البيانات ودالة `compute_metrics`.
3. استدعي [`~Trainer.train`] لضبط نموذجك.

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
...     data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك مع Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع الدرس الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة مُحسِّن وجدول معدل التعلم وبعض معلمات التدريب:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

ثم يمكنك تحميل BERT باستخدام [`TFAutoModelForMultipleChoice`]:

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

حوّل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

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

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة مناسبة للمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> model.compile(optimizer=optimizer)  # لا توجد وسيطة خسارة!
```

الخطوتان الأخيرتان قبل بدء التدريب هما: حساب دقة التنبؤات، وتوفير طريقة لرفع النموذج إلى Hub. ويمكن تحقيق ذلك باستخدام [استدعاءات Keras](../main_classes/keras_callbacks)

مرر دالتك `compute_metrics` إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد مكان دفع نموذجك ومعالجك في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

ثم قم بتضمين الاستدعاءات معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت جاهز لبدء تدريب نموذجك! استدعِ[`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق من الصحة وعدد الحقب والاستدعاءات لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للاختيار من متعدد، ألق نظرة على [دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) المقابل.

</Tip>

## الاستدلال  (Inference)

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال!

قم بإنشاء نص واقتراح إجابتين محتملتين:

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

<frameworkcontent>
<pt>
قم بتحليل كل مطالبة وزوج إجابة مرشح وأعد تنسورات PyTorch. يجب عليك أيضًا إنشاء بعض `العلامات`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

مرر مدخلاتك والعلامات إلى النموذج وأرجع`logits`:

```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("username/my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

استخرج الفئة ذات الاحتمالية الأكبر:

```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
0
```
</pt>
<tf>
قم بتحليل كل مطالبة وزوج إجابة مرشح وأعد موترات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

مرر مدخلاتك إلى النموذج وأعد القيم logits:

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("username/my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

استخرج الفئة ذات الاحتمالية الأكبر:

```py
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
0
```
</tf>
</frameworkcontent>

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

# الترجمة(Translation)

[[open-in-colab]]

<Youtube id="1JvfrvZgi6c"/>

الترجمة هي عملية تحويل سلسلة نصية من لغة إلى أخرى. وهي إحدى المهام التي يمكن صياغتها كمسألة تسلسل إلى تسلسل، وهو إطار عمل قوي لإنتاج مخرجات من مدخلات، مثل الترجمة أو التلخيص. تُستخدم أنظمة الترجمة عادةً للترجمة بين نصوص لغات مختلفة، ويمكن استخدامها أيضًا لترجمة الكلام أو لمهام تجمع بين النصوص والكلام، مثل تحويل النص إلى كلام أو تحويل الكلام إلى نص.

سيوضح لك هذا الدليل كيفية:

1. ضبط دقيق لنموذج [T5](https://huggingface.co/google-t5/t5-small) على المجموعة الفرعية الإنجليزية-الفرنسية من مجموعة بيانات [OPUS Books](https://huggingface.co/datasets/opus_books) لترجمة النص الإنجليزي إلى الفرنسية.
2. استخدام النموذج المضبوط بدقة للاستدلال.

<Tip>

لمشاهدة جميع البنى والنسخ المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/translation).

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate sacrebleu
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عند الطلب، أدخل الرمز المميز الخاص بك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات OPUS Books

ابدأ بتحميل المجموعة الفرعية الإنجليزية-الفرنسية من مجموعة بيانات [OPUS Books](https://huggingface.co/datasets/opus_books) من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

قسّم مجموعة البيانات إلى مجموعة تدريب ومجموعة اختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

ثم ألقِ نظرة على مثال:

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`: ترجمة إنجليزية وفرنسية للنص.

## المعالجة المسبقة(Preprocess)

<Youtube id="XAR8jnZZuUs"/>

الخطوة التالية هي تحميل مُجزئ T5 لمعالجة أزواج اللغة الإنجليزية-الفرنسية:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

يجب أن تقوم دالة المعالجة المسبقة التي تُريد إنشاءها بما يلي:

1. إضافة بادئة إلى المُدخل بمُوجه حتى يعرف T5 أن هذه مهمة ترجمة. تتطلب بعض النماذج القادرة على أداء مهام متعددة توجيهًا لمهام مُحددة.
2. تعيين اللغة الهدف (الفرنسية) في معامل `text_target` لضمان معالجة المُجزئ للنص بشكل صحيح. إذا لم تُعيّن `text_target`، فسيُعالج المُجزئ النص على أنه إنجليزي.
3. اقتطاع التسلسلات بحيث لا يزيد طولها عن الحد الأقصى الذي يحدده معامل `max_length`.

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "

>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] من 🤗 Datasets. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

الآن أنشئ دفعة من الأمثلة باستخدام [`DataCollatorForSeq2Seq`]. من الأكثر كفاءة *الحشو الديناميكي* للجمل إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الحد الأقصى للطول.

<frameworkcontent>
<pt>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt>
<tf>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>

## التقييم (Evaluate)

غالباً ما يكون تضمين مقياس أثناء التدريب مفيداً لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، حمّل مقياس [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) (راجع [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لـ 🤗 Evaluate لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

ثم أنشئ دالة تُمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب درجة SacreBLEU:

```py
>>> import numpy as np

>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels

>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

دالة `compute_metrics` الخاصة بك جاهزة الآن، وسوف تعود إليها عند إعداد التدريب.

## التدريب (Train)

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن معتادًا على ضبط دقيق نموذج باستخدام [`Trainer`], فألقِ نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت جاهز لبدء تدريب نموذجك الآن! حمّل T5 باستخدام [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. حدد مُعاملات للتدريب في [`Seq2SeqTrainingArguments`]. المُعامل الوحيدة المطلوبة هي `output_dir` التي تحدد مكان حفظ النموذج الخاص بك. ستقوم بدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب عليك تسجيل الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس SacreBLEU وحفظ نقطة تدقيق التدريب.
2. مرر مُعاملات التدريب إلى [`Seq2SeqTrainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمعالج اللغوي وجامع البيانات ووظيفة `compute_metrics`.
3. نفّذ [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
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

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فألق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة مُحسِّن وجدول معدل تعلم وبعض المعلمات الفائقة للتدريب:

```py
>>> from transformers import AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

ثم يمكنك تحميل T5 باستخدام [`TFAutoModelForSeq2SeqLM`]:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

حوّل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_books["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_books["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتكوين النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة إلا إذا كنت ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

آخر شيئين يجب إعدادهما قبل بدء التدريب هما حساب مقياس SacreBLEU من التوقعات، وتوفير طريقة لدفع نموذجك إلى Hub. يتم كلاهما باستخدام [استدعاءات Keras](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
```

حدد مكان دفع نموذجك ومعالجك اللغوي في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_opus_books_model",
...     tokenizer=tokenizer,
... )
```

ثم اجمع استدعاءاتك معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت جاهز لبدء تدريب نموذجك! اتصل بـ [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق من الصحة وعدد الحقب واستدعاءاتك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر تعمقًا لكيفية ضبط نموذج للترجمة، ألق نظرة على [دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb) المقابل
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb).

</Tip>

## الاستدلال (Inference)

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال!

أحضر بعض النصوص التي ترغب في ترجمتها إلى لغة أخرى. بالنسبة لـ T5، تحتاج إلى إضافة بادئة إلى مدخلاتك اعتمادًا على المهمة التي تعمل عليها. للترجمة من الإنجليزية إلى الفرنسية، يجب عليك إضافة بادئة إلى مدخلاتك كما هو موضح أدناه:

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. قم بإنشاء مثيل لـ `pipeline` للترجمة باستخدام نموذجك، ومرر النص الخاص بك إليه:

```py
>>> from transformers import pipeline

# تغيير `xx` إلى لغة الإدخال و `yy` إلى لغة المخرجات المطلوبة.
# أمثلة: "en" للغة الإنجليزية، "fr" للغة الفرنسية، "de" للغة الألمانية، "es" للغة الإسبانية، "zh" للغة الصينية، إلخ؛ translation_en_to_fr تترجم من الإنجليزية إلى الفرنسية
# يمكنك عرض جميع قوائم اللغات هنا - https://huggingface.co/languages
>>> translator = pipeline("translation_xx_to_yy", model="username/my_awesome_opus_books_model")
>>> translator(text)
[{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}]
```

يمكنك أيضًا تكرار نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
قم بتحويل النص إلى رموز وإرجاع `input_ids` كموترات PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

استخدم الدالة [`~generation.GenerationMixin.generate`] لإنشاء الترجمة. لمزيد من التفاصيل حول استراتيجيات توليد النصوص المختلفة والمعلمات للتحكم في التوليد، تحقق من واجهة برمجة تطبيقات [توليد النصوص](../main_classes/text_generation).

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```
</pt>
<tf>
قم بتحويل النص إلى رموز وإرجاع `input_ids` كموترات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

استخدم طريقة [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] لإنشاء الترجمة. لمزيد من التفاصيل حول استراتيجيات توليد النصوص المختلفة والمعلمات للتحكم في التوليد، تحقق من واجهة برمجة تطبيقات [توليد النصوص](../main_classes/text_generation).

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lugumes partagent les ressources avec des bactéries fixatrices d'azote.'
```
</tf>
</frameworkcontent>
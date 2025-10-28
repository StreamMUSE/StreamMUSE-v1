# المعالجة المسبقة Preprocessing

[[open-in-colab]]

قبل تدريب نموذج على مجموعة بيانات، يجب معالجتها مسبقًا وفقًا تنسيق  المتوقع لمدخلات النموذج. سواء كانت بياناتك نصية أو صورًا أو صوتًا، فيجب تحويلها وتجميعها في دفعات من الموترات. يوفر 🤗 Transformers مجموعة من فئات المعالجة المسبقة للمساعدة في إعداد بياناتك للنموذج. في هذا البرنامج التعليمي، ستتعلم أنه بالنسبة لـ:

* للنص، استخدم [مُجزّئ الرموز](./main_classes/tokenizer) لتحويل النص إلى تسلسل من الرموز، وإنشاء تمثيل رقمي للرموز، وتجميعها في موترات(tensors).
* للكلام والصوت، استخدم [مستخرج الميزات](./main_classes/feature_extractor) لاستخراج ميزات متسلسلة من أشكال موجات الصوت وتحويلها إلى موترات.
* تستخدم مدخلات الصورة [ImageProcessor](./main_classes/image_processor) لتحويل الصور إلى موترات.
* تستخدم مدخلات متعددة الوسائط [معالجًا](./main_classes/processors) لدمج مُجزّئ الرموز ومستخرج الميزات أو معالج الصور.

<Tip>

`AutoProcessor` **يعمل دائمًا** ويختار تلقائيًا الفئة الصحيحة للنموذج الذي تستخدمه، سواء كنت تستخدم مُجزّئ رموز أو معالج صور أو مستخرج ميزات أو معالجًا.

</Tip>

قبل البدء، قم بتثبيت 🤗 Datasets حتى تتمكن من تحميل بعض مجموعات البيانات لتجربتها:

```bash
pip install datasets
```

## معالجة اللغة الطبيعية (Natural Language Processing (NLP

<Youtube id="Yffk5aydLzg"/>

أداة المعالجة المسبقة الرئيسية للبيانات النصية هي [مُجزّئ اللغوي](main_classes/tokenizer). يقوم مُجزّئ اللغوي بتقسيم النص إلى  "أجزاء لغوية" (tokens) وفقًا لمجموعة من القواعد. يتم تحويل الأجزاء اللغوية إلى أرقام ثم إلى منسوجات، والتي تصبح مدخلات للنموذج. يقوم المجزئ اللغوي بإضافة أي مدخلات إضافية يحتاجها النموذج.

<Tip>

إذا كنت تخطط لاستخدام نموذج مُدرب مسبقًا، فمن المهم استخدامالمجزئ اللغوي المقترن بنفس ذلك النموذج. يضمن ذلك تقسيم النص بنفس الطريقة التي تم بها تقسيم النصوص ما قبل التدريب، واستخدام نفس  القاموس الذي يربط بين الأجزاء اللغوية وأرقامها ( يُشار إليها عادةً باسم المفردات *vocab*) أثناء التدريب المسبق.

</Tip>

ابدأ بتحميل  المُجزّئ اللغوي مُدرب مسبقًا باستخدام طريقة [`AutoTokenizer.from_pretrained`]. يقوم هذا بتنزيل المفردات *vocab* الذي تم تدريب النموذج عليه:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
```

ثم مرر نصك إلى المُجزّئ اللغوي:

```py
>>> encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
>>> print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

يعيد المُجزّئ اللغوي قاموسًا يحتوي على ثلاثة عناصر مهمة:

* [input_ids](glossary#input-ids) هي الفهارس المقابلة لكل رمز في الجملة.
* [attention_mask](glossary#attention-mask) يشير إلى ما إذا كان يجب الانتباه بالرمز أم لا.
* [token_type_ids](glossary#token-type-ids) يحدد التسلسل الذي ينتمي إليه الرمز عندما يكون هناك أكثر من تسلسل واحد.

أعد إدخالك الأصلي عن طريق فك ترميز `input_ids`:

```py
>>> tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```

كما ترى، أضاف المُجزّئ اللغوي رمزين خاصين - `CLS` و`SEP` (مصنف وفاصل) - إلى الجملة. لا تحتاج جميع النماذج إلى
رموز خاصة، ولكن إذا فعلوا ذلك، فإن المُجزّئ اللغوي يضيفها تلقائيًا لك.

إذا كان هناك عدة جمل تريد معالجتها مسبقًا، فقم بتمريرها كقائمة إلى مُجزّئ اللغوي:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_inputs = tokenizer(batch_sentences)
>>> print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]]}
```

### الحشو Padding

لا تكون الجمل دائمًا بنفس الطول،  وهذا يمكن أن يمثل مشكلة لأن الموترات،وهي مدخلات النموذج، تحتاج إلى شكل موحد. الحشو هو استراتيجية لضمان أن تكون الموترات مستطيلة عن طريق إضافة رمز حشو *padding* خاص إلى الجمل الأقصر.

قم بتعيين معلمة الحشو `padding` إلى `True` لحشو التسلسلات الأقصر في الدفعة لتطابق أطول تسلسل:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0، 0، 0، 0، 0، 0، 0، 0]]}
```

تم الآن حشو الجملتين الأولى والثالثة بـ `0` لأنهما أقصر.

### البتر Truncation

وعلى صعيد أخر، قد يكون التسلسل طويلًا جدًا بالنسبة للنموذج للتعامل معه. في هذه الحالة، ستحتاج إلى بتر التسلسل إلى طول أقصر.

قم بتعيين معلمة `truncation` إلى `True` لتقليم تسلسل إلى الطول الأقصى الذي يقبله النموذج:

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
>>> print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0، 0، 0، 0، 0]]،
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0، 0، 0، 0],
                    [1, 1, 1, 1, 1, 1, 1، 1، 1، 1، 1، 1، 1، 1، 1، 1],
                    [1، 1، 1، 1، 1، 1، 1، 0، 0، 0، 0، 0، 0، 0، 0، 0]]}
```

<Tip>

تحقق من دليل المفاهيم [Padding and truncation](./pad_truncation) لمعرفة المزيد حول معامﻻت الحشو و البتر المختلفة.

</Tip>

### بناء الموترات Build tensors

أخيرًا، تريد أن يقوم  المجزئ اللغوي بإرجاع موترات (tensors) الفعلية التي ستُغذي النموذج.

قم بتعيين معلمة `return_tensors` إلى إما `pt` لـ PyTorch، أو `tf` لـ TensorFlow:

<frameworkcontent>
<pt>

```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
>>> print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```
</pt>
<tf>
 
```py
>>> batch_sentences = [
...     "But what about second breakfast?",
...     "Don't think he knows about second breakfast, Pip.",
...     "What about elevensies?",
... ]
>>> encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
>>> print(encoded_input)
{'input_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
       [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
       [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
      dtype=int32)>,
 'token_type_ids': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>,
 'attention_mask': <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>}
```
</tf>
</frameworkcontent>

<Tip>

تدعم خطوط الأنابيب المختلفة معامل مُجزِّئ الرموز(tokenizer) بشكل مختلف في طريقة `()__call__` الخاصة بها.
و خطوط الأنابيب `text-2-text-generation` تدعم فقط `truncation`.
و خطوط الأنابيب `text-generation` تدعم `max_length` و`truncation` و`padding` و`add_special_tokens`.
أما في خطوط الأنابيب `fill-mask`، يمكن تمرير معامل مُجزِّئ الرموز (tokenizer) في المتغير `tokenizer_kwargs` (قاموس).

</Tip>

## الصوت Audio

بالنسبة للمهام الصوتية، ستحتاج إلى [مستخرج الميزات](main_classes/feature_extractor) لإعداد مجموعة البيانات الخاصة بك للنماذج. تم تصميم مستخرج الميزات لاستخراج الميزات من بيانات الصوت الخام، وتحويلها إلى موتورات.

قم بتحميل مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) (راجع البرنامج التعليمي لـ 🤗 [Datasets](https://huggingface.co/docs/datasets/load_hub) لمزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيفية استخدام مستخرج الميزات مع مجموعات البيانات الصوتية:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

الوصول إلى العنصر الأول من عمود `audio` لمعرفة المدخلات. يؤدي استدعاء عمود `audio` إلى تحميل ملف الصوت وإعادة أخذ العينات تلقائيًا:

```py
>>> dataset[0]["audio"]
{'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 8000}
```

يعيد هذا ثلاثة عناصر:

* `array` هو إشارة الكلام المحملة - وإعادة أخذ العينات المحتملة - كصفيف 1D.
* `path` يشير إلى موقع ملف الصوت.
* `sampling_rate` يشير إلى عدد نقاط البيانات في إشارة الكلام المقاسة في الثانية.

بالنسبة لهذا البرنامج التعليمي، ستستخدم نموذج [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base). الق نظرة على بطاقة النموذج، وستتعلم أن Wav2Vec2 مُدرب مسبقًا على صوت الكلام الذي تم أخذ عينات منه بمعدل 16 كيلو هرتز. من المهم أن يتطابق معدل أخذ العينات لبيانات الصوت مع معدل أخذ العينات لمجموعة البيانات المستخدمة لتدريب النموذج مسبقًا. إذا لم يكن معدل أخذ العينات لبياناتك هو نفسه، فيجب إعادة أخذ العينات من بياناتك.

1. استخدم طريقة [`~datasets.Dataset.cast_column`] في 🤗 Datasets لإعادة أخذ العينات بمعدل أخذ العينات 16 كيلو هرتز:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
```

2. استدعاء عمود `audio` مرة أخرى لأخذ عينات من ملف الصوت:

```py
>>> dataset[0]["audio"]
{'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'sampling_rate': 16000}
```

بعد ذلك، قم بتحميل مستخرج الميزات لتطبيع وحشو المدخلات. عند إضافة حشو للبيانات النصية، تتم إضافة "0" للتسلسلات الأقصر. تنطبق نفس الفكرة على بيانات الصوت. يضيف مستخرج الميزات "0" - الذي يتم تفسيره على أنه صمت - إلى "array".

قم بتحميل مستخرج الميزات باستخدام [`AutoFeatureExtractor.from_pretrained`]:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

مرر صفيف الصوت إلى مستخرج الميزات. كما نوصي بإضافة معامل `sampling_rate` في مستخرج الميزات من أجل تصحيح الأخطاء الصامتة التي قد تحدث بشكل أفضل.

```py
>>> audio_input = [dataset[0]["audio"]["array"]]
>>> feature_extractor(audio_input, sampling_rate=16000)
{'input_values': [array([ 3.8106556e-04,  2.7506407e-03,  2.8015103e-03, ...,
        5.6335266e-04,  4.6588284e-06, -1.7142107e-04], dtype=float32)]}
```

تمامًا مثل مُجزِّئ الرموز، يمكنك تطبيق الحشو أو البتر للتعامل مع التسلسلات المتغيرة في دفعة. الق نظرة على طول التسلسل لهاتين العينتين الصوتيتين:

```py
>>> dataset[0]["audio"]["array"].shape
(173398,)

>>> dataset[1]["audio"]["array"].shape
(106496,)
```

قم بإنشاء دالة لمعالجة مجموعة البيانات بحيث يكون للنماذج الصوتية نفس الأطوال. حدد أقصى طول للعينة ، وسيقوم مستخرج الميزات إما بإضافة حشو أو بتر التسلسلات لمطابقتها:

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays,
...         sampling_rate=16000,
...         padding=True,
...         max_length=100000,
...         truncation=True,
...     )
...     return inputs
```

قم بتطبيق `preprocess_function` على أول بضع أمثلة في مجموعة البيانات:

```py
>>> processed_dataset = preprocess_function(dataset[:5])
```

أطوال العينات الآن متساوية وتطابق الطول الأقصى المحدد. يمكنك الآن تمرير مجموعة البيانات المعالجة إلى النموذج!

```py
>>> processed_dataset["input_values"][0].shape
(100000,)

>>> processed_dataset["input_values"][1].shape
(100000,)
```

## رؤية الكمبيوتر Computer vision

بالنسبة لمهام رؤية الحاسوبية، ستحتاج إلى معالج صور [image processor](main_classes/image_processor) لإعداد مجموعة البيانات الخاصة بك لتناسب النموذج. تتكون معالجة الصور المسبقة من عدة خطوات لتحويل الصور إلى الشكل الذي يتوقعه النموذج. وتشمل هذه الخطوات، على سبيل المثال لا الحصر، تغيير الحجم والتطبيع وتصحيح قناة الألوان وتحويل الصور إلى موترات(tensors).

<Tip>

عادة ما تتبع معالجة الصور المسبقة شكلاً من أشكال زيادة البيانات (التضخيم). كلا العمليتين،  معالجة الصور المسبقة وزيادة الصور تغيران بيانات الصورة، ولكنها تخدم أغراضًا مختلفة:

*زيادة البيانات: تغيير الصور عن طريق زيادة الصور بطريقة يمكن أن تساعد في منع الإفراط في التعميم وزيادة متانة النموذج. يمكنك أن تكون مبدعًا في كيفية زيادة بياناتك - ضبط السطوع والألوان، واالقص، والدوران، تغيير الحجم، التكبير، إلخ. ومع ذلك، كن حذرًا من عدم تغيير معنى الصور بزياداتك.
*معالجة الصور المسبقة: تضمن معالجة الصور اتتطابق الصور مع تنسيق الإدخال المتوقع للنموذج. عند ضبط نموذج رؤية حاسوبية بدقة، يجب معالجة الصور بالضبط كما كانت عند تدريب النموذج في البداية.

يمكنك استخدام أي مكتبة تريدها لزيادة بيانات الصور. لمعالجة الصور المسبقة، استخدم `ImageProcessor` المرتبط بالنموذج.

</Tip>

قم بتحميل مجموعة بيانات [food101](https://huggingface.co/datasets/food101) (راجع دليل 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub) لمزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج الصور مع مجموعات بيانات رؤية الحاسب:

<Tip>

استخدم معامل `split` من 🤗 Datasets لتحميل عينة صغيرة فقط من مجموعة التدريب نظرًا لحجم البيانات كبيرة جدًا!

</Tip>

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("food101", split="train[:100]")
```

بعد ذلك، الق نظرة على الصورة مع ميزة 🤗 Datasets [`Image`](https://huggingface.co/docs/datasets/package_reference/main_classes?highlight=image#datasets.Image):

```py
>>> dataset[0]["image"]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png"/>
</div>

قم بتحميل معالج الصور باستخدام [`AutoImageProcessor.from_pretrained`]:

```py
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

أولاً، دعنا نضيف بعض الزيادات إلى الصور. يمكنك استخدام أي مكتبة تفضلها، ولكن في هذا الدليل، سنستخدم وحدة [`transforms`](https://pytorch.org/vision/stable/transforms.html) من torchvision. إذا كنت مهتمًا باستخدام مكتبة زيادة بيانات أخرى، فتعرف على كيفية القيام بذلك في [دفاتر Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb) أو [دفاتر Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb).

1. هنا نستخدم [`Compose`](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) لربط بعض التحولات معًا - [`RandomResizedCrop`](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) و [`ColorJitter`](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html).
لاحظ بالنسبة لتغيير الحجم، يمكننا الحصول على متطلبات حجم الصورة من `image_processor`. بالنسبة لبعض النماذج، يُتوقع ارتفاع وعرض دقيقين، بينما بالنسبة للنماذج الأخرى، يتم تحديد  الحافة الأقصر`shortest_edge` فقط.

```py
>>> from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )

>>> _transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

2. يقبل النموذج [`pixel_values`](model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel.forward.pixel_values)
كإدخال له. يمكن لـ `ImageProcessor` التعامل مع تطبيع الصور، وتوليد موترات(tensors) مناسبة.
قم بإنشاء دالة تجمع بين تضخيم بيانات الصور ومعالجة الصور المسبقة لمجموعة من الصور وتوليد `pixel_values`:

```py
>>> def transforms(examples):
...     images = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
...     return examples
```

<Tip>

في المثال أعلاه، قمنا بتعيين `do_resize=False` لأننا قمنا بالفعل بتغيير حجم الصور في تحويل زيادة الصور،
واستفدنا من خاصية `size` من `image_processor` المناسب. إذا لم تقم بتغيير حجم الصور أثناء زيادة الصور،
فاترك هذا المعلمة. بشكل افتراضي، ستتعامل `ImageProcessor` مع تغيير الحجم.

إذا كنت ترغب في تطبيع الصور كجزء من تحويل زيادة الصور، فاستخدم قيم `image_processor.image_mean`،
و `image_processor.image_std`.
</Tip>

3. ثم استخدم 🤗 Datasets[`~datasets.Dataset.set_transform`] لتطبيق التحولات أثناء التنقل:
```py
>>> dataset.set_transform(transforms)
```

4. الآن عند الوصول إلى الصورة، ستلاحظ أن معالج الصور قد أضاف `pixel_values`. يمكنك تمرير مجموعة البيانات المعالجة إلى النموذج الآن!

```py
>>> dataset[0].keys()
```

هكذا تبدو الصورة بعد تطبيق التحولات. تم اقتصاص الصورة بشكل عشوائي وتختلف خصائص الألوان بها.

```py
>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> img = dataset[0]["pixel_values"]
>>> plt.imshow(img.permute(1, 2, 0))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png"/>
</div>

<Tip>

بالنسبة للمهام مثل الكشف عن الأشياء، والتجزئة الدلالية، والتجزئة المثالية، والتجزئة الشاملة، يوفر `ImageProcessor`
تقوم هذه الطرق بتحويل النواتج الأولية للنموذج إلى تنبؤات ذات معنى مثل مربعات الحدود،
أو خرائط التجزئة.

</Tip>

### الحشو Pad

في بعض الحالات، على سبيل المثال، عند ضبط نموذج [DETR](./model_doc/detr) بدقة، يقوم النموذج بتطبيق زيادة المقياس أثناء التدريب. قد يتسبب ذلك في اختلاف أحجام الصور في دفعة واحدة. يمكنك استخدام [`DetrImageProcessor.pad`]
من [`DetrImageProcessor`] وتحديد دالة `collate_fn` مخصصة لتجميع الصور معًا.

```py
>>> def collate_fn(batch):
...     pixel_values = [item["pixel_values"] for item in batch]
...     encoding = image_processor.pad(pixel_values, return_tensors="pt")
...     labels = [item["labels"] for item in batch]
...     batch = {}
...     batch["pixel_values"] = encoding["pixel_values"]
...     batch["pixel_mask"] = encoding["pixel_mask"]
...     batch["labels"] = labels
...     return batch
```

## متعدد الوسائط Mulimodal

بالنسبة للمهام التي تتطلب مدخلات متعددة الوسائط، ستحتاج إلى معالج [processor](main_classes/processors) لإعداد مجموعة البيانات الخاصة بك لتناسب النموذج. يقترن المعالج بين  بمعالجين آخرين مثل محول النص إلى رمز ومستخرج الميزات.

قم بتحميل مجموعة بيانات [LJ Speech](https://huggingface.co/datasets/lj_speech) (راجع دليل 🤗 [Datasets tutorial](https://huggingface.co/docs/datasets/load_hub) لمزيد من التفاصيل حول كيفية تحميل مجموعة بيانات) لمعرفة كيف يمكنك استخدام معالج للتعرف التلقائي على الكلام (ASR):

```py
>>> from datasets import load_dataset

>>> lj_speech = load_dataset("lj_speech", split="train")
```

بالنسبة لـ ASR، فأنت تركز بشكل أساسي على `audio` و `text` لذا يمكنك إزالة الأعمدة الأخرى:

```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

الآن الق نظرة على أعمدة `audio` و `text`:
```py
>>> lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
```

الآن الق نظرة على أعمدة `audio` و `text`:

```py
>>> lj_speech[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

>>> lj_speech[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

تذكر أنه يجب عليك دائمًا [إعادة أخذ العينات](preprocessing#audio) لمعدل أخذ العينات في مجموعة البيانات الصوتية الخاصة بك لمطابقة معدل أخذ العينات في مجموعة البيانات المستخدمة لتدريب النموذج مسبقًا!

```py
>>> lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
```

قم بتحميل معالج باستخدام [`AutoProcessor.from_pretrained`]:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
```

1. قم بإنشاء دالة لمعالجة بيانات الصوت الموجودة في `array` إلى `input_values`، ورموز `text` إلى `labels`. هذه هي المدخلات للنموذج:

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))

...     return example
```

2. قم بتطبيق دالة `prepare_dataset` على عينة:

```py
>>> prepare_dataset(lj_speech[0])
```

لقد أضاف المعالج الآن `input_values` و `labels`، وتم أيضًا إعادة أخذ العينات لمعدل أخذ العينات بشكل صحيح إلى 16 كيلو هرتز. يمكنك تمرير مجموعة البيانات المعالجة إلى النموذج الآن!

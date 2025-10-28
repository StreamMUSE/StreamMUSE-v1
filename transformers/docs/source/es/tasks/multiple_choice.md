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

# Selección múltiple

La tarea de selección múltiple es parecida a la de responder preguntas, con la excepción de que se dan varias opciones de respuesta junto con el contexto. El modelo se entrena para escoger la respuesta correcta
entre varias opciones a partir del contexto dado.

Esta guía te mostrará como hacerle fine-tuning a [BERT](https://huggingface.co/google-bert/bert-base-uncased) en la configuración `regular` del dataset [SWAG](https://huggingface.co/datasets/swag), de forma
que seleccione la mejor respuesta a partir de varias opciones y algún contexto.

## Cargar el dataset SWAG

Carga el dataset SWAG con la biblioteca 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

Ahora, échale un vistazo a un ejemplo del dataset:

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

Los campos `sent1` y `sent2` muestran cómo comienza una oración, y cada campo `ending` indica cómo podría terminar. Dado el comienzo de la oración, el modelo debe escoger el final de oración correcto indicado por el campo `label`.

## Preprocesmaiento

Carga el tokenizer de BERT para procesar el comienzo de cada oración y los cuatro finales posibles:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

La función de preprocesmaiento debe hacer lo siguiente:

1. Hacer cuatro copias del campo `sent1` de forma que se pueda combinar cada una con el campo `sent2` para recrear la forma en que empieza la oración.
2. Combinar `sent2` con cada uno de los cuatro finales de oración posibles.
3. Aplanar las dos listas para que puedas tokenizarlas, y luego des-aplanarlas para que cada ejemplo tenga los campos `input_ids`, `attention_mask` y `labels` correspondientes.

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

Usa la función [`~datasets.Dataset.map`] de 🤗 Datasets para aplicarle la función de preprocesamiento al dataset entero. Puedes acelerar la función `map` haciendo `batched=True` para procesar varios elementos del dataset a la vez.

```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

Para crear un lote de ejemplos para selección múltiple, este también le *añadirá relleno de manera dinámica* a tu texto y a las etiquetas para que tengan la longitud del elemento más largo en su lote, de forma que tengan una longitud uniforme. Aunque es posible rellenar el texto en la función `tokenizer` haciendo
`padding=True`, el rellenado dinámico es más eficiente.

El [`DataCollatorForMultipleChoice`] aplanará todas las entradas del modelo, les aplicará relleno y luego des-aplanará los resultados.
```py
>>> from transformers import DataCollatorForMultipleChoice
>>> collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
```

## Entrenamiento

<frameworkcontent>
<pt>
Carga el modelo BERT con [`AutoModelForMultipleChoice`]:

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

<Tip>

Para familiarizarte con el fine-tuning con [`Trainer`], ¡mira el tutorial básico [aquí](../training#finetune-with-trainer)!

</Tip>

En este punto, solo quedan tres pasos:

1. Definir tus hiperparámetros de entrenamiento en [`TrainingArguments`].
2. Pasarle los argumentos del entrenamiento al [`Trainer`] jnto con el modelo, el dataset, el tokenizer y el collator de datos.
3. Invocar el método [`~Trainer.train`] para realizar el fine-tuning del modelo.

```py
>>> training_args = TrainingArguments(
...     output_dir="./results",
...     eval_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     processing_class=tokenizer,
...     data_collator=collator,
... )

>>> trainer.train()
```
</pt>
<tf>
Para realizar el fine-tuning de un modelo en TensorFlow, primero convierte tus datasets al formato `tf.data.Dataset` con el método [`~TFPreTrainedModel.prepare_tf_dataset`].

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

<Tip>

Para familiarizarte con el fine-tuning con Keras, ¡mira el tutorial básico [aquí](training#finetune-with-keras)!

</Tip>

Prepara una función de optimización, un programa para la tasa de aprendizaje y algunos hiperparámetros de entrenamiento:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

Carga el modelo BERT con [`TFAutoModelForMultipleChoice`]:

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

Configura el modelo para entrenarlo con [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> model.compile(optimizer=optimizer)
```

Invoca el método [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) para realizar el fine-tuning del modelo:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2)
```
</tf>
</frameworkcontent>

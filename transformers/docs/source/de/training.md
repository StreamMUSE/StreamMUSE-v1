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

# Optimierung eines vortrainierten Modells

[[open-in-colab]]

Die Verwendung eines vorab trainierten Modells hat erhebliche Vorteile. Es reduziert die Rechenkosten und den CO2-Fußabdruck und ermöglicht Ihnen die Verwendung von Modellen, die dem neuesten Stand der Technik entsprechen, ohne dass Sie ein Modell von Grund auf neu trainieren müssen. Transformers bietet Zugang zu Tausenden von vortrainierten Modellen für eine Vielzahl von Aufgaben. Wenn Sie ein vorab trainiertes Modell verwenden, trainieren Sie es auf einem für Ihre Aufgabe spezifischen Datensatz. Dies wird als Feinabstimmung bezeichnet und ist eine unglaublich leistungsfähige Trainingstechnik. In diesem Tutorial werden Sie ein vortrainiertes Modell mit einem Deep-Learning-Framework Ihrer Wahl feinabstimmen:

* Feinabstimmung eines vorab trainierten Modells mit 🤗 Transformers [`Trainer`].
* Feinabstimmung eines vorab trainierten Modells in TensorFlow mit Keras.
* Feinabstimmung eines vorab trainierten Modells in nativem PyTorch.

<a id='data-processing'></a>

## Vorbereitung eines Datensatzes

<Youtube id="_BZearw7f0w"/>

Bevor Sie die Feinabstimmung eines vortrainierten Modells vornehmen können, müssen Sie einen Datensatz herunterladen und für das Training vorbereiten. Im vorangegangenen Leitfaden haben Sie gelernt, wie man Daten für das Training aufbereitet, und jetzt haben Sie die Gelegenheit, diese Fähigkeiten zu testen!

Laden Sie zunächst den Datensatz [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full):

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

Wie Sie nun wissen, benötigen Sie einen Tokenizer, um den Text zu verarbeiten und eine Auffüll- und Abschneidungsstrategie einzubauen, um mit variablen Sequenzlängen umzugehen. Um Ihren Datensatz in einem Schritt zu verarbeiten, verwenden Sie die 🤗 Methode Datasets [`map`](https://huggingface.co/docs/datasets/process#map), um eine Vorverarbeitungsfunktion auf den gesamten Datensatz anzuwenden:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

Wenn Sie möchten, können Sie eine kleinere Teilmenge des gesamten Datensatzes für die Feinabstimmung erstellen, um den Zeitaufwand zu verringern:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## Training

An dieser Stelle sollten Sie dem Abschnitt folgen, der dem Rahmen entspricht, den Sie verwenden möchten. Sie können über die Links
in der rechten Seitenleiste können Sie zu dem gewünschten Abschnitt springen - und wenn Sie den gesamten Inhalt eines bestimmten Frameworks ausblenden möchten,
klicken Sie einfach auf die Schaltfläche oben rechts im Block des jeweiligen Frameworks!

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## Trainieren mit PyTorch Trainer

🤗 Transformers bietet eine [`Trainer`]-Klasse, die für das Training von 🤗 Transformers-Modellen optimiert ist und es einfacher macht, mit dem Training zu beginnen, ohne manuell eine eigene Trainingsschleife zu schreiben. Die [`Trainer`]-API unterstützt eine breite Palette von Trainingsoptionen und Funktionen wie Logging, Gradientenakkumulation und gemischte Präzision.

Beginnen Sie mit dem Laden Ihres Modells und geben Sie die Anzahl der erwarteten Labels an. Aus dem Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields) wissen Sie, dass es fünf Labels gibt:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

Es wird eine Warnung angezeigt, dass einige der trainierten Parameter nicht verwendet werden und einige Parameter zufällig
initialisiert werden. Machen Sie sich keine Sorgen, das ist völlig normal! Der vorher trainierte Kopf des BERT-Modells wird verworfen und durch einen zufällig initialisierten Klassifikationskopf ersetzt. Sie werden diesen neuen Modellkopf in Ihrer Sequenzklassifizierungsaufgabe feinabstimmen, indem Sie das Wissen des vortrainierten Modells auf ihn übertragen.

</Tip>

### Hyperparameter für das Training

Als Nächstes erstellen Sie eine Klasse [`TrainingArguments`], die alle Hyperparameter enthält, die Sie einstellen können, sowie Flags zur Aktivierung verschiedener Trainingsoptionen. Für dieses Lernprogramm können Sie mit den Standard- [Hyperparametern](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) beginnen, aber Sie können mit diesen experimentieren, um Ihre optimalen Einstellungen zu finden.

Geben Sie an, wo die Kontrollpunkte Ihres Trainings gespeichert werden sollen:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### Auswerten

Der [`Trainer`] wertet die Leistung des Modells während des Trainings nicht automatisch aus. Sie müssen [`Trainer`] eine Funktion übergeben, um Metriken zu berechnen und zu berichten. Die [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) Bibliothek bietet eine einfache [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) Funktion, die Sie mit der [`evaluate.load`] Funktion laden können (siehe diese [quicktour](https://huggingface.co/docs/evaluate/a_quick_tour) für weitere Informationen):

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

Rufen Sie [`~evaluate.compute`] auf `metric` auf, um die Genauigkeit Ihrer Vorhersagen zu berechnen. Bevor Sie Ihre Vorhersagen an `compute` übergeben, müssen Sie die Vorhersagen in Logits umwandeln (denken Sie daran, dass alle 🤗 Transformers-Modelle Logits zurückgeben):

```py
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     predictions = np.argmax(logits, axis=-1)
...     return metric.compute(predictions=predictions, references=labels)
```

Wenn Sie Ihre Bewertungsmetriken während der Feinabstimmung überwachen möchten, geben Sie den Parameter `eval_strategy` in Ihren Trainingsargumenten an, um die Bewertungsmetrik am Ende jeder Epoche zu ermitteln:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### Trainer

Erstellen Sie ein [`Trainer`]-Objekt mit Ihrem Modell, Trainingsargumenten, Trainings- und Testdatensätzen und einer Evaluierungsfunktion:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

Anschließend können Sie Ihr Modell durch den Aufruf von [`~transformers.Trainer.train`] optimieren:

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## Trainieren Sie ein TensorFlow-Modell mit Keras

Sie können auch 🤗 Transformers Modelle in TensorFlow mit der Keras API trainieren!

### Laden von Daten für Keras

Wenn Sie ein 🤗 Transformers Modell mit der Keras API trainieren wollen, müssen Sie Ihren Datensatz in ein Format konvertieren, das
Keras versteht. Wenn Ihr Datensatz klein ist, können Sie das Ganze einfach in NumPy-Arrays konvertieren und an Keras übergeben.
Probieren wir das zuerst aus, bevor wir etwas Komplizierteres tun.

Laden Sie zunächst ein Dataset. Wir werden den CoLA-Datensatz aus dem [GLUE-Benchmark](https://huggingface.co/datasets/glue) verwenden,
da es sich um eine einfache Aufgabe zur Klassifizierung von binärem Text handelt, und nehmen vorerst nur den Trainingssplit.

```py
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  # Just take the training split for now
```

Als nächstes laden Sie einen Tokenizer und tokenisieren die Daten als NumPy-Arrays. Beachten Sie, dass die Beschriftungen bereits eine Liste von 0 und 1en sind,
Wir können sie also ohne Tokenisierung direkt in ein NumPy-Array konvertieren!

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["text"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

Schließlich laden, [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) und [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) Sie das Modell:

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))

model.fit(tokenized_data, labels)
```

<Tip>

Sie müssen Ihren Modellen kein Verlustargument übergeben, wenn Sie sie `compile()`! Hugging-Face-Modelle wählen automatisch
einen Loss, der für ihre Aufgabe und Modellarchitektur geeignet ist, wenn dieses Argument leer gelassen wird. Sie können jederzeit außer Kraft setzen, indem Sie selbst einen Loss angeben, wenn Sie das möchten!

</Tip>

Dieser Ansatz eignet sich hervorragend für kleinere Datensätze, aber bei größeren Datensätzen kann er zu einem Problem werden. Warum?
Weil das tokenisierte Array und die Beschriftungen vollständig in den Speicher geladen werden müssten, und weil NumPy nicht mit
"gezackte" Arrays nicht verarbeiten kann, so dass jedes tokenisierte Sample auf die Länge des längsten Samples im gesamten Datensatz aufgefüllt werden müsste.
Datensatzes aufgefüllt werden. Dadurch wird das Array noch größer, und all die aufgefüllten Token verlangsamen auch das Training!

### Laden von Daten als tf.data.Dataset

Wenn Sie eine Verlangsamung des Trainings vermeiden wollen, können Sie Ihre Daten stattdessen als `tf.data.Dataset` laden. Sie können zwar Ihre eigene
tf.data"-Pipeline schreiben können, wenn Sie wollen, haben wir zwei bequeme Methoden, um dies zu tun:

- [`~TFPreTrainedModel.prepare_tf_dataset`]: Dies ist die Methode, die wir in den meisten Fällen empfehlen. Da es sich um eine Methode
Ihres Modells ist, kann sie das Modell inspizieren, um automatisch herauszufinden, welche Spalten als Modelleingaben verwendet werden können, und
verwirft die anderen, um einen einfacheren, leistungsfähigeren Datensatz zu erstellen.
- [`~datasets.Dataset.to_tf_dataset`]: Diese Methode ist eher auf niedriger Ebene angesiedelt und ist nützlich, wenn Sie genau kontrollieren wollen, wie
Dataset erstellt wird, indem man genau angibt, welche `columns` und `label_cols` einbezogen werden sollen.

Bevor Sie [`~TFPreTrainedModel.prepare_tf_dataset`] verwenden können, müssen Sie die Tokenizer-Ausgaben als Spalten zu Ihrem Datensatz hinzufügen, wie in
dem folgenden Codebeispiel:

```py
def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

Denken Sie daran, dass Hugging Face-Datensätze standardmäßig auf der Festplatte gespeichert werden, so dass dies nicht zu einem erhöhten Arbeitsspeicherbedarf führen wird! Sobald die
Spalten hinzugefügt wurden, können Sie Batches aus dem Datensatz streamen und zu jedem Batch Auffüllungen hinzufügen, was die Anzahl der Auffüllungs-Token im Vergleich zum Auffüllen des gesamten Datensatzes reduziert.


```py
>>> tf_dataset = model.prepare_tf_dataset(dataset, batch_size=16, shuffle=True, tokenizer=tokenizer)
```

Beachten Sie, dass Sie im obigen Codebeispiel den Tokenizer an `prepare_tf_dataset` übergeben müssen, damit die Stapel beim Laden korrekt aufgefüllt werden können.
Wenn alle Stichproben in Ihrem Datensatz die gleiche Länge haben und kein Auffüllen erforderlich ist, können Sie dieses Argument weglassen.
Wenn Sie etwas Komplexeres als nur das Auffüllen von Stichproben benötigen (z. B. das Korrumpieren von Token für die maskierte Sprachmodellierung), können Sie das Argument
Modellierung), können Sie stattdessen das Argument `collate_fn` verwenden, um eine Funktion zu übergeben, die aufgerufen wird, um die
Liste von Stichproben in einen Stapel umwandelt und alle gewünschten Vorverarbeitungen vornimmt. Siehe unsere
[examples](https://github.com/huggingface/transformers/tree/main/examples) oder
[notebooks](https://huggingface.co/docs/transformers/notebooks), um diesen Ansatz in Aktion zu sehen.

Sobald Sie einen `tf.data.Dataset` erstellt haben, können Sie das Modell wie zuvor kompilieren und anpassen:

```py
model.compile(optimizer=Adam(3e-5))

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>

<a id='pytorch_native'></a>

## Trainieren in nativem PyTorch

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] kümmert sich um die Trainingsschleife und ermöglicht die Feinabstimmung eines Modells in einer einzigen Codezeile. Für Benutzer, die es vorziehen, ihre eigene Trainingsschleife zu schreiben, können Sie auch eine Feinabstimmung eines 🤗 Transformers-Modells in nativem PyTorch vornehmen.

An diesem Punkt müssen Sie möglicherweise Ihr Notebook neu starten oder den folgenden Code ausführen, um etwas Speicher freizugeben:

```py
del model
del pytorch_model
del trainer
torch.cuda.empty_cache()
```

Als Nächstes müssen Sie den Datensatz `tokenized_dataset` manuell nachbearbeiten, um ihn für das Training vorzubereiten.

1. Entfernen Sie die Spalte "Text", da das Modell keinen Rohtext als Eingabe akzeptiert:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. Benennen Sie die Spalte "Label" in "Labels" um, da das Modell erwartet, dass das Argument "Labels" genannt wird:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    ```

3. Stellen Sie das Format des Datensatzes so ein, dass PyTorch-Tensoren anstelle von Listen zurückgegeben werden:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

Erstellen Sie dann eine kleinere Teilmenge des Datensatzes, wie zuvor gezeigt, um die Feinabstimmung zu beschleunigen:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

Erstellen Sie einen `DataLoader` für Ihre Trainings- und Testdatensätze, damit Sie über die Datenstapel iterieren können:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
```

Laden Sie Ihr Modell mit der Anzahl der erwarteten Kennzeichnungen:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

### Optimierer und Lernratensteuerung

Erstellen Sie einen Optimierer und einen Scheduler für die Lernrate, um das Modell fein abzustimmen. Wir verwenden den Optimierer [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) aus PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters(), lr=5e-5)
```

Erstellen Sie den Standard-Lernratenplaner aus [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
... )
```

Geben Sie schließlich `device` an, um einen Grafikprozessor zu verwenden, wenn Sie Zugang zu einem solchen haben. Andernfalls kann das Training auf einer CPU mehrere Stunden statt ein paar Minuten dauern.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

Holen Sie sich mit einem gehosteten Notebook wie [Colaboratory](https://colab.research.google.com/) oder [SageMaker StudioLab](https://studiolab.sagemaker.aws/) kostenlosen Zugang zu einem Cloud-GPU, wenn Sie noch keinen haben.

</Tip>

Großartig, Sie sind bereit für das Training! 🥳 

### Trainingsschleife

Um Ihren Trainingsfortschritt zu verfolgen, verwenden Sie die [tqdm](https://tqdm.github.io/) Bibliothek, um einen Fortschrittsbalken über die Anzahl der Trainingsschritte hinzuzufügen:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k, v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### Auswertung

Genauso wie Sie eine Bewertungsfunktion zu [`Trainer`] hinzugefügt haben, müssen Sie dasselbe tun, wenn Sie Ihre eigene Trainingsschleife schreiben. Aber anstatt die Metrik am Ende jeder Epoche zu berechnen und zu melden, werden Sie dieses Mal alle Stapel mit [`~evaluate.add_batch`] akkumulieren und die Metrik ganz am Ende berechnen.

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k, v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits, dim=-1)
...     metric.add_batch(predictions=predictions, references=batch["labels"])

>>> metric.compute()
```
</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## Zusätzliche Ressourcen

Weitere Beispiele für die Feinabstimmung finden Sie unter:

- [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) enthält Skripte
  um gängige NLP-Aufgaben in PyTorch und TensorFlow zu trainieren.

- [🤗 Transformers Notebooks](notebooks) enthält verschiedene Notebooks zur Feinabstimmung eines Modells für bestimmte Aufgaben in PyTorch und TensorFlow.
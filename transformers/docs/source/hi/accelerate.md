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

# वितरित प्रशिक्षण के साथ 🤗 Accelerate

जैसे-जैसे मॉडल बड़े होते हैं, समानांतरता सीमित हार्डवेयर पर बड़े मॉडल को प्रशिक्षित करने और प्रशिक्षण की गति को कई आदेशों के आकार में तेज करने के लिए एक रणनीति के रूप में उभरी है। हगिंग फेस में, हमने उपयोगकर्ताओं को किसी भी प्रकार के वितरित सेटअप पर 🤗 ट्रांसफार्मर्स मॉडल को आसानी से प्रशिक्षित करने में मदद करने के लिए [🤗 Accelerate](https://huggingface.co/docs/accelerate) पुस्तकालय बनाया है, चाहे वह एक मशीन पर कई GPU हों या कई मशीनों में कई GPU। इस ट्यूटोरियल में, जानें कि अपने मूल PyTorch प्रशिक्षण लूप को कैसे अनुकूलित किया जाए ताकि वितरित वातावरण में प्रशिक्षण सक्षम हो सके।

## सेटअप

🤗 Accelerate स्थापित करके शुरू करें:

```bash
pip install accelerate
```

फिर एक [`~accelerate.Accelerator`] ऑब्जेक्ट आयात करें और बनाएं। [`~accelerate.Accelerator`] स्वचालित रूप से आपके वितरित सेटअप के प्रकार का पता लगाएगा और प्रशिक्षण के लिए सभी आवश्यक घटकों को प्रारंभ करेगा। आपको अपने मॉडल को किसी डिवाइस पर स्पष्ट रूप से रखने की आवश्यकता नहीं है।

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## तेजी लाने की तैयारी

अगला कदम सभी प्रासंगिक प्रशिक्षण वस्तुओं को [`~accelerate.Accelerator.prepare`] विधि में पास करना है। इसमें आपके प्रशिक्षण और मूल्यांकन DataLoaders, एक मॉडल और एक ऑप्टिमाइज़र शामिल हैं:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## बैकवर्ड

अंतिम जोड़ यह है कि आपके प्रशिक्षण लूप में सामान्य `loss.backward()` को 🤗 Accelerate के [`~accelerate.Accelerator.backward`] विधि से बदलें:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

जैसा कि आप निम्नलिखित कोड में देख सकते हैं, आपको वितरित प्रशिक्षण सक्षम करने के लिए अपने प्रशिक्षण लूप में केवल चार अतिरिक्त कोड की पंक्तियाँ जोड़ने की आवश्यकता है!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## प्रशिक्षण

एक बार जब आपने प्रासंगिक कोड की पंक्तियाँ जोड़ दी हैं, तो अपने प्रशिक्षण को स्क्रिप्ट या कोलैबोरेटरी जैसे नोटबुक में लॉन्च करें।

### स्क्रिप्ट के साथ प्रशिक्षण

यदि आप स्क्रिप्ट से अपना प्रशिक्षण चला रहे हैं, तो एक कॉन्फ़िगरेशन फ़ाइल बनाने और सहेजने के लिए निम्नलिखित कमांड चलाएँ:

```bash
accelerate config
```

फिर अपने प्रशिक्षण को इस तरह लॉन्च करें:

```bash
accelerate launch train.py
```

### नोटबुक के साथ प्रशिक्षण

🤗 Accelerate एक नोटबुक में भी चल सकता है यदि आप Colaboratory के TPU का उपयोग करने की योजना बना रहे हैं। प्रशिक्षण के लिए जिम्मेदार सभी कोड को एक फ़ंक्शन में लपेटें, और इसे [`~accelerate.notebook_launcher`] में पास करें:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

🤗 Accelerate और इसकी समृद्ध सुविधाओं के बारे में अधिक जानकारी के लिए, [दस्तावेज़ीकरण](https://huggingface.co/docs/accelerate) देखें।

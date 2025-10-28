<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Inference on a Multiple GPUs

この文書には、複数のGPUで効率的に推論を行う方法に関する情報が含まれています。
<Tip>

注意: 複数のGPUセットアップは、[単一のGPUセクション](./perf_infer_gpu_one)で説明されているほとんどの戦略を使用できます。ただし、より良い使用法のために使用できる簡単なテクニックについても認識しておく必要があります。

</Tip>

## Flash Attention 2

Flash Attention 2の統合は、複数のGPUセットアップでも機能します。詳細については、[単一のGPUセクション](./perf_infer_gpu_one#Flash-Attention-2)の適切なセクションをご覧ください。

## BetterTransformer

[BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview)は、🤗 TransformersモデルをPyTorchネイティブの高速実行パスを使用するように変換し、その下でFlash Attentionなどの最適化されたカーネルを呼び出します。

BetterTransformerは、テキスト、画像、音声モデルの単一GPUおよび複数GPUでの高速推論もサポートしています。
<Tip>

Flash Attentionは、fp16またはbf16 dtypeを使用しているモデルにのみ使用できます。BetterTransformerを使用する前に、モデルを適切なdtypeにキャストしてください。

</Tip>

### Decoder models

テキストモデル、特にデコーダーベースのモデル（GPT、T5、Llamaなど）の場合、BetterTransformer APIはすべての注意操作を[`torch.nn.functional.scaled_dot_product_attention`オペレーター](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)（SDPA）を使用するように変換します。これはPyTorch 2.0以降でのみ使用可能です。

モデルをBetterTransformerに変換するには：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# convert the model to BetterTransformer
model.to_bettertransformer()

# Use it for training or inference
```

SDPAは、ハードウェアや問題のサイズなどの特定の設定で[Flash Attention](https://arxiv.org/abs/2205.14135)カーネルを呼び出すこともできます。Flash Attentionを有効にするか、特定の設定（ハードウェア、問題のサイズ）で利用可能かを確認するには、[`torch.nn.kernel.sdpa_kernel`](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)をコンテキストマネージャとして使用します。


```diff
import torch
+ from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

もしトレースバックで次のようなエラーメッセージが表示された場合：


```bash
RuntimeError: No available kernel.  Aborting execution.
```

当日、Flash Attentionのカバレッジが広範囲である可能性があるPyTorch Nightlyバージョンを試すようにお勧めします。

```bash
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

[このブログ投稿](https://pytorch.org/blog/out-of-the-box-acceleration/)をチェックして、BetterTransformer + SDPA APIで可能なことについて詳しく学びましょう。

### Encoder Models

推論中のエンコーダーモデルでは、BetterTransformerはエンコーダーレイヤーのforward呼び出しを、エンコーダーレイヤーの[`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)の相当するものにディスパッチします。これにより、エンコーダーレイヤーの高速実装が実行されます。

`torch.nn.TransformerEncoderLayer`の高速実装はトレーニングをサポートしていないため、代わりに`torch.nn.functional.scaled_dot_product_attention`にディスパッチされます。これにより、ネストされたテンソルを活用しないFlash AttentionまたはMemory-Efficient Attentionの融合カーネルを使用できます。

BetterTransformerのパフォーマンスの詳細については、この[ブログ投稿](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2)をご覧いただけます。また、エンコーダーモデル用のBetterTransformerについては、この[ブログ](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)で詳しく学ぶことができます。


## Advanced usage: mixing FP4 (or Int8) and BetterTransformer

モデルの最良のパフォーマンスを得るために、上記で説明した異なる方法を組み合わせることができます。例えば、FP4ミックスプレシジョン推論+Flash Attentionを使用したBetterTransformerを組み合わせることができます。


```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
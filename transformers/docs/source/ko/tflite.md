<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TFLite로 내보내기[[export-to-tflite]]

[TensorFlow Lite](https://www.tensorflow.org/lite/guide)는 자원이 제한된 휴대폰, 임베디드 시스템, 사물인터넷(IoT) 기기에서 
기계학습 모델을 배포하기 위한 경량 프레임워크입니다. 
TFLite는 연산 능력, 메모리, 전력 소비가 제한된 기기에서 모델을 효율적으로 최적화하고 실행하기 위해 
설계되었습니다. 
TensorFlow Lite 모델은 `.tflite` 파일 확장자로 식별되는 특수하고 효율적인 휴대용 포맷으로 표현됩니다. 

🤗 Optimum은 `exporters.tflite` 모듈로 🤗 Transformers 모델을 TFLite로 내보내는 기능을 제공합니다. 
지원되는 모델 아키텍처 목록은 [🤗 Optimum 문서](https://huggingface.co/docs/optimum/exporters/tflite/overview)를 참고하세요. 

모델을 TFLite로 내보내려면, 필요한 종속성을 설치하세요:
 
```bash
pip install optimum[exporters-tf]
```

모든 사용 가능한 인수를 확인하려면, [🤗 Optimum 문서](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model)를 참고하거나 
터미널에서 도움말을 살펴보세요:

```bash
optimum-cli export tflite --help
```

예를 들어 🤗 Hub에서의 `google-bert/bert-base-uncased` 모델 체크포인트를 내보내려면, 다음 명령을 실행하세요:

```bash
optimum-cli export tflite --model google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```

다음과 같이 진행 상황을 나타내는 로그와 결과물인 `model.tflite`가 저장된 위치를 보여주는 로그가 표시됩니다:

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
 ```

위 예제는 🤗 Hub에서의 체크포인트를 내보내는 방법을 보여줍니다. 
로컬 모델을 내보낸다면, 먼저 모델 가중치와 토크나이저 파일이 모두 같은 디렉터리( `local_path` )에 저장됐는지 확인하세요. 
CLI를 사용할 때, 🤗 Hub에서의 체크포인트 이름 대신 `model` 인수에 `local_path`를 전달하면 됩니다. 
# Model Exploration

## 가설
- 국어 문제를 잘 풀기 위해서는 지문에서 정답을 보다 잘 추론하기 위한 Reasoning 능력이 중요할 것 이라고 판단. →  **모델 사이즈가 크고 한국어로 사전 학습된 모델을 탐색.**
- 사회 문제는 관련된 배경지식이 사전 학습 되어야 한다고 판단. → **Wikipedia 혹은 Textbook을 사전 학습된 모델을 탐색.**
- Hugging Face에서 한국어가 가능한 LLM들에 대해 성능 평가 진행

| Model | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| [hungun/Qwen2.5-7B-Instruct-kowiki-qa](https://huggingface.co/hungun/Qwen2.5-7B-Instruct-kowiki-qa) | 0.7396 | 0.7126 |
| [beomi/Qwen2.5-7B-Instruct-kowiki-qa-context](https://huggingface.co/beomi/Qwen2.5-7B-Instruct-kowiki-qa-context) | 0.6797 | 0.6368 |
| [LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct) | 0.6544 | 0.6184 |
| [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) | 0.5899 | 0.5632 |
| [Saxo/Linkbricks-Horizon-AI-Korean-llama-3.1-sft-dpo-8B](https://huggingface.co/Saxo/Linkbricks-Horizon-AI-Korean-llama-3.1-sft-dpo-8B) | 0.5945 | 0.5517 |
| [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) | 0.5899 | 0.5632 |
| [lcw99/llama-3-10b-wiki-240709-f](https://huggingface.co/lcw99/llama-3-10b-wiki-240709-f?library=transformers) | 0.5760 | 0.5379 |
| [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) | 0.6705 | 0.6391 |



## 결과 분석
- 크기가 큰 모델일수록 높은 성능을 보이는 것을 확인할 수 있었고 이는 큰 모델이 훨씬 많은 데이터로 사전학습했기 때문에 일반화 성능이 더 높기 때문. 큰 모델이 가지고 있는 배경 지식도 더 많았고 Reasoning 능력도 더 좋았음.
- 실험을 통해 작은 모델로 Fine-Tuning을 시켜도 이런 선천적인 능력 차이를 극복하기 어렵다고 판단하였고 개발 환경인 V100에서 구동할 수 있는 가능한 가장 큰 크기의 모델을 사용하기로 판단.
- 비슷한 사이즈의 모델끼리 비교했을 때 Kowiki로 사전학습 된 모델이 test data에 대한 Accuracy가 더 높았음. 따라서 Kowiki로 사전학습 되지 않은 모델의 문제 해결 능력을 키우자는 인사이트 도출.

실험을 통해 작은 모델로 Fine-Tuning을 시켜도 이런 선천적인 능력 차이를 극복하기 어렵다고 판단하였고 개발 환경인 V100에서 구동할 수 있는 가능한 가장 큰 크기의 모델을 탐색하였음.

# 경량화
- Model Exploration 실험에 따라 모델의 크기가 커질 수록 Reasoning 능력이 향상되는 것을 확인했고 개발 환경인 V100에서 구동할 수 있는 가능한 가장 큰 크기의 모델을 탐색하였음.
- 이에, LLM의 Fine Tuning을 효율적으로 수행하기 위해 학습 속도와 메모리 사용량을 최적화 시킨 Unsloth Tool을 활용.
- 가장 크고 성능이 좋은 모델들을 실험해본 결과, 최종적으로 [unsloth/Qwen2.5-32B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit) 를 베이스라인 모델로 선정.

| Model | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| [unsloth/gemma-2-27b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit) | 0.6083 | 0.5816 |
| [unsloth/Qwen2.5-14B-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-14B-bnb-4bit) | 0.6889 | 0.6621|
| [unsloth/Qwen2.5-32B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit) | **0.7765** | 0.7448 |

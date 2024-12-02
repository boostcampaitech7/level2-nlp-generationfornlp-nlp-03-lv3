# 개요
- LLM이 지문을 읽고 정답을 추론하는 과정을 학습할 수 있게끔 **TOEFL listening (lecture)** 데이터를, 모델에 사회과목 및 세계사, 미국사 과목 배경 지식을 주입하기 위해 **한국어 위키백과 데이터셋(Kowiki)** 과 **SAT Questions and Answers for LLM** 데이터를 이용해 베이스라인 모델([`unsloth/Qwen2.5-32B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit))에 **1차적으로 Fine-Tuning을 진행**함.
- 베이스라인 모델이 영어 기반으로 사전학습된 모델이므로 TOEFL과 SAT 데이터를 영문으로 구축하여 영문 데이터에 대해 먼저 학습하고 그 후 train data를 학습하는 것으로 실험 진행.


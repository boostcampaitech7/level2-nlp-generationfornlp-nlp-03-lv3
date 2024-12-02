# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Generation for NLP Project : 한국어 수능 시험 문제 풀이 모델

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [Installation and Quick Start](#3-installation-and-quick-start)
4. [팀원 소개](#4-팀원-소개)
5. [프로젝트 진행](#5-프로젝트-진행)
6. [리더보드 결과](#6-리더보드-결과)

## 1. 프로젝트 소개
(1) 주제 및 목표
- 부스트캠프 AI Tech NLP 트랙 level 2 대회
- 주제 : 한국어 수능 시험 문제 풀이 모델 (Generation for NLP)    
      수능 특화 언어모델 만들기  <br>

(2) 평가지표
- Accuracy <br>

(3) 개발 환경 <br>
- GPU : Tesla V100 * 4 <br>

## 2. 프로젝트 구조
```sh
.
├── model
│   ├── fine_tune_gnn.py
│   └── SimCSE.py
├── preprocessing
│   ├── modeling
│   │   └── Clustering.ipynb
│   ├── DataCleaning.ipynb

├── resources
│   ├── auged
│   ├── merge
│   ├── processed
│   └── raw
│       ├── train.csv
│       ├── train_reformat.csv
│       ├── test.csv
│       └── test_reformat.csv
├── utils
│   ├── arguments.py
│   ├── clean.py
│   ├── dataloader.py
│   ├── helpers.py
│   ├── metric.py
│   └── prompt.py
├── train.py
├── train_unsloth.py
├── inference.py
└── inference_unsloth.py
```

## 3. Installation and Quick Start

**Step 1.** 해당 repository를 clone해서 사용

**Step 2.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`와 `requirements_unsloth.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행
```sh
$ python -m venv .venv
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
**Step 3.** `preprocessing` 폴더에서 원하는 전처리 작업 진행하여 데이터 가공

**Step 4.** 본인의 가상환경에서 Training 수행
```sh
$ . .venv/bin/activate

# 다양한 데이터로 학습
$ python train_unsloth.py
```

**Step 5.** 본인의 가상환경에서 Inference 수행
```sh
$ . .venv/bin/activate

# 단일 모델 Inference
$ python inference_unsloth.py

# 다양한 모델 앙상블
$ python run_ensemble.py

```

**Optional.** 원격 연결 끊어졌을 때도 돌아갈 수 있도록 Tmux 사용을 권장
```sh
# 새로운 세션 생성
$ tmux new -s (session_name)

# 세션 목록
$ tmux ls

# 세션 시작하기 (다시 불러오기)
tmux attach -t (session_name)

# 세션에서 나가기
(ctrl + b) d

# 특정 세션 강제 종료
$ tmux kill-session -t (session_name)
```


## 4. 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
|[Github](https://github.com/dongspam0209)|[Github](https://github.com/sunghoon014)|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github](https://github.com/kkobugi)| -->|


### 맡은 역할
|**Member**|**Team**|**Role**|
|:--|--|--|
|**김동한**|Data, Model|- **TOEFL+SAT 데이터 1차 Fine-Tuning**<br>- **PSAT기출문제 데이터 증강**<br>- **모델 훈련 실험**|
|**김성훈**|Data, Model|내용|
|**김수아**|Data, Model|- **검정고시 데이터 증강**<br>- **K-Fold 도입**<br>- **앙상블**|
|**김현욱**|Data, Model|- **MLflow 환경 구축**<br>- **모델 탐색**<br>- **Kowiki 사전 학습**<br>- **모델 훈련 실험**|
|**송수빈**|Data, Model|- **EDA**<br>- **데이터 증강(kbs 한국어능력시험, SAT history, KorQuAD)**<br>- **SAT 데이터 사전학습**<br>- **모델 훈련 실험**|
|**신수환**|Data, Model|내용|
<br>

## 5. 프로젝트 진행
| Task | **Task Description** |
| --- | --- |
| [**EDA**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/EDA.md) | 데이터의 특성을 살펴보기 위해 데이터 시각화 및 분석 |
| [**Model Exploration**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Model_Experiment.md) | 한국어 수능 문제 풀이에 적합한 pre-trained model 선정 |
| [**Fine-Tuning_1**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Fine-Tuning_1.md) | 모델의 문제 풀이 능력을 향상시키기 위한 1차 fine-tuning |
| [**Fine-Tuning_2**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Fine-Tuning_2.md) | 모델의 문제 풀이 능력을 향상시키기 위한 2차 fine-tuning |
| [**Post Processing & Result**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Result.md) | K-Fold, 앙상블 알고리즘을 사용해 후처리 |

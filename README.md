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
|**김동한**|Data, Model|- **TOEFL 데이터 사전학습**<br>- **pdf 데이터셋화**<br>- **모델 훈련 실험**|
|**김성훈**|Data, Model|내용|
|**김수아**|Data, Model|내용|
|**김현욱**|Data, Model|- **MLflow 환경 구축**<br>- **모델 탐색**<br>- **Kowiki 사전 학습**<br>- **모델 훈련 실험**|
|**송수빈**|Data, Model|내용|
|**신수환**|Data, Model|내용|
<br>

## 5. 프로젝트 진행
| Task | **Task Description** |
| --- | --- |
| **EDA** | 데이터의 특성을 살펴보기 위해 label별 분포 등 시각화 및 분석 |
| **Augmentation** | 데이터셋의 불균형성을 해소하기 위해 다양한 방법으로 데이터 증강 수행 |
| **Model Exploration** | 데이터셋과 STS  task를 수행하기 적합한 pre-trained model 선정 |
| Second-stream with GNN | 단어들 사이의 유의어 관계를 모델링하기 위해 Graph Neural Networks (GNN)을 second-stream으로 NLP 모델에 통합하고 성능 평가 / 최종 제출 때는 사용 x |
| **Soft Voting Ensemble** | 증강된 데이터셋으로 학습한 다양한 model의 예측확률을 평균하여 여러 모델의 강점을 결합해 성능 향상 |


### 사용한 데이터셋
데이터는 train.csv / test.csv의 2개의 파일로 되어있으며 각 파일의 column은 다음과 같이 구성되어있다. <br>

**id** : 문제 고유 id <br>
**paragraph** : 지문 <br>
**question** : 문제 <br>
**choices** : 선지 <br>
**answer** : 정답 <br>
**question_plus** : 보기 <br>


### Model Exploration
여러 실험으로 크기가 큰 모델일수록 높은 성능을 보이는 것을 확인할 수 있었고 이는 큰 모델이 훨씬 많은 데이터로 사전학습했기 때문에 일반화 성능이 더 높기 때문. 큰 모델이 가지고 있는 배경 지식도 더 많았고 reasoning 능력도 더 좋았음.

Hugging Face에서 한국어가 가능한 LLM들에 대해 성능 평가 진행

실험을 통해 작은 모델로 Fine-Tuning을 시켜도 이런 선천적인 능력 차이를 극복하기 어렵다고 판단하였고 개발 환경인 V100에서 구동할 수 있는 가능한 가장 큰 크기의 모델을 탐색하였음.

최종적으로 [unsloth/Qwen2.5-32B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit) 를 베이스라인 모델로 선정.

| Model | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) | 0.6705 | 0.6391 |
| [unsloth/gemma-2-27b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit) | 0.6083 | 0.5816 |
| [unsloth/Qwen2.5-32B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit) | **0.7765** | 0.7448 |
| [cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) | 0.4240 | 0.3908 |
| [hungun/Qwen2.5-7B-Instruct-kowiki-qa](https://huggingface.co/hungun/Qwen2.5-7B-Instruct-kowiki-qa) | 0.7396 | 0.7126 |


### 모델 1차 학습
#### 모델링 설명
- train, test dataset이 국어와 사회 과목 문제로 구성되어 있었기 때문에 국어 문제를 잘 풀기 위한 reasoning 능력과 사회 문제를 잘 풀기 위한 배경 지식을 가지고 있는 모델이 문제를 잘 풀 것이라고 판단
- reasoning 능력을 키울 수 있는 데이터(TOEFL, SAT)와 사회 사전 지식 데이터(Kowiki)로 모델을 사전 학습

| Data | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| merge_1125 | 0.7857 | 0.7586 |
| TOEFL -> merge_1125 | 0.7880 | 0.7793 |

#### 결과 분석
- 위 두 실험을 비교해보면, TOEFL로 1차 학습을 진행했을 때 Public, Private Accuracy 모두 상승하여 1차 학습이 효과가 있음을 확인할 수 있음.

### EDA
#### 대회 데이터 구성
- 수능형 문제
      - 수능 국어, 사회 영역(윤리, 정치, 사회)
- KMMLU
      - Korean History
- MMMLU
      - HighSchool 데이터 중 역사, 경제, 정치, 지리, 심리
- KLUE-MRC
      - 경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을
#### Source 별 분포
|  | Count | Ratio |
| --- | --- | --- |
| 수능형 문제 | 0 | 0.000000 |
| KMMLU | 72 | 0.035468 |
| MMMLU | 719 | 0.354187 |
| KLUE-MRC | 1239 | 0.610345 |
      
#### Subject 별 분포
|  | Count | Ratio |
| --- | --- | --- |
| korean history | 72 | 0.035468 |
| world history | 192 | 0.094581 |
| us history | 139 | 0.068473 |
| european history | 118 | 0.058128 |
| psychology | 140 | 0.068966 |
| macroeconomics | 78 | 0.038424 |
| microeconomics | 36 | 0.017734 |
| government | 9 | 0.004433 |
| geography | 7 | 0.003448 |
| KLUE-MRC | 1239 | 0.610345 |
        
#### 문장 길이 분포
|  | KMMLU | MMMLU | KLUE-MRC |
| --- | --- | --- | --- |
| count | 72 | 719 | 1239 |
| mean | 149.75 | 337.28 | 1000.82 |
| std | 65.20 | 261.87 | 357.90 |
| min | 23.00 | 15.00 | 503.00 |
| 25% | 103.00 | 80.50 | 711.50 |
| 50% | 141.50 | 329.00 | 909.00 |
| 75% | 190.25 | 520.50 | 1252.00 |
| max | 353.00 | 1292.00 | 2017.00 |
  
#### 데이터 분석 인사이트
1. 학습 데이터의 source는 KMMLU, MMMLU, KLUE-MRC로, 학습 데이터에는 수능형 문제가 포함되어 있지 않음
      - 주어진 학습 데이터만으로는국어 과목의 문제풀이 방식을 충분히 학습하지 못할 가능성이 높음
      - 수능형 문제가 테스트 데이터로 주어졌을 경우, 모델이 문제 해결에 어려움을 겪을 가능성이 큼
      - 따라서 수능형 문제를 증강하여 모델이 이 유형의 문제를 잘 학습하도록 설계하는 것이 중요하다고 판단
2. 학습 데이터의 source별 분포를 보면 KLUE-MRC의 비율이 가장 높으며, KLUE-MRC 데이터의 source는 Wikipedia의 비중이 높음
      - KLUE-MRC와 유사한 데이터를 증강하면 모델 성능 개선에 기여할 가능성이 높음
      - KorQuAD 데이터 역시 Wikipedia를 바탕으로 만들어진 QA 데이터셋이기 때문에 학습 데이터로 활용하기 좋다고 판단됨
3. 학습 데이터의 subject별 분포를 보면 과목별로 서로 같은 비율을 갖지 않음
      - 외부 데이터셋으로 학습 데이터 증강할 때, 학습 데이터 내 과목별 데이터 분포를 해치지 않도록 테이블 상의 비율을 참고하면서 증강해야 함
4. 학습 데이터의 paragraph 문장 길이 분포를 보면 source별로 확연한 차이를 보임
      - 문장 길이를 기준으로, RAG를 활용하여 문제 풀이를 진행할 데이터셋과 주어진 지문 내에서 문제 풀이를 진행할 데이터셋을 구분할 수 있음


### 모델 Fine-Tuning
#### 모델링 설명
- train, test 데이터와 같은 구조를 가지는 데이터로 Fine-Tuning
- (1) 증강 학습데이터로 학습
- (2) K-Fold

**(1) 증강 데이터로 학습**
| Data (1차 -> 2차) | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| curr_sat | 0.7765 | 0.7632 |
| merge_1122 | 0.7811 | 0.7862 |
| TOEFL+SAT -> merge_1127 | 0.7972 | 0.7770 |
| TOEFL+SAT -> merge_1127(지문생성) | 0.7949 | 0.7701 |

**curr_sat** : 검정고시 + KBS한국어능력시험 + PSAT<br>
**merge_1122** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '경제학', '교육학')<br>
**merge_1125** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '사회복지학')<br>
**merge_1127** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '경제학', '교육학', ‘사회복지학’)<br>
**merge_1127(지문생성)** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '경제학', '교육학', ‘사회복지학’) + 공무원 기출문제은행에서 지문이 없는 데이터의 지문을 LLM으로 생성<br>

**결과 분석**<br>
1. 증강을 통해 데이터의 양을 늘리는 것은 효과가 있었음.
2. merge_1127에서 지문을 생성한 데이터와 원본 그대로의 데이터의 학습 결과를 비교해보면 지문을 생성한 데이터의 Accuracy가 모두 작은 것을 볼 수 있는데 이로 미루어 보아 LLM으로 생성한 지문의 품질이 좋지 않음을 짐작할 수 있음.<br>

**(2) K-Fold**

**결과 분석**
| Model | . | . |
| --- | --- | --- |
| . | 0.926 | 0.9110 |
| . | 0.929 | 0.9164 |
| . | 0.929 | 0.9190 |

### Soft Voting Ensemble
**모델링 설명**
- Soft Voting은 앙상블 학습에서 사용되는 기법으로, 여러 개의 분류 모델의 예측 결과를 평균하여 최종 예측을 만드는 방법
- 각 모델이 예측한 logit을 평균하거나 가중 평균하여 최종 logit 결정
- Valid score 기반 가중 평균
    - 앙상블할 모델의 valid score만큼 비율로 곱하여 가중 평균
    - e.g) model A : 0.9 / model B : 0.8 인 경우
        
        $$
        \frac {A_i \times0.9+B_i\times 0.8} {0.9+0.8}
        $$

$$
0.8+\frac {x-x_{min}} {x_{max}-x_{min}}\times(1.2-0.8)
$$


**결과 분석**
- Data Aaugmentation 진행한 결과에 따른 4가지 version의 train data와 Model exploration&Modeling을 거쳐 선정된 model에 다양한 조합으로 실험하여 최적의 성능 도출
- **각 기법마다 best case에 대해서 비교해본 결과 min-max 평균을 취한 case가 가장 높은 92.98의 public pearson 값을 가지는 것을 확인하고 이를 최종 리더보드에 제출**

| 모델 | 활용 기법 | Validation Pearson | Min-Max 정규화 가중 평균 |
| --- | --- | --- | --- |
| deliciouscat/kf-deberta-base-cross-sts | raw + Contrastive Learning | 0.930 | 1.111 |
| deliciouscat/kf-deberta-base-cross-sts | raw + Cleaning | 0.930 | 1.111 |
| sorryhyun/sentence-embedding-klue-large | Augmentation v2 | 0.923 | 0.800 |
| snunlp/KR-ELECTRA-discriminator | Augmentation v2 | 0.932 | 1.200 |
| snunlp/KR-ELECTRA-discriminator | Augmentation v3 | 0.930 | 1.111 |

## 6. 리더보드 결과

**Public Leader Board 순위**

<img src="https://github.com/user-attachments/assets/7c4a1e2f-5f05-42a9-a4ee-baad6935f530"/>


**Private Leader Board 순위**

<img src="https://github.com/user-attachments/assets/54a48d5f-6b46-4740-8e66-914bcac52f0d"/>

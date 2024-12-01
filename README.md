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
| **EDA** | 데이터의 특성을 살펴보기 위해 데이터 시각화 및 분석 |
| **Augmentation** | 데이터셋 크기를 키우기 위해 데이터 증강 수행 |
| **Model Exploration** | 한국어 수능 문제 풀이에 적합한 풍부한 배경 지식을 갖춘 pre-trained model 선정 |
| **Model Train** | 증강 데이터 및 K-Fold 알고리즘을 사용하여 모델 훈련 |
| **Soft Voting Ensemble** | 증강된 데이터셋과 하이퍼 파라미터 튜닝으로 훈련한 모델의 추론 결과들을 결합해 성능 향상 |


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

<img src="https://github.com/user-attachments/assets/fd6fce6c-113e-4ca2-9427-5949dd461747" width="600" />
      
#### Subject 별 분포

<img src="https://github.com/user-attachments/assets/21e511ed-7033-4dc9-8e9a-f8e2f756f2db" width="600" />

        
#### 문장 길이 분포

| KMMLU | MMMLU | KLUE-MRC |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/a56c9e92-c8df-4b1b-a983-22c84dd620c8" width="600" /> | <img src="https://github.com/user-attachments/assets/0de39eb6-9ce3-4071-812f-245e9b49b905" width="600" /> | <img src="https://github.com/user-attachments/assets/a797e100-b84f-443c-991b-9c7b0244e867" width="600" /> |

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

### 데이터 증강
#### kbs 한국어 능력시험
- 데이터 설명
    - KBS가 국민의 국어 사용 능력을 높이고 국어문화를 발전시키는 데 기여하기 위해 시행하는 시험
    - 어휘, 어법 영역은 수능보다 어려울 수 있다고 판단하여 증강 데이터로 사용하지 않음.
    - 듣기, 창안, 국어문화 영역은 수능 형식과는 동떨어지기 때문에 증강 데이터로 사용하지 않음.
    - 쓰기, 말하기 영역은 문제들이 유기적으로 연결되어 있기 때문에 학습 데이터로 활용하기에 적합하지 않다고 판단하여 증강 데이터로 사용하지 않음.
    - 읽기 역역은 하나의 지문이 주어지고, 이에 대한 여러 개의 질문으로 구성되는 형식이 수능 국어 형식과 비슷하여 증강 데이터로 사용.
- 기출문제가 공개된 두 회차에 대해 데이터 증강 진행.
    - PDF Miner 활용
    - 제대로 데이터화되지 않은 부분에 대해서는 수작업을 진행.
- 실험 결과
    
    |  | Public Accuracy |  Private Accuracy |
    | --- | --- | --- |
    | origin | 0.6751 | 0.6207 |
    | kbs auged | **0.6843** | **0.6253** |
- 결과 분석 및 인사이트
    - 국어 문항 약 40개정도만 추가했는데도 성능이 오르는 것을 보아, 데이터 증강이 의미가 있음을 확인.
    - 특히 수능형 문제와 비슷한 문제만을 골라 증강한 것이 도움이 되었을 것으로 생각됨.

#### SAT Questions and Answers for LLM
- 데이터 설명
    - SAT는 미국의 수능으로, 이 데이터는 LLM(대규모 언어 모델) 학습 및 평가를 목적으로 만들어짐.
    - 세계사(World History) 및 미국사(U.S. History) 과목 시험의 문제와 보기, 선지, 그리고 정답이 주어짐.
    - 이는 MMMLU 데이터의 비중이 높은 상위 두 과목으로, 데이터 증강 시 성능 개선에 도움이 될 것이라 판단.
- DeepL API를 활용한 번역
    - SAT Questions and Answers for LLM 데이터셋은 영어로 구성되어 있어, 학습 데이터로 활용하기 위해 데이터를 한국어로 번역하였고 번역 과정에서 의미 전달의 정확성과 품질 저하를 최소화하기 위해 DeepL API를 활용.
- LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct 모델을 활용한 지문 생성
    - EXAONE은 한국어에 특화된 대규모 언어 모델로, 한국어 이해 및 생성 능력이 뛰어나며 자연스러운 문맥 처리와 정확한 텍스트 생성에서 우수한 성능을 보이기에 지문 생성에 적합할 것이라 판단.
- 실험 결과
    
    |  | Public Accuracy |  Private Accuracy |
    | --- | --- | --- |
    | kbs(korean) auged | 0.6843 | 0.6253 |
    | SAT(world history) auged with no paragraph | 0.6866 | **0.6322** |
    | SAT(world history) auged with paragraph | **0.6889** | 0.6230 |
    | SAT(world history, us history) auged with paragraph | 0.6636 | 0.6230 |
- 결과 분석 및 인사이트
    - World History (272문항)
        - World History 과목은 학습 데이터 중 KLUE-MRC 데이터 다음으로 가장 많은 비중을 차지함. (192문항, 9.4%)
        - 272문항을 합쳤을 때 KLUE-MRC 데이터의 개수를 넘지 않으면서, LLM이 생성한 세계사 지문을 학습함으로써 성능 향상이 있었던 것으로 판단.
    - U.S. History (1,107문항)
        - U.S. History 과목은 학습 데이터 중 World History 과목 다음으로 심리학 과목과 비슷하게 많은 비중을 차지함. (139문항, 6.8%)
        - World History 과목보다 적은 비중을 차지함에도 불구하고 5배 가량의 데이터를 증강했으며, 증강 후 U.S. History 과목의 총 개수가 KLUE-MRC 데이터의 개수를 넘겨버림으로써 성능 감소가 있었던 것으로 판단.
    - U.S. History 데이터는 사전학습 단계에서만 활용하고, Fine-Tuning에는 World History 데이터만 활용.

#### 국가공무원 시험 5급 (행정) 언어논리 데이터(PSAT) 증강
- 데이터 설명
    - 수능 국어 영역 문제와 유사한 데이터가 부족하여 증강.
    - PSAT에 수능 국어 문제와 동일하게 지문을 읽고 이해하여 올바른 선지를 고르는 유형의 문제들이 있어 증강 데이터로 사용하기로 결정.
- 공개된 3개년 기출문제(22, 23, 24) 데이터에 대해 증강.
    - PDF Miner 활용
    - 정규표현식으로  `2022년도\s+국가공무원\s+5급\s+공채\s+등\s+필기시험'` 과 같이 시험 자체의 정보를 담고 있는 불필요한 텍스트 제거.
    - 총 94문항 증강.
- 실험결과
 
    |-| Public Accuracy | Private Accuracy |
    | --- | --- | --- |
    | origin | 0.6751 | 0.6299 |
    | PSAT(22, 23,24) auged | 0.6705 | 0.6230 |

   - 2,030개의 원본데이터에 94문항 추가하였기 때문에 기존 문항에 비해 증강한 데이터셋이 너무 적었음.
   - origin train 데이터에 국어 문제를 증강하긴 했지만, 여전히 사회와 관련된 train 데이터가 많아서 큰 성능 향상이 있지 않았음.
   - 24년도 기출문제를 예시로 살펴보면 대부분의 question은 그저 “다음 글의 내용과 부합하는 것은?” 으로 전체적인 글을 읽고 문제를 푸는 case가 증강되었지만, 기존의 origin dataset에서는 빈칸에 들어갈 말이나 조금 더 지문에 관련된 단어가 포함된 질문 text로 구성되었음.

#### KorQuAD
- 데이터 설명
    - 한국어 기계 독해(MRC) 성능을 평가하기 위한 대표적인 한국어 데이터셋.
    - 주로 Wikipedia 문서에서 문단을 선택하고, 그에 맞는 질문과 답변을 작성하여 데이터셋을 구성.
    - KorQuAD는 질문에 대해 문서 내의 특정 부분에서 정확한 답을 찾는 단답형 QA 구조이며, KLUE-MRC는 추론이 필요한 질문이 더 많이 포함되어 있어 일반적으로 KorQuAD보다 난이도가 높음.
- 주제 필터링
    - KorQuAD 데이터셋 중 [역사, 경제, 정치, 지리, 심리, 경제, 교육산업, 국제, 부동산, 사회, 생활, 책마을] 관련 주제만 남겨서 데이터 증강에 활용하고자 함.
    - context 중복 제거 후 unique context에 대해 LLM에 주제와의 관련성 판단을 맡겨 유효 context에 해당하는 id를 가진 row 추출.
- 선지 생성
    - KorQuAD는 지문과 이에 대한 정답만 제공하며, 수능형 문제로 전환하기 위해 보기 선지가 필요했음.
    - EXAONE 모델에 보기 선지 생성을 요청하여 KorQuAD 데이터를 학습 데이터로 활용하기에 적합한 형식으로 만듦.
- 결과 분석
    - 선지가 제대로 생성되지 않았거나, 주어진 지문에 해당하지 않는 내용의 선지가 생성된 경우 발생.
    - 정답이 보기 선지에 포함되지 않은 경우가 다수 존재.
    - 생성된 선지의 품질이 수능형 문제의 요구 사항을 충족하지 못함.
    - 따라서 보기 선지를 생성하는 접근 방식은 원하는 수준의 결과를 얻지 못하여 적용하지 않기로 결정.

### 모델 1차 학습
#### 모델링 설명
- train, test dataset이 국어와 사회 과목 문제로 구성되어 있었기 때문에 국어 문제를 잘 풀기 위한 reasoning 능력과 사회 문제를 잘 풀기 위한 배경 지식을 가지고 있는 모델이 문제를 잘 풀 것이라고 판단
- reasoning 능력을 키울 수 있는 데이터(TOEFL, SAT)와 사회 사전 지식 데이터(Kowiki)로 모델을 사전 학습

| Data | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| merge_1125 | 0.7857 | 0.7586 |
| TOEFL → merge_1125 | 0.7880 | 0.7793 |

**merge_1125** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '사회복지학')<br>
#### 결과 분석
- 위 두 실험을 비교해보면, TOEFL로 1차 학습을 진행했을 때 Public, Private Accuracy 모두 상승하여 1차 학습이 효과가 있음을 확인할 수 있음.

  
### 모델 Fine-Tuning
#### 모델링 설명
- train, test 데이터와 같은 구조를 가지는 데이터로 Fine-Tuning
- (1) 증강 학습데이터로 학습
- (2) K-Fold

**(1) 증강 데이터로 학습**
| Data (1차 → 2차) | Public Accuracy | Private Accuracy |
| --- | --- | --- |
| curr_sat | 0.7765 | 0.7632 |
| merge_1122 | 0.7811 | 0.7862 |
| TOEFL+SAT → merge_1127 | 0.7972 | 0.7770 |
| TOEFL+SAT → merge_1127(지문생성) | 0.7949 | 0.7701 |

**curr_sat** : 검정고시 + KBS한국어능력시험 + PSAT<br>
**merge_1122** : 검정고시 + KBS한국어능력시험 + PSAT + 공무원 기출문제은행('국어', '사회', '한국사', '경제학', '교육학')<br>
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

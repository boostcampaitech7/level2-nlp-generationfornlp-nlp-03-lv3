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
│   ├──helpers.py
│   ├── metric.py
│   └── prompt.py
├── train.py
├── train_unsloth.py
├── inference.py
├── inference_unsloth.py
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
|**김동한**|Data, Model|-**Toefl 데이터 사전학습**<br>- **pdf 데이터셋화**<br>- **모델 훈련 실험**|
|**김성훈**|Data, Model|내용|
|**김수아**|Data, Model|내용|
|**김현욱**|Data|-**MLflow 환경 구축**<br>- **모델 탐색**<br>- **Kowiki 사전 학습**<br>- **모델 훈련 실험**|
|**송수빈**|Data, Model|내용|
|**신수환**|Data|내용|
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

| Model | Valid Accuracy | Public Accuracy |
| --- | --- | --- |
| [yanolja/EEVE-Korean-Instruct-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0) | 0.2958 | 0.6705 |
| [yusikyang/mistral-7b-kowiki-10p-instruct-16bit](https://huggingface.co/yusikyang/mistral-7b-kowiki-10p-instruct-16bit) | 0.3908 | X |
| [unsloth/gemma-2-27b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit) | 0.4501 | 0.6083 |
| [unsloth/Qwen2.5-32B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct-bnb-4bit) | 0.5884 | **0.7765** |
| [cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) | 0.3054 | 0.4240 |
| [hungun/Qwen2.5-7B-Instruct-kowiki-qa](https://huggingface.co/hungun/Qwen2.5-7B-Instruct-kowiki-qa) | 0.3408 | 0.7396 |


### 모델 사전학습
#### 모델링 설명
- train, test dataset이 국어와 사회 과목 문제로 구성되어 있었기 때문에 국어 문제를 잘 풀기 위한 reasoning 능력과 사회 문제를 잘 풀기 위한 배경 지식을 가지고 있는 모델이 문제를 잘 풀 것이라고 판단
- reasoning 능력을 키울 수 있는 데이터(Toefl, SAT)와 사회 사전 지식 데이터(Kowiki)로 모델을 사전 학습

#### 결과 분석
- 정량적 평가 : 제안하는 모델이 기존 backbone의 성능보다 더 좋은 것을 확인하였으나, 모델 학습의 cost가 커지는 것 대비 성능의 차이가 미미한 것으로 판단됨
- 정성적 평가 : 실제 데이터 예측 결과를 확인해본 결과 `주택가`와 `주거지역`이라는 유사한 단어가 존재하는 경우 반영하여 더 높은 점수를 부여하는 것을 확인
- 사전 그래프 구축 작업의 정교함에 따라 성능의 폭이 커질 것으로 예상되어 마지막 제출 전 시간이 남으면 진행할 것으로 결론냄


### 데이터 증강
|**Version**|**Abstract**|**num**|
|:--:|--|:--:|
|**V1_Downsampling**|label 0.0 데이터 1000개 downsampling|8,324|
|**V2_augmentation_biased**|`AugmentationV1` + `BERT-Token Insertion`|9,994|
|**V3_augmentation_uniform**|`AugmentationV2` + `Adverb Augmentation` + `Sentence Swap` + `BERT-Token Insertion`|15,541|
|**V4_augmentation_spellcheck**|`AugmentationV2` + `hanspell` + `Sentence Swap` |17,313|


### 증강 데이터 버전 설명
|**Version**|**Description**|
|:--:|--|
|**V1_Downsampling** |Downsampling된 1000개의 문장으로 V2에서 (4.0, 5.0] label의 data augmentation을 진행할 것이기 때문에, label이 0.0인 데이터셋에서 문장 내 token 수가 3개 이상이면서, K-TACC 증강 방법 중 random_masking_insertion을 진행했을 때 증강이 되는 문장을 선별했습니다. sentence_1과 sentence_2 모두 증강된 index만 고려하면서, sentence_1을 기준으로 유사도가 높은 상위 1000개의 index를 선별했습니다. 문장 간 유사도가 고려되지 못한 sentence_2 데이터셋에 대해서는 추후 data filtering을 거쳤습니다.|
|**V2_augmentation_biassed**|V1에서 Downsampling된 1000개 데이터셋을 증강한 데이터셋 중에서도 label이 5.0인 데이터셋은 큰 차이가 없어야 한다고 판단하여, 불용어를 제거하면 같은 문장인 데이터를 label 5.0에 할당했습니다. label이 (4.0, 5.0)인 데이터셋은 라벨 간의 비율을 직접 조정하면서, 유사도가 높은 순서대로 개수에 맞게 할당했습니다.|
|**V3_augmentation_uniform**| label 분포를 균형있게 맞추어 전체적인 데이터 분포를 고르게 하기 위해 **라벨별 증강 비율을 조정**하여 총 3단계에 걸쳐 증강했고 매 단계마다 데이터의 개수가 적은 label들을 집중적으로 증강했습니다. <br> 1단계로 label이 `0.5, 1.5, 1.6, 2.2, 2.4, 2.5, 3.5` 데이터에 대해 Adverb Augmentation 수행했습니다. 2단계로 label이 `0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6, 2.8, 3, 3.2, 3.4, 3.5` 데이터에 대해 Sentence Swap 수행하였습니다. 3단계로 `1.5, 2.5, 3.5` 데이터에 대해 random_masking_insertion을 수행하였으며 추가로 `1.5, 2.5` 데이터 중 Masking Insertion한 증강 데이터에 대해 Sentence Swap을 수행했습니다.|
|**V4_augmentation_spellcheck**|label이 0.0인 데이터셋 중 맞춤법 교정 라이브러리 hanspell이 sentence_1과 sentence_2 모두에 적용된 index 776개를 뽑고, 증강된 데이터셋들을 label 4.8에 493개, label 5.0에 1059개 할당하였습니다. label이 (0.0, 4.4]인 데이터셋은 sentence swapping을 진행하였습니다. V2의 데이터셋 중 500개를 뽑아와 label 4.6에 450개, 4.5에 50개 할당하여 라벨 간 비율이 비숫해지도록 조정하였습니다.|


### 모델 Fine-Tuning
#### 모델링 설명
- train, test 데이터와 같은 구조를 가지는 데이터로 Fine-Tuning
- (1) 증강 학습데이터로 학습
- (2) K-Fold

**증강 데이터로 학습**


**결과 분석**


**K-Fold**

**결과 분석**
| Model | Validation Pearson | Public Pearson |
| --- | --- | --- |
| deliciouscat/kf-deberta-base-cross-sts | 0.926 | 0.9110 |
| deliciouscat/kf-deberta-base-cross-sts + GNN | 0.929 | 0.9164 |
| deliciouscat/kf-deberta-base-cross-sts + CL | 0.929 | 0.9190 |

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
        
- Min-Max 정규화 가중 평균
    - 단순 Valid score 기반 가중 평균시, 대부분의 앙상블 대상 모델이 92~93의 유사한 valid score를 가짐
    - 더 좋은 성능을 가진 모델과 그렇지 않은 모델간 차이를 줄 수 있게 적합한 가중치 정규화의 필요성을 느낌
    - 앙상블할 모델의 valid score를 0.8~1.2 값으로 scaling하여 가중평균
    - 0.8~1.2로 scaling할때의 min-max 정규화 수식

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

<img src="[./markdownimg/Leader1.png](https://github.com/user-attachments/assets/2293845f-c342-4989-9c79-000dae6062c7)"/>


**Private Leader Board 순위**

<img src="[./markdownimg/Leader2.png](https://github.com/user-attachments/assets/cc48bcc5-d0fb-4d64-be8a-2a5f2ce876d7)"/>

# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Generation for NLP Project : 한국어 수능 시험 문제 풀이 모델

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [Installation and Quick Start](#2-installation-and-quick-start)
3. [팀원 소개](#3-팀원-소개)
4. [프로젝트 진행](#4-프로젝트-진행)
5. [제출 결과](#5-제출-결과)
## 1. 프로젝트 소개
(1) 주제 및 목표
- 부스트캠프 AI Tech NLP 트랙 level 2 대회
- 주제 : 한국어 수능 시험 문제 풀이 모델 (Generation for NLP)    
      수능 특화 언어모델 만들기  <br>

(2) 평가지표
- Accuracy <br>

(3) 개발 환경 <br>
- GPU : Tesla V100 * 4 <br>


## 2. Installation and Quick Start

**Step 1.** 해당 repository를 clone해서 사용

**Step 2.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`와 `requirements_unsloth.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행
```sh
$ python -m venv .venv
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
**Step 3.** `data_aug` 폴더에서 데이터 증강 진행, 자세한 증강 방안은 [프로젝트 진행](#4-프로젝트-진행)의 Fine-tuning 참고

**Step 4.** 본인의 가상환경에서 Training 수행, `utils/arguments.py` 에서 학습을 위한 파라미터 변경
- model_name_or_path : 사전 학습된 huggingface 모델 불러오기, Fine-tuning 된 체크포인트 불러오기
- dataset_name : 학습 데이터 경로, `merge_dataset_20241125.csv`는 원본과 증강 된 데이터를 합친 것
- valid_dataset_name : 검증 데이터 경로, 없다면 None

```sh
$ . .venv/bin/activate

# 기본 모델 training 코드
$ python train_baseline.py

# unsloth을 활용한 모델 학습, 지원하는 모델 목록은 unsloth 홈페이지 참고
$ python train_unsloth.py
```

**Step 5.** 본인의 가상환경에서 Inference 수행
- model_name_or_path : 학습 시 사용한 사전 학습된 huggingface 모델 이름
- checkpoint : Fine-tuning된 체크포인트
```sh
$ . .venv/bin/activate

# 기본 모델 inference 코드
$ python inference_baseline.py

# unsloth inference 코드
$ python inference_unsloth.py

```

**Step 6.** Inference된 output을 가지고 `ensemble.ipynb` 실행

**Optional.** 원격 연결 끊어졌을 때도 실행될 수 있도록 Tmux 사용을 권장
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


## 3. 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
|[Github](https://github.com/dongspam0209)|[Github](https://github.com/sunghoon014)|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github](https://github.com/kkobugi)| -->|


### 맡은 역할
|**Member**|**Team**|**Role**|
|:--|--|--|
|**김동한**|Data, Model|- **TOEFL+SAT 데이터 1차 Fine-Tuning**<br>- **PSAT기출문제 데이터 증강**<br>- **모델 훈련 실험**|
|**김성훈**|Data, Model|- **베이스라인 코드 모듈화**<br>- **모델 탐색, 모델 경량화**<br>- **공무원시험기출문제 데이터 증강**<br>- **모델 훈련 및 실험** |
|**김수아**|Data, Model|- **검정고시 데이터 증강**<br>- **K-Fold 도입**<br>- **앙상블**|
|**김현욱**|Data, Model|- **MLflow 환경 구축**<br>- **모델 탐색**<br>- **Kowiki 사전 학습**<br>- **모델 훈련 실험**|
|**송수빈**|Data, Model|- **EDA**<br>- **데이터 증강(kbs 한국어능력시험, SAT history, KorQuAD)**<br>- **SAT 데이터 사전학습**<br>- **모델 훈련 실험**|
|**신수환**|Data, Model| - **모델 사전 탐색**<br>- **Kowiki 필터링** |
<br>

## 4. 프로젝트 진행
| Task | **Task Description** |
| --- | --- |
| [**EDA**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/EDA.md) | 데이터의 특성을 살펴보기 위해 데이터 시각화 및 분석 |
| [**Model Exploration**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Model_Experiment.md) | 한국어 수능 문제 풀이에 적합한 pre-trained model 선정 |
| [**Fine-tuning_1**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Fine-Tuning_1.md) | 모델의 문제 풀이 능력을 향상시키기 위한 1차 fine-tuning |
| [**Fine-tuning_2**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Fine-Tuning_2.md) | 모델의 문제 풀이 능력을 향상시키기 위한 2차 fine-tuning |
| [**Post Processing & Result**](https://github.com/boostcampaitech7/level2-nlp-generationfornlp-nlp-03-lv3/blob/main/.doc/Post_Processing.md) | K-Fold, 앙상블 알고리즘을 사용해 후처리 & 실험 결과 |

## 5. 제출 결과
- Public 최고 성적 : (0.8249, 0.7678)
![image](https://github.com/user-attachments/assets/ef163889-201e-4868-8aa3-4ae34773c191)

        
- Private 최고 성적 : (0.7811, 0.7862)
![image2](https://github.com/user-attachments/assets/e2075550-7944-4ba5-b2f8-c66a47c5c9af)

### K-Fold<br>
1. **Random K-Fold** : 검정고시 증강 데이터를 5개의 fold로 랜덤 분할
      - fold 안에 국어, 사회 문제가 골고루 들어가도록 처리를 하지 않았음에도 public accuracy 기준 약 4%의 성능 향상이 확인됨.
        
      | | Public Accuracy | Private Accuracy |
      |---|---|---|
      | origin | 0.6774 | 0.6184 |
      | 5 fold + ensemble (hard voting) | 0.7143 | 0.6805 |
  
2. **Stratified K-Fold** : 데이터를 국어/사회 분야로 나눈 후 각 fold가 전체 데이터의 국어/사회 비율을 유지하도록 함.
      - fold 구성
        - 국어 분야: 검정고시 및 공무원 시험 국어 과목, KLUE-MRC 문제
        - 사회 분야: 나머지 문제들
      - Public accuracy 기준 약 9%의 성능 향상이 있었음.
      - 두 실험 간 fold 수와 데이터셋 구성에 차이가 있으나, 성능 향상 폭을 고려할 때 데이터를 균형적으로 분할하는 것이 모델 성능 개선에 더 큰 영향을 미친 것으로 판단됨.

      | | Public Accuracy | Private Accuracy |
      |---|---|---|
      | origin | 0.7143 | 0.6805 |
      | 4 fold + ensemble (hard voting) | 0.8065 | 0.7770 |
      | 4 fold + ensemble (soft voting) | 0.8088 | 0.7747 |
      
### 앙상블
- 방법론
    - Soft Voting
        - 각 선택지 별 확률값의 평균을 계산하여 가장 높은 확률을 가진 선택지를 최종 답안으로 선정.
    - Soft Voting ver2
        - Soft Voting 결과 답안으로 선택되는 선택지의 확률 값이 점점 낮아지는 현상이 있음.
        - 모델의 확신도가 높을수록 정답일 확률이 높아질 것이라는 가정에 따라 Soft Voting ver2를 제작함.
        - 평균 확률값이 임계값 미만인 경우, 개별 모델 중 가장 높은 확신도를 보인 답안을 선택하는 방식으로 보완.
    - Hard Voting
        - 다수결 방식으로 가장 많은 투표를 받은 선택지를 최종 답안으로 선정.
        - 동점이 발생한 경우, 동점인 선택지들 중에서 가장 높은 확률값을 가진 것을 최종 답안으로 선택.
    - Hard Voting ver2
        - 기본 hard voting과 비교했을 때 동점 발생 시 처리 방식에서 차이가 있음.
        - 동점인 선택지들에 대해 각 모델이 예측한 확률값들의 합이 가장 큰 선택지를 최종 답안으로 선택.
- 실험 결과 및 분석
    - 첫번째 실험에서 Public Accuracy 기준 Soft Voting이 Hard Voting보다 결과가 좋음.
    - 두번째 실험에서 soft voting ver2가 성능이 좋았음.
    - 실험결과에 따라 soft voting ver2의 성능이 가장 좋을 것이라 가정하고, public accuracy가 높았던 여러 결과들에대해 Soft Voting ver2로 앙상블을 수행함.
    - 실험 결과
        - 앙상블한 데이터셋 : 검정고시 증강 데이터 fold 4개
        
            |  | Public Accuracy |  Private Accuracy |
            | --- | --- | --- |
            | origin | 0.7143 | 0.6805 |
            | 4 fold + ensemble (Hard Voting) | 0.8065 | 0.7770 |
            | 4 fold + ensemble (Soft Voting) | 0.8088 | 0.7747 |
        - 앙상블한 데이터셋 : merge_dataset_20241124_fold_1 학습 결과 + 검정고시 증강 데이터 4 fold 후 soft voting한 결과 + toefl, sat 데이터 학습 결과
        
            |  | Public Accuracy |  Private Accuracy |
            | --- | --- | --- |
            | Soft Voting | 0.8088 | 0.7724 |
            | Soft Voting ver2 | 0.8134 | 0.7747 |


### 실험 결과

- 학습
    - 기본 성능이 가장 좋았던 **unsloth-Qwen2.5-32B-Instruct-bnb-4bit**으로 실험.
    - Filtering한 Kowiki 데이터들의 ‘text’ column의 길이가 최대 32,913개로 굉장히 길어 개발 환경에서 모델 학습이 불가했음. 따라서 길이 10,000 이상인 ‘text’를 가지고 있는 186개의 데이터를 제거한 후 남은 675개로 학습을 진행하였음.
    - LoRA Fine-Tuning을 수행했고 LoRA 하이퍼파라미터로 lora_r과 lora_alpha 값은 유동적으로, targe_modules는 다음과 같이 고정함. `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
    - batch size는 8로 모두 동일.
    - 앙상블 아닌 단일 모델 실험.
    - 학습 prompt 구조
        
        ```python
        """<|im_start|>system
        문제를 풀기 위해 단계별 사고 과정을 거쳐 정답을 도출하세요.<|im_end|>
        <|im_start|>user
        학습 목적:
        {}
        
        배경 지식:
        {}
        
        문제:
        {}
        
        사고 흐름:
        {}
        
        사고 흐름 답변:
        {}
        
        정답을 도출해 주세요.
        <|im_end|>
        <|im_start|>assistant
        {}
        <|im_end|>"""
        ```
        

| 1차 Fine Tuning | 2차 Fine tuning | lora_r | lora_alpha | Public Accuracy | Private Accuaracy |
| --- | --- | --- | --- | --- | --- |
| - | - | - | - | 0.7558 | 0.7080 |
| - | merge_dataset_20241124_fold_1 | 32 | 64 | 0.8088 | 0.7701 |
| TOEFL + SAT | merge_dataset_20241127 | 32 | 64 | 0.7972 | 0.7770 |
| TOEFL + SAT | merge_dataset_20241127 | 64 | 64 | 0.7949 | 0.7701 |
- **merge_dataset_20241124_fold_1** : 검정고시 + SAT(번역) + PSAT + 공무원기출문제은행[국어, 사회, 한국사, 경제학, 교육학]으로 만든 데이터셋을 K-fold를 위해 나눈 데이터셋 중 fold 1을 학습하기 위해 만든 데이터셋
- **merge_dataset_20241127** : 검정고시 + SAT(번역) + PSAT + 공무원기출문제은행[국어, 사회, 한국사, 경제학, 교육학, 사회복지학]

### 최종 성능 (Public, Private)

- 앙상블 기법 : Soft Voting ver2을 사용.
- 모델 조합 - Public, Private Accuracy : (0.8249, 0.7678)
    
    
    | 1차 Fine-Tuning | 2차 Fine-Tuning | lora_r | lora_alpha |
    | --- | --- | --- | --- |
    | - | 검정고시 + SAT(번역) + PSAT + 공무원기출문제은행[국어, 사회, 한국사, 경제학, 교육학] | 32 | 64 |
    | TOEFL + SAT | 검정고시 + SAT(번역) + PSAT + 공무원기출문제은행[국어, 사회, 한국사, 경제학, 교육학, 사회복지학] | 32 | 64 |
    | TOEFL + SAT + Kowiki | KBS한국어능력시험, SAT world, 공무원기출시험은행[사회, 한국사, 경제] | 32 | 64 |
    | - | KBS한국어능력시험, SAT world, 공무원기출시험은행[사회, 한국사, 한국사, 사회복지학] | 32 | 64 |
    | - | KBS한국어능력시험, SAT world, 공무원기출시험은행[사회, 한국사, 한국사, 사회복지학] | 64 | 128 |
    | TOEFL + SAT + Kowiki | 공무원기출시험은행[국어, 사회, 한국사, 경제, 교육학, 사회복지학] | 32 | 64 |
    | TOEFL + SAT | 검정고시 | 32 | 64 |
    | TOEFL + SAT + Kowiki | KBS한국어능력시험, SAT world | 32 | 64 |
- 결과
    - Public 최고 성적 : (0.8249, 0.7678)
        
        ![image](https://github.com/user-attachments/assets/ef163889-201e-4868-8aa3-4ae34773c191)

        
    - Private 최고 성적 : (0.7811, 0.7862)
        
        ![image2](https://github.com/user-attachments/assets/e2075550-7944-4ba5-b2f8-c66a47c5c9af)
